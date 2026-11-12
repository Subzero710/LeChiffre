// Upstream dashboard unit tests belong to rsp's own crate/module tree.
// This binary is validated by the arena integration tests and real CLI runs.
#![cfg(not(test))]
//! Mixed NoRegrets / stock rs_poker agents, with the original rs_poker TUI.
use clap::{Parser, Subcommand};
use lechiffre_noregrets::{
    arena_config::prepare_directory,
    comparison::PreparedComparison,
    table_arena::{PreparedTable, TableHand, TableReport},
};
use std::{
    fs,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

// Reuse the library's dashboard sources instead of maintaining a second UI.
// The vendor revision is pinned by the LeChiffre project.
#[allow(dead_code)]
#[path = "../../../../vendor/rs-poker/src/bin/rsp/tui/mod.rs"]
mod tui;

#[derive(Parser)]
#[command(about = "Table graphique : agents rs_poker et plusieurs blueprints NoRegrets")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Comparateur natif rs_poker : toutes les permutations des profils aux sieges.
    Compare {
        agents_dir: PathBuf,
        /// Joueurs par table, independamment du nombre de profils du dossier.
        #[arg(short = 'p', long = "players", default_value_t = 6)]
        players: usize,
        /// Series d'etats initiaux ; chaque serie execute toutes les permutations.
        #[arg(short = 'n', long = "num-games", default_value_t = 10)]
        num_games: usize,
        #[arg(long, default_value_t = 123)]
        seed: u64,
        #[arg(long)]
        stack: Option<u32>,
        #[arg(long)]
        reference: Option<PathBuf>,
        #[arg(short = 'o', long = "out", alias = "output-dir")]
        out: PathBuf,
        #[command(flatten)]
        display: tui::TuiFlags,
    },
    /// Un fichier JSON par siege, ordre alphabetique des fichiers (2 a 6).
    Table {
        agents_dir: PathBuf,
        /// Nombre total de mains, multiple du nombre de joueurs.
        #[arg(long, default_value_t = 6000)]
        hands: u64,
        #[arg(long, default_value_t = 123)]
        seed: u64,
        /// Tapis en jetons (BB = 2). Par defaut, lu dans les modeles.
        #[arg(long)]
        stack: Option<u32>,
        /// Modele utilise par les fichiers noregrets sans reference explicite.
        #[arg(long)]
        reference: Option<PathBuf>,
        #[arg(long)]
        out: PathBuf,
        #[command(flatten)]
        display: tui::TuiFlags,
    },
}

enum Job {
    Table {
        table: PreparedTable,
        hands: u64,
        seed: u64,
    },
    Compare(PreparedComparison),
}

enum Report {
    Table(TableReport),
    Compare(rs_poker::arena::comparison::ComparisonResult),
}

impl Report {
    fn save(&self, out: &Path) -> Result<(), String> {
        match self {
            Self::Table(report) => save_report(report, out),
            Self::Compare(report) => report.save_to_dir(out).map_err(|e| e.to_string()),
        }
    }

    fn print(&self) {
        match self {
            Self::Table(report) => print_report(report),
            Self::Compare(report) => {
                println!(
                    "Comparateur natif rs_poker : {} mains, sieges permutes.",
                    report.total_games()
                );
                for (name, stats) in report.get_rankings() {
                    let bb100 = stats.total_profit as f64 * 100.0
                        / stats.total_games as f64
                        / report.config().big_blind as f64;
                    println!("{name} | {bb100:+.3} bb/100 | {} mains", stats.total_games);
                }
                println!(
                    "Details natifs : results.json et results.md. Exploitabilite exacte non calculee."
                );
            }
        }
    }
}

impl Job {
    fn target(&self) -> Result<usize, String> {
        match self {
            Self::Table { hands, .. } => {
                usize::try_from(*hands).map_err(|_| "too many hands".into())
            }
            Self::Compare(prepared) => Ok(prepared.arena.total_games()),
        }
    }

    async fn run(
        self,
        history: &Path,
        mut on_hand: impl FnMut(TableHand) -> bool,
    ) -> Result<Option<Report>, String> {
        match self {
            Self::Table { table, hands, seed } => table
                .run_observed(hands, seed, Some(history), on_hand)
                .await
                .map(|r| r.map(Report::Table)),
            Self::Compare(prepared) => {
                let target = prepared.arena.total_games();
                let mut count = 0;
                let result = prepared
                    .arena
                    .run_with_cancellable_callback(|perm| {
                        count += 1;
                        let profits = (0..perm.agent_names.len())
                            .map(|p| perm.stats.total_profit[p])
                            .collect();
                        let keep_running = on_hand(TableHand {
                            number: count as u64,
                            names: perm.agent_names,
                            profits,
                            stats: perm.stats,
                        });
                        if keep_running {
                            std::ops::ControlFlow::Continue(())
                        } else {
                            std::ops::ControlFlow::Break(())
                        }
                    })
                    .await
                    .map_err(|e| e.to_string())?;
                // Upstream returns partial statistics on cancellation. Do not
                // label an incomplete set of permutations as a complete run.
                Ok((count == target).then_some(Report::Compare(result)))
            }
        }
    }
}

fn save_report(report: &TableReport, out: &Path) -> Result<(), String> {
    let mut csv = String::from("seat,agent,hands,total_bb,bb_per_100,ci95_low,ci95_high\n");
    for s in &report.seats {
        csv.push_str(&format!(
            "{},\"{}\",{},{:.6},{:.6},{:.6},{:.6}\n",
            s.seat,
            s.agent.replace('"', "\"\""),
            report.hands,
            s.total_bb,
            s.bb_per_100,
            s.ci95[0],
            s.ci95[1]
        ));
    }
    fs::write(out.join("table.csv"), csv).map_err(|e| e.to_string())?;
    fs::write(
        out.join("table.json"),
        serde_json::to_vec_pretty(report).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())
}

fn print_report(report: &TableReport) {
    println!(
        "{} joueurs, {} mains, {} BB ; bouton tournant.",
        report.players,
        report.hands,
        report.stack as f64 / 2.0
    );
    for s in &report.seats {
        println!(
            "{} | {:+.3} bb/100 | IC95 [{:+.3}, {:+.3}]",
            s.agent, s.bb_per_100, s.ci95[0], s.ci95[1]
        );
    }
}

async fn run_graphical(job: Job, out: PathBuf) -> Result<(), String> {
    use tui::{
        app::{self, App},
        event::{EventHandler, SimError, SimMessage},
        hand_store::HandStore,
        state::{GameResult, SeatStats, ending_round_from_stats},
    };
    let target = job.target()?;
    let history = out.join("hands.jsonl");
    let store = HandStore::new(history.clone());
    let worker_store = store.clone();
    let cancel = Arc::new(AtomicBool::new(false));
    let worker_cancel = cancel.clone();
    let (tx, rx) = std::sync::mpsc::sync_channel(1024);
    let worker = tokio::spawn(async move {
        let sender = tx.clone();
        let join = tokio::spawn(async move {
            let mut offset = 0;
            let report = job
                .run(&history, |hand| {
                    worker_store.push_offset(offset);
                    offset = fs::metadata(&history).map(|m| m.len()).unwrap_or(offset);
                    let n = hand.names.len();
                    let event = GameResult {
                        agent_names: hand.names,
                        profits: hand.profits,
                        ending_round: ending_round_from_stats(&hand.stats, n),
                        seat_stats: (0..n)
                            .map(|p| SeatStats::from_storage(&hand.stats, p))
                            .collect(),
                        big_blind: 2,
                    };
                    sender.send(SimMessage::GameResult(event)).is_ok()
                        && !worker_cancel.load(Ordering::Acquire)
                })
                .await?;
            if let Some(report) = report {
                report.save(&out)?;
            }
            Ok::<_, String>(())
        })
        .await;
        let message = match join {
            Ok(Ok(())) => SimMessage::Completed,
            Ok(Err(e)) => SimMessage::Error(SimError::ComparisonFailed {
                source: rs_poker::arena::comparison::ComparisonError::Io(std::io::Error::other(e)),
            }),
            Err(_) => SimMessage::Error(SimError::Panic),
        };
        let _ = tx.send(message);
    });
    tui::run_blocking_tui_loop(
        move || {
            let mut app = App::new(Some(target));
            app.hand_store = store;
            app::run_app(&mut app, &EventHandler::new(rx, Duration::from_millis(33)))
        },
        worker,
        || cancel.store(true, Ordering::Release),
    )
    .await
    .map_err(|e| e.to_string())
}

async fn run() -> Result<(), String> {
    let (job, out, display) = match Cli::parse().command {
        Command::Table {
            agents_dir,
            hands,
            seed,
            stack,
            reference,
            out,
            display,
        } => {
            let table = prepare_directory(&agents_dir, stack, reference.as_deref())?;
            let n = table.players() as u64;
            if hands < 2 * n || !hands.is_multiple_of(n) {
                return Err(format!(
                    "--hands must be a multiple of {n}, at least {}",
                    2 * n
                ));
            }
            (Job::Table { table, hands, seed }, out, display)
        }
        Command::Compare {
            agents_dir,
            players,
            num_games,
            seed,
            stack,
            reference,
            out,
            display,
        } => {
            let prepared = PreparedComparison::from_directory(
                &agents_dir,
                players,
                num_games,
                stack,
                reference.as_deref(),
                seed,
                Some(&out),
            )?;
            println!(
                "{} profils, {players} places, {} permutations x {num_games} series = {} mains.",
                prepared.arena.num_agents(),
                prepared.arena.total_permutations(),
                prepared.arena.total_games()
            );
            (Job::Compare(prepared), out, display)
        }
    };
    if out.exists() {
        return Err(format!(
            "{} already exists; choose a new --out directory",
            out.display()
        ));
    }
    fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    if let Job::Compare(prepared) = &job {
        prepared.save_manifest(&out)?;
    }
    if display.should_use_tui() {
        run_graphical(job, out).await
    } else {
        let report = job
            .run(&out.join("hands.jsonl"), |_| true)
            .await?
            .ok_or("comparison cancelled")?;
        report.save(&out)?;
        report.print();
        Ok(())
    }
}

fn main() {
    // Same deep-stack runtime as rsp; the blocking native TUI gets its own thread.
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .thread_stack_size(47 * 1024 * 1024)
        .build()
        .expect("tokio runtime");
    if let Err(e) = runtime.block_on(run()) {
        eprintln!("Erreur : {e}");
        std::process::exit(1);
    }
}
