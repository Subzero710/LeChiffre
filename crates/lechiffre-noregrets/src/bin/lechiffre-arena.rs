// Upstream dashboard unit tests belong to rsp's own crate/module tree.
// This binary is validated by the arena integration tests and real CLI runs.
#![cfg(not(test))]
//! Mixed NoRegrets / stock rs_poker agents, with the original rs_poker TUI.
use clap::{Parser, Subcommand};
use lechiffre_noregrets::{
    arena_config::prepare_directory,
    table_arena::{PreparedTable, TableReport},
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

async fn run_graphical(
    table: PreparedTable,
    hands: u64,
    seed: u64,
    out: PathBuf,
) -> Result<(), String> {
    use tui::{
        app::{self, App},
        event::{EventHandler, SimError, SimMessage},
        hand_store::HandStore,
        state::{GameResult, SeatStats, ending_round_from_stats},
    };
    let target = usize::try_from(hands).map_err(|_| "too many hands for this platform")?;
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
            let report = table
                .run_observed(hands, seed, Some(&history), |hand| {
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
                save_report(&report, &out)?;
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
    let Command::Table {
        agents_dir,
        hands,
        seed,
        stack,
        reference,
        out,
        display,
    } = Cli::parse().command;
    let table = prepare_directory(&agents_dir, stack, reference.as_deref())?;
    let n = table.players() as u64;
    if hands < 2 * n || !hands.is_multiple_of(n) {
        return Err(format!(
            "--hands must be a multiple of {n}, at least {}",
            2 * n
        ));
    }
    if out.exists() {
        return Err(format!(
            "{} already exists; choose a new --out directory",
            out.display()
        ));
    }
    fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    if display.should_use_tui() {
        run_graphical(table, hands, seed, out).await
    } else {
        let report = table
            .run_observed(hands, seed, Some(&out.join("hands.jsonl")), |_| true)
            .await?
            .ok_or("table cancelled")?;
        save_report(&report, &out)?;
        print_report(&report);
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
