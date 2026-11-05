use clap::{Parser, Subcommand};
use lechiffre_nlhe::game::GAME_ID;
use lechiffre_noregrets::{
    bridge,
    reference::{self, Reference, Settings},
    rs_agent::AgentReference,
    rs_arena,
    table_arena::PreparedTable,
};
use lechiffre_sdcfr::training::Trainer;
use std::{fs, path::PathBuf, time::Instant};

#[derive(Parser)]
#[command(
    about = "NoRegrets : Agent rs_poker, matchs contre agents existants et comparaison SD-CFR"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Entrainer un NoRegrets compatible Agent rs_poker, 2 a 6 joueurs.
    TrainAgent {
        #[arg(long, default_value_t = 10000)]
        iterations: u64,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long, default_value_t = 20)]
        stack: u32,
        #[arg(long, default_value_t = 2)]
        players: usize,
        #[arg(long)]
        out: PathBuf,
    },
    /// Choisir chaque siege d'une table de 2 a 6 joueurs.
    Table {
        #[arg(long, required_unless_present = "table_config")]
        players: Option<String>,
        #[arg(long, conflicts_with = "players")]
        table_config: Option<PathBuf>,
        #[arg(long)]
        reference: Option<PathBuf>,
        #[arg(long)]
        stack: Option<u32>,
        #[arg(long, default_value_t = 6000)]
        hands: u64,
        #[arg(long, default_value_t = 123)]
        seed: u64,
        #[arg(long)]
        out: PathBuf,
    },
    /// NoRegrets contre les agents deja implementes dans rs_poker.
    MatchAgents {
        #[arg(long)]
        reference: PathBuf,
        #[arg(long, default_value = "random")]
        opponent: String,
        #[arg(long)]
        opponent_config: Option<PathBuf>,
        #[arg(long, default_value_t = 1000)]
        hands: u64,
        #[arg(long, default_value_t = 123)]
        seed: u64,
        #[arg(long)]
        out: PathBuf,
    },
    /// Entrainer le vrai MCCFR tabulaire de NoRegrets sur le jeu reduit.
    Train {
        #[arg(long, default_value_t = 10_000)]
        iterations: u64,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long, default_value_t = 1)]
        threads: usize,
        #[arg(long, default_value_t = 12)]
        buckets: u16,
        #[arg(long, default_value_t = 64)]
        equity_rollouts: u32,
        /// Zero: buckets d'equite sans centroids. >0: k-medians EMD amont.
        #[arg(long, default_value_t = 0)]
        centroid_samples: usize,
        #[arg(long)]
        out: PathBuf,
    },
    /// Mesurer les gains du SD-CFR face au blueprint NoRegrets, agents figes.
    Match {
        #[arg(long)]
        checkpoint: PathBuf,
        #[arg(long)]
        reference: PathBuf,
        #[arg(long, default_value_t = 10_000)]
        hands: u64,
        #[arg(long, default_value_t = 123)]
        seed: u64,
        #[arg(long)]
        out: PathBuf,
    },
    /// Verifier la provenance et les regles d'une reference sauvegardee.
    Inspect {
        #[arg(long)]
        reference: PathBuf,
    },
}

async fn run() -> Result<(), String> {
    match Cli::parse().command {
        Command::TrainAgent {
            iterations,
            seed,
            stack,
            players,
            out,
        } => {
            if out.exists() {
                return Err("output directory already exists".into());
            }
            let r = AgentReference::train_for_players(iterations, seed, stack, players)?;
            r.save(&out)?;
            println!(
                "NoRegrets Agent sauvegarde : {} ; {players} joueurs, {} BB, relances libres",
                out.display(),
                stack as f64 / 2.0
            );
        }
        Command::Table {
            players,
            table_config,
            reference,
            stack,
            hands,
            seed,
            out,
        } => {
            if out.exists() {
                return Err("output directory already exists".into());
            }
            let (values, stack) = if let Some(path) = table_config {
                #[derive(serde::Deserialize)]
                struct Config {
                    players: Vec<serde_json::Value>,
                    stack: Option<u32>,
                }
                let c: Config = serde_json::from_slice(&fs::read(path).map_err(|e| e.to_string())?)
                    .map_err(|e| e.to_string())?;
                (c.players, stack.or(c.stack))
            } else {
                let names = players.ok_or("provide --players or --table-config")?;
                (
                    names
                        .split(',')
                        .map(|n| serde_json::json!({"type":n.trim()}))
                        .collect(),
                    stack,
                )
            };
            let table = PreparedTable::new(values, stack, reference.as_deref())?;
            let r = table.run(hands, seed).await?;
            println!(
                "Table {} joueurs, {} BB, {} mains ; bouton tournant ; blueprint NoRegrets sans recherche en ligne.",
                r.players,
                r.stack as f64 / 2.0,
                r.hands
            );
            for s in &r.seats {
                println!(
                    "Siege {} | {} | {:+.3} bb/100 | IC95 [{:+.3}, {:+.3}] | total {:+.2} BB",
                    s.seat, s.agent, s.bb_per_100, s.ci95[0], s.ci95[1], s.total_bb
                );
                if let Some(a) = &s.noregrets {
                    println!(
                        "  NoRegrets : inconnus {}/{} ; decisions apres mise hors menu {}",
                        a.unseen, a.decisions, a.decisions_after_off_tree_bet
                    );
                }
            }
            if let Some(p) = out.parent().filter(|p| !p.as_os_str().is_empty()) {
                fs::create_dir_all(p).map_err(|e| e.to_string())?;
            }
            fs::create_dir(&out).map_err(|e| e.to_string())?;
            let mut csv = String::from("seat,agent,hands,total_bb,bb_per_100,ci95_low,ci95_high\n");
            for s in &r.seats {
                csv.push_str(&format!(
                    "{},\"{}\",{},{:.6},{:.6},{:.6},{:.6}\n",
                    s.seat,
                    s.agent.replace('"', "\"\""),
                    r.hands,
                    s.total_bb,
                    s.bb_per_100,
                    s.ci95[0],
                    s.ci95[1]
                ));
            }
            fs::write(out.join("table.csv"), csv).map_err(|e| e.to_string())?;
            fs::write(
                out.join("table.json"),
                serde_json::to_vec_pretty(&r).map_err(|e| e.to_string())?,
            )
            .map_err(|e| e.to_string())?;
        }
        Command::MatchAgents {
            reference,
            opponent,
            opponent_config,
            hands,
            seed,
            out,
        } => {
            if out.exists() {
                return Err("output directory already exists".into());
            }
            let r = std::sync::Arc::new(AgentReference::load(&reference)?);
            let opponents = if let Some(path) = opponent_config {
                let b = rs_poker::arena::agent::ConfigAgentBuilder::from_file(&path)
                    .map_err(|e| e.to_string())?;
                let mut cfg = b.config().clone();
                rs_arena::configure_cfr(&mut cfg)?;
                vec![(path.display().to_string(), cfg)]
            } else {
                let names = if opponent == "all" {
                    vec![
                        "random",
                        "calling",
                        "all_in",
                        "folding",
                        "random_pot_control",
                        "cfr_basic",
                        "cfr_simple",
                    ]
                } else {
                    vec![opponent.as_str()]
                };
                names
                    .into_iter()
                    .map(|n| Ok((n.to_string(), rs_arena::opponent_config(n)?)))
                    .collect::<Result<Vec<_>, String>>()?
            };
            println!(
                "NoRegrets contre les vrais agents rs_poker ; HU {} BB, relances libres, blueprint seul.",
                r.metadata.stack as f64 / 2.0
            );
            let mut reports = Vec::new();
            for (name, cfg) in opponents {
                let report = rs_arena::duel(r.clone(), &name, cfg, hands, seed).await?;
                println!(
                    "{name}: NoRegrets {:+.3} bb/100 ; IC95 [{:+.3}, {:+.3}] ; inconnus {}/{} ; decisions apres mise hors menu {}",
                    report.bb_per_100,
                    report.ci95[0],
                    report.ci95[1],
                    report.noregrets.unseen,
                    report.noregrets.decisions,
                    report.noregrets.decisions_after_off_tree_bet
                );
                reports.push(report);
            }
            if let Some(p) = out.parent().filter(|p| !p.as_os_str().is_empty()) {
                fs::create_dir_all(p).map_err(|e| e.to_string())?;
            }
            fs::create_dir(&out).map_err(|e| e.to_string())?;
            let mut csv = String::from(
                "opponent,hands,seed,total_bb,bb_per_100,ci95_low,ci95_high,decisions,unseen,decisions_after_off_tree_bet\n",
            );
            for x in &reports {
                csv.push_str(&format!(
                    "\"{}\",{},{},{:.6},{:.6},{:.6},{:.6},{},{},{}\n",
                    x.opponent.replace('"', "\"\""),
                    x.hands,
                    x.seed,
                    x.total_bb,
                    x.bb_per_100,
                    x.ci95[0],
                    x.ci95[1],
                    x.noregrets.decisions,
                    x.noregrets.unseen,
                    x.noregrets.decisions_after_off_tree_bet
                ));
            }
            fs::write(out.join("matches.csv"), csv).map_err(|e| e.to_string())?;
            let json = serde_json::json!({"reference":r.metadata,"reports":reports,"engine":"rs_poker HoldemSimulation","exact_exploitability":null,"cfr_reproducibility":"CFR agents use internal thread RNG; seed alone does not guarantee identical CFR results"});
            fs::write(
                out.join("matches.json"),
                serde_json::to_vec_pretty(&json).map_err(|e| e.to_string())?,
            )
            .map_err(|e| e.to_string())?;
        }
        Command::Train {
            iterations,
            seed,
            threads,
            buckets,
            equity_rollouts,
            centroid_samples,
            out,
        } => {
            let settings = Settings {
                iterations,
                seed,
                threads,
                buckets,
                equity_rollouts,
                centroid_samples,
            };
            settings.validate()?;
            if out.exists() {
                return Err("output directory already exists; choose a new --out".into());
            }
            println!(
                "NoRegrets public {} ; HU 10 BB, menu LeChiffre, cap 1.",
                reference::UPSTREAM_COMMIT
            );
            println!(
                "MCCFR tabulaire, blueprint seul, pruning desactive. {iterations} parcours ; {threads} thread(s)."
            );
            println!(
                "Ce budget est un essai : aucun niveau GTO ou niveau de l'auteur n'est garanti."
            );
            let start = Instant::now();
            let model = reference::train(&settings)?;
            model.save(&out)?;
            println!(
                "Reference sauvegardee : {} ; {} infosets ; {:.2} s",
                out.display(),
                model.policy.blueprint.strategies.len(),
                start.elapsed().as_secs_f64()
            );
        }
        Command::Match {
            checkpoint,
            reference: path,
            hands,
            seed,
            out,
        } => {
            if out.exists() {
                return Err("output directory already exists; choose a new --out".into());
            }
            let hero = Trainer::load(&checkpoint, GAME_ID).map_err(|e| e.to_string())?;
            let reference = Reference::load(&path)?;
            let (r, coverage) = bridge::duel(&hero, &reference, hands, seed)?;
            if let Some(parent) = out.parent().filter(|p| !p.as_os_str().is_empty()) {
                fs::create_dir_all(parent).map_err(|e| e.to_string())?;
            }
            fs::create_dir(&out).map_err(|e| e.to_string())?;
            let report = serde_json::json!({
                "game_id": GAME_ID, "seed": seed, "hands": r.hands, "pairs": r.pairs,
                "hero_checkpoint": checkpoint, "hero_iterations": hero.completed(),
                "reference_directory": path, "reference": reference.metadata,
                "mode": "blueprint-only; no online search; no pruning; common reduced menu",
                "total_bb": r.total_bb, "bb_per_100": r.bb_per_100, "ci95": r.ci95,
                "mean_bb_by_hero_seat": r.by_seat, "reference_coverage": coverage,
                "exact_exploitability": serde_json::Value::Null
            });
            fs::write(
                out.join("comparison.json"),
                serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?,
            )
            .map_err(|e| e.to_string())?;
            fs::write(out.join("matches.csv"), format!("opponent,hands,pairs,seed,total_bb,bb_per_100,ci95_low,ci95_high\nnoregrets-blueprint-reduced,{},{},{},{:.6},{:.6},{:.6},{:.6}\n", r.hands,r.pairs,seed,r.total_bb,r.bb_per_100,r.ci95[0],r.ci95[1])).map_err(|e| e.to_string())?;
            println!(
                "SD-CFR vs NoRegrets blueprint reduit : {} mains appariees, places alternees.",
                r.hands
            );
            println!(
                "Gain du SD-CFR : {:+.3} bb/100 ; IC95 [{:+.3}, {:+.3}] ; total {:+.2} BB",
                r.bb_per_100, r.ci95[0], r.ci95[1], r.total_bb
            );
            for (i, street) in ["preflop", "flop", "turn", "river"].iter().enumerate() {
                println!(
                    "Reference {street}: {} decisions, {} replis check/call sur infoset inconnu.",
                    coverage.decisions[i], coverage.fallbacks[i]
                );
            }
            println!(
                "Ce resultat compare ces deux entrainements ; exploitabilite exacte non calculee."
            );
        }
        Command::Inspect { reference } => {
            let r = Reference::load(&reference)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&r.metadata).map_err(|e| e.to_string())?
            );
            println!("{} infosets", r.policy.blueprint.strategies.len());
        }
    }
    Ok(())
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    if let Err(error) = run().await {
        eprintln!("Erreur : {error}");
        std::process::exit(1);
    }
}
