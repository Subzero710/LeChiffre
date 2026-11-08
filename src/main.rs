use lechiffre::adapters;
use lechiffre::adapters::pokernow::PokerNowImport;
use lechiffre::adapters::rs_poker::RsPokerAdapter;
use lechiffre::datasets;
use lechiffre::datasets::behaviour_parquet::BehaviourParquetDatasetWriter;
use lechiffre::datasets::behaviour_parquet_loader::{
    BehaviourDatasetStatistics, BehaviourParquetDataset,
};
use rs_poker::arena::{Agent, GameStateBuilder, HoldemSimulationBuilder, agent::RandomAgent};

const EXPORT_BATCH_HANDS: usize = 10_000;

fn import_poker_with_the_boys_session(path: &str, session_id: &str) -> PokerNowImport {
    let source = std::fs::read_to_string(path).unwrap_or_else(|error| {
        eprintln!("failed to read {path}: {error}");
        std::process::exit(1);
    });
    let normalizer = datasets::poker_with_the_boys::normalize_player_name;
    let result = if path.ends_with(".json") {
        adapters::pokernow::PokerNowAdapter::adapt_pokerlater_json(
            session_id,
            datasets::poker_with_the_boys::IDENTITY_NAMESPACE,
            &source,
            normalizer,
        )
    } else if path.ends_with(".csv") {
        adapters::pokernow::PokerNowAdapter::adapt_csv(
            session_id,
            datasets::poker_with_the_boys::IDENTITY_NAMESPACE,
            &source,
            normalizer,
        )
    } else {
        eprintln!("unsupported extension for {path}: expected .csv or .json");
        std::process::exit(2);
    };

    result.unwrap_or_else(|error| {
        eprintln!("invalid input session {session_id} from {path}: {error}");
        std::process::exit(1);
    })
}

fn print_dataset_statistics(stats: &BehaviourDatasetStatistics) {
    println!("{stats:#?}");
}

fn export_poker_with_the_boys_parquet(args: &[String]) {
    if args.len() < 5 || !(args.len() - 3).is_multiple_of(2) {
        eprintln!(
            "usage: lechiffre --export-poker-with-the-boys-parquet <output-dir> \
             <file.csv|file.json> <session-id> [<file.csv|file.json> <session-id> ...]"
        );
        std::process::exit(2);
    }

    let output_dir = &args[2];
    let subject_ids = datasets::poker_with_the_boys::subject_ids();
    let mut writer = BehaviourParquetDatasetWriter::create(output_dir).unwrap_or_else(|error| {
        eprintln!("failed to create behavioural dataset at {output_dir}: {error}");
        std::process::exit(1);
    });

    let mut source_accepted_hands = 0usize;
    let mut source_rejected_hands = 0usize;
    let mut timing_dropped_hands = 0usize;
    let mut structurally_dropped_hands = 0usize;

    for pair in args[3..].as_chunks::<2>().0 {
        let path = &pair[0];
        let session_id = &pair[1];
        let imported = import_poker_with_the_boys_session(path, session_id);

        source_accepted_hands = source_accepted_hands
            .checked_add(imported.hands.len())
            .unwrap_or_else(|| {
                eprintln!("source accepted-hand counter overflow");
                std::process::exit(1);
            });
        source_rejected_hands = source_rejected_hands
            .checked_add(imported.rejected.len())
            .unwrap_or_else(|| {
                eprintln!("source rejected-hand counter overflow");
                std::process::exit(1);
            });

        for rejection in &imported.rejected {
            eprintln!(
                "rejected source hand session={} hand={}: {}",
                session_id, rejection.source_hand_id, rejection.reason
            );
        }

        for chunk in imported.hands.chunks(EXPORT_BATCH_HANDS) {
            let report = writer
                .write_hands(chunk, &subject_ids)
                .unwrap_or_else(|error| {
                    eprintln!("failed to write session {session_id} from {path}: {error}");
                    std::process::exit(1);
                });
            timing_dropped_hands = timing_dropped_hands
                .checked_add(report.dropped_hand_ids_missing_timing.len())
                .unwrap_or_else(|| {
                    eprintln!("timing-drop counter overflow");
                    std::process::exit(1);
                });
            for hand_id in &report.dropped_hand_ids_missing_timing {
                eprintln!("dropped hand missing timing: {hand_id}");
            }
            structurally_dropped_hands = structurally_dropped_hands
                .checked_add(report.dropped_hands_invalid.len())
                .unwrap_or_else(|| {
                    eprintln!("invalid-hand drop counter overflow");
                    std::process::exit(1);
                });
            for dropped in &report.dropped_hands_invalid {
                eprintln!(
                    "dropped invalid hand {}: {}",
                    dropped.hand_id, dropped.reason
                );
            }
        }
    }

    if writer.summary().hands_written == 0 {
        eprintln!("refusing to commit an empty behavioural dataset");
        std::process::exit(1);
    }

    let summary = writer.close().unwrap_or_else(|error| {
        eprintln!("failed to commit behavioural dataset at {output_dir}: {error}");
        std::process::exit(1);
    });
    println!(
        "source_accepted_hands={} source_rejected_hands={} timing_dropped_hands={} \
         structurally_dropped_hands={} retained_hands={} unique_events={} subject_hand_rows={}",
        source_accepted_hands,
        source_rejected_hands,
        timing_dropped_hands,
        structurally_dropped_hands,
        summary.hands_written,
        summary.events_written,
        summary.subject_hand_rows_written
    );

    let dataset = BehaviourParquetDataset::open(output_dir).unwrap_or_else(|error| {
        eprintln!("failed to reload committed dataset at {output_dir}: {error}");
        std::process::exit(1);
    });
    let stats = dataset.statistics().unwrap_or_else(|error| {
        eprintln!("failed to analyze committed dataset at {output_dir}: {error}");
        std::process::exit(1);
    });
    print_dataset_statistics(&stats);
}

fn print_behaviour_dataset_stats(args: &[String]) {
    if args.len() != 3 {
        eprintln!("usage: lechiffre --behaviour-dataset-stats <parquet-dir>");
        std::process::exit(2);
    }
    let dataset = BehaviourParquetDataset::open(&args[2]).unwrap_or_else(|error| {
        eprintln!("failed to load behavioural dataset at {}: {error}", args[2]);
        std::process::exit(1);
    });
    let stats = dataset.statistics().unwrap_or_else(|error| {
        eprintln!(
            "failed to analyze behavioural dataset at {}: {error}",
            args[2]
        );
        std::process::exit(1);
    });
    print_dataset_statistics(&stats);
}

struct BehaviourRunConfig {
    split: lechiffre::datasets::behaviour_parquet_loader::GeneratorHandSplit,
    training: lechiffre::datasets::behaviour_parquet_loader::GeneratorTrainingConfig,
    latent_dimension: usize,
    generator_hidden: usize,
    analyzer_hidden: usize,
    generator_lr: f64,
    analyzer_lr: f64,
    epochs: usize,
    subjects: Vec<u64>,
    seed: u64,
}

/// Strict, explicit run configuration. No omitted hyperparameter receives a default.
fn run_configuration(path: &str) -> Result<BehaviourRunConfig, String> {
    use lechiffre::datasets::behaviour_parquet_loader::{
        GeneratorHandSplit, GeneratorTrainingConfig,
    };
    let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let value: serde_json::Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    let object = value
        .as_object()
        .ok_or("configuration must be a JSON object")?;
    let keys = [
        "latent_dimension",
        "generator_hidden",
        "analyzer_hidden",
        "generator_lr",
        "analyzer_lr",
        "beta_kl",
        "seed",
        "validation_draws",
        "continuity_delta",
        "first_query_hand_index",
        "first_evaluation_hand_index",
        "support_hands",
        "query_hands",
        "evaluation_hands",
        "epochs",
        "subject_ids",
    ];
    if object.len() != keys.len() || keys.iter().any(|k| !object.contains_key(*k)) {
        return Err(format!("configuration keys must be exactly: {keys:?}"));
    }
    let int = |key: &str| -> Result<u64, String> {
        value[key]
            .as_u64()
            .ok_or_else(|| format!("{key}: expected unsigned integer"))
    };
    let size = |key: &str| -> Result<usize, String> {
        usize::try_from(int(key)?).map_err(|_| format!("{key}: integer overflow"))
    };
    let float = |key: &str| -> Result<f64, String> {
        value[key]
            .as_f64()
            .filter(|x| x.is_finite())
            .ok_or_else(|| format!("{key}: expected finite number"))
    };
    let subjects = value["subject_ids"]
        .as_array()
        .ok_or("subject_ids: expected array")?
        .iter()
        .map(|x| {
            x.as_u64()
                .ok_or_else(|| "subject_ids: each ID must be unsigned integer".to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    if subjects.is_empty()
        || size("epochs")? == 0
        || float("generator_lr")? <= 0.0
        || float("analyzer_lr")? <= 0.0
    {
        return Err("epochs, selected subjects and optimizer rates must be positive".into());
    }
    let split = GeneratorHandSplit::new(
        int("first_query_hand_index")?,
        int("first_evaluation_hand_index")?,
        size("support_hands")?,
        size("query_hands")?,
        size("evaluation_hands")?,
    )
    .map_err(|e| e.to_string())?;
    let train = GeneratorTrainingConfig::new(
        float("beta_kl")?,
        int("seed")?,
        size("validation_draws")?,
        float("continuity_delta")?,
    )
    .map_err(|e| e.to_string())?;
    Ok(BehaviourRunConfig {
        split,
        training: train,
        latent_dimension: size("latent_dimension")?,
        generator_hidden: size("generator_hidden")?,
        analyzer_hidden: size("analyzer_hidden")?,
        generator_lr: float("generator_lr")?,
        analyzer_lr: float("analyzer_lr")?,
        epochs: size("epochs")?,
        subjects,
        seed: int("seed")?,
    })
}

fn exact_source_legality(
    args: &[String],
) -> Result<
    std::collections::HashMap<
        (lechiffre::behavioural::HandId, u32),
        lechiffre::behavioural::LegalActionSpace,
    >,
    String,
> {
    use std::collections::hash_map::Entry;
    let mut all = std::collections::HashMap::new();
    if args.is_empty() || !args.len().is_multiple_of(2) {
        return Err("expected one or more <file.csv|file.json> <session-id> pairs".into());
    }
    for pair in args.as_chunks::<2>().0 {
        let imported = import_poker_with_the_boys_session(&pair[0], &pair[1]);
        // An import rejection is local to one source hand. The matching hand
        // cannot occur in a committed dataset produced by the same importer.
        // Every retained decision must still resolve to exact replay legality.
        for rejection in &imported.rejected {
            eprintln!(
                "excluded source hand session={} hand={}: {}",
                pair[1], rejection.source_hand_id, rejection.reason
            );
        }
        for (key, legal) in imported.exact_legal_actions {
            match all.entry(key) {
                Entry::Vacant(entry) => {
                    entry.insert(legal);
                }
                Entry::Occupied(entry) => {
                    let (hand, sequence) = entry.key();
                    return Err(format!(
                        "duplicate exact legality for hand {hand} decision {sequence}"
                    ));
                }
            }
        }
    }
    if all.is_empty() {
        return Err("no exact replay legality was obtained".into());
    }
    Ok(all)
}

fn print_ag_report(
    report: &lechiffre::datasets::behaviour_parquet_loader::GeneratorEpochReport,
) -> Result<(), String> {
    println!(
        "epoch={} subjects={} trained_decisions={} validation_decisions={}",
        report.epoch, report.subjects, report.train.decisions, report.held_out.decisions
    );
    if report.train.decisions > 0 {
        println!(
            "query_train_nll_per_decision={:.8} action_nll={:.8} sizing_nll={:.8} timing_nll={:.8}",
            report.train.mean_nll().map_err(|e| e.to_string())?,
            report.train.action_nll / report.train.decisions as f64,
            report.train.sizing_nll / report.train.decisions as f64,
            report.train.timing_nll / report.train.decisions as f64
        );
    }
    println!(
        "future_validation_nll_per_decision={:.8} neutral_z_nll_per_decision={:.8} marginal_mc_nll_per_decision={:.8}",
        report.held_out.mean_nll().map_err(|e| e.to_string())?,
        report.prior.mean_nll().map_err(|e| e.to_string())?,
        report.marginal_validation_nll / report.held_out.decisions as f64
    );
    println!(
        "mean_KL={:.8} posterior_std_mean={:.8} posterior_action_variance={:.10} action_ECE={:.8} continuity_JS={:.10} diversity_JS={:?}",
        report.mean_kl,
        report.posterior_std_mean,
        report.posterior_action_variance,
        report.action_ece,
        report.continuity_js,
        report.policy_diversity_js
    );
    Ok(())
}

fn train_ag(args: &[String]) -> Result<(), String> {
    use lechiffre::behavioural::{
        Adam, AnalyzerAdam, BehaviourLatentConfig, NeuralBehaviourAnalyzer,
        NeuralBehaviourGenerator,
    };
    use lechiffre::datasets::behaviour_parquet_loader::{
        BehaviourParquetDataset, GeneratorTrainer,
    };
    if args.len() < 7 || !(args.len() - 5).is_multiple_of(2) {
        return Err("usage: --train-behavioural <config.json> <parquet-dir> <checkpoint-out.json> <source.csv|.json> <session-id> ...".into());
    }
    let run = run_configuration(&args[2])?;
    let dataset = BehaviourParquetDataset::open(&args[3]).map_err(|e| e.to_string())?;
    let exact = exact_source_legality(&args[5..])?;
    let latent = BehaviourLatentConfig::new(run.latent_dimension).map_err(|e| e.to_string())?;
    let generator = NeuralBehaviourGenerator::initialize(latent, run.generator_hidden, run.seed)
        .map_err(|e| e.to_string())?;
    let analyzer_seed = run.seed.checked_add(1).ok_or("seed overflow")?;
    let analyzer = NeuralBehaviourAnalyzer::initialize(latent, run.analyzer_hidden, analyzer_seed)
        .map_err(|e| e.to_string())?;
    let generator_adam = Adam::new(&generator, run.generator_lr).map_err(|e| e.to_string())?;
    let analyzer_adam = AnalyzerAdam::new(&analyzer, run.analyzer_lr).map_err(|e| e.to_string())?;
    let mut trainer = GeneratorTrainer::new(
        generator,
        analyzer,
        generator_adam,
        analyzer_adam,
        run.training,
    )
    .map_err(|e| e.to_string())?;
    for _ in 0..run.epochs {
        let report = trainer
            .train_epoch(
                &dataset,
                &run.subjects,
                run.split,
                &mut |_, hand: &lechiffre::behavioural::HandId, seq| {
                    exact
                        .get(&(hand.clone(), seq))
                        .copied()
                        .ok_or_else(|| format!("source decision not found: {hand}:{seq}"))
                },
            )
            .map_err(|e| e.to_string())?;
        print_ag_report(&report)?;
    }
    trainer
        .save_checkpoint(&args[4])
        .map_err(|e| e.to_string())?;
    println!("checkpoint={}", args[4]);
    Ok(())
}

fn resume_ag(args: &[String]) -> Result<(), String> {
    use lechiffre::datasets::behaviour_parquet_loader::{
        BehaviourParquetDataset, GeneratorTrainer,
    };
    if args.len() < 8 || !(args.len() - 6).is_multiple_of(2) {
        return Err("usage: --resume-behavioural <checkpoint-in.json> <parquet-dir> <checkpoint-out.json> <extra-epochs> <source.csv|.json> <session-id> ...".into());
    }
    let mut trainer = GeneratorTrainer::load_checkpoint(&args[2]).map_err(|e| e.to_string())?;
    let dataset = BehaviourParquetDataset::open(&args[3]).map_err(|e| e.to_string())?;
    let split = trainer
        .training_split()
        .ok_or("checkpoint has no training split")?;
    let subjects = trainer.training_subjects().to_vec();
    let extra = args[5]
        .parse::<usize>()
        .map_err(|_| "invalid extra-epochs")?;
    if extra == 0 {
        return Err("extra-epochs must be positive".into());
    }
    let exact = exact_source_legality(&args[6..])?;
    for _ in 0..extra {
        let report = trainer
            .train_epoch(
                &dataset,
                &subjects,
                split,
                &mut |_, hand: &lechiffre::behavioural::HandId, seq| {
                    exact
                        .get(&(hand.clone(), seq))
                        .copied()
                        .ok_or_else(|| format!("source decision not found: {hand}:{seq}"))
                },
            )
            .map_err(|e| e.to_string())?;
        print_ag_report(&report)?;
    }
    trainer
        .save_checkpoint(&args[4])
        .map_err(|e| e.to_string())?;
    println!("checkpoint={}", args[4]);
    Ok(())
}

fn evaluate_ag(args: &[String]) -> Result<(), String> {
    use lechiffre::datasets::behaviour_parquet_loader::{
        BehaviourParquetDataset, GeneratorTrainer,
    };
    if args.len() < 6 || !(args.len() - 4).is_multiple_of(2) {
        return Err("usage: --evaluate-behavioural <checkpoint.json> <parquet-dir> <source.csv|.json> <session-id> ...".into());
    }
    let trainer = GeneratorTrainer::load_checkpoint(&args[2]).map_err(|e| e.to_string())?;
    let dataset = BehaviourParquetDataset::open(&args[3]).map_err(|e| e.to_string())?;
    let exact = exact_source_legality(&args[4..])?;
    let report = trainer
        .evaluate(&dataset, &mut |_,
                                  hand: &lechiffre::behavioural::HandId,
                                  seq| {
            exact
                .get(&(hand.clone(), seq))
                .copied()
                .ok_or_else(|| format!("source decision not found: {hand}:{seq}"))
        })
        .map_err(|e| e.to_string())?;
    print_ag_report(&report)
}

fn evaluate_unseen_ag(args: &[String]) -> Result<(), String> {
    use lechiffre::datasets::behaviour_parquet_loader::{
        BehaviourParquetDataset, GeneratorTrainer,
    };
    if args.len() < 7 || !(args.len() - 5).is_multiple_of(2) {
        return Err("usage: --evaluate-unseen-behavioural <checkpoint.json> <parquet-dir> <subject-id,subject-id,...> <source.csv|.json> <session-id> ...".into());
    }
    let trainer = GeneratorTrainer::load_checkpoint(&args[2]).map_err(|e| e.to_string())?;
    let dataset = BehaviourParquetDataset::open(&args[3]).map_err(|e| e.to_string())?;
    let ids = args[4]
        .split(',')
        .map(|s| {
            s.parse::<u64>()
                .map_err(|_| format!("invalid unseen subject-id {s}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let exact = exact_source_legality(&args[5..])?;
    let report = trainer
        .evaluate_unseen(
            &dataset,
            &ids,
            &mut |_, hand: &lechiffre::behavioural::HandId, seq| {
                exact
                    .get(&(hand.clone(), seq))
                    .copied()
                    .ok_or_else(|| format!("source decision not found: {hand}:{seq}"))
            },
        )
        .map_err(|e| e.to_string())?;
    print_ag_report(&report)
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() > 1 {
        let result = match args[1].as_str() {
            "--train-behavioural" => Some(train_ag(&args)),
            "--resume-behavioural" => Some(resume_ag(&args)),
            "--evaluate-behavioural" => Some(evaluate_ag(&args)),
            "--evaluate-unseen-behavioural" => Some(evaluate_unseen_ag(&args)),
            _ => None,
        };
        if let Some(result) = result {
            if let Err(error) = result {
                eprintln!("behavioural training failed: {error}");
                std::process::exit(1);
            }
            return;
        }
    }
    if args.len() > 1 && args[1] == "--export-poker-with-the-boys-parquet" {
        export_poker_with_the_boys_parquet(&args);
        return;
    }
    if args.len() > 1 && args[1] == "--behaviour-dataset-stats" {
        print_behaviour_dataset_stats(&args);
        return;
    }
    if args.len() > 1 && args[1] == "--import-poker-with-the-boys" {
        if args.len() != 5 {
            eprintln!(
                "usage: lechiffre --import-poker-with-the-boys <file.csv|file.json> <session-id> <canonical-player>"
            );
            std::process::exit(2);
        }
        let source = std::fs::read_to_string(&args[2]).expect("failed to read input file");
        let normalizer = datasets::poker_with_the_boys::normalize_player_name;
        let imported = if args[2].ends_with(".json") {
            adapters::pokernow::PokerNowAdapter::adapt_pokerlater_json(
                &args[3],
                datasets::poker_with_the_boys::IDENTITY_NAMESPACE,
                &source,
                normalizer,
            )
        } else if args[2].ends_with(".csv") {
            adapters::pokernow::PokerNowAdapter::adapt_csv(
                &args[3],
                datasets::poker_with_the_boys::IDENTITY_NAMESPACE,
                &source,
                normalizer,
            )
        } else {
            eprintln!("unsupported extension: expected .csv or .json");
            std::process::exit(2);
        }
        .expect("invalid input session format");
        let id = adapters::PlayerId::from_source(
            datasets::poker_with_the_boys::IDENTITY_NAMESPACE,
            &args[4],
        );
        let history = adapters::hand_history::HandHistoryAdapter::history_for(&id, &imported.hands);
        println!(
            "accepted_hands={} rejected_hands={} subject_hands={}",
            imported.hands.len(),
            imported.rejected.len(),
            history.len()
        );
        for rejection in &imported.rejected {
            eprintln!(
                "rejected hand {}: {}",
                rejection.source_hand_id, rejection.reason
            );
        }
        // Per-hand rejects are reported above; retained hands remain usable.
        return;
    }

    let game_state = GameStateBuilder::new()
        .num_players_with_stack(2, 100)
        .blinds(2, 1)
        .dealer_idx(0)
        .max_raises_per_round(None)
        .build()
        .expect("failed to build game state");

    let agents: Vec<Box<dyn Agent>> = vec![
        Box::new(RandomAgent::new_with_seed(
            "Random-0",
            vec![0.25, 0.30, 0.50],
            vec![0.50, 0.60, 0.45],
            1,
        )),
        Box::new(RandomAgent::new_with_seed(
            "Random-1",
            vec![0.25, 0.30, 0.50],
            vec![0.50, 0.60, 0.45],
            2,
        )),
    ];

    let adapter = RsPokerAdapter::new();
    let agents = adapter
        .wrap_agents(agents)
        .expect("LeChiffre supports only 2..=6-max");

    let mut simulation = HoldemSimulationBuilder::default()
        .game_state(game_state)
        .agents(agents)
        .historians(vec![adapter.historian()])
        .build()
        .expect("failed to build simulation");

    let hand_id = simulation.id;
    simulation.run().await;

    let sequence = adapter
        .sequence(hand_id)
        .expect("failed to read behavioural sequence")
        .expect("behavioural sequence was not recorded");

    println!("Random vs Random completed");
    println!("Behavioural events: {}", sequence.len());
    println!("Final stacks: {:?}", simulation.game_state.stacks);
}
