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

/// Atomically publishes new JSON without replacing an existing output.
fn commit_new_json_manifest(
    output: &std::path::Path,
    manifest: &serde_json::Value,
) -> Result<(), String> {
    use std::io::Write;

    let filename = output
        .file_name()
        .ok_or_else(|| format!("invalid output manifest path: {}", output.display()))?
        .to_string_lossy();
    let temporary = output.with_file_name(format!(
        ".{filename}.{}.inprogress",
        std::process::id()
    ));
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .map_err(|e| format!("cannot create {}: {e}", temporary.display()))?;
    let result: Result<(), String> = (|| {
        serde_json::to_writer_pretty(&mut file, manifest).map_err(|e| e.to_string())?;
        file.write_all(b"\n").map_err(|e| e.to_string())?;
        file.sync_all().map_err(|e| e.to_string())?;
        // On Unix, hard_link fails with AlreadyExists instead of overwriting.
        std::fs::hard_link(&temporary, output).map_err(|e| {
            format!(
                "cannot commit new manifest {} (existing files are never replaced): {e}",
                output.display()
            )
        })?;
        Ok(())
    })();
    drop(file);
    let removed = std::fs::remove_file(&temporary);
    result?;
    removed.map_err(|e| format!("could not clean up {}: {e}", temporary.display()))?;
    Ok(())
}

/// Exports an immutable manifest of complete hands, not a split of event rows.
/// `hand_index` denotes ingestion order only; source-time chronology is unverified.
fn split_behaviour_dataset_hands(args: &[String]) -> Result<(), String> {
    use lechiffre::datasets::behaviour_parquet_loader::HAND_SPLIT_ALGORITHM_VERSION;
    use serde_json::json;

    if args.len() != 4 {
        return Err(
            "usage: --split-behaviour-dataset-hands <parquet-dir> <new-manifest.json>".into(),
        );
    }
    let dataset = BehaviourParquetDataset::open(&args[2]).map_err(|e| e.to_string())?;
    let split = dataset
        .split_hands_by_ingestion(7_000, 1_500, 1_500)
        .map_err(|e| e.to_string())?;
    let stats = dataset.summary();

    // Existing hierarchical query hands are used for training, NOT validation.
    // These strict train/validation/test boundaries are intentionally not wired
    // into HierarchicalHandSplit, to prevent validation-to-training leakage.
    let to_entries = |hands: &[lechiffre::datasets::behaviour_parquet_loader::HandPartitionEntry]| {
        hands.iter().map(|entry| {
            json!({
                "hand_id": entry.hand_id.as_str(),
                "hand_index": entry.hand_index,
                "decisions": entry.decision_count,
                "players_at_table": entry.players_at_table,
            })
        }).collect::<Vec<_>>()
    };
    let subject_counts = split.subject_hand_counts().iter().map(|(&subject_id, &counts)| {
        json!({
            "subject_id": subject_id,
            "train_hands": counts[0],
            "validation_hands": counts[1],
            "test_hands": counts[2],
        })
    }).collect::<Vec<_>>();
    let decisions = split.decision_counts();
    let memberships = split.subject_hand_rows();
    let manifest = json!({
        "format": "lechiffre.ingestion-hand-split.v1",
        "algorithm_version": HAND_SPLIT_ALGORITHM_VERSION,
        "ordering": "persisted_hand_index_ingestion_order_not_verified_wall_clock_chronology",
        "ratios_basis_points": {"train": 7000, "validation": 1500, "test": 1500},
        "rounding": "floor_train_and_validation_remainder_to_test",
        "dataset": {
            "hands": stats.hands_written,
            "decisions": stats.events_written,
            "subject_hand_rows": stats.subject_hand_rows_written,
            "fingerprint_fnv128_noncryptographic": dataset.dataset_fingerprint(),
        },
        "boundaries": {
            "first_validation_hand_index": split.first_validation_hand_index(),
            "first_test_hand_index": split.first_test_hand_index(),
        },
        "hierarchical_training_integration": "not_compatible_query_partition_is_used_for_training",
        "partitions": {
            "train": {
                "hands": split.train().len(),
                "decisions": decisions[0],
                "subject_hand_rows": memberships[0],
                "entries": to_entries(split.train()),
            },
            "validation": {
                "hands": split.validation().len(),
                "decisions": decisions[1],
                "subject_hand_rows": memberships[1],
                "entries": to_entries(split.validation()),
            },
            "test": {
                "hands": split.test().len(),
                "decisions": decisions[2],
                "subject_hand_rows": memberships[2],
                "entries": to_entries(split.test()),
            },
        },
        "subjects": subject_counts,
    });

    let output = std::path::Path::new(&args[3]);
    commit_new_json_manifest(output, &manifest)?;
    println!(
        "manifest={} train_hands={} validation_hands={} test_hands={} train_decisions={} validation_decisions={} test_decisions={} first_validation_hand_index={} first_test_hand_index={} ordering=ingestion_not_verified_chronological",
        output.display(),
        split.train().len(),
        split.validation().len(),
        split.test().len(),
        decisions[0],
        decisions[1],
        decisions[2],
        split.first_validation_hand_index(),
        split.first_test_hand_index(),
    );
    Ok(())
}

/// Strict hierarchical run specification. No compatibility with the old A/G config.
struct BehaviourRunConfig {
    split: lechiffre::datasets::behaviour_parquet_loader::HierarchicalHandSplit,
    latent_dimension: usize,
    generator_hidden: usize,
    analyzer_hidden: usize,
    generator_lr: f64,
    analyzer_lr: f64,
    blueprint_lr: f64,
    baseline_lr: f64,
    baseline_max_step_prior_std: f64,
    beta_kl: f64,
    seed: u64,
    z0_initial: Vec<f32>,
    z0_prior_std: Vec<f32>,
    blueprint_epochs: usize,
    personalization_history_hands: Vec<usize>,
    momentary_history_hands: usize,
    momentary_epochs: usize,
    subjects: Vec<u64>,
}

fn configuration_object(path: &str) -> Result<serde_json::Value, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("config {path}: {e}"))?;
    serde_json::from_str(&text).map_err(|e| format!("config {path}: {e}"))
}

fn require_exact_keys(value: &serde_json::Value, keys: &[&str]) -> Result<(), String> {
    let object = value
        .as_object()
        .ok_or("configuration must be a JSON object")?;
    if object.len() != keys.len() || keys.iter().any(|key| !object.contains_key(*key)) {
        return Err(format!("configuration keys must be exactly: {keys:?}"));
    }
    Ok(())
}
fn config_u64(v: &serde_json::Value, key: &str) -> Result<u64, String> {
    v[key]
        .as_u64()
        .ok_or_else(|| format!("{key}: expected unsigned integer"))
}
fn config_usize(v: &serde_json::Value, key: &str) -> Result<usize, String> {
    usize::try_from(config_u64(v, key)?).map_err(|_| format!("{key}: integer overflow"))
}
fn config_float(v: &serde_json::Value, key: &str) -> Result<f64, String> {
    v[key]
        .as_f64()
        .filter(|x| x.is_finite())
        .ok_or_else(|| format!("{key}: expected finite number"))
}
fn config_unsigned_list(v: &serde_json::Value, key: &str) -> Result<Vec<u64>, String> {
    v[key]
        .as_array()
        .ok_or_else(|| format!("{key}: expected array"))?
        .iter()
        .map(|x| {
            x.as_u64()
                .ok_or_else(|| format!("{key}: expected unsigned ids"))
        })
        .collect()
}
fn config_sizes(v: &serde_json::Value, key: &str) -> Result<Vec<usize>, String> {
    config_unsigned_list(v, key)?
        .into_iter()
        .map(|x| usize::try_from(x).map_err(|_| format!("{key}: index overflow")))
        .collect()
}
fn config_f32_vec(v: &serde_json::Value, key: &str) -> Result<Vec<f32>, String> {
    v[key]
        .as_array()
        .ok_or_else(|| format!("{key}: expected array"))?
        .iter()
        .map(|x| {
            let f = x
                .as_f64()
                .filter(|f| f.is_finite())
                .ok_or_else(|| format!("{key}: expected finite numbers"))?;
            let f = f as f32;
            if !f.is_finite() {
                Err(format!("{key}: float32 range exceeded"))
            } else {
                Ok(f)
            }
        })
        .collect()
}
fn run_configuration(path: &str) -> Result<BehaviourRunConfig, String> {
    use lechiffre::datasets::behaviour_parquet_loader::HierarchicalHandSplit;
    let v = configuration_object(path)?;
    let keys = [
        "latent_dimension",
        "generator_hidden",
        "analyzer_hidden",
        "generator_lr",
        "analyzer_lr",
        "blueprint_lr",
        "baseline_lr",
        "baseline_max_step_prior_std",
        "beta_kl",
        "seed",
        "z0_initial",
        "z0_prior_std",
        "first_query_hand_index",
        "first_evaluation_hand_index",
        "support_hands",
        "query_hands",
        "evaluation_hands",
        "blueprint_epochs",
        "personalization_history_hands",
        "momentary_history_hands",
        "momentary_epochs",
        "subject_ids",
    ];
    require_exact_keys(&v, &keys)?;
    let split = HierarchicalHandSplit::new(
        config_u64(&v, "first_query_hand_index")?,
        config_u64(&v, "first_evaluation_hand_index")?,
        config_usize(&v, "support_hands")?,
        config_usize(&v, "query_hands")?,
        config_usize(&v, "evaluation_hands")?,
    )
    .map_err(|e| e.to_string())?;
    let history = config_sizes(&v, "personalization_history_hands")?;
    let momentary_history = config_usize(&v, "momentary_history_hands")?;
    let dim = config_usize(&v, "latent_dimension")?;
    let initial = config_f32_vec(&v, "z0_initial")?;
    let std = config_f32_vec(&v, "z0_prior_std")?;
    let subjects = config_unsigned_list(&v, "subject_ids")?;
    if config_usize(&v, "blueprint_epochs")? == 0
        || history.is_empty()
        || config_usize(&v, "momentary_epochs")? == 0
        || dim == 0
        || initial.len() != dim
        || std.len() != dim
        || subjects.is_empty()
        || history.iter().any(|&n| n > split.support_hands)
        || momentary_history > split.support_hands
        || std.iter().any(|s| *s <= 0.0)
        || [
            "generator_lr",
            "analyzer_lr",
            "blueprint_lr",
            "baseline_lr",
            "baseline_max_step_prior_std",
            "beta_kl",
        ]
        .iter()
        .any(|field| match config_float(&v, field) {
            Ok(x) => x <= 0.0,
            Err(_) => true,
        })
    {
        return Err("invalid hierarchical configuration: all phases, model dimensions, and rates are strictly required".into());
    }
    Ok(BehaviourRunConfig {
        split,
        latent_dimension: dim,
        generator_hidden: config_usize(&v, "generator_hidden")?,
        analyzer_hidden: config_usize(&v, "analyzer_hidden")?,
        generator_lr: config_float(&v, "generator_lr")?,
        analyzer_lr: config_float(&v, "analyzer_lr")?,
        blueprint_lr: config_float(&v, "blueprint_lr")?,
        baseline_lr: config_float(&v, "baseline_lr")?,
        baseline_max_step_prior_std: config_float(&v, "baseline_max_step_prior_std")?,
        beta_kl: config_float(&v, "beta_kl")?,
        seed: config_u64(&v, "seed")?,
        z0_initial: initial,
        z0_prior_std: std,
        blueprint_epochs: config_usize(&v, "blueprint_epochs")?,
        personalization_history_hands: history,
        momentary_history_hands: momentary_history,
        momentary_epochs: config_usize(&v, "momentary_epochs")?,
        subjects,
    })
}

struct ResumeConfig {
    blueprint_extra_epochs: usize,
    personalization_history_hands: Vec<usize>,
    momentary_history_hands: usize,
    momentary_extra_epochs: usize,
}
fn resume_configuration(path: &str) -> Result<ResumeConfig, String> {
    let v = configuration_object(path)?;
    require_exact_keys(
        &v,
        &[
            "blueprint_extra_epochs",
            "personalization_history_hands",
            "momentary_history_hands",
            "momentary_extra_epochs",
        ],
    )?;
    let result = ResumeConfig {
        blueprint_extra_epochs: config_usize(&v, "blueprint_extra_epochs")?,
        personalization_history_hands: config_sizes(&v, "personalization_history_hands")?,
        momentary_history_hands: config_usize(&v, "momentary_history_hands")?,
        momentary_extra_epochs: config_usize(&v, "momentary_extra_epochs")?,
    };
    if result.blueprint_extra_epochs == 0
        && result.personalization_history_hands.is_empty()
        && result.momentary_extra_epochs == 0
    {
        return Err("resume configuration requests no additional training".into());
    }
    Ok(result)
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

fn print_hierarchical_epoch(
    r: &lechiffre::datasets::behaviour_parquet_loader::HierarchicalEpochReport,
) {
    println!(
        "phase={:?} decisions={} mean_nll={:.8} conditional_KL={:?}",
        r.phase, r.decisions, r.mean_nll, r.mean_conditional_kl
    );
}
fn print_hierarchical_validation(
    r: &lechiffre::datasets::behaviour_parquet_loader::HierarchicalValidationReport,
) {
    println!(
        "validation_decisions={} z0_predictive_nll={:.8} z1_predictive_nll={:.8} z2_prior_predictive_nll={:.8} z2_posterior_RECONSTRUCTION_nll={:.8} mean_KL={:.8}",
        r.decisions,
        r.z0_predictive_nll,
        r.z1_predictive_nll,
        r.z2_prior_predictive_nll,
        r.z2_posterior_reconstruction_nll,
        r.mean_conditional_kl
    );
}

type ExactReplayMap = std::collections::HashMap<
    (lechiffre::behavioural::HandId, u32),
    lechiffre::behavioural::LegalActionSpace,
>;

fn replay_provider<'a>(
    exact: &'a ExactReplayMap,
) -> impl FnMut(
    u64,
    &lechiffre::behavioural::HandId,
    u32,
) -> Result<lechiffre::behavioural::LegalActionSpace, String>
+ 'a {
    move |_, hand, seq| {
        exact
            .get(&(hand.clone(), seq))
            .copied()
            .ok_or_else(|| format!("exact replay decision not found: {hand}:{seq}"))
    }
}

#[allow(clippy::too_many_arguments)]
fn execute_phases(
    trainer: &mut lechiffre::datasets::behaviour_parquet_loader::HierarchicalTrainer,
    dataset: &lechiffre::datasets::behaviour_parquet_loader::BehaviourParquetDataset,
    split: lechiffre::datasets::behaviour_parquet_loader::HierarchicalHandSplit,
    subjects: &[u64],
    blueprint_epochs: usize,
    personalization_histories: &[usize],
    momentary_history: usize,
    momentary_epochs: usize,
    exact: &ExactReplayMap,
) -> Result<(), String> {
    for _ in 0..blueprint_epochs {
        let report = trainer
            .train_blueprint_epoch(dataset, subjects, split, &mut replay_provider(exact))
            .map_err(|e| e.to_string())?;
        print_hierarchical_epoch(&report);
    }
    for &history in personalization_histories {
        let report = trainer
            .train_personalization_epoch(
                dataset,
                subjects,
                split,
                history,
                &mut replay_provider(exact),
            )
            .map_err(|e| e.to_string())?;
        print_hierarchical_epoch(&report);
    }
    for _ in 0..momentary_epochs {
        let report = trainer
            .train_momentary_epoch(
                dataset,
                subjects,
                split,
                momentary_history,
                &mut replay_provider(exact),
            )
            .map_err(|e| e.to_string())?;
        print_hierarchical_epoch(&report);
    }
    Ok(())
}

fn train_hierarchical(args: &[String]) -> Result<(), String> {
    use lechiffre::behavioural::{
        Adam, AnalyzerAdam, BaselineAdaptationConfig, BehaviourBlueprint, BehaviourLatentConfig,
        NeuralBehaviourAnalyzer, NeuralBehaviourGenerator,
    };
    use lechiffre::datasets::behaviour_parquet_loader::{
        BehaviourParquetDataset, HierarchicalTrainer,
    };
    if args.len() < 7 || !(args.len() - 5).is_multiple_of(2) {
        return Err("usage: --train-behavioural <hierarchical-config.json> <parquet-dir> <checkpoint-out.json> <source.csv|.json> <session-id> ...".into());
    }
    let run = run_configuration(&args[2])?;
    let dataset = BehaviourParquetDataset::open(&args[3]).map_err(|e| e.to_string())?;
    let exact = exact_source_legality(&args[5..])?;
    let latent = BehaviourLatentConfig::new(run.latent_dimension).map_err(|e| e.to_string())?;
    let blueprint = BehaviourBlueprint::new(
        latent.profile(run.z0_initial).map_err(|e| e.to_string())?,
        run.z0_prior_std,
    )
    .map_err(|e| e.to_string())?;
    let g = NeuralBehaviourGenerator::initialize(latent, run.generator_hidden, run.seed)
        .map_err(|e| e.to_string())?;
    let a_seed = run.seed.checked_add(1).ok_or("seed overflow")?;
    let noise_seed = run.seed.checked_add(2).ok_or("seed overflow")?;
    let a = NeuralBehaviourAnalyzer::initialize(latent, run.analyzer_hidden, a_seed)
        .map_err(|e| e.to_string())?;
    let go = Adam::new(&g, run.generator_lr).map_err(|e| e.to_string())?;
    let ao = AnalyzerAdam::new(&a, run.analyzer_lr).map_err(|e| e.to_string())?;
    let baseline = BaselineAdaptationConfig::new(run.baseline_lr, run.baseline_max_step_prior_std)
        .map_err(|e| e.to_string())?;
    let mut trainer = HierarchicalTrainer::new(
        blueprint,
        g,
        a,
        go,
        ao,
        baseline,
        run.blueprint_lr,
        run.beta_kl,
        noise_seed,
    )
    .map_err(|e| e.to_string())?;
    execute_phases(
        &mut trainer,
        &dataset,
        run.split,
        &run.subjects,
        run.blueprint_epochs,
        &run.personalization_history_hands,
        run.momentary_history_hands,
        run.momentary_epochs,
        &exact,
    )?;
    let report = trainer
        .evaluate(
            &dataset,
            &run.subjects,
            run.split,
            &mut replay_provider(&exact),
        )
        .map_err(|e| e.to_string())?;
    print_hierarchical_validation(&report);
    trainer
        .save_checkpoint(&args[4])
        .map_err(|e| e.to_string())?;
    println!(
        "checkpoint={} model_id={}",
        args[4],
        trainer.model_id().map_err(|e| e.to_string())?
    );
    Ok(())
}

fn resume_hierarchical(args: &[String]) -> Result<(), String> {
    use lechiffre::datasets::behaviour_parquet_loader::{
        BehaviourParquetDataset, HierarchicalTrainer,
    };
    if args.len() < 8 || !(args.len() - 6).is_multiple_of(2) {
        return Err("usage: --resume-behavioural <checkpoint-in.json> <parquet-dir> <checkpoint-out.json> <resume-config.json> <source.csv|.json> <session-id> ...".into());
    }
    let mut trainer = HierarchicalTrainer::load_checkpoint(&args[2]).map_err(|e| e.to_string())?;
    let dataset = BehaviourParquetDataset::open(&args[3]).map_err(|e| e.to_string())?;
    let config = resume_configuration(&args[5])?;
    let split = trainer
        .training_split()
        .ok_or("checkpoint missing chronological split")?;
    let subjects = trainer.training_subjects().to_vec();
    let exact = exact_source_legality(&args[6..])?;
    execute_phases(
        &mut trainer,
        &dataset,
        split,
        &subjects,
        config.blueprint_extra_epochs,
        &config.personalization_history_hands,
        config.momentary_history_hands,
        config.momentary_extra_epochs,
        &exact,
    )?;
    if trainer.completed_epochs().2 > 0 {
        let report = trainer
            .evaluate(&dataset, &subjects, split, &mut replay_provider(&exact))
            .map_err(|e| e.to_string())?;
        print_hierarchical_validation(&report);
    }
    trainer
        .save_checkpoint(&args[4])
        .map_err(|e| e.to_string())?;
    println!(
        "checkpoint={} model_id={}",
        args[4],
        trainer.model_id().map_err(|e| e.to_string())?
    );
    Ok(())
}

fn evaluate_hierarchical(args: &[String], unseen: bool) -> Result<(), String> {
    use lechiffre::datasets::behaviour_parquet_loader::{
        BehaviourParquetDataset, HierarchicalTrainer,
    };
    let start = if unseen { 5 } else { 4 };
    if args.len() < start + 2 || !(args.len() - start).is_multiple_of(2) {
        return Err(if unseen {
            "usage: --evaluate-unseen-behavioural <checkpoint.json> <parquet-dir> <subject-ids-comma-separated> <source.csv|.json> <session-id> ..."
        } else {
            "usage: --evaluate-behavioural <checkpoint.json> <parquet-dir> <source.csv|.json> <session-id> ..."
        }.into());
    }
    let trainer = HierarchicalTrainer::load_checkpoint(&args[2]).map_err(|e| e.to_string())?;
    let dataset = BehaviourParquetDataset::open(&args[3]).map_err(|e| e.to_string())?;
    let split = trainer.training_split().ok_or("checkpoint missing split")?;
    let subjects = if unseen {
        args[4]
            .split(',')
            .map(|id| {
                id.parse::<u64>()
                    .map_err(|_| format!("invalid subject id {id}"))
            })
            .collect::<Result<Vec<_>, _>>()?
    } else {
        trainer.training_subjects().to_vec()
    };
    let exact = exact_source_legality(&args[start..])?;
    let report = if unseen {
        trainer.evaluate_unseen(&dataset, &subjects, split, &mut replay_provider(&exact))
    } else {
        trainer.evaluate(&dataset, &subjects, split, &mut replay_provider(&exact))
    }
    .map_err(|e| e.to_string())?;
    print_hierarchical_validation(&report);
    Ok(())
}

fn player_baseline_cli(args: &[String], advance: bool) -> Result<(), String> {
    use lechiffre::behavioural::{PlayerBaselineAdapter, PlayerBaselineState};
    use lechiffre::datasets::behaviour_parquet_loader::{
        BehaviourParquetDataset, HierarchicalTrainer,
    };
    let source_start = if advance { 8 } else { 7 };
    if args.len() < source_start + 2 || !(args.len() - source_start).is_multiple_of(2) {
        return Err(if advance {
            "usage: --advance-player-baseline <model-checkpoint.json> <parquet-dir> <subject-id> <before-hand-index> <baseline-in.json> <baseline-out.json> <source.csv|.json> <session-id> ..."
        } else {
            "usage: --create-player-baseline <model-checkpoint.json> <parquet-dir> <subject-id> <before-hand-index> <baseline-out.json> <source.csv|.json> <session-id> ..."
        }.into());
    }
    let trainer = HierarchicalTrainer::load_checkpoint(&args[2]).map_err(|e| e.to_string())?;
    if trainer.completed_epochs().2 == 0 {
        return Err("model requires completed momentary training".into());
    }
    let dataset = BehaviourParquetDataset::open(&args[3]).map_err(|e| e.to_string())?;
    let subject = args[4].parse::<u64>().map_err(|_| "invalid subject id")?;
    let cutoff = args[5]
        .parse::<u64>()
        .map_err(|_| "invalid before-hand-index")?;
    let model_id = trainer.model_id().map_err(|e| e.to_string())?;
    let adapter =
        PlayerBaselineAdapter::new(trainer.blueprint().clone(), trainer.baseline_config())
            .map_err(|e| e.to_string())?;
    let mut state = if advance {
        PlayerBaselineState::load_checkpoint(&args[6], subject, &model_id, &adapter)
            .map_err(|e| e.to_string())?
    } else {
        adapter.initialize_player()
    };
    let exact = exact_source_legality(&args[source_start..])?;
    dataset
        .advance_baseline_before(
            subject,
            cutoff,
            trainer.generator(),
            &adapter,
            &mut state,
            &mut replay_provider(&exact),
        )
        .map_err(|e| e.to_string())?;
    let out = if advance { &args[7] } else { &args[6] };
    state
        .save_checkpoint(out, subject, &model_id)
        .map_err(|e| e.to_string())?;
    println!(
        "baseline={} subject={} model_id={} observed_hands={} observed_decisions={} covered_before={:?}",
        out,
        subject,
        model_id,
        state.baseline().observed_hands(),
        state.observed_decisions(),
        state.processed_before_hand_index()
    );
    Ok(())
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() > 1 && args[1] == "--split-behaviour-dataset-hands" {
        if let Err(error) = split_behaviour_dataset_hands(&args) {
            eprintln!("behavioural hand split failed: {error}");
            std::process::exit(1);
        }
        return;
    }
    if args.len() > 1 {
        let result = match args[1].as_str() {
            "--train-behavioural" => Some(train_hierarchical(&args)),
            "--resume-behavioural" => Some(resume_hierarchical(&args)),
            "--evaluate-behavioural" => Some(evaluate_hierarchical(&args, false)),
            "--evaluate-unseen-behavioural" => Some(evaluate_hierarchical(&args, true)),
            "--create-player-baseline" => Some(player_baseline_cli(&args, false)),
            "--advance-player-baseline" => Some(player_baseline_cli(&args, true)),
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

    if args.len() > 1 && args[1] != "--simulate-random" {
        eprintln!("unknown or obsolete command: {}", args[1]);
        std::process::exit(2);
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

#[cfg(test)]
mod hierarchical_cli_tests {
    use super::*;

    #[test]
    fn hand_split_manifest_refuses_overwriting_an_existing_file() {
        let path = std::env::temp_dir().join(format!(
            "lechiffre-hand-split-{}.json",
            std::process::id()
        ));
        let original = serde_json::json!({"format": "first"});
        let replacement = serde_json::json!({"format": "second"});
        commit_new_json_manifest(&path, &original).unwrap();
        assert!(commit_new_json_manifest(&path, &replacement).is_err());
        let actual: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(actual, original);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_obsolete_and_incomplete_configs() {
        let p = std::env::temp_dir().join(format!(
            "lechiffre-hierarchical-cli-{}.json",
            std::process::id()
        ));
        let original: serde_json::Value =
            serde_json::from_str(include_str!("../configs/hierarchical.example.json")).unwrap();
        std::fs::write(&p, serde_json::to_vec(&original).unwrap()).unwrap();
        assert!(run_configuration(p.to_str().unwrap()).is_ok());
        let mut missing = original.clone();
        missing.as_object_mut().unwrap().remove("z0_initial");
        std::fs::write(&p, serde_json::to_vec(&missing).unwrap()).unwrap();
        assert!(run_configuration(p.to_str().unwrap()).is_err());
        let mut obsolete = original;
        obsolete
            .as_object_mut()
            .unwrap()
            .insert("validation_draws".into(), serde_json::json!(16));
        std::fs::write(&p, serde_json::to_vec(&obsolete).unwrap()).unwrap();
        assert!(run_configuration(p.to_str().unwrap()).is_err());
        std::fs::remove_file(p).unwrap();
    }
}
