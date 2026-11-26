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
    let mut exact_legality = ExactReplayMap::new();

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

        merge_exact_legality(&mut exact_legality, imported.exact_legal_actions).unwrap_or_else(
            |error| {
                eprintln!("invalid exact replay: {error}");
                std::process::exit(1)
            },
        );

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

    BehaviourParquetDataset::attach_exact_legality(output_dir, &exact_legality).unwrap_or_else(
        |error| {
            eprintln!("failed to seal exact legality in {output_dir}: {error}");
            std::process::exit(1);
        },
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
fn commit_new_json(output: &std::path::Path, manifest: &serde_json::Value) -> Result<(), String> {
    use std::io::Write;

    let filename = output
        .file_name()
        .ok_or_else(|| format!("invalid output JSON path: {}", output.display()))?
        .to_string_lossy();
    let temporary = output.with_file_name(format!(".{filename}.{}.inprogress", std::process::id()));
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
                "cannot commit new JSON {} (existing files are never replaced): {e}",
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
    if args.len() != 4 {
        return Err(
            "usage: --split-behaviour-dataset-hands <parquet-dir> <new-manifest.json>".into(),
        );
    }
    let dataset = BehaviourParquetDataset::open(&args[2]).map_err(|e| e.to_string())?;
    let split = dataset
        .split_hands_by_ingestion(7_000, 1_500, 1_500)
        .map_err(|e| e.to_string())?;
    let manifest = split.manifest_value(&dataset);
    let decisions = split.decision_counts();
    let output = std::path::Path::new(&args[3]);
    commit_new_json(output, &manifest)?;
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
    episode: lechiffre::datasets::behaviour_parquet_loader::TrainingEpisodeConfig,
    optimization: lechiffre::datasets::behaviour_parquet_loader::HierarchicalOptimizationConfig,
    latent_dimension: usize,
    generator_hidden: usize,
    analyzer_hidden: usize,
    generator_gn_damping: f64,
    analyzer_lr: f64,
    blueprint_lr: f64,
    baseline_lr: f64,
    baseline_max_step_prior_std: f64,
    beta_kl: f64,
    seed: u64,
    z0_initial: Vec<f32>,
    z0_prior_std: Vec<f32>,
    blueprint_epochs: usize,
    personalization_epochs: usize,
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
    use lechiffre::datasets::behaviour_parquet_loader::{
        HierarchicalOptimizationConfig, TrainingEpisodeConfig,
    };
    let v = configuration_object(path)?;
    let keys = [
        "latent_dimension",
        "generator_hidden",
        "analyzer_hidden",
        "generator_gn_damping",
        "analyzer_lr",
        "blueprint_lr",
        "baseline_lr",
        "baseline_max_step_prior_std",
        "beta_kl",
        "seed",
        "z0_initial",
        "z0_prior_std",
        "support_hands",
        "query_hands",
        "blueprint_epochs",
        "blueprint_batch_decisions",
        "personalization_epochs",
        "momentary_history_hands",
        "momentary_epochs",
        "subject_ids",
    ];
    require_exact_keys(&v, &keys)?;
    let episode = TrainingEpisodeConfig::new(
        config_usize(&v, "support_hands")?,
        config_usize(&v, "query_hands")?,
    )
    .map_err(|e| e.to_string())?;
    let optimization =
        HierarchicalOptimizationConfig::new(config_usize(&v, "blueprint_batch_decisions")?)
            .map_err(|e| e.to_string())?;
    let personalization_epochs = config_usize(&v, "personalization_epochs")?;
    let momentary_history = config_usize(&v, "momentary_history_hands")?;
    let dim = config_usize(&v, "latent_dimension")?;
    let initial = config_f32_vec(&v, "z0_initial")?;
    let std = config_f32_vec(&v, "z0_prior_std")?;
    let subjects = config_unsigned_list(&v, "subject_ids")?;
    if config_usize(&v, "blueprint_epochs")? == 0
        || (personalization_epochs == 0 && config_usize(&v, "momentary_epochs")? > 0)
        || dim == 0
        || initial.len() != dim
        || std.len() != dim
        || subjects.is_empty()
        || momentary_history > episode.support_hands
        || std.iter().any(|s| *s <= 0.0)
        || [
            "generator_gn_damping",
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
        return Err("invalid hierarchical configuration: explicit valid phases, model dimensions, and rates required".into());
    }
    Ok(BehaviourRunConfig {
        episode,
        optimization,
        latent_dimension: dim,
        generator_hidden: config_usize(&v, "generator_hidden")?,
        analyzer_hidden: config_usize(&v, "analyzer_hidden")?,
        generator_gn_damping: config_float(&v, "generator_gn_damping")?,
        analyzer_lr: config_float(&v, "analyzer_lr")?,
        blueprint_lr: config_float(&v, "blueprint_lr")?,
        baseline_lr: config_float(&v, "baseline_lr")?,
        baseline_max_step_prior_std: config_float(&v, "baseline_max_step_prior_std")?,
        beta_kl: config_float(&v, "beta_kl")?,
        seed: config_u64(&v, "seed")?,
        z0_initial: initial,
        z0_prior_std: std,
        blueprint_epochs: config_usize(&v, "blueprint_epochs")?,
        personalization_epochs,
        momentary_history_hands: momentary_history,
        momentary_epochs: config_usize(&v, "momentary_epochs")?,
        subjects,
    })
}

struct ResumeConfig {
    blueprint_extra_epochs: usize,
    personalization_extra_epochs: usize,
    momentary_history_hands: usize,
    momentary_extra_epochs: usize,
}
fn resume_configuration(path: &str) -> Result<ResumeConfig, String> {
    let v = configuration_object(path)?;
    require_exact_keys(
        &v,
        &[
            "blueprint_extra_epochs",
            "personalization_extra_epochs",
            "momentary_history_hands",
            "momentary_extra_epochs",
        ],
    )?;
    let result = ResumeConfig {
        blueprint_extra_epochs: config_usize(&v, "blueprint_extra_epochs")?,
        personalization_extra_epochs: config_usize(&v, "personalization_extra_epochs")?,
        momentary_history_hands: config_usize(&v, "momentary_history_hands")?,
        momentary_extra_epochs: config_usize(&v, "momentary_extra_epochs")?,
    };
    if result.blueprint_extra_epochs == 0
        && result.personalization_extra_epochs == 0
        && result.momentary_extra_epochs == 0
    {
        return Err("resume configuration requests no additional training".into());
    }
    Ok(result)
}

fn merge_exact_legality(target: &mut ExactReplayMap, next: ExactReplayMap) -> Result<(), String> {
    use std::collections::hash_map::Entry;
    for (key, legal) in next {
        match target.entry(key) {
            Entry::Vacant(slot) => {
                slot.insert(legal);
            }
            Entry::Occupied(slot) => {
                let (hand, sequence) = slot.key();
                return Err(format!(
                    "duplicate exact replay legality for {hand}:{sequence}"
                ));
            }
        }
    }
    Ok(())
}

/// Raw sources are read only for a one-time exact-legality attachment.
/// Training, resume, validation, test and Z1 adaptation never call this.
fn exact_source_legality(args: &[String]) -> Result<ExactReplayMap, String> {
    if args.is_empty() || !args.len().is_multiple_of(2) {
        return Err("expected one or more <file.csv|file.json> <session-id> pairs".into());
    }
    let mut all = ExactReplayMap::new();
    for pair in args.as_chunks::<2>().0 {
        let imported = import_poker_with_the_boys_session(&pair[0], &pair[1]);
        for rejection in &imported.rejected {
            eprintln!(
                "excluded source hand session={} hand={}: {}",
                pair[1], rejection.source_hand_id, rejection.reason
            );
        }
        merge_exact_legality(&mut all, imported.exact_legal_actions)?;
    }
    if all.is_empty() {
        return Err("no exact replay legality was obtained".into());
    }
    Ok(all)
}

fn attach_behaviour_legality(args: &[String]) -> Result<(), String> {
    if args.len() < 5 || !(args.len() - 3).is_multiple_of(2) {
        return Err(
            "usage: --attach-behaviour-legality <parquet-dir> <source.csv|.json> <session-id> ..."
                .into(),
        );
    }
    let replay = exact_source_legality(&args[3..])?;
    let count = BehaviourParquetDataset::attach_exact_legality(&args[2], &replay)
        .map_err(|error| error.to_string())?;
    println!("exact_legality_sealed={} decisions={count}", args[2]);
    Ok(())
}

fn print_hierarchical_epoch(
    r: &lechiffre::datasets::behaviour_parquet_loader::HierarchicalEpochReport,
) {
    println!(
        "phase={:?} decisions={} optimizer_steps={} mean_objective={:.8} conditional_KL={:?}",
        r.phase, r.decisions, r.optimizer_steps, r.mean_objective, r.mean_conditional_kl
    );
}
fn validation_report_value(
    partition: lechiffre::datasets::behaviour_parquet_loader::EvaluationPartition,
    r: &lechiffre::datasets::behaviour_parquet_loader::HierarchicalValidationReport,
) -> serde_json::Value {
    let components =
        |c: &lechiffre::datasets::behaviour_parquet_loader::HierarchicalNllComponents| {
            serde_json::json!({"action":c.action, "sizing_given_wager":c.sizing,
                          "timing_log1p_ms":c.timing_log1p_ms})
        };
    serde_json::json!({
        "format":"lechiffre.behavioural.validation.v4",
        "partition":match partition {
            lechiffre::datasets::behaviour_parquet_loader::EvaluationPartition::Validation => "validation",
            lechiffre::datasets::behaviour_parquet_loader::EvaluationPartition::Test => "test",
        },
        "decisions":r.decisions,
        "wager_decisions":r.wager_decisions,
        "z2_prior_samples":r.z2_prior_samples,
        "history_hands":r.history_hands,
        "z0_action_accuracy":r.z0_action_accuracy,
        "z1_action_accuracy":r.z1_action_accuracy,
        "z2_prior_action_accuracy":r.z2_prior_action_accuracy,
        "z0_predictive_nll":r.z0_predictive_nll,
        "z1_predictive_nll":r.z1_predictive_nll,
        "z2_prior_marginal_nll":r.z2_prior_marginal_nll,
        "z2_posterior_reconstruction_nll":r.z2_posterior_reconstruction_nll,
        "z0_components":components(&r.z0_components),
        "z1_components":components(&r.z1_components),
        "z2_prior_marginal_components":components(&r.z2_prior_marginal_components),
        "mean_conditional_kl":r.mean_conditional_kl,
        "timing_measure":"normal_density_in_log1p_milliseconds",
        "sampling":"deterministic_common_standard_normal_stream",
        "ordering":"ingestion_not_verified_chronological",
    })
}

fn blueprint_report_value(
    r: &lechiffre::datasets::behaviour_parquet_loader::ReferenceValidationReport,
) -> serde_json::Value {
    serde_json::json!({
        "format":"lechiffre.behavioural.blueprint-validation.v1",
        "partition":"validation",
        "ordering":"ingestion_not_verified_chronological",
        "decisions":r.decisions,
        "wager_decisions":r.wager_decisions,
        "z0_predictive_nll":r.predictive_nll,
        "z0_action_accuracy":r.action_accuracy,
        "z0_components":{"action":r.components.action,"sizing_given_wager":r.components.sizing,
                         "timing_log1p_ms":r.components.timing_log1p_ms},
        "timing_measure":"normal_density_in_log1p_milliseconds",
    })
}

fn personalization_report_value(
    r: &lechiffre::datasets::behaviour_parquet_loader::ReferenceValidationReport,
    history_hands: usize,
) -> serde_json::Value {
    serde_json::json!({
        "format":"lechiffre.behavioural.personalization-validation.v1",
        "partition":"validation",
        "history_hands":history_hands,
        "decisions":r.decisions,
        "wager_decisions":r.wager_decisions,
        "z1_predictive_nll":r.predictive_nll,
        "z1_action_accuracy":r.action_accuracy,
        "z1_components":{"action":r.components.action,"sizing_given_wager":r.components.sizing,
                         "timing_log1p_ms":r.components.timing_log1p_ms},
        "timing_measure":"normal_density_in_log1p_milliseconds",
        "ordering":"ingestion_not_verified_chronological",
    })
}

fn print_hierarchical_validation(
    partition: lechiffre::datasets::behaviour_parquet_loader::EvaluationPartition,
    r: &lechiffre::datasets::behaviour_parquet_loader::HierarchicalValidationReport,
) {
    println!("{}", validation_report_value(partition, r));
}

type ExactReplayMap = std::collections::HashMap<
    (lechiffre::behavioural::HandId, u32),
    lechiffre::behavioural::LegalActionSpace,
>;

fn replay_provider(
    dataset: &BehaviourParquetDataset,
) -> impl FnMut(
    u64,
    &lechiffre::behavioural::HandId,
    u32,
) -> Result<lechiffre::behavioural::LegalActionSpace, String>
+ '_ {
    move |_, hand, sequence| {
        dataset
            .exact_legal_actions(hand, sequence)
            .map_err(|error| error.to_string())
    }
}

#[allow(clippy::too_many_arguments)]
fn execute_phases(
    trainer: &mut lechiffre::datasets::behaviour_parquet_loader::HierarchicalTrainer,
    dataset: &lechiffre::datasets::behaviour_parquet_loader::BehaviourParquetDataset,
    partitions: &lechiffre::datasets::behaviour_parquet_loader::DatasetPartitions,
    episode: lechiffre::datasets::behaviour_parquet_loader::TrainingEpisodeConfig,
    subjects: &[u64],
    blueprint_epochs: usize,
    personalization_epochs: usize,
    momentary_history: usize,
    momentary_epochs: usize,
) -> Result<(), String> {
    let (b, p, m) = trainer.completed_epochs();
    if (blueprint_epochs > 0 && (p > 0 || m > 0))
        || (personalization_epochs > 0 && m > 0)
        || (b == 0 && blueprint_epochs == 0)
        || (momentary_epochs > 0 && p == 0 && personalization_epochs == 0)
        || momentary_history > episode.support_hands
    {
        return Err("invalid training phase transition or history budget".into());
    }
    for _ in 0..blueprint_epochs {
        let report = trainer
            .train_blueprint_epoch(
                dataset,
                subjects,
                partitions,
                episode,
                &mut replay_provider(dataset),
            )
            .map_err(|e| e.to_string())?;
        print_hierarchical_epoch(&report);
        let validation = trainer
            .evaluate_blueprint(dataset, partitions, &mut replay_provider(dataset))
            .map_err(|e| e.to_string())?;
        println!("{}", blueprint_report_value(&validation));
    }
    for _ in 0..personalization_epochs {
        let report = trainer
            .train_personalization_epoch(
                dataset,
                subjects,
                partitions,
                episode,
                &mut replay_provider(dataset),
            )
            .map_err(|e| e.to_string())?;
        print_hierarchical_epoch(&report);
        let validation = trainer
            .evaluate_blueprint(dataset, partitions, &mut replay_provider(dataset))
            .map_err(|e| e.to_string())?;
        println!("{}", blueprint_report_value(&validation));
        let validation = trainer
            .evaluate_personalization(
                dataset,
                partitions,
                episode.support_hands,
                &mut replay_provider(dataset),
            )
            .map_err(|e| e.to_string())?;
        println!(
            "{}",
            personalization_report_value(&validation, episode.support_hands)
        );
    }
    for _ in 0..momentary_epochs {
        let report = trainer
            .train_momentary_epoch(
                dataset,
                subjects,
                partitions,
                episode,
                momentary_history,
                &mut replay_provider(dataset),
            )
            .map_err(|e| e.to_string())?;
        print_hierarchical_epoch(&report);
    }
    Ok(())
}

fn load_verified_partitions(
    dataset: &BehaviourParquetDataset,
    path: &str,
) -> Result<lechiffre::datasets::behaviour_parquet_loader::DatasetPartitions, String> {
    dataset
        .legality_fingerprint()
        .map_err(|error| error.to_string())?;
    lechiffre::datasets::behaviour_parquet_loader::DatasetPartitions::read_manifest(dataset, path)
        .map_err(|e| e.to_string())
}

fn evaluation_partition(
    arg: &str,
) -> Result<lechiffre::datasets::behaviour_parquet_loader::EvaluationPartition, String> {
    use lechiffre::datasets::behaviour_parquet_loader::EvaluationPartition;
    match arg {
        "validation" => Ok(EvaluationPartition::Validation),
        "test" => Ok(EvaluationPartition::Test),
        _ => Err("evaluation partition must be exactly validation or test".into()),
    }
}

fn train_hierarchical(args: &[String]) -> Result<(), String> {
    use lechiffre::behavioural::{
        AnalyzerAdam, BaselineAdaptationConfig, BehaviourBlueprint, BehaviourLatentConfig,
        GaussNewton, NeuralBehaviourAnalyzer, NeuralBehaviourGenerator,
    };
    use lechiffre::datasets::behaviour_parquet_loader::{
        BehaviourParquetDataset, HierarchicalTrainer,
    };
    if args.len() != 6 {
        return Err("usage: --train-behavioural <hierarchical-config.json> <parquet-dir> <manifest.json> <checkpoint-out.json>".into());
    }
    if std::path::Path::new(&args[5]).exists() {
        return Err(format!("checkpoint already exists: {}", args[5]));
    }
    let run = run_configuration(&args[2])?;
    let dataset = BehaviourParquetDataset::open(&args[3]).map_err(|e| e.to_string())?;
    let partitions = load_verified_partitions(&dataset, &args[4])?;
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
    let go = GaussNewton::new(&g, run.generator_gn_damping).map_err(|e| e.to_string())?;
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
        run.optimization,
        noise_seed,
    )
    .map_err(|e| e.to_string())?;
    execute_phases(
        &mut trainer,
        &dataset,
        &partitions,
        run.episode,
        &run.subjects,
        run.blueprint_epochs,
        run.personalization_epochs,
        run.momentary_history_hands,
        run.momentary_epochs,
    )?;
    if trainer.completed_epochs().2 > 0 {
        let report = trainer
            .evaluate(
                &dataset,
                &run.subjects,
                &partitions,
                lechiffre::datasets::behaviour_parquet_loader::EvaluationPartition::Validation,
                &mut replay_provider(&dataset),
            )
            .map_err(|e| e.to_string())?;
        print_hierarchical_validation(
            lechiffre::datasets::behaviour_parquet_loader::EvaluationPartition::Validation,
            &report,
        );
    }
    trainer
        .save_checkpoint(&args[5])
        .map_err(|e| e.to_string())?;
    println!(
        "checkpoint={} model_id={}",
        args[5],
        trainer.model_id().map_err(|e| e.to_string())?
    );
    Ok(())
}

fn resume_hierarchical(args: &[String]) -> Result<(), String> {
    use lechiffre::datasets::behaviour_parquet_loader::{
        BehaviourParquetDataset, HierarchicalTrainer,
    };
    if args.len() != 7 {
        return Err("usage: --resume-behavioural <checkpoint-in.json> <parquet-dir> <manifest.json> <checkpoint-out.json> <resume-config.json>".into());
    }
    if std::path::Path::new(&args[5]).exists() {
        return Err(format!("checkpoint already exists: {}", args[5]));
    }
    let mut trainer = HierarchicalTrainer::load_checkpoint(&args[2]).map_err(|e| e.to_string())?;
    let dataset = BehaviourParquetDataset::open(&args[3]).map_err(|e| e.to_string())?;
    let partitions = load_verified_partitions(&dataset, &args[4])?;
    let config = resume_configuration(&args[6])?;
    let episode = trainer
        .training_episode()
        .ok_or("checkpoint missing training episode")?;
    let subjects = trainer.training_subjects().to_vec();
    execute_phases(
        &mut trainer,
        &dataset,
        &partitions,
        episode,
        &subjects,
        config.blueprint_extra_epochs,
        config.personalization_extra_epochs,
        config.momentary_history_hands,
        config.momentary_extra_epochs,
    )?;
    if trainer.completed_epochs().2 > 0 {
        let report = trainer
            .evaluate(
                &dataset,
                &subjects,
                &partitions,
                lechiffre::datasets::behaviour_parquet_loader::EvaluationPartition::Validation,
                &mut replay_provider(&dataset),
            )
            .map_err(|e| e.to_string())?;
        print_hierarchical_validation(
            lechiffre::datasets::behaviour_parquet_loader::EvaluationPartition::Validation,
            &report,
        );
    }
    trainer
        .save_checkpoint(&args[5])
        .map_err(|e| e.to_string())?;
    println!(
        "checkpoint={} model_id={}",
        args[5],
        trainer.model_id().map_err(|e| e.to_string())?
    );
    Ok(())
}

fn evaluate_hierarchical(args: &[String]) -> Result<(), String> {
    use lechiffre::datasets::behaviour_parquet_loader::{
        BehaviourParquetDataset, HierarchicalTrainer,
    };
    if args.len() != 7 {
        return Err("usage: --evaluate-behavioural <checkpoint.json> <parquet-dir> <manifest.json> <validation|test> <new-report.json>".into());
    }
    let trainer = HierarchicalTrainer::load_checkpoint(&args[2]).map_err(|e| e.to_string())?;
    let dataset = BehaviourParquetDataset::open(&args[3]).map_err(|e| e.to_string())?;
    let partitions = load_verified_partitions(&dataset, &args[4])?;
    let partition = evaluation_partition(&args[5])?;
    let report = trainer
        .evaluate(
            &dataset,
            trainer.training_subjects(),
            &partitions,
            partition,
            &mut replay_provider(&dataset),
        )
        .map_err(|e| e.to_string())?;
    let mut result = validation_report_value(partition, &report);
    result["model_id"] = serde_json::json!(trainer.model_id().map_err(|e| e.to_string())?);
    result["dataset_fingerprint"] = serde_json::json!(dataset.dataset_fingerprint());
    result["manifest_fingerprint"] = serde_json::json!(partitions.manifest_fingerprint());
    result["legality_fingerprint"] =
        serde_json::json!(dataset.legality_fingerprint().map_err(|e| e.to_string())?);
    commit_new_json(std::path::Path::new(&args[6]), &result)?;
    println!("{result}");
    Ok(())
}

fn evaluate_blueprint_cli(args: &[String]) -> Result<(), String> {
    use lechiffre::datasets::behaviour_parquet_loader::HierarchicalTrainer;
    if args.len() != 6 {
        return Err("usage: --evaluate-blueprint <checkpoint.json> <parquet-dir> <manifest.json> <new-report.json>".into());
    }
    let trainer = HierarchicalTrainer::load_checkpoint(&args[2]).map_err(|e| e.to_string())?;
    let dataset = BehaviourParquetDataset::open(&args[3]).map_err(|e| e.to_string())?;
    let partitions = load_verified_partitions(&dataset, &args[4])?;
    let report = trainer
        .evaluate_blueprint(&dataset, &partitions, &mut replay_provider(&dataset))
        .map_err(|e| e.to_string())?;
    let mut result = blueprint_report_value(&report);
    result["model_id"] = serde_json::json!(trainer.model_id().map_err(|e| e.to_string())?);
    result["dataset_fingerprint"] = serde_json::json!(dataset.dataset_fingerprint());
    result["manifest_fingerprint"] = serde_json::json!(partitions.manifest_fingerprint());
    result["legality_fingerprint"] =
        serde_json::json!(dataset.legality_fingerprint().map_err(|e| e.to_string())?);
    commit_new_json(std::path::Path::new(&args[5]), &result)?;
    println!("{result}");
    Ok(())
}

fn benchmark_history_budgets(arg: &str) -> Result<Vec<usize>, String> {
    let histories = arg
        .split(',')
        .map(|n| n.parse::<usize>())
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| "history budgets must be unsigned integers")?;
    if histories.len() < 2
        || histories.first() != Some(&0)
        || histories.windows(2).any(|w| w[0] >= w[1])
    {
        return Err("history budgets must start at zero and increase strictly".into());
    }
    Ok(histories)
}

/// Isolate the effect of Z1 adaptation at fixed shared parameters.
fn benchmark_personalization(args: &[String]) -> Result<(), String> {
    use lechiffre::datasets::behaviour_parquet_loader::HierarchicalTrainer;
    if args.len() != 7 {
        return Err("usage: --benchmark-personalization <checkpoint.json> <parquet-dir> <manifest.json> <history1,history2,...> <new-report.json>".into());
    }
    if std::path::Path::new(&args[6]).exists() {
        return Err(format!("benchmark report already exists: {}", args[6]));
    }
    let histories = benchmark_history_budgets(&args[5])?;
    let trainer = HierarchicalTrainer::load_checkpoint(&args[2]).map_err(|e| e.to_string())?;
    let dataset = BehaviourParquetDataset::open(&args[3]).map_err(|e| e.to_string())?;
    let partitions = load_verified_partitions(&dataset, &args[4])?;
    let reference = trainer
        .evaluate_blueprint(&dataset, &partitions, &mut replay_provider(&dataset))
        .map_err(|e| e.to_string())?;
    let reference_value = blueprint_report_value(&reference);
    println!("{reference_value}");
    let mut rows = Vec::new();
    for history in histories {
        let report = trainer
            .evaluate_personalization(
                &dataset,
                &partitions,
                history,
                &mut replay_provider(&dataset),
            )
            .map_err(|e| format!("history={history}: {e}"))?;
        if report.decisions != reference.decisions
            || report.wager_decisions != reference.wager_decisions
        {
            return Err(format!("history={history}: evaluation coverage changed"));
        }
        if history == 0 && report != reference {
            return Err("zero-history Z1 must exactly reproduce Z0 scores".into());
        }
        let row = personalization_report_value(&report, history);
        println!("{row}");
        rows.push(row);
    }
    let (blueprint, personalization, momentary) = trainer.completed_epochs();
    let result = serde_json::json!({
        "format":"lechiffre.behavioural.personalization-benchmark.v1",
        "partition":"validation",
        "model_id":trainer.model_id().map_err(|e| e.to_string())?,
        "dataset_fingerprint":dataset.dataset_fingerprint(),
        "manifest_fingerprint":partitions.manifest_fingerprint(),
        "legality_fingerprint":dataset.legality_fingerprint().map_err(|e| e.to_string())?,
        "subject_ids":trainer.training_subjects(),
        "epochs":{"blueprint":blueprint,"personalization":personalization,"momentary":momentary},
        "ordering":"ingestion_not_verified_chronological",
        "history":"latest_exploitable_train_hands_per_subject",
        "reference":reference_value,
        "reports":rows,
    });
    commit_new_json(std::path::Path::new(&args[6]), &result)?;
    Ok(())
}

fn benchmark_hierarchical(args: &[String]) -> Result<(), String> {
    use lechiffre::datasets::behaviour_parquet_loader::{EvaluationPartition, HierarchicalTrainer};
    if args.len() != 7 {
        return Err("usage: --benchmark-behavioural <checkpoint.json> <parquet-dir> <manifest.json> <history1,history2,...> <new-report.json>".into());
    }
    if std::path::Path::new(&args[6]).exists() {
        return Err(format!("benchmark report already exists: {}", args[6]));
    }
    let histories = benchmark_history_budgets(&args[5])?;
    let trainer = HierarchicalTrainer::load_checkpoint(&args[2]).map_err(|e| e.to_string())?;
    let dataset = BehaviourParquetDataset::open(&args[3]).map_err(|e| e.to_string())?;
    let partitions = load_verified_partitions(&dataset, &args[4])?;
    let mut rows = Vec::new();
    for history in histories {
        let report = trainer
            .evaluate_with_history(
                &dataset,
                trainer.training_subjects(),
                &partitions,
                EvaluationPartition::Validation,
                history,
                &mut replay_provider(&dataset),
            )
            .map_err(|e| format!("history={history}: {e}"))?;
        let row = validation_report_value(EvaluationPartition::Validation, &report);
        println!("{row}");
        rows.push(row);
    }
    let result = serde_json::json!({
        "format":"lechiffre.behavioural.history-benchmark.v1",
        "partition":"validation",
        "model_id":trainer.model_id().map_err(|e| e.to_string())?,
        "dataset_fingerprint":dataset.dataset_fingerprint(),
        "manifest_fingerprint":partitions.manifest_fingerprint(),
        "legality_fingerprint":dataset.legality_fingerprint().map_err(|e| e.to_string())?,
        "subject_ids":trainer.training_subjects(),
        "ordering":"ingestion_not_verified_chronological",
        "history":"latest_exploitable_train_hands_per_subject",
        "reports":rows,
    });
    commit_new_json(std::path::Path::new(&args[6]), &result)?;
    Ok(())
}

/// Select exactly one immutable checkpoint by prospective Z2 prior NLL on
/// Validation. Test is never evaluated, even when ranking candidates.
fn select_best_hierarchical(args: &[String]) -> Result<(), String> {
    use lechiffre::datasets::behaviour_parquet_loader::{EvaluationPartition, HierarchicalTrainer};
    if args.len() != 6 {
        return Err("usage: --select-best-behavioural <parquet-dir> <manifest.json> <selected-checkpoint-out.json> <checkpoint1,checkpoint2,...>".into());
    }
    let dataset = BehaviourParquetDataset::open(&args[2]).map_err(|e| e.to_string())?;
    let partitions = load_verified_partitions(&dataset, &args[3])?;
    let output = std::path::Path::new(&args[4]);
    if output.exists() {
        return Err(format!(
            "selected checkpoint already exists: {}",
            output.display()
        ));
    }
    let candidates: Vec<_> = args[5].split(',').collect();
    if candidates.len() < 2 || candidates.iter().any(|candidate| candidate.is_empty()) {
        return Err("selection requires at least two distinct checkpoint paths".into());
    }
    let mut unique = std::collections::HashSet::new();
    if candidates
        .iter()
        .any(|candidate| !unique.insert(*candidate))
    {
        return Err("candidate checkpoint paths must be unique".into());
    }
    let mut best: Option<(f64, &str)> = None;
    let mut identity: Option<(
        Vec<u64>,
        lechiffre::datasets::behaviour_parquet_loader::TrainingEpisodeConfig,
    )> = None;
    for &path in &candidates {
        let trainer = HierarchicalTrainer::load_checkpoint(path).map_err(|e| e.to_string())?;
        let episode = trainer
            .training_episode()
            .ok_or("checkpoint missing training episode")?;
        let current = (trainer.training_subjects().to_vec(), episode);
        if let Some(expected) = &identity {
            if expected != &current {
                return Err("checkpoint subject selection or episode differs".into());
            }
        } else {
            identity = Some(current);
        }
        let report = trainer
            .evaluate(
                &dataset,
                trainer.training_subjects(),
                &partitions,
                EvaluationPartition::Validation,
                &mut replay_provider(&dataset),
            )
            .map_err(|e| format!("candidate {path}: {e}"))?;
        let score = report.z2_prior_marginal_nll;
        if !score.is_finite() {
            return Err(format!(
                "candidate {path} has a non-finite validation score"
            ));
        }
        println!("candidate={path} validation_z2_prior_marginal_nll={score:.8}");
        if best.as_ref().is_none_or(|(minimum, _)| score < *minimum) {
            best = Some((score, path));
        }
    }
    let (score, path) = best.ok_or("no candidate checkpoint")?;
    // Hard-linking enforces create-new semantics. Any FS error is fatal.
    std::fs::hard_link(path, output).map_err(|e| e.to_string())?;
    println!(
        "selected_checkpoint={} source={} validation_z2_prior_marginal_nll={score:.8}",
        output.display(),
        path
    );
    Ok(())
}

fn player_baseline_cli(args: &[String], advance: bool) -> Result<(), String> {
    use lechiffre::behavioural::{PlayerBaselineAdapter, PlayerBaselineState};
    use lechiffre::datasets::behaviour_parquet_loader::{
        BehaviourParquetDataset, HierarchicalTrainer,
    };
    let expected_arguments = if advance { 8 } else { 7 };
    if args.len() != expected_arguments {
        return Err(if advance {
            "usage: --advance-player-baseline <model-checkpoint.json> <parquet-dir> <subject-id> <before-hand-index> <baseline-in.json> <baseline-out.json>"
        } else {
            "usage: --create-player-baseline <model-checkpoint.json> <parquet-dir> <subject-id> <before-hand-index> <baseline-out.json>"
        }.into());
    }
    let trainer = HierarchicalTrainer::load_checkpoint(&args[2]).map_err(|e| e.to_string())?;
    if trainer.completed_epochs().2 == 0 {
        return Err("model requires completed momentary training".into());
    }
    let dataset = BehaviourParquetDataset::open(&args[3]).map_err(|e| e.to_string())?;
    trainer
        .verify_dataset_legality(&dataset)
        .map_err(|e| e.to_string())?;
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
    dataset
        .legality_fingerprint()
        .map_err(|error| error.to_string())?;
    dataset
        .advance_baseline_before(
            subject,
            cutoff,
            trainer.generator(),
            &adapter,
            &mut state,
            &mut replay_provider(&dataset),
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
    if args.len() > 1 && args[1] == "--attach-behaviour-legality" {
        if let Err(error) = attach_behaviour_legality(&args) {
            eprintln!("behavioural legality attachment failed: {error}");
            std::process::exit(1);
        }
        return;
    }
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
            "--evaluate-behavioural" => Some(evaluate_hierarchical(&args)),
            "--evaluate-blueprint" => Some(evaluate_blueprint_cli(&args)),
            "--benchmark-personalization" => Some(benchmark_personalization(&args)),
            "--benchmark-behavioural" => Some(benchmark_hierarchical(&args)),
            "--select-best-behavioural" => Some(select_best_hierarchical(&args)),
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
        let path =
            std::env::temp_dir().join(format!("lechiffre-hand-split-{}.json", std::process::id()));
        let original = serde_json::json!({"format": "first"});
        let replacement = serde_json::json!({"format": "second"});
        commit_new_json(&path, &original).unwrap();
        assert!(commit_new_json(&path, &replacement).is_err());
        let actual: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(actual, original);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn train_resume_evaluate_and_select_reject_source_arguments() {
        let argv = |values: &[&str]| values.iter().map(|s| (*s).to_owned()).collect::<Vec<_>>();
        assert!(
            train_hierarchical(&argv(&[
                "lechiffre",
                "--train-behavioural",
                "config",
                "parquet",
                "manifest",
                "model",
                "source.csv",
                "session"
            ]))
            .is_err()
        );
        assert!(
            resume_hierarchical(&argv(&[
                "lechiffre",
                "--resume-behavioural",
                "old",
                "parquet",
                "manifest",
                "new",
                "config",
                "source.csv",
                "session"
            ]))
            .is_err()
        );
        assert!(
            evaluate_hierarchical(&argv(&[
                "lechiffre",
                "--evaluate-behavioural",
                "model",
                "parquet",
                "manifest",
                "validation",
                "report",
                "source.csv",
                "session"
            ]))
            .is_err()
        );
        assert!(
            select_best_hierarchical(&argv(&[
                "lechiffre",
                "--select-best-behavioural",
                "parquet",
                "manifest",
                "selected",
                "a,b",
                "source.csv",
                "session"
            ]))
            .is_err()
        );
    }

    #[test]
    fn benchmark_requires_explicit_increasing_history_budgets() {
        assert_eq!(
            benchmark_history_budgets("0,1,10,50,100").unwrap(),
            vec![0, 1, 10, 50, 100]
        );
        for invalid in ["", "0", "1,10", "0,1,1", "0,10,1", "0,-1", "0,1,"] {
            assert!(benchmark_history_budgets(invalid).is_err());
        }
        let argv = [
            "lechiffre",
            "--benchmark-personalization",
            "model",
            "parquet",
            "manifest",
            "0,1",
            "report",
            "source.csv",
        ]
        .map(str::to_owned);
        assert!(benchmark_personalization(&argv).is_err());
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
        let mut blueprint_only = original.clone();
        blueprint_only["personalization_epochs"] = serde_json::json!(0);
        blueprint_only["momentary_epochs"] = serde_json::json!(0);
        std::fs::write(&p, serde_json::to_vec(&blueprint_only).unwrap()).unwrap();
        assert!(run_configuration(p.to_str().unwrap()).is_ok());
        blueprint_only["momentary_epochs"] = serde_json::json!(1);
        std::fs::write(&p, serde_json::to_vec(&blueprint_only).unwrap()).unwrap();
        assert!(run_configuration(p.to_str().unwrap()).is_err());
        let mut obsolete_history = original.clone();
        obsolete_history
            .as_object_mut()
            .unwrap()
            .remove("personalization_epochs");
        obsolete_history["personalization_history_hands"] = serde_json::json!([0, 1, 10]);
        std::fs::write(&p, serde_json::to_vec(&obsolete_history).unwrap()).unwrap();
        assert!(run_configuration(p.to_str().unwrap()).is_err());
        let mut obsolete_weight = original.clone();
        obsolete_weight["generator_z0_weight"] = serde_json::json!(1.0);
        std::fs::write(&p, serde_json::to_vec(&obsolete_weight).unwrap()).unwrap();
        assert!(run_configuration(p.to_str().unwrap()).is_err());
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
