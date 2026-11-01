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
        }
    }

    if source_accepted_hands == 0 {
        eprintln!("refusing to commit an empty behavioural dataset");
        std::process::exit(1);
    }

    let summary = writer.close().unwrap_or_else(|error| {
        eprintln!("failed to commit behavioural dataset at {output_dir}: {error}");
        std::process::exit(1);
    });
    println!(
        "source_accepted_hands={} source_rejected_hands={} timing_dropped_hands={} \
         retained_hands={} unique_events={} subject_hand_rows={}",
        source_accepted_hands,
        source_rejected_hands,
        timing_dropped_hands,
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

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
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
        if !imported.rejected.is_empty() {
            std::process::exit(1);
        }
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
