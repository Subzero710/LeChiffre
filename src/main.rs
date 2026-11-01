use lechiffre::adapters;
use lechiffre::adapters::rs_poker::RsPokerAdapter;
use lechiffre::datasets;
use rs_poker::arena::{Agent, GameStateBuilder, HoldemSimulationBuilder, agent::RandomAgent};

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
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
                "poker-with-the-boys",
                &source,
                normalizer,
            )
        } else if args[2].ends_with(".csv") {
            adapters::pokernow::PokerNowAdapter::adapt_csv(
                &args[3],
                "poker-with-the-boys",
                &source,
                normalizer,
            )
        } else {
            eprintln!("unsupported extension: expected .csv or .json");
            std::process::exit(2);
        }
        .expect("invalid input session format");
        let id = adapters::PlayerId::from_source("poker-with-the-boys", &args[4]);
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
