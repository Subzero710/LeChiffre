use rs_poker::arena::{Agent, GameStateBuilder, HoldemSimulationBuilder, agent::RandomAgent};

#[tokio::main]
async fn main() {
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

    let mut simulation = HoldemSimulationBuilder::default()
        .game_state(game_state)
        .agents(agents)
        .build()
        .expect("failed to build simulation");

    simulation.run().await;

    println!("Random vs Random completed");
    println!("Final stacks: {:?}", simulation.game_state.stacks);
}
