use lechiffre_noregrets::rs_agent::{AgentReference, NoRegretsAgent};
use rs_poker::arena::{Agent, GameStateBuilder, HoldemSimulationBuilder, agent::RandomAgent};
use std::{path::Path, sync::Arc};

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let reference = Arc::new(AgentReference::load(Path::new(
        "references/noregrets-hu10bb-agent-10000-seed42",
    ))?);
    let agents: Vec<Box<dyn Agent>> = vec![
        Box::new(NoRegretsAgent::new(reference.clone(), 42)),
        Box::new(RandomAgent::new_with_seed(
            "Random existant",
            vec![0.25, 0.30, 0.50],
            vec![0.50, 0.60, 0.45],
            43,
        )),
    ];
    let stack = reference.metadata.stack as i64;
    let gs = GameStateBuilder::new()
        .stacks([stack; 2])
        .blinds(2, 1)
        .dealer_idx(0)
        .max_raises_per_round(None)
        .build()?;
    let mut sim = HoldemSimulationBuilder::default()
        .game_state(gs)
        .agents(agents)
        .build_with_rng(rs_poker::arena::seeded_rng(Some(123)))?;
    sim.run().await;
    println!("Tapis finaux : {:?}", sim.game_state.stacks);
    Ok(())
}
