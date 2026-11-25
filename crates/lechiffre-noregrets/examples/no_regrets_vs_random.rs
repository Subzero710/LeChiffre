use lechiffre_noregrets::rs_agent::{AgentReference, NoRegretsAgent};
use rs_poker::arena::{Agent, GameStateBuilder, HoldemSimulationBuilder, agent::CallingAgent};
use std::{path::Path, sync::Arc};
#[tokio::main]
async fn main() -> Result<(), String> {
    let model = Arc::new(AgentReference::load(Path::new("runs/noregrets"))?);
    let n = model.metadata.num_players;
    let gs = GameStateBuilder::new()
        .stacks(vec![model.metadata.stack as i64; n])
        .blinds(2, 1)
        .max_raises_per_round(None)
        .build()
        .map_err(|e| e.to_string())?;
    let mut agents: Vec<Box<dyn Agent>> = vec![Box::new(NoRegretsAgent::new(model, 42))];
    agents.extend(
        (1..n).map(|p| Box::new(CallingAgent::new(format!("Calling-{p}"))) as Box<dyn Agent>),
    );
    let mut sim = HoldemSimulationBuilder::default()
        .game_state(gs)
        .agents(agents)
        .build_with_rng(rs_poker::arena::seeded_rng(Some(7)))
        .map_err(|e| e.to_string())?;
    sim.run().await;
    println!(
        "NoRegrets: {} BB",
        sim.game_state.player_reward(0) as f64 / 2.0
    );
    Ok(())
}
