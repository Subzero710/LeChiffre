//! Real rs_poker agents and real HoldemSimulation, no replacement policies.
use crate::rng::Rng;
use crate::rs_agent::{AgentReference, AgentStats, NoRegretsAgent};
use rs_poker::arena::{
    Agent, GameStateBuilder, HoldemSimulationBuilder,
    agent::{AgentConfig, ConfigAgentBuilder},
    cfr::BudgetConfig,
};
use serde::Serialize;
use std::sync::Arc;

pub fn opponent_config(name: &str) -> Result<AgentConfig, String> {
    let value=match name {
        "random"=>serde_json::json!({"type":"random"}),
        "random_pot_control"=>serde_json::json!({"type":"random_pot_control","percent_call":[0.5,0.3]}),
        "calling"|"all_in"|"folding"=>serde_json::json!({"type":name}),
        "cfr_basic"|"cfr_simple"=>serde_json::json!({"type":name,"hand_estimator":"uniform"}),
        _=>return Err("unknown rs_poker opponent: use random, random_pot_control, calling, all_in, folding, cfr_basic, cfr_simple, all or --opponent-config".into()),
    };
    let mut cfg: AgentConfig = serde_json::from_value(value).map_err(|e| e.to_string())?;
    configure_cfr(&mut cfg)?;
    Ok(cfg)
}

pub fn configure_cfr(cfg: &mut AgentConfig) -> Result<(), String> {
    if cfg.is_cfr() {
        // The upstream default 'known' sees real hidden cards. Use its existing
        // Uniform estimator for this imperfect-information benchmark.
        let mut value = serde_json::to_value(&*cfg).map_err(|e| e.to_string())?;
        value["hand_estimator"] = serde_json::json!("uniform");
        *cfg = serde_json::from_value(value).map_err(|e| e.to_string())?;
        let budget: BudgetConfig = serde_json::from_value(serde_json::json!([
            {"type":"deadline","millis":50},
            {"type":"max_width","recursive_widths":[]}
        ]))
        .map_err(|e| e.to_string())?;
        cfg.fill_default_budget(&budget);
    }
    Ok(())
}

#[derive(Serialize, Debug)]
pub struct ArenaReport {
    pub opponent: String,
    pub opponent_config: AgentConfig,
    pub hands: u64,
    pub seed: u64,
    pub total_bb: f64,
    pub bb_per_100: f64,
    pub ci95: [f64; 2],
    pub noregrets: AgentStats,
}

pub async fn duel(
    reference: Arc<AgentReference>,
    name: &str,
    config: AgentConfig,
    hands: u64,
    seed: u64,
) -> Result<ArenaReport, String> {
    if reference.metadata.num_players != 2 {
        return Err("match-agents is heads-up; use table with a multi-player reference".into());
    }
    if hands < 4 || !hands.is_multiple_of(2) {
        return Err("hands must be even and >= 4".into());
    }
    let mut master = Rng::new(seed);
    let (mut mean, mut m2, mut total) = (0.0, 0.0, 0.0);
    let mut aggregate = AgentStats::default();
    for pair in 0..hands / 2 {
        let deal_seed = master.next_u64();
        let agent_seed = master.next_u64();
        let mut gain = 0.0;
        for hero in 0..2 {
            let m = &reference.metadata;
            let gs = GameStateBuilder::new()
                .stacks([m.stack as i64; 2])
                .blinds(m.big_blind as i64, m.small_blind as i64)
                .dealer_idx(0)
                .max_raises_per_round(None)
                .build()
                .map_err(|e| e.to_string())?;
            let agent = NoRegretsAgent::new(reference.clone(), agent_seed);
            let stats = agent.stats();
            let context = AgentConfig::maybe_shared_cfr_context([&config], &gs, 2);
            let mut builder = ConfigAgentBuilder::new(config.clone())
                .map_err(|e| e.to_string())?
                .player_idx(1 - hero)
                .game_state(gs.clone())
                .rng_seed(agent_seed);
            if let Some((ref cfr, ref traversal)) = context {
                builder = builder.cfr_context(cfr.clone(), traversal.clone());
            }
            let other = builder.build();
            let agents: Vec<Box<dyn Agent>> = if hero == 0 {
                vec![Box::new(agent), other]
            } else {
                vec![other, Box::new(agent)]
            };
            let mut sim_builder = HoldemSimulationBuilder::default()
                .game_state(gs)
                .agents(agents);
            if let Some((cfr, traversal)) = context {
                sim_builder = sim_builder.cfr_context(cfr, traversal, true);
            }
            let mut sim = sim_builder
                .build_with_rng(rs_poker::arena::seeded_rng(Some(deal_seed)))
                .map_err(|e| e.to_string())?;
            sim.run().await;
            let stats = stats.lock().map_err(|_| "agent stats poisoned")?;
            if !stats.errors.is_empty() {
                return Err(format!(
                    "NoRegrets adapter failed: {}",
                    stats.errors.join("; ")
                ));
            }
            aggregate.decisions += stats.decisions;
            aggregate.unseen += stats.unseen;
            aggregate.decisions_after_off_tree_bet += stats.decisions_after_off_tree_bet;
            let bb = sim.game_state.player_reward(hero) as f64 / m.big_blind as f64;
            total += bb;
            gain += bb / 2.0;
        }
        let delta = gain - mean;
        mean += delta / (pair + 1) as f64;
        m2 += delta * (gain - mean);
    }
    let pairs = hands / 2;
    let width =
        1.959963984540054 * (m2.max(0.0) / (pairs - 1) as f64 / pairs as f64).sqrt() * 100.0;
    Ok(ArenaReport {
        opponent: name.into(),
        opponent_config: config,
        hands,
        seed,
        total_bb: total,
        bb_per_100: mean * 100.0,
        ci95: [mean * 100.0 - width, mean * 100.0 + width],
        noregrets: aggregate,
    })
}
