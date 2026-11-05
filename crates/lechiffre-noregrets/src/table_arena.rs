//! Fixed seats, rotating button, actual rs_poker agents. 2..6 players.
use crate::{
    rs_agent::{AgentReference, AgentStats, NoRegretsAgent},
    rs_arena::configure_cfr,
};
use lechiffre_sdcfr::rng::Rng;
use rs_poker::arena::{
    Agent, GameStateBuilder, HoldemSimulationBuilder,
    agent::{AgentConfig, ConfigAgentBuilder},
};
use serde::Serialize;
use serde_json::Value;
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

enum Bot {
    NoRegrets(Arc<AgentReference>),
    Existing(AgentConfig),
}
struct Seat {
    label: String,
    config: Value,
    bot: Bot,
}
pub struct PreparedTable {
    seats: Vec<Seat>,
    stack: u32,
}

#[derive(Debug, Serialize)]
pub struct SeatReport {
    pub seat: usize,
    pub agent: String,
    pub config: Value,
    pub total_bb: f64,
    pub bb_per_100: f64,
    pub ci95: [f64; 2],
    pub noregrets: Option<AgentStats>,
}
#[derive(Debug, Serialize)]
pub struct TableReport {
    pub players: usize,
    pub stack: u32,
    pub big_blind: u32,
    pub hands: u64,
    pub cycles: u64,
    pub seed: u64,
    pub seats: Vec<SeatReport>,
    pub exact_exploitability: Option<f64>,
}

impl PreparedTable {
    /// Configs use existing rs_poker JSON, plus {type:noregrets,reference:path}.
    pub fn new(
        configs: Vec<Value>,
        stack: Option<u32>,
        default_reference: Option<&Path>,
    ) -> Result<Self, String> {
        let n = configs.len();
        if !(2..=6).contains(&n) {
            return Err("choose 2 to 6 players, one configuration per seat".into());
        }
        let mut seats = Vec::new();
        let mut common_stack = stack;
        for value in configs {
            let kind = value
                .get("type")
                .and_then(Value::as_str)
                .ok_or("each player needs a type")?
                .to_owned();
            if kind == "noregrets" {
                let path = value
                    .get("reference")
                    .and_then(Value::as_str)
                    .map(PathBuf::from)
                    .or_else(|| default_reference.map(Path::to_path_buf))
                    .ok_or("NoRegrets seat needs --reference or a reference path in JSON")?;
                let r = Arc::new(AgentReference::load(&path)?);
                if r.metadata.num_players != n {
                    return Err(format!(
                        "reference trained for {} players; table has {n}. Train with --players {n}",
                        r.metadata.num_players
                    ));
                }
                let expected = *common_stack.get_or_insert(r.metadata.stack);
                if expected != r.metadata.stack {
                    return Err("all references must match table stack".into());
                }
                seats.push(Seat {label:"noregrets".into(),config:serde_json::json!({"type":"noregrets","reference":path,"metadata":r.metadata}),bot:Bot::NoRegrets(r)});
            } else {
                let mut value = value;
                // Same convenience default as match-agents; other settings
                // remain the supplied ConfigAgentBuilder configuration.
                if kind == "random_pot_control" && value.get("percent_call").is_none() {
                    value["percent_call"] = serde_json::json!([0.5, 0.3]);
                }
                let mut cfg: AgentConfig =
                    serde_json::from_value(value).map_err(|e| e.to_string())?;
                cfg.validate().map_err(|e| e.to_string())?;
                configure_cfr(&mut cfg)?;
                seats.push(Seat {
                    label: kind,
                    config: serde_json::to_value(&cfg).map_err(|e| e.to_string())?,
                    bot: Bot::Existing(cfg),
                });
            }
        }
        let stack = common_stack.unwrap_or(20);
        if stack <= 2 {
            return Err("stack must exceed big blind (2 chips)".into());
        }
        Ok(Self { seats, stack })
    }

    pub async fn run(&self, hands: u64, seed: u64) -> Result<TableReport, String> {
        let n = self.seats.len();
        let size = n as u64;
        if hands < 2 * size || !hands.is_multiple_of(size) {
            return Err(format!(
                "hands must be a multiple of {n}, at least {} (two complete button rotations)",
                2 * n
            ));
        }
        let cycles = hands / size;
        let mut rng = Rng::new(seed);
        let (mut total, mut mean, mut m2) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        let mut stats: Vec<Option<AgentStats>> = self
            .seats
            .iter()
            .map(|s| matches!(s.bot, Bot::NoRegrets(_)).then(AgentStats::default))
            .collect();
        for cycle in 0..cycles {
            let mut block = vec![0.0; n];
            for dealer in 0..n {
                let gs = GameStateBuilder::new()
                    .stacks(vec![self.stack as i64; n])
                    .blinds(2, 1)
                    .dealer_idx(dealer)
                    .max_raises_per_round(None)
                    .build()
                    .map_err(|e| e.to_string())?;
                let configs = self.seats.iter().filter_map(|s| {
                    if let Bot::Existing(ref c) = s.bot {
                        Some(c)
                    } else {
                        None
                    }
                });
                let context = AgentConfig::maybe_shared_cfr_context(configs, &gs, n);
                let mut agents: Vec<Box<dyn Agent>> = Vec::with_capacity(n);
                let mut counters = Vec::new();
                for (p, seat) in self.seats.iter().enumerate() {
                    let seed = rng.next_u64();
                    match &seat.bot {
                        Bot::NoRegrets(r) => {
                            let a = NoRegretsAgent::new(r.clone(), seed);
                            counters.push((p, a.stats()));
                            agents.push(Box::new(a));
                        }
                        Bot::Existing(config) => {
                            let mut b = ConfigAgentBuilder::new(config.clone())
                                .map_err(|e| e.to_string())?
                                .player_idx(p)
                                .game_state(gs.clone())
                                .rng_seed(seed);
                            if let Some((ref cfr, ref traversal)) = context {
                                b = b.cfr_context(cfr.clone(), traversal.clone());
                            }
                            agents.push(b.build());
                        }
                    }
                }
                let mut builder = HoldemSimulationBuilder::default()
                    .game_state(gs)
                    .agents(agents);
                if let Some((cfr, traversal)) = context {
                    builder = builder.cfr_context(cfr, traversal, true);
                }
                let mut sim = builder
                    .build_with_rng(rs_poker::arena::seeded_rng(Some(rng.next_u64())))
                    .map_err(|e| e.to_string())?;
                sim.run().await;
                let rewards: Vec<i64> = (0..n).map(|p| sim.game_state.player_reward(p)).collect();
                if rewards.iter().sum::<i64>() != 0 {
                    return Err("non-zero-sum result in no-rake table".into());
                }
                for (p, reward) in rewards.into_iter().enumerate() {
                    let bb = reward as f64 / 2.0;
                    total[p] += bb;
                    block[p] += bb / size as f64;
                }
                for (p, counter) in counters {
                    let s = counter.lock().map_err(|_| "agent statistics poisoned")?;
                    if !s.errors.is_empty() {
                        return Err(s.errors.join("; "));
                    }
                    let a = stats[p].as_mut().unwrap();
                    a.decisions += s.decisions;
                    a.unseen += s.unseen;
                    a.decisions_after_off_tree_bet += s.decisions_after_off_tree_bet;
                }
            }
            for p in 0..n {
                let delta = block[p] - mean[p];
                mean[p] += delta / (cycle + 1) as f64;
                m2[p] += delta * (block[p] - mean[p]);
            }
        }
        let seats = self
            .seats
            .iter()
            .enumerate()
            .map(|(p, s)| {
                let width = 1.959963984540054
                    * (m2[p].max(0.0) / (cycles - 1) as f64 / cycles as f64).sqrt()
                    * 100.0;
                SeatReport {
                    seat: p + 1,
                    agent: s.label.clone(),
                    config: s.config.clone(),
                    total_bb: total[p],
                    bb_per_100: mean[p] * 100.0,
                    ci95: [mean[p] * 100.0 - width, mean[p] * 100.0 + width],
                    noregrets: stats[p].take(),
                }
            })
            .collect();
        Ok(TableReport {
            players: n,
            stack: self.stack,
            big_blind: 2,
            hands,
            cycles,
            seed,
            seats,
            exact_exploitability: None,
        })
    }
}
