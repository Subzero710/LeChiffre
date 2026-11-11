use std::ops::ControlFlow;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use rand::{RngExt, SeedableRng, rngs::StdRng};
use tokio::io::AsyncWriteExt;
use tokio::task::JoinSet;
use tracing::event;

use crate::arena::agent::{AgentConfig, ConfigAgentBuilder};
use crate::arena::comparison::AgentStatsBuilder;
use crate::arena::historian::{OpenHandHistoryVecHistorian, StatsStorage, StatsTrackingHistorian};
use crate::arena::{Agent, Chips, GameStateBuilder, Historian, HoldemSimulationBuilder};
use crate::open_hand_history::write_hand;

use super::config::RotationConfig;
use super::error::{Result, RotationError};
use super::result::{RebuyStats, RotationComparisonResult};
use super::scheduler::{RotationJob, build_schedule, circular_seatings, mix64};

/// Result of one hand inside a table rotation.
#[derive(Debug, Clone)]
pub struct RotationHandResult {
    pub rotation_idx: usize,
    pub hand_idx: usize,
    pub dealer_idx: usize,
    /// Agent index assigned to each physical seat.
    pub seating: Vec<usize>,
    pub agent_names: Vec<String>,
    pub stats: StatsStorage,
    pub duration: Duration,
    pub(crate) ohh_record: Option<Vec<u8>>,
}

#[derive(Debug)]
struct RotationWorkerResult {
    hands: Vec<RotationHandResult>,
    /// (agent_idx, chips injected from outside the table)
    rebuys: Vec<(usize, Chips)>,
    completed: bool,
}

/// Rotation-based cash-game benchmark.
///
/// A job fixes one seating, plays exactly one full lap of the dealer button,
/// preserves stacks between those hands, and resets before the next job.
#[derive(Debug)]
pub struct ArenaRotation {
    config: RotationConfig,
    agents: Vec<(String, AgentConfig)>,
}

impl ArenaRotation {
    pub(crate) fn new(config: RotationConfig, agents: Vec<(String, AgentConfig)>) -> Self {
        Self { config, agents }
    }

    pub fn config(&self) -> &RotationConfig {
        &self.config
    }

    pub fn agents(&self) -> &[(String, AgentConfig)] {
        &self.agents
    }

    pub fn num_agents(&self) -> usize {
        self.agents.len()
    }

    pub fn unique_seatings(&self) -> usize {
        circular_seatings(self.agents.len(), self.config.players_per_table).len()
    }

    pub fn total_rotations(&self) -> usize {
        self.config.num_rotations
    }

    pub fn total_hands(&self) -> usize {
        self.config.num_rotations * self.config.players_per_table
    }

    pub async fn run(&self) -> Result<RotationComparisonResult> {
        self.run_with_callback(|_| {}).await
    }

    pub async fn run_with_callback<F>(&self, mut on_hand: F) -> Result<RotationComparisonResult>
    where
        F: FnMut(RotationHandResult),
    {
        self.run_with_cancellable_callback(|hand| {
            on_hand(hand);
            ControlFlow::Continue(())
        })
        .await
    }

    pub async fn run_with_cancellable_callback<F>(
        &self,
        mut on_hand: F,
    ) -> Result<RotationComparisonResult>
    where
        F: FnMut(RotationHandResult) -> ControlFlow<()>,
    {
        let names: Vec<String> = self.agents.iter().map(|(name, _)| name.clone()).collect();
        let configs = Arc::new(
            self.agents
                .iter()
                .map(|(_, cfg)| cfg.clone())
                .collect::<Vec<AgentConfig>>(),
        );
        let shared_names = Arc::new(names.clone());
        let mut stats_builder = AgentStatsBuilder::new(names.clone());
        let mut rebuy_totals = vec![RebuyStats::default(); self.agents.len()];
        let unique_seatings = self.unique_seatings();
        let effective_seed = self.config.seed.unwrap_or_else(|| {
            let mut rng = rand::rng();
            rng.random::<u64>()
        });
        let schedule = build_schedule(
            self.agents.len(),
            self.config.players_per_table,
            self.config.num_rotations,
            effective_seed,
        );
        let jobs = self.config.resolved_jobs();
        let cancelled = Arc::new(AtomicBool::new(false));
        let mut join_set = JoinSet::new();
        let mut next_job = 0usize;
        let mut rotations_run = 0usize;
        let mut hands_run = 0usize;

        event!(
            tracing::Level::INFO,
            rotations = self.config.num_rotations,
            players_per_table = self.config.players_per_table,
            unique_seatings,
            jobs,
            "Starting rotation comparison"
        );

        while next_job < schedule.len() && join_set.len() < jobs {
            spawn_rotation(
                &mut join_set,
                schedule[next_job].clone(),
                self.config.clone(),
                Arc::clone(&configs),
                Arc::clone(&shared_names),
                Arc::clone(&cancelled),
            );
            next_job += 1;
        }

        let ohh_path = self
            .config
            .output_dir
            .as_ref()
            .map(|d| d.join("hands.jsonl"));
        let mut user_cancelled = false;

        while let Some(joined) = join_set.join_next().await {
            let worker = joined??;

            if user_cancelled {
                continue;
            }

            for &(agent_idx, chips) in &worker.rebuys {
                rebuy_totals[agent_idx].count += 1;
                rebuy_totals[agent_idx].chips += chips;
            }

            if worker.completed {
                rotations_run += 1;
            }

            for mut hand in worker.hands {
                let positions = relative_positions(self.config.players_per_table, hand.dealer_idx);
                stats_builder.merge_hand_stats(&hand.seating, &positions, &hand.stats);
                hands_run += 1;

                if let (Some(path), Some(record)) = (ohh_path.as_ref(), hand.ohh_record.take()) {
                    append_record(path, &record).await?;
                }

                if on_hand(hand).is_break() {
                    cancelled.store(true, Ordering::Release);
                    user_cancelled = true;
                    break;
                }
            }

            if !user_cancelled && next_job < schedule.len() {
                spawn_rotation(
                    &mut join_set,
                    schedule[next_job].clone(),
                    self.config.clone(),
                    Arc::clone(&configs),
                    Arc::clone(&shared_names),
                    Arc::clone(&cancelled),
                );
                next_job += 1;
            }
        }

        let agent_stats = stats_builder.build();
        let rebuy_stats = names.iter().cloned().zip(rebuy_totals).collect();

        Ok(RotationComparisonResult::new(
            names,
            agent_stats,
            rebuy_stats,
            self.config.clone(),
            unique_seatings,
            rotations_run,
            hands_run,
        ))
    }

    pub fn print_configuration_summary(&self) {
        let unique = self.unique_seatings();
        println!("Agent Rotation Comparison");
        println!("=========================");
        println!();
        println!("Number of Agents: {}", self.agents.len());
        println!("Players per Table: {}", self.config.players_per_table);
        println!("Table Rotations: {}", self.config.num_rotations);
        println!("Hands per Rotation: {}", self.config.players_per_table);
        println!("Total Hands: {}", self.total_hands());
        println!("Unique Circular Seatings: {unique}");
        println!(
            "Seating Coverage: {} complete cycles + {}/{}",
            self.config.num_rotations / unique,
            self.config.num_rotations % unique,
            unique
        );
        println!("Workers: {}", self.config.resolved_jobs());
        println!("Rebuy on Bust: {:.1} BB", self.config.rebuy_to_bb);
        println!();
        println!("Loaded Agents:");
        for (name, _) in &self.agents {
            println!("  - {name}");
        }
        println!();
    }
}

fn spawn_rotation(
    set: &mut JoinSet<Result<RotationWorkerResult>>,
    job: RotationJob,
    config: RotationConfig,
    agent_configs: Arc<Vec<AgentConfig>>,
    agent_names: Arc<Vec<String>>,
    cancelled: Arc<AtomicBool>,
) {
    set.spawn(
        async move { run_rotation(job, config, agent_configs, agent_names, cancelled).await },
    );
}

async fn run_rotation(
    job: RotationJob,
    config: RotationConfig,
    agent_configs: Arc<Vec<AgentConfig>>,
    agent_names: Arc<Vec<String>>,
    cancelled: Arc<AtomicBool>,
) -> Result<RotationWorkerResult> {
    let players = config.players_per_table;
    let mut stack_rng = StdRng::seed_from_u64(mix64(job.seed ^ 0x243F_6A88_85A3_08D3));
    let min_stack = config.min_stack();
    let max_stack = config.max_stack();
    let mut stacks: Vec<Chips> = (0..players)
        .map(|_| {
            if min_stack == max_stack {
                min_stack
            } else {
                stack_rng.random_range(min_stack..max_stack)
            }
        })
        .collect();
    let rebuy_stack = config.rebuy_stack();
    let mut hands = Vec::with_capacity(players);
    let mut rebuys = Vec::new();

    for hand_idx in 0..players {
        if cancelled.load(Ordering::Acquire) {
            return Ok(RotationWorkerResult {
                hands,
                rebuys,
                completed: false,
            });
        }

        let dealer_idx = (job.starting_dealer + hand_idx) % players;
        let hand_seed = mix64(job.seed ^ (hand_idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let game_state = GameStateBuilder::new()
            .stacks(&stacks)
            .big_blind(config.big_blind)
            .small_blind(config.small_blind)
            .ante(config.ante)
            .dealer_idx(dealer_idx)
            .rake(config.rake)
            .build()?;

        let cfr_context = AgentConfig::maybe_shared_cfr_context(
            job.seating
                .iter()
                .map(|&agent_idx| &agent_configs[agent_idx]),
            &game_state,
            players,
        );

        let boxed_agents: Vec<Box<dyn Agent>> = job
            .seating
            .iter()
            .enumerate()
            .map(|(seat_idx, &agent_idx)| {
                let mut builder = ConfigAgentBuilder::new(agent_configs[agent_idx].clone())
                    .expect("validated agent config failed to create builder")
                    .player_idx(seat_idx);
                if let Some((ref cfr_state, ref traversal_set)) = cfr_context {
                    builder = builder.cfr_context(cfr_state.clone(), traversal_set.clone());
                }
                builder = builder.game_state(game_state.clone()).rng_seed(mix64(
                    hand_seed
                        ^ (agent_idx as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93)
                        ^ seat_idx as u64,
                ));
                builder.build()
            })
            .collect();

        let stats_historian = StatsTrackingHistorian::new_with_num_players(players);
        let stats_storage = stats_historian.get_storage();
        let mut historians: Vec<Box<dyn Historian>> = vec![Box::new(stats_historian)];

        let ohh_storage = if config.output_dir.is_some() {
            let historian = OpenHandHistoryVecHistorian::new();
            let storage = historian.get_storage();
            historians.push(Box::new(historian));
            Some(storage)
        } else {
            None
        };

        let mut sim_builder = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(boxed_agents)
            .historians(historians);
        if let Some((cfr_state, traversal_set)) = cfr_context {
            sim_builder = sim_builder.cfr_context(cfr_state, traversal_set, true);
        }
        let sim_rng = StdRng::seed_from_u64(mix64(hand_seed ^ 0x1319_8A2E_0370_7344));
        let mut sim = sim_builder.build_with_rng(sim_rng)?;

        let started = Instant::now();
        sim.run().await;
        let duration = started.elapsed();
        stacks = sim.game_state.stacks.to_vec();
        drop(sim);

        let stats = {
            let guard = stats_storage
                .try_read()
                .map_err(|e| RotationError::StatsUnavailable {
                    reason: e.to_string(),
                })?;
            guard.clone()
        };

        let ohh_record = if let Some(storage) = ohh_storage {
            let hand = storage
                .lock()
                .map_err(|_| RotationError::StatsUnavailable {
                    reason: "open-hand-history storage lock poisoned".to_string(),
                })?
                .last()
                .cloned();
            if let Some(hand) = hand {
                let mut record = Vec::new();
                write_hand(&mut record, hand)?;
                Some(record)
            } else {
                None
            }
        } else {
            None
        };

        hands.push(RotationHandResult {
            rotation_idx: job.rotation_idx,
            hand_idx,
            dealer_idx,
            seating: job.seating.clone(),
            agent_names: job
                .seating
                .iter()
                .map(|&agent_idx| agent_names[agent_idx].clone())
                .collect(),
            stats,
            duration,
            ohh_record,
        });

        // A rebuy is external bankroll, never poker profit. Per-hand StatsStorage
        // already closed the busted hand at the pre-rebuy stack, so injecting
        // chips here cannot create a fake +rebuy result.
        if hand_idx + 1 < players {
            for (seat_idx, stack) in stacks.iter_mut().enumerate() {
                if *stack == 0 {
                    *stack = rebuy_stack;
                    rebuys.push((job.seating[seat_idx], rebuy_stack));
                }
            }
        }
    }

    Ok(RotationWorkerResult {
        hands,
        rebuys,
        completed: true,
    })
}

fn relative_positions(players: usize, dealer_idx: usize) -> Vec<usize> {
    (0..players)
        .map(|seat_idx| (seat_idx + players - dealer_idx) % players)
        .collect()
}

async fn append_record(path: &std::path::Path, record: &[u8]) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    let mut file = tokio::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .await?;
    file.write_all(record).await?;
    file.flush().await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positions_follow_the_button() {
        assert_eq!(relative_positions(6, 0), vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(relative_positions(6, 2), vec![4, 5, 0, 1, 2, 3]);
    }
}
