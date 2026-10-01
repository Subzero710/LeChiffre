use crate::arena::{Chips, RakeConfig, money::apply_ratio};
use std::path::PathBuf;

use super::error::{Result, RotationConfigError};

/// Configuration for table-rotation agent benchmarks.
#[derive(Debug, Clone)]
pub struct RotationConfig {
    /// Number of independent table rotations to run.
    pub num_rotations: usize,
    /// Number of players seated at each table.
    pub players_per_table: usize,
    pub big_blind: Chips,
    pub small_blind: Chips,
    pub min_stack_bb: f32,
    pub max_stack_bb: f32,
    pub ante: Chips,
    pub output_dir: Option<PathBuf>,
    pub seed: Option<u64>,
    pub rake: RakeConfig,
    /// Stack to buy back in for after a player reaches exactly zero chips.
    pub rebuy_to_bb: f32,
    /// Maximum number of table rotations executed concurrently. `None` = auto.
    pub jobs: Option<usize>,
}

impl Default for RotationConfig {
    fn default() -> Self {
        Self {
            num_rotations: 1000,
            players_per_table: 3,
            big_blind: 10,
            small_blind: 5,
            min_stack_bb: 100.0,
            max_stack_bb: 100.0,
            ante: 0,
            output_dir: None,
            seed: None,
            rake: RakeConfig::none(),
            rebuy_to_bb: 100.0,
            jobs: None,
        }
    }
}

impl RotationConfig {
    pub fn validate(&self, num_agents: usize) -> Result<()> {
        if self.players_per_table < 2 {
            return Err(
                RotationConfigError::PlayersPerTableTooSmall(self.players_per_table).into(),
            );
        }
        if self.players_per_table > num_agents {
            return Err(RotationConfigError::PlayersPerTableExceedsAgents {
                players: self.players_per_table,
                num_agents,
            }
            .into());
        }
        if self.num_rotations == 0 {
            return Err(RotationConfigError::NumRotationsZero.into());
        }
        if self.big_blind <= 0 {
            return Err(RotationConfigError::NonPositiveBigBlind(self.big_blind).into());
        }
        if self.small_blind <= 0 {
            return Err(RotationConfigError::NonPositiveSmallBlind(self.small_blind).into());
        }
        if self.small_blind >= self.big_blind {
            return Err(RotationConfigError::SmallBlindNotLessThanBig {
                small: self.small_blind,
                big: self.big_blind,
            }
            .into());
        }
        if self.min_stack_bb <= 0.0 {
            return Err(RotationConfigError::NonPositiveMinStack(self.min_stack_bb).into());
        }
        if self.max_stack_bb <= 0.0 {
            return Err(RotationConfigError::NonPositiveMaxStack(self.max_stack_bb).into());
        }
        if self.min_stack_bb > self.max_stack_bb {
            return Err(RotationConfigError::MinStackExceedsMax {
                min: self.min_stack_bb,
                max: self.max_stack_bb,
            }
            .into());
        }
        if self.ante < 0 {
            return Err(RotationConfigError::NegativeAnte(self.ante).into());
        }
        if self.rebuy_to_bb <= 0.0 {
            return Err(RotationConfigError::NonPositiveRebuy(self.rebuy_to_bb).into());
        }
        if self.jobs == Some(0) {
            return Err(RotationConfigError::JobsZero.into());
        }
        Ok(())
    }

    pub fn min_stack(&self) -> Chips {
        apply_ratio(self.big_blind, self.min_stack_bb)
    }

    pub fn max_stack(&self) -> Chips {
        apply_ratio(self.big_blind, self.max_stack_bb)
    }

    pub fn rebuy_stack(&self) -> Chips {
        apply_ratio(self.big_blind, self.rebuy_to_bb)
    }

    /// Number of concurrent table rotations. Auto mode uses roughly 80% of the
    /// parallelism the OS exposes to this process, with a floor of one worker.
    pub fn resolved_jobs(&self) -> usize {
        if let Some(jobs) = self.jobs {
            return jobs.max(1).min(self.num_rotations);
        }
        let available = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let auto = ((available * 4) / 5).max(1);
        auto.min(self.num_rotations)
    }
}
