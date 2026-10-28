use crate::arena::Chips;
use crate::arena::agent::AgentConfigError;
use crate::arena::errors::HoldemSimulationError;
use crate::arena::game_state::GameStateBuilderError;
use crate::arena::historian::HistorianError;
use thiserror::Error;

#[derive(Debug, Error, PartialEq)]
pub enum RotationConfigError {
    #[error("players_per_table must be at least 2, got {0}")]
    PlayersPerTableTooSmall(usize),

    #[error("players_per_table ({players}) cannot exceed number of agents ({num_agents})")]
    PlayersPerTableExceedsAgents { players: usize, num_agents: usize },

    #[error("num_rotations must be greater than 0")]
    NumRotationsZero,

    #[error("big_blind must be positive, got {0}")]
    NonPositiveBigBlind(Chips),

    #[error("small_blind must be positive, got {0}")]
    NonPositiveSmallBlind(Chips),

    #[error("small_blind ({small}) must be less than big_blind ({big})")]
    SmallBlindNotLessThanBig { small: Chips, big: Chips },

    #[error("min_stack_bb must be positive, got {0}")]
    NonPositiveMinStack(f32),

    #[error("max_stack_bb must be positive, got {0}")]
    NonPositiveMaxStack(f32),

    #[error("min_stack_bb ({min}) cannot exceed max_stack_bb ({max})")]
    MinStackExceedsMax { min: f32, max: f32 },

    #[error("ante must be non-negative, got {0}")]
    NegativeAnte(Chips),

    #[error("rebuy_to_bb must be positive, got {0}")]
    NonPositiveRebuy(f32),

    #[error("jobs must be greater than 0 when specified")]
    JobsZero,
}

#[derive(Debug, Error)]
pub enum RotationError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Failed to parse agent config from {path}: {source}")]
    ParseConfig {
        path: String,
        #[source]
        source: serde_json::Error,
    },

    #[error("Failed to validate agent config: {0}")]
    InvalidAgentConfig(#[from] AgentConfigError),

    #[error("Invalid rotation configuration: {0}")]
    InvalidConfig(#[from] RotationConfigError),

    #[error("No agent config files found in directory: {0}")]
    NoAgentsFound(String),

    #[error("Simulation error: {0}")]
    Simulation(#[from] HoldemSimulationError),

    #[error("Game-state build error: {0}")]
    GameState(#[from] GameStateBuilderError),

    #[error("Historian error: {0}")]
    Historian(#[from] HistorianError),

    #[error("Failed to read stats from historian storage: {reason}")]
    StatsUnavailable { reason: String },

    #[error("Rotation worker task failed: {0}")]
    WorkerJoin(#[from] tokio::task::JoinError),

    #[error("Failed to serialize JSON: {0}")]
    JsonSerialize(#[from] serde_json::Error),

    #[error("Missing required configuration: {0}")]
    MissingConfig(String),
}

pub type Result<T> = std::result::Result<T, RotationError>;
