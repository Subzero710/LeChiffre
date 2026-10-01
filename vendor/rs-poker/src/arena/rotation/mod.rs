//! Rotation-based cash-game benchmarking.
//!
//! Unlike the legacy `comparison` module, a rotation keeps one circular
//! seating fixed for a complete lap of the dealer button and preserves stacks
//! between hands. Independent rotations can be executed concurrently.

mod builder;
mod config;
mod error;
mod result;
mod runner;
mod scheduler;

pub use builder::RotationBuilder;
pub use config::RotationConfig;
pub use error::{Result, RotationConfigError, RotationError};
pub use result::{RebuyStats, RotationComparisonResult, position_name};
pub use runner::{ArenaRotation, RotationHandResult};
pub use scheduler::{RotationJob, build_schedule, circular_seatings};
