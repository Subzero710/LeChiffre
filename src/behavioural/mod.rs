//! Behavioural Engine data contracts.

mod event;
mod profile;
mod sequence;

pub use event::{
    ActionKind, BehaviourEvent, BehaviourEventError, DecisionContext, ObservedAction, PublicCard,
    Rank, Street, Suit,
};
pub use profile::{BehaviourProfile, BehaviourProfileError};
pub use sequence::{BehaviourSequence, BehaviourSequenceError};

/// Infers the behavioural profile of `subject_seat` from the public action sequence.
pub trait BehaviourAnalyzer {
    type Error;

    fn analyze(
        &self,
        sequence: &BehaviourSequence,
        subject_seat: usize,
    ) -> Result<BehaviourProfile, Self::Error>;
}

/// Generates behaviour for a supplied profile in the given public context.
pub trait BehaviourGenerator {
    type Error;

    fn generate(
        &self,
        profile: &BehaviourProfile,
        context: &DecisionContext,
    ) -> Result<ObservedAction, Self::Error>;
}
