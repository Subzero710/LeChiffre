//! Behavioural Engine data contracts.

mod event;
mod history;
mod profile;
mod sequence;

pub use event::{
    ActionKind, BehaviourEvent, BehaviourEventError, DecisionContext, HandId, ObservedAction,
    Position, PublicCard, Rank, Street, Suit,
};
pub use history::{BehaviourHistory, SubjectHand};
pub use profile::{BehaviourProfile, BehaviourProfileError};
pub use sequence::{BehaviourSequence, BehaviourSequenceError};

/// Infers a behavioural profile from multiple real hands of the same subject.
pub trait BehaviourAnalyzer {
    type Error;

    fn analyze(&self, history: &BehaviourHistory) -> Result<BehaviourProfile, Self::Error>;
}

/// Generates behaviour for a supplied profile from the public history and current context.
pub trait BehaviourGenerator {
    type Error;

    fn generate(
        &self,
        profile: &BehaviourProfile,
        sequence: &BehaviourSequence,
        context: &DecisionContext,
    ) -> Result<ObservedAction, Self::Error>;
}
