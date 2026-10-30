use std::fmt;

use crate::behavioural::{BehaviourSequence, Position, SubjectHand};

pub mod hand_history;
pub mod rs_poker;

/// Source-observed player identity used by ingestion adapters.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PlayerId(String);

impl PlayerId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub fn from_source(source: &str, identity: &str) -> Self {
        Self(format!("{source}:{identity}"))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for PlayerId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObservedPlayer {
    pub player_id: PlayerId,
    pub position: Position,
}

/// Source-independent result of adapting one real hand.
#[derive(Debug, Clone, PartialEq)]
pub struct AdaptedHand {
    sequence: BehaviourSequence,
    players: Vec<ObservedPlayer>,
}

impl AdaptedHand {
    pub(crate) fn new(sequence: BehaviourSequence, players: Vec<ObservedPlayer>) -> Self {
        Self { sequence, players }
    }

    pub fn sequence(&self) -> &BehaviourSequence {
        &self.sequence
    }

    pub fn players(&self) -> &[ObservedPlayer] {
        &self.players
    }

    pub fn player(&self, player_id: &PlayerId) -> Option<&ObservedPlayer> {
        self.players
            .iter()
            .find(|player| &player.player_id == player_id)
    }

    pub fn subject_hand(&self, player_id: &PlayerId) -> Option<SubjectHand> {
        self.player(player_id)
            .map(|player| SubjectHand::new(self.sequence.clone(), player.position))
    }
}
