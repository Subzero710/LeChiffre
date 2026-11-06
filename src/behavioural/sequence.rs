use std::fmt;

use super::event::{BehaviourEvent, HandId, Position};

/// Ordered public behavioural observations for one poker hand.
///
/// The sequence contains every observed player action, not only the actions of
/// the player being profiled. This preserves the stimulus/response relation
/// required by the Analyzer.
#[derive(Debug, Clone, PartialEq)]
pub struct BehaviourSequence {
    hand_id: HandId,
    events: Vec<BehaviourEvent>,
}

impl BehaviourSequence {
    pub fn new(hand_id: impl Into<HandId>) -> Self {
        Self {
            hand_id: hand_id.into(),
            events: Vec::new(),
        }
    }

    pub fn hand_id(&self) -> &HandId {
        &self.hand_id
    }

    pub fn events(&self) -> &[BehaviourEvent] {
        &self.events
    }

    pub fn events_for_position(&self, position: Position) -> impl Iterator<Item = &BehaviourEvent> {
        self.events
            .iter()
            .filter(move |event| event.actor_position() == position)
    }

    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    pub fn len(&self) -> usize {
        self.events.len()
    }

    pub fn push(&mut self, event: BehaviourEvent) -> Result<(), BehaviourSequenceError> {
        if event.hand_id != self.hand_id {
            return Err(BehaviourSequenceError::WrongHand {
                expected: self.hand_id.clone(),
                actual: event.hand_id,
            });
        }

        let expected_index = u32::try_from(self.events.len())
            .map_err(|_| BehaviourSequenceError::SequenceTooLong)?;
        if event.sequence_index != expected_index {
            return Err(BehaviourSequenceError::UnexpectedSequenceIndex {
                expected: expected_index,
                actual: event.sequence_index,
            });
        }

        self.events.push(event);
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BehaviourSequenceError {
    SequenceTooLong,
    WrongHand { expected: HandId, actual: HandId },
    UnexpectedSequenceIndex { expected: u32, actual: u32 },
}

impl fmt::Display for BehaviourSequenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SequenceTooLong => write!(f, "behaviour sequence has more than u32::MAX events"),
            Self::WrongHand { expected, actual } => {
                write!(f, "event belongs to hand {actual}, expected {expected}")
            }
            Self::UnexpectedSequenceIndex { expected, actual } => {
                write!(f, "event index is {actual}, expected {expected}")
            }
        }
    }
}

impl std::error::Error for BehaviourSequenceError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::behavioural::{ActionKind, BehaviourEvent, DecisionContext, ObservedAction, Street};

    fn event(hand_id: &str, sequence_index: u32, position: Position) -> BehaviourEvent {
        let other = if position == Position::Button {
            Position::BigBlind
        } else {
            Position::Button
        };
        let context = DecisionContext {
            street: Street::Preflop,
            actor_position: position,
            pot_bb: 2.0,
            players: vec![
                crate::behavioural::PlayerState {
                    position,
                    status: crate::behavioural::PlayerStatus::Active,
                    stack_remaining_bb: 99.0,
                    committed_this_street_bb: 1.0,
                    committed_this_hand_bb: 1.0,
                },
                crate::behavioural::PlayerState {
                    position: other,
                    status: crate::behavioural::PlayerStatus::Active,
                    stack_remaining_bb: 99.0,
                    committed_this_street_bb: 1.0,
                    committed_this_hand_bb: 1.0,
                },
            ],
            current_bet_bb: 1.0,
            to_call_bb: 0.0,
            min_raise_to_bb: Some(2.0),
            players_yet_to_act: 1,
            bets_and_raises_this_street: 0,
            calls_this_street: 0,
            folds_this_street: 0,
            actor_has_acted_this_street: false,
            actor_aggression_streak_before: 0,
            table_aggression_streak_before: 0,
            last_aggressor: None,
            preflop_last_aggressor_position: None,
        };
        BehaviourEvent::new(
            hand_id,
            sequence_index,
            context,
            ObservedAction {
                kind: ActionKind::Check,
                is_all_in: false,
                sizing: Some(crate::behavioural::ActionSizing {
                    amount_over_pot_before: 0.0,
                    amount_over_stack_before: 0.0,
                    raise_increment_over_pot_before: None,
                }),
                actor_aggression_streak_after: 0,
                table_aggression_streak_after: 0,
                decision_time_ms: Some(320),
            },
        )
        .unwrap()
    }

    #[test]
    fn keeps_actions_from_multiple_positions_in_order() {
        let mut seq = BehaviourSequence::new("hand-10");
        seq.push(event("hand-10", 0, Position::Cutoff)).unwrap();
        seq.push(event("hand-10", 1, Position::Button)).unwrap();

        assert_eq!(seq.len(), 2);
        assert_eq!(seq.events_for_position(Position::Cutoff).count(), 1);
        assert_eq!(seq.events_for_position(Position::Button).count(), 1);
    }

    #[test]
    fn rejects_out_of_order_event() {
        let mut seq = BehaviourSequence::new("hand-10");
        assert_eq!(
            seq.push(event("hand-10", 2, Position::Cutoff)),
            Err(BehaviourSequenceError::UnexpectedSequenceIndex {
                expected: 0,
                actual: 2,
            })
        );
    }

    #[test]
    fn rejects_event_from_another_hand() {
        let mut seq = BehaviourSequence::new("hand-10");
        assert_eq!(
            seq.push(event("hand-11", 0, Position::Cutoff)),
            Err(BehaviourSequenceError::WrongHand {
                expected: HandId::from("hand-10"),
                actual: HandId::from("hand-11"),
            })
        );
    }
}
