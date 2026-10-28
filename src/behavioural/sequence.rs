use std::fmt;

use super::event::BehaviourEvent;

/// Ordered public behavioural observations for one poker hand.
///
/// The sequence contains every observed player action, not only the actions of
/// the player being profiled. This preserves the stimulus/response relation
/// required by the Analyzer (e.g. "tank 20s while facing a 3-bet").
#[derive(Debug, Clone, PartialEq)]
pub struct BehaviourSequence {
    hand_id: u128,
    events: Vec<BehaviourEvent>,
}

impl BehaviourSequence {
    pub fn new(hand_id: u128) -> Self {
        Self {
            hand_id,
            events: Vec::new(),
        }
    }

    pub fn hand_id(&self) -> u128 {
        self.hand_id
    }

    pub fn events(&self) -> &[BehaviourEvent] {
        &self.events
    }

    pub fn events_for_seat(&self, seat: usize) -> impl Iterator<Item = &BehaviourEvent> {
        self.events
            .iter()
            .filter(move |event| event.actor_seat() == seat)
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
                expected: self.hand_id,
                actual: event.hand_id,
            });
        }

        let expected_index = self.events.len() as u32;
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
    WrongHand { expected: u128, actual: u128 },
    UnexpectedSequenceIndex { expected: u32, actual: u32 },
}

impl fmt::Display for BehaviourSequenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
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
    use crate::behavioural::{
        ActionKind, BehaviourEvent, DecisionContext, ObservedAction, Street,
    };

    fn event(hand_id: u128, sequence_index: u32, actor_seat: usize) -> BehaviourEvent {
        BehaviourEvent::new(
            hand_id,
            sequence_index,
            DecisionContext {
                street: Street::Preflop,
                board: vec![],
                actor_seat,
                dealer_seat: 1,
                players_at_table: 2,
                players_active: 2,
                pot_bb: 2.0,
                actor_stack_bb: 99.0,
                effective_stack_bb: 99.0,
                actor_street_commitment_bb: 1.0,
                current_bet_bb: 1.0,
                to_call_bb: 0.0,
                min_raise_to_bb: Some(2.0),
                bet_or_raise_count_this_street: 0,
            },
            ObservedAction {
                kind: ActionKind::Check,
                amount_added_bb: 0.0,
                street_total_after_bb: 1.0,
                decision_time_ms: Some(320),
            },
        )
        .unwrap()
    }

    #[test]
    fn keeps_actions_from_multiple_players_in_order() {
        let mut seq = BehaviourSequence::new(10);
        seq.push(event(10, 0, 0)).unwrap();
        seq.push(event(10, 1, 1)).unwrap();

        assert_eq!(seq.len(), 2);
        assert_eq!(seq.events_for_seat(0).count(), 1);
        assert_eq!(seq.events_for_seat(1).count(), 1);
    }

    #[test]
    fn rejects_out_of_order_event() {
        let mut seq = BehaviourSequence::new(10);
        assert_eq!(
            seq.push(event(10, 2, 0)),
            Err(BehaviourSequenceError::UnexpectedSequenceIndex {
                expected: 0,
                actual: 2,
            })
        );
    }

    #[test]
    fn rejects_event_from_another_hand() {
        let mut seq = BehaviourSequence::new(10);
        assert_eq!(
            seq.push(event(11, 0, 0)),
            Err(BehaviourSequenceError::WrongHand {
                expected: 10,
                actual: 11,
            })
        );
    }
}
