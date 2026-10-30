use std::fmt;

use rs_poker::arena::action::AgentAction;
use rs_poker::arena::RakeConfig;
use rs_poker::open_hand_history::{replay_hand, HandHistory, ReplayError, ReplayStep};

use crate::behavioural::{
    ActionKind, BehaviourEvent, BehaviourEventError, BehaviourSequence, BehaviourSequenceError,
    HandId, ObservedAction,
};

use super::rs_poker::{decision_context, to_bb};

pub struct HandHistoryAdapter;

impl HandHistoryAdapter {
    pub fn adapt(
        hand: &HandHistory,
        rake: RakeConfig,
    ) -> Result<BehaviourSequence, HandHistoryAdapterError> {
        let replayed = replay_hand(hand, rake)?;
        let hand_id = HandId::new(format!("{}:{}", hand.site_name, hand.game_number));
        let mut sequence = BehaviourSequence::new(hand_id.clone());

        for (sequence_index, step) in replayed.transitions().enumerate() {
            let sequence_index = u32::try_from(sequence_index)
                .map_err(|_| HandHistoryAdapterError::SequenceTooLong)?;
            let event = BehaviourEvent::new(
                hand_id.clone(),
                sequence_index,
                decision_context(&step.state_before),
                observed_action(step),
            )?;
            sequence.push(event)?;
        }

        Ok(sequence)
    }
}

fn observed_action(step: &ReplayStep) -> ObservedAction {
    let actor_seat = step.state_before.to_act_idx();
    let before_player_bet = step.state_before.current_round_player_bet(actor_seat);
    let after_player_bet = step.state_after.current_round_player_bet(actor_seat);
    let amount_added = after_player_bet - before_player_bet;
    let starting_bet = step.state_before.current_round_bet();
    let final_bet = step.state_after.current_round_bet();

    let kind = if matches!(&step.action, AgentAction::Fold) {
        ActionKind::Fold
    } else if final_bet > starting_bet {
        if starting_bet == 0 {
            ActionKind::Bet
        } else {
            ActionKind::Raise
        }
    } else if amount_added > 0 {
        ActionKind::Call
    } else {
        ActionKind::Check
    };

    ObservedAction {
        kind,
        is_all_in: step.state_after.player_all_in.get(actor_seat),
        amount_added_bb: to_bb(amount_added, step.state_before.big_blind),
        street_total_after_bb: to_bb(after_player_bet, step.state_before.big_blind),
        decision_time_ms: None,
    }
}

#[derive(Debug)]
pub enum HandHistoryAdapterError {
    SequenceTooLong,
    Replay(ReplayError),
    BehaviourEvent(BehaviourEventError),
    BehaviourSequence(BehaviourSequenceError),
}

impl fmt::Display for HandHistoryAdapterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SequenceTooLong => write!(f, "hand history has more than u32::MAX decisions"),
            Self::Replay(error) => write!(f, "hand history replay failed: {error}"),
            Self::BehaviourEvent(error) => write!(f, "invalid behaviour event: {error}"),
            Self::BehaviourSequence(error) => write!(f, "invalid behaviour sequence: {error}"),
        }
    }
}

impl std::error::Error for HandHistoryAdapterError {}

impl From<ReplayError> for HandHistoryAdapterError {
    fn from(error: ReplayError) -> Self {
        Self::Replay(error)
    }
}

impl From<BehaviourEventError> for HandHistoryAdapterError {
    fn from(error: BehaviourEventError) -> Self {
        Self::BehaviourEvent(error)
    }
}

impl From<BehaviourSequenceError> for HandHistoryAdapterError {
    fn from(error: BehaviourSequenceError) -> Self {
        Self::BehaviourSequence(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rs_poker::open_hand_history::{HandHistoryParser, PokerStarsParser};

    #[test]
    fn adapts_parsed_room_history_without_inventing_timing() {
        let hand = PokerStarsParser::default()
            .parse(include_str!(
                "../../vendor/rs-poker/tests/fixtures/rooms/pokerstars.txt"
            ))
            .unwrap()
            .remove(0);

        let sequence = HandHistoryAdapter::adapt(&hand, RakeConfig::none()).unwrap();

        assert_eq!(sequence.hand_id().as_str(), "PokerStars:171562910425");
        assert_eq!(sequence.len(), 14);
        assert!(
            sequence
                .events()
                .iter()
                .all(|event| event.action.decision_time_ms.is_none())
        );
    }
}
