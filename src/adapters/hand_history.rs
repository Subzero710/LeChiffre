use std::collections::HashSet;
use std::fmt;

use rs_poker::arena::action::AgentAction;
use rs_poker::arena::RakeConfig;
use rs_poker::open_hand_history::{replay_hand, HandHistory, ReplayError, ReplayStep, ReplayedHand};

use crate::behavioural::{
    ActionKind, BehaviourEvent, BehaviourEventError, BehaviourHistory, BehaviourSequence,
    BehaviourSequenceError, HandId, ObservedAction, Position,
};

use super::{AdaptedHand, ObservedPlayer, PlayerId};
use super::rs_poker::{decision_context, position_for_index, to_bb};

pub struct HandHistoryAdapter;

impl HandHistoryAdapter {
    pub fn adapt(
        hand: &HandHistory,
        rake: RakeConfig,
    ) -> Result<AdaptedHand, HandHistoryAdapterError> {
        let replayed = replay_hand(hand, rake)?;
        let players_at_table = replayed.initial_state.num_players;
        if !(2..=6).contains(&players_at_table) {
            return Err(HandHistoryAdapterError::UnsupportedPlayerCount {
                players: players_at_table,
            });
        }

        let hand_id = HandId::new(format!("{}:{}", hand.site_name, hand.game_number));
        let players = observed_players(hand, &replayed)?;
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

        Ok(AdaptedHand::new(sequence, players))
    }

    /// Builds a multi-hand behavioural history by exact source-observed
    /// `PlayerId` equality. No seat-based or heuristic identity matching occurs.
    pub fn history_for(player_id: &PlayerId, hands: &[AdaptedHand]) -> BehaviourHistory {
        let mut history = BehaviourHistory::new();
        for hand in hands {
            if let Some(subject_hand) = hand.subject_hand(player_id) {
                history.push(subject_hand);
            }
        }
        history
    }
}

fn observed_players(
    hand: &HandHistory,
    replayed: &ReplayedHand,
) -> Result<Vec<ObservedPlayer>, HandHistoryAdapterError> {
    let mut identities = HashSet::new();
    let mut players = Vec::with_capacity(replayed.player_ids.len());

    for (index, source_player_id) in replayed.player_ids.iter().copied().enumerate() {
        let source_player = hand
            .players
            .iter()
            .find(|player| player.id == source_player_id)
            .ok_or(HandHistoryAdapterError::MissingSourcePlayer {
                source_player_id,
            })?;

        let player_id = PlayerId::from_source(&hand.site_name, &source_player.name);
        if !identities.insert(player_id.clone()) {
            return Err(HandHistoryAdapterError::DuplicatePlayerIdentity { player_id });
        }

        players.push(ObservedPlayer {
            player_id,
            position: position_for_index(
                replayed.initial_state.num_players,
                replayed.initial_state.dealer_idx,
                index,
            ),
        });
    }

    Ok(players)
}

fn observed_action(step: &ReplayStep) -> ObservedAction {
    let actor_index = step.state_before.to_act_idx();
    let before_player_bet = step.state_before.current_round_player_bet(actor_index);
    let after_player_bet = step.state_after.current_round_player_bet(actor_index);
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
        is_all_in: step.state_after.player_all_in.get(actor_index),
        amount_added_bb: to_bb(amount_added, step.state_before.big_blind),
        street_total_after_bb: to_bb(after_player_bet, step.state_before.big_blind),
        decision_time_ms: None,
    }
}

#[derive(Debug)]
pub enum HandHistoryAdapterError {
    UnsupportedPlayerCount { players: usize },
    SequenceTooLong,
    MissingSourcePlayer { source_player_id: u64 },
    DuplicatePlayerIdentity { player_id: PlayerId },
    Replay(ReplayError),
    BehaviourEvent(BehaviourEventError),
    BehaviourSequence(BehaviourSequenceError),
}

impl fmt::Display for HandHistoryAdapterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedPlayerCount { players } => {
                write!(f, "LeChiffre supports only 2..=6-max, got {players} players")
            }
            Self::SequenceTooLong => write!(f, "hand history has more than u32::MAX decisions"),
            Self::MissingSourcePlayer { source_player_id } => {
                write!(f, "replay references unknown source player {source_player_id}")
            }
            Self::DuplicatePlayerIdentity { player_id } => {
                write!(f, "duplicate source player identity {player_id} in one hand")
            }
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

        let adapted = HandHistoryAdapter::adapt(&hand, RakeConfig::none()).unwrap();
        let sequence = adapted.sequence();

        assert_eq!(sequence.hand_id().as_str(), "PokerStars:171562910425");
        assert_eq!(sequence.len(), 14);
        assert!(
            sequence
                .events()
                .iter()
                .all(|event| event.action.decision_time_ms.is_none())
        );
    }

    #[test]
    fn maps_real_six_max_fixture_to_canonical_positions() {
        let hand = PokerStarsParser::default()
            .parse(include_str!(
                "../../vendor/rs-poker/tests/fixtures/rooms/pokerstars.txt"
            ))
            .unwrap()
            .remove(0);

        let adapted = HandHistoryAdapter::adapt(&hand, RakeConfig::none()).unwrap();

        let expected = [
            ("PokerStars:zelittle", Position::Cutoff),
            ("PokerStars:yakka34", Position::Button),
            ("PokerStars:Zaco7319", Position::SmallBlind),
            ("PokerStars:Pat10man", Position::BigBlind),
            ("PokerStars:jumpshotftw", Position::UnderTheGun),
            ("PokerStars:Rihan2", Position::Hijack),
        ];

        for (id, position) in expected {
            let player = adapted.player(&PlayerId::new(id)).unwrap();
            assert_eq!(player.position, position);
        }
    }

    #[test]
    fn builds_history_only_from_exact_player_identity() {
        let hand = PokerStarsParser::default()
            .parse(include_str!(
                "../../vendor/rs-poker/tests/fixtures/rooms/pokerstars.txt"
            ))
            .unwrap()
            .remove(0);

        let adapted = HandHistoryAdapter::adapt(&hand, RakeConfig::none()).unwrap();
        let hands = vec![adapted];

        let history = HandHistoryAdapter::history_for(
            &PlayerId::new("PokerStars:yakka34"),
            &hands,
        );
        assert_eq!(history.len(), 1);
        assert_eq!(history.hands()[0].subject_position(), Position::Button);

        let absent = HandHistoryAdapter::history_for(
            &PlayerId::new("PokerStars:not-present"),
            &hands,
        );
        assert!(absent.is_empty());
    }
}
