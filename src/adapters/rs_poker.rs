use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use async_trait::async_trait;
use rs_poker::arena::action::{Action, AgentAction, PlayedActionPayload};
use rs_poker::arena::game_state::Round;
use rs_poker::arena::{Agent, GameState, Historian, HistorianError};
use rs_poker::core::{Card, Suit as RsSuit, Value as RsValue};

use crate::behavioural::{
    ActionKind, BehaviourEvent, BehaviourEventError, BehaviourSequence, BehaviourSequenceError,
    DecisionContext, HandId, ObservedAction, Position, PublicCard, Rank, Street, Suit,
};

type DecisionKey = (u128, usize);

#[derive(Default)]
struct AdapterState {
    pending: HashMap<DecisionKey, PendingDecision>,
    sequences: HashMap<u128, BehaviourSequence>,
}

struct PendingDecision {
    context: DecisionContext,
    big_blind: rs_poker::arena::Chips,
    decision_time_ms: u32,
}

#[derive(Clone, Default)]
pub struct RsPokerAdapter {
    state: Arc<Mutex<AdapterState>>,
}

impl RsPokerAdapter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn wrap_agents(
        &self,
        agents: Vec<Box<dyn Agent>>,
    ) -> Result<Vec<Box<dyn Agent>>, RsPokerAdapterError> {
        validate_player_count(agents.len())?;

        Ok(agents
            .into_iter()
            .map(|agent| {
                Box::new(ObservedAgent {
                    inner: agent,
                    state: Arc::clone(&self.state),
                }) as Box<dyn Agent>
            })
            .collect())
    }

    pub fn historian(&self) -> Box<dyn Historian> {
        Box::new(BehaviourHistorian {
            state: Arc::clone(&self.state),
        })
    }

    pub fn sequence(&self, hand_id: u128) -> Result<Option<BehaviourSequence>, RsPokerAdapterError> {
        let state = self
            .state
            .lock()
            .map_err(|_| RsPokerAdapterError::StoragePoisoned)?;
        Ok(state.sequences.get(&hand_id).cloned())
    }

    pub fn take_sequence(
        &self,
        hand_id: u128,
    ) -> Result<Option<BehaviourSequence>, RsPokerAdapterError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| RsPokerAdapterError::StoragePoisoned)?;
        Ok(state.sequences.remove(&hand_id))
    }
}

struct ObservedAgent {
    inner: Box<dyn Agent>,
    state: Arc<Mutex<AdapterState>>,
}

#[async_trait]
impl Agent for ObservedAgent {
    async fn act(&mut self, id: u128, game_state: &GameState) -> AgentAction {
        let actor_seat = game_state.to_act_idx();
        let context = decision_context(game_state);
        let started = Instant::now();
        let action = self.inner.act(id, game_state).await;
        let decision_time_ms = u32::try_from(started.elapsed().as_millis())
            .expect("decision time exceeded u32 milliseconds");

        self.state
            .lock()
            .expect("rs-poker adapter state lock poisoned")
            .pending
            .insert(
                (id, actor_seat),
                PendingDecision {
                    context,
                    big_blind: game_state.big_blind,
                    decision_time_ms,
                },
            );

        action
    }

    fn name(&self) -> &str {
        self.inner.name()
    }

    fn historian(&self) -> Option<Box<dyn Historian>> {
        self.inner.historian()
    }
}

struct BehaviourHistorian {
    state: Arc<Mutex<AdapterState>>,
}

#[async_trait]
impl Historian for BehaviourHistorian {
    async fn record_action(
        &mut self,
        id: u128,
        _game_state: &GameState,
        action: &Action,
    ) -> Result<(), HistorianError> {
        let payload = match action {
            Action::PlayedAction(payload) => payload,
            Action::FailedAction(payload) => &payload.result,
            _ => return Ok(()),
        };

        record_played_action(&self.state, id, payload)
            .map_err(|_| HistorianError::UnableToRecordAction)
    }
}

fn record_played_action(
    state: &Arc<Mutex<AdapterState>>,
    hand_id: u128,
    payload: &PlayedActionPayload,
) -> Result<(), RsPokerAdapterError> {
    let mut state = state
        .lock()
        .map_err(|_| RsPokerAdapterError::StoragePoisoned)?;

    let pending = state
        .pending
        .remove(&(hand_id, payload.idx))
        .ok_or(RsPokerAdapterError::MissingDecision {
            hand_id,
            seat: payload.idx,
        })?;

    let sequence = state
        .sequences
        .entry(hand_id)
        .or_insert_with(|| BehaviourSequence::new(rs_poker_hand_id(hand_id)));

    let sequence_index = u32::try_from(sequence.len())
        .map_err(|_| RsPokerAdapterError::SequenceTooLong { hand_id })?;
    let event = BehaviourEvent::new(
        rs_poker_hand_id(hand_id),
        sequence_index,
        pending.context,
        observed_action(payload, pending.big_blind, pending.decision_time_ms),
    )?;
    sequence.push(event)?;

    Ok(())
}

pub(super) fn decision_context(game_state: &GameState) -> DecisionContext {
    let actor_seat = game_state.to_act_idx();
    let current_bet = game_state.current_round_bet();
    let actor_commitment = game_state.current_round_player_bet(actor_seat);
    let live_players = game_state.player_active | game_state.player_all_in;

    DecisionContext {
        street: street(game_state.round),
        board: game_state.board.iter().copied().map(public_card).collect(),
        actor_position: position_for_index(
            game_state.num_players,
            game_state.dealer_idx,
            actor_seat,
        ),
        players_at_table: game_state.num_players,
        players_active: live_players.count(),
        pot_bb: to_bb(game_state.total_pot, game_state.big_blind),
        actor_stack_bb: to_bb(game_state.stacks[actor_seat], game_state.big_blind),
        effective_stack_bb: to_bb(
            effective_stack_behind(game_state, actor_seat),
            game_state.big_blind,
        ),
        actor_street_commitment_bb: to_bb(actor_commitment, game_state.big_blind),
        current_bet_bb: to_bb(current_bet, game_state.big_blind),
        to_call_bb: to_bb((current_bet - actor_commitment).max(0), game_state.big_blind),
        min_raise_to_bb: Some(
            to_bb(current_bet, game_state.big_blind)
                + to_bb(game_state.current_round_min_raise(), game_state.big_blind),
        ),
        bet_or_raise_count_this_street: game_state.round_data.total_raise_count,
    }
}

pub(super) fn validate_player_count(players: usize) -> Result<(), RsPokerAdapterError> {
    if (2..=6).contains(&players) {
        Ok(())
    } else {
        Err(RsPokerAdapterError::UnsupportedPlayerCount { players })
    }
}

pub(super) fn position_for_index(
    num_players: usize,
    dealer_idx: usize,
    actor_idx: usize,
) -> Position {
    assert!(
        (2..=6).contains(&num_players),
        "LeChiffre supports only 2..=6-max"
    );
    assert!(dealer_idx < num_players, "dealer index is outside the table");
    assert!(actor_idx < num_players, "actor index is outside the table");

    let distance_from_button = (actor_idx + num_players - dealer_idx) % num_players;

    match (num_players, distance_from_button) {
        (2, 0) => Position::ButtonSmallBlind,
        (2, 1) => Position::BigBlind,
        (_, 0) => Position::Button,
        (_, 1) => Position::SmallBlind,
        (_, 2) => Position::BigBlind,
        (4, 3) => Position::Cutoff,
        (5, 3) => Position::Hijack,
        (5, 4) => Position::Cutoff,
        (6, 3) => Position::UnderTheGun,
        (6, 4) => Position::Hijack,
        (6, 5) => Position::Cutoff,
        _ => unreachable!("validated 2..=6-max position mapping is exhaustive"),
    }
}

fn effective_stack_behind(game_state: &GameState, actor_seat: usize) -> rs_poker::arena::Chips {
    let live_players = game_state.player_active | game_state.player_all_in;
    let deepest_opponent = live_players
        .ones()
        .filter(|&seat| seat != actor_seat)
        .map(|seat| game_state.stacks[seat])
        .max()
        .expect("agent decision occurred without a live opponent");

    game_state.stacks[actor_seat].min(deepest_opponent)
}

fn observed_action(
    payload: &PlayedActionPayload,
    big_blind: rs_poker::arena::Chips,
    decision_time_ms: u32,
) -> ObservedAction {
    let amount_added = payload.final_player_bet - payload.starting_player_bet;

    ObservedAction {
        kind: semantic_action(payload, amount_added),
        is_all_in: payload.players_all_in.get(payload.idx),
        amount_added_bb: to_bb(amount_added, big_blind),
        street_total_after_bb: to_bb(payload.final_player_bet, big_blind),
        decision_time_ms: Some(decision_time_ms),
    }
}

fn semantic_action(payload: &PlayedActionPayload, amount_added: rs_poker::arena::Chips) -> ActionKind {
    if matches!(payload.action, AgentAction::Fold) {
        return ActionKind::Fold;
    }

    if payload.final_bet > payload.starting_bet {
        if payload.starting_bet == 0 {
            ActionKind::Bet
        } else {
            ActionKind::Raise
        }
    } else if amount_added > 0 {
        ActionKind::Call
    } else {
        ActionKind::Check
    }
}

fn street(round: Round) -> Street {
    match round {
        Round::Preflop => Street::Preflop,
        Round::Flop => Street::Flop,
        Round::Turn => Street::Turn,
        Round::River => Street::River,
        other => panic!("agent decision occurred outside a betting street: {other:?}"),
    }
}

fn public_card(card: Card) -> PublicCard {
    PublicCard {
        rank: match card.value {
            RsValue::Two => Rank::Two,
            RsValue::Three => Rank::Three,
            RsValue::Four => Rank::Four,
            RsValue::Five => Rank::Five,
            RsValue::Six => Rank::Six,
            RsValue::Seven => Rank::Seven,
            RsValue::Eight => Rank::Eight,
            RsValue::Nine => Rank::Nine,
            RsValue::Ten => Rank::Ten,
            RsValue::Jack => Rank::Jack,
            RsValue::Queen => Rank::Queen,
            RsValue::King => Rank::King,
            RsValue::Ace => Rank::Ace,
        },
        suit: match card.suit {
            RsSuit::Club => Suit::Clubs,
            RsSuit::Diamond => Suit::Diamonds,
            RsSuit::Heart => Suit::Hearts,
            RsSuit::Spade => Suit::Spades,
        },
    }
}

pub(super) fn to_bb(chips: rs_poker::arena::Chips, big_blind: rs_poker::arena::Chips) -> f32 {
    assert!(big_blind > 0, "big blind must be positive");
    chips as f32 / big_blind as f32
}

fn rs_poker_hand_id(hand_id: u128) -> HandId {
    HandId::new(format!("rs-poker:{hand_id}"))
}

#[derive(Debug)]
pub enum RsPokerAdapterError {
    StoragePoisoned,
    UnsupportedPlayerCount { players: usize },
    SequenceTooLong { hand_id: u128 },
    MissingDecision { hand_id: u128, seat: usize },
    BehaviourEvent(BehaviourEventError),
    BehaviourSequence(BehaviourSequenceError),
}

impl fmt::Display for RsPokerAdapterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StoragePoisoned => write!(f, "behaviour adapter storage lock poisoned"),
            Self::UnsupportedPlayerCount { players } => {
                write!(f, "LeChiffre supports only 2..=6-max, got {players} players")
            }
            Self::SequenceTooLong { hand_id } => {
                write!(f, "hand {hand_id} has more than u32::MAX behavioural events")
            }
            Self::MissingDecision { hand_id, seat } => {
                write!(f, "missing decision snapshot for hand {hand_id}, seat {seat}")
            }
            Self::BehaviourEvent(error) => write!(f, "invalid behaviour event: {error}"),
            Self::BehaviourSequence(error) => write!(f, "invalid behaviour sequence: {error}"),
        }
    }
}

impl std::error::Error for RsPokerAdapterError {}

impl From<BehaviourEventError> for RsPokerAdapterError {
    fn from(error: BehaviourEventError) -> Self {
        Self::BehaviourEvent(error)
    }
}

impl From<BehaviourSequenceError> for RsPokerAdapterError {
    fn from(error: BehaviourSequenceError) -> Self {
        Self::BehaviourSequence(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rs_poker::arena::action::PlayedActionPayload;
    use rs_poker::arena::agent::CallingAgent;
    use rs_poker::arena::{GameStateBuilder, HoldemSimulationBuilder};
    use rs_poker::core::PlayerBitSet;

    #[test]
    fn maps_engine_actions_to_semantic_actions() {
        let players = PlayerBitSet::new(2);
        let base = PlayedActionPayload {
            action: AgentAction::Call,
            idx: 0,
            round: Round::Flop,
            player_stack: 100,
            starting_pot: 20,
            final_pot: 20,
            starting_bet: 0,
            final_bet: 0,
            starting_min_raise: 2,
            final_min_raise: 2,
            starting_player_bet: 0,
            final_player_bet: 0,
            players_active: players,
            players_all_in: PlayerBitSet::default(),
        };

        assert_eq!(semantic_action(&base, 0), ActionKind::Check);

        let mut call = base.clone();
        call.starting_bet = 4;
        call.final_bet = 4;
        call.final_player_bet = 4;
        assert_eq!(semantic_action(&call, 4), ActionKind::Call);

        let mut bet = base.clone();
        bet.final_bet = 4;
        bet.final_player_bet = 4;
        assert_eq!(semantic_action(&bet, 4), ActionKind::Bet);

        let mut raise = call.clone();
        raise.final_bet = 8;
        raise.final_player_bet = 8;
        assert_eq!(semantic_action(&raise, 8), ActionKind::Raise);
    }

    #[test]
    fn maps_positions_for_two_to_six_players() {
        assert_eq!(position_for_index(2, 0, 0), Position::ButtonSmallBlind);
        assert_eq!(position_for_index(2, 0, 1), Position::BigBlind);

        assert_eq!(position_for_index(3, 0, 0), Position::Button);
        assert_eq!(position_for_index(3, 0, 1), Position::SmallBlind);
        assert_eq!(position_for_index(3, 0, 2), Position::BigBlind);

        assert_eq!(position_for_index(4, 0, 3), Position::Cutoff);
        assert_eq!(position_for_index(5, 0, 3), Position::Hijack);
        assert_eq!(position_for_index(5, 0, 4), Position::Cutoff);
        assert_eq!(position_for_index(6, 0, 3), Position::UnderTheGun);
        assert_eq!(position_for_index(6, 0, 4), Position::Hijack);
        assert_eq!(position_for_index(6, 0, 5), Position::Cutoff);

        assert_eq!(position_for_index(6, 4, 4), Position::Button);
        assert_eq!(position_for_index(6, 4, 5), Position::SmallBlind);
        assert_eq!(position_for_index(6, 4, 0), Position::BigBlind);
    }

    #[test]
    fn rejects_tables_above_six_max() {
        assert!(matches!(
            validate_player_count(7),
            Err(RsPokerAdapterError::UnsupportedPlayerCount { players: 7 })
        ));
    }

    #[tokio::test]
    async fn records_behaviour_sequence_from_rs_poker() {
        let adapter = RsPokerAdapter::new();
        let agents: Vec<Box<dyn Agent>> = vec![
            Box::<CallingAgent>::default(),
            Box::<CallingAgent>::default(),
        ];
        let agents = adapter.wrap_agents(agents).unwrap();
        let game_state = GameStateBuilder::new()
            .num_players_with_stack(2, 100)
            .blinds(2, 1)
            .dealer_idx(0)
            .build()
            .unwrap();

        let mut simulation = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .historians(vec![adapter.historian()])
            .build()
            .unwrap();
        let hand_id = simulation.id;

        simulation.run().await;

        let sequence = adapter.sequence(hand_id).unwrap().unwrap();
        assert!(!sequence.is_empty());
        assert!(
            sequence
                .events()
                .iter()
                .all(|event| event.action.decision_time_ms.is_some())
        );
        assert!(
            sequence
                .events()
                .iter()
                .all(|event| event.hand_id == rs_poker_hand_id(hand_id))
        );
    }
}
