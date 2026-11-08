use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use async_trait::async_trait;
use rs_poker::arena::action::{Action, AgentAction, PlayedActionPayload};
use rs_poker::arena::game_state::Round;
use rs_poker::arena::{Agent, GameState, Historian, HistorianError};

use crate::behavioural::{
    ActionHistory, ActionKind, BehaviourEvent, BehaviourEventError, BehaviourSequence,
    BehaviourSequenceError, DecisionContext, ExactAction, GeneratorError, HandId, LegalActionSpace,
    PlayerState, PlayerStatus, Position, Street, WagerBounds,
};

type DecisionKey = (u128, usize);

#[derive(Default)]
struct AdapterState {
    pending: HashMap<DecisionKey, PendingDecision>,
    sequences: HashMap<u128, BehaviourSequence>,
    histories: HashMap<u128, ActionHistory>,
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

    pub fn sequence(
        &self,
        hand_id: u128,
    ) -> Result<Option<BehaviourSequence>, RsPokerAdapterError> {
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
        state.histories.remove(&hand_id);
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
        let mut context = decision_context(game_state);
        self.state
            .lock()
            .expect("rs-poker adapter state lock poisoned")
            .histories
            .entry(id)
            .or_default()
            .prepare(&mut context);
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

    let pending = state.pending.remove(&(hand_id, payload.idx)).ok_or(
        RsPokerAdapterError::MissingDecision {
            hand_id,
            seat: payload.idx,
        },
    )?;

    let action = state.histories.entry(hand_id).or_default().record(
        &pending.context,
        ExactAction {
            kind: semantic_action(
                payload,
                payload.final_player_bet - payload.starting_player_bet,
            ),
            amount_added_bb: to_bb(
                payload.final_player_bet - payload.starting_player_bet,
                pending.big_blind,
            ),
            highest_bet_after_bb: to_bb(payload.final_bet, pending.big_blind),
            is_all_in: payload.players_all_in.get(payload.idx),
            decision_time_ms: Some(pending.decision_time_ms),
        },
    )?;
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
        action,
    )?;
    sequence.push(event)?;

    Ok(())
}

/// Derive the complete NLHE action mask and exact chip bounds from the engine.
/// The behavioural model never infers legal actions from an observed choice.
pub fn exact_legal_actions(game_state: &GameState) -> Result<LegalActionSpace, GeneratorError> {
    let seat = game_state.to_act_idx();
    let stack = game_state.stacks[seat];
    let street_commitment = game_state.current_round_player_bet(seat);
    let current_bet = game_state.current_round_bet();
    let minimum_raise = game_state.current_round_min_raise();
    if stack <= 0 || game_state.big_blind <= 0 {
        return Err(GeneratorError::InvalidLegalActions);
    }
    let to_call = (current_bet - street_commitment).max(0);
    let mut legal = LegalActionSpace {
        fold: true,
        check: to_call == 0,
        call: to_call > 0,
        bet: None,
        raise: None,
    };
    let max_total = street_commitment
        .checked_add(stack)
        .ok_or(GeneratorError::InvalidWagerBounds)?;
    let min_total = current_bet
        .checked_add(minimum_raise)
        .ok_or(GeneratorError::InvalidWagerBounds)?;
    let min_legal_total = min_total.min(max_total);
    if min_legal_total > current_bet
        && !game_state.is_raise_capped()
        && game_state.validate_bet_amount(min_legal_total).is_ok()
        && game_state.validate_bet_amount(max_total).is_ok()
    {
        let bounds = WagerBounds {
            min_added_chips: min_legal_total - street_commitment,
            max_added_chips: stack,
            chip_unit: 1,
            big_blind_chips: game_state.big_blind,
        };
        if current_bet == 0 {
            legal.bet = Some(bounds);
        } else {
            legal.raise = Some(bounds);
        }
    }
    legal.validate(&decision_context(game_state))?;
    Ok(legal)
}

pub(super) fn decision_context(game_state: &GameState) -> DecisionContext {
    let actor_seat = game_state.to_act_idx();
    let position = |seat| position_for_index(game_state.num_players, game_state.dealer_idx, seat);
    let players = (0..game_state.num_players)
        .map(|seat| PlayerState {
            position: position(seat),
            status: if game_state.player_all_in.get(seat) {
                PlayerStatus::AllIn
            } else if game_state.player_active.get(seat) {
                PlayerStatus::Active
            } else {
                PlayerStatus::Folded
            },
            stack_remaining_bb: to_bb(game_state.stacks[seat], game_state.big_blind),
            committed_this_street_bb: to_bb(
                game_state.current_round_player_bet(seat),
                game_state.big_blind,
            ),
            committed_this_hand_bb: to_bb(game_state.player_bet[seat], game_state.big_blind),
        })
        .collect();
    let current_bet = game_state.current_round_bet();
    let actor_commitment = game_state.current_round_player_bet(actor_seat);
    DecisionContext {
        street: street(game_state.round),
        actor_position: position(actor_seat),
        pot_bb: to_bb(game_state.total_pot, game_state.big_blind),
        players,
        current_bet_bb: to_bb(current_bet, game_state.big_blind),
        to_call_bb: to_bb(
            (current_bet - actor_commitment).max(0),
            game_state.big_blind,
        ),
        min_raise_to_bb: Some(to_bb(
            current_bet + game_state.current_round_min_raise(),
            game_state.big_blind,
        )),
        players_yet_to_act: u8::try_from(
            game_state.round_data.needs_action.count().saturating_sub(1),
        )
        .expect("rs-poker supports only six players"),
        bets_and_raises_this_street: 0,
        calls_this_street: 0,
        folds_this_street: 0,
        actor_has_acted_this_street: false,
        actor_aggression_streak_before: 0,
        table_aggression_streak_before: 0,
        last_aggressor: None,
        preflop_last_aggressor_position: None,
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
    assert!(
        dealer_idx < num_players,
        "dealer index is outside the table"
    );
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

fn semantic_action(
    payload: &PlayedActionPayload,
    amount_added: rs_poker::arena::Chips,
) -> ActionKind {
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
                write!(
                    f,
                    "LeChiffre supports only 2..=6-max, got {players} players"
                )
            }
            Self::SequenceTooLong { hand_id } => {
                write!(
                    f,
                    "hand {hand_id} has more than u32::MAX behavioural events"
                )
            }
            Self::MissingDecision { hand_id, seat } => {
                write!(
                    f,
                    "missing decision snapshot for hand {hand_id}, seat {seat}"
                )
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

    #[test]
    fn exact_legality_matches_engine_minimum_and_all_in() {
        let game = GameStateBuilder::new()
            .num_players_with_stack(2, 100)
            .blinds(2, 1)
            .round(Round::Flop)
            .player_bet([2, 2])
            .max_raises_per_round(None)
            .build()
            .unwrap();
        assert_eq!(game.total_pot, 4);
        assert_eq!(game.current_round_bet(), 0);
        let legal = exact_legal_actions(&game).unwrap();
        assert!(legal.check);
        assert!(!legal.call);
        assert_eq!(legal.bet.unwrap().min_added_chips, 2);
        assert_eq!(legal.bet.unwrap().max_added_chips, 100);
        assert_eq!(legal.bet.unwrap().big_blind_chips, 2);

        let mut raised = game;
        raised.do_bet(10, false).unwrap();
        let facing = exact_legal_actions(&raised).unwrap();
        assert!(facing.fold);
        assert!(facing.call);
        assert!(!facing.check);
        assert!(facing.raise.is_some());
        assert_eq!(facing.raise.unwrap().min_added_chips, 20);
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
