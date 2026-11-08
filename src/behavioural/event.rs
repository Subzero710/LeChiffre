use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct HandId(String);

impl HandId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn into_string(self) -> String {
        self.0
    }
}

impl From<String> for HandId {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl From<&str> for HandId {
    fn from(value: &str) -> Self {
        Self(value.to_owned())
    }
}

impl fmt::Display for HandId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Canonical table position for LeChiffre's 2..=6-max NLHE scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Position {
    UnderTheGun,
    Hijack,
    Cutoff,
    Button,
    ButtonSmallBlind,
    SmallBlind,
    BigBlind,
}

/// Betting street visible to every player.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Street {
    Preflop,
    Flop,
    Turn,
    River,
}

/// Semantic action observed at the table, with check/call and bet/raise kept distinct.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ActionKind {
    Fold,
    Check,
    Call,
    Bet,
    Raise,
}

use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlayerStatus {
    Active,
    Folded,
    AllIn,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PlayerState {
    pub position: Position,
    pub status: PlayerStatus,
    pub stack_remaining_bb: f32,
    pub committed_this_street_bb: f32,
    pub committed_this_hand_bb: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ActionSizing {
    /// Newly contributed chips divided by the pot immediately before the action.
    pub amount_over_pot_before: f32,
    /// Newly contributed chips divided by available stack immediately before the action.
    pub amount_over_stack_before: f32,
    /// Increase of the highest street bet, divided by pre-action pot; only Raise.
    pub raise_increment_over_pot_before: Option<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ObservedAction {
    pub kind: ActionKind,
    pub is_all_in: bool,
    /// None iff Fold; zero ratios iff Check.
    pub sizing: Option<ActionSizing>,
    /// Consecutive aggressive decisions taken by this actor, after this action.
    pub actor_aggression_streak_after: u32,
    /// Consecutive Bet/Raise table decisions within the current street, after this action.
    pub table_aggression_streak_after: u32,
    pub decision_time_ms: Option<u32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PreviousAggression {
    pub actor_position: Position,
    pub street: Street,
    pub sequence_index: u32,
    pub action: ObservedAction,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecisionContext {
    pub street: Street,
    pub actor_position: Position,
    /// Includes every chip currently in the pot, including live street bets.
    pub pot_bb: f32,
    pub players: Vec<PlayerState>,
    pub current_bet_bb: f32,
    pub to_call_bb: f32,
    /// Total street commitment required for a full legal raise, when available.
    pub min_raise_to_bb: Option<f32>,
    pub players_yet_to_act: u8,
    pub bets_and_raises_this_street: u32,
    pub calls_this_street: u32,
    pub folds_this_street: u32,
    pub actor_has_acted_this_street: bool,
    pub actor_aggression_streak_before: u32,
    pub table_aggression_streak_before: u32,
    pub last_aggressor: Option<PreviousAggression>,
    pub preflop_last_aggressor_position: Option<Position>,
}

impl DecisionContext {
    pub fn actor(&self) -> Result<&PlayerState, BehaviourEventError> {
        self.players
            .iter()
            .find(|p| p.position == self.actor_position)
            .ok_or(BehaviourEventError::MissingActor)
    }

    pub fn validate(&self) -> Result<(), BehaviourEventError> {
        if !(2..=6).contains(&self.players.len()) {
            return Err(BehaviourEventError::InvalidPlayerCount);
        }
        for (name, number) in [
            ("pot_bb", self.pot_bb),
            ("current_bet_bb", self.current_bet_bb),
            ("to_call_bb", self.to_call_bb),
        ] {
            nonnegative(name, number)?;
        }
        if let Some(value) = self.min_raise_to_bb {
            nonnegative("min_raise_to_bb", value)?;
        }
        for (i, player) in self.players.iter().enumerate() {
            if self.players[..i]
                .iter()
                .any(|p| p.position == player.position)
            {
                return Err(BehaviourEventError::DuplicatePosition);
            }
            nonnegative("stack_remaining_bb", player.stack_remaining_bb)?;
            nonnegative("committed_this_street_bb", player.committed_this_street_bb)?;
            nonnegative("committed_this_hand_bb", player.committed_this_hand_bb)?;
            if player.committed_this_street_bb > player.committed_this_hand_bb + 1e-4 {
                return Err(BehaviourEventError::InconsistentCommitment);
            }
        }
        let committed_total: f32 = self.players.iter().map(|p| p.committed_this_hand_bb).sum();
        let max_street_bet = self
            .players
            .iter()
            .map(|p| p.committed_this_street_bb)
            .fold(0.0_f32, f32::max);
        if !close(committed_total, self.pot_bb) || !close(max_street_bet, self.current_bet_bb) {
            return Err(BehaviourEventError::InconsistentPot);
        }
        let actor = self.actor()?;
        if actor.status != PlayerStatus::Active {
            return Err(BehaviourEventError::ActorNotActive);
        }
        let expected_call = (self.current_bet_bb - actor.committed_this_street_bb).max(0.0);
        if !close(self.to_call_bb, expected_call) {
            return Err(BehaviourEventError::InconsistentCall);
        }
        if self.players_yet_to_act as usize >= self.players.len() {
            return Err(BehaviourEventError::InvalidPlayerCount);
        }
        if let Some(aggression) = &self.last_aggressor {
            if !is_aggression(aggression.action.kind) {
                return Err(BehaviourEventError::InvalidAggressor);
            }
            if !self
                .players
                .iter()
                .any(|p| p.position == aggression.actor_position)
            {
                return Err(BehaviourEventError::InvalidAggressor);
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExactAction {
    pub kind: ActionKind,
    /// Fresh chips placed during the action, measured by the game engine.
    pub amount_added_bb: f32,
    /// Final highest bet on the current street, measured by the game engine.
    pub highest_bet_after_bb: f32,
    pub is_all_in: bool,
    pub decision_time_ms: Option<u32>,
}

impl ObservedAction {
    /// Strict projection of an exact engine action onto behavioural ratios.
    pub fn from_exact(
        context: &DecisionContext,
        exact: ExactAction,
    ) -> Result<Self, BehaviourEventError> {
        context.validate()?;
        nonnegative("amount_added_bb", exact.amount_added_bb)?;
        nonnegative("highest_bet_after_bb", exact.highest_bet_after_bb)?;
        let actor = context.actor()?;
        if exact.amount_added_bb > actor.stack_remaining_bb + 1e-4 {
            return Err(BehaviourEventError::ExceedsStack);
        }
        if exact.is_all_in != close(exact.amount_added_bb, actor.stack_remaining_bb) {
            return Err(BehaviourEventError::InconsistentAllIn);
        }
        let sizing = match exact.kind {
            ActionKind::Fold => {
                if exact.amount_added_bb != 0.0 || exact.is_all_in {
                    return Err(BehaviourEventError::InvalidPassiveAction);
                }
                None
            }
            ActionKind::Check => {
                if exact.amount_added_bb != 0.0 || context.to_call_bb != 0.0 || exact.is_all_in {
                    return Err(BehaviourEventError::InvalidPassiveAction);
                }
                Some(ActionSizing {
                    amount_over_pot_before: 0.0,
                    amount_over_stack_before: 0.0,
                    raise_increment_over_pot_before: None,
                })
            }
            ActionKind::Call | ActionKind::Bet | ActionKind::Raise => {
                if exact.kind == ActionKind::Call
                    && !close(exact.amount_added_bb, context.to_call_bb)
                    && !(exact.is_all_in && exact.amount_added_bb < context.to_call_bb)
                {
                    return Err(BehaviourEventError::InconsistentCall);
                }
                if exact.kind == ActionKind::Bet && context.current_bet_bb != 0.0 {
                    return Err(BehaviourEventError::InvalidAggressiveAction);
                }
                if exact.kind != ActionKind::Raise
                    && !close(
                        exact.highest_bet_after_bb,
                        if exact.kind == ActionKind::Bet {
                            exact.amount_added_bb + actor.committed_this_street_bb
                        } else {
                            context.current_bet_bb
                        },
                    )
                {
                    return Err(BehaviourEventError::InvalidAggressiveAction);
                }
                if exact.kind == ActionKind::Raise
                    && !close(
                        exact.highest_bet_after_bb,
                        exact.amount_added_bb + actor.committed_this_street_bb,
                    )
                {
                    return Err(BehaviourEventError::InvalidAggressiveAction);
                }
                if exact.kind == ActionKind::Raise
                    && exact.highest_bet_after_bb <= context.current_bet_bb
                {
                    return Err(BehaviourEventError::InvalidAggressiveAction);
                }
                if exact.amount_added_bb <= 0.0 {
                    return Err(BehaviourEventError::InvalidAggressiveAction);
                }
                if context.pot_bb <= 0.0 || actor.stack_remaining_bb <= 0.0 {
                    return Err(BehaviourEventError::UndefinedRatio);
                }
                let raise_increment_over_pot_before = if exact.kind == ActionKind::Raise {
                    Some((exact.highest_bet_after_bb - context.current_bet_bb) / context.pot_bb)
                } else {
                    None
                };
                Some(ActionSizing {
                    amount_over_pot_before: exact.amount_added_bb / context.pot_bb,
                    amount_over_stack_before: exact.amount_added_bb / actor.stack_remaining_bb,
                    raise_increment_over_pot_before,
                })
            }
        };
        let aggressive = is_aggression(exact.kind);
        Ok(Self {
            kind: exact.kind,
            is_all_in: exact.is_all_in,
            sizing,
            actor_aggression_streak_after: if aggressive {
                context
                    .actor_aggression_streak_before
                    .checked_add(1)
                    .ok_or(BehaviourEventError::CounterOverflow)?
            } else {
                0
            },
            table_aggression_streak_after: if aggressive {
                context
                    .table_aggression_streak_before
                    .checked_add(1)
                    .ok_or(BehaviourEventError::CounterOverflow)?
            } else {
                0
            },
            decision_time_ms: exact.decision_time_ms,
        })
    }
}

/// A complete behavioural observation. Hand identity and row order remain metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct BehaviourEvent {
    pub hand_id: HandId,
    pub sequence_index: u32,
    pub context: DecisionContext,
    pub action: ObservedAction,
}

/// Chronological state used to annotate an event before its response is known.
/// One recorder is used per hand; it never reads cards.
#[derive(Debug, Default)]
pub struct ActionHistory {
    actor_streaks: HashMap<Position, u32>,
    table_streak: u32,
    street: Option<Street>,
    last_aggressor: Option<PreviousAggression>,
    preflop_last_aggressor_position: Option<Position>,
    actor_acted_street: HashMap<Position, bool>,
    bets_and_raises: u32,
    calls: u32,
    folds: u32,
    sequence_index: u32,
}

impl ActionHistory {
    pub fn new() -> Self {
        Self::default()
    }

    /// Prepare the pre-action context. All historical values are strictly prior.
    pub fn prepare(&mut self, context: &mut DecisionContext) {
        if self.street != Some(context.street) {
            self.street = Some(context.street);
            self.table_streak = 0;
            self.actor_acted_street.clear();
            self.bets_and_raises = 0;
            self.calls = 0;
            self.folds = 0;
        }
        context.actor_aggression_streak_before = self
            .actor_streaks
            .get(&context.actor_position)
            .copied()
            .unwrap_or(0);
        context.table_aggression_streak_before = self.table_streak;
        context.actor_has_acted_this_street = self
            .actor_acted_street
            .contains_key(&context.actor_position);
        context.bets_and_raises_this_street = self.bets_and_raises;
        context.calls_this_street = self.calls;
        context.folds_this_street = self.folds;
        context.last_aggressor = self.last_aggressor.clone();
        context.preflop_last_aggressor_position = self.preflop_last_aggressor_position;
    }

    /// Compute the response and advance the history only on success.
    pub fn record(
        &mut self,
        context: &DecisionContext,
        exact: ExactAction,
    ) -> Result<ObservedAction, BehaviourEventError> {
        if self.street != Some(context.street) {
            return Err(BehaviourEventError::UnpreparedContext);
        }
        if context.actor_aggression_streak_before
            != self
                .actor_streaks
                .get(&context.actor_position)
                .copied()
                .unwrap_or(0)
            || context.table_aggression_streak_before != self.table_streak
            || context.last_aggressor != self.last_aggressor
            || context.actor_has_acted_this_street
                != self
                    .actor_acted_street
                    .contains_key(&context.actor_position)
            || context.bets_and_raises_this_street != self.bets_and_raises
            || context.calls_this_street != self.calls
            || context.folds_this_street != self.folds
            || context.preflop_last_aggressor_position != self.preflop_last_aggressor_position
        {
            return Err(BehaviourEventError::UnpreparedContext);
        }
        let response = ObservedAction::from_exact(context, exact)?;
        let next_index = self
            .sequence_index
            .checked_add(1)
            .ok_or(BehaviourEventError::CounterOverflow)?;
        let next_bets = if is_aggression(response.kind) {
            self.bets_and_raises
                .checked_add(1)
                .ok_or(BehaviourEventError::CounterOverflow)?
        } else {
            self.bets_and_raises
        };
        let next_calls = if response.kind == ActionKind::Call {
            self.calls
                .checked_add(1)
                .ok_or(BehaviourEventError::CounterOverflow)?
        } else {
            self.calls
        };
        let next_folds = if response.kind == ActionKind::Fold {
            self.folds
                .checked_add(1)
                .ok_or(BehaviourEventError::CounterOverflow)?
        } else {
            self.folds
        };
        self.bets_and_raises = next_bets;
        self.calls = next_calls;
        self.folds = next_folds;
        self.actor_acted_street.insert(context.actor_position, true);
        self.actor_streaks.insert(
            context.actor_position,
            response.actor_aggression_streak_after,
        );
        self.table_streak = response.table_aggression_streak_after;
        if is_aggression(response.kind) {
            self.last_aggressor = Some(PreviousAggression {
                actor_position: context.actor_position,
                street: context.street,
                sequence_index: self.sequence_index,
                action: response.clone(),
            });
            if context.street == Street::Preflop {
                self.preflop_last_aggressor_position = Some(context.actor_position);
            }
        }
        self.sequence_index = next_index;
        Ok(response)
    }
}

fn is_aggression(kind: ActionKind) -> bool {
    matches!(kind, ActionKind::Bet | ActionKind::Raise)
}
fn close(left: f32, right: f32) -> bool {
    (left - right).abs() <= 1e-4 * left.abs().max(right.abs()).max(1.0)
}
fn nonnegative(field: &'static str, value: f32) -> Result<(), BehaviourEventError> {
    if !value.is_finite() || value < 0.0 {
        Err(BehaviourEventError::InvalidAmount(field))
    } else {
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BehaviourEventError {
    InvalidAmount(&'static str),
    InvalidPlayerCount,
    DuplicatePosition,
    MissingActor,
    ActorNotActive,
    InconsistentCommitment,
    InconsistentPot,
    InconsistentCall,
    InvalidAggressor,
    InvalidPassiveAction,
    InvalidAggressiveAction,
    ExceedsStack,
    InconsistentAllIn,
    UndefinedRatio,
    CounterOverflow,
    UnpreparedContext,
    InvalidActionSizing,
    InconsistentStreak,
}
impl fmt::Display for BehaviourEventError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for BehaviourEventError {}

impl BehaviourEvent {
    pub fn new(
        hand_id: impl Into<HandId>,
        sequence_index: u32,
        context: DecisionContext,
        action: ObservedAction,
    ) -> Result<Self, BehaviourEventError> {
        context.validate()?;
        action.validate()?;
        action.validate_with_context(&context)?;
        if context
            .last_aggressor
            .as_ref()
            .is_some_and(|a| a.sequence_index >= sequence_index)
        {
            return Err(BehaviourEventError::InvalidAggressor);
        }
        Ok(Self {
            hand_id: hand_id.into(),
            sequence_index,
            context,
            action,
        })
    }
    pub fn actor_position(&self) -> Position {
        self.context.actor_position
    }
}

impl ObservedAction {
    fn validate(&self) -> Result<(), BehaviourEventError> {
        match (self.kind, self.sizing) {
            (ActionKind::Fold, None) => {}
            (ActionKind::Check, Some(s))
                if s.amount_over_pot_before == 0.0
                    && s.amount_over_stack_before == 0.0
                    && s.raise_increment_over_pot_before.is_none() => {}
            (ActionKind::Call | ActionKind::Bet, Some(s))
                if s.raise_increment_over_pot_before.is_none() =>
            {
                nonnegative("amount_over_pot_before", s.amount_over_pot_before)?;
                nonnegative("amount_over_stack_before", s.amount_over_stack_before)?;
            }
            (ActionKind::Raise, Some(s)) if s.raise_increment_over_pot_before.is_some() => {
                nonnegative("amount_over_pot_before", s.amount_over_pot_before)?;
                nonnegative("amount_over_stack_before", s.amount_over_stack_before)?;
                nonnegative(
                    "raise_increment_over_pot_before",
                    s.raise_increment_over_pot_before.unwrap(),
                )?;
            }
            _ => return Err(BehaviourEventError::InvalidActionSizing),
        }
        Ok(())
    }
    pub(crate) fn validate_with_context(
        &self,
        context: &DecisionContext,
    ) -> Result<(), BehaviourEventError> {
        self.validate()?;
        let actor = context.actor()?;
        let aggressive = is_aggression(self.kind);
        let expected_actor_streak = if aggressive {
            context
                .actor_aggression_streak_before
                .checked_add(1)
                .ok_or(BehaviourEventError::CounterOverflow)?
        } else {
            0
        };
        let expected_table_streak = if aggressive {
            context
                .table_aggression_streak_before
                .checked_add(1)
                .ok_or(BehaviourEventError::CounterOverflow)?
        } else {
            0
        };
        if self.actor_aggression_streak_after != expected_actor_streak
            || self.table_aggression_streak_after != expected_table_streak
        {
            return Err(BehaviourEventError::InconsistentStreak);
        }
        let Some(s) = self.sizing else {
            if self.kind != ActionKind::Fold || self.is_all_in {
                return Err(BehaviourEventError::InvalidActionSizing);
            }
            return Ok(());
        };
        if self.kind == ActionKind::Check {
            if context.to_call_bb != 0.0 || self.is_all_in {
                return Err(BehaviourEventError::InvalidPassiveAction);
            }
            return Ok(());
        }
        if context.pot_bb <= 0.0 || actor.stack_remaining_bb <= 0.0 {
            return Err(BehaviourEventError::UndefinedRatio);
        }
        let amount = actor.stack_remaining_bb * s.amount_over_stack_before;
        if !close(amount, context.pot_bb * s.amount_over_pot_before) {
            return Err(BehaviourEventError::InvalidActionSizing);
        }
        if amount <= 0.0 || amount > actor.stack_remaining_bb + 1e-4 {
            return Err(BehaviourEventError::ExceedsStack);
        }
        if self.is_all_in != close(amount, actor.stack_remaining_bb) {
            return Err(BehaviourEventError::InconsistentAllIn);
        }
        match self.kind {
            ActionKind::Call => {
                if context.to_call_bb <= 0.0
                    || !close(amount, context.to_call_bb.min(actor.stack_remaining_bb))
                {
                    return Err(BehaviourEventError::InconsistentCall);
                }
            }
            ActionKind::Bet => {
                if context.current_bet_bb != 0.0 {
                    return Err(BehaviourEventError::InvalidAggressiveAction);
                }
            }
            ActionKind::Raise => {
                let increment = (actor.committed_this_street_bb + amount - context.current_bet_bb)
                    / context.pot_bb;
                if increment <= 0.0 || !close(s.raise_increment_over_pot_before.unwrap(), increment)
                {
                    return Err(BehaviourEventError::InvalidAggressiveAction);
                }
            }
            ActionKind::Fold | ActionKind::Check => {
                return Err(BehaviourEventError::InvalidActionSizing);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn context(
        street: Street,
        actor: Position,
        pot: f32,
        current: f32,
        actor_stack: f32,
        actor_commitment: f32,
    ) -> DecisionContext {
        let other = if actor == Position::Button {
            Position::BigBlind
        } else {
            Position::Button
        };
        DecisionContext {
            street,
            actor_position: actor,
            pot_bb: pot,
            players: vec![
                PlayerState {
                    position: actor,
                    status: PlayerStatus::Active,
                    stack_remaining_bb: actor_stack,
                    committed_this_street_bb: actor_commitment,
                    committed_this_hand_bb: actor_commitment + 10.0,
                },
                PlayerState {
                    position: other,
                    status: PlayerStatus::Active,
                    stack_remaining_bb: 100.0,
                    committed_this_street_bb: current,
                    committed_this_hand_bb: current + 10.0,
                },
            ],
            current_bet_bb: current,
            to_call_bb: (current - actor_commitment).max(0.0),
            min_raise_to_bb: Some(current + 1.0),
            players_yet_to_act: 1,
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

    #[test]
    fn fold_has_no_sizing_check_has_explicit_zero() {
        for (kind, expected_sizing) in [(ActionKind::Fold, false), (ActionKind::Check, true)] {
            let mut ctx = context(Street::Flop, Position::Button, 20.0, 0.0, 50.0, 0.0);
            let mut recorder = ActionHistory::new();
            recorder.prepare(&mut ctx);
            let response = recorder
                .record(
                    &ctx,
                    ExactAction {
                        kind,
                        amount_added_bb: 0.0,
                        highest_bet_after_bb: 0.0,
                        is_all_in: false,
                        decision_time_ms: Some(320),
                    },
                )
                .unwrap();
            assert_eq!(response.sizing.is_some(), expected_sizing);
            if let Some(s) = response.sizing {
                assert_eq!(s.amount_over_pot_before, 0.0);
                assert_eq!(s.amount_over_stack_before, 0.0);
            }
            BehaviourEvent::new("test", 0, ctx, response).unwrap();
        }
    }

    #[test]
    fn aggressor_uses_original_pot_and_streaks_are_chronological() {
        let mut recorder = ActionHistory::new();
        let mut first = context(Street::Flop, Position::Button, 20.0, 0.0, 50.0, 0.0);
        recorder.prepare(&mut first);
        let bet = recorder
            .record(
                &first,
                ExactAction {
                    kind: ActionKind::Bet,
                    amount_added_bb: 5.0,
                    highest_bet_after_bb: 5.0,
                    is_all_in: false,
                    decision_time_ms: Some(600),
                },
            )
            .unwrap();
        assert_eq!(bet.actor_aggression_streak_after, 1);
        assert_eq!(bet.table_aggression_streak_after, 1);
        let mut second = context(Street::Flop, Position::BigBlind, 25.0, 5.0, 50.0, 0.0);
        recorder.prepare(&mut second);
        assert_eq!(second.table_aggression_streak_before, 1);
        let previous = second.last_aggressor.as_ref().unwrap();
        assert_eq!(previous.actor_position, Position::Button);
        assert_eq!(previous.sequence_index, 0);
        assert_eq!(previous.action.sizing.unwrap().amount_over_pot_before, 0.25);
        let raise = recorder
            .record(
                &second,
                ExactAction {
                    kind: ActionKind::Raise,
                    amount_added_bb: 15.0,
                    highest_bet_after_bb: 15.0,
                    is_all_in: false,
                    decision_time_ms: Some(900),
                },
            )
            .unwrap();
        assert_eq!(raise.table_aggression_streak_after, 2);
        assert_eq!(
            raise.sizing.unwrap().raise_increment_over_pot_before,
            Some(0.4)
        );
        BehaviourEvent::new("test", 1, second, raise).unwrap();
        let mut third = context(Street::Turn, Position::Button, 40.0, 0.0, 45.0, 0.0);
        recorder.prepare(&mut third);
        assert_eq!(third.table_aggression_streak_before, 0);
        assert_eq!(third.actor_aggression_streak_before, 1);
        assert_eq!(third.last_aggressor.unwrap().action.kind, ActionKind::Raise);
    }

    #[test]
    fn tampered_ratio_is_rejected() {
        let mut ctx = context(Street::Flop, Position::Button, 20.0, 0.0, 50.0, 0.0);
        let mut recorder = ActionHistory::new();
        recorder.prepare(&mut ctx);
        let mut action = recorder
            .record(
                &ctx,
                ExactAction {
                    kind: ActionKind::Bet,
                    amount_added_bb: 5.0,
                    highest_bet_after_bb: 5.0,
                    is_all_in: false,
                    decision_time_ms: Some(900),
                },
            )
            .unwrap();
        action.sizing.as_mut().unwrap().amount_over_pot_before = 0.9;
        assert_eq!(
            BehaviourEvent::new("test", 0, ctx, action),
            Err(BehaviourEventError::InvalidActionSizing)
        );
    }
}
