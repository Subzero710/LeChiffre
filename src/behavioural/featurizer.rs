use super::codebook::{action_id, position_id, street_id};
use super::{BehaviourEvent, BehaviourHistory, BehaviourSequence, HandId, Position, SubjectHand};

pub const MAX_OPPONENTS: usize = 5;
pub const PLAYER_STATE_WIDTH: usize = 5;
pub const PRE_ACTION_CONTEXT_WIDTH: usize = 26 + MAX_OPPONENTS * PLAYER_STATE_WIDTH;

/// Width of the categorical event matrix produced by [`BehaviourFeaturizer`].
pub const EVENT_CATEGORICAL_WIDTH: usize = 5;

/// Width of the continuous event matrix produced by [`BehaviourFeaturizer`].
pub const EVENT_CONTINUOUS_WIDTH: usize = PRE_ACTION_CONTEXT_WIDTH + 6;

/// Stable columns of [`FeaturizedHand::categorical`].
///
/// Values are integer IDs intended to be embedded by the training model. The
/// ID mappings come from the shared behavioural codebook and are covered by
/// tests so persisted datasets and runtime features cannot silently diverge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub enum CategoricalFeature {
    Street = 0,
    ActorPosition = 1,
    Action = 2,
    IsSubject = 3,
    IsAllIn = 4,
}

/// Stable columns of [`FeaturizedHand::continuous`].
///
/// Values are copied mechanically from `BehaviourEvent` without learned
/// normalization. Big-blind values remain in BB and timing remains in ms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub enum ContinuousFeature {
    PotBb = 0,
    StackRemainingBb = 1,
    CommittedThisStreetBb = 2,
    CommittedThisHandBb = 3,
    CurrentBetBb = 4,
    ToCallBb = 5,
    MinRaiseToBb = 6,
    PlayersYetToAct = 7,
    BetsAndRaisesThisStreet = 8,
    CallsThisStreet = 9,
    FoldsThisStreet = 10,
    ActorHasActedThisStreet = 11,
    ActorAggressionStreakBefore = 12,
    TableAggressionStreakBefore = 13,
    PreflopLastAggressorPosition = 14,
    LastAggressorPosition = 15,
    LastAggressorStreet = 16,
    LastAggressorSequenceIndex = 17,
    LastAggressorActionKind = 18,
    LastAggressorIsAllIn = 19,
    LastAggressorAmountOverPot = 20,
    LastAggressorAmountOverStack = 21,
    LastAggressorRaiseIncrementOverPot = 22,
    LastAggressorActorStreak = 23,
    LastAggressorTableStreak = 24,
    LastAggressorTimingMs = 25,
    // Slots 26..50 contain opponents' position, status and three chip values.
    // Slots are sorted by canonical position ID, and unused slots are masked.
    AmountOverPot = 51,
    AmountOverStack = 52,
    RaiseIncrementOverPot = 53,
    ActorAggressionStreakAfter = 54,
    TableAggressionStreakAfter = 55,
    DecisionTimeMs = 56,
}

/// Tensor-ready representation of one complete behavioural hand.
///
/// Every vector has `len() == sequence_len()`. The row order is exactly the
/// original `BehaviourSequence` order, so no behavioural chronology is lost.
/// Hand IDs and sequence indexes are metadata only and are not model features.
#[derive(Debug, Clone, PartialEq)]
pub struct FeaturizedHand {
    hand_id: HandId,
    subject_position: Position,
    sequence_indexes: Vec<u32>,
    categorical: Vec<[i64; EVENT_CATEGORICAL_WIDTH]>,
    continuous: Vec<[f32; EVENT_CONTINUOUS_WIDTH]>,
    continuous_present: Vec<[u8; EVENT_CONTINUOUS_WIDTH]>,
    decision_time_ms: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FeaturizedHandData {
    pub(crate) hand_id: HandId,
    pub(crate) subject_position: Position,
    pub(crate) sequence_indexes: Vec<u32>,
    pub(crate) categorical: Vec<[i64; EVENT_CATEGORICAL_WIDTH]>,
    pub(crate) continuous: Vec<[f32; EVENT_CONTINUOUS_WIDTH]>,
    pub(crate) continuous_present: Vec<[u8; EVENT_CONTINUOUS_WIDTH]>,
    pub(crate) decision_time_ms: Vec<u32>,
}

impl FeaturizedHand {
    pub(crate) fn from_data(data: FeaturizedHandData) -> Self {
        Self {
            hand_id: data.hand_id,
            subject_position: data.subject_position,
            sequence_indexes: data.sequence_indexes,
            categorical: data.categorical,
            continuous: data.continuous,
            continuous_present: data.continuous_present,
            decision_time_ms: data.decision_time_ms,
        }
    }

    pub fn hand_id(&self) -> &HandId {
        &self.hand_id
    }

    pub fn subject_position(&self) -> Position {
        self.subject_position
    }

    pub fn sequence_len(&self) -> usize {
        self.categorical.len()
    }

    pub fn is_empty(&self) -> bool {
        self.categorical.is_empty()
    }

    pub fn sequence_indexes(&self) -> &[u32] {
        &self.sequence_indexes
    }

    pub fn categorical(&self) -> &[[i64; EVENT_CATEGORICAL_WIDTH]] {
        &self.categorical
    }

    pub fn continuous(&self) -> &[[f32; EVENT_CONTINUOUS_WIDTH]] {
        &self.continuous
    }

    /// Presence mask for `continuous` (`1` = observed, `0` = absent).
    ///
    /// Retained hands always have observed `DecisionTimeMs`. Missing optional
    /// pre-action information, unused opponent slots, and inapplicable sizing
    /// fields are represented by a zero presence mask, not a synthetic value.
    pub fn continuous_present(&self) -> &[[u8; EVENT_CONTINUOUS_WIDTH]] {
        &self.continuous_present
    }

    /// Exact source timing retained separately from the model-ready f32 matrix.
    pub fn decision_time_ms(&self) -> &[u32] {
        &self.decision_time_ms
    }
}

/// Multi-hand tensor-ready representation for one behavioural subject.
///
/// Hands remain separate. Padding/batching across players belongs to the
/// dataset/training layer so this representation never invents synthetic
/// events or silently truncates a hand.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FeaturizedHistory {
    hands: Vec<FeaturizedHand>,
    dropped_hands_missing_timing: usize,
}

impl FeaturizedHistory {
    pub fn hands(&self) -> &[FeaturizedHand] {
        &self.hands
    }

    /// Number of source hands excluded because at least one decision had no
    /// observed wall-clock timing. Such hands are never partially featurized.
    pub fn dropped_hands_missing_timing(&self) -> usize {
        self.dropped_hands_missing_timing
    }

    pub fn len(&self) -> usize {
        self.hands.len()
    }

    pub fn is_empty(&self) -> bool {
        self.hands.is_empty()
    }

    pub fn total_events(&self) -> usize {
        self.hands.iter().map(FeaturizedHand::sequence_len).sum()
    }
}

/// Mechanical conversion from behavioural domain objects to numeric matrices.
///
/// This component does not infer player style, aggregate statistics, normalize
/// from a population, impute missing observations, or choose training labels.
/// A hand containing any event without `decision_time_ms` is excluded entirely
/// so the retained dataset never contains partially observed timing sequences.
#[derive(Debug, Clone, Copy, Default)]
pub struct BehaviourFeaturizer;

impl BehaviourFeaturizer {
    pub fn new() -> Self {
        Self
    }

    pub fn featurize_history(
        &self,
        history: &BehaviourHistory,
    ) -> Result<FeaturizedHistory, BehaviourFeaturizerError> {
        let mut hands = Vec::with_capacity(history.len());
        let mut dropped_hands_missing_timing = 0usize;

        for hand in history.hands() {
            match self.featurize_hand(hand)? {
                Some(hand) => hands.push(hand),
                None => dropped_hands_missing_timing += 1,
            }
        }

        Ok(FeaturizedHistory {
            hands,
            dropped_hands_missing_timing,
        })
    }

    /// Returns `Ok(None)` when any event in the hand has no observed decision
    /// timing. The hand is dropped as a whole to preserve its action sequence.
    pub fn featurize_hand(
        &self,
        hand: &SubjectHand,
    ) -> Result<Option<FeaturizedHand>, BehaviourFeaturizerError> {
        self.featurize_sequence(hand.sequence(), hand.subject_position())
    }

    /// Returns `Ok(None)` when timing is incomplete anywhere in the sequence.
    pub fn featurize_sequence(
        &self,
        sequence: &BehaviourSequence,
        subject_position: Position,
    ) -> Result<Option<FeaturizedHand>, BehaviourFeaturizerError> {
        if sequence
            .events()
            .iter()
            .any(|event| event.action.decision_time_ms.is_none())
        {
            return Ok(None);
        }

        let len = sequence.len();
        let mut sequence_indexes = Vec::with_capacity(len);
        let mut categorical = Vec::with_capacity(len);
        let mut continuous = Vec::with_capacity(len);
        let mut continuous_present = Vec::with_capacity(len);
        let mut decision_time_ms = Vec::with_capacity(len);

        for event in sequence.events() {
            event.context.validate()?;
            event.action.validate_with_context(&event.context)?;
            sequence_indexes.push(event.sequence_index);
            categorical.push([
                i64::from(street_id(event.context.street)),
                i64::from(position_id(event.context.actor_position)),
                i64::from(action_id(event.action.kind)),
                if event.context.actor_position == subject_position {
                    1
                } else {
                    0
                },
                if event.action.is_all_in { 1 } else { 0 },
            ]);

            let (values, present) = encode_continuous(event);
            let Some(exact_decision_time_ms) = event.action.decision_time_ms else {
                return Ok(None);
            };

            continuous.push(values);
            continuous_present.push(present);
            decision_time_ms.push(exact_decision_time_ms);
        }

        Ok(Some(FeaturizedHand {
            hand_id: sequence.hand_id().clone(),
            subject_position,
            sequence_indexes,
            categorical,
            continuous,
            continuous_present,
            decision_time_ms,
        }))
    }
}

pub(crate) fn continuous_feature_name(index: usize) -> String {
    const BASE: [&str; 26] = [
        "pot_bb",
        "actor_stack_remaining_bb",
        "actor_committed_this_street_bb",
        "actor_committed_this_hand_bb",
        "current_bet_bb",
        "to_call_bb",
        "min_raise_to_bb",
        "players_yet_to_act",
        "bets_and_raises_this_street",
        "calls_this_street",
        "folds_this_street",
        "actor_has_acted_this_street",
        "actor_aggression_streak_before",
        "table_aggression_streak_before",
        "preflop_last_aggressor_position",
        "last_aggressor_position",
        "last_aggressor_street",
        "last_aggressor_sequence_index",
        "last_aggressor_action_kind",
        "last_aggressor_is_all_in",
        "last_aggressor_amount_over_pot",
        "last_aggressor_amount_over_stack",
        "last_aggressor_raise_increment_over_pot",
        "last_aggressor_actor_streak",
        "last_aggressor_table_streak",
        "last_aggressor_timing_ms",
    ];
    const PLAYER_FIELDS: [&str; PLAYER_STATE_WIDTH] = [
        "position",
        "status",
        "stack_remaining_bb",
        "committed_this_street_bb",
        "committed_this_hand_bb",
    ];
    const RESPONSE: [&str; 6] = [
        "amount_over_pot_before",
        "amount_over_stack_before",
        "raise_increment_over_pot_before",
        "actor_aggression_streak_after",
        "table_aggression_streak_after",
        "decision_time_ms_f32",
    ];
    match index {
        0..26 => BASE[index].to_owned(),
        26..PRE_ACTION_CONTEXT_WIDTH => {
            let slot = (index - 26) / PLAYER_STATE_WIDTH;
            let field = (index - 26) % PLAYER_STATE_WIDTH;
            format!("opponent_{slot}_{}", PLAYER_FIELDS[field])
        }
        PRE_ACTION_CONTEXT_WIDTH..EVENT_CONTINUOUS_WIDTH => {
            RESPONSE[index - PRE_ACTION_CONTEXT_WIDTH].to_owned()
        }
        _ => panic!("feature index outside the canonical encoding"),
    }
}

/// Canonical encoding shared by live featurization and Parquet persistence.
/// Indices strictly below PRE_ACTION_CONTEXT_WIDTH contain no current-action response.
pub(crate) fn encode_context(
    ctx: &super::DecisionContext,
) -> (
    [f32; PRE_ACTION_CONTEXT_WIDTH],
    [u8; PRE_ACTION_CONTEXT_WIDTH],
) {
    let mut values = [0.0; PRE_ACTION_CONTEXT_WIDTH];
    let mut present = [1; PRE_ACTION_CONTEXT_WIDTH];
    let actor = ctx.actor().expect("validated event includes its actor");
    values[ContinuousFeature::PotBb as usize] = ctx.pot_bb;
    values[ContinuousFeature::StackRemainingBb as usize] = actor.stack_remaining_bb;
    values[ContinuousFeature::CommittedThisStreetBb as usize] = actor.committed_this_street_bb;
    values[ContinuousFeature::CommittedThisHandBb as usize] = actor.committed_this_hand_bb;
    values[ContinuousFeature::CurrentBetBb as usize] = ctx.current_bet_bb;
    values[ContinuousFeature::ToCallBb as usize] = ctx.to_call_bb;
    set_optional(
        &mut values,
        &mut present,
        ContinuousFeature::MinRaiseToBb as usize,
        ctx.min_raise_to_bb,
    );
    values[ContinuousFeature::PlayersYetToAct as usize] = f32::from(ctx.players_yet_to_act);
    values[ContinuousFeature::BetsAndRaisesThisStreet as usize] =
        ctx.bets_and_raises_this_street as f32;
    values[ContinuousFeature::CallsThisStreet as usize] = ctx.calls_this_street as f32;
    values[ContinuousFeature::FoldsThisStreet as usize] = ctx.folds_this_street as f32;
    values[ContinuousFeature::ActorHasActedThisStreet as usize] = if ctx.actor_has_acted_this_street
    {
        1.0
    } else {
        0.0
    };
    values[ContinuousFeature::ActorAggressionStreakBefore as usize] =
        ctx.actor_aggression_streak_before as f32;
    values[ContinuousFeature::TableAggressionStreakBefore as usize] =
        ctx.table_aggression_streak_before as f32;
    set_optional(
        &mut values,
        &mut present,
        ContinuousFeature::PreflopLastAggressorPosition as usize,
        ctx.preflop_last_aggressor_position
            .map(|p| f32::from(position_id(p))),
    );
    let previous = ctx.last_aggressor.as_ref();
    set_optional(
        &mut values,
        &mut present,
        ContinuousFeature::LastAggressorPosition as usize,
        previous.map(|a| f32::from(position_id(a.actor_position))),
    );
    set_optional(
        &mut values,
        &mut present,
        ContinuousFeature::LastAggressorStreet as usize,
        previous.map(|a| f32::from(street_id(a.street))),
    );
    set_optional(
        &mut values,
        &mut present,
        ContinuousFeature::LastAggressorSequenceIndex as usize,
        previous.map(|a| a.sequence_index as f32),
    );
    set_optional(
        &mut values,
        &mut present,
        ContinuousFeature::LastAggressorActionKind as usize,
        previous.map(|a| f32::from(action_id(a.action.kind))),
    );
    set_optional(
        &mut values,
        &mut present,
        ContinuousFeature::LastAggressorIsAllIn as usize,
        previous.map(|a| if a.action.is_all_in { 1.0 } else { 0.0 }),
    );
    set_optional(
        &mut values,
        &mut present,
        ContinuousFeature::LastAggressorAmountOverPot as usize,
        previous.and_then(|a| a.action.sizing.map(|s| s.amount_over_pot_before)),
    );
    set_optional(
        &mut values,
        &mut present,
        ContinuousFeature::LastAggressorAmountOverStack as usize,
        previous.and_then(|a| a.action.sizing.map(|s| s.amount_over_stack_before)),
    );
    set_optional(
        &mut values,
        &mut present,
        ContinuousFeature::LastAggressorRaiseIncrementOverPot as usize,
        previous.and_then(|a| {
            a.action
                .sizing
                .and_then(|s| s.raise_increment_over_pot_before)
        }),
    );
    set_optional(
        &mut values,
        &mut present,
        ContinuousFeature::LastAggressorActorStreak as usize,
        previous.map(|a| a.action.actor_aggression_streak_after as f32),
    );
    set_optional(
        &mut values,
        &mut present,
        ContinuousFeature::LastAggressorTableStreak as usize,
        previous.map(|a| a.action.table_aggression_streak_after as f32),
    );
    set_optional(
        &mut values,
        &mut present,
        ContinuousFeature::LastAggressorTimingMs as usize,
        previous.and_then(|a| a.action.decision_time_ms.map(|t| t as f32)),
    );
    let mut opponents = ctx
        .players
        .iter()
        .filter(|p| p.position != ctx.actor_position)
        .collect::<Vec<_>>();
    opponents.sort_unstable_by_key(|p| position_id(p.position));
    for slot in 0..MAX_OPPONENTS {
        let offset = 26 + slot * PLAYER_STATE_WIDTH;
        if let Some(player) = opponents.get(slot) {
            values[offset] = f32::from(position_id(player.position));
            values[offset + 1] = match player.status {
                super::PlayerStatus::Active => 0.0,
                super::PlayerStatus::Folded => 1.0,
                super::PlayerStatus::AllIn => 2.0,
            };
            values[offset + 2] = player.stack_remaining_bb;
            values[offset + 3] = player.committed_this_street_bb;
            values[offset + 4] = player.committed_this_hand_bb;
        } else {
            present[offset..offset + PLAYER_STATE_WIDTH].fill(0);
        }
    }
    (values, present)
}

pub(crate) fn encode_continuous(
    event: &BehaviourEvent,
) -> ([f32; EVENT_CONTINUOUS_WIDTH], [u8; EVENT_CONTINUOUS_WIDTH]) {
    let mut values = [0.0; EVENT_CONTINUOUS_WIDTH];
    let mut present = [1; EVENT_CONTINUOUS_WIDTH];
    let (context_values, context_present) = encode_context(&event.context);
    values[..PRE_ACTION_CONTEXT_WIDTH].copy_from_slice(&context_values);
    present[..PRE_ACTION_CONTEXT_WIDTH].copy_from_slice(&context_present);
    set_optional(
        &mut values,
        &mut present,
        ContinuousFeature::AmountOverPot as usize,
        event.action.sizing.map(|s| s.amount_over_pot_before),
    );
    set_optional(
        &mut values,
        &mut present,
        ContinuousFeature::AmountOverStack as usize,
        event.action.sizing.map(|s| s.amount_over_stack_before),
    );
    set_optional(
        &mut values,
        &mut present,
        ContinuousFeature::RaiseIncrementOverPot as usize,
        event
            .action
            .sizing
            .and_then(|s| s.raise_increment_over_pot_before),
    );
    values[ContinuousFeature::ActorAggressionStreakAfter as usize] =
        event.action.actor_aggression_streak_after as f32;
    values[ContinuousFeature::TableAggressionStreakAfter as usize] =
        event.action.table_aggression_streak_after as f32;
    set_optional(
        &mut values,
        &mut present,
        ContinuousFeature::DecisionTimeMs as usize,
        event.action.decision_time_ms.map(|t| t as f32),
    );
    (values, present)
}

fn set_optional<const N: usize>(
    values: &mut [f32; N],
    present: &mut [u8; N],
    idx: usize,
    value: Option<f32>,
) {
    match value {
        Some(value) => values[idx] = value,
        None => present[idx] = 0,
    }
}

/// Invalid public observation records are rejected before featurization.
/// Missing timing remains represented as `Ok(None)`.
pub type BehaviourFeaturizerError = super::BehaviourEventError;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::behavioural::{
        ActionHistory, ActionKind, DecisionContext, ExactAction, PlayerState, PlayerStatus, Street,
    };

    fn event(kind: ActionKind) -> BehaviourEvent {
        let mut context = DecisionContext {
            street: Street::Flop,
            actor_position: Position::Button,
            pot_bb: 20.0,
            players: vec![
                PlayerState {
                    position: Position::Button,
                    status: PlayerStatus::Active,
                    stack_remaining_bb: 50.0,
                    committed_this_street_bb: 0.0,
                    committed_this_hand_bb: 10.0,
                },
                PlayerState {
                    position: Position::BigBlind,
                    status: PlayerStatus::Active,
                    stack_remaining_bb: 50.0,
                    committed_this_street_bb: 0.0,
                    committed_this_hand_bb: 10.0,
                },
            ],
            current_bet_bb: 0.0,
            to_call_bb: 0.0,
            min_raise_to_bb: Some(1.0),
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
        let mut tracker = ActionHistory::new();
        tracker.prepare(&mut context);
        let amount = if kind == ActionKind::Bet { 5.0 } else { 0.0 };
        let action = tracker
            .record(
                &context,
                ExactAction {
                    kind,
                    amount_added_bb: amount,
                    highest_bet_after_bb: amount,
                    is_all_in: false,
                    decision_time_ms: Some(321),
                },
            )
            .unwrap();
        BehaviourEvent::new("one", 0, context, action).unwrap()
    }

    #[test]
    fn check_has_zero_sizing_fold_has_no_sizing() {
        let check = event(ActionKind::Check);
        let fold = event(ActionKind::Fold);
        let (cv, cp) = encode_continuous(&check);
        let (_, fp) = encode_continuous(&fold);
        assert_eq!(cv[ContinuousFeature::AmountOverPot as usize], 0.0);
        assert_eq!(cp[ContinuousFeature::AmountOverPot as usize], 1);
        assert_eq!(fp[ContinuousFeature::AmountOverPot as usize], 0);
    }

    #[test]
    fn pre_action_features_exclude_response() {
        let bet = event(ActionKind::Bet);
        let check = event(ActionKind::Check);
        let (a, _) = encode_continuous(&bet);
        let (b, _) = encode_continuous(&check);
        assert_eq!(
            &a[..PRE_ACTION_CONTEXT_WIDTH],
            &b[..PRE_ACTION_CONTEXT_WIDTH]
        );
        assert_eq!(a[ContinuousFeature::AmountOverPot as usize], 0.25);
        assert_eq!(a[ContinuousFeature::AmountOverStack as usize], 0.1);
    }

    #[test]
    fn featurizer_keeps_exact_timing() {
        let mut sequence = BehaviourSequence::new("one");
        sequence.push(event(ActionKind::Bet)).unwrap();
        let hand = BehaviourFeaturizer
            .featurize_sequence(&sequence, Position::Button)
            .unwrap()
            .unwrap();
        assert_eq!(hand.decision_time_ms(), &[321]);
        assert_eq!(hand.sequence_len(), 1);
    }
    #[test]
    fn incomplete_timing_drops_the_whole_hand() {
        let mut sequence = BehaviourSequence::new("one");
        sequence.push(event(ActionKind::Check)).unwrap();
        let mut missing = event(ActionKind::Bet);
        missing.sequence_index = 1;
        missing.action.decision_time_ms = None;
        sequence.push(missing).unwrap();
        assert!(
            BehaviourFeaturizer
                .featurize_sequence(&sequence, Position::Button)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn timing_preserves_u32_precision() {
        let mut sequence = BehaviourSequence::new("one");
        let mut precise = event(ActionKind::Check);
        precise.action.decision_time_ms = Some(16_777_217);
        sequence.push(precise).unwrap();
        let encoded = BehaviourFeaturizer
            .featurize_sequence(&sequence, Position::Button)
            .unwrap()
            .unwrap();
        assert_eq!(encoded.decision_time_ms(), &[16_777_217]);
    }

    #[test]
    fn subject_marker_is_position_based() {
        let mut sequence = BehaviourSequence::new("one");
        let mut other = event(ActionKind::Check);
        other.context.actor_position = Position::BigBlind;
        // Construct a matching, valid actor state in the existing player list.
        sequence.push(other).unwrap();
        let mut actor = event(ActionKind::Bet);
        actor.sequence_index = 1;
        sequence.push(actor).unwrap();
        let encoded = BehaviourFeaturizer
            .featurize_sequence(&sequence, Position::Button)
            .unwrap()
            .unwrap();
        assert_eq!(
            encoded.categorical()[0][CategoricalFeature::IsSubject as usize],
            0
        );
        assert_eq!(
            encoded.categorical()[1][CategoricalFeature::IsSubject as usize],
            1
        );
    }
}
