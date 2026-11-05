use super::codebook::{action_id, position_id, street_id};
use super::{BehaviourHistory, BehaviourSequence, HandId, Position, SubjectHand};

/// Width of the categorical event matrix produced by [`BehaviourFeaturizer`].
pub const EVENT_CATEGORICAL_WIDTH: usize = 5;

/// Width of the continuous event matrix produced by [`BehaviourFeaturizer`].
pub const EVENT_CONTINUOUS_WIDTH: usize = 13;

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
    PlayersAtTable = 0,
    PlayersActive = 1,
    PotBb = 2,
    ActorStackBb = 3,
    EffectiveStackBb = 4,
    ActorStreetCommitmentBb = 5,
    CurrentBetBb = 6,
    ToCallBb = 7,
    MinRaiseToBb = 8,
    BetOrRaiseCountThisStreet = 9,
    AmountAddedBb = 10,
    StreetTotalAfterBb = 11,
    DecisionTimeMs = 12,
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
    /// Retained hands always have observed `DecisionTimeMs`. At present only
    /// `MinRaiseToBb` can be absent; its `0.0` storage value must be ignored
    /// whenever the corresponding mask value is `0`.
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

            let mut values = [0.0; EVENT_CONTINUOUS_WIDTH];
            let mut present = [1; EVENT_CONTINUOUS_WIDTH];

            values[ContinuousFeature::PlayersAtTable as usize] =
                event.context.players_at_table as f32;
            values[ContinuousFeature::PlayersActive as usize] = event.context.players_active as f32;
            values[ContinuousFeature::PotBb as usize] = event.context.pot_bb;
            values[ContinuousFeature::ActorStackBb as usize] = event.context.actor_stack_bb;
            values[ContinuousFeature::EffectiveStackBb as usize] = event.context.effective_stack_bb;
            values[ContinuousFeature::ActorStreetCommitmentBb as usize] =
                event.context.actor_street_commitment_bb;
            values[ContinuousFeature::CurrentBetBb as usize] = event.context.current_bet_bb;
            values[ContinuousFeature::ToCallBb as usize] = event.context.to_call_bb;
            values[ContinuousFeature::BetOrRaiseCountThisStreet as usize] =
                f32::from(event.context.bet_or_raise_count_this_street);
            values[ContinuousFeature::AmountAddedBb as usize] = event.action.amount_added_bb;
            values[ContinuousFeature::StreetTotalAfterBb as usize] =
                event.action.street_total_after_bb;

            match event.context.min_raise_to_bb {
                Some(value) => values[ContinuousFeature::MinRaiseToBb as usize] = value,
                None => present[ContinuousFeature::MinRaiseToBb as usize] = 0,
            }

            let Some(exact_decision_time_ms) = event.action.decision_time_ms else {
                return Ok(None);
            };
            values[ContinuousFeature::DecisionTimeMs as usize] = exact_decision_time_ms as f32;

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

/// Feature encoding has no recoverable error after context validation.
/// Missing timing remains represented as `Ok(None)`.
pub type BehaviourFeaturizerError = std::convert::Infallible;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::behavioural::{ActionKind, BehaviourEvent, DecisionContext, ObservedAction, Street};

    fn event(
        hand_id: &str,
        sequence_index: u32,
        position: Position,
        action: ActionKind,
        street: Street,
        min_raise_to_bb: Option<f32>,
        decision_time_ms: Option<u32>,
    ) -> BehaviourEvent {
        BehaviourEvent::new(
            hand_id,
            sequence_index,
            DecisionContext {
                street,
                actor_position: position,
                players_at_table: 6,
                players_active: 4,
                pot_bb: 8.5,
                actor_stack_bb: 73.0,
                effective_stack_bb: 61.0,
                actor_street_commitment_bb: 1.5,
                current_bet_bb: 4.0,
                to_call_bb: 2.5,
                min_raise_to_bb,
                bet_or_raise_count_this_street: 2,
            },
            ObservedAction {
                kind: action,
                is_all_in: false,
                amount_added_bb: 2.5,
                street_total_after_bb: 4.0,
                decision_time_ms,
            },
        )
        .unwrap()
    }

    #[test]
    fn copies_observed_event_fields_without_imputation() {
        let mut sequence = BehaviourSequence::new("hand-1");
        sequence
            .push(event(
                "hand-1",
                0,
                Position::Cutoff,
                ActionKind::Call,
                Street::Preflop,
                None,
                Some(18_400),
            ))
            .unwrap();

        let encoded = BehaviourFeaturizer::new()
            .featurize_sequence(&sequence, Position::Cutoff)
            .unwrap()
            .unwrap();
        let cat = encoded.categorical()[0];
        let values = encoded.continuous()[0];
        let present = encoded.continuous_present()[0];

        assert_eq!(cat[CategoricalFeature::Street as usize], 0);
        assert_eq!(cat[CategoricalFeature::ActorPosition as usize], 2);
        assert_eq!(cat[CategoricalFeature::Action as usize], 2);
        assert_eq!(cat[CategoricalFeature::IsSubject as usize], 1);
        assert_eq!(cat[CategoricalFeature::IsAllIn as usize], 0);

        assert_eq!(values[ContinuousFeature::PotBb as usize], 8.5);
        assert_eq!(values[ContinuousFeature::ActorStackBb as usize], 73.0);
        assert_eq!(values[ContinuousFeature::EffectiveStackBb as usize], 61.0);
        assert_eq!(values[ContinuousFeature::AmountAddedBb as usize], 2.5);
        assert_eq!(values[ContinuousFeature::StreetTotalAfterBb as usize], 4.0);
        assert_eq!(values[ContinuousFeature::DecisionTimeMs as usize], 18_400.0);
        assert_eq!(encoded.decision_time_ms(), &[18_400]);

        assert_eq!(present[ContinuousFeature::MinRaiseToBb as usize], 0);
        assert_eq!(values[ContinuousFeature::MinRaiseToBb as usize], 0.0);
        assert_eq!(present[ContinuousFeature::DecisionTimeMs as usize], 1);
    }

    #[test]
    fn drops_entire_hand_when_any_decision_time_is_missing() {
        let mut sequence = BehaviourSequence::new("hand-missing-timing");
        sequence
            .push(event(
                "hand-missing-timing",
                0,
                Position::Hijack,
                ActionKind::Raise,
                Street::Preflop,
                Some(5.0),
                Some(900),
            ))
            .unwrap();
        sequence
            .push(event(
                "hand-missing-timing",
                1,
                Position::Button,
                ActionKind::Call,
                Street::Preflop,
                Some(5.0),
                None,
            ))
            .unwrap();

        assert_eq!(
            BehaviourFeaturizer::new()
                .featurize_sequence(&sequence, Position::Button)
                .unwrap(),
            None
        );
    }

    #[test]
    fn retains_exact_integer_timing_beyond_f32_precision() {
        let exact = 16_777_217u32;
        let mut sequence = BehaviourSequence::new("hand-exact-timing");
        sequence
            .push(event(
                "hand-exact-timing",
                0,
                Position::Button,
                ActionKind::Call,
                Street::Preflop,
                Some(2.0),
                Some(exact),
            ))
            .unwrap();

        let encoded = BehaviourFeaturizer::new()
            .featurize_sequence(&sequence, Position::Button)
            .unwrap()
            .unwrap();

        assert_eq!(encoded.decision_time_ms(), &[exact]);
    }

    #[test]
    fn present_optional_values_keep_their_exact_observation() {
        let mut sequence = BehaviourSequence::new("hand-2");
        sequence
            .push(event(
                "hand-2",
                0,
                Position::Button,
                ActionKind::Raise,
                Street::Preflop,
                Some(9.0),
                Some(18_400),
            ))
            .unwrap();

        let encoded = BehaviourFeaturizer::new()
            .featurize_sequence(&sequence, Position::Button)
            .unwrap()
            .unwrap();
        let values = encoded.continuous()[0];
        let present = encoded.continuous_present()[0];

        assert_eq!(present[ContinuousFeature::MinRaiseToBb as usize], 1);
        assert_eq!(values[ContinuousFeature::MinRaiseToBb as usize], 9.0);
        assert_eq!(present[ContinuousFeature::DecisionTimeMs as usize], 1);
        assert_eq!(values[ContinuousFeature::DecisionTimeMs as usize], 18_400.0);
    }

    #[test]
    fn subject_marker_uses_subject_position_and_preserves_other_players() {
        let mut sequence = BehaviourSequence::new("hand-3");
        sequence
            .push(event(
                "hand-3",
                0,
                Position::Hijack,
                ActionKind::Raise,
                Street::Preflop,
                Some(5.0),
                Some(900),
            ))
            .unwrap();
        sequence
            .push(event(
                "hand-3",
                1,
                Position::Button,
                ActionKind::Call,
                Street::Preflop,
                Some(5.0),
                Some(2_100),
            ))
            .unwrap();

        let encoded = BehaviourFeaturizer::new()
            .featurize_sequence(&sequence, Position::Button)
            .unwrap()
            .unwrap();

        assert_eq!(encoded.sequence_len(), 2);
        assert_eq!(
            encoded.categorical()[0][CategoricalFeature::IsSubject as usize],
            0
        );
        assert_eq!(
            encoded.categorical()[1][CategoricalFeature::IsSubject as usize],
            1
        );
        assert_eq!(encoded.sequence_indexes(), &[0, 1]);
    }

    #[test]
    fn encodes_street_without_card_inputs() {
        let mut sequence = BehaviourSequence::new("hand-flop");
        sequence
            .push(event(
                "hand-flop",
                0,
                Position::BigBlind,
                ActionKind::Check,
                Street::Flop,
                Some(2.0),
                Some(320),
            ))
            .unwrap();
        let encoded = BehaviourFeaturizer::new()
            .featurize_sequence(&sequence, Position::BigBlind)
            .unwrap()
            .unwrap();
        assert_eq!(
            encoded.categorical()[0][CategoricalFeature::Street as usize],
            1
        );
        assert_eq!(
            encoded.continuous()[0][ContinuousFeature::DecisionTimeMs as usize],
            320.0
        );
    }

    #[test]
    fn history_keeps_complete_hands_and_counts_dropped_timing_hands() {
        let mut complete = BehaviourSequence::new("hand-a");
        complete
            .push(event(
                "hand-a",
                0,
                Position::UnderTheGun,
                ActionKind::Fold,
                Street::Preflop,
                Some(2.0),
                Some(700),
            ))
            .unwrap();

        let mut incomplete = BehaviourSequence::new("hand-b");
        incomplete
            .push(event(
                "hand-b",
                0,
                Position::Button,
                ActionKind::Call,
                Street::Preflop,
                Some(3.0),
                Some(1_000),
            ))
            .unwrap();
        incomplete
            .push(event(
                "hand-b",
                1,
                Position::BigBlind,
                ActionKind::Check,
                Street::Preflop,
                Some(3.0),
                None,
            ))
            .unwrap();

        let mut history = BehaviourHistory::new();
        history.push(SubjectHand::new(complete, Position::UnderTheGun));
        history.push(SubjectHand::new(incomplete, Position::Button));

        let encoded = BehaviourFeaturizer::new()
            .featurize_history(&history)
            .unwrap();

        assert_eq!(encoded.len(), 1);
        assert_eq!(encoded.total_events(), 1);
        assert_eq!(encoded.dropped_hands_missing_timing(), 1);
        assert_eq!(encoded.hands()[0].hand_id().as_str(), "hand-a");
        assert_eq!(encoded.hands()[0].subject_position(), Position::UnderTheGun);
    }

    #[test]
    fn history_preserves_boundaries_between_complete_hands() {
        let mut first = BehaviourSequence::new("hand-a");
        first
            .push(event(
                "hand-a",
                0,
                Position::UnderTheGun,
                ActionKind::Fold,
                Street::Preflop,
                Some(2.0),
                Some(700),
            ))
            .unwrap();

        let mut second = BehaviourSequence::new("hand-b");
        second
            .push(event(
                "hand-b",
                0,
                Position::Button,
                ActionKind::Call,
                Street::Preflop,
                Some(3.0),
                Some(1_000),
            ))
            .unwrap();
        second
            .push(event(
                "hand-b",
                1,
                Position::BigBlind,
                ActionKind::Check,
                Street::Preflop,
                Some(3.0),
                Some(400),
            ))
            .unwrap();

        let mut history = BehaviourHistory::new();
        history.push(SubjectHand::new(first, Position::UnderTheGun));
        history.push(SubjectHand::new(second, Position::Button));

        let encoded = BehaviourFeaturizer::new()
            .featurize_history(&history)
            .unwrap();

        assert_eq!(encoded.len(), 2);
        assert_eq!(encoded.total_events(), 3);
        assert_eq!(encoded.dropped_hands_missing_timing(), 0);
        assert_eq!(encoded.hands()[0].sequence_len(), 1);
        assert_eq!(encoded.hands()[1].sequence_len(), 2);
    }

    #[test]
    fn categorical_codebook_is_stable() {
        assert_eq!(street_id(Street::Preflop), 0);
        assert_eq!(street_id(Street::River), 3);
        assert_eq!(position_id(Position::UnderTheGun), 0);
        assert_eq!(position_id(Position::BigBlind), 6);
        assert_eq!(action_id(ActionKind::Fold), 0);
        assert_eq!(action_id(ActionKind::Raise), 4);
    }
}
