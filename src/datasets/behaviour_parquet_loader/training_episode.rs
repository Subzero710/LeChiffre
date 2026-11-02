use std::collections::HashSet;
use std::fmt;

use crate::behavioural::codebook::action_from_id;
use crate::behavioural::{
    ActionKind, CategoricalFeature, ContinuousFeature, EVENT_CATEGORICAL_WIDTH,
    EVENT_CONTINUOUS_WIDTH, FeaturizedHand, HandId, MAX_BOARD_CARDS, Position,
};

use super::sampling::SupportQuerySample;

/// Categorical width exposed for the current query decision context.
///
/// Response-dependent fields are deliberately absent. Every query target is a
/// subject decision, so `IsSubject` would be constant and is not duplicated.
pub const QUERY_CONTEXT_CATEGORICAL_WIDTH: usize = 2;

/// Continuous width exposed for the current query decision context.
///
/// These are exactly the pre-action continuous fields from `FeaturizedHand`.
/// Response fields (`AmountAddedBb`, `StreetTotalAfterBb`, `DecisionTimeMs`)
/// are intentionally excluded from the input contract.
pub const QUERY_CONTEXT_CONTINUOUS_WIDTH: usize = 10;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub enum QueryContextCategoricalFeature {
    Street = 0,
    ActorPosition = 1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub enum QueryContextContinuousFeature {
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
}

/// Public information available immediately before one held-out subject action.
///
/// This type cannot represent the current action, all-in flag, sizing response,
/// or decision time. Those values live only in [`BehaviourQueryTarget`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BehaviourQueryContext {
    categorical: [i64; QUERY_CONTEXT_CATEGORICAL_WIDTH],
    continuous: [f32; QUERY_CONTEXT_CONTINUOUS_WIDTH],
    continuous_present: [u8; QUERY_CONTEXT_CONTINUOUS_WIDTH],
    board_rank: [i64; MAX_BOARD_CARDS],
    board_suit: [i64; MAX_BOARD_CARDS],
    board_present: [u8; MAX_BOARD_CARDS],
}

impl BehaviourQueryContext {
    pub fn categorical(&self) -> &[i64; QUERY_CONTEXT_CATEGORICAL_WIDTH] {
        &self.categorical
    }

    pub fn continuous(&self) -> &[f32; QUERY_CONTEXT_CONTINUOUS_WIDTH] {
        &self.continuous
    }

    pub fn continuous_present(&self) -> &[u8; QUERY_CONTEXT_CONTINUOUS_WIDTH] {
        &self.continuous_present
    }

    pub fn board_rank(&self) -> &[i64; MAX_BOARD_CARDS] {
        &self.board_rank
    }

    pub fn board_suit(&self) -> &[i64; MAX_BOARD_CARDS] {
        &self.board_suit
    }

    pub fn board_present(&self) -> &[u8; MAX_BOARD_CARDS] {
        &self.board_present
    }
}

/// Exact observed response used as supervision for one query decision.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BehaviourQueryTarget {
    action: ActionKind,
    is_all_in: bool,
    amount_added_bb: f32,
    street_total_after_bb: f32,
    decision_time_ms: u32,
}

impl BehaviourQueryTarget {
    pub fn action(&self) -> ActionKind {
        self.action
    }

    pub fn is_all_in(&self) -> bool {
        self.is_all_in
    }

    pub fn amount_added_bb(&self) -> f32 {
        self.amount_added_bb
    }

    pub fn street_total_after_bb(&self) -> f32 {
        self.street_total_after_bb
    }

    pub fn decision_time_ms(&self) -> u32 {
        self.decision_time_ms
    }
}

#[derive(Debug, Clone, PartialEq)]
struct QueryDecisionIndex {
    hand_slot: usize,
    row: usize,
    action: ActionKind,
    is_all_in: bool,
}

/// One subject decision from a held-out query hand.
///
/// The full current row is intentionally not exposed. The caller can inspect
/// complete rows only for events strictly before the target decision, then gets
/// a response-free current context plus a separate target.
#[derive(Debug, Clone, Copy)]
pub struct BehaviourQueryDecision<'a> {
    hand: &'a FeaturizedHand,
    index: &'a QueryDecisionIndex,
}

impl<'a> BehaviourQueryDecision<'a> {
    pub fn hand_id(&self) -> &HandId {
        self.hand.hand_id()
    }

    pub fn subject_position(&self) -> Position {
        self.hand.subject_position()
    }

    pub fn target_sequence_index(&self) -> u32 {
        self.hand.sequence_indexes()[self.index.row]
    }

    pub fn prefix_len(&self) -> usize {
        self.index.row
    }

    pub fn prefix_sequence_indexes(&self) -> &[u32] {
        &self.hand.sequence_indexes()[..self.index.row]
    }

    pub fn prefix_categorical(&self) -> &[[i64; EVENT_CATEGORICAL_WIDTH]] {
        &self.hand.categorical()[..self.index.row]
    }

    pub fn prefix_continuous(&self) -> &[[f32; EVENT_CONTINUOUS_WIDTH]] {
        &self.hand.continuous()[..self.index.row]
    }

    pub fn prefix_continuous_present(&self) -> &[[u8; EVENT_CONTINUOUS_WIDTH]] {
        &self.hand.continuous_present()[..self.index.row]
    }

    pub fn prefix_decision_time_ms(&self) -> &[u32] {
        &self.hand.decision_time_ms()[..self.index.row]
    }

    pub fn prefix_board_rank(&self) -> &[[i64; MAX_BOARD_CARDS]] {
        &self.hand.board_rank()[..self.index.row]
    }

    pub fn prefix_board_suit(&self) -> &[[i64; MAX_BOARD_CARDS]] {
        &self.hand.board_suit()[..self.index.row]
    }

    pub fn prefix_board_present(&self) -> &[[u8; MAX_BOARD_CARDS]] {
        &self.hand.board_present()[..self.index.row]
    }

    pub fn context(&self) -> BehaviourQueryContext {
        let categorical = self.hand.categorical()[self.index.row];
        let continuous = self.hand.continuous()[self.index.row];
        let continuous_present = self.hand.continuous_present()[self.index.row];

        BehaviourQueryContext {
            categorical: [
                categorical[CategoricalFeature::Street as usize],
                categorical[CategoricalFeature::ActorPosition as usize],
            ],
            continuous: [
                continuous[ContinuousFeature::PlayersAtTable as usize],
                continuous[ContinuousFeature::PlayersActive as usize],
                continuous[ContinuousFeature::PotBb as usize],
                continuous[ContinuousFeature::ActorStackBb as usize],
                continuous[ContinuousFeature::EffectiveStackBb as usize],
                continuous[ContinuousFeature::ActorStreetCommitmentBb as usize],
                continuous[ContinuousFeature::CurrentBetBb as usize],
                continuous[ContinuousFeature::ToCallBb as usize],
                continuous[ContinuousFeature::MinRaiseToBb as usize],
                continuous[ContinuousFeature::BetOrRaiseCountThisStreet as usize],
            ],
            continuous_present: [
                continuous_present[ContinuousFeature::PlayersAtTable as usize],
                continuous_present[ContinuousFeature::PlayersActive as usize],
                continuous_present[ContinuousFeature::PotBb as usize],
                continuous_present[ContinuousFeature::ActorStackBb as usize],
                continuous_present[ContinuousFeature::EffectiveStackBb as usize],
                continuous_present[ContinuousFeature::ActorStreetCommitmentBb as usize],
                continuous_present[ContinuousFeature::CurrentBetBb as usize],
                continuous_present[ContinuousFeature::ToCallBb as usize],
                continuous_present[ContinuousFeature::MinRaiseToBb as usize],
                continuous_present[ContinuousFeature::BetOrRaiseCountThisStreet as usize],
            ],
            board_rank: self.hand.board_rank()[self.index.row],
            board_suit: self.hand.board_suit()[self.index.row],
            board_present: self.hand.board_present()[self.index.row],
        }
    }

    pub fn target(&self) -> BehaviourQueryTarget {
        let continuous = self.hand.continuous()[self.index.row];
        BehaviourQueryTarget {
            action: self.index.action,
            is_all_in: self.index.is_all_in,
            amount_added_bb: continuous[ContinuousFeature::AmountAddedBb as usize],
            street_total_after_bb: continuous[ContinuousFeature::StreetTotalAfterBb as usize],
            decision_time_ms: self.hand.decision_time_ms()[self.index.row],
        }
    }
}

/// Architecture-agnostic training episode for the behavioural Analyzer.
///
/// Support hands remain complete because they are evidence used to infer the
/// profile. Query hands stay private and are exposed only through
/// [`BehaviourQueryDecision`] views that cannot reveal the current response.
#[derive(Debug, Clone, PartialEq)]
pub struct BehaviourTrainingEpisode {
    subject_id: u64,
    support: Vec<FeaturizedHand>,
    query_hands: Vec<FeaturizedHand>,
    decisions: Vec<QueryDecisionIndex>,
}

impl BehaviourTrainingEpisode {
    fn build(sample: SupportQuerySample) -> Result<Self, BehaviourTrainingEpisodeError> {
        validate_disjoint_hands(&sample.support, &sample.query)?;

        for hand in sample.support.iter().chain(&sample.query) {
            validate_hand_shape(hand)?;
        }

        let mut decisions = Vec::new();
        for (hand_slot, hand) in sample.query.iter().enumerate() {
            for row in 0..hand.sequence_len() {
                let categorical = hand.categorical()[row];
                let sequence_index = hand.sequence_indexes()[row];
                let subject_marker = categorical[CategoricalFeature::IsSubject as usize];
                match subject_marker {
                    0 => continue,
                    1 => {}
                    value => {
                        return Err(BehaviourTrainingEpisodeError::InvalidBooleanCode {
                            hand_id: hand.hand_id().clone(),
                            sequence_index,
                            feature: "is_subject",
                            value,
                        });
                    }
                }

                let action_code = categorical[CategoricalFeature::Action as usize];
                let action = u8::try_from(action_code)
                    .ok()
                    .and_then(action_from_id)
                    .ok_or_else(|| BehaviourTrainingEpisodeError::InvalidActionCode {
                        hand_id: hand.hand_id().clone(),
                        sequence_index,
                        value: action_code,
                    })?;

                let is_all_in_code = categorical[CategoricalFeature::IsAllIn as usize];
                let is_all_in = match is_all_in_code {
                    0 => false,
                    1 => true,
                    value => {
                        return Err(BehaviourTrainingEpisodeError::InvalidBooleanCode {
                            hand_id: hand.hand_id().clone(),
                            sequence_index,
                            feature: "is_all_in",
                            value,
                        });
                    }
                };

                decisions.push(QueryDecisionIndex {
                    hand_slot,
                    row,
                    action,
                    is_all_in,
                });
            }
        }

        Ok(Self {
            subject_id: sample.subject_id,
            support: sample.support,
            query_hands: sample.query,
            decisions,
        })
    }

    pub fn subject_id(&self) -> u64 {
        self.subject_id
    }

    pub fn support(&self) -> &[FeaturizedHand] {
        &self.support
    }

    pub fn query_hand_count(&self) -> usize {
        self.query_hands.len()
    }

    pub fn query_decision_count(&self) -> usize {
        self.decisions.len()
    }

    pub fn query_decisions(
        &self,
    ) -> impl ExactSizeIterator<Item = BehaviourQueryDecision<'_>> + '_ {
        self.decisions.iter().map(|index| BehaviourQueryDecision {
            hand: &self.query_hands[index.hand_slot],
            index,
        })
    }
}

impl SupportQuerySample {
    /// Consumes the sampled hands and creates leakage-safe held-out decisions.
    pub fn into_training_episode(
        self,
    ) -> Result<BehaviourTrainingEpisode, BehaviourTrainingEpisodeError> {
        BehaviourTrainingEpisode::build(self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BehaviourTrainingEpisodeError {
    DuplicateSupportHand {
        hand_id: HandId,
    },
    DuplicateQueryHand {
        hand_id: HandId,
    },
    SupportQueryHandOverlap {
        hand_id: HandId,
    },
    InconsistentHandShape {
        hand_id: HandId,
        field: &'static str,
        expected: usize,
        observed: usize,
    },
    InvalidActionCode {
        hand_id: HandId,
        sequence_index: u32,
        value: i64,
    },
    InvalidBooleanCode {
        hand_id: HandId,
        sequence_index: u32,
        feature: &'static str,
        value: i64,
    },
}

impl fmt::Display for BehaviourTrainingEpisodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateSupportHand { hand_id } => {
                write!(f, "support contains duplicate hand {hand_id}")
            }
            Self::DuplicateQueryHand { hand_id } => {
                write!(f, "query contains duplicate hand {hand_id}")
            }
            Self::SupportQueryHandOverlap { hand_id } => {
                write!(f, "hand {hand_id} appears in both support and query")
            }
            Self::InconsistentHandShape {
                hand_id,
                field,
                expected,
                observed,
            } => write!(
                f,
                "hand {hand_id} has {observed} {field} rows; expected {expected}"
            ),
            Self::InvalidActionCode {
                hand_id,
                sequence_index,
                value,
            } => write!(
                f,
                "hand {hand_id} event {sequence_index} has invalid action code {value}"
            ),
            Self::InvalidBooleanCode {
                hand_id,
                sequence_index,
                feature,
                value,
            } => write!(
                f,
                "hand {hand_id} event {sequence_index} has invalid {feature} code {value}"
            ),
        }
    }
}

impl std::error::Error for BehaviourTrainingEpisodeError {}

fn validate_disjoint_hands(
    support: &[FeaturizedHand],
    query: &[FeaturizedHand],
) -> Result<(), BehaviourTrainingEpisodeError> {
    let mut support_ids = HashSet::with_capacity(support.len());
    for hand in support {
        if !support_ids.insert(hand.hand_id().clone()) {
            return Err(BehaviourTrainingEpisodeError::DuplicateSupportHand {
                hand_id: hand.hand_id().clone(),
            });
        }
    }

    let mut query_ids = HashSet::with_capacity(query.len());
    for hand in query {
        if support_ids.contains(hand.hand_id()) {
            return Err(BehaviourTrainingEpisodeError::SupportQueryHandOverlap {
                hand_id: hand.hand_id().clone(),
            });
        }
        if !query_ids.insert(hand.hand_id().clone()) {
            return Err(BehaviourTrainingEpisodeError::DuplicateQueryHand {
                hand_id: hand.hand_id().clone(),
            });
        }
    }

    Ok(())
}

fn validate_hand_shape(hand: &FeaturizedHand) -> Result<(), BehaviourTrainingEpisodeError> {
    let expected = hand.sequence_len();
    for (field, observed) in [
        ("sequence_indexes", hand.sequence_indexes().len()),
        ("continuous", hand.continuous().len()),
        ("continuous_present", hand.continuous_present().len()),
        ("decision_time_ms", hand.decision_time_ms().len()),
        ("board_rank", hand.board_rank().len()),
        ("board_suit", hand.board_suit().len()),
        ("board_present", hand.board_present().len()),
    ] {
        if observed != expected {
            return Err(BehaviourTrainingEpisodeError::InconsistentHandShape {
                hand_id: hand.hand_id().clone(),
                field,
                expected,
                observed,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::behavioural::{FeaturizedHandData, HandId, Position};

    use super::*;

    fn hand(
        hand_id: &str,
        subject_position: Position,
        categorical: Vec<[i64; EVENT_CATEGORICAL_WIDTH]>,
        continuous: Vec<[f32; EVENT_CONTINUOUS_WIDTH]>,
        exact_timing: Vec<u32>,
    ) -> FeaturizedHand {
        let len = categorical.len();
        FeaturizedHand::from_data(FeaturizedHandData {
            hand_id: HandId::new(hand_id),
            subject_position,
            sequence_indexes: (0..u32::try_from(len).unwrap()).collect(),
            categorical,
            continuous,
            continuous_present: vec![[1; EVENT_CONTINUOUS_WIDTH]; len],
            decision_time_ms: exact_timing,
            board_rank: vec![[0; MAX_BOARD_CARDS]; len],
            board_suit: vec![[0; MAX_BOARD_CARDS]; len],
            board_present: vec![[0; MAX_BOARD_CARDS]; len],
        })
    }

    fn row(
        street: i64,
        actor_position: i64,
        action: i64,
        is_subject: i64,
        is_all_in: i64,
    ) -> [i64; EVENT_CATEGORICAL_WIDTH] {
        [street, actor_position, action, is_subject, is_all_in]
    }

    fn continuous(
        pot: f32,
        to_call: f32,
        amount_added: f32,
        street_total: f32,
        timing_f32: f32,
    ) -> [f32; EVENT_CONTINUOUS_WIDTH] {
        let mut values = [0.0; EVENT_CONTINUOUS_WIDTH];
        values[ContinuousFeature::PlayersAtTable as usize] = 3.0;
        values[ContinuousFeature::PlayersActive as usize] = 3.0;
        values[ContinuousFeature::PotBb as usize] = pot;
        values[ContinuousFeature::ActorStackBb as usize] = 90.0;
        values[ContinuousFeature::EffectiveStackBb as usize] = 75.0;
        values[ContinuousFeature::ActorStreetCommitmentBb as usize] = 2.0;
        values[ContinuousFeature::CurrentBetBb as usize] = 4.0;
        values[ContinuousFeature::ToCallBb as usize] = to_call;
        values[ContinuousFeature::MinRaiseToBb as usize] = 9.0;
        values[ContinuousFeature::BetOrRaiseCountThisStreet as usize] = 1.0;
        values[ContinuousFeature::AmountAddedBb as usize] = amount_added;
        values[ContinuousFeature::StreetTotalAfterBb as usize] = street_total;
        values[ContinuousFeature::DecisionTimeMs as usize] = timing_f32;
        values
    }

    fn sample() -> SupportQuerySample {
        let support = hand(
            "support",
            Position::Button,
            vec![row(0, 3, 3, 1, 0)],
            vec![continuous(3.0, 0.0, 2.5, 2.5, 700.0)],
            vec![700],
        );
        let query = hand(
            "query",
            Position::Button,
            vec![
                row(0, 0, 3, 0, 0),
                row(0, 3, 2, 1, 0),
                row(1, 5, 1, 0, 0),
                row(1, 3, 4, 1, 1),
            ],
            vec![
                continuous(1.5, 0.0, 2.5, 2.5, 900.0),
                continuous(4.0, 2.5, 2.5, 2.5, 1_830.0),
                continuous(8.0, 0.0, 0.0, 0.0, 600.0),
                continuous(8.0, 0.0, 24.0, 24.0, 16_777_216.0),
            ],
            vec![900, 1_830, 600, 16_777_217],
        );

        SupportQuerySample {
            subject_id: 42,
            support: vec![support],
            query: vec![query],
        }
    }

    #[test]
    fn builds_one_target_per_subject_decision_only() {
        let episode = sample().into_training_episode().unwrap();
        assert_eq!(episode.subject_id(), 42);
        assert_eq!(episode.support().len(), 1);
        assert_eq!(episode.query_hand_count(), 1);
        assert_eq!(episode.query_decision_count(), 2);

        let sequence_indexes = episode
            .query_decisions()
            .map(|decision| decision.target_sequence_index())
            .collect::<Vec<_>>();
        assert_eq!(sequence_indexes, vec![1, 3]);
    }

    #[test]
    fn current_context_excludes_response_and_prefix_stops_before_target() {
        let episode = sample().into_training_episode().unwrap();
        let first = episode.query_decisions().next().unwrap();

        assert_eq!(first.prefix_len(), 1);
        assert_eq!(first.prefix_sequence_indexes(), &[0]);
        assert_eq!(
            first.prefix_categorical()[0][CategoricalFeature::Action as usize],
            3
        );
        assert_eq!(
            first.prefix_continuous()[0][ContinuousFeature::AmountAddedBb as usize],
            2.5
        );

        let context = first.context();
        assert_eq!(
            context.categorical()[QueryContextCategoricalFeature::Street as usize],
            0
        );
        assert_eq!(
            context.categorical()[QueryContextCategoricalFeature::ActorPosition as usize],
            3
        );
        assert_eq!(
            context.continuous()[QueryContextContinuousFeature::PotBb as usize],
            4.0
        );
        assert_eq!(
            context.continuous()[QueryContextContinuousFeature::ToCallBb as usize],
            2.5
        );

        let target = first.target();
        assert_eq!(target.action(), ActionKind::Call);
        assert!(!target.is_all_in());
        assert_eq!(target.amount_added_bb(), 2.5);
        assert_eq!(target.street_total_after_bb(), 2.5);
        assert_eq!(target.decision_time_ms(), 1_830);
    }

    #[test]
    fn exact_timing_target_uses_u32_side_channel() {
        let episode = sample().into_training_episode().unwrap();
        let second = episode.query_decisions().nth(1).unwrap();
        let target = second.target();

        assert_eq!(target.action(), ActionKind::Raise);
        assert!(target.is_all_in());
        assert_eq!(target.decision_time_ms(), 16_777_217);
        assert_eq!(second.prefix_decision_time_ms(), &[900, 1_830, 600]);
    }

    #[test]
    fn query_hand_with_no_subject_action_produces_no_fake_target() {
        let mut sample = sample();
        sample.query = vec![hand(
            "walk",
            Position::BigBlind,
            vec![row(0, 3, 0, 0, 0)],
            vec![continuous(1.5, 0.0, 0.0, 0.0, 500.0)],
            vec![500],
        )];

        let episode = sample.into_training_episode().unwrap();
        assert_eq!(episode.query_hand_count(), 1);
        assert_eq!(episode.query_decision_count(), 0);
    }

    #[test]
    fn rejects_support_query_hand_overlap() {
        let support = hand(
            "same",
            Position::Button,
            vec![row(0, 3, 2, 1, 0)],
            vec![continuous(2.0, 1.0, 1.0, 1.0, 100.0)],
            vec![100],
        );
        let query = support.clone();
        let sample = SupportQuerySample {
            subject_id: 1,
            support: vec![support],
            query: vec![query],
        };

        assert_eq!(
            sample.into_training_episode(),
            Err(BehaviourTrainingEpisodeError::SupportQueryHandOverlap {
                hand_id: HandId::new("same")
            })
        );
    }

    #[test]
    fn rejects_invalid_subject_marker_instead_of_guessing() {
        let sample = SupportQuerySample {
            subject_id: 7,
            support: vec![],
            query: vec![hand(
                "bad",
                Position::Button,
                vec![row(0, 3, 2, 2, 0)],
                vec![continuous(2.0, 1.0, 1.0, 1.0, 100.0)],
                vec![100],
            )],
        };

        assert_eq!(
            sample.into_training_episode(),
            Err(BehaviourTrainingEpisodeError::InvalidBooleanCode {
                hand_id: HandId::new("bad"),
                sequence_index: 0,
                feature: "is_subject",
                value: 2,
            })
        );
    }
}
