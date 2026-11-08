use std::collections::HashSet;
use std::fmt;

use crate::behavioural::codebook::action_from_id;
use crate::behavioural::{
    ActionKind, ActionSizing, CategoricalFeature, ContinuousFeature, EVENT_CATEGORICAL_WIDTH,
    EVENT_CONTINUOUS_WIDTH, FeaturizedHand, HandId, PRE_ACTION_CONTEXT_WIDTH, Position,
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
/// Action sizing and post-action streaks are excluded from the query context.
pub const QUERY_CONTEXT_CONTINUOUS_WIDTH: usize = PRE_ACTION_CONTEXT_WIDTH;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub enum QueryContextCategoricalFeature {
    Street = 0,
    ActorPosition = 1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub enum QueryContextContinuousFeature {
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
    // Other pre-action dimensions are represented in the full context array.
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
}

/// Exact observed response used as supervision for one query decision.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BehaviourQueryTarget {
    action: ActionKind,
    is_all_in: bool,
    sizing: Option<ActionSizing>,
    actor_aggression_streak_after: u32,
    table_aggression_streak_after: u32,
    decision_time_ms: u32,
}

impl BehaviourQueryTarget {
    pub fn action(&self) -> ActionKind {
        self.action
    }
    pub fn is_all_in(&self) -> bool {
        self.is_all_in
    }
    pub fn sizing(&self) -> Option<ActionSizing> {
        self.sizing
    }
    pub fn actor_aggression_streak_after(&self) -> u32 {
        self.actor_aggression_streak_after
    }
    pub fn table_aggression_streak_after(&self) -> u32 {
        self.table_aggression_streak_after
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

    pub fn context(&self) -> BehaviourQueryContext {
        let categorical = self.hand.categorical()[self.index.row];
        let continuous = self.hand.continuous()[self.index.row];
        let continuous_present = self.hand.continuous_present()[self.index.row];

        BehaviourQueryContext {
            categorical: [
                categorical[CategoricalFeature::Street as usize],
                categorical[CategoricalFeature::ActorPosition as usize],
            ],
            continuous: continuous[..PRE_ACTION_CONTEXT_WIDTH]
                .try_into()
                .expect("pre-action tensor has fixed width"),
            continuous_present: continuous_present[..PRE_ACTION_CONTEXT_WIDTH]
                .try_into()
                .expect("pre-action presence tensor has fixed width"),
        }
    }

    pub fn target(&self) -> BehaviourQueryTarget {
        let values = self.hand.continuous()[self.index.row];
        let presence = self.hand.continuous_present()[self.index.row];
        let sizing = if self.index.action == ActionKind::Fold {
            None
        } else {
            Some(ActionSizing {
                amount_over_pot_before: values[ContinuousFeature::AmountOverPot as usize],
                amount_over_stack_before: values[ContinuousFeature::AmountOverStack as usize],
                raise_increment_over_pot_before: if presence
                    [ContinuousFeature::RaiseIncrementOverPot as usize]
                    == 1
                {
                    Some(values[ContinuousFeature::RaiseIncrementOverPot as usize])
                } else {
                    None
                },
            })
        };
        BehaviourQueryTarget {
            action: self.index.action,
            is_all_in: self.index.is_all_in,
            sizing,
            actor_aggression_streak_after: values
                [ContinuousFeature::ActorAggressionStreakAfter as usize]
                as u32,
            table_aggression_streak_after: values
                [ContinuousFeature::TableAggressionStreakAfter as usize]
                as u32,
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

                let mask = hand.continuous_present()[row];
                let amount_pot = ContinuousFeature::AmountOverPot as usize;
                let amount_stack = ContinuousFeature::AmountOverStack as usize;
                let increment = ContinuousFeature::RaiseIncrementOverPot as usize;
                let valid_sizing = match action {
                    ActionKind::Fold => {
                        mask[amount_pot] == 0 && mask[amount_stack] == 0 && mask[increment] == 0
                    }
                    ActionKind::Raise => {
                        mask[amount_pot] == 1 && mask[amount_stack] == 1 && mask[increment] == 1
                    }
                    _ => mask[amount_pot] == 1 && mask[amount_stack] == 1 && mask[increment] == 0,
                };
                if !valid_sizing {
                    return Err(BehaviourTrainingEpisodeError::InvalidActionSizing {
                        hand_id: hand.hand_id().clone(),
                        sequence_index,
                    });
                }
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
    InvalidActionSizing {
        hand_id: HandId,
        sequence_index: u32,
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
            Self::InvalidActionSizing {
                hand_id,
                sequence_index,
            } => write!(
                f,
                "hand {hand_id} event {sequence_index} has inconsistent sizing masks"
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
    use super::*;
    use crate::behavioural::{FeaturizedHandData, HandId, Position};

    fn one_hand() -> FeaturizedHand {
        let mut categorical = [0; EVENT_CATEGORICAL_WIDTH];
        categorical[CategoricalFeature::Action as usize] = 3;
        categorical[CategoricalFeature::IsSubject as usize] = 1;
        categorical[CategoricalFeature::ActorPosition as usize] = 3;
        let mut continuous = [0.0; EVENT_CONTINUOUS_WIDTH];
        continuous[ContinuousFeature::PotBb as usize] = 20.0;
        continuous[ContinuousFeature::AmountOverPot as usize] = 0.25;
        continuous[ContinuousFeature::AmountOverStack as usize] = 0.10;
        continuous[ContinuousFeature::DecisionTimeMs as usize] = 700.0;
        let mut presence = [1; EVENT_CONTINUOUS_WIDTH];
        presence[ContinuousFeature::RaiseIncrementOverPot as usize] = 0;
        FeaturizedHand::from_data(FeaturizedHandData {
            hand_id: HandId::new("hand"),
            subject_position: Position::Button,
            sequence_indexes: vec![0],
            categorical: vec![categorical],
            continuous: vec![continuous],
            continuous_present: vec![presence],
            decision_time_ms: vec![700],
        })
    }

    #[test]
    fn context_does_not_include_its_current_response() {
        let episode = SupportQuerySample {
            subject_id: 42,
            support: vec![],
            query: vec![one_hand()],
        }
        .into_training_episode()
        .unwrap();
        let decision = episode.query_decisions().next().unwrap();
        assert_eq!(decision.prefix_len(), 0);
        assert_eq!(
            decision.context().continuous().len(),
            PRE_ACTION_CONTEXT_WIDTH
        );
        assert_eq!(
            decision.context().continuous()[QueryContextContinuousFeature::PotBb as usize],
            20.0
        );
        let target = decision.target();
        assert_eq!(target.action(), ActionKind::Bet);
        assert_eq!(target.sizing().unwrap().amount_over_pot_before, 0.25);
        assert_eq!(target.decision_time_ms(), 700);
    }
    #[test]
    fn support_and_query_must_be_disjoint() {
        let hand = one_hand();
        let result = SupportQuerySample {
            subject_id: 42,
            support: vec![hand.clone()],
            query: vec![hand],
        }
        .into_training_episode();
        assert_eq!(
            result,
            Err(BehaviourTrainingEpisodeError::SupportQueryHandOverlap {
                hand_id: HandId::new("hand"),
            })
        );
    }

    #[test]
    fn invalid_sizing_presence_is_rejected() {
        let source = one_hand();
        let mut presence = source.continuous_present().to_vec();
        presence[0][ContinuousFeature::AmountOverPot as usize] = 0;
        let bad = FeaturizedHand::from_data(FeaturizedHandData {
            hand_id: source.hand_id().clone(),
            subject_position: source.subject_position(),
            sequence_indexes: source.sequence_indexes().to_vec(),
            categorical: source.categorical().to_vec(),
            continuous: source.continuous().to_vec(),
            continuous_present: presence,
            decision_time_ms: source.decision_time_ms().to_vec(),
        });
        let result = SupportQuerySample {
            subject_id: 42,
            support: vec![],
            query: vec![bad],
        }
        .into_training_episode();
        assert_eq!(
            result,
            Err(BehaviourTrainingEpisodeError::InvalidActionSizing {
                hand_id: HandId::new("hand"),
                sequence_index: 0,
            })
        );
    }
}
