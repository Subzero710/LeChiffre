//! Shared, strict training data contracts for the hierarchical Z0/Z1/Z2 engine.
//! No legacy model or optimizer is constructed here.
use std::collections::HashSet;
use std::fmt;

use super::BehaviourParquetDataset;
use crate::behavioural::checkpoint::{dimension, exact_keys, unsigned};
use crate::behavioural::codebook::{action_from_id, position_from_id, position_id, street_from_id};
use crate::behavioural::{
    ActionKind, ActionSizing, CategoricalFeature, ContinuousFeature, FeaturizedHand,
    GeneratorObservation, HandId, LegalActionSpace, ObservedAction, PRE_ACTION_CONTEXT_WIDTH,
};
use serde_json::{Value, json};

pub trait ExactLegalityProvider {
    type Error: fmt::Display;
    fn legal_actions(
        &mut self,
        subject_id: u64,
        hand_id: &HandId,
        sequence_index: u32,
    ) -> Result<LegalActionSpace, Self::Error>;
}
impl<F, E> ExactLegalityProvider for F
where
    F: FnMut(u64, &HandId, u32) -> Result<LegalActionSpace, E>,
    E: fmt::Display,
{
    type Error = E;
    fn legal_actions(
        &mut self,
        subject_id: u64,
        hand_id: &HandId,
        sequence_index: u32,
    ) -> Result<LegalActionSpace, E> {
        self(subject_id, hand_id, sequence_index)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HierarchicalHandSplit {
    pub first_query_hand_index: u64,
    pub first_evaluation_hand_index: u64,
    pub support_hands: usize,
    pub query_hands: usize,
    pub evaluation_hands: usize,
}
impl HierarchicalHandSplit {
    pub fn new(
        first_query_hand_index: u64,
        first_evaluation_hand_index: u64,
        support_hands: usize,
        query_hands: usize,
        evaluation_hands: usize,
    ) -> Result<Self, TrainingDataError> {
        if first_evaluation_hand_index <= first_query_hand_index
            || support_hands == 0
            || query_hands == 0
            || evaluation_hands == 0
        {
            return Err(TrainingDataError::InvalidConfiguration);
        }
        Ok(Self {
            first_query_hand_index,
            first_evaluation_hand_index,
            support_hands,
            query_hands,
            evaluation_hands,
        })
    }
    pub(crate) fn checkpoint_value(self) -> Value {
        json!({"first_query_hand_index":self.first_query_hand_index,
               "first_evaluation_hand_index":self.first_evaluation_hand_index,
               "support_hands":self.support_hands,"query_hands":self.query_hands,"evaluation_hands":self.evaluation_hands})
    }
    pub(crate) fn from_checkpoint(value: &Value) -> Result<Self, String> {
        exact_keys(
            value,
            &[
                "first_query_hand_index",
                "first_evaluation_hand_index",
                "support_hands",
                "query_hands",
                "evaluation_hands",
            ],
        )?;
        Self::new(
            unsigned(&value["first_query_hand_index"])?,
            unsigned(&value["first_evaluation_hand_index"])?,
            dimension(&value["support_hands"])?,
            dimension(&value["query_hands"])?,
            dimension(&value["evaluation_hands"])?,
        )
        .map_err(|e| e.to_string())
    }
}

#[derive(Debug, Clone)]
pub(super) struct SubjectSample {
    pub(super) id: u64,
    pub(super) query: Vec<FeaturizedHand>,
    pub(super) validation: Vec<FeaturizedHand>,
}

pub(super) fn sample_subjects(
    dataset: &BehaviourParquetDataset,
    selected: &[u64],
    split: HierarchicalHandSplit,
) -> Result<Vec<SubjectSample>, TrainingDataError> {
    if selected.is_empty() {
        return Err(TrainingDataError::EmptySubjectSelection);
    }
    let mut seen = HashSet::new();
    let mut result = Vec::new();
    for &id in selected {
        if !seen.insert(id) {
            return Err(TrainingDataError::DuplicateSubject { subject_id: id });
        }
        let memberships = dataset
            .subjects
            .get(&id)
            .ok_or(TrainingDataError::UnknownSubject { subject_id: id })?;
        let mut support = Vec::new();
        let mut query = Vec::new();
        let mut validation = Vec::new();
        for membership in memberships {
            let hand = &dataset.hands[membership.hand_slot];
            let materialized = hand.materialize(membership.subject_position);
            if hand.hand_index < split.first_query_hand_index {
                support.push(materialized);
            } else if hand.hand_index < split.first_evaluation_hand_index {
                query.push(materialized);
            } else {
                validation.push(materialized);
            }
        }
        if support.len() < split.support_hands
            || query.len() < split.query_hands
            || validation.len() < split.evaluation_hands
        {
            return Err(TrainingDataError::InsufficientPartition {
                subject_id: id,
                available_support: support.len(),
                available_query: query.len(),
                available_evaluation: validation.len(),
            });
        }
        let support = support.split_off(support.len() - split.support_hands);
        query.truncate(split.query_hands);
        validation.truncate(split.evaluation_hands);
        for (name, hands) in [
            ("support", &support),
            ("query", &query),
            ("evaluation", &validation),
        ] {
            if !hands
                .iter()
                .any(|hand| subject_decisions(hand).next().is_some())
            {
                return Err(TrainingDataError::NoSubjectDecisions {
                    subject_id: id,
                    partition: name,
                });
            }
        }
        result.push(SubjectSample {
            id,
            query,
            validation,
        });
    }
    Ok(result)
}

/// Deterministic FNV-1a fingerprint over every loaded row and subject membership.
/// This detects accidental data replacement across checkpoint resumes; it is not
/// a cryptographic authentication tag.
pub(super) fn dataset_fingerprint(dataset: &BehaviourParquetDataset) -> String {
    let mut hash = 0x6c62_272e_07bb_0142_62b8_2175_6295_c58d_u128;
    fn write(hash: &mut u128, bytes: &[u8]) {
        const PRIME: u128 = 0x0000_0000_0100_0000_0000_0000_0000_013b;
        for &byte in bytes {
            *hash = (*hash ^ u128::from(byte)).wrapping_mul(PRIME);
        }
    }
    for hand in &dataset.hands {
        write(
            &mut hash,
            &(hand.hand_id.as_str().len() as u64).to_le_bytes(),
        );
        write(&mut hash, hand.hand_id.as_str().as_bytes());
        write(&mut hash, &hand.hand_index.to_le_bytes());
        write(&mut hash, &[hand.players_at_table]);
        for &sequence in &hand.sequence_indexes {
            write(&mut hash, &sequence.to_le_bytes());
        }
        for cat in &hand.categorical_without_subject {
            for value in cat {
                write(&mut hash, &value.to_le_bytes());
            }
        }
        for continuous in &hand.continuous {
            for value in continuous {
                write(&mut hash, &value.to_bits().to_le_bytes());
            }
        }
        for present in &hand.continuous_present {
            write(&mut hash, present);
        }
        for &duration in &hand.decision_time_ms {
            write(&mut hash, &duration.to_le_bytes());
        }
    }
    for (&id, memberships) in &dataset.subjects {
        write(&mut hash, &id.to_le_bytes());
        write(&mut hash, &(memberships.len() as u64).to_le_bytes());
        for membership in memberships {
            write(&mut hash, &(membership.hand_slot as u64).to_le_bytes());
            write(&mut hash, &[position_id(membership.subject_position)]);
        }
    }
    format!("{hash:032x}")
}

pub(super) fn subject_decisions(hand: &FeaturizedHand) -> impl Iterator<Item = usize> + '_ {
    hand.categorical()
        .iter()
        .enumerate()
        .filter(|(_, cat)| cat[CategoricalFeature::IsSubject as usize] == 1)
        .map(|(i, _)| i)
}
pub(super) fn lookup<P: ExactLegalityProvider>(
    provider: &mut P,
    id: u64,
    hand: &FeaturizedHand,
    row: usize,
) -> Result<LegalActionSpace, TrainingDataError> {
    provider
        .legal_actions(id, hand.hand_id(), hand.sequence_indexes()[row])
        .map_err(|e| TrainingDataError::MissingLegality {
            hand_id: hand.hand_id().clone(),
            sequence_index: hand.sequence_indexes()[row],
            message: e.to_string(),
        })
}

pub(super) fn decode_decision(
    hand: &FeaturizedHand,
    row: usize,
) -> Result<GeneratorObservation, TrainingDataError> {
    let categorical = hand.categorical()[row];
    let values = hand.continuous()[row];
    let present = hand.continuous_present()[row];
    let invalid = || TrainingDataError::InvalidData {
        hand_id: hand.hand_id().clone(),
        sequence_index: hand.sequence_indexes()[row],
    };
    let street = u8::try_from(categorical[CategoricalFeature::Street as usize])
        .ok()
        .and_then(street_from_id)
        .ok_or_else(invalid)?;
    let actor_position = u8::try_from(categorical[CategoricalFeature::ActorPosition as usize])
        .ok()
        .and_then(position_from_id)
        .ok_or_else(invalid)?;
    let kind = u8::try_from(categorical[CategoricalFeature::Action as usize])
        .ok()
        .and_then(action_from_id)
        .ok_or_else(invalid)?;
    let is_all_in = match categorical[CategoricalFeature::IsAllIn as usize] {
        0 => false,
        1 => true,
        _ => return Err(invalid()),
    };
    let sizing = if kind == ActionKind::Fold {
        if present[ContinuousFeature::AmountOverPot as usize] != 0
            || present[ContinuousFeature::AmountOverStack as usize] != 0
        {
            return Err(invalid());
        }
        None
    } else {
        if present[ContinuousFeature::AmountOverPot as usize] != 1
            || present[ContinuousFeature::AmountOverStack as usize] != 1
        {
            return Err(invalid());
        }
        let increment = if kind == ActionKind::Raise {
            if present[ContinuousFeature::RaiseIncrementOverPot as usize] != 1 {
                return Err(invalid());
            }
            Some(values[ContinuousFeature::RaiseIncrementOverPot as usize])
        } else {
            if present[ContinuousFeature::RaiseIncrementOverPot as usize] != 0 {
                return Err(invalid());
            }
            None
        };
        Some(ActionSizing {
            amount_over_pot_before: values[ContinuousFeature::AmountOverPot as usize],
            amount_over_stack_before: values[ContinuousFeature::AmountOverStack as usize],
            raise_increment_over_pot_before: increment,
        })
    };
    if kind == ActionKind::Fold && present[ContinuousFeature::RaiseIncrementOverPot as usize] != 0 {
        return Err(invalid());
    }
    let actor_streak = values[ContinuousFeature::ActorAggressionStreakAfter as usize];
    let table_streak = values[ContinuousFeature::TableAggressionStreakAfter as usize];
    if ![actor_streak, table_streak]
        .iter()
        .all(|v| v.is_finite() && *v >= 0.0 && v.fract() == 0.0 && *v <= u32::MAX as f32)
    {
        return Err(invalid());
    }
    let context_values: [f32; PRE_ACTION_CONTEXT_WIDTH] = values[..PRE_ACTION_CONTEXT_WIDTH]
        .try_into()
        .map_err(|_| invalid())?;
    let context_present: [u8; PRE_ACTION_CONTEXT_WIDTH] = present[..PRE_ACTION_CONTEXT_WIDTH]
        .try_into()
        .map_err(|_| invalid())?;
    Ok(GeneratorObservation {
        street,
        actor_position,
        context_values,
        context_present,
        observed: ObservedAction {
            kind,
            is_all_in,
            sizing,
            actor_aggression_streak_after: actor_streak as u32,
            table_aggression_streak_after: table_streak as u32,
            decision_time_ms: Some(hand.decision_time_ms()[row]),
        },
    })
}

#[derive(Debug)]
pub enum TrainingDataError {
    InvalidConfiguration,
    NonIncreasingCutoff,
    Baseline(crate::behavioural::BaselineAdaptationError),
    CountOverflow,
    EmptySubjectSelection,
    DuplicateSubject {
        subject_id: u64,
    },
    UnknownSubject {
        subject_id: u64,
    },
    InsufficientPartition {
        subject_id: u64,
        available_support: usize,
        available_query: usize,
        available_evaluation: usize,
    },
    NoSubjectDecisions {
        subject_id: u64,
        partition: &'static str,
    },
    InvalidData {
        hand_id: HandId,
        sequence_index: u32,
    },
    MissingLegality {
        hand_id: HandId,
        sequence_index: u32,
        message: String,
    },
}

impl fmt::Display for TrainingDataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidData {
                hand_id,
                sequence_index,
            } => write!(f, "invalid features for {hand_id}:{sequence_index}"),
            Self::MissingLegality {
                hand_id,
                sequence_index,
                message,
            } => write!(
                f,
                "exact legality missing for {hand_id}:{sequence_index}: {message}"
            ),
            other => write!(f, "{other:?}"),
        }
    }
}
impl std::error::Error for TrainingDataError {}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn enforces_strict_chronological_partitions() {
        assert!(HierarchicalHandSplit::new(5, 5, 1, 1, 1).is_err());
        assert!(HierarchicalHandSplit::new(5, 6, 1, 0, 1).is_err());
        assert!(HierarchicalHandSplit::new(5, 6, 1, 1, 1).is_ok());
    }
}
