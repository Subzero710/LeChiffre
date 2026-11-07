//! Joint generator and per-subject latent training on retained Parquet hands.
//!
//! Each legal action space must come from an exact pre-decision rules-engine
//! replay. Parquet stores only behavioural observations; it cannot determine
//! exact chip increments or whether betting is reopened after a short all-in.

use std::collections::{BTreeMap, HashSet};
use std::fmt;

use crate::behavioural::codebook::{action_from_id, position_from_id, street_from_id};
use crate::behavioural::{
    ActionKind, ActionSizing, Adam, BehaviourLatentModel, BehaviourProfile, CategoricalFeature,
    ContinuousFeature, FeaturizedHand, GeneratorError, GeneratorObservation, HandId,
    LegalActionSpace, NeuralBehaviourGenerator, ObservedAction, PRE_ACTION_CONTEXT_WIDTH,
    TrainingLoss,
};

use super::BehaviourParquetDataset;

/// Return the complete exact action space from the original rules-engine state.
/// No legal mask, blind denomination, or chip increment is inferred from labels.
/// The key is the canonical hand ID and absolute sequence index within the hand.
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
    ) -> Result<LegalActionSpace, Self::Error> {
        self(subject_id, hand_id, sequence_index)
    }
}

/// Split by the dataset's *global* hand index, not per subject. A hand can
/// contain several subjects: splitting independently by subject would leak
/// held-out actions through another subject's support-hand context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratorHandSplit {
    pub first_query_hand_index: u64,
    pub support_hands: usize,
    pub query_hands: usize,
}

impl GeneratorHandSplit {
    pub fn new(
        first_query_hand_index: u64,
        support_hands: usize,
        query_hands: usize,
    ) -> Result<Self, GeneratorTrainingError> {
        if support_hands == 0 || query_hands == 0 {
            return Err(GeneratorTrainingError::InvalidConfiguration);
        }
        Ok(Self {
            first_query_hand_index,
            support_hands,
            query_hands,
        })
    }
}

/// All optimization hyperparameters are explicitly supplied by the caller.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeneratorTrainingConfig {
    pub latent_learning_rate: f64,
    pub latent_l2: f64,
}

impl GeneratorTrainingConfig {
    pub fn new(latent_learning_rate: f64, latent_l2: f64) -> Result<Self, GeneratorTrainingError> {
        if !latent_learning_rate.is_finite()
            || latent_learning_rate <= 0.0
            || !latent_l2.is_finite()
            || latent_l2 < 0.0
        {
            return Err(GeneratorTrainingError::InvalidConfiguration);
        }
        Ok(Self {
            latent_learning_rate,
            latent_l2,
        })
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct GeneratorLossSummary {
    pub decisions: usize,
    pub total_nll: f64,
    pub action_nll: f64,
    pub sizing_nll: f64,
    pub timing_nll: f64,
}

impl GeneratorLossSummary {
    fn record(&mut self, loss: &TrainingLoss) -> Result<(), GeneratorTrainingError> {
        self.decisions = self
            .decisions
            .checked_add(1)
            .ok_or(GeneratorTrainingError::CountOverflow)?;
        self.total_nll += loss.total;
        self.action_nll += loss.action;
        self.sizing_nll += loss.sizing;
        self.timing_nll += loss.timing;
        if ![
            self.total_nll,
            self.action_nll,
            self.sizing_nll,
            self.timing_nll,
        ]
        .iter()
        .all(|v| v.is_finite())
        {
            return Err(GeneratorTrainingError::NonFiniteLoss);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeneratorEpochReport {
    pub subjects: usize,
    /// Gradients use only support-hand decisions by the current subject.
    pub train: GeneratorLossSummary,
    /// Query-hand decisions are evaluated after the epoch, without updates.
    pub held_out: GeneratorLossSummary,
}

/// The single generator is shared by all subjects; Z is learned separately for
/// each subject and is never inserted into an input context as an ID feature.
#[derive(Debug, Clone)]
pub struct GeneratorTrainer {
    model: NeuralBehaviourGenerator,
    adam: Adam,
    profiles: BTreeMap<u64, BehaviourProfile>,
    config: GeneratorTrainingConfig,
}

impl GeneratorTrainer {
    pub fn new(
        model: NeuralBehaviourGenerator,
        adam: Adam,
        config: GeneratorTrainingConfig,
    ) -> Self {
        Self {
            model,
            adam,
            profiles: BTreeMap::new(),
            config,
        }
    }

    pub fn model(&self) -> &NeuralBehaviourGenerator {
        &self.model
    }

    pub fn profile(&self, subject_id: u64) -> Option<&BehaviourProfile> {
        self.profiles.get(&subject_id)
    }

    pub fn trained_subjects(&self) -> usize {
        self.profiles.len()
    }

    pub fn optimizer_steps(&self) -> u64 {
        self.adam.steps()
    }

    /// One epoch over explicitly selected subjects. The split is global by hand,
    /// so no target hand enters any model input in the training partition.
    /// An invalid or missing exact legal space aborts the epoch atomically.
    pub fn train_epoch<P: ExactLegalityProvider>(
        &mut self,
        dataset: &BehaviourParquetDataset,
        selected_subjects: &[u64],
        split: GeneratorHandSplit,
        provider: &mut P,
    ) -> Result<GeneratorEpochReport, GeneratorTrainingError> {
        let mut working = self.clone();
        let mut seen = HashSet::new();
        let mut samples = Vec::new();
        if selected_subjects.is_empty() {
            return Err(GeneratorTrainingError::EmptySubjectSelection);
        }
        for &subject_id in selected_subjects {
            if !seen.insert(subject_id) {
                return Err(GeneratorTrainingError::DuplicateSubject { subject_id });
            }
            let memberships = dataset
                .subjects
                .get(&subject_id)
                .ok_or(GeneratorTrainingError::UnknownSubject { subject_id })?;
            let mut support = Vec::new();
            let mut query = Vec::new();
            for membership in memberships {
                let hand = &dataset.hands[membership.hand_slot];
                if hand.hand_index < split.first_query_hand_index {
                    support.push(hand.materialize(membership.subject_position));
                } else {
                    query.push(hand.materialize(membership.subject_position));
                }
            }
            if support.len() < split.support_hands || query.len() < split.query_hands {
                return Err(GeneratorTrainingError::InsufficientPartition {
                    subject_id,
                    available_support: support.len(),
                    available_query: query.len(),
                });
            }
            let recent = support.split_off(support.len() - split.support_hands);
            query.truncate(split.query_hands);
            if !recent
                .iter()
                .any(|hand| subject_decisions(hand).next().is_some())
            {
                return Err(GeneratorTrainingError::NoSubjectDecisions {
                    subject_id,
                    partition: "support",
                });
            }
            if !query
                .iter()
                .any(|hand| subject_decisions(hand).next().is_some())
            {
                return Err(GeneratorTrainingError::NoSubjectDecisions {
                    subject_id,
                    partition: "query",
                });
            }
            samples.push((subject_id, recent, query));
        }

        let mut train = GeneratorLossSummary::default();
        let mut held_out = GeneratorLossSummary::default();

        for (subject_id, support, _) in &samples {
            let latent = working.model.latent_config();
            let mut profile = working
                .profiles
                .get(subject_id)
                .cloned()
                .unwrap_or_else(|| {
                    latent
                        .profile(vec![0.0; latent.dimension()])
                        .expect("finite zero profile")
                });
            for hand in support {
                for row in subject_decisions(hand) {
                    let obs = decode_decision(hand, row)?;
                    let legal = lookup(provider, *subject_id, hand, row)?;
                    let result =
                        working
                            .adam
                            .train_encoded(&mut working.model, &profile, &obs, &legal)?;
                    train.record(&result)?;
                    let updated = profile
                        .as_slice()
                        .iter()
                        .zip(&result.latent_gradient)
                        .map(|(&z, &gradient)| {
                            (f64::from(z)
                                - working.config.latent_learning_rate
                                    * (f64::from(gradient)
                                        + working.config.latent_l2 * f64::from(z)))
                                as f32
                        })
                        .collect();
                    profile = latent
                        .profile(updated)
                        .map_err(|_| GeneratorTrainingError::NonFiniteProfile)?;
                }
            }
            working.profiles.insert(*subject_id, profile);
        }

        // No optimizer steps and no latent updates in this entire loop.
        for (subject_id, _, query) in &samples {
            let profile = working
                .profiles
                .get(subject_id)
                .ok_or(GeneratorTrainingError::MissingProfile)?;
            for hand in query {
                for row in subject_decisions(hand) {
                    let obs = decode_decision(hand, row)?;
                    let legal = lookup(provider, *subject_id, hand, row)?;
                    let result =
                        working
                            .adam
                            .evaluate_encoded(&working.model, profile, &obs, &legal)?;
                    held_out.record(&result)?;
                }
            }
        }

        let report = GeneratorEpochReport {
            subjects: samples.len(),
            train,
            held_out,
        };
        *self = working;
        Ok(report)
    }
}

fn subject_decisions(hand: &FeaturizedHand) -> impl Iterator<Item = usize> + '_ {
    hand.categorical()
        .iter()
        .enumerate()
        .filter(|(_, row)| row[CategoricalFeature::IsSubject as usize] == 1)
        .map(|(index, _)| index)
}

fn lookup<P: ExactLegalityProvider>(
    provider: &mut P,
    subject_id: u64,
    hand: &FeaturizedHand,
    row: usize,
) -> Result<LegalActionSpace, GeneratorTrainingError> {
    provider
        .legal_actions(subject_id, hand.hand_id(), hand.sequence_indexes()[row])
        .map_err(|error| GeneratorTrainingError::MissingLegality {
            hand_id: hand.hand_id().clone(),
            sequence_index: hand.sequence_indexes()[row],
            message: error.to_string(),
        })
}

fn decode_decision(
    hand: &FeaturizedHand,
    row: usize,
) -> Result<GeneratorObservation, GeneratorTrainingError> {
    let categorical = hand.categorical()[row];
    let values = hand.continuous()[row];
    let present = hand.continuous_present()[row];
    let invalid = || GeneratorTrainingError::InvalidData {
        hand_id: hand.hand_id().clone(),
        sequence_index: hand.sequence_indexes()[row],
    };
    let street = u8::try_from(categorical[CategoricalFeature::Street as usize])
        .ok()
        .and_then(street_from_id)
        .ok_or_else(&invalid)?;
    let actor_position = u8::try_from(categorical[CategoricalFeature::ActorPosition as usize])
        .ok()
        .and_then(position_from_id)
        .ok_or_else(&invalid)?;
    let kind = u8::try_from(categorical[CategoricalFeature::Action as usize])
        .ok()
        .and_then(action_from_id)
        .ok_or_else(&invalid)?;
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
pub enum GeneratorTrainingError {
    InvalidConfiguration,
    CountOverflow,
    NonFiniteLoss,
    NonFiniteProfile,
    MissingProfile,
    NoSubjectDecisions {
        subject_id: u64,
        partition: &'static str,
    },
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
    },
    Generator(GeneratorError),
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

impl fmt::Display for GeneratorTrainingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingLegality {
                hand_id,
                sequence_index,
                message,
            } => write!(
                f,
                "missing or invalid exact legal space for {hand_id}:{sequence_index}: {message}"
            ),
            Self::InvalidData {
                hand_id,
                sequence_index,
            } => write!(
                f,
                "invalid behavioural features for {hand_id}:{sequence_index}"
            ),
            other => write!(f, "{other:?}"),
        }
    }
}
impl std::error::Error for GeneratorTrainingError {}
impl From<GeneratorError> for GeneratorTrainingError {
    fn from(error: GeneratorError) -> Self {
        Self::Generator(error)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::super::{PersistedHand, SubjectMembership};
    use super::*;
    use crate::behavioural::codebook::{action_id, position_id, street_id};
    use crate::behavioural::{
        Adam, BehaviourLatentConfig, ContinuousFeature, EVENT_CONTINUOUS_WIDTH, HandId,
        NeuralBehaviourGenerator, Position, Street,
    };
    use crate::datasets::behaviour_parquet::BehaviourParquetDatasetSummary;

    fn dataset() -> BehaviourParquetDataset {
        let mut hands = Vec::new();
        for index in 0_u64..4 {
            let kind = if index.is_multiple_of(2) {
                ActionKind::Check
            } else {
                ActionKind::Bet
            };
            let mut continuous = [0.0; EVENT_CONTINUOUS_WIDTH];
            continuous[ContinuousFeature::PotBb as usize] = 10.0;
            continuous[ContinuousFeature::StackRemainingBb as usize] = 80.0;
            continuous[ContinuousFeature::CommittedThisHandBb as usize] = 5.0;
            continuous[ContinuousFeature::MinRaiseToBb as usize] = 1.0;
            continuous[ContinuousFeature::PlayersYetToAct as usize] = 1.0;
            continuous[ContinuousFeature::AmountOverPot as usize] =
                if kind == ActionKind::Bet { 0.5 } else { 0.0 };
            continuous[ContinuousFeature::AmountOverStack as usize] =
                if kind == ActionKind::Bet { 0.0625 } else { 0.0 };
            continuous[ContinuousFeature::DecisionTimeMs as usize] = 600.0;
            let mut present = [1; EVENT_CONTINUOUS_WIDTH];
            present[ContinuousFeature::RaiseIncrementOverPot as usize] = 0;
            hands.push(PersistedHand {
                hand_id: HandId::new(format!("h{index}")),
                hand_index: index,
                players_at_table: 2,
                sequence_indexes: vec![0],
                categorical_without_subject: vec![[
                    street_id(Street::Flop) as i64,
                    position_id(Position::Button) as i64,
                    action_id(kind) as i64,
                    0,
                ]],
                continuous: vec![continuous],
                continuous_present: vec![present],
                decision_time_ms: vec![600],
            });
        }
        BehaviourParquetDataset {
            summary: BehaviourParquetDatasetSummary {
                hands_seen: 4,
                hands_written: 4,
                events_written: 4,
                subject_hand_rows_written: 4,
                dropped_hands_missing_timing: 0,
            },
            hands,
            subjects: BTreeMap::from([(
                42,
                (0..4)
                    .map(|hand_slot| SubjectMembership {
                        hand_slot,
                        subject_position: Position::Button,
                    })
                    .collect(),
            )]),
            event_count: 4,
        }
    }

    fn trainer() -> GeneratorTrainer {
        let model =
            NeuralBehaviourGenerator::initialize(BehaviourLatentConfig::new(2).unwrap(), 8, 42)
                .unwrap();
        let adam = Adam::new(&model, 0.001).unwrap();
        GeneratorTrainer::new(
            model,
            adam,
            GeneratorTrainingConfig::new(0.005, 0.01).unwrap(),
        )
    }

    fn space() -> LegalActionSpace {
        LegalActionSpace {
            fold: false,
            check: true,
            call: false,
            bet: Some(crate::behavioural::WagerBounds {
                min_added_chips: 1,
                max_added_chips: 80,
                chip_unit: 1,
                big_blind_chips: 1,
            }),
            raise: None,
        }
    }

    #[test]
    fn learns_generator_and_latent_using_only_support_decisions() {
        let mut trainer = trainer();
        let before = trainer.model().parameters().clone();
        let mut queried = 0;
        let report = trainer
            .train_epoch(
                &dataset(),
                &[42],
                GeneratorHandSplit::new(2, 2, 2).unwrap(),
                &mut |_, _: &HandId, _| -> Result<_, &'static str> {
                    queried += 1;
                    Ok(space())
                },
            )
            .unwrap();
        assert_eq!(report.subjects, 1);
        assert_eq!(report.train.decisions, 2);
        assert_eq!(report.held_out.decisions, 2);
        assert_eq!(queried, 4);
        assert_eq!(trainer.optimizer_steps(), 2);
        assert_ne!(trainer.model().parameters(), &before);
        assert_eq!(trainer.trained_subjects(), 1);
        assert!(
            trainer
                .profile(42)
                .unwrap()
                .as_slice()
                .iter()
                .all(|v| v.is_finite())
        );
    }

    #[test]
    fn missing_exact_legality_is_error_and_does_not_update_state() {
        let mut trainer = trainer();
        let before = trainer.model().parameters().clone();
        let error = trainer
            .train_epoch(
                &dataset(),
                &[42],
                GeneratorHandSplit::new(2, 2, 2).unwrap(),
                &mut |_, _: &HandId, _| -> Result<LegalActionSpace, &'static str> {
                    Err("rules engine replay not supplied")
                },
            )
            .unwrap_err();
        assert!(matches!(
            error,
            GeneratorTrainingError::MissingLegality { .. }
        ));
        assert_eq!(trainer.optimizer_steps(), 0);
        assert_eq!(trainer.model().parameters(), &before);
        assert!(trainer.profile(42).is_none());
    }
}
