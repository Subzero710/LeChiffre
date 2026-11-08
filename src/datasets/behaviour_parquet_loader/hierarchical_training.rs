//! Hierarchical Z0/Z1/Z2 training: strictly chronological public observations.
//!
//! Training is deliberately divided into explicit phases. No profile is used
//! before its observations exist, and G/A gradients use one fixed model snapshot
//! per batch. This module is an in-memory trainer: versioned checkpoints/CLI
//! are part of deployment stage 7, not silently compatible with v1 checkpoints.
use std::collections::HashSet;
use std::fmt;

use super::BehaviourParquetDataset;
use super::generator_training::{
    ExactLegalityProvider, GeneratorHandSplit, GeneratorTrainingError, SubjectSample,
    decode_decision, lookup, subject_decisions,
};
use crate::behavioural::{
    Adam, AnalyzerAdam, AnalyzerError, AnalyzerGradientBatch, BaselineAdaptationConfig,
    BaselineAdaptationError, BehaviourBlueprint, BehaviourLatentModel, BehaviourProfileError,
    GeneratorError, GeneratorGradientBatch, GeneratorObservation, LegalActionSpace,
    NeuralBehaviourAnalyzer, NeuralBehaviourGenerator, NoiseStream, PlayerBaseline,
    PlayerBaselineAdapter,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HierarchicalPhase {
    Blueprint,
    Personalization,
    Momentary,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HierarchicalEpochReport {
    pub phase: HierarchicalPhase,
    pub decisions: usize,
    pub mean_nll: f64,
    /// The conditional KL is available only for Momentary training.
    pub mean_conditional_kl: Option<f64>,
}

/// These are different quantities: posterior reconstruction is not forecasting.
#[derive(Debug, Clone, PartialEq)]
pub struct HierarchicalValidationReport {
    pub decisions: usize,
    pub z0_predictive_nll: f64,
    pub z1_predictive_nll: f64,
    pub z2_prior_predictive_nll: f64,
    pub z2_posterior_reconstruction_nll: f64,
    pub mean_conditional_kl: f64,
}

/// Contains one global blueprint, one shared A and G, and their optimizers.
/// Z1 is ephemeral, keyed by a subject only during sequential adaptation.
#[derive(Debug, Clone)]
pub struct HierarchicalTrainer {
    blueprint: BehaviourBlueprint,
    generator: NeuralBehaviourGenerator,
    analyzer: NeuralBehaviourAnalyzer,
    generator_adam: Adam,
    analyzer_adam: AnalyzerAdam,
    baseline_config: BaselineAdaptationConfig,
    blueprint_learning_rate: f64,
    beta_kl: f64,
    noise: NoiseStream,
    blueprint_epochs: u64,
    personalization_epochs: u64,
    momentary_epochs: u64,
}

impl HierarchicalTrainer {
    /// All hyperparameters and the initial population prior are explicit.
    /// The initial Z0 is not a pretrained population model. Use
    /// `train_blueprint_epoch` before personalizing anyone.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        blueprint: BehaviourBlueprint,
        generator: NeuralBehaviourGenerator,
        analyzer: NeuralBehaviourAnalyzer,
        generator_adam: Adam,
        analyzer_adam: AnalyzerAdam,
        baseline_config: BaselineAdaptationConfig,
        blueprint_learning_rate: f64,
        beta_kl: f64,
        noise_seed: u64,
    ) -> Result<Self, HierarchicalTrainingError> {
        if generator.latent_config() != analyzer.latent_config()
            || generator.latent_config().dimension() != blueprint.centre().dimension()
        {
            return Err(HierarchicalTrainingError::LatentDimensionMismatch);
        }
        if !blueprint_learning_rate.is_finite()
            || blueprint_learning_rate <= 0.0
            || !beta_kl.is_finite()
            || beta_kl <= 0.0
        {
            return Err(HierarchicalTrainingError::InvalidConfiguration);
        }
        let noise = NoiseStream::new(noise_seed)?;
        Ok(Self {
            blueprint,
            generator,
            analyzer,
            generator_adam,
            analyzer_adam,
            baseline_config,
            blueprint_learning_rate,
            beta_kl,
            noise,
            blueprint_epochs: 0,
            personalization_epochs: 0,
            momentary_epochs: 0,
        })
    }

    pub fn blueprint(&self) -> &BehaviourBlueprint {
        &self.blueprint
    }
    pub fn generator(&self) -> &NeuralBehaviourGenerator {
        &self.generator
    }
    pub fn analyzer(&self) -> &NeuralBehaviourAnalyzer {
        &self.analyzer
    }
    pub fn completed_epochs(&self) -> (u64, u64, u64) {
        (
            self.blueprint_epochs,
            self.personalization_epochs,
            self.momentary_epochs,
        )
    }

    /// Population pretraining: only hands before the global query boundary
    /// contribute. Update Z0 as an explicit learned latent vector alongside G.
    /// The prior standard deviation remains an explicitly supplied hyperparameter,
    /// not a calibrated population dispersion estimate.
    pub fn train_blueprint_epoch<P: ExactLegalityProvider>(
        &mut self,
        dataset: &BehaviourParquetDataset,
        subjects: &[u64],
        split: GeneratorHandSplit,
        provider: &mut P,
    ) -> Result<HierarchicalEpochReport, HierarchicalTrainingError> {
        let mut working = self.clone();
        let selected = selected_subjects(dataset, subjects)?;
        let mut batch = GeneratorGradientBatch::new(&working.generator);
        for id in selected {
            let memberships = dataset
                .subjects
                .get(&id)
                .ok_or(HierarchicalTrainingError::UnknownSubject(id))?;
            for membership in memberships {
                let hand = &dataset.hands[membership.hand_slot];
                if hand.hand_index >= split.first_query_hand_index {
                    continue;
                }
                let observed = hand.materialize(membership.subject_position);
                for row in subject_decisions(&observed) {
                    let observation = decode_decision(&observed, row)?;
                    let legal = lookup(provider, id, &observed, row)?;
                    batch.accumulate_encoded(
                        &working.generator,
                        working.blueprint.centre(),
                        &observation,
                        &legal,
                    )?;
                }
            }
        }
        let mean = batch.mean_loss()?;
        let decisions = batch.len();
        let mut centre = Vec::with_capacity(working.blueprint.centre().dimension());
        for (&value, &gradient) in working
            .blueprint
            .centre()
            .as_slice()
            .iter()
            .zip(&mean.latent_gradient)
        {
            let next = f64::from(value) - working.blueprint_learning_rate * f64::from(gradient);
            if !next.is_finite() || next.abs() > f64::from(f32::MAX) {
                return Err(HierarchicalTrainingError::NonFiniteUpdate);
            }
            centre.push(next as f32);
        }
        let next_centre = working.generator.latent_config().profile(centre)?;
        let next_blueprint =
            BehaviourBlueprint::new(next_centre, working.blueprint.prior_std().to_vec())?;
        working
            .generator_adam
            .apply_batch(&mut working.generator, batch)?;
        working.blueprint = next_blueprint;
        working.blueprint_epochs = working
            .blueprint_epochs
            .checked_add(1)
            .ok_or(HierarchicalTrainingError::CountOverflow)?;
        *self = working;
        Ok(HierarchicalEpochReport {
            phase: HierarchicalPhase::Blueprint,
            decisions,
            mean_nll: mean.total,
            mean_conditional_kl: None,
        })
    }

    /// First-order personalization training: adapt a private Z1 on the latest
    /// `history_hands` before the global query split, then train shared G only on
    /// later query hands. Zero is valid (exact Z1=Z0); too few hands is an error.
    /// Use different *explicit* history_hands across epochs (0,10,100,1000...).
    pub fn train_personalization_epoch<P: ExactLegalityProvider>(
        &mut self,
        dataset: &BehaviourParquetDataset,
        subjects: &[u64],
        split: GeneratorHandSplit,
        history_hands: usize,
        provider: &mut P,
    ) -> Result<HierarchicalEpochReport, HierarchicalTrainingError> {
        if self.blueprint_epochs == 0 || history_hands > split.support_hands {
            return Err(HierarchicalTrainingError::InvalidPhaseOrHistory);
        }
        let samples =
            super::generator_training::GeneratorTrainer::samples(dataset, subjects, split)?;
        let mut working = self.clone();
        let mut sum_nll = 0.0;
        let mut count = 0usize;
        for sample in &samples {
            let baseline = working.personalize(
                dataset,
                sample.id,
                split.first_query_hand_index,
                history_hands,
                provider,
            )?;
            let mut batch = GeneratorGradientBatch::new(&working.generator);
            for hand in &sample.query {
                for row in subject_decisions(hand) {
                    let observation = decode_decision(hand, row)?;
                    let legal = lookup(provider, sample.id, hand, row)?;
                    batch.accumulate_encoded(
                        &working.generator,
                        baseline.profile(),
                        &observation,
                        &legal,
                    )?;
                }
            }
            let mean = batch.mean_loss()?;
            sum_nll += mean.total * batch.len() as f64;
            count = count
                .checked_add(batch.len())
                .ok_or(HierarchicalTrainingError::CountOverflow)?;
            working
                .generator_adam
                .apply_batch(&mut working.generator, batch)?;
        }
        if count == 0 {
            return Err(HierarchicalTrainingError::EmptyTraining);
        }
        working.personalization_epochs = working
            .personalization_epochs
            .checked_add(1)
            .ok_or(HierarchicalTrainingError::CountOverflow)?;
        *self = working;
        Ok(HierarchicalEpochReport {
            phase: HierarchicalPhase::Personalization,
            decisions: count,
            mean_nll: sum_nll / count as f64,
            mean_conditional_kl: None,
        })
    }

    /// Decision-level variational learning: prior Z2 sees only C/Z1, posterior
    /// Z2 sees C/ObservedAction/Z1. G learns both branches under the complete
    /// action+sizing+timing likelihood; A receives both reverse-mode gradients
    /// and conditional KL. Z1 is fitted only on the preceding support hands.
    pub fn train_momentary_epoch<P: ExactLegalityProvider>(
        &mut self,
        dataset: &BehaviourParquetDataset,
        subjects: &[u64],
        split: GeneratorHandSplit,
        history_hands: usize,
        provider: &mut P,
    ) -> Result<HierarchicalEpochReport, HierarchicalTrainingError> {
        if self.blueprint_epochs == 0
            || self.personalization_epochs == 0
            || history_hands > split.support_hands
        {
            return Err(HierarchicalTrainingError::InvalidPhaseOrHistory);
        }
        let samples =
            super::generator_training::GeneratorTrainer::samples(dataset, subjects, split)?;
        let mut working = self.clone();
        let mut nll_sum = 0.0;
        let mut kl_sum = 0.0;
        let mut decisions = 0usize;
        for sample in &samples {
            let baseline = working.personalize(
                dataset,
                sample.id,
                split.first_query_hand_index,
                history_hands,
                provider,
            )?;
            let observations = collect_query(sample, provider)?;
            let (total_nll, total_kl) =
                working.train_momentary_decisions(&baseline, &observations)?;
            nll_sum += total_nll;
            kl_sum += total_kl;
            decisions = decisions
                .checked_add(observations.len())
                .ok_or(HierarchicalTrainingError::CountOverflow)?;
        }
        if decisions == 0 {
            return Err(HierarchicalTrainingError::EmptyTraining);
        }
        working.momentary_epochs = working
            .momentary_epochs
            .checked_add(1)
            .ok_or(HierarchicalTrainingError::CountOverflow)?;
        *self = working;
        Ok(HierarchicalEpochReport {
            phase: HierarchicalPhase::Momentary,
            decisions,
            mean_nll: nll_sum / decisions as f64,
            mean_conditional_kl: Some(kl_sum / decisions as f64),
        })
    }

    fn train_momentary_decisions(
        &mut self,
        baseline: &PlayerBaseline,
        decisions: &[(GeneratorObservation, LegalActionSpace)],
    ) -> Result<(f64, f64), HierarchicalTrainingError> {
        let mut g_batch = GeneratorGradientBatch::new(&self.generator);
        let mut a_batch = AnalyzerGradientBatch::new(&self.analyzer);
        let mut next_noise = self.noise;
        let mut nll = 0.0;
        let mut kl = 0.0;
        for (observation, legal) in decisions {
            let posterior = self.analyzer.infer_encoded(observation, baseline)?;
            let prior_noise = next_noise.sample(self.generator.latent_config().dimension());
            let post_noise = next_noise.sample(self.generator.latent_config().dimension());
            let prior_z = posterior.prior().sample(&prior_noise)?;
            let post_z = posterior.posterior().sample(&post_noise)?;
            let prior_loss =
                g_batch.accumulate_encoded(&self.generator, &prior_z, observation, legal)?;
            let post_loss =
                g_batch.accumulate_encoded(&self.generator, &post_z, observation, legal)?;
            kl += a_batch.accumulate_encoded(
                &self.analyzer,
                baseline,
                observation,
                &prior_noise,
                &post_noise,
                &prior_loss.latent_gradient,
                &post_loss.latent_gradient,
                self.beta_kl,
            )?;
            nll += (prior_loss.total + post_loss.total) * 0.5;
        }
        if decisions.is_empty() {
            return Err(HierarchicalTrainingError::EmptyTraining);
        }
        // Compute both candidate updates before committing either one (or RNG).
        let mut next_g = self.generator.clone();
        let mut next_a = self.analyzer.clone();
        let mut next_g_optimizer = self.generator_adam.clone();
        let mut next_a_optimizer = self.analyzer_adam.clone();
        next_g_optimizer.apply_batch(&mut next_g, g_batch)?;
        next_a_optimizer.apply_decision_batch(&mut next_a, a_batch)?;
        self.generator = next_g;
        self.analyzer = next_a;
        self.generator_adam = next_g_optimizer;
        self.analyzer_adam = next_a_optimizer;
        self.noise = next_noise;
        Ok((nll, kl))
    }

    fn personalize<P: ExactLegalityProvider>(
        &self,
        dataset: &BehaviourParquetDataset,
        subject: u64,
        before_index: u64,
        count: usize,
        provider: &mut P,
    ) -> Result<PlayerBaseline, HierarchicalTrainingError> {
        let adapter = PlayerBaselineAdapter::new(self.blueprint.clone(), self.baseline_config)?;
        let mut state = adapter.initialize_player();
        let memberships = dataset
            .subjects
            .get(&subject)
            .ok_or(HierarchicalTrainingError::UnknownSubject(subject))?;
        let mut selected = memberships
            .iter()
            .filter(|membership| dataset.hands[membership.hand_slot].hand_index < before_index)
            .rev()
            .take(count)
            .collect::<Vec<_>>();
        if selected.len() != count {
            return Err(HierarchicalTrainingError::InsufficientHistory {
                subject,
                required: count,
                available: selected.len(),
            });
        }
        selected.reverse();
        for membership in selected {
            let hand = &dataset.hands[membership.hand_slot];
            let observed = hand.materialize(membership.subject_position);
            let mut decisions = Vec::new();
            for row in subject_decisions(&observed) {
                decisions.push((
                    decode_decision(&observed, row)?,
                    lookup(provider, subject, &observed, row)?,
                ));
            }
            if !decisions.is_empty() {
                adapter.adapt_hand(&mut state, &self.generator, hand.hand_index, &decisions)?;
            }
        }
        Ok(state.baseline().clone())
    }

    /// Evaluation is read-only and strictly after `first_evaluation_hand_index`.
    /// Z1 is allowed to use all earlier hands, including query training hands,
    /// but never a validation decision. Posterior NLL is labelled reconstruction.
    pub fn evaluate<P: ExactLegalityProvider>(
        &self,
        dataset: &BehaviourParquetDataset,
        subjects: &[u64],
        split: GeneratorHandSplit,
        provider: &mut P,
    ) -> Result<HierarchicalValidationReport, HierarchicalTrainingError> {
        let samples =
            super::generator_training::GeneratorTrainer::samples(dataset, subjects, split)?;
        let adapter = PlayerBaselineAdapter::new(self.blueprint.clone(), self.baseline_config)?;
        let mut sum = [0.0_f64; 4];
        let mut kl = 0.0;
        let mut count = 0usize;
        for sample in &samples {
            let baseline = dataset.adapt_baseline_before(
                sample.id,
                split.first_evaluation_hand_index,
                &self.generator,
                &adapter,
                provider,
            )?;
            for hand in &sample.validation {
                for row in subject_decisions(hand) {
                    let observation = decode_decision(hand, row)?;
                    let legal = lookup(provider, sample.id, hand, row)?;
                    let z2 = self
                        .analyzer
                        .infer_encoded(&observation, baseline.baseline())?;
                    let profiles = [
                        self.blueprint.centre().clone(),
                        baseline.baseline().profile().clone(),
                        z2.prior().mean_profile()?,
                        z2.posterior().mean_profile()?,
                    ];
                    for (total, profile) in sum.iter_mut().zip(&profiles) {
                        *total += self
                            .generator
                            .evaluate_observation_encoded(profile, &observation, &legal)?
                            .total;
                    }
                    kl += z2.kl_to_prior()?;
                    count = count
                        .checked_add(1)
                        .ok_or(HierarchicalTrainingError::CountOverflow)?;
                }
            }
        }
        if count == 0 {
            return Err(HierarchicalTrainingError::EmptyValidation);
        }
        if !kl.is_finite() || sum.iter().any(|v| !v.is_finite()) {
            return Err(HierarchicalTrainingError::NonFiniteUpdate);
        }
        Ok(HierarchicalValidationReport {
            decisions: count,
            z0_predictive_nll: sum[0] / count as f64,
            z1_predictive_nll: sum[1] / count as f64,
            z2_prior_predictive_nll: sum[2] / count as f64,
            z2_posterior_reconstruction_nll: sum[3] / count as f64,
            mean_conditional_kl: kl / count as f64,
        })
    }
}

fn selected_subjects(
    dataset: &BehaviourParquetDataset,
    subjects: &[u64],
) -> Result<Vec<u64>, HierarchicalTrainingError> {
    if subjects.is_empty() {
        return Err(HierarchicalTrainingError::EmptyTraining);
    }
    let mut unique = HashSet::new();
    for &id in subjects {
        if !unique.insert(id) {
            return Err(HierarchicalTrainingError::DuplicateSubject(id));
        }
        if !dataset.subjects.contains_key(&id) {
            return Err(HierarchicalTrainingError::UnknownSubject(id));
        }
    }
    Ok(subjects.to_vec())
}

fn collect_query<P: ExactLegalityProvider>(
    sample: &SubjectSample,
    provider: &mut P,
) -> Result<Vec<(GeneratorObservation, LegalActionSpace)>, HierarchicalTrainingError> {
    let mut result = Vec::new();
    for hand in &sample.query {
        for row in subject_decisions(hand) {
            result.push((
                decode_decision(hand, row)?,
                lookup(provider, sample.id, hand, row)?,
            ));
        }
    }
    if result.is_empty() {
        return Err(HierarchicalTrainingError::EmptyTraining);
    }
    Ok(result)
}

#[derive(Debug)]
pub enum HierarchicalTrainingError {
    InvalidConfiguration,
    InvalidPhaseOrHistory,
    LatentDimensionMismatch,
    UnknownSubject(u64),
    DuplicateSubject(u64),
    InsufficientHistory {
        subject: u64,
        required: usize,
        available: usize,
    },
    EmptyTraining,
    EmptyValidation,
    NonFiniteUpdate,
    CountOverflow,
    Generator(GeneratorError),
    Analyzer(AnalyzerError),
    Baseline(BaselineAdaptationError),
    Profile(BehaviourProfileError),
    Data(GeneratorTrainingError),
}
impl fmt::Display for HierarchicalTrainingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for HierarchicalTrainingError {}
impl From<GeneratorError> for HierarchicalTrainingError {
    fn from(value: GeneratorError) -> Self {
        Self::Generator(value)
    }
}
impl From<AnalyzerError> for HierarchicalTrainingError {
    fn from(value: AnalyzerError) -> Self {
        Self::Analyzer(value)
    }
}
impl From<BaselineAdaptationError> for HierarchicalTrainingError {
    fn from(value: BaselineAdaptationError) -> Self {
        Self::Baseline(value)
    }
}
impl From<BehaviourProfileError> for HierarchicalTrainingError {
    fn from(value: BehaviourProfileError) -> Self {
        Self::Profile(value)
    }
}
impl From<GeneratorTrainingError> for HierarchicalTrainingError {
    fn from(value: GeneratorTrainingError) -> Self {
        Self::Data(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::behavioural::{
        ActionKind, ActionSizing, BehaviourLatentConfig, ObservedAction, PRE_ACTION_CONTEXT_WIDTH,
        Position, Street, WagerBounds,
    };

    fn trainer() -> HierarchicalTrainer {
        let latent = BehaviourLatentConfig::new(4).unwrap();
        let blueprint =
            BehaviourBlueprint::new(latent.profile(vec![0.0; 4]).unwrap(), vec![1.0; 4]).unwrap();
        let generator = NeuralBehaviourGenerator::initialize(latent, 8, 42).unwrap();
        let analyzer = NeuralBehaviourAnalyzer::initialize(latent, 8, 43).unwrap();
        let gopt = Adam::new(&generator, 0.0005).unwrap();
        let aopt = AnalyzerAdam::new(&analyzer, 0.0005).unwrap();
        HierarchicalTrainer::new(
            blueprint,
            generator,
            analyzer,
            gopt,
            aopt,
            BaselineAdaptationConfig::new(0.2, 0.1).unwrap(),
            0.001,
            0.01,
            41,
        )
        .unwrap()
    }

    fn decision(ms: u32) -> (GeneratorObservation, LegalActionSpace) {
        let mut context = [0.0_f32; PRE_ACTION_CONTEXT_WIDTH];
        context[0] = 10.0;
        context[1] = 80.0;
        context[3] = 5.0;
        context[6] = 1.0;
        context[7] = 1.0;
        (
            GeneratorObservation {
                street: Street::Flop,
                actor_position: Position::Button,
                context_values: context,
                context_present: [1; PRE_ACTION_CONTEXT_WIDTH],
                observed: ObservedAction {
                    kind: ActionKind::Check,
                    is_all_in: false,
                    sizing: Some(ActionSizing {
                        amount_over_pot_before: 0.0,
                        amount_over_stack_before: 0.0,
                        raise_increment_over_pot_before: None,
                    }),
                    actor_aggression_streak_after: 0,
                    table_aggression_streak_after: 0,
                    decision_time_ms: Some(ms),
                },
            },
            LegalActionSpace {
                fold: false,
                check: true,
                call: false,
                bet: Some(WagerBounds {
                    min_added_chips: 1,
                    max_added_chips: 80,
                    chip_unit: 1,
                    big_blind_chips: 1,
                }),
                raise: None,
            },
        )
    }

    fn dataset() -> BehaviourParquetDataset {
        use super::super::{PersistedHand, SubjectMembership};
        use crate::behavioural::codebook::{action_id, position_id, street_id};
        use crate::behavioural::{ContinuousFeature, EVENT_CONTINUOUS_WIDTH, HandId};
        use crate::datasets::behaviour_parquet::BehaviourParquetDatasetSummary;
        use std::collections::BTreeMap;
        let mut hands = Vec::new();
        for index in 0_u64..9 {
            let kind = if index.is_multiple_of(2) {
                ActionKind::Check
            } else {
                ActionKind::Bet
            };
            let mut values = [0.0_f32; EVENT_CONTINUOUS_WIDTH];
            values[ContinuousFeature::PotBb as usize] = 10.0;
            values[ContinuousFeature::StackRemainingBb as usize] = 80.0;
            values[ContinuousFeature::CommittedThisHandBb as usize] = 5.0;
            values[ContinuousFeature::MinRaiseToBb as usize] = 1.0;
            values[ContinuousFeature::PlayersYetToAct as usize] = 1.0;
            values[ContinuousFeature::AmountOverPot as usize] =
                if kind == ActionKind::Bet { 0.5 } else { 0.0 };
            values[ContinuousFeature::AmountOverStack as usize] =
                if kind == ActionKind::Bet { 0.0625 } else { 0.0 };
            values[ContinuousFeature::DecisionTimeMs as usize] = 600.0;
            let mut present = [1_u8; EVENT_CONTINUOUS_WIDTH];
            present[ContinuousFeature::RaiseIncrementOverPot as usize] = 0;
            hands.push(PersistedHand {
                hand_id: HandId::new(format!("h{index}")),
                hand_index: index,
                players_at_table: 2,
                sequence_indexes: vec![0],
                categorical_without_subject: vec![[
                    i64::from(street_id(Street::Flop)),
                    i64::from(position_id(Position::Button)),
                    i64::from(action_id(kind)),
                    0,
                ]],
                continuous: vec![values],
                continuous_present: vec![present],
                decision_time_ms: vec![600],
            });
        }
        let memberships = (0..9)
            .map(|hand_slot| SubjectMembership {
                hand_slot,
                subject_position: Position::Button,
            })
            .collect::<Vec<_>>();
        BehaviourParquetDataset {
            summary: BehaviourParquetDatasetSummary {
                hands_seen: 9,
                hands_written: 9,
                events_written: 9,
                subject_hand_rows_written: 9,
                dropped_hands_missing_timing: 0,
                dropped_hands_invalid: 0,
            },
            hands,
            subjects: BTreeMap::from([(42, memberships)]),
            event_count: 9,
        }
    }

    #[test]
    fn three_training_phases_preserve_global_future_partition() {
        let dataset = dataset();
        let split = GeneratorHandSplit::new(3, 6, 3, 3, 3).unwrap();
        let mut trainer = trainer();
        let seen = std::cell::RefCell::new(Vec::<String>::new());
        let mut legal_source = |_: u64,
                                hand: &crate::behavioural::HandId,
                                _: u32|
         -> Result<LegalActionSpace, &'static str> {
            seen.borrow_mut().push(hand.as_str().to_owned());
            Ok(decision(600).1)
        };
        let baseline = trainer
            .train_blueprint_epoch(&dataset, &[42], split, &mut legal_source)
            .unwrap();
        assert_eq!(baseline.phase, HierarchicalPhase::Blueprint);
        assert_eq!(baseline.decisions, 3);
        assert_eq!(
            seen.borrow().iter().map(String::as_str).collect::<Vec<_>>(),
            vec!["h0", "h1", "h2"]
        );
        seen.borrow_mut().clear();
        let personal = trainer
            .train_personalization_epoch(&dataset, &[42], split, 2, &mut legal_source)
            .unwrap();
        assert_eq!(personal.decisions, 3);
        assert!(
            !seen
                .borrow()
                .iter()
                .any(|x| ["h6", "h7", "h8"].contains(&x.as_str()))
        );
        seen.borrow_mut().clear();
        let momentary = trainer
            .train_momentary_epoch(&dataset, &[42], split, 2, &mut legal_source)
            .unwrap();
        assert_eq!(momentary.decisions, 3);
        assert!(momentary.mean_conditional_kl.unwrap().is_finite());
        assert!(
            !seen
                .borrow()
                .iter()
                .any(|x| ["h6", "h7", "h8"].contains(&x.as_str()))
        );
        let report = trainer
            .evaluate(&dataset, &[42], split, &mut legal_source)
            .unwrap();
        assert_eq!(report.decisions, 3);
        for value in [
            report.z0_predictive_nll,
            report.z1_predictive_nll,
            report.z2_prior_predictive_nll,
            report.z2_posterior_reconstruction_nll,
        ] {
            assert!(value.is_finite());
        }
        assert_eq!(trainer.completed_epochs(), (1, 1, 1));
    }

    #[test]
    fn encoded_prior_never_observes_action_or_timing() {
        let trainer = trainer();
        let baseline = PlayerBaselineAdapter::new(
            trainer.blueprint().clone(),
            BaselineAdaptationConfig::new(0.2, 0.1).unwrap(),
        )
        .unwrap()
        .initialize_player();
        let a = decision(500);
        let b = decision(20000);
        let prior_a = trainer
            .analyzer
            .prior_encoded(&a.0, baseline.baseline())
            .unwrap();
        let prior_b = trainer
            .analyzer
            .prior_encoded(&b.0, baseline.baseline())
            .unwrap();
        assert_eq!(prior_a, prior_b);
        let post_a = trainer
            .analyzer
            .infer_encoded(&a.0, baseline.baseline())
            .unwrap();
        let post_b = trainer
            .analyzer
            .infer_encoded(&b.0, baseline.baseline())
            .unwrap();
        assert_ne!(post_a.posterior(), post_b.posterior());
    }

    #[test]
    fn momentary_training_updates_shared_a_and_g_without_mutating_z0_or_z1() {
        let mut trainer = trainer();
        let baseline = PlayerBaselineAdapter::new(
            trainer.blueprint().clone(),
            BaselineAdaptationConfig::new(0.2, 0.1).unwrap(),
        )
        .unwrap()
        .initialize_player();
        let original_baseline = baseline.baseline().clone();
        let original_blueprint = trainer.blueprint().clone();
        let original_a = trainer.analyzer.parameters().clone();
        let original_g = trainer.generator.parameters().clone();
        let (loss, kl) = trainer
            .train_momentary_decisions(baseline.baseline(), &[decision(500), decision(10000)])
            .unwrap();
        assert!(loss.is_finite() && kl.is_finite() && kl >= 0.0);
        assert_ne!(&original_a, trainer.analyzer.parameters());
        assert_ne!(&original_g, trainer.generator.parameters());
        assert_eq!(&original_blueprint, trainer.blueprint());
        assert_eq!(&original_baseline, baseline.baseline());
    }

    #[test]
    fn invalid_second_decision_is_atomic_for_both_models() {
        let mut trainer = trainer();
        let before_g = trainer.generator.parameters().clone();
        let before_a = trainer.analyzer.parameters().clone();
        let baseline = PlayerBaselineAdapter::new(
            trainer.blueprint().clone(),
            BaselineAdaptationConfig::new(0.2, 0.1).unwrap(),
        )
        .unwrap()
        .initialize_player();
        let (observation, mut bad_legal) = decision(500);
        bad_legal.check = false;
        let error = trainer
            .train_momentary_decisions(
                baseline.baseline(),
                &[decision(500), (observation, bad_legal)],
            )
            .unwrap_err();
        assert!(matches!(error, HierarchicalTrainingError::Generator(_)));
        assert_eq!(&before_g, trainer.generator.parameters());
        assert_eq!(&before_a, trainer.analyzer.parameters());
    }
}
