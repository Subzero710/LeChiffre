//! Joint A/G variational learning from strict support, query and future validation hands.
//! The three partitions are global by hand index, so no player's target hand can
//! appear in another subject's Analyzer support or Generator optimization.
use std::collections::HashSet;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use super::BehaviourParquetDataset;
use crate::behavioural::checkpoint::{dimension, exact_keys, number, unsigned};
use crate::behavioural::codebook::{
    action_from_id, action_id, position_from_id, position_id, street_from_id,
};
use crate::behavioural::{
    ACTION_COUNT, ActionKind, ActionSizing, Adam, AnalyzerAdam, AnalyzerError,
    BaselineAdaptationError, BehaviourLatentModel, BehaviourProfile, CategoricalFeature,
    ContinuousFeature, FeaturizedHand, GeneratorError, GeneratorGradientBatch,
    GeneratorObservation, HandId, LegalActionSpace, NeuralBehaviourAnalyzer,
    NeuralBehaviourGenerator, NoiseStream, ObservedAction, PRE_ACTION_CONTEXT_WIDTH, TrainingLoss,
};
use serde_json::{Value, json};

/// Legal-space source must replay the exact pre-decision game state, never infer
/// legal actions from Parquet observations or the action that was chosen.
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

/// Support < first_query <= query < first_evaluation <= evaluation.
/// All counts are explicit and nonzero; no per-subject split can cross a boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratorHandSplit {
    pub first_query_hand_index: u64,
    pub first_evaluation_hand_index: u64,
    pub support_hands: usize,
    pub query_hands: usize,
    pub evaluation_hands: usize,
}
impl GeneratorHandSplit {
    pub fn new(
        first_query_hand_index: u64,
        first_evaluation_hand_index: u64,
        support_hands: usize,
        query_hands: usize,
        evaluation_hands: usize,
    ) -> Result<Self, GeneratorTrainingError> {
        if first_evaluation_hand_index <= first_query_hand_index
            || support_hands == 0
            || query_hands == 0
            || evaluation_hands == 0
        {
            return Err(GeneratorTrainingError::InvalidConfiguration);
        }
        Ok(Self {
            first_query_hand_index,
            first_evaluation_hand_index,
            support_hands,
            query_hands,
            evaluation_hands,
        })
    }
    fn checkpoint_value(self) -> Value {
        json!({"first_query_hand_index":self.first_query_hand_index,
               "first_evaluation_hand_index":self.first_evaluation_hand_index,
               "support_hands":self.support_hands,"query_hands":self.query_hands,"evaluation_hands":self.evaluation_hands})
    }
    fn from_checkpoint(value: &Value) -> Result<Self, String> {
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

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeneratorTrainingConfig {
    pub beta_kl: f64,
    pub initial_noise_seed: u64,
    pub validation_draws: usize,
    pub continuity_delta: f64,
}
impl GeneratorTrainingConfig {
    pub fn new(
        beta_kl: f64,
        initial_noise_seed: u64,
        validation_draws: usize,
        continuity_delta: f64,
    ) -> Result<Self, GeneratorTrainingError> {
        if !beta_kl.is_finite()
            || beta_kl < 0.0
            || initial_noise_seed == 0
            || !(2..=1024).contains(&validation_draws)
            || !continuity_delta.is_finite()
            || continuity_delta <= 0.0
            || continuity_delta > 1.0
        {
            return Err(GeneratorTrainingError::InvalidConfiguration);
        }
        Ok(Self {
            beta_kl,
            initial_noise_seed,
            validation_draws,
            continuity_delta,
        })
    }
    fn checkpoint_value(self) -> Value {
        json!({"beta_kl":self.beta_kl,"initial_noise_seed":self.initial_noise_seed,
               "validation_draws":self.validation_draws,"continuity_delta":self.continuity_delta})
    }
    fn from_checkpoint(value: &Value) -> Result<Self, String> {
        exact_keys(
            value,
            &[
                "beta_kl",
                "initial_noise_seed",
                "validation_draws",
                "continuity_delta",
            ],
        )?;
        Self::new(
            number(&value["beta_kl"])?,
            unsigned(&value["initial_noise_seed"])?,
            dimension(&value["validation_draws"])?,
            number(&value["continuity_delta"])?,
        )
        .map_err(|e| e.to_string())
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
        if [
            self.total_nll,
            self.action_nll,
            self.sizing_nll,
            self.timing_nll,
        ]
        .iter()
        .any(|x| !x.is_finite())
        {
            return Err(GeneratorTrainingError::NonFiniteLoss);
        }
        Ok(())
    }
    pub fn mean_nll(&self) -> Result<f64, GeneratorTrainingError> {
        if self.decisions == 0 {
            return Err(GeneratorTrainingError::NoDecisions);
        }
        Ok(self.total_nll / self.decisions as f64)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GeneratorEpochReport {
    pub epoch: u64,
    pub subjects: usize,
    /// Query likelihood that backpropagated into both A and G.
    pub train: GeneratorLossSummary,
    /// Future validation likelihood at q(Z | support).mean, never optimized.
    pub held_out: GeneratorLossSummary,
    /// Same validation decisions at neutral Z=0.
    pub prior: GeneratorLossSummary,
    /// Monte-Carlo marginal predictive NLL on held-out decisions.
    pub marginal_validation_nll: f64,
    pub mean_kl: f64,
    pub posterior_std_mean: f64,
    /// Mean variance of action probabilities under posterior Z sampling.
    /// A value of zero indicates no measured action-policy sensitivity to Z.
    pub posterior_action_variance: f64,
    /// Ten-bin expected calibration error of the top action on future decisions.
    pub action_ece: f64,
    /// Mean pairwise Jensen-Shannon divergence across subjects on the SAME public context.
    pub policy_diversity_js: Option<f64>,
    /// Jensen-Shannon divergence when perturbing z[0] by continuity_delta.
    pub continuity_js: f64,
}

#[derive(Debug, Clone)]
struct SubjectSample {
    id: u64,
    support: Vec<FeaturizedHand>,
    query: Vec<FeaturizedHand>,
    validation: Vec<FeaturizedHand>,
}

#[derive(Debug, Clone)]
pub struct GeneratorTrainer {
    model: NeuralBehaviourGenerator,
    analyzer: NeuralBehaviourAnalyzer,
    generator_adam: Adam,
    analyzer_adam: AnalyzerAdam,
    config: GeneratorTrainingConfig,
    noise: NoiseStream,
    epochs: u64,
    last_split: Option<GeneratorHandSplit>,
    last_subjects: Vec<u64>,
    dataset_fingerprint: Option<String>,
}
impl GeneratorTrainer {
    pub fn new(
        model: NeuralBehaviourGenerator,
        analyzer: NeuralBehaviourAnalyzer,
        generator_adam: Adam,
        analyzer_adam: AnalyzerAdam,
        config: GeneratorTrainingConfig,
    ) -> Result<Self, GeneratorTrainingError> {
        if model.latent_config() != analyzer.latent_config() {
            return Err(GeneratorTrainingError::InvalidLatentDimensions);
        }
        let noise = NoiseStream::new(config.initial_noise_seed)?;
        Ok(Self {
            model,
            analyzer,
            generator_adam,
            analyzer_adam,
            config,
            noise,
            epochs: 0,
            last_split: None,
            last_subjects: Vec::new(),
            dataset_fingerprint: None,
        })
    }
    pub fn model(&self) -> &NeuralBehaviourGenerator {
        &self.model
    }
    pub fn analyzer(&self) -> &NeuralBehaviourAnalyzer {
        &self.analyzer
    }
    pub fn generator_optimizer_steps(&self) -> u64 {
        self.generator_adam.steps()
    }
    pub fn analyzer_optimizer_steps(&self) -> u64 {
        self.analyzer_adam.steps()
    }
    pub fn completed_epochs(&self) -> u64 {
        self.epochs
    }
    pub fn training_split(&self) -> Option<GeneratorHandSplit> {
        self.last_split
    }
    pub fn training_subjects(&self) -> &[u64] {
        &self.last_subjects
    }

    fn samples(
        dataset: &BehaviourParquetDataset,
        selected: &[u64],
        split: GeneratorHandSplit,
    ) -> Result<Vec<SubjectSample>, GeneratorTrainingError> {
        if selected.is_empty() {
            return Err(GeneratorTrainingError::EmptySubjectSelection);
        }
        let mut seen = HashSet::new();
        let mut result = Vec::new();
        for &id in selected {
            if !seen.insert(id) {
                return Err(GeneratorTrainingError::DuplicateSubject { subject_id: id });
            }
            let memberships = dataset
                .subjects
                .get(&id)
                .ok_or(GeneratorTrainingError::UnknownSubject { subject_id: id })?;
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
                return Err(GeneratorTrainingError::InsufficientPartition {
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
                    return Err(GeneratorTrainingError::NoSubjectDecisions {
                        subject_id: id,
                        partition: name,
                    });
                }
            }
            result.push(SubjectSample {
                id,
                support,
                query,
                validation,
            });
        }
        Ok(result)
    }

    /// Atomic epoch: changes are committed only after every source decision,
    /// every exact legality lookup, and the entire validation pass succeeds.
    pub fn train_epoch<P: ExactLegalityProvider>(
        &mut self,
        dataset: &BehaviourParquetDataset,
        selected_subjects: &[u64],
        split: GeneratorHandSplit,
        provider: &mut P,
    ) -> Result<GeneratorEpochReport, GeneratorTrainingError> {
        if self.epochs > 0
            && (self.last_split != Some(split)
                || self.last_subjects.as_slice() != selected_subjects)
        {
            return Err(GeneratorTrainingError::TrainingPartitionChanged);
        }
        let fingerprint = dataset_fingerprint(dataset);
        if self
            .dataset_fingerprint
            .as_ref()
            .is_some_and(|previous| previous != &fingerprint)
        {
            return Err(GeneratorTrainingError::DatasetChanged);
        }
        let samples = Self::samples(dataset, selected_subjects, split)?;
        let mut working = self.clone();
        let mut train = GeneratorLossSummary::default();
        for sample in &samples {
            let posterior = working.analyzer.infer(&sample.support)?;
            let epsilon = working
                .noise
                .sample(working.model.latent_config().dimension());
            let profile = posterior.sample(&epsilon)?;
            let mut latent_grads = vec![0.0; epsilon.len()];
            let mut gradient_batch = GeneratorGradientBatch::new(&working.model);
            let mut decisions = 0_usize;
            for hand in &sample.query {
                for row in subject_decisions(hand) {
                    let observation = decode_decision(hand, row)?;
                    let legal = lookup(provider, sample.id, hand, row)?;
                    let loss = gradient_batch.accumulate_encoded(
                        &working.model,
                        &profile,
                        &observation,
                        &legal,
                    )?;
                    train.record(&loss)?;
                    for (total, g) in latent_grads.iter_mut().zip(&loss.latent_gradient) {
                        *total += f64::from(*g);
                    }
                    decisions += 1;
                }
            }
            if decisions == 0 {
                return Err(GeneratorTrainingError::NoDecisions);
            }
            for gradient in &mut latent_grads {
                *gradient /= decisions as f64;
            }
            // G was held fixed for every query observation for this subject.
            // Apply the mean G gradient once, after collecting all dL/dZ values.
            working
                .generator_adam
                .apply_batch(&mut working.model, gradient_batch)?;
            working.analyzer_adam.train_step(
                &mut working.analyzer,
                &sample.support,
                &epsilon,
                &latent_grads,
                working.config.beta_kl,
            )?;
        }
        let epoch = working
            .epochs
            .checked_add(1)
            .ok_or(GeneratorTrainingError::CountOverflow)?;
        let mut report = working.evaluate_samples(&samples, provider)?;
        report.epoch = epoch;
        report.train = train;
        working.epochs = epoch;
        working.last_split = Some(split);
        working.last_subjects = selected_subjects.to_vec();
        working.dataset_fingerprint = Some(fingerprint);
        *self = working;
        Ok(report)
    }

    pub fn evaluate<P: ExactLegalityProvider>(
        &self,
        dataset: &BehaviourParquetDataset,
        provider: &mut P,
    ) -> Result<GeneratorEpochReport, GeneratorTrainingError> {
        let split = self.last_split.ok_or(GeneratorTrainingError::NotTrained)?;
        if self.dataset_fingerprint.as_ref() != Some(&dataset_fingerprint(dataset)) {
            return Err(GeneratorTrainingError::DatasetChanged);
        }
        let samples = Self::samples(dataset, &self.last_subjects, split)?;
        self.evaluate_samples(&samples, provider)
    }

    /// Test on subjects never selected for parameter updates. Their support
    /// histories are processed by A without fitting any per-subject parameters.
    pub fn evaluate_unseen<P: ExactLegalityProvider>(
        &self,
        dataset: &BehaviourParquetDataset,
        unseen_subjects: &[u64],
        provider: &mut P,
    ) -> Result<GeneratorEpochReport, GeneratorTrainingError> {
        let split = self.last_split.ok_or(GeneratorTrainingError::NotTrained)?;
        if self.dataset_fingerprint.as_ref() != Some(&dataset_fingerprint(dataset)) {
            return Err(GeneratorTrainingError::DatasetChanged);
        }
        if let Some(&id) = unseen_subjects
            .iter()
            .find(|id| self.last_subjects.contains(*id))
        {
            return Err(GeneratorTrainingError::PreviouslyTrainedSubject { subject_id: id });
        }
        let samples = Self::samples(dataset, unseen_subjects, split)?;
        self.evaluate_samples(&samples, provider)
    }

    fn evaluate_samples<P: ExactLegalityProvider>(
        &self,
        samples: &[SubjectSample],
        provider: &mut P,
    ) -> Result<GeneratorEpochReport, GeneratorTrainingError> {
        let mut held_out = GeneratorLossSummary::default();
        let mut prior = GeneratorLossSummary::default();
        let mut marginal_validation_nll = 0.0;
        let mut posterior_action_variance = 0.0;
        let mut total_kl = 0.0;
        let mut total_std = 0.0;
        let mut std_count = 0_usize;
        let mut first_context: Option<(GeneratorObservation, LegalActionSpace, BehaviourProfile)> =
            None;
        let mut profiles = Vec::with_capacity(samples.len());
        let mut eval_noise = self.noise;
        let mut calibration_count = [0_usize; 10];
        let mut calibration_confidence = [0.0_f64; 10];
        let mut calibration_correct = [0_usize; 10];
        let zero = self
            .model
            .latent_config()
            .profile(vec![0.0; self.model.latent_config().dimension()])
            .map_err(|_| GeneratorTrainingError::InvalidLatentDimensions)?;
        for sample in samples {
            let posterior = self.analyzer.infer(&sample.support)?;
            let profile = posterior.mean_profile()?;
            profiles.push(profile.clone());
            total_kl += posterior.kl_to_standard_normal();
            for &std in posterior.std() {
                total_std += std;
                std_count += 1;
            }
            for hand in &sample.validation {
                for row in subject_decisions(hand) {
                    let observation = decode_decision(hand, row)?;
                    let legal = lookup(provider, sample.id, hand, row)?;
                    held_out.record(&self.generator_adam.evaluate_encoded(
                        &self.model,
                        &profile,
                        &observation,
                        &legal,
                    )?)?;
                    let probabilities =
                        self.model
                            .action_probabilities_encoded(&profile, &observation, &legal)?;
                    let (top, confidence) = probabilities
                        .iter()
                        .copied()
                        .enumerate()
                        .max_by(|(_, x), (_, y)| x.total_cmp(y))
                        .ok_or(GeneratorTrainingError::NoDecisions)?;
                    let bin = ((confidence * 10.0).floor() as usize).min(9);
                    calibration_count[bin] += 1;
                    calibration_confidence[bin] += confidence;
                    if top == usize::from(action_id(observation.observed.kind)) {
                        calibration_correct[bin] += 1;
                    }
                    prior.record(&self.generator_adam.evaluate_encoded(
                        &self.model,
                        &zero,
                        &observation,
                        &legal,
                    )?)?;
                    let mut observed_nll = Vec::with_capacity(self.config.validation_draws);
                    let mut average_action = [0.0_f64; ACTION_COUNT];
                    let mut squared_action = [0.0_f64; ACTION_COUNT];
                    for _ in 0..self.config.validation_draws {
                        let eps = eval_noise.sample(self.model.latent_config().dimension());
                        let z = posterior.sample(&eps)?;
                        observed_nll.push(
                            self.generator_adam
                                .evaluate_encoded(&self.model, &z, &observation, &legal)?
                                .total,
                        );
                        let policy =
                            self.model
                                .action_probabilities_encoded(&z, &observation, &legal)?;
                        for i in 0..ACTION_COUNT {
                            average_action[i] += policy[i] / self.config.validation_draws as f64;
                            squared_action[i] +=
                                policy[i] * policy[i] / self.config.validation_draws as f64;
                        }
                    }
                    posterior_action_variance += (0..ACTION_COUNT)
                        .map(|i| {
                            (squared_action[i] - average_action[i] * average_action[i]).max(0.0)
                        })
                        .sum::<f64>()
                        / ACTION_COUNT as f64;
                    let smallest = observed_nll
                        .iter()
                        .copied()
                        .reduce(f64::min)
                        .ok_or(GeneratorTrainingError::NoDecisions)?;
                    let mixture: f64 = observed_nll.iter().map(|v| (smallest - *v).exp()).sum();
                    marginal_validation_nll +=
                        smallest - (mixture / self.config.validation_draws as f64).ln();
                    if first_context.is_none() {
                        first_context = Some((observation, legal, profile.clone()));
                    }
                }
            }
        }
        let (observation, legal, first_profile) =
            first_context.ok_or(GeneratorTrainingError::NoDecisions)?;
        let first_policy =
            self.model
                .action_probabilities_encoded(&first_profile, &observation, &legal)?;
        let policy_diversity_js = if profiles.len() > 1 {
            let mut distances = 0.0;
            let mut comparisons = 0_usize;
            let policies = profiles
                .iter()
                .map(|z| {
                    self.model
                        .action_probabilities_encoded(z, &observation, &legal)
                })
                .collect::<Result<Vec<_>, _>>()?;
            for i in 0..policies.len() {
                for j in i + 1..policies.len() {
                    distances += js_divergence(&policies[i], &policies[j])?;
                    comparisons += 1;
                }
            }
            Some(distances / comparisons as f64)
        } else {
            None
        };
        let mut shifted = first_profile.into_vec();
        shifted[0] = (f64::from(shifted[0]) + self.config.continuity_delta) as f32;
        if !shifted[0].is_finite() {
            return Err(GeneratorTrainingError::NonFiniteLoss);
        }
        let shifted = self
            .model
            .latent_config()
            .profile(shifted)
            .map_err(|_| GeneratorTrainingError::NonFiniteLoss)?;
        let shifted_policy =
            self.model
                .action_probabilities_encoded(&shifted, &observation, &legal)?;
        let continuity_js = js_divergence(&first_policy, &shifted_policy)?;
        if !marginal_validation_nll.is_finite()
            || !posterior_action_variance.is_finite()
            || !total_kl.is_finite()
            || !total_std.is_finite()
            || std_count == 0
        {
            return Err(GeneratorTrainingError::NonFiniteLoss);
        }
        let action_ece = (0..10)
            .filter(|&i| calibration_count[i] > 0)
            .map(|i| {
                let count = calibration_count[i] as f64;
                (calibration_confidence[i] / count - calibration_correct[i] as f64 / count).abs()
                    * count
                    / held_out.decisions as f64
            })
            .sum::<f64>();
        Ok(GeneratorEpochReport {
            epoch: self.epochs,
            subjects: samples.len(),
            train: GeneratorLossSummary::default(),
            held_out,
            prior,
            marginal_validation_nll,
            mean_kl: total_kl / samples.len() as f64,
            posterior_std_mean: total_std / std_count as f64,
            posterior_action_variance: posterior_action_variance / held_out.decisions as f64,
            action_ece,
            policy_diversity_js,
            continuity_js,
        })
    }

    pub fn save_checkpoint(&self, path: impl AsRef<Path>) -> Result<(), GeneratorTrainingError> {
        if self.epochs == 0 {
            return Err(GeneratorTrainingError::NotTrained);
        }
        let value = json!({"format":"lechiffre.ag.checkpoint","generator":self.model.checkpoint_value(),
            "analyzer":self.analyzer.checkpoint_value(),"generator_adam":self.generator_adam.checkpoint_value(),
            "analyzer_adam":self.analyzer_adam.checkpoint_value(),"config":self.config.checkpoint_value(),
            "rng_state":self.noise.state,"epochs":self.epochs,
            "split":self.last_split.ok_or(GeneratorTrainingError::NotTrained)?.checkpoint_value(),
            "subjects":self.last_subjects,
            "dataset_fingerprint":self.dataset_fingerprint.as_ref().ok_or(GeneratorTrainingError::NotTrained)?});
        let destination = path.as_ref();
        if destination.exists() {
            return Err(GeneratorTrainingError::CheckpointAlreadyExists(
                destination.to_path_buf(),
            ));
        }
        let mut temporary = destination.as_os_str().to_os_string();
        temporary.push(".inprogress");
        let temporary = PathBuf::from(temporary);
        let result = (|| -> Result<(), GeneratorTrainingError> {
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temporary)?;
            serde_json::to_writer_pretty(&mut file, &value)?;
            file.write_all(b"\n")?;
            file.sync_all()?;
            // Hard-link creates the destination atomically and never overwrites it.
            fs::hard_link(&temporary, destination)?;
            fs::remove_file(&temporary)?;
            Ok(())
        })();
        if result.is_err() {
            let _ = fs::remove_file(&temporary);
        }
        result
    }

    pub fn load_checkpoint(path: impl AsRef<Path>) -> Result<Self, GeneratorTrainingError> {
        let value: Value = serde_json::from_reader(File::open(path)?)?;
        let error = GeneratorTrainingError::InvalidCheckpoint;
        exact_keys(
            &value,
            &[
                "format",
                "generator",
                "analyzer",
                "generator_adam",
                "analyzer_adam",
                "config",
                "rng_state",
                "epochs",
                "split",
                "subjects",
                "dataset_fingerprint",
            ],
        )
        .map_err(error)?;
        if value["format"].as_str() != Some("lechiffre.ag.checkpoint") {
            return Err(GeneratorTrainingError::InvalidCheckpoint(
                "checkpoint format mismatch".into(),
            ));
        }
        let model =
            NeuralBehaviourGenerator::from_checkpoint(&value["generator"]).map_err(error)?;
        let analyzer =
            NeuralBehaviourAnalyzer::from_checkpoint(&value["analyzer"]).map_err(error)?;
        let generator_adam =
            Adam::from_checkpoint(&value["generator_adam"], &model).map_err(error)?;
        let analyzer_adam =
            AnalyzerAdam::from_checkpoint(&value["analyzer_adam"], &analyzer).map_err(error)?;
        let config = GeneratorTrainingConfig::from_checkpoint(&value["config"]).map_err(error)?;
        let mut trainer = Self::new(model, analyzer, generator_adam, analyzer_adam, config)?;
        trainer.noise = NoiseStream::new(unsigned(&value["rng_state"]).map_err(error)?)?;
        trainer.epochs = unsigned(&value["epochs"]).map_err(error)?;
        trainer.last_split =
            Some(GeneratorHandSplit::from_checkpoint(&value["split"]).map_err(error)?);
        trainer.last_subjects = value["subjects"]
            .as_array()
            .ok_or_else(|| {
                GeneratorTrainingError::InvalidCheckpoint("invalid subjects array".into())
            })?
            .iter()
            .map(|s| unsigned(s).map_err(GeneratorTrainingError::InvalidCheckpoint))
            .collect::<Result<Vec<_>, _>>()?;
        let digest = value["dataset_fingerprint"]
            .as_str()
            .filter(|s| s.len() == 32 && s.bytes().all(|ch| ch.is_ascii_hexdigit()))
            .ok_or_else(|| {
                GeneratorTrainingError::InvalidCheckpoint("invalid dataset fingerprint".into())
            })?;
        trainer.dataset_fingerprint = Some(digest.to_string());
        if trainer.epochs == 0
            || trainer.last_subjects.is_empty()
            || trainer
                .last_subjects
                .iter()
                .copied()
                .collect::<HashSet<_>>()
                .len()
                != trainer.last_subjects.len()
        {
            return Err(GeneratorTrainingError::InvalidCheckpoint(
                "invalid training progress".into(),
            ));
        }
        Ok(trainer)
    }
}

/// Deterministic FNV-1a fingerprint over every loaded row and subject membership.
/// This detects accidental data replacement across checkpoint resumes; it is not
/// a cryptographic authentication tag.
fn dataset_fingerprint(dataset: &BehaviourParquetDataset) -> String {
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

fn js_divergence(a: &[f64], b: &[f64]) -> Result<f64, GeneratorTrainingError> {
    if a.len() != b.len() || a.is_empty() || a.iter().chain(b).any(|v| !v.is_finite() || *v < 0.0) {
        return Err(GeneratorTrainingError::NonFiniteLoss);
    }
    let mut js = 0.0;
    for (&p, &q) in a.iter().zip(b) {
        let m = (p + q) * 0.5;
        if p > 0.0 {
            js += 0.5 * p * (p / m).ln();
        }
        if q > 0.0 {
            js += 0.5 * q * (q / m).ln();
        }
    }
    if !js.is_finite() {
        return Err(GeneratorTrainingError::NonFiniteLoss);
    }
    Ok(js.max(0.0))
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
) -> Result<LegalActionSpace, GeneratorTrainingError> {
    provider
        .legal_actions(id, hand.hand_id(), hand.sequence_indexes()[row])
        .map_err(|e| GeneratorTrainingError::MissingLegality {
            hand_id: hand.hand_id().clone(),
            sequence_index: hand.sequence_indexes()[row],
            message: e.to_string(),
        })
}

pub(super) fn decode_decision(
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
pub enum GeneratorTrainingError {
    InvalidConfiguration,
    InvalidLatentDimensions,
    CountOverflow,
    NonFiniteLoss,
    NoDecisions,
    EmptySubjectSelection,
    DuplicateSubject {
        subject_id: u64,
    },
    UnknownSubject {
        subject_id: u64,
    },
    DatasetChanged,
    PreviouslyTrainedSubject {
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
    TrainingPartitionChanged,
    NotTrained,
    Generator(GeneratorError),
    Analyzer(AnalyzerError),
    BaselineAdaptation(BaselineAdaptationError),
    InvalidData {
        hand_id: HandId,
        sequence_index: u32,
    },
    MissingLegality {
        hand_id: HandId,
        sequence_index: u32,
        message: String,
    },
    InvalidCheckpoint(String),
    CheckpointAlreadyExists(PathBuf),
    Io(io::Error),
    Json(serde_json::Error),
}
impl fmt::Display for GeneratorTrainingError {
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
            Self::InvalidCheckpoint(message) => write!(f, "checkpoint rejected: {message}"),
            Self::Io(error) => write!(f, "checkpoint I/O: {error}"),
            Self::Json(error) => write!(f, "checkpoint JSON: {error}"),
            other => write!(f, "{other:?}"),
        }
    }
}
impl std::error::Error for GeneratorTrainingError {}
impl From<GeneratorError> for GeneratorTrainingError {
    fn from(err: GeneratorError) -> Self {
        Self::Generator(err)
    }
}
impl From<BaselineAdaptationError> for GeneratorTrainingError {
    fn from(err: BaselineAdaptationError) -> Self {
        Self::BaselineAdaptation(err)
    }
}
impl From<AnalyzerError> for GeneratorTrainingError {
    fn from(err: AnalyzerError) -> Self {
        Self::Analyzer(err)
    }
}
impl From<io::Error> for GeneratorTrainingError {
    fn from(err: io::Error) -> Self {
        Self::Io(err)
    }
}
impl From<serde_json::Error> for GeneratorTrainingError {
    fn from(err: serde_json::Error) -> Self {
        Self::Json(err)
    }
}

#[cfg(test)]
mod tests {
    use super::super::{PersistedHand, SubjectMembership};
    use super::*;
    use crate::behavioural::codebook::{action_id, position_id, street_id};
    use crate::behavioural::{
        BehaviourLatentConfig, EVENT_CONTINUOUS_WIDTH, Position, Street, WagerBounds,
    };
    use crate::datasets::behaviour_parquet::BehaviourParquetDatasetSummary;
    use std::collections::BTreeMap;

    fn dataset() -> BehaviourParquetDataset {
        let mut hands = Vec::new();
        for index in 0_u64..9 {
            let kind = if index.is_multiple_of(2) {
                ActionKind::Check
            } else {
                ActionKind::Bet
            };
            let mut values = [0.0; EVENT_CONTINUOUS_WIDTH];
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
            let mut mask = [1; EVENT_CONTINUOUS_WIDTH];
            mask[ContinuousFeature::RaiseIncrementOverPot as usize] = 0;
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
                continuous: vec![values],
                continuous_present: vec![mask],
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
    fn trainer() -> GeneratorTrainer {
        let latent = BehaviourLatentConfig::new(2).unwrap();
        let g = NeuralBehaviourGenerator::initialize(latent, 8, 42).unwrap();
        let a = NeuralBehaviourAnalyzer::initialize(latent, 8, 19).unwrap();
        let ga = Adam::new(&g, 0.001).unwrap();
        let aa = AnalyzerAdam::new(&a, 0.001).unwrap();
        GeneratorTrainer::new(
            g,
            a,
            ga,
            aa,
            GeneratorTrainingConfig::new(0.01, 23, 4, 0.001).unwrap(),
        )
        .unwrap()
    }
    fn split() -> GeneratorHandSplit {
        GeneratorHandSplit::new(3, 6, 3, 3, 3).unwrap()
    }
    fn legal() -> LegalActionSpace {
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
        }
    }
    #[test]
    fn joint_learning_respects_three_global_partitions() {
        let mut trainer = trainer();
        let before = trainer.model().parameters().clone();
        let mut looked = 0;
        let report = trainer
            .train_epoch(&dataset(), &[42], split(), &mut |_,
                                                           _: &HandId,
                                                           _|
             -> Result<
                _,
                &'static str,
            > {
                looked += 1;
                Ok(legal())
            })
            .unwrap();
        assert_eq!(report.train.decisions, 3);
        assert_eq!(report.held_out.decisions, 3);
        assert_eq!(report.prior.decisions, 3);
        assert_eq!(looked, 6);
        assert_eq!(trainer.generator_optimizer_steps(), 1);
        assert_eq!(trainer.analyzer_optimizer_steps(), 1);
        assert_ne!(trainer.model().parameters(), &before);
        assert!(report.mean_kl.is_finite());
        assert!(report.posterior_std_mean > 0.0);
        assert!(report.posterior_action_variance >= 0.0);
        assert!((0.0..=1.0).contains(&report.action_ece));
        assert!(report.continuity_js >= 0.0);
        assert!(report.policy_diversity_js.is_none());
        assert_eq!(trainer.completed_epochs(), 1);
    }
    #[test]
    fn provider_error_does_not_commit_partial_training() {
        let mut trainer = trainer();
        let before = trainer.model().parameters().clone();
        let error = trainer
            .train_epoch(&dataset(), &[42], split(), &mut |_,
                                                           _: &HandId,
                                                           _|
             -> Result<
                LegalActionSpace,
                &'static str,
            > {
                Err("missing replay")
            })
            .unwrap_err();
        assert!(matches!(
            error,
            GeneratorTrainingError::MissingLegality { .. }
        ));
        assert_eq!(trainer.generator_optimizer_steps(), 0);
        assert_eq!(trainer.analyzer_optimizer_steps(), 0);
        assert_eq!(*trainer.model().parameters(), before);
    }
    #[test]
    fn checkpoint_preserves_exact_optimizer_and_rng_state() {
        let mut trainer = trainer();
        trainer
            .train_epoch(&dataset(), &[42], split(), &mut |_,
                                                           _: &HandId,
                                                           _|
             -> Result<
                _,
                &'static str,
            > {
                Ok(legal())
            })
            .unwrap();
        let path = std::env::temp_dir().join(format!("lechiffre-ag-{}.json", std::process::id()));
        let _ = std::fs::remove_file(&path);
        trainer.save_checkpoint(&path).unwrap();
        let mut resumed = GeneratorTrainer::load_checkpoint(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_eq!(resumed.model().parameters(), trainer.model().parameters());
        assert_eq!(
            resumed.analyzer().parameters(),
            trainer.analyzer().parameters()
        );
        assert_eq!(resumed.generator_optimizer_steps(), 1);
        assert_eq!(resumed.analyzer_optimizer_steps(), 1);
        let report = resumed
            .train_epoch(&dataset(), &[42], split(), &mut |_,
                                                           _: &HandId,
                                                           _|
             -> Result<
                _,
                &'static str,
            > {
                Ok(legal())
            })
            .unwrap();
        assert_eq!(report.epoch, 2);
    }
    #[test]
    fn resume_rejects_changed_dataset_and_leaves_weights_unchanged() {
        let mut trainer = trainer();
        let original = dataset();
        trainer
            .train_epoch(&original, &[42], split(), &mut |_,
                                                          _: &HandId,
                                                          _|
             -> Result<
                _,
                &'static str,
            > {
                Ok(legal())
            })
            .unwrap();
        let mut altered = dataset();
        altered.hands[0].continuous[0][ContinuousFeature::PotBb as usize] = 11.0;
        let before = trainer.model().parameters().clone();
        let failure = trainer
            .train_epoch(&altered, &[42], split(), &mut |_,
                                                         _: &HandId,
                                                         _|
             -> Result<
                _,
                &'static str,
            > { Ok(legal()) })
            .unwrap_err();
        assert!(matches!(failure, GeneratorTrainingError::DatasetChanged));
        assert_eq!(trainer.model().parameters(), &before);
        assert_eq!(trainer.completed_epochs(), 1);
    }
    #[test]
    fn validation_does_not_change_optimizer_state() {
        let mut trainer = trainer();
        let source = dataset();
        trainer
            .train_epoch(&source, &[42], split(), &mut |_,
                                                        _: &HandId,
                                                        _|
             -> Result<
                _,
                &'static str,
            > { Ok(legal()) })
            .unwrap();
        let g_steps = trainer.generator_optimizer_steps();
        let a_steps = trainer.analyzer_optimizer_steps();
        let report = trainer
            .evaluate(&source, &mut |_,
                                     _: &HandId,
                                     _|
             -> Result<_, &'static str> {
                Ok(legal())
            })
            .unwrap();
        assert_eq!(report.held_out.decisions, 3);
        assert_eq!(trainer.generator_optimizer_steps(), g_steps);
        assert_eq!(trainer.analyzer_optimizer_steps(), a_steps);
    }
    #[test]
    fn unseen_subject_evaluation_rejects_trained_identity() {
        let mut trainer = trainer();
        let source = dataset();
        trainer
            .train_epoch(&source, &[42], split(), &mut |_,
                                                        _: &HandId,
                                                        _|
             -> Result<
                _,
                &'static str,
            > { Ok(legal()) })
            .unwrap();
        let result = trainer.evaluate_unseen(
            &source,
            &[42],
            &mut |_, _: &HandId, _| -> Result<_, &'static str> { Ok(legal()) },
        );
        assert!(matches!(
            result,
            Err(GeneratorTrainingError::PreviouslyTrainedSubject { subject_id: 42 })
        ));
    }
    #[test]
    fn z1_uses_only_past_hands_and_never_trains_generator() {
        use crate::behavioural::{
            BaselineAdaptationConfig, BehaviourBlueprint, PlayerBaselineAdapter,
        };
        let source = dataset();
        let trainer = trainer();
        let generator = trainer.model();
        let weights_before = generator.parameters().clone();
        let latent = generator.latent_config();
        let blueprint = BehaviourBlueprint::new(
            latent.profile(vec![0.2; latent.dimension()]).unwrap(),
            vec![1.0; latent.dimension()],
        )
        .unwrap();
        let adapter = PlayerBaselineAdapter::new(
            blueprint.clone(),
            BaselineAdaptationConfig::new(0.1, 0.2).unwrap(),
        )
        .unwrap();
        let mut calls = 0_usize;
        let empty = source
            .adapt_baseline_before(42, 0, generator, &adapter, &mut |_,
                                                                     _: &HandId,
                                                                     _|
             -> Result<
                _,
                &'static str,
            > {
                calls += 1;
                Err("no replay should be required for future hands")
            })
            .unwrap();
        assert_eq!(empty.baseline().observed_hands(), 0);
        assert_eq!(empty.baseline().profile(), blueprint.centre());
        assert_eq!(calls, 0);

        let adapted = source
            .adapt_baseline_before(42, 3, generator, &adapter, &mut |_,
                                                                     _: &HandId,
                                                                     _|
             -> Result<
                _,
                &'static str,
            > {
                calls += 1;
                Ok(legal())
            })
            .unwrap();
        assert_eq!(calls, 3);
        assert_eq!(adapted.baseline().observed_hands(), 3);
        assert_eq!(adapted.observed_decisions(), 3);
        assert_eq!(adapted.last_hand_index(), Some(2));
        assert_eq!(generator.parameters(), &weights_before);

        let mut changed_future = dataset();
        changed_future.hands[7].decision_time_ms[0] = 99999;
        changed_future.hands[7].continuous[0][ContinuousFeature::DecisionTimeMs as usize] = 99999.0;
        let still_same = changed_future
            .adapt_baseline_before(42, 3, generator, &adapter, &mut |_,
                                                                     _: &HandId,
                                                                     _|
             -> Result<
                _,
                &'static str,
            > {
                Ok(legal())
            })
            .unwrap();
        assert_eq!(adapted, still_same);
    }

    #[test]
    fn three_partition_boundaries_must_be_strictly_ordered() {
        assert!(GeneratorHandSplit::new(5, 5, 1, 1, 1).is_err());
        assert!(GeneratorHandSplit::new(5, 6, 1, 0, 1).is_err());
    }
    #[test]
    fn js_is_zero_for_same_policy_and_nonnegative() {
        assert_eq!(js_divergence(&[0.2, 0.8], &[0.2, 0.8]).unwrap(), 0.0);
        assert!(js_divergence(&[0.2, 0.8], &[0.8, 0.2]).unwrap() > 0.0);
    }
}
