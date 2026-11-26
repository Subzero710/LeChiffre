//! Hierarchical Z0/Z1/Z2 training: ordered public observations (ingestion order).
//!
//! Training is deliberately divided into explicit phases. No profile is used
//! before its observations exist, and G/A gradients use one fixed model snapshot
//! per batch. Versioned hierarchical checkpoints reject all obsolete A/G formats.
use crate::behavioural::checkpoint::{dimension, exact_keys, float_array, number, unsigned};
use crate::behavioural::codebook::action_id;

/// A fixed quadrature budget and independent evaluation stream are part of the
/// v3 scoring contract. They never consume the training checkpoint RNG.
pub const Z2_PRIOR_MARGINAL_SAMPLES: usize = 64;
const VALIDATION_NOISE_SEED: u64 = 0x8c0a_f13e_61ad_90b7;

use serde_json::{Value, json};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use super::training_data::{
    ExactLegalityProvider, SubjectSample, TrainingDataError, TrainingEpisodeConfig,
    dataset_fingerprint, decode_decision, lookup, sample_subjects, subject_decisions,
};
use super::{BehaviourParquetDataset, DatasetPartitions, EvaluationPartition};
use crate::behavioural::{
    ACTION_COUNT, Adam, AnalyzerAdam, AnalyzerError, AnalyzerGradientBatch,
    BaselineAdaptationConfig, BaselineAdaptationError, BehaviourBlueprint, BehaviourLatentModel,
    BehaviourProfileError, GeneratorError, GeneratorGradientBatch, GeneratorObservation,
    LegalActionSpace, NeuralBehaviourAnalyzer, NeuralBehaviourGenerator, NoiseStream,
    PlayerBaseline, PlayerBaselineAdapter, PlayerBaselineState,
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
    /// G optimizer steps in this epoch (A steps coincide during Momentary).
    pub optimizer_steps: usize,
    /// Blueprint: Z0 NLL; Personalization: Z1 NLL; Momentary: mean prior/posterior
    /// sampled NLL. Does not include KL.
    pub mean_nll: f64,
    /// The conditional KL is available only for Momentary training.
    pub mean_conditional_kl: Option<f64>,
}

/// Exact chain-rule NLL components on the *same* evaluated decisions.
/// Sizing is conditional on a Bet/Raise and has its own denominator; when no wager is
/// observed, sizing is explicitly absent (not silently assigned zero).
#[derive(Debug, Clone, PartialEq)]
pub struct HierarchicalNllComponents {
    pub action: f64,
    pub sizing: Option<f64>,
    pub timing_log1p_ms: f64,
}

/// Z2 prior is the Monte Carlo marginal likelihood, not G evaluated at E[Z2].
/// Posterior reconstruction is after observing the target, never forecasting.
#[derive(Debug, Clone, PartialEq)]
pub struct HierarchicalValidationReport {
    pub decisions: usize,
    pub wager_decisions: usize,
    pub z2_prior_samples: usize,
    pub history_hands: usize,
    pub z0_action_accuracy: f64,
    pub z1_action_accuracy: f64,
    pub z2_prior_action_accuracy: f64,
    pub z0_predictive_nll: f64,
    pub z1_predictive_nll: f64,
    pub z2_prior_marginal_nll: f64,
    pub z2_posterior_reconstruction_nll: f64,
    pub z0_components: HierarchicalNllComponents,
    pub z1_components: HierarchicalNllComponents,
    pub z2_prior_marginal_components: HierarchicalNllComponents,
    pub mean_conditional_kl: f64,
}

/// Explicit decision mini-batch size for Blueprint and Personalization.
/// Each phase has its own fixed objective.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HierarchicalOptimizationConfig {
    blueprint_batch_decisions: usize,
}

impl HierarchicalOptimizationConfig {
    pub fn new(blueprint_batch_decisions: usize) -> Result<Self, HierarchicalTrainingError> {
        if blueprint_batch_decisions == 0 {
            return Err(HierarchicalTrainingError::InvalidConfiguration);
        }
        Ok(Self {
            blueprint_batch_decisions,
        })
    }

    fn checkpoint_value(self) -> Value {
        json!({"blueprint_batch_decisions":self.blueprint_batch_decisions})
    }

    fn from_checkpoint(v: &Value) -> Result<Self, String> {
        exact_keys(v, &["blueprint_batch_decisions"])?;
        Self::new(dimension(&v["blueprint_batch_decisions"])?).map_err(|e| e.to_string())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ReferenceValidationReport {
    pub decisions: usize,
    pub wager_decisions: usize,
    pub predictive_nll: f64,
    pub components: HierarchicalNllComponents,
    pub action_accuracy: f64,
}

/// Contains one global blueprint, one shared A and G, and their optimizers.
/// Individual Z1 persistence is separate from shared A/G/Z0 checkpoints.
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
    optimization: HierarchicalOptimizationConfig,
    noise: NoiseStream,
    blueprint_epochs: u64,
    personalization_epochs: u64,
    momentary_epochs: u64,
    last_episode: Option<TrainingEpisodeConfig>,
    manifest_fingerprint: Option<String>,
    legality_fingerprint: Option<String>,
    last_subjects: Vec<u64>,
    dataset_fingerprint: Option<String>,
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
        optimization: HierarchicalOptimizationConfig,
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
            optimization,
            noise,
            blueprint_epochs: 0,
            personalization_epochs: 0,
            momentary_epochs: 0,
            last_episode: None,
            manifest_fingerprint: None,
            legality_fingerprint: None,
            last_subjects: Vec::new(),
            dataset_fingerprint: None,
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

    pub fn baseline_config(&self) -> BaselineAdaptationConfig {
        self.baseline_config
    }
    pub fn training_episode(&self) -> Option<TrainingEpisodeConfig> {
        self.last_episode
    }
    pub fn training_subjects(&self) -> &[u64] {
        &self.last_subjects
    }

    fn bind_training_context(
        &mut self,
        dataset: &BehaviourParquetDataset,
        subjects: &[u64],
        partitions: &DatasetPartitions,
        episode: TrainingEpisodeConfig,
    ) -> Result<(), HierarchicalTrainingError> {
        let fingerprint = dataset_fingerprint(dataset);
        let legality = dataset.legality_fingerprint()?.to_owned();
        if self
            .dataset_fingerprint
            .as_ref()
            .is_some_and(|expected| expected != &fingerprint)
        {
            return Err(HierarchicalTrainingError::DatasetChanged);
        }
        partitions
            .verify_dataset(dataset)
            .map_err(|_| HierarchicalTrainingError::TrainingPartitionChanged)?;
        let manifest = partitions.manifest_fingerprint();
        selected_subjects(dataset, subjects)?;
        if let Some(expected) = &self.dataset_fingerprint {
            if expected != &fingerprint {
                return Err(HierarchicalTrainingError::DatasetChanged);
            }
            if self.manifest_fingerprint.as_deref() != Some(manifest)
                || self.legality_fingerprint.as_deref() != Some(&legality)
                || self.last_episode != Some(episode)
                || self.last_subjects != subjects
            {
                return Err(HierarchicalTrainingError::TrainingPartitionChanged);
            }
        } else {
            self.dataset_fingerprint = Some(fingerprint);
            self.manifest_fingerprint = Some(manifest.to_owned());
            self.legality_fingerprint = Some(legality);
            self.last_episode = Some(episode);
            self.last_subjects = subjects.to_vec();
        }
        Ok(())
    }

    /// A persisted Z1 update must use the same dataset and exact legal action
    /// bounds as the global checkpoint, even though it does not need the split.
    pub fn verify_dataset_legality(
        &self,
        dataset: &BehaviourParquetDataset,
    ) -> Result<(), HierarchicalTrainingError> {
        if self.dataset_fingerprint.as_deref() != Some(dataset_fingerprint(dataset).as_str()) {
            return Err(HierarchicalTrainingError::DatasetChanged);
        }
        if self.legality_fingerprint.as_deref() != Some(dataset.legality_fingerprint()?) {
            return Err(HierarchicalTrainingError::TrainingPartitionChanged);
        }
        Ok(())
    }

    fn check_evaluation_dataset(
        &self,
        dataset: &BehaviourParquetDataset,
        partitions: &DatasetPartitions,
    ) -> Result<(), HierarchicalTrainingError> {
        if self.blueprint_epochs == 0 {
            return Err(HierarchicalTrainingError::NotTrained);
        }
        self.verify_dataset_legality(dataset)?;
        partitions
            .verify_dataset(dataset)
            .map_err(|_| HierarchicalTrainingError::TrainingPartitionChanged)?;
        if self.manifest_fingerprint.as_deref() != Some(partitions.manifest_fingerprint()) {
            return Err(HierarchicalTrainingError::TrainingPartitionChanged);
        }
        Ok(())
    }

    /// Stable identifier for the exact shared learned model (Z0, A, G and adaptation policy).
    /// FNV-128 detects accidental model mismatches; it is not a cryptographic signature.
    pub fn model_id(&self) -> Result<String, HierarchicalTrainingError> {
        let data = serde_json::to_vec(&json!({
            "format":"lechiffre.hierarchical.model.v10-shared-bb",
            "z0":self.blueprint.centre().as_slice(),
            "prior_std":self.blueprint.prior_std(),
            "generator":self.generator.checkpoint_value(),
            "analyzer":self.analyzer.checkpoint_value(),
            "baseline_learning_rate":self.baseline_config.learning_rate(),
            "baseline_max_step":self.baseline_config.max_step_prior_std(),
        }))?;
        let mut hash = 0x6c62_272e_07bb_0142_62b8_2175_6295_c58d_u128;
        const PRIME: u128 = 0x0000_0000_0100_0000_0000_0000_0000_013b;
        for byte in data {
            hash = (hash ^ u128::from(byte)).wrapping_mul(PRIME);
        }
        Ok(format!("hierarchical-v10-shared-bb-{hash:032x}"))
    }

    /// Hierarchical-only checkpoint. Exact model, optimizer, RNG, phase, and dataset provenance.
    pub fn save_checkpoint(&self, path: impl AsRef<Path>) -> Result<(), HierarchicalTrainingError> {
        if self.blueprint_epochs == 0 {
            return Err(HierarchicalTrainingError::NotTrained);
        }
        let value = json!({
            "format":"lechiffre.hierarchical.checkpoint.v10-shared-bb",
            "model_id":self.model_id()?,
            "blueprint":{"centre":self.blueprint.centre().as_slice(),
                         "prior_std":self.blueprint.prior_std()},
            "generator":self.generator.checkpoint_value(),
            "analyzer":self.analyzer.checkpoint_value(),
            "generator_adam":self.generator_adam.checkpoint_value(),
            "analyzer_adam":self.analyzer_adam.checkpoint_value(),
            "baseline_config":{"learning_rate":self.baseline_config.learning_rate(),
                               "max_step_prior_std":self.baseline_config.max_step_prior_std()},
            "blueprint_learning_rate":self.blueprint_learning_rate,
            "beta_kl":self.beta_kl,
            "optimization":self.optimization.checkpoint_value(),
            "rng_state":self.noise.state,
            "epochs":{"blueprint":self.blueprint_epochs,"personalization":self.personalization_epochs,
                      "momentary":self.momentary_epochs},
            "training_episode":self.last_episode.ok_or(HierarchicalTrainingError::NotTrained)?.checkpoint_value(),
            "manifest_fingerprint":self.manifest_fingerprint.as_ref().ok_or(HierarchicalTrainingError::NotTrained)?,
            "legality_fingerprint":self.legality_fingerprint.as_ref().ok_or(HierarchicalTrainingError::NotTrained)?,
            "subjects":self.last_subjects,
            "dataset_fingerprint":self.dataset_fingerprint.as_ref().ok_or(HierarchicalTrainingError::NotTrained)?,
        });
        let destination = path.as_ref();
        if destination.exists() {
            return Err(HierarchicalTrainingError::CheckpointAlreadyExists(
                destination.to_path_buf(),
            ));
        }
        let mut temporary = destination.as_os_str().to_os_string();
        temporary.push(".inprogress");
        let temporary = PathBuf::from(temporary);
        let result = (|| -> Result<(), HierarchicalTrainingError> {
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temporary)?;
            serde_json::to_writer_pretty(&mut file, &value)?;
            file.write_all(b"\n")?;
            file.sync_all()?;
            fs::hard_link(&temporary, destination)?;
            fs::remove_file(&temporary)?;
            Ok(())
        })();
        if result.is_err() {
            let _ = fs::remove_file(&temporary);
        }
        result
    }

    pub fn load_checkpoint(path: impl AsRef<Path>) -> Result<Self, HierarchicalTrainingError> {
        let value: Value = serde_json::from_reader(File::open(path)?)?;
        let bad = HierarchicalTrainingError::InvalidCheckpoint;
        exact_keys(
            &value,
            &[
                "format",
                "model_id",
                "blueprint",
                "generator",
                "analyzer",
                "generator_adam",
                "analyzer_adam",
                "baseline_config",
                "blueprint_learning_rate",
                "beta_kl",
                "optimization",
                "rng_state",
                "epochs",
                "training_episode",
                "manifest_fingerprint",
                "legality_fingerprint",
                "subjects",
                "dataset_fingerprint",
            ],
        )
        .map_err(bad)?;
        let expected_format = "lechiffre.hierarchical.checkpoint.v10-shared-bb";
        if value["format"].as_str() != Some(expected_format) {
            return Err(bad(format!("expected {expected_format}")));
        }
        exact_keys(&value["blueprint"], &["centre", "prior_std"]).map_err(bad)?;
        exact_keys(
            &value["baseline_config"],
            &["learning_rate", "max_step_prior_std"],
        )
        .map_err(bad)?;
        exact_keys(
            &value["epochs"],
            &["blueprint", "personalization", "momentary"],
        )
        .map_err(bad)?;
        let generator =
            NeuralBehaviourGenerator::from_checkpoint(&value["generator"]).map_err(bad)?;
        let analyzer = NeuralBehaviourAnalyzer::from_checkpoint(&value["analyzer"]).map_err(bad)?;
        let config = generator.latent_config();
        let floats = |field: &Value| -> Result<Vec<f32>, HierarchicalTrainingError> {
            float_array(field)
                .map_err(bad)?
                .into_iter()
                .map(|v| {
                    let v = v as f32;
                    if !v.is_finite() {
                        Err(bad("latent value outside f32 range".into()))
                    } else {
                        Ok(v)
                    }
                })
                .collect()
        };
        let blueprint = BehaviourBlueprint::new(
            config.profile(floats(&value["blueprint"]["centre"])?)?,
            floats(&value["blueprint"]["prior_std"])?,
        )?;
        let gopt = Adam::from_checkpoint(&value["generator_adam"], &generator).map_err(bad)?;
        let aopt =
            AnalyzerAdam::from_checkpoint(&value["analyzer_adam"], &analyzer).map_err(bad)?;
        let baseline_config = BaselineAdaptationConfig::new(
            number(&value["baseline_config"]["learning_rate"]).map_err(bad)?,
            number(&value["baseline_config"]["max_step_prior_std"]).map_err(bad)?,
        )?;
        let optimization =
            HierarchicalOptimizationConfig::from_checkpoint(&value["optimization"]).map_err(bad)?;
        let mut trainer = Self::new(
            blueprint,
            generator,
            analyzer,
            gopt,
            aopt,
            baseline_config,
            number(&value["blueprint_learning_rate"]).map_err(bad)?,
            number(&value["beta_kl"]).map_err(bad)?,
            optimization,
            unsigned(&value["rng_state"]).map_err(bad)?,
        )?;
        let epochs = &value["epochs"];
        trainer.blueprint_epochs = unsigned(&epochs["blueprint"]).map_err(bad)?;
        trainer.personalization_epochs = unsigned(&epochs["personalization"]).map_err(bad)?;
        trainer.momentary_epochs = unsigned(&epochs["momentary"]).map_err(bad)?;
        trainer.last_episode =
            Some(TrainingEpisodeConfig::from_checkpoint(&value["training_episode"]).map_err(bad)?);
        trainer.manifest_fingerprint = Some(
            value["manifest_fingerprint"]
                .as_str()
                .filter(|s| s.len() == 32 && s.bytes().all(|c| c.is_ascii_hexdigit()))
                .ok_or_else(|| bad("invalid manifest fingerprint".into()))?
                .to_owned(),
        );
        trainer.legality_fingerprint = Some(
            value["legality_fingerprint"]
                .as_str()
                .filter(|s| s.len() == 32 && s.bytes().all(|c| c.is_ascii_hexdigit()))
                .ok_or_else(|| bad("invalid exact legality fingerprint".into()))?
                .to_owned(),
        );
        trainer.last_subjects = value["subjects"]
            .as_array()
            .ok_or_else(|| bad("subjects must be an array".into()))?
            .iter()
            .map(|v| unsigned(v).map_err(bad))
            .collect::<Result<Vec<_>, _>>()?;
        trainer.dataset_fingerprint = Some(
            value["dataset_fingerprint"]
                .as_str()
                .filter(|s| s.len() == 32 && s.bytes().all(|c| c.is_ascii_hexdigit()))
                .ok_or_else(|| bad("invalid dataset fingerprint".into()))?
                .to_owned(),
        );
        if trainer.blueprint_epochs == 0
            || (trainer.momentary_epochs > 0 && trainer.personalization_epochs == 0)
            || trainer.last_subjects.is_empty()
            || trainer
                .last_subjects
                .iter()
                .copied()
                .collect::<HashSet<_>>()
                .len()
                != trainer.last_subjects.len()
        {
            return Err(bad("inconsistent training progress".into()));
        }
        if value["model_id"].as_str() != Some(trainer.model_id()?.as_str()) {
            return Err(bad("shared model fingerprint mismatch".into()));
        }
        Ok(trainer)
    }

    /// Population pretraining: only globally verified Train hands contribute. Update Z0 as an explicit learned latent vector alongside G.
    /// The prior standard deviation remains an explicitly supplied hyperparameter,
    /// not a calibrated population dispersion estimate.
    pub fn train_blueprint_epoch<P: ExactLegalityProvider>(
        &mut self,
        dataset: &BehaviourParquetDataset,
        subjects: &[u64],
        partitions: &DatasetPartitions,
        episode: TrainingEpisodeConfig,
        provider: &mut P,
    ) -> Result<HierarchicalEpochReport, HierarchicalTrainingError> {
        if self.personalization_epochs > 0 || self.momentary_epochs > 0 {
            return Err(HierarchicalTrainingError::TrainingPhaseCompleted);
        }
        let mut working = self.clone();
        working.bind_training_context(dataset, subjects, partitions, episode)?;
        let mut memberships = dataset
            .subjects
            .iter()
            .flat_map(|(&id, rows)| rows.iter().map(move |row| (id, row)))
            .filter(|(_, row)| partitions.contains_train(dataset.hands[row.hand_slot].hand_index))
            .collect::<Vec<_>>();
        memberships
            .sort_unstable_by_key(|(id, row)| (dataset.hands[row.hand_slot].hand_index, *id));
        let mut batch = GeneratorGradientBatch::new(&working.generator);
        let mut decisions = 0usize;
        let mut optimizer_steps = 0usize;
        let mut nll_sum = 0.0;
        for (id, membership) in memberships {
            let hand = &dataset.hands[membership.hand_slot];
            let observed = hand.materialize(membership.subject_position);
            for row in subject_decisions(&observed) {
                let observation = decode_decision(&observed, row)?;
                let legal = lookup(provider, id, &observed, row)?;
                batch.accumulate_snapshot_encoded(
                    working.blueprint.centre(),
                    &observation,
                    &legal,
                )?;
                if batch.len() == working.optimization.blueprint_batch_decisions {
                    decisions += batch.len();
                    nll_sum += working.apply_blueprint_batch(batch)?;
                    optimizer_steps += 1;
                    batch = GeneratorGradientBatch::new(&working.generator);
                }
            }
        }
        if !batch.is_empty() {
            decisions += batch.len();
            nll_sum += working.apply_blueprint_batch(batch)?;
            optimizer_steps += 1;
        }
        if decisions == 0 {
            return Err(HierarchicalTrainingError::EmptyTraining);
        }
        if !nll_sum.is_finite() {
            return Err(HierarchicalTrainingError::NonFiniteUpdate);
        }
        let expected_decisions: usize = dataset
            .hands
            .iter()
            .filter(|hand| partitions.contains_train(hand.hand_index))
            .map(|hand| hand.sequence_indexes.len())
            .sum();
        if decisions != expected_decisions {
            return Err(HierarchicalTrainingError::PopulationCoverageMismatch {
                expected: expected_decisions,
                observed: decisions,
            });
        }
        working.blueprint_epochs = working
            .blueprint_epochs
            .checked_add(1)
            .ok_or(HierarchicalTrainingError::CountOverflow)?;
        *self = working;
        Ok(HierarchicalEpochReport {
            phase: HierarchicalPhase::Blueprint,
            decisions,
            optimizer_steps,
            mean_nll: nll_sum / decisions as f64,
            mean_conditional_kl: None,
        })
    }

    /// One coherent G/Z0 step. Called only on the private epoch copy.
    fn apply_blueprint_batch(
        &mut self,
        batch: GeneratorGradientBatch,
    ) -> Result<f64, HierarchicalTrainingError> {
        let mean = batch.mean_loss()?;
        let mut centre = Vec::with_capacity(self.blueprint.centre().dimension());
        for (&value, &gradient) in self
            .blueprint
            .centre()
            .as_slice()
            .iter()
            .zip(&mean.latent_gradient)
        {
            let next = f64::from(value) - self.blueprint_learning_rate * f64::from(gradient);
            if !next.is_finite() || next.abs() > f64::from(f32::MAX) {
                return Err(HierarchicalTrainingError::NonFiniteUpdate);
            }
            centre.push(next as f32);
        }
        let next_centre = self.generator.latent_config().profile(centre)?;
        let next_blueprint =
            BehaviourBlueprint::new(next_centre, self.blueprint.prior_std().to_vec())?;
        let nll_sum = mean.total * batch.len() as f64;
        self.generator_adam
            .apply_batch(&mut self.generator, batch)?;
        self.blueprint = next_blueprint;
        Ok(nll_sum)
    }

    /// Online personalization on disjoint trajectories of `support_hands`
    /// exploitable Train hands per subject. Reset the complete adaptation state
    /// to Z0 at each trajectory boundary, including at the start of each epoch.
    /// Each hand is scored with its pre-hand Z1 and adapted once after its last
    /// decision, before Adam, using that decision's G snapshot. Keep the state
    /// across G updates within each trajectory. No replay or gradient through Z1.
    pub fn train_personalization_epoch<P: ExactLegalityProvider>(
        &mut self,
        dataset: &BehaviourParquetDataset,
        subjects: &[u64],
        partitions: &DatasetPartitions,
        episode: TrainingEpisodeConfig,
        provider: &mut P,
    ) -> Result<HierarchicalEpochReport, HierarchicalTrainingError> {
        if self.blueprint_epochs == 0 || self.momentary_epochs > 0 {
            return Err(HierarchicalTrainingError::InvalidPhaseOrHistory);
        }
        let trajectory_hands = u64::try_from(episode.support_hands)
            .map_err(|_| HierarchicalTrainingError::CountOverflow)?;
        if trajectory_hands == 0 {
            return Err(HierarchicalTrainingError::InvalidConfiguration);
        }
        let mut working = self.clone();
        working.bind_training_context(dataset, subjects, partitions, episode)?;
        let adapter =
            PlayerBaselineAdapter::new(working.blueprint.clone(), working.baseline_config)?;
        let mut states = HashMap::<u64, PlayerBaselineState>::new();
        let mut memberships = Vec::new();
        for &id in subjects {
            states.insert(id, adapter.initialize_player());
            for membership in &dataset.subjects[&id] {
                let hand = &dataset.hands[membership.hand_slot];
                if partitions.contains_train(hand.hand_index) {
                    memberships.push((hand.hand_index, id, membership));
                }
            }
        }
        memberships.sort_unstable_by_key(|&(index, id, _)| (index, id));
        for pair in memberships.windows(2) {
            if (pair[0].0, pair[0].1) == (pair[1].0, pair[1].1) {
                return Err(HierarchicalTrainingError::DuplicateTrainingHand {
                    hand_index: pair[0].0,
                    subject_id: pair[0].1,
                });
            }
        }
        let mut batch = GeneratorGradientBatch::new(&working.generator);
        let mut count = 0usize;
        let mut sum_nll = 0.0;
        let mut optimizer_steps = 0;
        for (hand_index, id, membership) in memberships {
            let hand = dataset.hands[membership.hand_slot].materialize(membership.subject_position);
            let mut observations = Vec::new();
            let mut rows = subject_decisions(&hand).collect::<Vec<_>>();
            rows.sort_unstable_by_key(|&row| hand.sequence_indexes()[row]);
            for pair in rows.windows(2) {
                if hand.sequence_indexes()[pair[0]] == hand.sequence_indexes()[pair[1]] {
                    return Err(HierarchicalTrainingError::DuplicateQueryDecision {
                        hand_index,
                        subject_id: id,
                        sequence_index: hand.sequence_indexes()[pair[0]],
                    });
                }
            }
            for row in rows {
                observations.push((
                    decode_decision(&hand, row)?,
                    lookup(provider, id, &hand, row)?,
                ));
            }
            if observations.is_empty() {
                continue;
            }
            let state = states
                .get_mut(&id)
                .ok_or(HierarchicalTrainingError::UnknownSubject(id))?;
            // Reset only before the next exploitable hand, never mid-hand or
            // at a batch boundary. This also preserves the final partial trajectory.
            if state.baseline().observed_hands() == trajectory_hands {
                *state = adapter.initialize_player();
            }
            for (i, (observation, legal)) in observations.iter().enumerate() {
                let loss = batch.accumulate_snapshot_encoded(
                    state.baseline().profile(),
                    observation,
                    legal,
                )?;
                sum_nll += loss.total;
                count = count
                    .checked_add(1)
                    .ok_or(HierarchicalTrainingError::CountOverflow)?;
                // The complete hand is only evidence after all its decisions
                // have been scored. If a hand crosses batches, adapt with G
                // from the batch containing its final decision.
                if i + 1 == observations.len() {
                    adapter.adapt_hand(state, &working.generator, hand_index, &observations)?;
                }
                if batch.len() == working.optimization.blueprint_batch_decisions {
                    working
                        .generator_adam
                        .apply_batch(&mut working.generator, batch)?;
                    optimizer_steps += 1;
                    batch = GeneratorGradientBatch::new(&working.generator);
                }
            }
        }
        if !batch.is_empty() {
            working
                .generator_adam
                .apply_batch(&mut working.generator, batch)?;
            optimizer_steps += 1;
        }
        if count == 0 || states.values().any(|state| state.observed_decisions() == 0) {
            return Err(HierarchicalTrainingError::EmptyTraining);
        }
        if !sum_nll.is_finite() {
            return Err(HierarchicalTrainingError::NonFiniteUpdate);
        }
        working.personalization_epochs = working
            .personalization_epochs
            .checked_add(1)
            .ok_or(HierarchicalTrainingError::CountOverflow)?;
        *self = working;
        Ok(HierarchicalEpochReport {
            phase: HierarchicalPhase::Personalization,
            decisions: count,
            optimizer_steps,
            mean_nll: sum_nll / count as f64,
            mean_conditional_kl: None,
        })
    }

    /// Decision-level variational learning: prior Z2 sees only C/Z1, posterior
    /// Z2 sees C/ObservedAction/Z1. G learns only the two Z2 branches under
    /// the complete action+sizing+timing likelihood;
    /// A receives both reverse-mode gradients
    /// and conditional KL. Z1 is fitted only on the preceding support hands.
    pub fn train_momentary_epoch<P: ExactLegalityProvider>(
        &mut self,
        dataset: &BehaviourParquetDataset,
        subjects: &[u64],
        partitions: &DatasetPartitions,
        episode: TrainingEpisodeConfig,
        history_hands: usize,
        provider: &mut P,
    ) -> Result<HierarchicalEpochReport, HierarchicalTrainingError> {
        if self.blueprint_epochs == 0
            || self.personalization_epochs == 0
            || history_hands > episode.support_hands
        {
            return Err(HierarchicalTrainingError::InvalidPhaseOrHistory);
        }
        let samples = sample_subjects(dataset, subjects, partitions, episode)?;
        let mut working = self.clone();
        working.bind_training_context(dataset, subjects, partitions, episode)?;
        let mut nll_sum = 0.0;
        let mut kl_sum = 0.0;
        let mut decisions = 0usize;
        for sample in &samples {
            let baseline =
                working.personalize(sample.id, &sample.support, history_hands, provider)?;
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
            optimizer_steps: samples.len(),
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

    /// Fit Z1 only from the explicitly provided, ordered support history.
    fn personalize<P: ExactLegalityProvider>(
        &self,
        subject: u64,
        hands: &[(u64, crate::behavioural::FeaturizedHand)],
        count: usize,
        provider: &mut P,
    ) -> Result<PlayerBaseline, HierarchicalTrainingError> {
        let adapter = PlayerBaselineAdapter::new(self.blueprint.clone(), self.baseline_config)?;
        let mut state = adapter.initialize_player();
        if hands.len() < count {
            return Err(HierarchicalTrainingError::InsufficientHistory {
                subject,
                required: count,
                available: hands.len(),
            });
        }
        for (hand_index, hand) in &hands[hands.len() - count..] {
            let mut decisions = Vec::new();
            for row in subject_decisions(hand) {
                decisions.push((
                    decode_decision(hand, row)?,
                    lookup(provider, subject, hand, row)?,
                ));
            }
            if !decisions.is_empty() {
                adapter.adapt_hand(&mut state, &self.generator, *hand_index, &decisions)?;
            }
        }
        Ok(state.baseline().clone())
    }

    /// Score the population reference before A or personalization are trained.
    /// Use the final evaluation cohort for comparable stage-to-stage scores.
    pub fn evaluate_blueprint<P: ExactLegalityProvider>(
        &self,
        dataset: &BehaviourParquetDataset,
        partitions: &DatasetPartitions,
        provider: &mut P,
    ) -> Result<ReferenceValidationReport, HierarchicalTrainingError> {
        self.evaluate_reference(dataset, partitions, None, provider)
    }

    /// Score a fresh Z1 using a frozen G and only exploitable Train hands.
    /// A trained Blueprint is sufficient; neither Personalization nor Momentary
    /// training is required. This is evaluation only, never an optimizer step.
    pub fn evaluate_personalization<P: ExactLegalityProvider>(
        &self,
        dataset: &BehaviourParquetDataset,
        partitions: &DatasetPartitions,
        history_hands: usize,
        provider: &mut P,
    ) -> Result<ReferenceValidationReport, HierarchicalTrainingError> {
        self.evaluate_reference(dataset, partitions, Some(history_hands), provider)
    }

    fn evaluation_baseline<P: ExactLegalityProvider>(
        &self,
        dataset: &BehaviourParquetDataset,
        partitions: &DatasetPartitions,
        id: u64,
        history_hands: usize,
        provider: &mut P,
    ) -> Result<PlayerBaseline, HierarchicalTrainingError> {
        let memberships = dataset
            .subjects
            .get(&id)
            .ok_or(HierarchicalTrainingError::UnknownSubject(id))?;
        let mut train_history = memberships
            .iter()
            .filter_map(|membership| {
                let hand = &dataset.hands[membership.hand_slot];
                partitions.contains_train(hand.hand_index).then(|| {
                    (
                        hand.hand_index,
                        hand.materialize(membership.subject_position),
                    )
                })
            })
            .collect::<Vec<_>>();
        // Count actual exploitable hands, not memberships without decisions.
        train_history.retain(|(_, hand)| subject_decisions(hand).next().is_some());
        train_history.sort_unstable_by_key(|(index, _)| *index);
        self.personalize(id, &train_history, history_hands, provider)
    }

    fn evaluate_reference<P: ExactLegalityProvider>(
        &self,
        dataset: &BehaviourParquetDataset,
        partitions: &DatasetPartitions,
        history_hands: Option<usize>,
        provider: &mut P,
    ) -> Result<ReferenceValidationReport, HierarchicalTrainingError> {
        self.check_evaluation_dataset(dataset, partitions)?;
        selected_subjects(dataset, &self.last_subjects)?;
        let mut total = 0.0;
        let mut action = 0.0;
        let mut sizing = 0.0;
        let mut timing = 0.0;
        let mut decisions = 0usize;
        let mut wagers = 0usize;
        let mut correct = 0usize;
        for &id in &self.last_subjects {
            let profile = match history_hands {
                None => self.blueprint.centre().clone(),
                Some(count) => self
                    .evaluation_baseline(dataset, partitions, id, count, provider)?
                    .profile()
                    .clone(),
            };
            let mut subject_count = 0usize;
            for membership in &dataset.subjects[&id] {
                let hand = &dataset.hands[membership.hand_slot];
                if !partitions.contains_evaluation(EvaluationPartition::Validation, hand.hand_index)
                {
                    continue;
                }
                let observed = hand.materialize(membership.subject_position);
                for row in subject_decisions(&observed) {
                    let observation = decode_decision(&observed, row)?;
                    let legal = lookup(provider, id, &observed, row)?;
                    let (loss, probs) = self
                        .generator
                        .score_observation_with_probabilities_encoded(
                            &profile,
                            &observation,
                            &legal,
                        )?;
                    if predicted_action(&probs) == usize::from(action_id(observation.observed.kind))
                    {
                        correct += 1;
                    }
                    total += loss.total;
                    action += loss.action;
                    timing += loss.timing;
                    if matches!(
                        observation.observed.kind,
                        crate::behavioural::ActionKind::Bet | crate::behavioural::ActionKind::Raise
                    ) {
                        sizing += loss.sizing;
                        wagers += 1;
                    }
                    decisions += 1;
                    subject_count += 1;
                }
            }
            if subject_count == 0 {
                return Err(HierarchicalTrainingError::EmptyEvaluationSubject(
                    id,
                    EvaluationPartition::Validation,
                ));
            }
        }
        if decisions == 0 {
            return Err(HierarchicalTrainingError::EmptyValidation);
        }
        if ![total, action, sizing, timing]
            .iter()
            .all(|x| x.is_finite())
        {
            return Err(HierarchicalTrainingError::NonFiniteUpdate);
        }
        Ok(ReferenceValidationReport {
            decisions,
            wager_decisions: wagers,
            predictive_nll: total / decisions as f64,
            components: HierarchicalNllComponents {
                action: action / decisions as f64,
                sizing: (wagers > 0).then(|| sizing / wagers as f64),
                timing_log1p_ms: timing / decisions as f64,
            },
            action_accuracy: correct as f64 / decisions as f64,
        })
    }

    /// Model parameters and evaluation baselines are read-only. Z1 uses only
    /// full hands inside Train; neither Validation nor Test may adapt the player.
    /// In particular Test cannot consume Validation as support.
    pub fn evaluate<P: ExactLegalityProvider>(
        &self,
        dataset: &BehaviourParquetDataset,
        subjects: &[u64],
        partitions: &DatasetPartitions,
        partition: EvaluationPartition,
        provider: &mut P,
    ) -> Result<HierarchicalValidationReport, HierarchicalTrainingError> {
        let history_hands = self
            .last_episode
            .ok_or(HierarchicalTrainingError::NotTrained)?
            .support_hands;
        self.evaluate_with_history(
            dataset,
            subjects,
            partitions,
            partition,
            history_hands,
            provider,
        )
    }

    /// Compare explicit evidence budgets on the same held-out decisions. All
    /// shared parameters and per-subject evaluation baselines remain fixed.
    pub fn evaluate_with_history<P: ExactLegalityProvider>(
        &self,
        dataset: &BehaviourParquetDataset,
        subjects: &[u64],
        partitions: &DatasetPartitions,
        partition: EvaluationPartition,
        history_hands: usize,
        provider: &mut P,
    ) -> Result<HierarchicalValidationReport, HierarchicalTrainingError> {
        if self.momentary_epochs == 0 {
            return Err(HierarchicalTrainingError::NotTrained);
        }
        self.check_evaluation_dataset(dataset, partitions)?;
        selected_subjects(dataset, subjects)?;
        let mut correct = [0usize; 3];
        let mut sum = [0.0_f64; 4]; // z0, z1, marginal prior, posterior reconstruction
        let mut component_sums = [[0.0_f64; 3]; 3]; // z0, z1, marginal prior
        let mut kl = 0.0;
        let mut count = 0usize;
        let mut wager_count = 0usize;
        // The same standard-normal draws are used across candidate checkpoints.
        // This stream never touches self.noise and cannot change a resumed run.
        let mut eval_noise = NoiseStream::new(VALIDATION_NOISE_SEED)?;
        for &id in subjects {
            let memberships = dataset
                .subjects
                .get(&id)
                .ok_or(HierarchicalTrainingError::UnknownSubject(id))?;
            let baseline =
                self.evaluation_baseline(dataset, partitions, id, history_hands, provider)?;
            let mut subject_decisions_seen = 0usize;
            for membership in memberships {
                let hand = &dataset.hands[membership.hand_slot];
                if !partitions.contains_evaluation(partition, hand.hand_index) {
                    continue;
                }
                let observed = hand.materialize(membership.subject_position);
                for row in subject_decisions(&observed) {
                    let observation = decode_decision(&observed, row)?;
                    let legal = lookup(provider, id, &observed, row)?;
                    let z2 = self.analyzer.infer_encoded(&observation, &baseline)?;
                    let (z0, z0_probs) = self
                        .generator
                        .score_observation_with_probabilities_encoded(
                            self.blueprint.centre(),
                            &observation,
                            &legal,
                        )?;
                    let (z1, z1_probs) = self
                        .generator
                        .score_observation_with_probabilities_encoded(
                            baseline.profile(),
                            &observation,
                            &legal,
                        )?;
                    let target = usize::from(action_id(observation.observed.kind));
                    for (slot, probabilities) in [(0, z0_probs), (1, z1_probs)] {
                        if predicted_action(&probabilities) == target {
                            correct[slot] += 1;
                        }
                    }
                    let reconstruction_profile = z2.posterior().mean_profile()?;
                    let posterior = self.generator.score_observation_encoded(
                        &reconstruction_profile,
                        &observation,
                        &legal,
                    )?;
                    // Marginalize the joint likelihood before applying -log.
                    // Conditional components use the exact chain rule:
                    // NLL(action, sizing, timing) = NLL(action)
                    // + NLL(sizing | action) + NLL(timing | action, sizing).
                    // Marginalizing each head independently would be incorrect.
                    let mut draws = Vec::with_capacity(Z2_PRIOR_MARGINAL_SAMPLES);
                    let mut marginal_probabilities = [0.0; ACTION_COUNT];
                    for _ in 0..Z2_PRIOR_MARGINAL_SAMPLES {
                        let noise = eval_noise.sample(self.generator.latent_config().dimension());
                        let profile = z2.prior().sample(&noise)?;
                        let (loss, probabilities) = self
                            .generator
                            .score_observation_with_probabilities_encoded(
                                &profile,
                                &observation,
                                &legal,
                            )?;
                        for (sum, probability) in
                            marginal_probabilities.iter_mut().zip(probabilities)
                        {
                            *sum += probability / Z2_PRIOR_MARGINAL_SAMPLES as f64;
                        }
                        draws.push(loss);
                    }
                    if predicted_action(&marginal_probabilities) == target {
                        correct[2] += 1;
                    }
                    let marginal = marginal_nll(draws.iter().map(|x| x.total))?;
                    let marginal_action = marginal_nll(draws.iter().map(|x| x.action))?;
                    let marginal_action_sizing =
                        marginal_nll(draws.iter().map(|x| x.action + x.sizing))?;
                    let conditional_sizing = marginal_action_sizing - marginal_action;
                    let conditional_timing = marginal - marginal_action_sizing;
                    let is_wager = matches!(
                        observation.observed.kind,
                        crate::behavioural::ActionKind::Bet | crate::behavioural::ActionKind::Raise
                    );
                    sum[0] += z0.total;
                    sum[1] += z1.total;
                    sum[2] += marginal;
                    sum[3] += posterior.total;
                    for (slot, loss) in [(0, &z0), (1, &z1)] {
                        component_sums[slot][0] += loss.action;
                        if is_wager {
                            component_sums[slot][1] += loss.sizing;
                        }
                        component_sums[slot][2] += loss.timing;
                    }
                    component_sums[2][0] += marginal_action;
                    if is_wager {
                        component_sums[2][1] += conditional_sizing;
                    }
                    component_sums[2][2] += conditional_timing;
                    if is_wager {
                        wager_count = wager_count
                            .checked_add(1)
                            .ok_or(HierarchicalTrainingError::CountOverflow)?;
                    }
                    kl += z2.kl_to_prior()?;
                    count = count
                        .checked_add(1)
                        .ok_or(HierarchicalTrainingError::CountOverflow)?;
                    subject_decisions_seen += 1;
                }
            }
            if subject_decisions_seen == 0 {
                return Err(HierarchicalTrainingError::EmptyEvaluationSubject(
                    id, partition,
                ));
            }
        }
        if count == 0 {
            return Err(HierarchicalTrainingError::EmptyValidation);
        }
        if !kl.is_finite()
            || sum.iter().any(|v| !v.is_finite())
            || component_sums.iter().flatten().any(|v| !v.is_finite())
        {
            return Err(HierarchicalTrainingError::NonFiniteUpdate);
        }
        let components = |values: [f64; 3]| HierarchicalNllComponents {
            action: values[0] / count as f64,
            sizing: (wager_count > 0).then(|| values[1] / wager_count as f64),
            timing_log1p_ms: values[2] / count as f64,
        };
        Ok(HierarchicalValidationReport {
            decisions: count,
            wager_decisions: wager_count,
            z2_prior_samples: Z2_PRIOR_MARGINAL_SAMPLES,
            history_hands,
            z0_action_accuracy: correct[0] as f64 / count as f64,
            z1_action_accuracy: correct[1] as f64 / count as f64,
            z2_prior_action_accuracy: correct[2] as f64 / count as f64,
            z0_predictive_nll: sum[0] / count as f64,
            z1_predictive_nll: sum[1] / count as f64,
            z2_prior_marginal_nll: sum[2] / count as f64,
            z2_posterior_reconstruction_nll: sum[3] / count as f64,
            z0_components: components(component_sums[0]),
            z1_components: components(component_sums[1]),
            z2_prior_marginal_components: components(component_sums[2]),
            mean_conditional_kl: kl / count as f64,
        })
    }
}

/// Deterministic argmax; ties retain the lowest action code.
fn predicted_action(probabilities: &[f64; ACTION_COUNT]) -> usize {
    let mut best = 0;
    for i in 1..ACTION_COUNT {
        if probabilities[i] > probabilities[best] {
            best = i;
        }
    }
    best
}

/// Stable negative log of the arithmetic mean of probabilities/densities.
/// Each input is already a joint or conditional negative log likelihood.
fn marginal_nll(losses: impl IntoIterator<Item = f64>) -> Result<f64, HierarchicalTrainingError> {
    let losses = losses.into_iter().collect::<Vec<_>>();
    if losses.is_empty() || losses.iter().any(|v| !v.is_finite()) {
        return Err(HierarchicalTrainingError::NonFiniteUpdate);
    }
    let maximum = losses.iter().map(|v| -v).fold(f64::NEG_INFINITY, f64::max);
    let mean_exp = losses
        .iter()
        .map(|loss| (-loss - maximum).exp())
        .sum::<f64>()
        / losses.len() as f64;
    let result = -maximum - mean_exp.ln();
    if !result.is_finite() {
        return Err(HierarchicalTrainingError::NonFiniteUpdate);
    }
    Ok(result)
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
    for (_, hand) in &sample.query {
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
    TrainingPhaseCompleted,
    TrainingPartitionChanged,
    DatasetChanged,
    NotTrained,
    InvalidCheckpoint(String),
    CheckpointAlreadyExists(PathBuf),
    Io(io::Error),
    Json(serde_json::Error),
    LatentDimensionMismatch,
    UnknownSubject(u64),
    DuplicateSubject(u64),
    DuplicateTrainingHand {
        hand_index: u64,
        subject_id: u64,
    },
    DuplicateQueryDecision {
        hand_index: u64,
        subject_id: u64,
        sequence_index: u32,
    },
    InsufficientHistory {
        subject: u64,
        required: usize,
        available: usize,
    },
    EmptyTraining,
    PopulationCoverageMismatch {
        expected: usize,
        observed: usize,
    },
    EmptyValidation,
    EmptyEvaluationSubject(u64, EvaluationPartition),
    NonFiniteUpdate,
    CountOverflow,
    Generator(GeneratorError),
    Analyzer(AnalyzerError),
    Baseline(BaselineAdaptationError),
    Profile(BehaviourProfileError),
    Data(TrainingDataError),
    Legality(super::exact_legality::ExactLegalityError),
}
impl fmt::Display for HierarchicalTrainingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidCheckpoint(message) => write!(f, "checkpoint rejected: {message}"),
            Self::CheckpointAlreadyExists(path) => {
                write!(f, "checkpoint already exists: {}", path.display())
            }
            Self::Io(source) => write!(f, "I/O: {source}"),
            Self::Json(source) => write!(f, "JSON: {source}"),
            Self::Legality(source) => write!(f, "exact replay legality: {source}"),
            other => write!(f, "{other:?}"),
        }
    }
}
impl std::error::Error for HierarchicalTrainingError {}
impl From<io::Error> for HierarchicalTrainingError {
    fn from(v: io::Error) -> Self {
        Self::Io(v)
    }
}
impl From<serde_json::Error> for HierarchicalTrainingError {
    fn from(v: serde_json::Error) -> Self {
        Self::Json(v)
    }
}
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
impl From<super::exact_legality::ExactLegalityError> for HierarchicalTrainingError {
    fn from(value: super::exact_legality::ExactLegalityError) -> Self {
        Self::Legality(value)
    }
}

impl From<TrainingDataError> for HierarchicalTrainingError {
    fn from(value: TrainingDataError) -> Self {
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
            HierarchicalOptimizationConfig::new(4).unwrap(),
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
            exact_legality: Some(super::super::exact_legality::ExactLegalityIndex {
                by_decision: std::collections::HashMap::new(),
                fingerprint: "00000000000000000000000000000001".to_owned(),
            }),
        }
    }

    #[test]
    fn three_training_phases_preserve_global_future_partition() {
        let dataset = dataset();
        let partitions = DatasetPartitions::canonical_for_test(&dataset);
        let episode = TrainingEpisodeConfig::new(3, 3).unwrap();
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
            .train_blueprint_epoch(&dataset, &[42], &partitions, episode, &mut legal_source)
            .unwrap();
        assert_eq!(baseline.phase, HierarchicalPhase::Blueprint);
        assert_eq!(baseline.decisions, 6);
        assert_eq!(baseline.optimizer_steps, 2); // final incomplete mini-batch included
        assert_eq!(
            seen.borrow().iter().map(String::as_str).collect::<Vec<_>>(),
            vec!["h0", "h1", "h2", "h3", "h4", "h5"]
        );
        seen.borrow_mut().clear();
        let personal = trainer
            .train_personalization_epoch(&dataset, &[42], &partitions, episode, &mut legal_source)
            .unwrap();
        assert_eq!(personal.decisions, 6);
        assert_eq!(personal.optimizer_steps, 2);
        assert_eq!(
            seen.borrow().iter().map(String::as_str).collect::<Vec<_>>(),
            vec!["h0", "h1", "h2", "h3", "h4", "h5"]
        );
        assert!(
            !seen
                .borrow()
                .iter()
                .any(|x| ["h6", "h7", "h8"].contains(&x.as_str()))
        );
        let p = trainer
            .evaluate_personalization(&dataset, &partitions, 2, &mut legal_source)
            .unwrap();
        assert_eq!(p.decisions, 1);
        assert!(p.predictive_nll.is_finite());
        seen.borrow_mut().clear();
        let momentary = trainer
            .train_momentary_epoch(&dataset, &[42], &partitions, episode, 2, &mut legal_source)
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
            .evaluate(
                &dataset,
                &[42],
                &partitions,
                EvaluationPartition::Validation,
                &mut legal_source,
            )
            .unwrap();
        assert_eq!(report.decisions, 1);
        assert_eq!(report.z2_prior_samples, Z2_PRIOR_MARGINAL_SAMPLES);
        assert!(report.z2_prior_marginal_nll.is_finite());
        let repeat = trainer
            .evaluate(
                &dataset,
                &[42],
                &partitions,
                EvaluationPartition::Validation,
                &mut legal_source,
            )
            .unwrap();
        assert_eq!(report, repeat);
        let zero = trainer
            .evaluate_with_history(
                &dataset,
                &[42],
                &partitions,
                EvaluationPartition::Validation,
                0,
                &mut legal_source,
            )
            .unwrap();
        assert_eq!(zero.decisions, report.decisions);
        assert_eq!(zero.z0_predictive_nll, zero.z1_predictive_nll);
        assert_eq!(zero.z0_action_accuracy, zero.z1_action_accuracy);
        assert_eq!(zero.history_hands, 0);
        assert!((0.0..=1.0).contains(&zero.z2_prior_action_accuracy));
        assert!(matches!(
            trainer.evaluate_with_history(
                &dataset,
                &[42],
                &partitions,
                EvaluationPartition::Validation,
                7,
                &mut legal_source
            ),
            Err(HierarchicalTrainingError::InsufficientHistory { .. })
        ));

        assert!(
            !seen
                .borrow()
                .iter()
                .any(|id| ["h7", "h8"].contains(&id.as_str()))
        );
        seen.borrow_mut().clear();
        let test_report = trainer
            .evaluate(
                &dataset,
                &[42],
                &partitions,
                EvaluationPartition::Test,
                &mut legal_source,
            )
            .unwrap();
        assert_eq!(test_report.decisions, 2);
        assert!(!seen.borrow().iter().any(|id| id == "h6"));
        for value in [
            report.z0_predictive_nll,
            report.z1_predictive_nll,
            report.z2_prior_marginal_nll,
            report.z2_posterior_reconstruction_nll,
        ] {
            assert!(value.is_finite());
        }
        assert_eq!(trainer.completed_epochs(), (1, 1, 1));
    }

    #[test]
    fn blueprint_includes_unselected_subject_decisions() {
        use super::super::SubjectMembership;
        use crate::behavioural::codebook::position_id;
        let mut data = dataset();
        let hand = &mut data.hands[0];
        let mut categorical = hand.categorical_without_subject[0];
        categorical[1] = i64::from(position_id(Position::BigBlind));
        hand.categorical_without_subject.push(categorical);
        hand.sequence_indexes.push(1);
        let continuous = hand.continuous[0];
        let present = hand.continuous_present[0];
        let timing = hand.decision_time_ms[0];
        hand.continuous.push(continuous);
        hand.continuous_present.push(present);
        hand.decision_time_ms.push(timing);
        data.event_count += 1;
        data.summary.events_written += 1;
        data.summary.subject_hand_rows_written += 1;
        data.subjects.insert(
            99,
            vec![SubjectMembership {
                hand_slot: 0,
                subject_position: Position::BigBlind,
            }],
        );
        let partitions = DatasetPartitions::canonical_for_test(&data);
        let mut trainer = trainer();
        let mut seen = Vec::new();
        let mut provider =
            |id: u64, _: &crate::behavioural::HandId, sequence: u32| -> Result<_, &'static str> {
                seen.push((id, sequence));
                Ok(decision(600).1)
            };
        let report = trainer
            .train_blueprint_epoch(
                &data,
                &[42],
                &partitions,
                TrainingEpisodeConfig::new(3, 3).unwrap(),
                &mut provider,
            )
            .unwrap();
        assert_eq!(report.decisions, 7);
        assert_eq!(report.optimizer_steps, 2);
        assert!(seen.contains(&(99, 1)));
        assert_eq!(trainer.training_subjects(), &[42]);
    }

    #[test]
    fn blueprint_checkpoint_is_evaluable_before_momentary() {
        let data = dataset();
        let partitions = DatasetPartitions::canonical_for_test(&data);
        let episode = TrainingEpisodeConfig::new(3, 3).unwrap();
        let mut trainer = trainer();
        let mut provider = |_: u64,
                            _: &crate::behavioural::HandId,
                            _: u32|
         -> Result<_, &'static str> { Ok(decision(600).1) };
        trainer
            .train_blueprint_epoch(&data, &[42], &partitions, episode, &mut provider)
            .unwrap();
        let report = trainer
            .evaluate_blueprint(&data, &partitions, &mut provider)
            .unwrap();
        assert_eq!(report.decisions, 1);
        assert!(report.predictive_nll.is_finite());
        assert!(matches!(
            trainer.evaluate(
                &data,
                &[42],
                &partitions,
                EvaluationPartition::Validation,
                &mut provider
            ),
            Err(HierarchicalTrainingError::NotTrained)
        ));
    }

    #[test]
    fn frozen_personalization_benchmark_requires_only_trained_blueprint() {
        let data = dataset();
        let partitions = DatasetPartitions::canonical_for_test(&data);
        let episode = TrainingEpisodeConfig::new(3, 3).unwrap();
        let mut trainer = trainer();
        let mut provider = |_: u64,
                            _: &crate::behavioural::HandId,
                            _: u32|
         -> Result<_, &'static str> { Ok(decision(600).1) };
        assert!(matches!(
            trainer.evaluate_personalization(&data, &partitions, 0, &mut provider),
            Err(HierarchicalTrainingError::NotTrained)
        ));
        trainer
            .train_blueprint_epoch(&data, &[42], &partitions, episode, &mut provider)
            .unwrap();
        let model_id = trainer.model_id().unwrap();
        let generator_optimizer = trainer.generator_adam.checkpoint_value();
        let analyzer_optimizer = trainer.analyzer_adam.checkpoint_value();
        let noise_state = trainer.noise.state;
        let reference = trainer
            .evaluate_blueprint(&data, &partitions, &mut provider)
            .unwrap();
        let zero = trainer
            .evaluate_personalization(&data, &partitions, 0, &mut provider)
            .unwrap();
        assert_eq!(reference, zero);
        let adapted = trainer
            .evaluate_personalization(&data, &partitions, 2, &mut provider)
            .unwrap();
        assert_eq!(adapted.decisions, reference.decisions);
        assert_eq!(adapted.wager_decisions, reference.wager_decisions);
        assert!(adapted.predictive_nll.is_finite());
        assert_eq!(
            adapted,
            trainer
                .evaluate_personalization(&data, &partitions, 2, &mut provider)
                .unwrap()
        );
        assert!(matches!(
            trainer.evaluate_personalization(&data, &partitions, 7, &mut provider),
            Err(HierarchicalTrainingError::InsufficientHistory { .. })
        ));
        assert_eq!(trainer.completed_epochs(), (1, 0, 0));
        assert_eq!(trainer.model_id().unwrap(), model_id);
        assert_eq!(
            trainer.generator_adam.checkpoint_value(),
            generator_optimizer
        );
        assert_eq!(trainer.analyzer_adam.checkpoint_value(), analyzer_optimizer);
        assert_eq!(trainer.noise.state, noise_state);
    }

    #[test]
    fn marginal_nll_uses_logmeanexp_not_mean_nll() {
        let equal = marginal_nll([3.0, 3.0]).unwrap();
        assert!((equal - 3.0).abs() < 1e-12);
        let expected = -(((-0.1_f64).exp() + (-2.0_f64).exp()) / 2.0).ln();
        assert!((marginal_nll([0.1, 2.0]).unwrap() - expected).abs() < 1e-12);
        let action = marginal_nll([0.1, 1.0]).unwrap();
        let action_sizing = marginal_nll([0.1 + 2.0, 1.0 + 0.5]).unwrap();
        let joint = marginal_nll([0.1 + 2.0 + 0.7, 1.0 + 0.5 + 1.2]).unwrap();
        let sizing_given_action = action_sizing - action;
        let timing_given_action_sizing = joint - action_sizing;
        assert!((action + sizing_given_action + timing_given_action_sizing - joint).abs() < 1e-12);
        assert!(marginal_nll([f64::INFINITY]).is_err());
        assert!(marginal_nll(std::iter::empty()).is_err());
        let stable = marginal_nll([50_000.0, 50_001.0]).unwrap();
        assert!(stable.is_finite());
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
    fn personalization_mixes_subjects_and_resets_independent_trajectories() {
        use super::super::SubjectMembership;
        use crate::behavioural::ContinuousFeature;
        use crate::behavioural::codebook::position_id;

        let mut data = dataset();
        // Two subjects with different timing histories, both present in every hand.
        for hand in &mut data.hands {
            let mut categorical = hand.categorical_without_subject[0];
            categorical[1] = i64::from(position_id(Position::BigBlind));
            let mut continuous = hand.continuous[0];
            continuous[ContinuousFeature::DecisionTimeMs as usize] = 6000.0;
            let present = hand.continuous_present[0];
            hand.categorical_without_subject.push(categorical);
            hand.sequence_indexes.push(1);
            hand.continuous.push(continuous);
            hand.continuous_present.push(present);
            hand.decision_time_ms.push(6000);
        }
        let hand_count = data.hands.len();
        data.event_count += hand_count;
        data.summary.events_written += hand_count as u64;
        data.summary.subject_hand_rows_written += hand_count as u64;
        data.subjects.insert(
            99,
            (0..hand_count)
                .map(|hand_slot| SubjectMembership {
                    hand_slot,
                    subject_position: Position::BigBlind,
                })
                .collect(),
        );
        let partitions = DatasetPartitions::canonical_for_test(&data);
        let episode = TrainingEpisodeConfig::new(2, 3).unwrap();
        let mut initial = trainer();
        initial.optimization = HierarchicalOptimizationConfig::new(3).unwrap();
        let subjects = [99, 42]; // deliberately opposite the ingestion tie-break
        let mut provider = |_: u64,
                            _: &crate::behavioural::HandId,
                            _: u32|
         -> Result<_, &'static str> { Ok(decision(600).1) };
        initial
            .train_blueprint_epoch(&data, &subjects, &partitions, episode, &mut provider)
            .unwrap();
        // Independent oracle: two-hand trajectories per player, one update per
        // hand, exact three-decision batches in ingestion order.
        let ordered = (0_usize..6)
            .flat_map(|index| {
                [
                    (index, 42, Position::Button, 0),
                    (index, 99, Position::BigBlind, 1),
                ]
            })
            .collect::<Vec<_>>();
        let mut actual = initial.clone();
        let mut expected = initial.clone();
        let adapter =
            PlayerBaselineAdapter::new(initial.blueprint.clone(), initial.baseline_config).unwrap();
        let mut states = [adapter.initialize_player(), adapter.initialize_player()];
        let mut nll = 0.0;
        for chunk in ordered.chunks(3) {
            let mut batch = GeneratorGradientBatch::new(&expected.generator);
            for &(index, id, position, row) in chunk {
                let state = &mut states[if id == 99 { 1 } else { 0 }];
                if index.is_multiple_of(2) {
                    *state = adapter.initialize_player();
                    assert_eq!(state.baseline().profile(), initial.blueprint.centre());
                    assert_eq!(
                        state.baseline().uncertainty_std(),
                        initial.blueprint.prior_std()
                    );
                }
                assert_eq!(state.baseline().observed_hands(), (index % 2) as u64);
                let hand = data.hands[index].materialize(position);
                let observation = decode_decision(&hand, row).unwrap();
                let legal = lookup(&mut provider, id, &hand, row).unwrap();
                nll += batch
                    .accumulate_snapshot_encoded(state.baseline().profile(), &observation, &legal)
                    .unwrap()
                    .total;
                adapter
                    .adapt_hand(
                        state,
                        &expected.generator,
                        index as u64,
                        &[(observation, legal)],
                    )
                    .unwrap();
            }
            expected
                .generator_adam
                .apply_batch(&mut expected.generator, batch)
                .unwrap();
        }
        let report = actual
            .train_personalization_epoch(&data, &subjects, &partitions, episode, &mut provider)
            .unwrap();
        assert_eq!(report.decisions, 12);
        assert_eq!(report.optimizer_steps, 4);
        assert_eq!(report.mean_nll, nll / 12.0);
        assert_eq!(
            actual.generator.parameters(),
            expected.generator.parameters()
        );
        assert_eq!(
            actual.generator_adam.checkpoint_value(),
            expected.generator_adam.checkpoint_value()
        );
        assert_eq!(actual.blueprint, initial.blueprint);
        assert_eq!(actual.analyzer.parameters(), initial.analyzer.parameters());
        assert_eq!(actual.completed_epochs(), (1, 1, 0));
        assert_ne!(
            states[0].baseline().profile(),
            states[1].baseline().profile()
        );
    }

    #[test]
    fn personalization_adapts_once_after_complete_hand_across_batches() {
        let mut data = dataset();
        for hand in &mut data.hands {
            hand.categorical_without_subject
                .push(hand.categorical_without_subject[0]);
            hand.sequence_indexes.push(1);
            let mut continuous = hand.continuous[0];
            continuous[crate::behavioural::ContinuousFeature::DecisionTimeMs as usize] = 6000.0;
            hand.continuous.push(continuous);
            hand.continuous_present.push(hand.continuous_present[0]);
            hand.decision_time_ms.push(6000);
        }
        data.event_count += data.hands.len();
        data.summary.events_written += data.hands.len() as u64;
        let partitions = DatasetPartitions::canonical_for_test(&data);
        let episode = TrainingEpisodeConfig::new(4, 3).unwrap();
        let mut actual = trainer();
        actual.optimization = HierarchicalOptimizationConfig::new(5).unwrap();
        let mut provider = |_: u64,
                            _: &crate::behavioural::HandId,
                            _: u32|
         -> Result<_, &'static str> { Ok(decision(600).1) };
        actual
            .train_blueprint_epoch(&data, &[42], &partitions, episode, &mut provider)
            .unwrap();
        let mut expected = actual.clone();
        let adapter =
            PlayerBaselineAdapter::new(expected.blueprint.clone(), expected.baseline_config)
                .unwrap();
        let mut state = adapter.initialize_player();
        let mut batch = GeneratorGradientBatch::new(&expected.generator);
        let mut total = 0.0;
        for index in 0..6 {
            if index == 4 {
                state = adapter.initialize_player();
            }
            let hand = data.hands[index].materialize(Position::Button);
            let observations =
                [0, 1].map(|row| (decode_decision(&hand, row).unwrap(), decision(600).1));
            let pre_hand = state.baseline().profile().clone();
            for (row, (observation, legal)) in observations.iter().enumerate() {
                total += batch
                    .accumulate_snapshot_encoded(&pre_hand, observation, legal)
                    .unwrap()
                    .total;
                if row == 1 {
                    adapter
                        .adapt_hand(&mut state, &expected.generator, index as u64, &observations)
                        .unwrap();
                }
                if batch.len() == 5 {
                    expected
                        .generator_adam
                        .apply_batch(&mut expected.generator, batch)
                        .unwrap();
                    batch = GeneratorGradientBatch::new(&expected.generator);
                }
            }
        }
        assert_eq!(state.baseline().observed_hands(), 2);
        assert_eq!(state.observed_decisions(), 4);
        assert_eq!(batch.len(), 2);
        expected
            .generator_adam
            .apply_batch(&mut expected.generator, batch)
            .unwrap();
        let report = actual
            .train_personalization_epoch(&data, &[42], &partitions, episode, &mut provider)
            .unwrap();
        assert_eq!(report.decisions, 12);
        assert_eq!(report.optimizer_steps, 3);
        assert_eq!(report.mean_nll, total / 12.0);
        assert_eq!(
            actual.generator.parameters(),
            expected.generator.parameters()
        );
        assert_eq!(
            actual.generator_adam.checkpoint_value(),
            expected.generator_adam.checkpoint_value()
        );
        // A second pass starts from a fresh Z0; no duplicate-hand error.
        assert_eq!(
            actual
                .train_personalization_epoch(&data, &[42], &partitions, episode, &mut provider)
                .unwrap()
                .decisions,
            12
        );
        assert_eq!(actual.completed_epochs(), (1, 2, 0));
    }

    #[test]
    fn momentary_generator_uses_only_z2_gradients() {
        let mut actual = trainer();
        let adapter =
            PlayerBaselineAdapter::new(actual.blueprint.clone(), actual.baseline_config).unwrap();
        let state = adapter.initialize_player();
        let baseline = state.baseline();
        let decisions = [decision(600), decision(10000)];
        let mut expected = actual.clone();
        let mut noise = expected.noise;
        let mut batch = GeneratorGradientBatch::new(&expected.generator);
        for (observation, legal) in &decisions {
            let posterior = expected
                .analyzer
                .infer_encoded(observation, baseline)
                .unwrap();
            let prior_noise = noise.sample(expected.generator.latent_config().dimension());
            let post_noise = noise.sample(expected.generator.latent_config().dimension());
            let prior = posterior.prior().sample(&prior_noise).unwrap();
            let post = posterior.posterior().sample(&post_noise).unwrap();
            for profile in [&prior, &post] {
                batch
                    .accumulate_encoded(&expected.generator, profile, observation, legal)
                    .unwrap();
            }
        }
        expected
            .generator_adam
            .apply_batch(&mut expected.generator, batch)
            .unwrap();
        actual
            .train_momentary_decisions(baseline, &decisions)
            .unwrap();
        assert_eq!(
            actual.generator.checkpoint_value(),
            expected.generator.checkpoint_value()
        );
        assert_eq!(
            actual.generator_adam.checkpoint_value(),
            expected.generator_adam.checkpoint_value()
        );
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

    #[test]
    fn hierarchical_checkpoint_roundtrips_all_state_and_rejects_legacy() {
        let source = dataset();
        let partitions = DatasetPartitions::canonical_for_test(&source);
        let episode = TrainingEpisodeConfig::new(3, 3).unwrap();
        let mut trainer = trainer();
        let mut provider = |_: u64,
                            _: &crate::behavioural::HandId,
                            _: u32|
         -> Result<_, &'static str> { Ok(decision(600).1) };
        trainer
            .train_blueprint_epoch(&source, &[42], &partitions, episode, &mut provider)
            .unwrap();
        trainer
            .train_personalization_epoch(&source, &[42], &partitions, episode, &mut provider)
            .unwrap();
        trainer
            .train_momentary_epoch(&source, &[42], &partitions, episode, 2, &mut provider)
            .unwrap();
        let path = std::env::temp_dir().join(format!(
            "lechiffre-hierarchical-{}-{}.json",
            std::process::id(),
            trainer.model_id().unwrap()
        ));
        let _ = std::fs::remove_file(&path);
        trainer.save_checkpoint(&path).unwrap();
        assert!(matches!(
            trainer.save_checkpoint(&path),
            Err(HierarchicalTrainingError::CheckpointAlreadyExists(_))
        ));
        let mut restored = HierarchicalTrainer::load_checkpoint(&path).unwrap();
        assert_eq!(trainer.model_id().unwrap(), restored.model_id().unwrap());
        assert_eq!(trainer.completed_epochs(), restored.completed_epochs());
        assert_eq!(trainer.training_subjects(), restored.training_subjects());
        restored
            .train_momentary_epoch(&source, &[42], &partitions, episode, 2, &mut provider)
            .unwrap();
        assert_eq!(restored.completed_epochs(), (1, 1, 2));
        let mut obsolete: Value = serde_json::from_reader(File::open(&path).unwrap()).unwrap();
        let invalid_path = path.with_extension("obsolete.json");
        for obsolete_version in [
            "lechiffre.hierarchical.checkpoint.v1",
            "lechiffre.hierarchical.checkpoint.v2",
            "lechiffre.hierarchical.checkpoint.v3",
            "lechiffre.hierarchical.checkpoint.v4",
            "lechiffre.hierarchical.checkpoint.v5",
            "lechiffre.hierarchical.checkpoint.v6",
            "lechiffre.hierarchical.checkpoint.v7",
            "lechiffre.hierarchical.checkpoint.v8",
            "lechiffre.hierarchical.checkpoint.v9",
            "lechiffre.hierarchical.checkpoint.v10",
        ] {
            obsolete["format"] = json!(obsolete_version);
            std::fs::write(&invalid_path, serde_json::to_vec(&obsolete).unwrap()).unwrap();
            assert!(matches!(
                HierarchicalTrainer::load_checkpoint(&invalid_path),
                Err(HierarchicalTrainingError::InvalidCheckpoint(_))
            ));
        }
        std::fs::remove_file(&path).unwrap();
        std::fs::remove_file(&invalid_path).unwrap();
    }

    #[test]
    fn checkpoint_resume_rejects_replaced_dataset() {
        let source = dataset();
        let partitions = DatasetPartitions::canonical_for_test(&source);
        let episode = TrainingEpisodeConfig::new(3, 3).unwrap();
        let mut trainer = trainer();
        let mut provider = |_: u64,
                            _: &crate::behavioural::HandId,
                            _: u32|
         -> Result<_, &'static str> { Ok(decision(600).1) };
        trainer
            .train_blueprint_epoch(&source, &[42], &partitions, episode, &mut provider)
            .unwrap();
        let mut changed = dataset();
        changed.hands[0].continuous[0][0] += 1.0;
        assert!(matches!(
            trainer.train_blueprint_epoch(&changed, &[42], &partitions, episode, &mut provider),
            Err(HierarchicalTrainingError::DatasetChanged)
        ));
        assert_eq!(trainer.completed_epochs(), (1, 0, 0));
        let mut changed_legality = dataset();
        changed_legality
            .exact_legality
            .as_mut()
            .unwrap()
            .fingerprint = "00000000000000000000000000000002".into();
        assert!(matches!(
            trainer.train_blueprint_epoch(
                &changed_legality,
                &[42],
                &partitions,
                episode,
                &mut provider
            ),
            Err(HierarchicalTrainingError::TrainingPartitionChanged)
        ));
        let changed_episode = TrainingEpisodeConfig::new(2, 3).unwrap();
        assert!(matches!(
            trainer.train_blueprint_epoch(
                &source,
                &[42],
                &partitions,
                changed_episode,
                &mut provider
            ),
            Err(HierarchicalTrainingError::TrainingPartitionChanged)
        ));
    }
}

// Explicit diagnostic: no training, no model save, no Test decisions.
#[cfg(test)]
mod sizing_diagnostic {
    use super::*;
    use crate::behavioural::ActionKind;

    #[test]
    #[ignore = "requires the real dataset, manifest, and two explicit V10 checkpoints"]
    fn sizing_diagnostic_z32_g64() {
        let env = |key: &str| std::env::var(key).expect(key);
        let data = BehaviourParquetDataset::open(env("LECHIFFRE_DIAG_DATASET")).unwrap();
        let partitions =
            DatasetPartitions::read_manifest(&data, env("LECHIFFRE_DIAG_MANIFEST")).unwrap();
        let mut provider = |_: u64, hand: &crate::behavioural::HandId, sequence: u32| {
            data.exact_legal_actions(hand, sequence)
                .map_err(|e| e.to_string())
        };
        let blueprint =
            HierarchicalTrainer::load_checkpoint(env("LECHIFFRE_DIAG_BLUEPRINT")).unwrap();
        let personalized =
            HierarchicalTrainer::load_checkpoint(env("LECHIFFRE_DIAG_PERSONALIZATION")).unwrap();
        let mut output = std::io::BufWriter::new(
            OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(env("LECHIFFRE_DIAG_OUTPUT"))
                .unwrap(),
        );
        for (phase, trainer, epochs) in [
            ("B", &blueprint, (25, 0, 0)),
            ("P", &personalized, (25, 5, 0)),
        ] {
            trainer
                .check_evaluation_dataset(&data, &partitions)
                .unwrap();
            assert_eq!(trainer.completed_epochs(), epochs);
            assert_eq!(trainer.training_subjects(), &[3, 6, 8, 11, 13, 14]);
            assert_eq!(trainer.generator.latent_config().dimension(), 32);
            assert_eq!(trainer.generator.parameters().state.output_width(), 64);
            let model_id = trainer.model_id().unwrap();
            let mut n = 0;
            let mut wagers = 0;
            for &id in trainer.training_subjects() {
                let baseline = trainer
                    .evaluation_baseline(&data, &partitions, id, 100, &mut provider)
                    .unwrap();
                for membership in &data.subjects[&id] {
                    let hand = &data.hands[membership.hand_slot];
                    if !partitions
                        .contains_evaluation(EvaluationPartition::Validation, hand.hand_index)
                    {
                        continue;
                    }
                    let observed = hand.materialize(membership.subject_position);
                    for row in subject_decisions(&observed) {
                        n += 1;
                        let obs = decode_decision(&observed, row).unwrap();
                        if !matches!(obs.observed.kind, ActionKind::Bet | ActionKind::Raise) {
                            continue;
                        }
                        wagers += 1;
                        let legal = lookup(&mut provider, id, &observed, row).unwrap();
                        for (history, profile) in
                            [(0, trainer.blueprint.centre()), (100, baseline.profile())]
                        {
                            let mut value = trainer
                                .generator
                                .sizing_diagnostic_row(profile, &obs, &legal)
                                .unwrap();
                            value["phase"] = json!(phase);
                            value["model_id"] = json!(&model_id);
                            value["history"] = json!(history);
                            value["subject"] = json!(id);
                            value["hand_index"] = json!(hand.hand_index);
                            value["sequence_index"] = json!(observed.sequence_indexes()[row]);
                            writeln!(output, "{value}").unwrap();
                        }
                    }
                }
            }
            assert_eq!((n, wagers), (42_693, 6_656));
            println!("sizing diagnostic {phase}: {model_id} {wagers} wagers");
        }
        output.flush().unwrap();
    }
}
