//! Online, per-hand personalization of Z1 with *fixed* shared A/G parameters.
//!
//! One complete observed hand produces one adaptation, regardless of its number
//! of subject decisions. Every score is computed with the pre-update Z1 and
//! the pre-existing Generator. Future hands cannot enter this operation.
//!
//! The uncertainty update is a **diagonal empirical-Fisher approximation**,
//! not an exact Bayesian posterior or a calibrated confidence certificate.
use super::checkpoint::{exact_keys, float_array, unsigned};
use serde_json::{Value, json};
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use super::{
    BehaviourBlueprint, BehaviourLatentModel, BehaviourProfileError, GeneratorError,
    GeneratorObservation, LegalActionSpace, NeuralBehaviourGenerator, PlayerBaseline,
};

/// Explicit online adaptation settings. No per-player G/A weight updates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BaselineAdaptationConfig {
    /// Multiplier of the precision-preconditioned negative log-likelihood step.
    learning_rate: f64,
    /// Maximum coordinate movement per observed hand, expressed in the original
    /// Z0 prior standard deviation for that coordinate.
    max_step_prior_std: f64,
}

impl BaselineAdaptationConfig {
    pub fn new(
        learning_rate: f64,
        max_step_prior_std: f64,
    ) -> Result<Self, BaselineAdaptationError> {
        if !learning_rate.is_finite()
            || learning_rate <= 0.0
            || !max_step_prior_std.is_finite()
            || max_step_prior_std <= 0.0
        {
            return Err(BaselineAdaptationError::InvalidConfiguration);
        }
        Ok(Self {
            learning_rate,
            max_step_prior_std,
        })
    }
    pub fn learning_rate(&self) -> f64 {
        self.learning_rate
    }
    pub fn max_step_prior_std(&self) -> f64 {
        self.max_step_prior_std
    }
}

/// Z1 and its streaming optimization statistics for exactly one subject.
/// The owner keeps this value keyed by subject *outside* A/G model features.
#[derive(Debug, Clone, PartialEq)]
pub struct PlayerBaselineState {
    blueprint: BehaviourBlueprint,
    baseline: PlayerBaseline,
    precision: Vec<f64>,
    last_hand_index: Option<u64>,
    observed_decisions: u64,
    processed_before_hand_index: Option<u64>,
}

impl PlayerBaselineState {
    pub fn baseline(&self) -> &PlayerBaseline {
        &self.baseline
    }

    pub fn last_hand_index(&self) -> Option<u64> {
        self.last_hand_index
    }

    pub fn observed_decisions(&self) -> u64 {
        self.observed_decisions
    }

    /// All hands with global index strictly below this value have been processed.
    pub fn processed_before_hand_index(&self) -> Option<u64> {
        self.processed_before_hand_index
    }

    /// Used only after a complete, successful incremental dataset scan.
    pub(crate) fn mark_processed_before(
        &mut self,
        cutoff: u64,
    ) -> Result<(), BaselineAdaptationError> {
        if self
            .processed_before_hand_index
            .is_some_and(|old| cutoff < old)
            || self.last_hand_index.is_some_and(|last| cutoff <= last)
        {
            return Err(BaselineAdaptationError::NonIncreasingProcessedCutoff);
        }
        self.processed_before_hand_index = Some(cutoff);
        Ok(())
    }

    /// Persist a complete Z1 optimizer state. The saved model id binds its latent geometry
    /// to precisely one Z0/A/G snapshot, without writing any player id into model features.
    pub fn save_checkpoint(
        &self,
        path: impl AsRef<Path>,
        subject_id: u64,
        model_id: &str,
    ) -> Result<(), BaselineStateStorageError> {
        if model_id.is_empty() {
            return Err(BaselineStateStorageError::InvalidState(
                "empty model id".into(),
            ));
        }
        self.validate_state()?;
        let value = json!({
            "format":"lechiffre.player_baseline.v1",
            "model_id":model_id,
            "subject_id":subject_id,
            "profile":self.baseline.profile().as_slice(),
            "uncertainty_std":self.baseline.uncertainty_std(),
            "precision":self.precision,
            "observed_hands":self.baseline.observed_hands(),
            "observed_decisions":self.observed_decisions,
            "last_hand_index":self.last_hand_index,
            "processed_before_hand_index":self.processed_before_hand_index,
        });
        let destination = path.as_ref();
        if destination.exists() {
            return Err(BaselineStateStorageError::AlreadyExists(
                destination.to_path_buf(),
            ));
        }
        let mut name = destination.as_os_str().to_os_string();
        name.push(".inprogress");
        let temporary = PathBuf::from(name);
        let result = (|| -> Result<(), BaselineStateStorageError> {
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

    pub fn load_checkpoint(
        path: impl AsRef<Path>,
        subject_id: u64,
        model_id: &str,
        adapter: &PlayerBaselineAdapter,
    ) -> Result<Self, BaselineStateStorageError> {
        let v: Value = serde_json::from_reader(File::open(path)?)?;
        exact_keys(
            &v,
            &[
                "format",
                "model_id",
                "subject_id",
                "profile",
                "uncertainty_std",
                "precision",
                "observed_hands",
                "observed_decisions",
                "last_hand_index",
                "processed_before_hand_index",
            ],
        )
        .map_err(BaselineStateStorageError::InvalidState)?;
        if v["format"].as_str() != Some("lechiffre.player_baseline.v1") {
            return Err(BaselineStateStorageError::InvalidState(
                "unknown baseline format".into(),
            ));
        }
        if v["subject_id"].as_u64() != Some(subject_id) || v["model_id"].as_str() != Some(model_id)
        {
            return Err(BaselineStateStorageError::ModelOrSubjectMismatch);
        }
        let floats = |field: &Value| -> Result<Vec<f32>, BaselineStateStorageError> {
            float_array(field)
                .map_err(BaselineStateStorageError::InvalidState)?
                .into_iter()
                .map(|v| {
                    let x = v as f32;
                    if !x.is_finite() {
                        Err(BaselineStateStorageError::InvalidState(
                            "float outside f32 range".into(),
                        ))
                    } else {
                        Ok(x)
                    }
                })
                .collect()
        };
        let value = floats(&v["profile"])?;
        let std = floats(&v["uncertainty_std"])?;
        let precision =
            float_array(&v["precision"]).map_err(BaselineStateStorageError::InvalidState)?;
        let hands =
            unsigned(&v["observed_hands"]).map_err(BaselineStateStorageError::InvalidState)?;
        let decisions =
            unsigned(&v["observed_decisions"]).map_err(BaselineStateStorageError::InvalidState)?;
        let last = if v["last_hand_index"].is_null() {
            None
        } else {
            Some(unsigned(&v["last_hand_index"]).map_err(BaselineStateStorageError::InvalidState)?)
        };
        let processed = if v["processed_before_hand_index"].is_null() {
            None
        } else {
            Some(
                unsigned(&v["processed_before_hand_index"])
                    .map_err(BaselineStateStorageError::InvalidState)?,
            )
        };
        let mut state = adapter.initialize_player();
        if hands > 0 {
            let estimate = adapter.blueprint.centre().dimension();
            let profile = super::BehaviourLatentConfig::new(estimate)
                .map_err(|e| BaselineStateStorageError::InvalidState(e.to_string()))?
                .profile(value)
                .map_err(|e| BaselineStateStorageError::InvalidState(e.to_string()))?;
            state
                .baseline
                .set_estimate(profile, std, hands)
                .map_err(|e| BaselineStateStorageError::InvalidState(e.to_string()))?;
        } else if value.as_slice() != adapter.blueprint.centre().as_slice()
            || std.as_slice() != adapter.blueprint.prior_std()
        {
            return Err(BaselineStateStorageError::InvalidState(
                "zero-hand Z1 must equal Z0".into(),
            ));
        }
        state.precision = precision;
        state.observed_decisions = decisions;
        state.last_hand_index = last;
        state.processed_before_hand_index = processed;
        state.validate_state()?;
        Ok(state)
    }

    fn validate_state(&self) -> Result<(), BaselineStateStorageError> {
        let dim = self.blueprint.centre().dimension();
        let hands = self.baseline.observed_hands();
        if self.baseline.profile().dimension() != dim
            || self.baseline.uncertainty_std().len() != dim
            || self.precision.len() != dim
            || self.observed_decisions < hands
            || (hands > 0 && self.last_hand_index.is_none())
            || (hands == 0 && (self.observed_decisions != 0 || self.last_hand_index.is_some()))
            || self
                .last_hand_index
                .zip(self.processed_before_hand_index)
                .is_some_and(|(last, cutoff)| last >= cutoff)
            || self
                .precision
                .iter()
                .zip(self.baseline.uncertainty_std())
                .any(|(&prec, &std)| {
                    !prec.is_finite()
                        || prec <= 0.0
                        || !std.is_finite()
                        || std <= 0.0
                        || (1.0 / prec.sqrt() - f64::from(std)).abs()
                            > 1e-6 * f64::from(std).max(1.0)
                })
        {
            return Err(BaselineStateStorageError::InvalidState(
                "inconsistent baseline state".into(),
            ));
        }
        if hands == 0
            && (self.baseline.profile() != self.blueprint.centre()
                || self.baseline.uncertainty_std() != self.blueprint.prior_std())
        {
            return Err(BaselineStateStorageError::InvalidState(
                "zero-hand Z1 must equal Z0".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct BaselineHandReport {
    pub hand_index: u64,
    pub observed_hands: u64,
    pub decisions: usize,
    pub mean_nll_before: f64,
    pub mean_step_norm: f64,
}

/// A blueprint-bound online optimizer for individual Z1 values.
/// It **never** mutates the shared Generator, Analyzer, or Z0.
#[derive(Debug, Clone)]
pub struct PlayerBaselineAdapter {
    blueprint: BehaviourBlueprint,
    config: BaselineAdaptationConfig,
    initial_precision: Vec<f64>,
}

impl PlayerBaselineAdapter {
    pub fn new(
        blueprint: BehaviourBlueprint,
        config: BaselineAdaptationConfig,
    ) -> Result<Self, BaselineAdaptationError> {
        let initial_precision: Vec<_> = blueprint
            .prior_std()
            .iter()
            .map(|&std| 1.0 / f64::from(std).powi(2))
            .collect();
        if initial_precision
            .iter()
            .any(|v| !v.is_finite() || *v <= 0.0)
            || blueprint
                .prior_std()
                .iter()
                .any(|&std| !(config.max_step_prior_std * f64::from(std)).is_finite())
        {
            return Err(BaselineAdaptationError::InvalidBlueprint);
        }
        Ok(Self {
            blueprint,
            config,
            initial_precision,
        })
    }

    pub fn initialize_player(&self) -> PlayerBaselineState {
        PlayerBaselineState {
            blueprint: self.blueprint.clone(),
            baseline: PlayerBaseline::from_blueprint(&self.blueprint),
            precision: self.initial_precision.clone(),
            last_hand_index: None,
            observed_decisions: 0,
            processed_before_hand_index: None,
        }
    }

    pub fn blueprint(&self) -> &BehaviourBlueprint {
        &self.blueprint
    }

    pub fn validate_player_state(
        &self,
        state: &PlayerBaselineState,
    ) -> Result<(), BaselineAdaptationError> {
        if state.blueprint != self.blueprint
            || state.precision.len() != self.initial_precision.len()
        {
            return Err(BaselineAdaptationError::BlueprintMismatch);
        }
        Ok(())
    }

    /// Update Z1 **after** a complete past hand. `hand_index` must come from
    /// the globally ordered dataset, not a player-local counter. Each tuple
    /// is one subject decision and its exact pre-decision legality from replay.
    ///
    /// On any failure neither the state nor the shared Generator is modified.
    pub fn adapt_hand(
        &self,
        state: &mut PlayerBaselineState,
        generator: &NeuralBehaviourGenerator,
        hand_index: u64,
        decisions: &[(GeneratorObservation, LegalActionSpace)],
    ) -> Result<BaselineHandReport, BaselineAdaptationError> {
        if state.blueprint != self.blueprint
            || state.precision.len() != self.initial_precision.len()
        {
            return Err(BaselineAdaptationError::BlueprintMismatch);
        }
        if generator.latent_config().dimension() != self.blueprint.centre().dimension() {
            return Err(BaselineAdaptationError::LatentDimensionMismatch);
        }
        if state
            .processed_before_hand_index
            .is_some_and(|cutoff| hand_index < cutoff)
        {
            return Err(BaselineAdaptationError::NonIncreasingProcessedCutoff);
        }
        if let Some(previous) = state.last_hand_index
            && hand_index <= previous
        {
            return Err(BaselineAdaptationError::NonIncreasingHandIndex {
                previous,
                observed: hand_index,
            });
        }
        if decisions.is_empty() {
            return Err(BaselineAdaptationError::NoSubjectDecisions);
        }
        let count = state
            .baseline
            .observed_hands()
            .checked_add(1)
            .ok_or(BaselineAdaptationError::CountOverflow)?;
        let observed_decisions = state
            .observed_decisions
            .checked_add(
                u64::try_from(decisions.len())
                    .map_err(|_| BaselineAdaptationError::CountOverflow)?,
            )
            .ok_or(BaselineAdaptationError::CountOverflow)?;
        let dim = self.initial_precision.len();
        let mut summed_gradients = vec![0.0_f64; dim];
        let mut summed_square_gradients = vec![0.0_f64; dim];
        let mut total_nll = 0.0;
        for (observation, legal) in decisions {
            // This validates the observation against the *exact* legal space;
            // no legality or missing response is synthesized from target data.
            let loss = generator.evaluate_observation_encoded(
                state.baseline.profile(),
                observation,
                legal,
            )?;
            total_nll += loss.total;
            for ((sum, square), &grad) in summed_gradients
                .iter_mut()
                .zip(&mut summed_square_gradients)
                .zip(&loss.latent_gradient)
            {
                let grad = f64::from(grad);
                *sum += grad;
                *square += grad * grad;
            }
        }
        let divisor = decisions.len() as f64;
        let mut next_values = Vec::with_capacity(dim);
        let mut next_std = Vec::with_capacity(dim);
        let mut next_precision = Vec::with_capacity(dim);
        let mut squared_step = 0.0;
        for i in 0..dim {
            let information = summed_square_gradients[i] / divisor;
            let precision = state.precision[i] + information;
            let gradient = summed_gradients[i] / divisor;
            let max_step =
                self.config.max_step_prior_std * f64::from(self.blueprint.prior_std()[i]);
            let delta =
                (-self.config.learning_rate * gradient / precision).clamp(-max_step, max_step);
            let value = f64::from(state.baseline.profile().as_slice()[i]) + delta;
            let std = 1.0 / precision.sqrt();
            if !precision.is_finite()
                || precision <= 0.0
                || !value.is_finite()
                || value.abs() > f64::from(f32::MAX)
                || !std.is_finite()
                || std <= 0.0
                || std < f64::from(f32::MIN_POSITIVE)
                || !delta.is_finite()
            {
                return Err(BaselineAdaptationError::NonFiniteUpdate);
            }
            let value = value as f32;
            let std = std as f32;
            if !value.is_finite() || !std.is_finite() || std <= 0.0 {
                return Err(BaselineAdaptationError::NonFiniteUpdate);
            }
            next_values.push(value);
            next_std.push(std);
            next_precision.push(precision);
            squared_step += delta * delta;
        }
        let mean_nll = total_nll / divisor;
        let mean_step_norm = (squared_step / dim as f64).sqrt();
        if !mean_nll.is_finite() || !mean_step_norm.is_finite() {
            return Err(BaselineAdaptationError::NonFiniteUpdate);
        }
        // Build the entire next state before committing, preserving atomicity.
        let mut updated = state.clone();
        let profile = generator
            .latent_config()
            .profile(next_values)
            .map_err(BaselineAdaptationError::Profile)?;
        updated.baseline.set_estimate(profile, next_std, count)?;
        updated.precision = next_precision;
        updated.last_hand_index = Some(hand_index);
        updated.observed_decisions = observed_decisions;
        *state = updated;
        Ok(BaselineHandReport {
            hand_index,
            observed_hands: count,
            decisions: decisions.len(),
            mean_nll_before: mean_nll,
            mean_step_norm,
        })
    }
}

#[derive(Debug)]
pub enum BaselineAdaptationError {
    InvalidConfiguration,
    InvalidBlueprint,
    BlueprintMismatch,
    LatentDimensionMismatch,
    NonIncreasingHandIndex { previous: u64, observed: u64 },
    NoSubjectDecisions,
    CountOverflow,
    NonIncreasingProcessedCutoff,
    NonFiniteUpdate,
    Profile(BehaviourProfileError),
    Generator(GeneratorError),
}

impl fmt::Display for BaselineAdaptationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for BaselineAdaptationError {}
impl From<GeneratorError> for BaselineAdaptationError {
    fn from(value: GeneratorError) -> Self {
        Self::Generator(value)
    }
}
impl From<BehaviourProfileError> for BaselineAdaptationError {
    fn from(value: BehaviourProfileError) -> Self {
        Self::Profile(value)
    }
}

#[derive(Debug)]
pub enum BaselineStateStorageError {
    Io(io::Error),
    Json(serde_json::Error),
    AlreadyExists(PathBuf),
    ModelOrSubjectMismatch,
    InvalidState(String),
}
impl fmt::Display for BaselineStateStorageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "baseline I/O: {e}"),
            Self::Json(e) => write!(f, "baseline JSON: {e}"),
            Self::AlreadyExists(path) => write!(f, "baseline already exists: {}", path.display()),
            Self::ModelOrSubjectMismatch => {
                write!(f, "baseline belongs to another model or subject")
            }
            Self::InvalidState(s) => write!(f, "invalid baseline: {s}"),
        }
    }
}
impl std::error::Error for BaselineStateStorageError {}
impl From<io::Error> for BaselineStateStorageError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}
impl From<serde_json::Error> for BaselineStateStorageError {
    fn from(e: serde_json::Error) -> Self {
        Self::Json(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::behavioural::{
        ActionKind, ActionSizing, BehaviourLatentConfig, ObservedAction, PRE_ACTION_CONTEXT_WIDTH,
        Position, Street, WagerBounds,
    };

    fn fixtures() -> (PlayerBaselineAdapter, NeuralBehaviourGenerator) {
        let latent = BehaviourLatentConfig::new(4).unwrap();
        let blueprint = BehaviourBlueprint::new(
            latent.profile(vec![0.1, -0.2, 0.3, 0.4]).unwrap(),
            vec![1.0, 1.2, 0.8, 1.0],
        )
        .unwrap();
        let adapter = PlayerBaselineAdapter::new(
            blueprint,
            BaselineAdaptationConfig::new(0.25, 0.2).unwrap(),
        )
        .unwrap();
        let generator = NeuralBehaviourGenerator::initialize(latent, 8, 42).unwrap();
        (adapter, generator)
    }

    fn decision() -> (GeneratorObservation, LegalActionSpace) {
        let mut context = [0.0_f32; PRE_ACTION_CONTEXT_WIDTH];
        context[0] = 10.0; // pot
        context[1] = 80.0; // stack
        context[3] = 5.0; // committed this hand
        context[6] = 1.0; // min-raise-to
        context[7] = 1.0; // players yet to act
        let observation = GeneratorObservation {
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
                decision_time_ms: Some(600),
            },
        };
        let legal = LegalActionSpace {
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
        };
        (observation, legal)
    }

    #[test]
    fn exactly_z0_before_any_hand_then_updates_without_changing_generator() {
        let (adapter, generator) = fixtures();
        let mut state = adapter.initialize_player();
        assert_eq!(state.baseline().profile(), adapter.blueprint().centre());
        assert_eq!(
            state.baseline().uncertainty_std(),
            adapter.blueprint().prior_std()
        );
        assert_eq!(state.baseline().observed_hands(), 0);
        let generator_before = generator.parameters().clone();
        let input = vec![decision()];
        let result = adapter
            .adapt_hand(&mut state, &generator, 7, &input)
            .unwrap();
        assert_eq!(result.observed_hands, 1);
        assert_eq!(result.decisions, 1);
        assert_eq!(state.last_hand_index(), Some(7));
        assert_eq!(state.observed_decisions(), 1);
        assert_ne!(state.baseline().profile(), adapter.blueprint().centre());
        assert!(
            state
                .baseline()
                .uncertainty_std()
                .iter()
                .zip(adapter.blueprint().prior_std())
                .all(|(new, old)| new <= old)
        );
        assert_eq!(&generator_before, generator.parameters());
    }

    #[test]
    fn many_hands_refine_uncertainty_without_changing_blueprint() {
        let (adapter, generator) = fixtures();
        let mut state = adapter.initialize_player();
        let initial = state.baseline().uncertainty_std().to_vec();
        let observation = vec![decision()];
        for hand in 0..1000_u64 {
            adapter
                .adapt_hand(&mut state, &generator, hand, &observation)
                .unwrap();
        }
        assert_eq!(state.baseline().observed_hands(), 1000);
        assert_eq!(state.observed_decisions(), 1000);
        assert!(
            state
                .baseline()
                .uncertainty_std()
                .iter()
                .zip(&initial)
                .all(|(updated, initial)| updated <= initial)
        );
        assert_eq!(
            adapter.blueprint().centre().as_slice(),
            &[0.1, -0.2, 0.3, 0.4]
        );
    }

    #[test]
    fn rejects_duplicate_or_out_of_order_hands_without_mutating_state() {
        let (adapter, generator) = fixtures();
        let mut state = adapter.initialize_player();
        adapter
            .adapt_hand(&mut state, &generator, 20, &[decision()])
            .unwrap();
        assert_eq!(state.processed_before_hand_index(), None);
        let saved = state.clone();
        let error = adapter
            .adapt_hand(&mut state, &generator, 20, &[decision()])
            .unwrap_err();
        assert!(matches!(
            error,
            BaselineAdaptationError::NonIncreasingHandIndex { .. }
        ));
        assert_eq!(state, saved);
    }

    #[test]
    fn rejects_illegal_actions_atomically() {
        let (adapter, generator) = fixtures();
        let mut state = adapter.initialize_player();
        let (observation, mut legal) = decision();
        legal.check = false;
        let before = state.clone();
        assert!(
            adapter
                .adapt_hand(&mut state, &generator, 1, &[(observation, legal)])
                .is_err()
        );
        assert_eq!(state, before);
    }

    #[test]
    fn a_hand_counts_once_even_with_multiple_subject_decisions() {
        let (adapter, generator) = fixtures();
        let mut state = adapter.initialize_player();
        let decisions = vec![decision(), decision(), decision()];
        adapter
            .adapt_hand(&mut state, &generator, 10, &decisions)
            .unwrap();
        assert_eq!(state.baseline().observed_hands(), 1);
        assert_eq!(state.observed_decisions(), 3);
    }

    #[test]
    fn sizing_and_timing_observations_can_change_z1() {
        let (adapter, generator) = fixtures();
        let mut first = adapter.initialize_player();
        let mut second = adapter.initialize_player();
        let (mut observation, legal) = decision();
        adapter
            .adapt_hand(&mut first, &generator, 0, &[(observation.clone(), legal)])
            .unwrap();
        observation.observed.decision_time_ms = Some(30_000);
        adapter
            .adapt_hand(&mut second, &generator, 0, &[(observation, legal)])
            .unwrap();
        assert_ne!(first.baseline().profile(), second.baseline().profile());
    }

    #[test]
    fn later_invalid_decision_leaves_whole_hand_uncommitted() {
        let (adapter, generator) = fixtures();
        let mut state = adapter.initialize_player();
        let (observation, mut invalid_legal) = decision();
        invalid_legal.check = false;
        let before = state.clone();
        let observations = vec![decision(), (observation, invalid_legal)];
        assert!(
            adapter
                .adapt_hand(&mut state, &generator, 1, &observations)
                .is_err()
        );
        assert_eq!(state, before);
    }

    #[test]
    fn baseline_state_is_bound_to_one_blueprint() {
        let (adapter, generator) = fixtures();
        let mut state = adapter.initialize_player();
        let latent = BehaviourLatentConfig::new(4).unwrap();
        let other = PlayerBaselineAdapter::new(
            BehaviourBlueprint::new(latent.profile(vec![0.0; 4]).unwrap(), vec![1.0; 4]).unwrap(),
            BaselineAdaptationConfig::new(0.2, 0.2).unwrap(),
        )
        .unwrap();
        assert!(matches!(
            other.adapt_hand(&mut state, &generator, 1, &[decision()]),
            Err(BaselineAdaptationError::BlueprintMismatch)
        ));
    }

    #[test]
    fn baseline_checkpoint_roundtrips_and_rejects_other_model_or_subject() {
        let (adapter, generator) = fixtures();
        let mut state = adapter.initialize_player();
        adapter
            .adapt_hand(&mut state, &generator, 4, &[decision()])
            .unwrap();
        let model_id = "hierarchical-v1-example";
        let path = std::env::temp_dir().join(format!(
            "lechiffre-baseline-{}-{}.json",
            std::process::id(),
            model_id
        ));
        let _ = std::fs::remove_file(&path);
        state.save_checkpoint(&path, 42, model_id).unwrap();
        let loaded = PlayerBaselineState::load_checkpoint(&path, 42, model_id, &adapter).unwrap();
        assert_eq!(state, loaded);
        assert!(matches!(
            state.save_checkpoint(&path, 42, model_id),
            Err(BaselineStateStorageError::AlreadyExists(_))
        ));
        assert!(matches!(
            PlayerBaselineState::load_checkpoint(&path, 43, model_id, &adapter),
            Err(BaselineStateStorageError::ModelOrSubjectMismatch)
        ));
        assert!(matches!(
            PlayerBaselineState::load_checkpoint(&path, 42, "different", &adapter),
            Err(BaselineStateStorageError::ModelOrSubjectMismatch)
        ));
        std::fs::remove_file(&path).unwrap();
    }
}
