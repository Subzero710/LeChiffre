//! Online, per-hand personalization of Z1 with *fixed* shared A/G parameters.
//!
//! One complete observed hand produces one adaptation, regardless of its number
//! of subject decisions. Every score is computed with the pre-update Z1 and
//! the pre-existing Generator. Future hands cannot enter this operation.
//!
//! The uncertainty update is a **diagonal empirical-Fisher approximation**,
//! not an exact Bayesian posterior or a calibrated confidence certificate.
use std::fmt;

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
        }
    }

    pub fn blueprint(&self) -> &BehaviourBlueprint {
        &self.blueprint
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
}
