//! Differentiable, conditional behavioural generator G(C, Z).
//!
//! The NLHE rules engine provides a *complete* action mask and exact wager
//! bounds. The model never guesses legality or enumerates arbitrary bet sizes.
//! Trainable reverse-mode gradients and Adam are implemented in Rust directly.

use std::f64::consts::PI;
use std::fmt;

use super::checkpoint::{
    dense_from_value, dense_value, dimension, exact_keys, float_array, number, unsigned,
};
use serde_json::{Value, json};

use super::codebook::{action_id, position_id, street_id};
use super::featurizer::encode_context;
use super::{
    ActionKind, ActionSizing, BehaviourEventError, BehaviourLatentConfig, BehaviourLatentModel,
    BehaviourProfile, DecisionContext, ExactAction, ObservedAction, PRE_ACTION_CONTEXT_WIDTH,
    Position, Street,
};

const STREET_COUNT: usize = 4;
const POSITION_COUNT: usize = 7;
pub const ACTION_COUNT: usize = 5;
const MIN_SIZING_SCALE: f64 = 0.1;
const MIN_TIMING_SCALE: f64 = 0.05;

pub const GENERATOR_CONTEXT_WIDTH: usize =
    STREET_COUNT + POSITION_COUNT + 2 * PRE_ACTION_CONTEXT_WIDTH;
pub const GENERATOR_SIZING_WIDTH: usize = ACTION_COUNT;
pub const GENERATOR_TIMING_WIDTH: usize = ACTION_COUNT + 1;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WagerBounds {
    /// Exact minimum added by the player in integral engine chips.
    /// A short all-in can be represented with min == max.
    pub min_added_chips: i64,
    /// NLHE maximum: all remaining chips in the actor's stack.
    pub max_added_chips: i64,
    /// Smallest legal wager increment in exact engine chips.
    pub chip_unit: i64,
    /// Big blind expressed in exact engine chips.
    pub big_blind_chips: i64,
}

impl WagerBounds {
    /// Number of non-all-in wager choices; the maximum is a separate atom.
    /// The all-in amount itself does not have to lie on the ordinary chip grid.
    fn ticks(self) -> Result<usize, GeneratorError> {
        if self.min_added_chips <= 0
            || self.max_added_chips < self.min_added_chips
            || self.chip_unit <= 0
            || self.big_blind_chips <= 0
        {
            return Err(GeneratorError::InvalidWagerBounds);
        }
        let delta = self
            .max_added_chips
            .checked_sub(self.min_added_chips)
            .ok_or(GeneratorError::InvalidWagerBounds)?;
        let count = if delta == 0 {
            0
        } else {
            (delta - 1) / self.chip_unit + 1
        };
        if count > 1_000_000_000 {
            return Err(GeneratorError::InvalidWagerBounds);
        }
        usize::try_from(count).map_err(|_| GeneratorError::InvalidWagerBounds)
    }

    fn chips(self, index: usize) -> i64 {
        let count = self.ticks().expect("validated bounds");
        if index == count {
            self.max_added_chips
        } else {
            self.min_added_chips + index as i64 * self.chip_unit
        }
    }

    fn amount(self, index: usize) -> f32 {
        (self.chips(index) as f64 / self.big_blind_chips as f64) as f32
    }

    fn index(self, amount: f32) -> Result<usize, GeneratorError> {
        let count = self.ticks()?;
        if !amount.is_finite() {
            return Err(GeneratorError::InvalidObservedWager);
        }
        let observed_chips = f64::from(amount) * self.big_blind_chips as f64;
        let index = if (observed_chips - self.max_added_chips as f64).abs() <= 0.1 {
            count
        } else {
            let candidate =
                ((observed_chips - self.min_added_chips as f64) / self.chip_unit as f64).round();
            if candidate < 0.0 || candidate >= count as f64 {
                return Err(GeneratorError::InvalidObservedWager);
            }
            candidate as usize
        };
        if (observed_chips - self.chips(index) as f64).abs() > 0.1 {
            // The behavioral f32 observation no longer identifies the exact chip.
            return Err(GeneratorError::InvalidObservedWager);
        }
        Ok(index)
    }
}

/// Complete legal action space, supplied *before* the decision by the rules engine.
/// The wager maximum is an explicit all-in atom, never an arbitrary candidate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LegalActionSpace {
    pub fold: bool,
    pub check: bool,
    pub call: bool,
    pub bet: Option<WagerBounds>,
    pub raise: Option<WagerBounds>,
}

impl LegalActionSpace {
    fn allowed(self, kind: ActionKind) -> bool {
        match kind {
            ActionKind::Fold => self.fold,
            ActionKind::Check => self.check,
            ActionKind::Call => self.call,
            ActionKind::Bet => self.bet.is_some(),
            ActionKind::Raise => self.raise.is_some(),
        }
    }

    fn bounds(self, kind: ActionKind) -> Option<WagerBounds> {
        match kind {
            ActionKind::Bet => self.bet,
            ActionKind::Raise => self.raise,
            _ => None,
        }
    }

    pub fn validate(self, context: &DecisionContext) -> Result<(), GeneratorError> {
        context.validate().map_err(GeneratorError::InvalidContext)?;
        let actor = context.actor().map_err(GeneratorError::InvalidContext)?;
        self.validate_facts(
            context.pot_bb,
            actor.stack_remaining_bb,
            actor.committed_this_street_bb,
            context.current_bet_bb,
            context.to_call_bb,
        )
    }

    /// Invariants shared by live inference and stored Parquet observations.
    fn validate_facts(
        self,
        pot_bb: f32,
        stack_bb: f32,
        actor_street_bb: f32,
        current_bet_bb: f32,
        to_call_bb: f32,
    ) -> Result<(), GeneratorError> {
        if ![
            pot_bb,
            stack_bb,
            actor_street_bb,
            current_bet_bb,
            to_call_bb,
        ]
        .iter()
        .all(|x| x.is_finite() && *x >= 0.0)
            || stack_bb <= 0.0
            || ![
                self.fold,
                self.check,
                self.call,
                self.bet.is_some(),
                self.raise.is_some(),
            ]
            .contains(&true)
        {
            return Err(GeneratorError::InvalidLegalActions);
        }
        if (self.bet.is_some() || self.raise.is_some()) && pot_bb <= 0.0
            || self.check && to_call_bb != 0.0
            || self.call && to_call_bb <= 0.0
            || self.bet.is_some() && current_bet_bb != 0.0
            || self.raise.is_some() && current_bet_bb <= 0.0
        {
            return Err(GeneratorError::InvalidLegalActions);
        }
        for (kind, bounds) in [(ActionKind::Bet, self.bet), (ActionKind::Raise, self.raise)] {
            if let Some(bounds) = bounds {
                let count = bounds.ticks()?;
                if !close(bounds.amount(count), stack_bb)
                    || (kind == ActionKind::Raise
                        && actor_street_bb + bounds.amount(0) <= current_bet_bb)
                {
                    return Err(GeneratorError::InvalidWagerBounds);
                }
            }
        }
        Ok(())
    }
}

/// A differentiable linear transform with row-major [output, input] weights.
#[derive(Debug, Clone, PartialEq)]
pub struct DenseLayer {
    input: usize,
    output: usize,
    weights: Vec<f32>,
    bias: Vec<f32>,
}

impl DenseLayer {
    pub fn new(
        input: usize,
        output: usize,
        weights: Vec<f32>,
        bias: Vec<f32>,
    ) -> Result<Self, GeneratorError> {
        if input == 0
            || output == 0
            || input.checked_mul(output) != Some(weights.len())
            || output != bias.len()
        {
            return Err(GeneratorError::InvalidLayerShape);
        }
        if weights.iter().chain(bias.iter()).any(|v| !v.is_finite()) {
            return Err(GeneratorError::NonFiniteParameter);
        }
        Ok(Self {
            input,
            output,
            weights,
            bias,
        })
    }

    pub fn input_width(&self) -> usize {
        self.input
    }
    pub fn output_width(&self) -> usize {
        self.output
    }
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }
    pub fn biases(&self) -> &[f32] {
        &self.bias
    }

    fn forward(&self, input: &[f64]) -> Vec<f64> {
        debug_assert_eq!(input.len(), self.input);
        self.weights
            .chunks_exact(self.input)
            .zip(&self.bias)
            .map(|(row, b)| {
                row.iter()
                    .zip(input)
                    .fold(f64::from(*b), |s, (w, x)| s + f64::from(*w) * x)
            })
            .collect()
    }

    fn backward(
        &self,
        input: &[f64],
        output_gradient: &[f64],
        gradients: &mut LayerGradients,
    ) -> Vec<f64> {
        debug_assert_eq!(input.len(), self.input);
        debug_assert_eq!(output_gradient.len(), self.output);
        let mut input_gradient = vec![0.0; self.input];
        for (out, &g) in output_gradient.iter().enumerate() {
            gradients.bias[out] += g;
            for i in 0..self.input {
                gradients.weights[out * self.input + i] += g * input[i];
                input_gradient[i] += g * f64::from(self.weights[out * self.input + i]);
            }
        }
        input_gradient
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GeneratorParameters {
    pub state: DenseLayer,  // tanh(context || Z)
    pub action: DenseLayer, // five action logits
    pub sizing: DenseLayer, // conditional logistic location, scale, all-in logit
    pub timing: DenseLayer, // conditional log1p(ms) normal mean and scale
}

#[derive(Debug, Clone)]
pub struct NeuralBehaviourGenerator {
    latent: BehaviourLatentConfig,
    parameters: GeneratorParameters,
}

/// One subject decision read from the canonical behavioural features.
/// The current action is kept out of the pre-decision tensor by construction.
#[derive(Debug, Clone, PartialEq)]
pub struct GeneratorObservation {
    pub street: Street,
    pub actor_position: Position,
    pub context_values: [f32; PRE_ACTION_CONTEXT_WIDTH],
    pub context_present: [u8; PRE_ACTION_CONTEXT_WIDTH],
    pub observed: ObservedAction,
}

impl GeneratorObservation {
    fn validated_facts(&self, legal: LegalActionSpace) -> Result<(f32, f32), GeneratorError> {
        if self.context_present.iter().any(|&v| v != 0 && v != 1)
            || self
                .context_values
                .iter()
                .zip(&self.context_present)
                .any(|(&v, &present)| !v.is_finite() || v < 0.0 || (present == 0 && v != 0.0))
            || [0_usize, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13]
                .iter()
                .any(|&index| self.context_present[index] == 0)
        {
            return Err(GeneratorError::InvalidEncodedObservation);
        }
        let pot = self.context_values[0];
        let stack = self.context_values[1];
        let committed_street = self.context_values[2];
        let bet = self.context_values[4];
        let to_call = self.context_values[5];
        legal.validate_facts(pot, stack, committed_street, bet, to_call)?;
        if !legal.allowed(self.observed.kind) {
            return Err(GeneratorError::IllegalObservedAction);
        }
        let amount = match (self.observed.kind, self.observed.sizing) {
            (ActionKind::Fold, None) => 0.0,
            (ActionKind::Check, Some(sizing))
                if sizing.amount_over_pot_before == 0.0
                    && sizing.amount_over_stack_before == 0.0
                    && sizing.raise_increment_over_pot_before.is_none() =>
            {
                0.0
            }
            (ActionKind::Call, Some(sizing))
                if sizing.raise_increment_over_pot_before.is_none() =>
            {
                let value = stack * sizing.amount_over_stack_before;
                if !close(value, to_call.min(stack)) {
                    return Err(GeneratorError::InvalidEncodedObservation);
                }
                value
            }
            (ActionKind::Bet, Some(sizing)) if sizing.raise_increment_over_pot_before.is_none() => {
                stack * sizing.amount_over_stack_before
            }
            (ActionKind::Raise, Some(sizing))
                if sizing.raise_increment_over_pot_before.is_some() =>
            {
                stack * sizing.amount_over_stack_before
            }
            _ => return Err(GeneratorError::InvalidEncodedObservation),
        };
        if !amount.is_finite()
            || amount < 0.0
            || amount > stack + 1e-4 * stack
            || self.observed.is_all_in
                != (matches!(
                    self.observed.kind,
                    ActionKind::Call | ActionKind::Bet | ActionKind::Raise
                ) && close(amount, stack))
        {
            return Err(GeneratorError::InvalidEncodedObservation);
        }
        if let Some(ActionSizing {
            amount_over_pot_before,
            amount_over_stack_before,
            raise_increment_over_pot_before,
        }) = self.observed.sizing
        {
            if !amount_over_pot_before.is_finite()
                || !amount_over_stack_before.is_finite()
                || amount_over_pot_before < 0.0
                || amount_over_stack_before < 0.0
                || !close(amount, pot * amount_over_pot_before)
            {
                return Err(GeneratorError::InvalidEncodedObservation);
            }
            if self.observed.kind == ActionKind::Raise {
                let increment = committed_street + amount - bet;
                if increment <= 0.0
                    || !raise_increment_over_pot_before
                        .is_some_and(|ratio| ratio.is_finite() && close(increment, pot * ratio))
                {
                    return Err(GeneratorError::InvalidEncodedObservation);
                }
            }
        }
        if let Some(bounds) = legal.bounds(self.observed.kind) {
            bounds.index(amount)?;
        }
        Ok((stack, to_call))
    }
}

#[derive(Debug, Clone)]
struct ForwardPass {
    input: Vec<f64>,
    state: Vec<f64>,
    probabilities: [f64; ACTION_COUNT],
}

impl NeuralBehaviourGenerator {
    pub fn new(
        latent: BehaviourLatentConfig,
        parameters: GeneratorParameters,
    ) -> Result<Self, GeneratorError> {
        let hidden = parameters.state.output;
        let input_width = GENERATOR_CONTEXT_WIDTH
            .checked_add(latent.dimension())
            .ok_or(GeneratorError::InvalidModelShape)?;
        if latent.dimension() > 4096
            || parameters.state.input != input_width
            || parameters.action.input != hidden
            || parameters.action.output != ACTION_COUNT
            || parameters.sizing.input != hidden + GENERATOR_SIZING_WIDTH
            || parameters.sizing.output != 3
            || parameters.timing.input != hidden + GENERATOR_TIMING_WIDTH
            || parameters.timing.output != 2
        {
            return Err(GeneratorError::InvalidModelShape);
        }
        Ok(Self { latent, parameters })
    }

    /// Deterministically initialize one model; seed zero is rejected.
    pub fn initialize(
        latent: BehaviourLatentConfig,
        hidden: usize,
        seed: u64,
    ) -> Result<Self, GeneratorError> {
        if hidden == 0 || hidden > 4096 || latent.dimension() > 4096 || seed == 0 {
            return Err(GeneratorError::InvalidInitialization);
        }
        let mut rng = SeededWeights(seed);
        fn layer(rng: &mut SeededWeights, input: usize, output: usize) -> DenseLayer {
            let radius = (6.0 / (input + output) as f64).sqrt();
            let weights = (0..input * output)
                .map(|_| ((rng.next() * 2.0 - 1.0) * radius) as f32)
                .collect();
            DenseLayer::new(input, output, weights, vec![0.0; output])
                .expect("generated finite weights")
        }
        Self::new(
            latent,
            GeneratorParameters {
                state: layer(
                    &mut rng,
                    GENERATOR_CONTEXT_WIDTH + latent.dimension(),
                    hidden,
                ),
                action: layer(&mut rng, hidden, ACTION_COUNT),
                sizing: layer(&mut rng, hidden + GENERATOR_SIZING_WIDTH, 3),
                timing: layer(&mut rng, hidden + GENERATOR_TIMING_WIDTH, 2),
            },
        )
    }

    fn forward(
        &self,
        profile: &BehaviourProfile,
        context: &DecisionContext,
        legal: LegalActionSpace,
    ) -> Result<ForwardPass, GeneratorError> {
        legal.validate(context)?;
        self.forward_features(profile, context_features(context), legal)
    }

    fn forward_features(
        &self,
        profile: &BehaviourProfile,
        mut input: Vec<f64>,
        legal: LegalActionSpace,
    ) -> Result<ForwardPass, GeneratorError> {
        if profile.dimension() != self.latent.dimension() {
            return Err(GeneratorError::WrongLatentDimension {
                expected: self.latent.dimension(),
                actual: profile.dimension(),
            });
        }
        if input.len() != GENERATOR_CONTEXT_WIDTH {
            return Err(GeneratorError::InvalidEncodedObservation);
        }
        input.extend(profile.as_slice().iter().map(|&x| f64::from(x)));
        let state = self
            .parameters
            .state
            .forward(&input)
            .into_iter()
            .map(f64::tanh)
            .collect::<Vec<_>>();
        let logits = self.parameters.action.forward(&state);
        let probabilities = masked_softmax(&logits, legal)?;
        Ok(ForwardPass {
            input,
            state,
            probabilities,
        })
    }

    pub fn predict(
        &self,
        profile: &BehaviourProfile,
        context: &DecisionContext,
        legal: &LegalActionSpace,
    ) -> Result<BehaviourPolicy, GeneratorError> {
        let pass = self.forward(profile, context, *legal)?;
        Ok(BehaviourPolicy {
            context: context.clone(),
            legal: *legal,
            probabilities: pass.probabilities,
            state: pass.state,
            sizing_layer: self.parameters.sizing.clone(),
            timing_layer: self.parameters.timing.clone(),
        })
    }

    pub fn parameters(&self) -> &GeneratorParameters {
        &self.parameters
    }
}

impl BehaviourLatentModel for NeuralBehaviourGenerator {
    fn latent_config(&self) -> BehaviourLatentConfig {
        self.latent
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeneratorDraws {
    pub action: f64,
    pub sizing: f64,
    pub all_in: f64,
    pub timing_1: f64,
    pub timing_2: f64,
}

impl GeneratorDraws {
    fn validate(self) -> Result<(), GeneratorError> {
        if ![self.action, self.sizing, self.all_in, self.timing_2]
            .iter()
            .all(|&x| x.is_finite() && (0.0..1.0).contains(&x))
            || !self.timing_1.is_finite()
            || self.timing_1 <= 0.0
            || self.timing_1 >= 1.0
        {
            return Err(GeneratorError::InvalidRandomDraw);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimingDistribution {
    /// Mean of log(1 + decision_time_ms).
    pub log1p_ms_mean: f64,
    pub log1p_ms_std: f64,
}

impl TimingDistribution {
    pub fn sample_ms(self, u1: f64, u2: f64) -> Result<u32, GeneratorError> {
        if !u1.is_finite() || u1 <= 0.0 || u1 >= 1.0 || !(0.0..1.0).contains(&u2) {
            return Err(GeneratorError::InvalidRandomDraw);
        }
        let standard = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
        let ms = (self.log1p_ms_mean + self.log1p_ms_std * standard)
            .exp_m1()
            .max(0.0)
            .round();
        if !ms.is_finite() || ms > f64::from(u32::MAX) {
            return Err(GeneratorError::InvalidGeneratedTiming);
        }
        Ok(ms as u32)
    }
}

#[derive(Debug, Clone)]
pub struct BehaviourPolicy {
    context: DecisionContext,
    legal: LegalActionSpace,
    probabilities: [f64; ACTION_COUNT],
    state: Vec<f64>,
    sizing_layer: DenseLayer,
    timing_layer: DenseLayer,
}

impl BehaviourPolicy {
    pub fn action_probability(&self, kind: ActionKind) -> f64 {
        self.probabilities[usize::from(action_id(kind))]
    }

    pub fn action_probabilities(&self) -> &[f64; ACTION_COUNT] {
        &self.probabilities
    }

    pub fn timing_distribution(
        &self,
        kind: ActionKind,
        amount_bb: f32,
    ) -> Result<TimingDistribution, GeneratorError> {
        if !self.legal.allowed(kind) {
            return Err(GeneratorError::IllegalObservedAction);
        }
        let stack = self
            .context
            .actor()
            .map_err(GeneratorError::InvalidContext)?
            .stack_remaining_bb;
        let fraction = if stack > 0.0 {
            f64::from(amount_bb / stack)
        } else {
            0.0
        };
        if !amount_bb.is_finite() || amount_bb < 0.0 || amount_bb > stack {
            return Err(GeneratorError::InvalidObservedWager);
        }
        if let Some(bounds) = self.legal.bounds(kind) {
            bounds.index(amount_bb)?;
        }
        if matches!(kind, ActionKind::Fold | ActionKind::Check) && amount_bb != 0.0 {
            return Err(GeneratorError::InvalidObservedWager);
        }
        if kind == ActionKind::Call && !close(amount_bb, self.context.to_call_bb.min(stack)) {
            return Err(GeneratorError::InvalidObservedWager);
        }
        let input = timing_input(&self.state, kind, fraction);
        let raw = self.timing_layer.forward(&input);
        let distribution = TimingDistribution {
            log1p_ms_mean: raw[0],
            log1p_ms_std: softplus(raw[1]) + MIN_TIMING_SCALE,
        };
        if !distribution.log1p_ms_mean.is_finite() || !distribution.log1p_ms_std.is_finite() {
            return Err(GeneratorError::NonFiniteOutput);
        }
        Ok(distribution)
    }

    pub fn sample(&self, draws: GeneratorDraws) -> Result<ObservedAction, GeneratorError> {
        draws.validate()?;
        let mut cumulative = 0.0;
        let mut chosen = None;
        for kind in KINDS {
            cumulative += self.action_probability(kind);
            if draws.action < cumulative {
                chosen = Some(kind);
                break;
            }
        }
        let kind = chosen.ok_or(GeneratorError::InvalidProbabilityMass)?;
        let actor = self
            .context
            .actor()
            .map_err(GeneratorError::InvalidContext)?;
        let amount = match kind {
            ActionKind::Fold | ActionKind::Check => 0.0,
            ActionKind::Call => self.context.to_call_bb.min(actor.stack_remaining_bb),
            ActionKind::Bet | ActionKind::Raise => {
                let bounds = self
                    .legal
                    .bounds(kind)
                    .ok_or(GeneratorError::IllegalObservedAction)?;
                let raw = self.sizing_layer.forward(&sizing_input(&self.state, kind));
                sample_wager(bounds, &raw, draws.sizing, draws.all_in)?
            }
        };
        let timing = self
            .timing_distribution(kind, amount)?
            .sample_ms(draws.timing_1, draws.timing_2)?;
        let highest = match kind {
            ActionKind::Bet | ActionKind::Raise => actor.committed_this_street_bb + amount,
            _ => self.context.current_bet_bb,
        };
        ObservedAction::from_exact(
            &self.context,
            ExactAction {
                kind,
                is_all_in: matches!(kind, ActionKind::Call | ActionKind::Bet | ActionKind::Raise)
                    && close(amount, actor.stack_remaining_bb),
                amount_added_bb: amount,
                highest_bet_after_bb: highest,
                decision_time_ms: Some(timing),
            },
        )
        .map_err(GeneratorError::InvalidContext)
    }
}

const KINDS: [ActionKind; ACTION_COUNT] = [
    ActionKind::Fold,
    ActionKind::Check,
    ActionKind::Call,
    ActionKind::Bet,
    ActionKind::Raise,
];

fn context_features(ctx: &DecisionContext) -> Vec<f64> {
    let (numeric, mask) = encode_context(ctx);
    encoded_context_features(ctx.street, ctx.actor_position, &numeric, &mask)
}

fn encoded_context_features(
    street: Street,
    position: Position,
    numeric: &[f32; PRE_ACTION_CONTEXT_WIDTH],
    mask: &[u8; PRE_ACTION_CONTEXT_WIDTH],
) -> Vec<f64> {
    let mut values = vec![0.0; GENERATOR_CONTEXT_WIDTH];
    values[usize::from(street_id(street))] = 1.0;
    values[STREET_COUNT + usize::from(position_id(position))] = 1.0;
    let start = STREET_COUNT + POSITION_COUNT;
    for i in 0..PRE_ACTION_CONTEXT_WIDTH {
        // Identical normalization for observed Parquet and live game contexts.
        values[start + i] = f64::from(numeric[i]).ln_1p();
        values[start + PRE_ACTION_CONTEXT_WIDTH + i] = f64::from(mask[i]);
    }
    values
}

fn sizing_input(state: &[f64], kind: ActionKind) -> Vec<f64> {
    let mut input = state.to_vec();
    input.extend((0..ACTION_COUNT).map(|i| {
        if i == usize::from(action_id(kind)) {
            1.0
        } else {
            0.0
        }
    }));
    input
}

fn timing_input(state: &[f64], kind: ActionKind, fraction: f64) -> Vec<f64> {
    let mut input = sizing_input(state, kind);
    input.push(fraction);
    input
}

fn masked_softmax(
    logits: &[f64],
    legal: LegalActionSpace,
) -> Result<[f64; ACTION_COUNT], GeneratorError> {
    if logits.len() != ACTION_COUNT || logits.iter().any(|x| !x.is_finite()) {
        return Err(GeneratorError::NonFiniteOutput);
    }
    let maximum = KINDS
        .iter()
        .filter(|&&k| legal.allowed(k))
        .map(|&k| logits[usize::from(action_id(k))])
        .reduce(f64::max)
        .ok_or(GeneratorError::NoLegalActions)?;
    let mut values = [0.0; ACTION_COUNT];
    for kind in KINDS {
        let i = usize::from(action_id(kind));
        if legal.allowed(kind) {
            values[i] = (logits[i] - maximum).exp();
        }
    }
    let mass: f64 = values.iter().sum();
    if !mass.is_finite() || mass <= 0.0 {
        return Err(GeneratorError::InvalidProbabilityMass);
    }
    for v in &mut values {
        *v /= mass;
    }
    Ok(values)
}

fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}
fn softplus(x: f64) -> f64 {
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}
fn logit(x: f64) -> f64 {
    (x / (1.0 - x)).ln()
}
fn close(x: f32, y: f32) -> bool {
    (x - y).abs() <= 1e-4 * x.abs().max(y.abs()).max(1.0)
}

fn sample_wager(
    bounds: WagerBounds,
    raw: &[f64],
    size_draw: f64,
    allin_draw: f64,
) -> Result<f32, GeneratorError> {
    let nonallin = bounds.ticks()?;
    if nonallin == 0 {
        return Ok(bounds.amount(nonallin));
    }
    if allin_draw < sigmoid(raw[2]) {
        return Ok(bounds.amount(nonallin));
    }
    if nonallin == 1 {
        return Ok(bounds.amount(0));
    }
    let mu = sigmoid(raw[0]);
    let scale = softplus(raw[1]) + MIN_SIZING_SCALE;
    let cdf0 = sigmoid(-mu / scale);
    let cdf1 = sigmoid((1.0 - mu) / scale);
    let q = cdf0 + size_draw * (cdf1 - cdf0);
    if !(q > 0.0 && q < 1.0) {
        return Err(GeneratorError::InvalidProbabilityMass);
    }
    let normalized = (mu + scale * logit(q)).clamp(0.0, 1.0);
    let index = (normalized * (nonallin - 1) as f64).round() as usize;
    Ok(bounds.amount(index.min(nonallin - 1)))
}

#[derive(Debug, Clone)]
struct LayerGradients {
    weights: Vec<f64>,
    bias: Vec<f64>,
}
impl LayerGradients {
    fn new(layer: &DenseLayer) -> Self {
        Self {
            weights: vec![0.0; layer.weights.len()],
            bias: vec![0.0; layer.bias.len()],
        }
    }
}
#[derive(Debug, Clone)]
struct Gradients {
    state: LayerGradients,
    action: LayerGradients,
    sizing: LayerGradients,
    timing: LayerGradients,
}
impl Gradients {
    fn new(p: &GeneratorParameters) -> Self {
        Self {
            state: LayerGradients::new(&p.state),
            action: LayerGradients::new(&p.action),
            sizing: LayerGradients::new(&p.sizing),
            timing: LayerGradients::new(&p.timing),
        }
    }
}

/// Gradients accumulated against one immutable Generator parameter snapshot.
/// No optimizer step occurs until `Adam::apply_batch` is called. Per-decision
/// `TrainingLoss.latent_gradient` can be propagated to A before that update.
#[derive(Debug, Clone)]
pub struct GeneratorGradientBatch {
    parameter_snapshot: GeneratorParameters,
    gradients: Gradients,
    observations: usize,
    loss_total: f64,
    loss_action: f64,
    loss_sizing: f64,
    loss_timing: f64,
    latent_gradient_sum: Vec<f64>,
}

impl GeneratorGradientBatch {
    pub fn new(model: &NeuralBehaviourGenerator) -> Self {
        Self {
            parameter_snapshot: model.parameters.clone(),
            gradients: Gradients::new(&model.parameters),
            observations: 0,
            loss_total: 0.0,
            loss_action: 0.0,
            loss_sizing: 0.0,
            loss_timing: 0.0,
            latent_gradient_sum: vec![0.0; model.latent.dimension()],
        }
    }

    pub fn len(&self) -> usize {
        self.observations
    }

    pub fn is_empty(&self) -> bool {
        self.observations == 0
    }

    pub fn accumulate_encoded(
        &mut self,
        model: &NeuralBehaviourGenerator,
        profile: &BehaviourProfile,
        observation: &GeneratorObservation,
        legal: &LegalActionSpace,
    ) -> Result<TrainingLoss, GeneratorError> {
        self.check_model(model)?;
        let (loss, gradients) = loss_gradients_encoded(model, profile, observation, *legal)?;
        self.add_loss_gradients(loss, gradients)
    }

    pub fn accumulate(
        &mut self,
        model: &NeuralBehaviourGenerator,
        profile: &BehaviourProfile,
        context: &DecisionContext,
        legal: &LegalActionSpace,
        observed: &ObservedAction,
    ) -> Result<TrainingLoss, GeneratorError> {
        self.check_model(model)?;
        let (loss, gradients) = loss_gradients(model, profile, context, *legal, observed)?;
        self.add_loss_gradients(loss, gradients)
    }

    fn check_model(&self, model: &NeuralBehaviourGenerator) -> Result<(), GeneratorError> {
        if model.parameters != self.parameter_snapshot
            || model.latent.dimension() != self.latent_gradient_sum.len()
        {
            return Err(GeneratorError::BatchModelChanged);
        }
        Ok(())
    }

    fn add_loss_gradients(
        &mut self,
        loss: TrainingLoss,
        gradients: Gradients,
    ) -> Result<TrainingLoss, GeneratorError> {
        let observations = self
            .observations
            .checked_add(1)
            .ok_or(GeneratorError::BatchCountOverflow)?;
        let mut summed = self.gradients.clone();
        fn sum_layer(target: &mut LayerGradients, source: &LayerGradients) {
            for (total, delta) in target.weights.iter_mut().zip(&source.weights) {
                *total += delta;
            }
            for (total, delta) in target.bias.iter_mut().zip(&source.bias) {
                *total += delta;
            }
        }
        sum_layer(&mut summed.state, &gradients.state);
        sum_layer(&mut summed.action, &gradients.action);
        sum_layer(&mut summed.sizing, &gradients.sizing);
        sum_layer(&mut summed.timing, &gradients.timing);
        let totals = [
            self.loss_total + loss.total,
            self.loss_action + loss.action,
            self.loss_sizing + loss.sizing,
            self.loss_timing + loss.timing,
        ];
        let next_latent = self
            .latent_gradient_sum
            .iter()
            .zip(&loss.latent_gradient)
            .map(|(&current, &delta)| current + f64::from(delta))
            .collect::<Vec<_>>();
        let finite_layer = |layer: &LayerGradients| {
            layer
                .weights
                .iter()
                .chain(&layer.bias)
                .all(|value| value.is_finite())
        };
        if !totals.iter().all(|value| value.is_finite())
            || !next_latent.iter().all(|value| value.is_finite())
            || ![
                &summed.state,
                &summed.action,
                &summed.sizing,
                &summed.timing,
            ]
            .iter()
            .all(|layer| finite_layer(layer))
        {
            return Err(GeneratorError::NonFiniteGradient);
        }
        self.gradients = summed;
        self.observations = observations;
        self.loss_total = totals[0];
        self.loss_action = totals[1];
        self.loss_sizing = totals[2];
        self.loss_timing = totals[3];
        self.latent_gradient_sum = next_latent;
        Ok(loss)
    }

    pub fn mean_loss(&self) -> Result<TrainingLoss, GeneratorError> {
        if self.observations == 0 {
            return Err(GeneratorError::EmptyBatch);
        }
        let n = self.observations as f64;
        let latent_gradient = self
            .latent_gradient_sum
            .iter()
            .map(|&value| (value / n) as f32)
            .collect::<Vec<_>>();
        if latent_gradient.iter().any(|x| !x.is_finite()) {
            return Err(GeneratorError::NonFiniteGradient);
        }
        Ok(TrainingLoss {
            total: self.loss_total / n,
            action: self.loss_action / n,
            sizing: self.loss_sizing / n,
            timing: self.loss_timing / n,
            latent_gradient,
        })
    }

    fn averaged_gradients(&self) -> Result<Gradients, GeneratorError> {
        if self.observations == 0 {
            return Err(GeneratorError::EmptyBatch);
        }
        let mut result = self.gradients.clone();
        let n = self.observations as f64;
        for layer in [
            &mut result.state,
            &mut result.action,
            &mut result.sizing,
            &mut result.timing,
        ] {
            for g in layer.weights.iter_mut().chain(&mut layer.bias) {
                *g /= n;
            }
        }
        Ok(result)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TrainingLoss {
    pub total: f64,
    pub action: f64,
    pub sizing: f64,
    pub timing: f64,
    /// d(loss)/dZ; exposed for future subject-embedding training.
    pub latent_gradient: Vec<f32>,
}

/// Explicit, differentiable negative-log-likelihood and reverse-mode gradients.
/// The training caller supplies each observed decision's *actual* legal space.
fn loss_gradients(
    model: &NeuralBehaviourGenerator,
    profile: &BehaviourProfile,
    context: &DecisionContext,
    legal: LegalActionSpace,
    action: &ObservedAction,
) -> Result<(TrainingLoss, Gradients), GeneratorError> {
    action
        .validate_with_context(context)
        .map_err(GeneratorError::InvalidContext)?;
    if !legal.allowed(action.kind) {
        return Err(GeneratorError::IllegalObservedAction);
    }
    legal.validate(context)?;
    let actor = context.actor().map_err(GeneratorError::InvalidContext)?;
    loss_gradients_features(
        model,
        profile,
        context_features(context),
        legal,
        action,
        actor.stack_remaining_bb,
        context.to_call_bb,
    )
}

fn loss_gradients_encoded(
    model: &NeuralBehaviourGenerator,
    profile: &BehaviourProfile,
    observation: &GeneratorObservation,
    legal: LegalActionSpace,
) -> Result<(TrainingLoss, Gradients), GeneratorError> {
    let (stack, to_call) = observation.validated_facts(legal)?;
    loss_gradients_features(
        model,
        profile,
        encoded_context_features(
            observation.street,
            observation.actor_position,
            &observation.context_values,
            &observation.context_present,
        ),
        legal,
        &observation.observed,
        stack,
        to_call,
    )
}

fn loss_gradients_features(
    model: &NeuralBehaviourGenerator,
    profile: &BehaviourProfile,
    encoded: Vec<f64>,
    legal: LegalActionSpace,
    action: &ObservedAction,
    actor_stack: f32,
    to_call: f32,
) -> Result<(TrainingLoss, Gradients), GeneratorError> {
    let forward = model.forward_features(profile, encoded, legal)?;
    let k = usize::from(action_id(action.kind));
    let probability = forward.probabilities[k];
    if !probability.is_finite() || probability <= 0.0 {
        return Err(GeneratorError::InvalidProbabilityMass);
    }
    let action_loss = -probability.ln();
    let mut grads = Gradients::new(&model.parameters);
    let mut state_gradient = vec![0.0; forward.state.len()];
    let mut logits_gradient = forward.probabilities;
    logits_gradient[k] -= 1.0;
    add(
        &mut state_gradient,
        &model
            .parameters
            .action
            .backward(&forward.state, &logits_gradient, &mut grads.action),
    );

    let amount = match action.kind {
        ActionKind::Fold | ActionKind::Check => 0.0,
        ActionKind::Call => to_call.min(actor_stack),
        ActionKind::Bet | ActionKind::Raise => {
            actor_stack
                * action
                    .sizing
                    .ok_or(GeneratorError::InvalidObservedWager)?
                    .amount_over_stack_before
        }
    };
    let mut sizing_loss = 0.0;
    if let Some(bounds) = legal.bounds(action.kind) {
        let index = bounds.index(amount)?;
        let input = sizing_input(&forward.state, action.kind);
        let raw = model.parameters.sizing.forward(&input);
        let (loss, grad) = sizing_nll(bounds, index, &raw)?;
        sizing_loss = loss;
        let prev = model
            .parameters
            .sizing
            .backward(&input, &grad, &mut grads.sizing);
        add(&mut state_gradient, &prev[..forward.state.len()]);
    }

    let mut timing_loss = 0.0;
    if let Some(ms) = action.decision_time_ms {
        let fraction = if actor_stack > 0.0 {
            f64::from(amount / actor_stack)
        } else {
            0.0
        };
        let input = timing_input(&forward.state, action.kind, fraction);
        let raw = model.parameters.timing.forward(&input);
        let sigma = softplus(raw[1]) + MIN_TIMING_SCALE;
        let residual = raw[0] - f64::from(ms).ln_1p();
        timing_loss = 0.5 * (residual / sigma).powi(2) + sigma.ln() + 0.5 * (2.0 * PI).ln();
        let grad = [
            residual / (sigma * sigma),
            (1.0 / sigma - residual * residual / sigma.powi(3)) * sigmoid(raw[1]),
        ];
        let prev = model
            .parameters
            .timing
            .backward(&input, &grad, &mut grads.timing);
        add(&mut state_gradient, &prev[..forward.state.len()]);
    }
    for (g, &s) in state_gradient.iter_mut().zip(&forward.state) {
        *g *= 1.0 - s * s;
    }
    let input_gradient =
        model
            .parameters
            .state
            .backward(&forward.input, &state_gradient, &mut grads.state);
    let latent_gradient = input_gradient[GENERATOR_CONTEXT_WIDTH..]
        .iter()
        .map(|&x| x as f32)
        .collect();
    let total = action_loss + sizing_loss + timing_loss;
    if !total.is_finite() || input_gradient.iter().any(|x| !x.is_finite()) {
        return Err(GeneratorError::NonFiniteGradient);
    }
    Ok((
        TrainingLoss {
            total,
            action: action_loss,
            sizing: sizing_loss,
            timing: timing_loss,
            latent_gradient,
        },
        grads,
    ))
}

fn add(target: &mut [f64], source: &[f64]) {
    for (a, b) in target.iter_mut().zip(source) {
        *a += b;
    }
}

/// Mixed distribution: exact atom at the stack maximum, logistic-binned mass
/// at every other *legal chip increment*. No arbitrarily sampled candidate grid.
fn sizing_nll(
    bounds: WagerBounds,
    index: usize,
    raw: &[f64],
) -> Result<(f64, [f64; 3]), GeneratorError> {
    let count = bounds.ticks()?;
    if count == 0 {
        return Ok((0.0, [0.0; 3]));
    }
    let allin = index == count;
    let atom = sigmoid(raw[2]);
    let mut loss = if allin {
        softplus(-raw[2])
    } else {
        softplus(raw[2])
    };
    let mut gradient = [0.0, 0.0, atom - if allin { 1.0 } else { 0.0 }];
    if allin || count == 1 {
        return Ok((loss, gradient));
    }

    let denominator = (count - 1) as f64;
    let low = ((index as f64 - 0.5) / denominator).max(0.0);
    let high = ((index as f64 + 0.5) / denominator).min(1.0);
    let mu = sigmoid(raw[0]);
    let sigma = softplus(raw[1]) + MIN_SIZING_SCALE;
    let cdf = |x: f64| sigmoid((x - mu) / sigma);
    let (a, b, c, d) = (cdf(low), cdf(high), cdf(0.0), cdf(1.0));
    let mass = b - a;
    let normalizer = d - c;
    if mass <= 0.0 || normalizer <= 0.0 || !mass.is_finite() {
        return Err(GeneratorError::InvalidProbabilityMass);
    }
    loss -= (mass / normalizer).ln();
    let dmu = |_x: f64, f: f64| -f * (1.0 - f) / sigma;
    let dsigma = |x: f64, f: f64| -(x - mu) * f * (1.0 - f) / (sigma * sigma);
    let dloss_dmu =
        -((dmu(high, b) - dmu(low, a)) / mass - (dmu(1.0, d) - dmu(0.0, c)) / normalizer);
    let dloss_dsigma = -((dsigma(high, b) - dsigma(low, a)) / mass
        - (dsigma(1.0, d) - dsigma(0.0, c)) / normalizer);
    gradient[0] = dloss_dmu * mu * (1.0 - mu);
    gradient[1] = dloss_dsigma * sigmoid(raw[1]);
    if !loss.is_finite() || gradient.iter().any(|v| !v.is_finite()) {
        return Err(GeneratorError::NonFiniteGradient);
    }
    Ok((loss, gradient))
}

/// Adam optimizer owns all of its moments, with no second model implementation.
#[derive(Debug, Clone)]
pub struct Adam {
    learning_rate: f64,
    step: u64,
    moments: AdamMoments,
}
#[derive(Debug, Clone)]
struct LayerMoments {
    mw: Vec<f64>,
    vw: Vec<f64>,
    mb: Vec<f64>,
    vb: Vec<f64>,
}
impl LayerMoments {
    fn new(layer: &DenseLayer) -> Self {
        Self {
            mw: vec![0.0; layer.weights.len()],
            vw: vec![0.0; layer.weights.len()],
            mb: vec![0.0; layer.bias.len()],
            vb: vec![0.0; layer.bias.len()],
        }
    }
    fn matches(&self, layer: &DenseLayer) -> bool {
        self.mw.len() == layer.weights.len()
            && self.vw.len() == layer.weights.len()
            && self.mb.len() == layer.bias.len()
            && self.vb.len() == layer.bias.len()
    }
    fn update(
        &mut self,
        layer: &mut DenseLayer,
        gradients: &LayerGradients,
        rate: f64,
        beta1_correction: f64,
        beta2_correction: f64,
    ) -> Result<(), GeneratorError> {
        for ((w, g), (m, v)) in layer
            .weights
            .iter_mut()
            .zip(&gradients.weights)
            .zip(self.mw.iter_mut().zip(self.vw.iter_mut()))
        {
            adam_value(w, *g, m, v, rate, beta1_correction, beta2_correction)?;
        }
        for ((b, g), (m, v)) in layer
            .bias
            .iter_mut()
            .zip(&gradients.bias)
            .zip(self.mb.iter_mut().zip(self.vb.iter_mut()))
        {
            adam_value(b, *g, m, v, rate, beta1_correction, beta2_correction)?;
        }
        Ok(())
    }
}
fn adam_value(
    weight: &mut f32,
    grad: f64,
    m: &mut f64,
    v: &mut f64,
    rate: f64,
    correction1: f64,
    correction2: f64,
) -> Result<(), GeneratorError> {
    if !grad.is_finite() {
        return Err(GeneratorError::NonFiniteGradient);
    }
    *m = 0.9 * *m + 0.1 * grad;
    *v = 0.999 * *v + 0.001 * grad * grad;
    let updated =
        f64::from(*weight) - rate * (*m / correction1) / ((*v / correction2).sqrt() + 1e-8);
    if !updated.is_finite() || updated.abs() > f64::from(f32::MAX) {
        return Err(GeneratorError::NonFiniteParameter);
    }
    *weight = updated as f32;
    Ok(())
}
#[derive(Debug, Clone)]
struct AdamMoments {
    state: LayerMoments,
    action: LayerMoments,
    sizing: LayerMoments,
    timing: LayerMoments,
}
impl AdamMoments {
    fn matches(&self, p: &GeneratorParameters) -> bool {
        self.state.matches(&p.state)
            && self.action.matches(&p.action)
            && self.sizing.matches(&p.sizing)
            && self.timing.matches(&p.timing)
    }
    fn new(p: &GeneratorParameters) -> Self {
        Self {
            state: LayerMoments::new(&p.state),
            action: LayerMoments::new(&p.action),
            sizing: LayerMoments::new(&p.sizing),
            timing: LayerMoments::new(&p.timing),
        }
    }
    fn update(
        &mut self,
        p: &mut GeneratorParameters,
        g: &Gradients,
        rate: f64,
        step: u64,
    ) -> Result<(), GeneratorError> {
        let c1 = 1.0 - 0.9f64.powf(step as f64);
        let c2 = 1.0 - 0.999f64.powf(step as f64);
        self.state.update(&mut p.state, &g.state, rate, c1, c2)?;
        self.action.update(&mut p.action, &g.action, rate, c1, c2)?;
        self.sizing.update(&mut p.sizing, &g.sizing, rate, c1, c2)?;
        self.timing.update(&mut p.timing, &g.timing, rate, c1, c2)?;
        Ok(())
    }
}
impl Adam {
    pub fn new(
        model: &NeuralBehaviourGenerator,
        learning_rate: f64,
    ) -> Result<Self, GeneratorError> {
        if !learning_rate.is_finite() || learning_rate <= 0.0 {
            return Err(GeneratorError::InvalidLearningRate);
        }
        Ok(Self {
            learning_rate,
            step: 0,
            moments: AdamMoments::new(&model.parameters),
        })
    }

    /// One raw game decision: NLL, reverse-mode backprop, then Adam.
    pub fn train_step(
        &mut self,
        model: &mut NeuralBehaviourGenerator,
        profile: &BehaviourProfile,
        context: &DecisionContext,
        legal: &LegalActionSpace,
        observed: &ObservedAction,
    ) -> Result<TrainingLoss, GeneratorError> {
        let (loss, gradients) = loss_gradients(model, profile, context, *legal, observed)?;
        self.apply(model, loss, gradients)
    }

    /// The same training path, fed by canonical pre-action Parquet tensors.
    pub fn train_encoded(
        &mut self,
        model: &mut NeuralBehaviourGenerator,
        profile: &BehaviourProfile,
        observation: &GeneratorObservation,
        legal: &LegalActionSpace,
    ) -> Result<TrainingLoss, GeneratorError> {
        let (loss, gradients) = loss_gradients_encoded(model, profile, observation, *legal)?;
        self.apply(model, loss, gradients)
    }

    /// Average all gradients at fixed G parameters and apply Adam exactly once.
    pub fn apply_batch(
        &mut self,
        model: &mut NeuralBehaviourGenerator,
        batch: GeneratorGradientBatch,
    ) -> Result<TrainingLoss, GeneratorError> {
        batch.check_model(model)?;
        let loss = batch.mean_loss()?;
        let gradients = batch.averaged_gradients()?;
        self.apply(model, loss, gradients)
    }

    /// Convenience batch path for canonical Parquet decisions. The caller can
    /// also use GeneratorGradientBatch directly to route each dL/dZ back to A.
    pub fn train_encoded_batch(
        &mut self,
        model: &mut NeuralBehaviourGenerator,
        observations: &[(&BehaviourProfile, &GeneratorObservation, &LegalActionSpace)],
    ) -> Result<TrainingLoss, GeneratorError> {
        let mut batch = GeneratorGradientBatch::new(model);
        for (profile, observation, legal) in observations {
            batch.accumulate_encoded(model, profile, observation, legal)?;
        }
        self.apply_batch(model, batch)
    }

    fn apply(
        &mut self,
        model: &mut NeuralBehaviourGenerator,
        loss: TrainingLoss,
        gradients: Gradients,
    ) -> Result<TrainingLoss, GeneratorError> {
        if !self.moments.matches(&model.parameters) {
            return Err(GeneratorError::OptimizerModelMismatch);
        }
        let step = self
            .step
            .checked_add(1)
            .ok_or(GeneratorError::OptimizerStepOverflow)?;
        let mut parameters = model.parameters.clone();
        let mut moments = self.moments.clone();
        moments.update(&mut parameters, &gradients, self.learning_rate, step)?;
        model.parameters = parameters;
        self.moments = moments;
        self.step = step;
        Ok(loss)
    }

    pub fn evaluate(
        &self,
        model: &NeuralBehaviourGenerator,
        profile: &BehaviourProfile,
        context: &DecisionContext,
        legal: &LegalActionSpace,
        observed: &ObservedAction,
    ) -> Result<TrainingLoss, GeneratorError> {
        loss_gradients(model, profile, context, *legal, observed).map(|(loss, _)| loss)
    }

    pub fn evaluate_encoded(
        &self,
        model: &NeuralBehaviourGenerator,
        profile: &BehaviourProfile,
        observation: &GeneratorObservation,
        legal: &LegalActionSpace,
    ) -> Result<TrainingLoss, GeneratorError> {
        loss_gradients_encoded(model, profile, observation, *legal).map(|(loss, _)| loss)
    }

    pub fn steps(&self) -> u64 {
        self.step
    }
}

impl NeuralBehaviourGenerator {
    /// Read-only joint NLL (action, conditional sizing, timing) and dL/dZ for
    /// personalization of Z1. Does not update any Generator parameters.
    pub fn evaluate_observation_encoded(
        &self,
        profile: &BehaviourProfile,
        observation: &GeneratorObservation,
        legal: &LegalActionSpace,
    ) -> Result<TrainingLoss, GeneratorError> {
        loss_gradients_encoded(self, profile, observation, *legal).map(|(loss, _)| loss)
    }

    /// Canonical encoded pre-action probabilities, without reading the target response.
    pub fn action_probabilities_encoded(
        &self,
        profile: &BehaviourProfile,
        observation: &GeneratorObservation,
        legal: &LegalActionSpace,
    ) -> Result<[f64; ACTION_COUNT], GeneratorError> {
        if observation
            .context_present
            .iter()
            .any(|&p| p != 0 && p != 1)
            || observation
                .context_values
                .iter()
                .zip(&observation.context_present)
                .any(|(&v, &p)| !v.is_finite() || v < 0.0 || (p == 0 && v != 0.0))
            || [0_usize, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13]
                .iter()
                .any(|&i| observation.context_present[i] == 0)
        {
            return Err(GeneratorError::InvalidEncodedObservation);
        }
        legal.validate_facts(
            observation.context_values[0],
            observation.context_values[1],
            observation.context_values[2],
            observation.context_values[4],
            observation.context_values[5],
        )?;
        Ok(self
            .forward_features(
                profile,
                encoded_context_features(
                    observation.street,
                    observation.actor_position,
                    &observation.context_values,
                    &observation.context_present,
                ),
                *legal,
            )?
            .probabilities)
    }

    pub(crate) fn checkpoint_value(&self) -> Value {
        json!({"dimension":self.latent.dimension(),
               "state":dense_value(&self.parameters.state),
               "action":dense_value(&self.parameters.action),
               "sizing":dense_value(&self.parameters.sizing),
               "timing":dense_value(&self.parameters.timing)})
    }

    pub(crate) fn from_checkpoint(value: &Value) -> Result<Self, String> {
        exact_keys(value, &["dimension", "state", "action", "sizing", "timing"])?;
        let config = BehaviourLatentConfig::new(dimension(&value["dimension"])?)
            .map_err(|e| e.to_string())?;
        Self::new(
            config,
            GeneratorParameters {
                state: dense_from_value(&value["state"])?,
                action: dense_from_value(&value["action"])?,
                sizing: dense_from_value(&value["sizing"])?,
                timing: dense_from_value(&value["timing"])?,
            },
        )
        .map_err(|e| e.to_string())
    }
}

impl LayerMoments {
    fn checkpoint_value(&self) -> Value {
        json!({"mw":self.mw,"vw":self.vw,"mb":self.mb,"vb":self.vb})
    }
    fn from_checkpoint(value: &Value, layer: &DenseLayer) -> Result<Self, String> {
        exact_keys(value, &["mw", "vw", "mb", "vb"])?;
        let moments = Self {
            mw: float_array(&value["mw"])?,
            vw: float_array(&value["vw"])?,
            mb: float_array(&value["mb"])?,
            vb: float_array(&value["vb"])?,
        };
        if !moments.matches(layer) || moments.vw.iter().chain(&moments.vb).any(|v| *v < 0.0) {
            return Err("invalid generator optimizer moments".into());
        }
        Ok(moments)
    }
}

impl Adam {
    pub(crate) fn checkpoint_value(&self) -> Value {
        json!({"learning_rate":self.learning_rate,"steps":self.step,
            "state":self.moments.state.checkpoint_value(),
            "action":self.moments.action.checkpoint_value(),
            "sizing":self.moments.sizing.checkpoint_value(),
            "timing":self.moments.timing.checkpoint_value()})
    }
    pub(crate) fn from_checkpoint(
        value: &Value,
        model: &NeuralBehaviourGenerator,
    ) -> Result<Self, String> {
        exact_keys(
            value,
            &[
                "learning_rate",
                "steps",
                "state",
                "action",
                "sizing",
                "timing",
            ],
        )?;
        let rate = number(&value["learning_rate"])?;
        let mut optimizer = Self::new(model, rate).map_err(|e| e.to_string())?;
        optimizer.step = unsigned(&value["steps"])?;
        optimizer.moments = AdamMoments {
            state: LayerMoments::from_checkpoint(&value["state"], &model.parameters.state)?,
            action: LayerMoments::from_checkpoint(&value["action"], &model.parameters.action)?,
            sizing: LayerMoments::from_checkpoint(&value["sizing"], &model.parameters.sizing)?,
            timing: LayerMoments::from_checkpoint(&value["timing"], &model.parameters.timing)?,
        };
        Ok(optimizer)
    }
}

struct SeededWeights(u64);
impl SeededWeights {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum GeneratorError {
    InvalidContext(BehaviourEventError),
    InvalidLayerShape,
    InvalidModelShape,
    InvalidInitialization,
    NonFiniteParameter,
    NonFiniteOutput,
    NonFiniteGradient,
    WrongLatentDimension { expected: usize, actual: usize },
    NoLegalActions,
    InvalidLegalActions,
    InvalidEncodedObservation,
    InvalidWagerBounds,
    InvalidObservedWager,
    IllegalObservedAction,
    InvalidRandomDraw,
    InvalidProbabilityMass,
    InvalidGeneratedTiming,
    InvalidLearningRate,
    OptimizerStepOverflow,
    OptimizerModelMismatch,
    BatchModelChanged,
    BatchCountOverflow,
    EmptyBatch,
}
impl fmt::Display for GeneratorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for GeneratorError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::behavioural::{PlayerState, PlayerStatus, Position, Street};

    fn ctx() -> DecisionContext {
        DecisionContext {
            street: Street::Flop,
            actor_position: Position::Button,
            pot_bb: 10.0,
            players: vec![
                PlayerState {
                    position: Position::Button,
                    status: PlayerStatus::Active,
                    stack_remaining_bb: 80.0,
                    committed_this_street_bb: 0.0,
                    committed_this_hand_bb: 5.0,
                },
                PlayerState {
                    position: Position::BigBlind,
                    status: PlayerStatus::Active,
                    stack_remaining_bb: 80.0,
                    committed_this_street_bb: 0.0,
                    committed_this_hand_bb: 5.0,
                },
            ],
            current_bet_bb: 0.0,
            to_call_bb: 0.0,
            min_raise_to_bb: Some(1.0),
            players_yet_to_act: 1,
            bets_and_raises_this_street: 0,
            calls_this_street: 0,
            folds_this_street: 0,
            actor_has_acted_this_street: false,
            actor_aggression_streak_before: 0,
            table_aggression_streak_before: 0,
            last_aggressor: None,
            preflop_last_aggressor_position: None,
        }
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
    fn model() -> NeuralBehaviourGenerator {
        NeuralBehaviourGenerator::initialize(BehaviourLatentConfig::new(2).unwrap(), 8, 42).unwrap()
    }
    fn profile() -> BehaviourProfile {
        BehaviourLatentConfig::new(2)
            .unwrap()
            .profile(vec![0.0, 0.0])
            .unwrap()
    }
    fn observed_bet(amount: f32) -> ObservedAction {
        ObservedAction::from_exact(
            &ctx(),
            ExactAction {
                kind: ActionKind::Bet,
                is_all_in: amount == 80.0,
                amount_added_bb: amount,
                highest_bet_after_bb: amount,
                decision_time_ms: Some(600),
            },
        )
        .unwrap()
    }

    #[test]
    fn batch_calculates_at_fixed_weights_and_applies_one_optimizer_step() {
        let mut generator = model();
        let mut adam = Adam::new(&generator, 0.001).unwrap();
        let original = generator.parameters().clone();
        let mut batch = GeneratorGradientBatch::new(&generator);
        let first = batch
            .accumulate(&generator, &profile(), &ctx(), &legal(), &observed_bet(5.0))
            .unwrap();
        let second = batch
            .accumulate(&generator, &profile(), &ctx(), &legal(), &observed_bet(7.0))
            .unwrap();
        assert_eq!(batch.len(), 2);
        assert_eq!(*generator.parameters(), original);
        assert_eq!(adam.steps(), 0);
        let avg = batch.mean_loss().unwrap();
        assert!((avg.total - (first.total + second.total) / 2.0).abs() < 1e-12);
        let result = adam.apply_batch(&mut generator, batch).unwrap();
        assert!((result.total - avg.total).abs() < 1e-12);
        assert_eq!(adam.steps(), 1);
        assert_ne!(*generator.parameters(), original);
    }

    #[test]
    fn batch_rejects_empty_or_mutated_generator() {
        let mut generator = model();
        let mut adam = Adam::new(&generator, 0.001).unwrap();
        let batch = GeneratorGradientBatch::new(&generator);
        assert!(matches!(
            adam.apply_batch(&mut generator, batch),
            Err(GeneratorError::EmptyBatch)
        ));
        let mut batch = GeneratorGradientBatch::new(&generator);
        batch
            .accumulate(&generator, &profile(), &ctx(), &legal(), &observed_bet(5.0))
            .unwrap();
        adam.train_step(
            &mut generator,
            &profile(),
            &ctx(),
            &legal(),
            &observed_bet(5.0),
        )
        .unwrap();
        let before = generator.parameters().clone();
        assert!(matches!(
            adam.apply_batch(&mut generator, batch),
            Err(GeneratorError::BatchModelChanged)
        ));
        assert_eq!(*generator.parameters(), before);
    }

    #[test]
    fn legal_action_mask_and_probability_normalization() {
        let p = model().predict(&profile(), &ctx(), &legal()).unwrap();
        assert!(p.action_probability(ActionKind::Check) > 0.0);
        assert!(p.action_probability(ActionKind::Bet) > 0.0);
        assert_eq!(p.action_probability(ActionKind::Fold), 0.0);
        assert_eq!(p.action_probability(ActionKind::Raise), 0.0);
        assert!((p.action_probabilities().iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn sizing_uses_exact_chip_ticks_and_allin_atom() {
        let bounds = legal().bet.unwrap();
        assert_eq!(bounds.ticks().unwrap(), 79);
        for i in 0..=79 {
            assert_eq!(bounds.index(bounds.amount(i)).unwrap(), i);
        }
        let odd_allin = WagerBounds {
            min_added_chips: 10,
            max_added_chips: 157,
            chip_unit: 5,
            big_blind_chips: 10,
        };
        assert_eq!(odd_allin.ticks().unwrap(), 30);
        assert_eq!(odd_allin.chips(29), 155);
        assert_eq!(odd_allin.chips(30), 157);
        assert_eq!(odd_allin.index(odd_allin.amount(30)).unwrap(), 30);
        let p = model().predict(&profile(), &ctx(), &legal()).unwrap();
        for seed in [0.0, 0.1, 0.5, 0.9] {
            let result = p
                .sample(GeneratorDraws {
                    action: p.action_probability(ActionKind::Check) + 1e-8,
                    sizing: seed,
                    all_in: 0.999,
                    timing_1: 0.5,
                    timing_2: 0.5,
                })
                .unwrap();
            assert_eq!(result.kind, ActionKind::Bet);
            assert!(!result.is_all_in);
            assert!(result.sizing.unwrap().amount_over_stack_before < 1.0);
        }
        let result = p
            .sample(GeneratorDraws {
                action: p.action_probability(ActionKind::Check) + 1e-8,
                sizing: 0.0,
                all_in: 0.0,
                timing_1: 0.5,
                timing_2: 0.5,
            })
            .unwrap();
        assert!(result.is_all_in);
    }

    #[test]
    fn check_is_zero_and_fold_is_absent() {
        let p = model().predict(&profile(), &ctx(), &legal()).unwrap();
        let check = p
            .sample(GeneratorDraws {
                action: 0.0,
                sizing: 0.5,
                all_in: 0.5,
                timing_1: 0.5,
                timing_2: 0.5,
            })
            .unwrap();
        assert_eq!(check.kind, ActionKind::Check);
        assert_eq!(check.sizing.unwrap().amount_over_pot_before, 0.0);
        let mut with_fold = legal();
        with_fold.fold = true;
        with_fold.check = false;
        with_fold.bet = None;
        let fold = model()
            .predict(&profile(), &ctx(), &with_fold)
            .unwrap()
            .sample(GeneratorDraws {
                action: 0.0,
                sizing: 0.5,
                all_in: 0.5,
                timing_1: 0.5,
                timing_2: 0.5,
            })
            .unwrap();
        assert_eq!(fold.kind, ActionKind::Fold);
        assert_eq!(fold.sizing, None);
    }

    #[test]
    fn finite_differences_verify_sizing_gradients() {
        let bounds = legal().bet.unwrap();
        let raw = [0.2, -0.4, 0.1];
        let (_, grad) = sizing_nll(bounds, 17, &raw).unwrap();
        for j in 0..3 {
            let mut plus = raw;
            let mut minus = raw;
            plus[j] += 1e-5;
            minus[j] -= 1e-5;
            let numeric = (sizing_nll(bounds, 17, &plus).unwrap().0
                - sizing_nll(bounds, 17, &minus).unwrap().0)
                / 2e-5;
            assert!(
                (numeric - grad[j]).abs() < 1e-5,
                "{j}: {numeric} vs {}",
                grad[j]
            );
        }
    }

    #[test]
    fn adam_reduces_observed_decision_loss() {
        let mut g = model();
        let mut adam = Adam::new(&g, 0.002).unwrap();
        let action = observed_bet(20.0);
        let initial = adam
            .evaluate(&g, &profile(), &ctx(), &legal(), &action)
            .unwrap()
            .total;
        for _ in 0..150 {
            adam.train_step(&mut g, &profile(), &ctx(), &legal(), &action)
                .unwrap();
        }
        let final_loss = adam
            .evaluate(&g, &profile(), &ctx(), &legal(), &action)
            .unwrap()
            .total;
        assert!(final_loss < initial, "{final_loss} >= {initial}");
        assert_eq!(adam.steps(), 150);
    }

    #[test]
    fn latent_gradient_matches_finite_differences() {
        let g = model();
        let adam = Adam::new(&g, 0.001).unwrap();
        let action = observed_bet(20.0);
        let base = vec![0.0f32, 0.0f32];
        let reference = adam
            .evaluate(&g, &profile(), &ctx(), &legal(), &action)
            .unwrap();
        for i in 0..2 {
            let mut plus = base.clone();
            let mut minus = base.clone();
            plus[i] += 0.005;
            minus[i] -= 0.005;
            let config = BehaviourLatentConfig::new(2).unwrap();
            let lplus = adam
                .evaluate(
                    &g,
                    &config.profile(plus).unwrap(),
                    &ctx(),
                    &legal(),
                    &action,
                )
                .unwrap()
                .total;
            let lminus = adam
                .evaluate(
                    &g,
                    &config.profile(minus).unwrap(),
                    &ctx(),
                    &legal(),
                    &action,
                )
                .unwrap()
                .total;
            let numeric = (lplus - lminus) / 0.01;
            assert!(
                (numeric - f64::from(reference.latent_gradient[i])).abs() < 0.02,
                "latent {i}: {numeric} != {}",
                reference.latent_gradient[i]
            );
        }
    }

    #[test]
    fn parquet_features_and_live_context_have_identical_nll_and_latent_gradient() {
        let game = ctx();
        let observed = observed_bet(5.0);
        let (numeric, mask) = encode_context(&game);
        let encoded = GeneratorObservation {
            street: game.street,
            actor_position: game.actor_position,
            context_values: numeric,
            context_present: mask,
            observed: observed.clone(),
        };
        let model = model();
        let adam = Adam::new(&model, 0.001).unwrap();
        let live = adam
            .evaluate(&model, &profile(), &game, &legal(), &observed)
            .unwrap();
        let stored = adam
            .evaluate_encoded(&model, &profile(), &encoded, &legal())
            .unwrap();
        assert!((live.total - stored.total).abs() < 1e-12);
        assert_eq!(live.latent_gradient, stored.latent_gradient);
    }

    #[test]
    fn gradients_reach_latent_z() {
        let g = model();
        let (_, gradients) =
            loss_gradients(&g, &profile(), &ctx(), legal(), &observed_bet(20.0)).unwrap();
        assert!(gradients.state.weights.iter().any(|x| x.abs() > 0.0));
        let grad = Adam::new(&g, 0.001)
            .unwrap()
            .evaluate(&g, &profile(), &ctx(), &legal(), &observed_bet(20.0))
            .unwrap();
        assert_eq!(grad.latent_gradient.len(), 2);
        assert!(grad.latent_gradient.iter().all(|v| v.is_finite()));
    }
}
