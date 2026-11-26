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

// Debian's LP64 OpenBLAS CBLAS ABI. All buffers are owned Rust allocations;
// every call below checks its matrix/vector dimensions before entering C.
const ROW_MAJOR: i32 = 101;
const NO_TRANS: i32 = 111;
const TRANS: i32 = 112;

#[link(name = "openblas")]
unsafe extern "C" {
    fn cblas_dgemv(
        layout: i32,
        trans: i32,
        m: i32,
        n: i32,
        alpha: f64,
        a: *const f64,
        lda: i32,
        x: *const f64,
        incx: i32,
        beta: f64,
        y: *mut f64,
        incy: i32,
    );
    fn cblas_dger(
        layout: i32,
        m: i32,
        n: i32,
        alpha: f64,
        x: *const f64,
        incx: i32,
        y: *const f64,
        incy: i32,
        a: *mut f64,
        lda: i32,
    );
    fn cblas_dgemm(
        layout: i32,
        trans_a: i32,
        trans_b: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f64,
        a: *const f64,
        lda: i32,
        b: *const f64,
        ldb: i32,
        beta: f64,
        c: *mut f64,
        ldc: i32,
    );
}

const STREET_COUNT: usize = 4;
const POSITION_COUNT: usize = 7;
pub const ACTION_COUNT: usize = 5;
// One all-in logit plus (intercept, slope, readout) per shared ReLU unit.
// Reuse the generator's hidden width, not a fixed count of candidate sizes.
fn sizing_outputs(hidden: usize) -> usize {
    1 + 3 * hidden
}
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
    // Exact f32 -> f64 conversion, refreshed only when Adam changes weights.
    weights_f64: Vec<f64>,
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
            || input > i32::MAX as usize
            || output > i32::MAX as usize
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
            weights_f64: weights.iter().map(|&w| f64::from(w)).collect(),
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
        assert_eq!(input.len(), self.input);
        assert_eq!(self.weights_f64.len(), self.input * self.output);
        let mut output = self.bias.iter().map(|&b| f64::from(b)).collect::<Vec<_>>();
        assert_eq!(output.len(), self.output);
        let m = self.output as i32;
        let n = self.input as i32;
        // SAFETY: contiguous row-major m*n matrix, n input and m output
        // elements, unit strides; dimensions validated in DenseLayer::new.
        unsafe {
            cblas_dgemv(
                ROW_MAJOR,
                NO_TRANS,
                m,
                n,
                1.0,
                self.weights_f64.as_ptr(),
                n,
                input.as_ptr(),
                1,
                1.0,
                output.as_mut_ptr(),
                1,
            );
        }
        output
    }

    fn backward(
        &self,
        input: &[f64],
        output_gradient: &[f64],
        gradients: Option<&mut LayerGradients>,
    ) -> Vec<f64> {
        assert_eq!(input.len(), self.input);
        assert_eq!(output_gradient.len(), self.output);
        assert_eq!(self.weights_f64.len(), self.input * self.output);
        let m = self.output as i32;
        let n = self.input as i32;
        let mut input_gradient = vec![0.0; self.input];
        // SAFETY: transpose of the same validated matrix; m input and n output.
        unsafe {
            cblas_dgemv(
                ROW_MAJOR,
                TRANS,
                m,
                n,
                1.0,
                self.weights_f64.as_ptr(),
                n,
                output_gradient.as_ptr(),
                1,
                0.0,
                input_gradient.as_mut_ptr(),
                1,
            );
        }
        if let Some(gradients) = gradients {
            assert_eq!(gradients.weights.len(), self.input * self.output);
            assert_eq!(gradients.bias.len(), self.output);
            // SAFETY: outer product of m and n contiguous vectors into m*n.
            unsafe {
                cblas_dger(
                    ROW_MAJOR,
                    m,
                    n,
                    1.0,
                    output_gradient.as_ptr(),
                    1,
                    input.as_ptr(),
                    1,
                    gradients.weights.as_mut_ptr(),
                    n,
                );
            }
            for (b, &delta) in gradients.bias.iter_mut().zip(output_gradient) {
                *b += delta;
            }
        }
        input_gradient
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GeneratorParameters {
    pub state: DenseLayer,  // tanh(context || Z)
    pub action: DenseLayer, // five action logits
    pub sizing: DenseLayer, // all-in logit and shared BB-bin scorer coefficients
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
    /// Check exact pre-action legality and exact observed wager against the
    /// stored chip grid, without requiring a Generator or calculating gradients.
    pub fn validate_legality(&self, legal: LegalActionSpace) -> Result<(), GeneratorError> {
        self.validated_facts(legal).map(|_| ())
    }

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
            || parameters.sizing.output != sizing_outputs(hidden)
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
        let state = layer(
            &mut rng,
            GENERATOR_CONTEXT_WIDTH + latent.dimension(),
            hidden,
        );
        let action = layer(&mut rng, hidden, ACTION_COUNT);
        let input = hidden + GENERATOR_SIZING_WIDTH;
        // Preserve V9's initial state, action, all-in and timing parameters.
        let initial_allin = layer(&mut rng, input, 3);
        let timing = layer(&mut rng, hidden + GENERATOR_TIMING_WIDTH, 2);
        let mut sizing = layer(&mut rng, input, sizing_outputs(hidden));
        sizing.weights[..input].copy_from_slice(&initial_allin.weights[2 * input..]);
        sizing.weights_f64[..input].copy_from_slice(&initial_allin.weights_f64[2 * input..]);
        Self::new(
            latent,
            GeneratorParameters {
                state,
                action,
                sizing,
                timing,
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
fn close(x: f32, y: f32) -> bool {
    (x - y).abs() <= 1e-4 * x.abs().max(y.abs()).max(1.0)
}

// Nearest tenth of a BB (halfway ties upward), computed in exact integers.
// Ordinary bins contain at least one legal non-all-in chip amount. All-in is
// separate even when it shares a BB cell with an ordinary amount.
#[derive(Clone, Copy)]
struct BbSizingGrid {
    bounds: WagerBounds,
    ticks: usize,
    count: usize,
    first: i128,
    dense: bool,
}

impl BbSizingGrid {
    fn new(bounds: WagerBounds) -> Result<Self, GeneratorError> {
        let ticks = bounds.ticks()?;
        let first = Self::bin(bounds, 0);
        let dense = 10 * i128::from(bounds.chip_unit) <= i128::from(bounds.big_blind_chips);
        let count = if ticks == 0 {
            0
        } else if dense {
            usize::try_from(Self::bin(bounds, ticks - 1) - first + 1)
                .map_err(|_| GeneratorError::InvalidWagerBounds)?
        } else {
            ticks
        };
        Ok(Self {
            bounds,
            ticks,
            count,
            first,
            dense,
        })
    }

    fn bin(bounds: WagerBounds, tick: usize) -> i128 {
        let bb = i128::from(bounds.big_blind_chips);
        (20 * i128::from(bounds.chips(tick)) + bb) / (2 * bb)
    }

    fn bin_at(self, position: usize) -> i128 {
        if self.dense {
            self.first + position as i128
        } else {
            Self::bin(self.bounds, position)
        }
    }

    fn position(self, tick: usize) -> Result<usize, GeneratorError> {
        if tick >= self.ticks {
            return Err(GeneratorError::InvalidObservedWager);
        }
        let position = if self.dense {
            usize::try_from(Self::bin(self.bounds, tick) - self.first)
                .map_err(|_| GeneratorError::InvalidObservedWager)?
        } else {
            tick
        };
        if position >= self.count {
            return Err(GeneratorError::InvalidObservedWager);
        }
        Ok(position)
    }

    // Project the bin centre onto the legal ordinary chip grid. On an exact
    // tie choose the lower tick. This is the definition of execution, not a
    // fallback. A selected bin is guaranteed to have at least one legal tick.
    fn representative(self, position: usize) -> Result<usize, GeneratorError> {
        if position >= self.count {
            return Err(GeneratorError::InvalidObservedWager);
        }
        if !self.dense {
            return Ok(position);
        }
        let numerator = self.bin_at(position) * i128::from(self.bounds.big_blind_chips)
            - 10 * i128::from(self.bounds.min_added_chips);
        let denominator = 10 * i128::from(self.bounds.chip_unit);
        let numerator = numerator.max(0);
        let nearest = numerator / denominator
            + if 2 * (numerator % denominator) > denominator {
                1
            } else {
                0
            };
        let tick = usize::try_from(nearest.min(self.ticks as i128 - 1))
            .map_err(|_| GeneratorError::InvalidWagerBounds)?;
        if self.position(tick)? != position {
            return Err(GeneratorError::InvalidWagerBounds);
        }
        Ok(tick)
    }
}

fn sizing_softplus(x: f64) -> f64 {
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}

// The scorer is shared by every legal bin: E(x) = sum_i v_i ReLU(b_i+w_i*x).
// b,w,v are generated from the same (state, action) input. No fixed knot,
// centre, width, or stack-dependent parameter matrix. x = log1p(BB).
fn sizing_width(raw: &[f64]) -> Result<usize, GeneratorError> {
    if raw.len() < 4 || !(raw.len() - 1).is_multiple_of(3) || raw.iter().any(|x| !x.is_finite()) {
        return Err(GeneratorError::NonFiniteOutput);
    }
    Ok((raw.len() - 1) / 3)
}

struct SizingShape {
    coordinates: Vec<f64>,
    log_probs: Vec<f64>,
    active: Vec<std::ops::Range<usize>>,
}

impl SizingShape {
    fn new(grid: BbSizingGrid, raw: &[f64]) -> Result<Self, GeneratorError> {
        let width = sizing_width(raw)?;
        if grid.count == 0 {
            return Err(GeneratorError::InvalidProbabilityMass);
        }
        let coordinates: Vec<f64> = (0..grid.count)
            .map(|i| (grid.bin_at(i) as f64 / 10.0).ln_1p())
            .collect();
        if coordinates.iter().any(|x| !x.is_finite())
            || coordinates.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(GeneratorError::InvalidWagerBounds);
        }
        let n = grid.count;
        // Each ReLU is active on a prefix or suffix of the ordered grid.
        // Add its affine contribution to that range by two boundary events.
        // A prefix scan then evaluates all logits in O(N + H log N).
        let mut events = vec![(0.0, 0.0); n + 1];
        let mut active = Vec::with_capacity(width);
        for i in 0..width {
            let b = raw[1 + i];
            let w = raw[1 + width + i];
            let v = raw[1 + 2 * width + i];
            if ![
                b + w * coordinates[0],
                b + w * coordinates[n - 1],
                v * b,
                v * w,
            ]
            .iter()
            .all(|x| x.is_finite())
            {
                return Err(GeneratorError::NonFiniteOutput);
            }
            let range = if w >= 0.0 {
                coordinates.partition_point(|&x| b + w * x <= 0.0)..n
            } else {
                0..coordinates.partition_point(|&x| b + w * x > 0.0)
            };
            if !range.is_empty() {
                events[range.start].0 += v * b;
                events[range.start].1 += v * w;
                events[range.end].0 -= v * b;
                events[range.end].1 -= v * w;
            }
            active.push(range);
        }
        let mut intercept = 0.0;
        let mut slope = 0.0;
        let mut log_probs = Vec::with_capacity(n);
        for (&x, &(db, dw)) in coordinates.iter().zip(&events) {
            intercept += db;
            slope += dw;
            let score = intercept + slope * x;
            if !score.is_finite() {
                return Err(GeneratorError::NonFiniteOutput);
            }
            log_probs.push(score);
        }
        let maximum = log_probs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let sum = log_probs.iter().map(|x| (x - maximum).exp()).sum::<f64>();
        if !sum.is_finite() || sum <= 0.0 {
            return Err(GeneratorError::InvalidProbabilityMass);
        }
        let log_sum = sum.ln();
        for x in &mut log_probs {
            *x = (*x - maximum) - log_sum;
            if !x.is_finite() {
                return Err(GeneratorError::InvalidProbabilityMass);
            }
        }
        Ok(Self {
            coordinates,
            log_probs,
            active,
        })
    }

    fn log_probability(&self, position: usize) -> Result<f64, GeneratorError> {
        self.log_probs
            .get(position)
            .copied()
            .ok_or(GeneratorError::InvalidObservedWager)
    }

    fn sample(&self, draw: f64) -> Result<usize, GeneratorError> {
        if !draw.is_finite() || !(0.0..1.0).contains(&draw) {
            return Err(GeneratorError::InvalidRandomDraw);
        }
        // Normalize the floating-point sum used by the sampler itself.
        let total = self.log_probs.iter().map(|p| p.exp()).sum::<f64>();
        if !total.is_finite() || total <= 0.0 {
            return Err(GeneratorError::InvalidProbabilityMass);
        }
        let mut cumulative = 0.0;
        for (i, &p) in self.log_probs.iter().enumerate() {
            cumulative += p.exp();
            if draw < cumulative / total {
                return Ok(i);
            }
        }
        Err(GeneratorError::InvalidProbabilityMass)
    }

    // dNLL/dE_j = q_j - 1(j=observed). Every neuron's derivative only needs
    // the sum of that residual and x times that residual on its active range.
    // Prefix/suffix scans avoid a dense bins-by-hidden Jacobian.
    fn gradient(
        &self,
        observed: usize,
        raw: &[f64],
        grad: &mut [f64],
    ) -> Result<(), GeneratorError> {
        let n = self.log_probs.len();
        let width = self.active.len();
        if observed >= n || raw.len() != sizing_outputs(width) || grad.len() != raw.len() {
            return Err(GeneratorError::InvalidObservedWager);
        }
        let residuals: Vec<f64> = self
            .log_probs
            .iter()
            .enumerate()
            .map(|(j, p)| p.exp() - if j == observed { 1.0 } else { 0.0 })
            .collect();
        self.score_gradient(&residuals, raw, grad)
    }

    fn score_gradient(
        &self,
        residuals: &[f64],
        raw: &[f64],
        grad: &mut [f64],
    ) -> Result<(), GeneratorError> {
        let n = self.log_probs.len();
        let width = self.active.len();
        if residuals.len() != n || raw.len() != sizing_outputs(width) || grad.len() != raw.len() {
            return Err(GeneratorError::InvalidObservedWager);
        }
        let mut prefix = vec![(0.0, 0.0); n + 1];
        let mut suffix = vec![(0.0, 0.0); n + 1];
        for (j, &d) in residuals.iter().enumerate() {
            prefix[j + 1] = (prefix[j].0 + d, prefix[j].1 + d * self.coordinates[j]);
        }
        for (j, &d) in residuals.iter().enumerate().rev() {
            suffix[j] = (
                suffix[j + 1].0 + d,
                suffix[j + 1].1 + d * self.coordinates[j],
            );
        }
        for (i, range) in self.active.iter().enumerate() {
            if range.is_empty() {
                continue;
            }
            let (sum, xsum) = if range.start == 0 {
                prefix[range.end]
            } else {
                suffix[range.start]
            };
            let b = raw[1 + i];
            let w = raw[1 + width + i];
            let v = raw[1 + 2 * width + i];
            grad[1 + i] = v * sum;
            grad[1 + width + i] = v * xsum;
            grad[1 + 2 * width + i] = b * sum + w * xsum;
        }
        if grad.iter().any(|x| !x.is_finite()) {
            return Err(GeneratorError::NonFiniteGradient);
        }
        Ok(())
    }
}

fn sample_wager(
    bounds: WagerBounds,
    raw: &[f64],
    size_draw: f64,
    allin_draw: f64,
) -> Result<f32, GeneratorError> {
    sizing_width(raw)?;
    let grid = BbSizingGrid::new(bounds)?;
    if grid.ticks == 0 || allin_draw < sigmoid(raw[0]) {
        return Ok(bounds.amount(grid.ticks));
    }
    if grid.count == 1 {
        return Ok(bounds.amount(grid.representative(0)?));
    }
    let shape = SizingShape::new(grid, raw)?;
    let selected = grid.representative(shape.sample(size_draw)?)?;
    Ok(bounds.amount(selected))
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

/// Rows of inputs X and output derivatives D from one fixed G snapshot.
/// Weight gradients are D^T X; no per-decision weight matrix is allocated.
#[derive(Debug, Clone, Default)]
struct LayerProducts {
    rows: usize,
    inputs: Vec<f64>,
    deltas: Vec<f64>,
}
impl LayerProducts {
    fn record(&mut self, input: &[f64], delta: &[f64]) -> Result<(), GeneratorError> {
        if !input.iter().chain(delta).all(|x| x.is_finite()) {
            return Err(GeneratorError::NonFiniteGradient);
        }
        self.rows = self
            .rows
            .checked_add(1)
            .ok_or(GeneratorError::BatchCountOverflow)?;
        self.inputs.extend_from_slice(input);
        self.deltas.extend_from_slice(delta);
        Ok(())
    }

    fn append(&mut self, other: Self) {
        self.rows += other.rows;
        self.inputs.extend(other.inputs);
        self.deltas.extend(other.deltas);
    }

    fn add_to(
        &self,
        layer: &DenseLayer,
        gradients: &mut LayerGradients,
    ) -> Result<(), GeneratorError> {
        if self.rows == 0 {
            return Ok(()); // No observed target for this conditional head.
        }
        if self.rows.checked_mul(layer.input) != Some(self.inputs.len())
            || self.rows.checked_mul(layer.output) != Some(self.deltas.len())
            || layer.input.checked_mul(layer.output) != Some(gradients.weights.len())
            || gradients.bias.len() != layer.output
        {
            return Err(GeneratorError::InvalidLayerShape);
        }
        let k = i32::try_from(self.rows).map_err(|_| GeneratorError::BatchCountOverflow)?;
        let m = layer.output as i32;
        let n = layer.input as i32;
        // SAFETY: D is k*m, X is k*n, output m*n, all contiguous row-major.
        // Transpose D, accumulate D^T X in the existing gradient buffer.
        unsafe {
            cblas_dgemm(
                ROW_MAJOR,
                TRANS,
                NO_TRANS,
                m,
                n,
                k,
                1.0,
                self.deltas.as_ptr(),
                m,
                self.inputs.as_ptr(),
                n,
                1.0,
                gradients.weights.as_mut_ptr(),
                n,
            );
        }
        for row in self.deltas.chunks_exact(layer.output) {
            for (b, &d) in gradients.bias.iter_mut().zip(row) {
                *b += d;
            }
        }
        if !gradients
            .weights
            .iter()
            .chain(&gradients.bias)
            .all(|x| x.is_finite())
        {
            return Err(GeneratorError::NonFiniteGradient);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Default)]
struct MatrixProducts {
    state: LayerProducts,
    action: LayerProducts,
    sizing: LayerProducts,
    timing: LayerProducts,
}
impl MatrixProducts {
    fn append(&mut self, other: Self) {
        self.state.append(other.state);
        self.action.append(other.action);
        self.sizing.append(other.sizing);
        self.timing.append(other.timing);
    }
    fn add_to(
        &self,
        model: &GeneratorParameters,
        grads: &mut Gradients,
    ) -> Result<(), GeneratorError> {
        self.state.add_to(&model.state, &mut grads.state)?;
        self.action.add_to(&model.action, &mut grads.action)?;
        self.sizing.add_to(&model.sizing, &mut grads.sizing)?;
        self.timing.add_to(&model.timing, &mut grads.timing)?;
        Ok(())
    }
}

/// Gradients accumulated against one immutable Generator parameter snapshot.
/// No optimizer step occurs until `Adam::apply_batch` is called. Per-decision
/// `TrainingLoss.latent_gradient` can be propagated to A before that update.
#[derive(Debug, Clone)]
pub struct GeneratorGradientBatch {
    model_snapshot: NeuralBehaviourGenerator,
    gradients: Gradients,
    matrix_products: MatrixProducts,
    observations: usize,
    total_weight: f64,
    loss_total: f64,
    loss_action: f64,
    loss_sizing: f64,
    loss_timing: f64,
    latent_gradient_sum: Vec<f64>,
}

impl GeneratorGradientBatch {
    pub fn new(model: &NeuralBehaviourGenerator) -> Self {
        Self {
            model_snapshot: model.clone(),
            gradients: Gradients::new(&model.parameters),
            matrix_products: MatrixProducts::default(),
            observations: 0,
            total_weight: 0.0,
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
        self.accumulate_weighted_encoded(model, profile, observation, legal, 1.0)
    }

    /// Score against this batch's owned immutable snapshot. The caller cannot
    /// supply a different G; Adam::apply_batch still checks the live model.
    /// This removes a full parameter equality scan for each training decision.
    pub fn accumulate_snapshot_encoded(
        &mut self,
        profile: &BehaviourProfile,
        observation: &GeneratorObservation,
        legal: &LegalActionSpace,
    ) -> Result<TrainingLoss, GeneratorError> {
        let mut products = MatrixProducts::default();
        let loss = backprop_encoded(
            &self.model_snapshot,
            profile,
            observation,
            *legal,
            SizingObjective::CrpsOverPot(f64::from(observation.context_values[0])),
            None,
            Some(&mut products),
        )?;
        let observations = self
            .observations
            .checked_add(1)
            .ok_or(GeneratorError::BatchCountOverflow)?;
        if observations > i32::MAX as usize {
            return Err(GeneratorError::BatchCountOverflow);
        }
        let total_weight = self.total_weight + 1.0;
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
        if !total_weight.is_finite() || !totals.iter().chain(&next_latent).all(|x| x.is_finite()) {
            return Err(GeneratorError::NonFiniteGradient);
        }
        // All per-row checks passed. GEMM is deferred to apply_batch.
        self.matrix_products.append(products);
        self.observations = observations;
        self.total_weight = total_weight;
        self.loss_total = totals[0];
        self.loss_action = totals[1];
        self.loss_sizing = totals[2];
        self.loss_timing = totals[3];
        self.latent_gradient_sum = next_latent;
        Ok(loss)
    }

    /// Accumulate a positive, explicit objective weight. The returned loss and
    /// latent gradient are unweighted; callers may use them for a separate A objective.
    pub fn accumulate_weighted_encoded(
        &mut self,
        model: &NeuralBehaviourGenerator,
        profile: &BehaviourProfile,
        observation: &GeneratorObservation,
        legal: &LegalActionSpace,
        weight: f64,
    ) -> Result<TrainingLoss, GeneratorError> {
        self.check_model(model)?;
        let (loss, gradients) = loss_gradients_encoded(model, profile, observation, *legal)?;
        self.add_loss_gradients(loss, gradients, weight)
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
        self.add_loss_gradients(loss, gradients, 1.0)
    }

    fn check_model(&self, model: &NeuralBehaviourGenerator) -> Result<(), GeneratorError> {
        if model.parameters != self.model_snapshot.parameters
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
        weight: f64,
    ) -> Result<TrainingLoss, GeneratorError> {
        let total_weight = self.total_weight + weight;
        if !weight.is_finite() || weight <= 0.0 || !total_weight.is_finite() {
            return Err(GeneratorError::InvalidBatchWeight);
        }
        let observations = self
            .observations
            .checked_add(1)
            .ok_or(GeneratorError::BatchCountOverflow)?;
        // Reuse the consumed per-observation buffer as the candidate sum.
        // The previous accumulator is untouched until all checks pass.
        let mut summed = gradients;
        fn sum_layer(candidate: &mut LayerGradients, previous: &LayerGradients, weight: f64) {
            for (delta, total) in candidate.weights.iter_mut().zip(&previous.weights) {
                *delta = *total + weight * *delta;
            }
            for (delta, total) in candidate.bias.iter_mut().zip(&previous.bias) {
                *delta = *total + weight * *delta;
            }
        }
        sum_layer(&mut summed.state, &self.gradients.state, weight);
        sum_layer(&mut summed.action, &self.gradients.action, weight);
        sum_layer(&mut summed.sizing, &self.gradients.sizing, weight);
        sum_layer(&mut summed.timing, &self.gradients.timing, weight);
        let totals = [
            self.loss_total + weight * loss.total,
            self.loss_action + weight * loss.action,
            self.loss_sizing + weight * loss.sizing,
            self.loss_timing + weight * loss.timing,
        ];
        let next_latent = self
            .latent_gradient_sum
            .iter()
            .zip(&loss.latent_gradient)
            .map(|(&current, &delta)| current + weight * f64::from(delta))
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
        self.total_weight = total_weight;
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
        let n = self.total_weight;
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
        self.matrix_products
            .add_to(&self.model_snapshot.parameters, &mut result)?;
        let n = self.total_weight;
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
/// Mixed training score, or joint NLL when returned by evaluate_observation_encoded.
pub struct TrainingLoss {
    pub total: f64,
    pub action: f64,
    pub sizing: f64,
    pub timing: f64,
    /// d(loss)/dZ; exposed for future subject-embedding training.
    pub latent_gradient: Vec<f32>,
}

/// Read-only NLL decomposition without constructing or backpropagating model gradients.
/// The joint total uses the exact same action mask, wager bins, and timing
/// density as Z1 adaptation; training uses CRPS for sizing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScoringLoss {
    pub total: f64,
    pub action: f64,
    pub sizing: f64,
    pub timing: f64,
}

#[derive(Clone, Copy)]
enum SizingObjective {
    Nll,
    CrpsOverPot(f64),
}

/// Differentiable training objective and reverse-mode gradients.
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
        SizingObjective::CrpsOverPot(f64::from(context.pot_bb)),
    )
}

fn loss_gradients_encoded(
    model: &NeuralBehaviourGenerator,
    profile: &BehaviourProfile,
    observation: &GeneratorObservation,
    legal: LegalActionSpace,
) -> Result<(TrainingLoss, Gradients), GeneratorError> {
    let mut gradients = Gradients::new(&model.parameters);
    let loss = backprop_encoded(
        model,
        profile,
        observation,
        legal,
        SizingObjective::CrpsOverPot(f64::from(observation.context_values[0])),
        Some(&mut gradients),
        None,
    )?;
    Ok((loss, gradients))
}

fn backprop_encoded(
    model: &NeuralBehaviourGenerator,
    profile: &BehaviourProfile,
    observation: &GeneratorObservation,
    legal: LegalActionSpace,
    objective: SizingObjective,
    gradients: Option<&mut Gradients>,
    products: Option<&mut MatrixProducts>,
) -> Result<TrainingLoss, GeneratorError> {
    let (stack, to_call) = observation.validated_facts(legal)?;
    backprop_features(
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
        objective,
        gradients,
        products,
    )
}

#[allow(clippy::too_many_arguments)]
fn loss_gradients_features(
    model: &NeuralBehaviourGenerator,
    profile: &BehaviourProfile,
    encoded: Vec<f64>,
    legal: LegalActionSpace,
    action: &ObservedAction,
    actor_stack: f32,
    to_call: f32,
    objective: SizingObjective,
) -> Result<(TrainingLoss, Gradients), GeneratorError> {
    let mut gradients = Gradients::new(&model.parameters);
    let loss = backprop_features(
        model,
        profile,
        encoded,
        legal,
        action,
        actor_stack,
        to_call,
        objective,
        Some(&mut gradients),
        None,
    )?;
    Ok((loss, gradients))
}

#[allow(clippy::too_many_arguments)]
fn backprop_features(
    model: &NeuralBehaviourGenerator,
    profile: &BehaviourProfile,
    encoded: Vec<f64>,
    legal: LegalActionSpace,
    action: &ObservedAction,
    actor_stack: f32,
    to_call: f32,
    objective: SizingObjective,
    mut grads: Option<&mut Gradients>,
    mut products: Option<&mut MatrixProducts>,
) -> Result<TrainingLoss, GeneratorError> {
    let forward = model.forward_features(profile, encoded, legal)?;
    let k = usize::from(action_id(action.kind));
    let probability = forward.probabilities[k];
    if !probability.is_finite() || probability <= 0.0 {
        return Err(GeneratorError::InvalidProbabilityMass);
    }
    let action_loss = -probability.ln();
    let mut state_gradient = vec![0.0; forward.state.len()];
    let mut logits_gradient = forward.probabilities;
    logits_gradient[k] -= 1.0;
    if let Some(p) = products.as_deref_mut() {
        p.action.record(&forward.state, &logits_gradient)?;
    }
    add(
        &mut state_gradient,
        &model.parameters.action.backward(
            &forward.state,
            &logits_gradient,
            grads.as_deref_mut().map(|g| &mut g.action),
        ),
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
        let (loss, grad) = match objective {
            SizingObjective::Nll => sizing_nll(bounds, index, &raw)?,
            SizingObjective::CrpsOverPot(pot) => sizing_crps(bounds, index, &raw, pot)?,
        };
        sizing_loss = loss;
        if let Some(p) = products.as_deref_mut() {
            p.sizing.record(&input, &grad)?;
        }
        let prev = model.parameters.sizing.backward(
            &input,
            &grad,
            grads.as_deref_mut().map(|g| &mut g.sizing),
        );
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
        if let Some(p) = products.as_deref_mut() {
            p.timing.record(&input, &grad)?;
        }
        let prev = model.parameters.timing.backward(
            &input,
            &grad,
            grads.as_deref_mut().map(|g| &mut g.timing),
        );
        add(&mut state_gradient, &prev[..forward.state.len()]);
    }
    for (g, &s) in state_gradient.iter_mut().zip(&forward.state) {
        *g *= 1.0 - s * s;
    }
    if let Some(p) = products {
        p.state.record(&forward.input, &state_gradient)?;
    }
    let input_gradient = model.parameters.state.backward(
        &forward.input,
        &state_gradient,
        grads.map(|g| &mut g.state),
    );
    let latent_gradient = input_gradient[GENERATOR_CONTEXT_WIDTH..]
        .iter()
        .map(|&x| x as f32)
        .collect();
    let total = action_loss + sizing_loss + timing_loss;
    if !total.is_finite() || input_gradient.iter().any(|x| !x.is_finite()) {
        return Err(GeneratorError::NonFiniteGradient);
    }
    Ok(TrainingLoss {
        total,
        action: action_loss,
        sizing: sizing_loss,
        timing: timing_loss,
        latent_gradient,
    })
}

fn add(target: &mut [f64], source: &[f64]) {
    for (a, b) in target.iter_mut().zip(source) {
        *a += b;
    }
}

/// Exact mixed mass: all-in Bernoulli, then softmax over occupied 0.1 BB bins.
fn sizing_objective(
    bounds: WagerBounds,
    index: usize,
    raw: &[f64],
    mut gradient: Option<&mut [f64]>,
) -> Result<f64, GeneratorError> {
    sizing_width(raw)?;
    let grid = BbSizingGrid::new(bounds)?;
    if index > grid.ticks {
        return Err(GeneratorError::InvalidObservedWager);
    }
    if grid.ticks == 0 {
        return Ok(0.0);
    }
    let allin = index == grid.ticks;
    let mut loss = sizing_softplus(if allin { -raw[0] } else { raw[0] });
    if let Some(g) = gradient.as_deref_mut() {
        g[0] = if allin {
            -sigmoid(-raw[0])
        } else {
            sigmoid(raw[0])
        };
    }
    if !allin && grid.count > 1 {
        let position = grid.position(index)?;
        let shape = SizingShape::new(grid, raw)?;
        loss -= shape.log_probability(position)?;
        if let Some(g) = gradient {
            shape.gradient(position, raw, g)?;
        }
    }
    if !loss.is_finite() {
        return Err(GeneratorError::NonFiniteGradient);
    }
    Ok(loss)
}

fn sizing_nll(
    bounds: WagerBounds,
    index: usize,
    raw: &[f64],
) -> Result<(f64, Vec<f64>), GeneratorError> {
    let mut gradient = vec![0.0; raw.len()];
    let loss = sizing_objective(bounds, index, raw, Some(&mut gradient))?;
    Ok((loss, gradient))
}

/// Exact CRPS of the executable distribution, divided by the pre-action pot.
/// The observed amount is the exact legal amount in BB, not a chip class.
/// Integer differences avoid subtracting large nearly equal floating amounts.
/// The all-in atom is included in the same CDF, not an extra Bernoulli loss.
fn sizing_crps(
    bounds: WagerBounds,
    index: usize,
    raw: &[f64],
    pot_bb: f64,
) -> Result<(f64, Vec<f64>), GeneratorError> {
    sizing_width(raw)?;
    let grid = BbSizingGrid::new(bounds)?;
    if index > grid.ticks {
        return Err(GeneratorError::InvalidObservedWager);
    }
    let pot_chips = pot_bb * bounds.big_blind_chips as f64;
    if !pot_chips.is_finite() || pot_chips <= 0.0 {
        return Err(GeneratorError::InvalidLegalActions);
    }
    let mut gradient = vec![0.0; raw.len()];
    if grid.ticks == 0 {
        return Ok((0.0, gradient));
    }
    let shape = SizingShape::new(grid, raw)?;
    let q: Vec<f64> = shape.log_probs.iter().map(|p| p.exp()).collect();
    let a = sigmoid(raw[0]);
    let b = sigmoid(-raw[0]);
    let observed = bounds.chips(index);
    let mut lo = bounds.chips(grid.representative(0)?);
    let mut loss = (lo - observed).max(0) as f64 / pot_chips;
    let mut cumulative = 0.0;
    let mut cdf_gradient = Vec::with_capacity(grid.count);
    for (j, &probability) in q.iter().enumerate() {
        cumulative += b * probability;
        let hi = if j + 1 == grid.count {
            bounds.max_added_chips
        } else {
            bounds.chips(grid.representative(j + 1)?)
        };
        if hi <= lo {
            return Err(GeneratorError::InvalidWagerBounds);
        }
        let split = observed.clamp(lo, hi);
        let left = (split - lo) as f64 / pot_chips;
        let right = (hi - split) as f64 / pot_chips;
        loss += left * cumulative.powi(2) + right * (cumulative - 1.0).powi(2);
        cdf_gradient.push(2.0 * (left * cumulative + right * (cumulative - 1.0)));
        lo = hi;
    }
    // dL/d(mass_j) is a suffix sum of interval CDF derivatives.
    let mut suffix = 0.0;
    for value in cdf_gradient.iter_mut().rev() {
        suffix += *value;
        *value = suffix;
    }
    let mean: f64 = q.iter().zip(&cdf_gradient).map(|(p, d)| p * d).sum();
    gradient[0] = -a * b * mean;
    for (d, &p) in cdf_gradient.iter_mut().zip(&q) {
        *d = b * p * (*d - mean);
    }
    shape.score_gradient(&cdf_gradient, raw, &mut gradient)?;
    if !loss.is_finite() || gradient.iter().any(|x| !x.is_finite()) {
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
        for (cached, &weight) in layer.weights_f64.iter_mut().zip(&layer.weights) {
            *cached = f64::from(weight);
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

    /// One raw game decision: action/timing NLL + sizing CRPS/pot, then Adam.
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
        backprop_encoded(
            model,
            profile,
            observation,
            *legal,
            SizingObjective::CrpsOverPot(f64::from(observation.context_values[0])),
            None,
            None,
        )
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
        backprop_encoded(
            self,
            profile,
            observation,
            *legal,
            SizingObjective::Nll,
            None,
            None,
        )
    }

    /// Forward-only joint NLL for Monte Carlo validation. Never computes dL/dZ
    /// or model gradients; use evaluate_observation_encoded for Z1 adaptation.
    pub fn score_observation_encoded(
        &self,
        profile: &BehaviourProfile,
        observation: &GeneratorObservation,
        legal: &LegalActionSpace,
    ) -> Result<ScoringLoss, GeneratorError> {
        self.score_observation_with_probabilities_encoded(profile, observation, legal)
            .map(|(loss, _)| loss)
    }

    /// The same forward pass also exposes its pre-action distribution for accuracy.
    pub fn score_observation_with_probabilities_encoded(
        &self,
        profile: &BehaviourProfile,
        observation: &GeneratorObservation,
        legal: &LegalActionSpace,
    ) -> Result<(ScoringLoss, [f64; ACTION_COUNT]), GeneratorError> {
        let (stack, to_call) = observation.validated_facts(*legal)?;
        let encoded = encoded_context_features(
            observation.street,
            observation.actor_position,
            &observation.context_values,
            &observation.context_present,
        );
        let pass = self.forward_features(profile, encoded, *legal)?;
        let action = &observation.observed;
        let probability = pass.probabilities[usize::from(action_id(action.kind))];
        if !probability.is_finite() || probability <= 0.0 {
            return Err(GeneratorError::InvalidProbabilityMass);
        }
        let action_loss = -probability.ln();
        let amount = match action.kind {
            ActionKind::Fold | ActionKind::Check => 0.0,
            ActionKind::Call => to_call.min(stack),
            ActionKind::Bet | ActionKind::Raise => {
                stack
                    * action
                        .sizing
                        .ok_or(GeneratorError::InvalidObservedWager)?
                        .amount_over_stack_before
            }
        };
        let sizing_loss = if let Some(bounds) = legal.bounds(action.kind) {
            let index = bounds.index(amount)?;
            let raw = self
                .parameters
                .sizing
                .forward(&sizing_input(&pass.state, action.kind));
            sizing_objective(bounds, index, &raw, None)?
        } else {
            0.0
        };
        let timing_loss = if let Some(ms) = action.decision_time_ms {
            let fraction = f64::from(amount / stack);
            let raw =
                self.parameters
                    .timing
                    .forward(&timing_input(&pass.state, action.kind, fraction));
            let sigma = softplus(raw[1]) + MIN_TIMING_SCALE;
            let residual = raw[0] - f64::from(ms).ln_1p();
            0.5 * (residual / sigma).powi(2) + sigma.ln() + 0.5 * (2.0 * PI).ln()
        } else {
            0.0
        };
        let total = action_loss + sizing_loss + timing_loss;
        if !total.is_finite() {
            return Err(GeneratorError::NonFiniteGradient);
        }
        Ok((
            ScoringLoss {
                total,
                action: action_loss,
                sizing: sizing_loss,
                timing: timing_loss,
            },
            pass.probabilities,
        ))
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
    InvalidBatchWeight,
    EmptyBatch,
}
impl fmt::Display for GeneratorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for GeneratorError {}

// Validation-only exact sizing diagnostics; never used by training.
#[cfg(test)]
impl NeuralBehaviourGenerator {
    pub(crate) fn sizing_diagnostic_row(
        &self,
        profile: &BehaviourProfile,
        observation: &GeneratorObservation,
        legal: &LegalActionSpace,
    ) -> Result<Value, GeneratorError> {
        let (stack, _) = observation.validated_facts(*legal)?;
        let kind = observation.observed.kind;
        let bounds = legal
            .bounds(kind)
            .ok_or(GeneratorError::IllegalObservedAction)?;
        let sizing = observation
            .observed
            .sizing
            .ok_or(GeneratorError::InvalidObservedWager)?;
        let actual = bounds.index(stack * sizing.amount_over_stack_before)?;
        let count = bounds.ticks()?;
        let pot = observation.context_values[0];
        let pot_chips = f64::from(pot) * bounds.big_blind_chips as f64;
        let grid = BbSizingGrid::new(bounds)?;
        let encoded = encoded_context_features(
            observation.street,
            observation.actor_position,
            &observation.context_values,
            &observation.context_present,
        );
        let pass = self.forward_features(profile, encoded, *legal)?;
        let raw = self
            .parameters
            .sizing
            .forward(&sizing_input(&pass.state, kind));
        let loss = sizing_objective(bounds, actual, &raw, None)?;
        let atom = if count == 0 { 1.0 } else { sigmoid(raw[0]) };
        let log_atom = if count == 0 {
            0.0
        } else {
            -sizing_softplus(-raw[0])
        };
        let log_nonatom = -sizing_softplus(raw[0]);
        let shape = if grid.count > 1 {
            Some(SizingShape::new(grid, &raw)?)
        } else {
            None
        };
        let p_min_bb_bin_given_nonallin = if grid.count == 0 {
            None
        } else if let Some(s) = &shape {
            Some(s.log_probability(0)?.exp())
        } else {
            Some(1.0)
        };
        let mut best = count;
        let mut best_log = log_atom;
        let mut nonallin_best: Option<(usize, f64)> = None;
        let mut cumulative = 0.0;
        let mut crps = if grid.count > 0 {
            // The actual amount can be below the first executable bin centre.
            (bounds.chips(grid.representative(0)?) - bounds.chips(actual)).max(0) as f64 / pot_chips
        } else {
            0.0
        };
        let mut shape_nll = None;
        let mut median = count;
        let mut median_found = false;
        let mut close10_mass = 0.0;
        let mut close25_mass = 0.0;
        let mut add_tolerance = |index: usize, mass: f64| {
            let distance =
                (bounds.chips(index) - bounds.chips(actual)).unsigned_abs() as f64 / pot_chips;
            if distance <= 0.1 {
                close10_mass += mass;
            }
            if distance <= 0.25 {
                close25_mass += mass;
            }
        };
        for position in 0..grid.count {
            let i = grid.representative(position)?;
            let log_shape = match &shape {
                Some(s) => s.log_probability(position)?,
                None => 0.0,
            };
            let log_mass = log_nonatom + log_shape;
            if log_mass > best_log || (log_mass == best_log && i < best) {
                best = i;
                best_log = log_mass;
            }
            if nonallin_best.is_none_or(|(_, old)| log_shape > old) {
                nonallin_best = Some((i, log_shape));
            }
            if actual < count && grid.position(actual)? == position {
                shape_nll = Some(-log_shape);
            }
            let mass = log_mass.exp();
            add_tolerance(i, mass);
            cumulative += mass;
            if !median_found && cumulative >= 0.5 {
                median = i;
                median_found = true;
            }
            let next = if position + 1 == grid.count {
                count
            } else {
                grid.representative(position + 1)?
            };
            let lo = bounds.chips(i);
            let hi = bounds.chips(next);
            let split = bounds.chips(actual).clamp(lo, hi);
            crps += (split - lo) as f64 / pot_chips * cumulative.powi(2)
                + (hi - split) as f64 / pot_chips * (cumulative - 1.0).powi(2);
        }
        add_tolerance(count, atom);
        if (cumulative + atom - 1.0).abs() > 1e-8 || !crps.is_finite() {
            return Err(GeneratorError::InvalidProbabilityMass);
        }
        Ok(json!({
            "street":format!("{:?}", observation.street), "action":format!("{:?}", kind),
            "legal_choices":count+1, "nonallin_choices":count,
            "actual_chips":bounds.chips(actual), "map_chips":bounds.chips(best),
            "median_chips":bounds.chips(median), "chip_unit":bounds.chip_unit,
            "big_blind_chips":bounds.big_blind_chips, "min_chips":bounds.min_added_chips,
            "max_chips":bounds.max_added_chips, "pot_bb":pot,
            "observed_bb":bounds.chips(actual) as f64 / bounds.big_blind_chips as f64,
            "map_bb":bounds.chips(best) as f64 / bounds.big_blind_chips as f64,
            "observed_allin":actual==count, "map_allin":best==count,
            "sizing_measure":"bb_nearest_0.1_shared_relu_plus_allin_atom",
            "bb_ordinary_bins":grid.count,
            "actual_bb_bin":(actual < count).then(|| BbSizingGrid::bin(bounds, actual).to_string()),
            "map_bb_bin":(best < count).then(|| BbSizingGrid::bin(bounds, best).to_string()),
            "p_min_bb_bin_given_nonallin":p_min_bb_bin_given_nonallin,
            "p_allin":atom, "nll":loss, "uniform_nll":((grid.count+1) as f64).ln(),
            "p_actual":(-loss).exp(), "p_map":best_log.exp(),
            "shape_nll":shape_nll, "nonallin_map_chips":nonallin_best.map(|(i,_)| bounds.chips(i)),
            "crps_over_pot":crps, "mass_within_10pct_pot":close10_mass,
            "mass_within_25pct_pot":close25_mass,
        }))
    }
}

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
    fn forward_only_scoring_matches_nll_adaptation_components() {
        let context = ctx();
        let (context_values, context_present) = encode_context(&context);
        let observation = GeneratorObservation {
            street: context.street,
            actor_position: context.actor_position,
            context_values,
            context_present,
            observed: observed_bet(5.0),
        };
        let g = model();
        let score = g
            .score_observation_encoded(&profile(), &observation, &legal())
            .unwrap();
        let with_gradients = g
            .evaluate_observation_encoded(&profile(), &observation, &legal())
            .unwrap();
        for (a, b) in [
            (score.total, with_gradients.total),
            (score.action, with_gradients.action),
            (score.sizing, with_gradients.sizing),
            (score.timing, with_gradients.timing),
        ] {
            assert!((a - b).abs() < 1e-12, "{a} vs {b}");
        }
    }

    #[test]
    fn openblas_dense_matches_scalar_reference() {
        let layer =
            DenseLayer::new(3, 2, vec![0.1, -0.2, 0.3, 0.4, -0.5, 0.6], vec![0.25, -0.5]).unwrap();
        let input = [0.7, 1.2, -0.9];
        let delta = [0.3, -0.8];
        let expected = layer
            .weights()
            .as_chunks::<3>()
            .0
            .iter()
            .zip(layer.biases())
            .map(|(row, &b)| {
                row.iter()
                    .zip(input)
                    .fold(f64::from(b), |sum, (&w, x)| sum + f64::from(w) * x)
            })
            .collect::<Vec<_>>();
        for (a, b) in layer.forward(&input).iter().zip(expected) {
            assert!((a - b).abs() < 1e-12);
        }
        let mut gradients = LayerGradients::new(&layer);
        let actual = layer.backward(&input, &delta, Some(&mut gradients));
        for (i, (&value, &x)) in actual.iter().zip(&input).enumerate() {
            let expected = delta
                .iter()
                .enumerate()
                .map(|(j, &d)| d * f64::from(layer.weights()[j * 3 + i]))
                .sum::<f64>();
            assert!((value - expected).abs() < 1e-12);
            for (j, &d) in delta.iter().enumerate() {
                assert!((gradients.weights[j * 3 + i] - d * x).abs() < 1e-12);
            }
        }
        let mut products = LayerProducts::default();
        products.record(&input, &delta).unwrap();
        products.record(&[1.0, 2.0, 3.0], &[0.5, 0.25]).unwrap();
        let mut sum = LayerGradients::new(&layer);
        products.add_to(&layer, &mut sum).unwrap();
        for (j, (&d, d2)) in delta.iter().zip([0.5, 0.25]).enumerate() {
            for (i, (&x, x2)) in input.iter().zip([1.0, 2.0, 3.0]).enumerate() {
                let expected = d * x + d2 * x2;
                assert!((sum.weights[j * 3 + i] - expected).abs() < 1e-12);
            }
        }
        assert!((sum.bias[0] - 0.8).abs() < 1e-12);
        assert!((sum.bias[1] + 0.55).abs() < 1e-12);
    }

    #[test]
    fn latent_only_backprop_matches_full_gradient_exactly() {
        let context = ctx();
        let (context_values, context_present) = encode_context(&context);
        let g = model();
        for amount in [0.0, 5.0, 80.0] {
            let observed = if amount == 0.0 {
                ObservedAction::from_exact(
                    &context,
                    ExactAction {
                        kind: ActionKind::Check,
                        is_all_in: false,
                        amount_added_bb: 0.0,
                        highest_bet_after_bb: 0.0,
                        decision_time_ms: Some(600),
                    },
                )
                .unwrap()
            } else {
                observed_bet(amount)
            };
            for timing in [None, Some(600), Some(30_000)] {
                let mut observed = observed.clone();
                observed.decision_time_ms = timing;
                let observation = GeneratorObservation {
                    street: context.street,
                    actor_position: context.actor_position,
                    context_values,
                    context_present,
                    observed,
                };
                let mut gradients = Gradients::new(&g.parameters);
                let full = backprop_encoded(
                    &g,
                    &profile(),
                    &observation,
                    legal(),
                    SizingObjective::Nll,
                    Some(&mut gradients),
                    None,
                )
                .unwrap();
                let latent = g
                    .evaluate_observation_encoded(&profile(), &observation, &legal())
                    .unwrap();
                assert_eq!(full, latent);
            }
        }
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
    fn owned_snapshot_accumulation_matches_checked_accumulation() {
        let context = ctx();
        let (context_values, context_present) = encode_context(&context);
        let mut g = model();
        let mut expected_g = g.clone();
        let mut opt = Adam::new(&g, 0.001).unwrap();
        let mut expected_opt = opt.clone();
        let mut batch = GeneratorGradientBatch::new(&g);
        let mut expected = GeneratorGradientBatch::new(&expected_g);
        for amount in [5.0, 7.0, 80.0] {
            let observation = GeneratorObservation {
                street: context.street,
                actor_position: context.actor_position,
                context_values,
                context_present,
                observed: observed_bet(amount),
            };
            assert_eq!(
                batch
                    .accumulate_snapshot_encoded(&profile(), &observation, &legal())
                    .unwrap(),
                expected
                    .accumulate_encoded(&expected_g, &profile(), &observation, &legal())
                    .unwrap()
            );
        }
        let a = batch.averaged_gradients().unwrap();
        let b = expected.averaged_gradients().unwrap();
        for (a, b) in [
            (&a.state, &b.state),
            (&a.action, &b.action),
            (&a.sizing, &b.sizing),
            (&a.timing, &b.timing),
        ] {
            for (&a, &b) in a
                .weights
                .iter()
                .chain(&a.bias)
                .zip(b.weights.iter().chain(&b.bias))
            {
                assert!((a - b).abs() <= 1e-12 * (1.0 + a.abs().max(b.abs())));
            }
        }
        assert_eq!(
            opt.apply_batch(&mut g, batch).unwrap(),
            expected_opt.apply_batch(&mut expected_g, expected).unwrap()
        );
        for (a, b) in [
            (&g.parameters.state, &expected_g.parameters.state),
            (&g.parameters.action, &expected_g.parameters.action),
            (&g.parameters.sizing, &expected_g.parameters.sizing),
            (&g.parameters.timing, &expected_g.parameters.timing),
        ] {
            for (&a, &b) in a
                .weights()
                .iter()
                .chain(a.biases())
                .zip(b.weights().iter().chain(b.biases()))
            {
                assert!((a - b).abs() <= 1e-6 * (1.0 + a.abs().max(b.abs())));
            }
            assert!(
                a.weights
                    .iter()
                    .zip(&a.weights_f64)
                    .all(|(&w, &cached)| f64::from(w) == cached)
            );
        }
        assert_eq!(opt.steps(), expected_opt.steps());
    }

    #[test]
    fn weighted_batch_matches_repeated_observations() {
        let generator = model();
        let mut weighted = GeneratorGradientBatch::new(&generator);
        let mut repeated = GeneratorGradientBatch::new(&generator);
        for (amount, weight) in [(5.0, 2), (7.0, 1)] {
            let (loss, gradients) = loss_gradients(
                &generator,
                &profile(),
                &ctx(),
                legal(),
                &observed_bet(amount),
            )
            .unwrap();
            weighted
                .add_loss_gradients(loss.clone(), gradients.clone(), f64::from(weight))
                .unwrap();
            for _ in 0..weight {
                repeated
                    .add_loss_gradients(loss.clone(), gradients.clone(), 1.0)
                    .unwrap();
            }
        }
        assert_eq!(weighted.len(), 2);
        let a = weighted.mean_loss().unwrap();
        let b = repeated.mean_loss().unwrap();
        assert!((a.total - b.total).abs() < 1e-12);
        let a = weighted.averaged_gradients().unwrap();
        let b = repeated.averaged_gradients().unwrap();
        for (a, b) in [
            (&a.state, &b.state),
            (&a.action, &b.action),
            (&a.sizing, &b.sizing),
            (&a.timing, &b.timing),
        ] {
            assert!(
                a.weights
                    .iter()
                    .chain(&a.bias)
                    .zip(b.weights.iter().chain(&b.bias))
                    .all(|(x, y)| (x - y).abs() < 1e-12)
            );
        }
        let (loss, gradients) =
            loss_gradients(&generator, &profile(), &ctx(), legal(), &observed_bet(5.0)).unwrap();
        let before = weighted.mean_loss().unwrap();
        assert!(matches!(
            weighted.add_loss_gradients(loss, gradients, 0.0),
            Err(GeneratorError::InvalidBatchWeight)
        ));
        assert_eq!(weighted.mean_loss().unwrap(), before);
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

    fn sizing_test_raw() -> Vec<f64> {
        // Nonzero arbitrary coefficients for gradient tests, not initialisation.
        vec![-2.0, -0.3, 0.7, -0.9, 0.8, -0.55, 0.6, 0.4, -0.6, 0.9]
    }

    #[test]
    fn finite_differences_verify_sizing_gradients() {
        let raw = sizing_test_raw();
        for bounds in [
            WagerBounds {
                min_added_chips: 11,
                max_added_chips: 187,
                chip_unit: 1,
                big_blind_chips: 100,
            },
            WagerBounds {
                min_added_chips: 10,
                max_added_chips: 157,
                chip_unit: 5,
                big_blind_chips: 10,
            },
        ] {
            let count = bounds.ticks().unwrap();
            for i in [0, 17, count - 1, count] {
                let (_, grad) = sizing_nll(bounds, i, &raw).unwrap();
                for j in 0..raw.len() {
                    let mut plus = raw.clone();
                    let mut minus = raw.clone();
                    plus[j] += 1e-5;
                    minus[j] -= 1e-5;
                    let numeric = (sizing_nll(bounds, i, &plus).unwrap().0
                        - sizing_nll(bounds, i, &minus).unwrap().0)
                        / 2e-5;
                    assert!(
                        (numeric - grad[j]).abs() < 1e-5,
                        "{i}:{j}: {numeric} vs {}",
                        grad[j]
                    );
                }
            }
        }
    }

    #[test]
    fn shared_scorer_matches_direct_network() {
        let raw = sizing_test_raw();
        let bounds = WagerBounds {
            min_added_chips: 10,
            max_added_chips: 60_000,
            chip_unit: 1,
            big_blind_chips: 100,
        };
        let grid = BbSizingGrid::new(bounds).unwrap();
        let shape = SizingShape::new(grid, &raw).unwrap();
        let width = sizing_width(&raw).unwrap();
        let scores: Vec<f64> = shape
            .coordinates
            .iter()
            .map(|&x| {
                (0..width)
                    .map(|i| {
                        raw[1 + 2 * width + i] * (raw[1 + i] + raw[1 + width + i] * x).max(0.0)
                    })
                    .sum()
            })
            .collect();
        let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let log_sum = scores.iter().map(|x| (x - max).exp()).sum::<f64>().ln();
        for (i, score) in scores.iter().enumerate() {
            assert!((shape.log_probability(i).unwrap() - (score - max - log_sum)).abs() < 1e-10);
        }
    }

    #[test]
    fn shared_scorer_is_normalized_and_sampler_matches_bb_bins() {
        let raw = sizing_test_raw();
        for bounds in [
            legal().bet.unwrap(),
            WagerBounds {
                min_added_chips: 11,
                max_added_chips: 187,
                chip_unit: 1,
                big_blind_chips: 100,
            },
            WagerBounds {
                min_added_chips: 10,
                max_added_chips: 157,
                chip_unit: 5,
                big_blind_chips: 10,
            },
            WagerBounds {
                min_added_chips: 10,
                max_added_chips: 12,
                chip_unit: 5,
                big_blind_chips: 10,
            },
            WagerBounds {
                min_added_chips: 10,
                max_added_chips: 15,
                chip_unit: 1,
                big_blind_chips: 100,
            },
            WagerBounds {
                min_added_chips: 10,
                max_added_chips: 10,
                chip_unit: 5,
                big_blind_chips: 10,
            },
        ] {
            let grid = BbSizingGrid::new(bounds).unwrap();
            let count = bounds.ticks().unwrap();
            let mut mass = (-sizing_nll(bounds, count, &raw).unwrap().0).exp();
            let mut conditional_cdf = 0.0;
            for position in 0..grid.count {
                let i = grid.representative(position).unwrap();
                let p = (-sizing_nll(bounds, i, &raw).unwrap().0).exp();
                mass += p;
                let q = p / sigmoid(-raw[0]);
                let u = conditional_cdf + q * 0.5;
                assert_eq!(
                    bounds
                        .index(sample_wager(bounds, &raw, u, 0.999).unwrap())
                        .unwrap(),
                    i
                );
                conditional_cdf += q;
            }
            for i in 0..count {
                let position = grid.position(i).unwrap();
                let representative = grid.representative(position).unwrap();
                assert_eq!(
                    BbSizingGrid::bin(bounds, representative),
                    BbSizingGrid::bin(bounds, i)
                );
                let centre = grid.bin_at(position) * i128::from(bounds.big_blind_chips);
                let distance = |tick| (10 * i128::from(bounds.chips(tick)) - centre).abs();
                assert!(distance(representative) <= distance(i));
                assert_eq!(
                    sizing_nll(bounds, i, &raw).unwrap(),
                    sizing_nll(bounds, representative, &raw).unwrap()
                );
            }
            assert!((mass - 1.0).abs() < 1e-10, "{mass}");
            assert_eq!(
                bounds
                    .index(sample_wager(bounds, &raw, 0.5, 0.0).unwrap())
                    .unwrap(),
                count
            );
        }
    }

    #[test]
    fn sizing_is_invariant_to_currency_and_finer_chip_resolution() {
        let raw = sizing_test_raw();
        let a = WagerBounds {
            min_added_chips: 10,
            max_added_chips: 200,
            chip_unit: 1,
            big_blind_chips: 10,
        };
        let b = WagerBounds {
            min_added_chips: 100,
            max_added_chips: 2000,
            chip_unit: 10,
            big_blind_chips: 100,
        };
        let c = WagerBounds {
            max_added_chips: 1995,
            chip_unit: 1,
            ..b
        };
        let ga = BbSizingGrid::new(a).unwrap();
        let gb = BbSizingGrid::new(b).unwrap();
        // c has the same occupied ordinary bins, even though its final chip is off-grid.
        let gc = BbSizingGrid::new(c).unwrap();
        assert_eq!(ga.count, gb.count);
        assert_eq!(ga.count, gc.count);
        for position in 0..ga.count {
            let ia = ga.representative(position).unwrap();
            let ib = gb.representative(position).unwrap();
            let ic = gc.representative(position).unwrap();
            let x = sizing_nll(a, ia, &raw).unwrap();
            assert_eq!(x, sizing_nll(b, ib, &raw).unwrap());
            assert_eq!(x, sizing_nll(c, ic, &raw).unwrap());
        }
        assert_eq!(
            sizing_nll(a, ga.ticks, &raw).unwrap(),
            sizing_nll(b, gb.ticks, &raw).unwrap()
        );
    }

    #[test]
    fn shared_scorer_can_form_two_separate_peaks() {
        let bounds = WagerBounds {
            min_added_chips: 1,
            max_added_chips: 1200,
            chip_unit: 1,
            big_blind_chips: 100,
        };
        // Two triangular bumps in log1p(BB), represented by three hinges each.
        // These are test parameters only, never used for initialization.
        let width = 6;
        let mut raw = vec![0.0; sizing_outputs(width)];
        raw[0] = -2.0;
        for (j, centre) in [5.0_f64.ln_1p(), 10.0_f64.ln_1p()].into_iter().enumerate() {
            for (k, (offset, v)) in [(-0.02, 1000.0), (0.0, -2000.0), (0.02, 1000.0)]
                .into_iter()
                .enumerate()
            {
                let i = 3 * j + k;
                raw[1 + i] = -(centre + offset);
                raw[1 + width + i] = 1.0;
                raw[1 + 2 * width + i] = v;
            }
        }
        let nll = |chips: usize| sizing_nll(bounds, chips - 1, &raw).unwrap().0;
        assert_eq!(nll(999), nll(998));
        assert!(nll(998) < nll(12));
        assert!(nll(500) + 10.0 < nll(750));
        assert!(nll(999) + 10.0 < nll(750));
    }

    #[test]
    fn crps_matches_energy_identity_and_finite_differences() {
        let raw = [0.3, -0.4, 0.7, 0.2, 1.1, -0.6, 0.9];
        for bounds in [
            WagerBounds {
                min_added_chips: 13,
                max_added_chips: 117,
                chip_unit: 1,
                big_blind_chips: 100,
            },
            WagerBounds {
                min_added_chips: 2,
                max_added_chips: 19,
                chip_unit: 3,
                big_blind_chips: 10,
            },
            WagerBounds {
                min_added_chips: 10,
                max_added_chips: 11,
                chip_unit: 1,
                big_blind_chips: 100,
            },
            WagerBounds {
                min_added_chips: 10,
                max_added_chips: 10,
                chip_unit: 1,
                big_blind_chips: 100,
            },
        ] {
            let grid = BbSizingGrid::new(bounds).unwrap();
            let mut masses = Vec::new();
            if grid.count > 0 {
                let shape = SizingShape::new(grid, &raw).unwrap();
                for (j, &p) in shape.log_probs.iter().enumerate() {
                    masses.push((
                        bounds.chips(grid.representative(j).unwrap()) as f64,
                        sigmoid(-raw[0]) * p.exp(),
                    ));
                }
            }
            masses.push((
                bounds.max_added_chips as f64,
                if grid.count == 0 {
                    1.0
                } else {
                    sigmoid(raw[0])
                },
            ));
            let pot = 2.3;
            for index in 0..=grid.ticks {
                let (loss, grad) = sizing_crps(bounds, index, &raw, pot).unwrap();
                let y = bounds.chips(index) as f64;
                let first: f64 = masses.iter().map(|(x, p)| p * (x - y).abs()).sum();
                let mut second = 0.0;
                for &(x, p) in &masses {
                    for &(z, q) in &masses {
                        second += p * q * (x - z).abs();
                    }
                }
                let expected = (first - 0.5 * second) / (pot * bounds.big_blind_chips as f64);
                assert!((loss - expected).abs() < 1e-12);
                for j in 0..raw.len() {
                    let mut plus = raw;
                    let mut minus = raw;
                    plus[j] += 1e-5;
                    minus[j] -= 1e-5;
                    let numeric = (sizing_crps(bounds, index, &plus, pot).unwrap().0
                        - sizing_crps(bounds, index, &minus, pot).unwrap().0)
                        / 2e-5;
                    assert!((numeric - grad[j]).abs() < 1e-7, "{numeric} vs {}", grad[j]);
                }
                let (half, half_grad) = sizing_crps(bounds, index, &raw, 2.0 * pot).unwrap();
                assert!((half * 2.0 - loss).abs() < 1e-12);
                for (a, b) in half_grad.iter().zip(&grad) {
                    assert!((a * 2.0 - b).abs() < 1e-12);
                }
                let scaled = WagerBounds {
                    min_added_chips: bounds.min_added_chips * 10,
                    max_added_chips: bounds.max_added_chips * 10,
                    chip_unit: bounds.chip_unit * 10,
                    big_blind_chips: bounds.big_blind_chips * 10,
                };
                let (other, other_grad) = sizing_crps(scaled, index, &raw, pot).unwrap();
                assert!((other - loss).abs() < 1e-12);
                for (a, b) in other_grad.iter().zip(&grad) {
                    assert!((a - b).abs() < 1e-12);
                }
            }
            assert!(sizing_crps(bounds, 0, &raw, 0.0).is_err());
        }
    }

    #[test]
    fn training_crps_matches_diagnostic_and_preserves_nll_adaptation() {
        let context = ctx();
        let (context_values, context_present) = encode_context(&context);
        let g = model();
        for amount in [5.0, 20.0, 80.0] {
            let obs = GeneratorObservation {
                street: context.street,
                actor_position: context.actor_position,
                context_values,
                context_present,
                observed: observed_bet(amount),
            };
            let training = loss_gradients_encoded(&g, &profile(), &obs, legal())
                .unwrap()
                .0;
            let diag = g.sizing_diagnostic_row(&profile(), &obs, &legal()).unwrap();
            let nll = g
                .evaluate_observation_encoded(&profile(), &obs, &legal())
                .unwrap();
            assert!((training.sizing - diag["crps_over_pot"].as_f64().unwrap()).abs() < 1e-12);
            assert!((nll.sizing - diag["nll"].as_f64().unwrap()).abs() < 1e-12);
            assert_eq!(training.action, nll.action);
            assert_eq!(training.timing, nll.timing);
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
    fn parquet_features_and_live_context_have_identical_objective_and_latent_gradient() {
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
