//! Differentiable, conditional behavioural generator G(C, Z).
//!
//! The NLHE rules engine provides a *complete* action mask and exact wager
//! bounds. The model never guesses legality or enumerates arbitrary bet sizes.
//! Trainable reverse-mode gradients and Adam are implemented in Rust directly.

use std::f64::consts::PI;
use std::fmt;

use super::codebook::{action_id, position_id, street_id};
use super::featurizer::encode_context;
use super::{
    ActionKind, BehaviourEventError, BehaviourLatentConfig, BehaviourLatentModel, BehaviourProfile,
    DecisionContext, ExactAction, ObservedAction, PRE_ACTION_CONTEXT_WIDTH,
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
        if ![
            self.fold,
            self.check,
            self.call,
            self.bet.is_some(),
            self.raise.is_some(),
        ]
        .contains(&true)
        {
            return Err(GeneratorError::NoLegalActions);
        }
        if (self.bet.is_some() || self.raise.is_some()) && context.pot_bb <= 0.0 {
            return Err(GeneratorError::InvalidLegalActions);
        }
        if self.check && context.to_call_bb != 0.0
            || self.call && context.to_call_bb <= 0.0
            || self.bet.is_some() && context.current_bet_bb != 0.0
            || self.raise.is_some() && context.current_bet_bb <= 0.0
        {
            return Err(GeneratorError::InvalidLegalActions);
        }
        let stack = context
            .actor()
            .map_err(GeneratorError::InvalidContext)?
            .stack_remaining_bb;
        if stack <= 0.0 {
            return Err(GeneratorError::InvalidLegalActions);
        }
        for (kind, bounds) in [(ActionKind::Bet, self.bet), (ActionKind::Raise, self.raise)] {
            if let Some(bounds) = bounds {
                bounds.ticks()?;
                if (bounds.amount(bounds.ticks()?) - stack).abs() > 1e-4 * stack.max(1.0)
                    || (kind == ActionKind::Raise
                        && context
                            .actor()
                            .map_err(GeneratorError::InvalidContext)?
                            .committed_this_street_bb
                            + bounds.amount(0)
                            <= context.current_bet_bb)
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
        if parameters.state.input != GENERATOR_CONTEXT_WIDTH + latent.dimension()
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
        if profile.dimension() != self.latent.dimension() {
            return Err(GeneratorError::WrongLatentDimension {
                expected: self.latent.dimension(),
                actual: profile.dimension(),
            });
        }
        legal.validate(context)?;
        let mut input = context_features(context);
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
    let mut values = vec![0.0; GENERATOR_CONTEXT_WIDTH];
    values[usize::from(street_id(ctx.street))] = 1.0;
    values[STREET_COUNT + usize::from(position_id(ctx.actor_position))] = 1.0;
    let start = STREET_COUNT + POSITION_COUNT;
    let (numeric, mask) = encode_context(ctx);
    for i in 0..PRE_ACTION_CONTEXT_WIDTH {
        // All numeric public fields are nonnegative. Log compression prevents
        // large stacks and historic timing from saturating the tanh MLP.
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
    let forward = model.forward(profile, context, legal)?;
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

    let actor = context.actor().map_err(GeneratorError::InvalidContext)?;
    let amount = match action.kind {
        ActionKind::Fold | ActionKind::Check => 0.0,
        ActionKind::Call => context.to_call_bb.min(actor.stack_remaining_bb),
        ActionKind::Bet | ActionKind::Raise => {
            actor.stack_remaining_bb
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
        let fraction = if actor.stack_remaining_bb > 0.0 {
            f64::from(amount / actor.stack_remaining_bb)
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

    /// One observed decision: NLL, reverse-mode backprop, then one Adam update.
    /// Latent Z is *not* silently optimized: its gradient is returned explicitly.
    pub fn train_step(
        &mut self,
        model: &mut NeuralBehaviourGenerator,
        profile: &BehaviourProfile,
        context: &DecisionContext,
        legal: &LegalActionSpace,
        observed: &ObservedAction,
    ) -> Result<TrainingLoss, GeneratorError> {
        if !self.moments.matches(&model.parameters) {
            return Err(GeneratorError::OptimizerModelMismatch);
        }
        let (loss, gradients) = loss_gradients(model, profile, context, *legal, observed)?;
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

    pub fn steps(&self) -> u64 {
        self.step
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
    InvalidWagerBounds,
    InvalidObservedWager,
    IllegalObservedAction,
    InvalidRandomDraw,
    InvalidProbabilityMass,
    InvalidGeneratedTiming,
    InvalidLearningRate,
    OptimizerStepOverflow,
    OptimizerModelMismatch,
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
