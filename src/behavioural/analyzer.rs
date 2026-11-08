//! Amortized posterior q(Z | subject support hands), with exact reverse-mode training.
//! No cards, player identifiers, query responses, or hidden opponent information enter A.
use std::f64::consts::PI;
use std::fmt;

use serde_json::{Value, json};

use super::checkpoint::{
    dense_from_value, dense_value, dimension, exact_keys, float_array, number, unsigned,
};
use super::codebook::position_id;
use super::{
    ACTION_COUNT, BehaviourLatentConfig, BehaviourLatentModel, BehaviourProfile,
    CategoricalFeature, DenseLayer, EVENT_CONTINUOUS_WIDTH, FeaturizedHand,
};

pub const ANALYZER_EVENT_WIDTH: usize = 4 + 7 + ACTION_COUNT + 1 + 2 * EVENT_CONTINUOUS_WIDTH;
const LOG_STD_CENTER: f64 = -1.0;
const LOG_STD_RANGE: f64 = 3.0;

#[derive(Debug, Clone, PartialEq)]
pub struct BehaviourPosterior {
    mean: Vec<f64>,
    std: Vec<f64>,
}

impl BehaviourPosterior {
    pub fn mean(&self) -> &[f64] {
        &self.mean
    }
    pub fn std(&self) -> &[f64] {
        &self.std
    }

    pub fn sample(&self, epsilon: &[f64]) -> Result<BehaviourProfile, AnalyzerError> {
        if epsilon.len() != self.mean.len() || epsilon.iter().any(|x| !x.is_finite()) {
            return Err(AnalyzerError::InvalidNoise);
        }
        let z = self
            .mean
            .iter()
            .zip(&self.std)
            .zip(epsilon)
            .map(|((&m, &s), &e)| {
                let value = (m + s * e) as f32;
                if !value.is_finite() {
                    Err(AnalyzerError::NonFiniteOutput)
                } else {
                    Ok(value)
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        let config =
            BehaviourLatentConfig::new(self.mean.len()).map_err(|_| AnalyzerError::InvalidShape)?;
        config
            .profile(z)
            .map_err(|_| AnalyzerError::NonFiniteOutput)
    }

    pub fn mean_profile(&self) -> Result<BehaviourProfile, AnalyzerError> {
        self.sample(&vec![0.0; self.mean.len()])
    }

    pub fn kl_to_standard_normal(&self) -> f64 {
        self.mean
            .iter()
            .zip(&self.std)
            .map(|(&m, &s)| 0.5 * (m * m + s * s - 1.0 - 2.0 * s.ln()))
            .sum()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AnalyzerParameters {
    pub state: DenseLayer,
    pub posterior: DenseLayer,
}

#[derive(Debug, Clone)]
pub struct NeuralBehaviourAnalyzer {
    latent: BehaviourLatentConfig,
    parameters: AnalyzerParameters,
}

#[derive(Debug)]
struct AnalyzerPass {
    events: Vec<Vec<f64>>,
    states: Vec<Vec<f64>>,
    pooled: Vec<f64>,
    raw: Vec<f64>,
    posterior: BehaviourPosterior,
}

impl NeuralBehaviourAnalyzer {
    pub fn new(
        latent: BehaviourLatentConfig,
        parameters: AnalyzerParameters,
    ) -> Result<Self, AnalyzerError> {
        let output_width = latent
            .dimension()
            .checked_mul(2)
            .ok_or(AnalyzerError::InvalidShape)?;
        if latent.dimension() > 4096
            || parameters.state.input_width() != ANALYZER_EVENT_WIDTH
            || parameters.posterior.input_width() != parameters.state.output_width()
            || parameters.posterior.output_width() != output_width
        {
            return Err(AnalyzerError::InvalidShape);
        }
        Ok(Self { latent, parameters })
    }

    pub fn initialize(
        latent: BehaviourLatentConfig,
        hidden: usize,
        seed: u64,
    ) -> Result<Self, AnalyzerError> {
        if hidden == 0 || hidden > 4096 || latent.dimension() > 4096 || seed == 0 {
            return Err(AnalyzerError::InvalidInitialization);
        }
        let mut rng = seed;
        let mut new_layer = |input: usize, output: usize| -> Result<DenseLayer, AnalyzerError> {
            let radius = (6.0 / (input + output) as f64).sqrt();
            let weights = (0..input * output)
                .map(|_| {
                    rng ^= rng << 13;
                    rng ^= rng >> 7;
                    rng ^= rng << 17;
                    ((rng as f64 / u64::MAX as f64 * 2.0 - 1.0) * radius) as f32
                })
                .collect();
            DenseLayer::new(input, output, weights, vec![0.0; output])
                .map_err(|_| AnalyzerError::InvalidShape)
        };
        Self::new(
            latent,
            AnalyzerParameters {
                state: new_layer(ANALYZER_EVENT_WIDTH, hidden)?,
                posterior: new_layer(hidden, latent.dimension() * 2)?,
            },
        )
    }

    pub fn parameters(&self) -> &AnalyzerParameters {
        &self.parameters
    }

    pub fn infer(&self, support: &[FeaturizedHand]) -> Result<BehaviourPosterior, AnalyzerError> {
        Ok(self.pass(support)?.posterior)
    }

    fn pass(&self, support: &[FeaturizedHand]) -> Result<AnalyzerPass, AnalyzerError> {
        let events = encode_support(support)?;
        let states: Vec<_> = events
            .iter()
            .map(|event| {
                dense_forward(&self.parameters.state, event)
                    .into_iter()
                    .map(f64::tanh)
                    .collect::<Vec<_>>()
            })
            .collect();
        let mut pooled = vec![0.0; self.parameters.state.output_width()];
        for state in &states {
            for (p, &value) in pooled.iter_mut().zip(state) {
                *p += value / events.len() as f64;
            }
        }
        let raw = dense_forward(&self.parameters.posterior, &pooled);
        let mut mean = Vec::with_capacity(self.latent.dimension());
        let mut std = Vec::with_capacity(self.latent.dimension());
        for i in 0..self.latent.dimension() {
            mean.push(raw[i]);
            std.push(
                (LOG_STD_CENTER + LOG_STD_RANGE * raw[self.latent.dimension() + i].tanh()).exp(),
            );
        }
        if mean.iter().any(|x| !x.is_finite()) || std.iter().any(|x| !x.is_finite() || *x <= 0.0) {
            return Err(AnalyzerError::NonFiniteOutput);
        }
        Ok(AnalyzerPass {
            events,
            states,
            pooled,
            raw,
            posterior: BehaviourPosterior { mean, std },
        })
    }

    pub(crate) fn checkpoint_value(&self) -> Value {
        json!({"dimension":self.latent.dimension(), "state":dense_value(&self.parameters.state),
               "posterior":dense_value(&self.parameters.posterior)})
    }

    pub(crate) fn from_checkpoint(value: &Value) -> Result<Self, String> {
        exact_keys(value, &["dimension", "state", "posterior"])?;
        let latent = BehaviourLatentConfig::new(dimension(&value["dimension"])?)
            .map_err(|e| e.to_string())?;
        Self::new(
            latent,
            AnalyzerParameters {
                state: dense_from_value(&value["state"])?,
                posterior: dense_from_value(&value["posterior"])?,
            },
        )
        .map_err(|e| e.to_string())
    }
}

impl BehaviourLatentModel for NeuralBehaviourAnalyzer {
    fn latent_config(&self) -> BehaviourLatentConfig {
        self.latent
    }
}

fn encode_support(support: &[FeaturizedHand]) -> Result<Vec<Vec<f64>>, AnalyzerError> {
    let mut observations = Vec::new();
    for hand in support {
        let len = hand.sequence_len();
        if hand.categorical().len() != len
            || hand.continuous().len() != len
            || hand.continuous_present().len() != len
            || hand.sequence_indexes().len() != len
            || hand.decision_time_ms().len() != len
        {
            return Err(AnalyzerError::InvalidSupport);
        }
        for row in 0..len {
            let cat = hand.categorical()[row];
            let expected_subject = i64::from(position_id(hand.subject_position()))
                == cat[CategoricalFeature::ActorPosition as usize];
            match (
                cat[CategoricalFeature::IsSubject as usize],
                expected_subject,
            ) {
                (0, false) => continue,
                (1, true) => {}
                _ => return Err(AnalyzerError::InvalidSupport),
            }
            let street = usize::try_from(cat[CategoricalFeature::Street as usize])
                .map_err(|_| AnalyzerError::InvalidSupport)?;
            let position = usize::try_from(cat[CategoricalFeature::ActorPosition as usize])
                .map_err(|_| AnalyzerError::InvalidSupport)?;
            let action = usize::try_from(cat[CategoricalFeature::Action as usize])
                .map_err(|_| AnalyzerError::InvalidSupport)?;
            let allin = cat[CategoricalFeature::IsAllIn as usize];
            if street >= 4 || position >= 7 || action >= ACTION_COUNT || !(0..=1).contains(&allin) {
                return Err(AnalyzerError::InvalidSupport);
            }
            let mut features = vec![0.0; ANALYZER_EVENT_WIDTH];
            features[street] = 1.0;
            features[4 + position] = 1.0;
            features[11 + action] = 1.0;
            features[11 + ACTION_COUNT] = allin as f64;
            let start = 12 + ACTION_COUNT;
            for i in 0..EVENT_CONTINUOUS_WIDTH {
                let present = hand.continuous_present()[row][i];
                let value = hand.continuous()[row][i];
                if present > 1
                    || !value.is_finite()
                    || value < 0.0
                    || (present == 0 && value != 0.0)
                {
                    return Err(AnalyzerError::InvalidSupport);
                }
                features[start + i] = f64::from(value).ln_1p();
                features[start + EVENT_CONTINUOUS_WIDTH + i] = f64::from(present);
            }
            observations.push(features);
        }
    }
    if observations.is_empty() {
        return Err(AnalyzerError::NoSubjectDecisions);
    }
    Ok(observations)
}

fn dense_forward(layer: &DenseLayer, input: &[f64]) -> Vec<f64> {
    layer
        .weights()
        .chunks_exact(layer.input_width())
        .zip(layer.biases())
        .map(|(row, bias)| {
            row.iter()
                .zip(input)
                .fold(f64::from(*bias), |sum, (w, x)| sum + f64::from(*w) * x)
        })
        .collect()
}

#[derive(Debug, Clone)]
struct MatrixGradient {
    weights: Vec<f64>,
    bias: Vec<f64>,
}
impl MatrixGradient {
    fn zeros(layer: &DenseLayer) -> Self {
        Self {
            weights: vec![0.0; layer.weights().len()],
            bias: vec![0.0; layer.biases().len()],
        }
    }
}

fn dense_backward(
    layer: &DenseLayer,
    input: &[f64],
    output_gradient: &[f64],
    gradients: &mut MatrixGradient,
) -> Vec<f64> {
    let mut input_grad = vec![0.0; layer.input_width()];
    for (j, &g) in output_gradient.iter().enumerate() {
        gradients.bias[j] += g;
        for k in 0..layer.input_width() {
            gradients.weights[j * layer.input_width() + k] += g * input[k];
            input_grad[k] += g * f64::from(layer.weights()[j * layer.input_width() + k]);
        }
    }
    input_grad
}

#[derive(Debug, Clone)]
pub struct AnalyzerAdam {
    learning_rate: f64,
    steps: u64,
    state: MatrixMoments,
    posterior: MatrixMoments,
}

#[derive(Debug, Clone)]
struct MatrixMoments {
    mw: Vec<f64>,
    vw: Vec<f64>,
    mb: Vec<f64>,
    vb: Vec<f64>,
}
impl MatrixMoments {
    fn zero(layer: &DenseLayer) -> Self {
        Self {
            mw: vec![0.0; layer.weights().len()],
            vw: vec![0.0; layer.weights().len()],
            mb: vec![0.0; layer.biases().len()],
            vb: vec![0.0; layer.biases().len()],
        }
    }
    fn matches(&self, layer: &DenseLayer) -> bool {
        self.mw.len() == layer.weights().len()
            && self.vw.len() == layer.weights().len()
            && self.mb.len() == layer.biases().len()
            && self.vb.len() == layer.biases().len()
    }
    fn value(&self) -> Value {
        json!({"mw":self.mw, "vw":self.vw, "mb":self.mb, "vb":self.vb})
    }
    fn load(value: &Value, layer: &DenseLayer) -> Result<Self, String> {
        exact_keys(value, &["mw", "vw", "mb", "vb"])?;
        let this = Self {
            mw: float_array(&value["mw"])?,
            vw: float_array(&value["vw"])?,
            mb: float_array(&value["mb"])?,
            vb: float_array(&value["vb"])?,
        };
        if !this.matches(layer) || this.vw.iter().chain(&this.vb).any(|v| *v < 0.0) {
            return Err("invalid Analyzer Adam moments".into());
        }
        Ok(this)
    }
    fn update(
        &mut self,
        layer: &DenseLayer,
        gradients: &MatrixGradient,
        rate: f64,
        step: u64,
    ) -> Result<DenseLayer, AnalyzerError> {
        let c1 = 1.0 - 0.9_f64.powf(step as f64);
        let c2 = 1.0 - 0.999_f64.powf(step as f64);
        fn element(
            v: f32,
            g: f64,
            m: &mut f64,
            q: &mut f64,
            rate: f64,
            c1: f64,
            c2: f64,
        ) -> Result<f32, AnalyzerError> {
            if !g.is_finite() {
                return Err(AnalyzerError::NonFiniteGradient);
            }
            *m = 0.9 * *m + 0.1 * g;
            *q = 0.999 * *q + 0.001 * g * g;
            let new = f64::from(v) - rate * (*m / c1) / ((*q / c2).sqrt() + 1e-8);
            if !new.is_finite() || new.abs() > f64::from(f32::MAX) {
                return Err(AnalyzerError::NonFiniteGradient);
            }
            Ok(new as f32)
        }
        let weights = layer
            .weights()
            .iter()
            .zip(&gradients.weights)
            .zip(self.mw.iter_mut().zip(self.vw.iter_mut()))
            .map(|((&v, &g), (m, q))| element(v, g, m, q, rate, c1, c2))
            .collect::<Result<Vec<_>, _>>()?;
        let bias = layer
            .biases()
            .iter()
            .zip(&gradients.bias)
            .zip(self.mb.iter_mut().zip(self.vb.iter_mut()))
            .map(|((&v, &g), (m, q))| element(v, g, m, q, rate, c1, c2))
            .collect::<Result<Vec<_>, _>>()?;
        DenseLayer::new(layer.input_width(), layer.output_width(), weights, bias)
            .map_err(|_| AnalyzerError::NonFiniteGradient)
    }
}

impl AnalyzerAdam {
    pub fn new(model: &NeuralBehaviourAnalyzer, learning_rate: f64) -> Result<Self, AnalyzerError> {
        if !learning_rate.is_finite() || learning_rate <= 0.0 {
            return Err(AnalyzerError::InvalidLearningRate);
        }
        Ok(Self {
            learning_rate,
            steps: 0,
            state: MatrixMoments::zero(&model.parameters.state),
            posterior: MatrixMoments::zero(&model.parameters.posterior),
        })
    }
    pub fn steps(&self) -> u64 {
        self.steps
    }

    /// Update A using the gradient of query NLL w.r.t. the sampled z and the KL prior penalty.
    pub fn train_step(
        &mut self,
        model: &mut NeuralBehaviourAnalyzer,
        support: &[FeaturizedHand],
        epsilon: &[f64],
        latent_gradient: &[f64],
        beta: f64,
    ) -> Result<f64, AnalyzerError> {
        if !beta.is_finite()
            || beta < 0.0
            || latent_gradient.len() != model.latent.dimension()
            || epsilon.len() != model.latent.dimension()
            || latent_gradient
                .iter()
                .chain(epsilon)
                .any(|x| !x.is_finite())
        {
            return Err(AnalyzerError::InvalidNoise);
        }
        if !self.state.matches(&model.parameters.state)
            || !self.posterior.matches(&model.parameters.posterior)
        {
            return Err(AnalyzerError::OptimizerModelMismatch);
        }
        let pass = model.pass(support)?;
        let d = model.latent.dimension();
        let mut output_grad = vec![0.0; d * 2];
        for i in 0..d {
            let mean = pass.posterior.mean[i];
            let sigma = pass.posterior.std[i];
            output_grad[i] = latent_gradient[i] + beta * mean;
            let raw = pass.raw[d + i];
            let log_std_grad =
                latent_gradient[i] * sigma * epsilon[i] + beta * (sigma * sigma - 1.0);
            output_grad[d + i] = log_std_grad * LOG_STD_RANGE * (1.0 - raw.tanh().powi(2));
        }
        let mut posterior_gradient = MatrixGradient::zeros(&model.parameters.posterior);
        let pooled_gradient = dense_backward(
            &model.parameters.posterior,
            &pass.pooled,
            &output_grad,
            &mut posterior_gradient,
        );
        let mut state_gradient = MatrixGradient::zeros(&model.parameters.state);
        for (event, state) in pass.events.iter().zip(&pass.states) {
            let gradients = pooled_gradient
                .iter()
                .zip(state)
                .map(|(&g, &s)| g / pass.events.len() as f64 * (1.0 - s * s))
                .collect::<Vec<_>>();
            let _ = dense_backward(
                &model.parameters.state,
                event,
                &gradients,
                &mut state_gradient,
            );
        }
        let step = self
            .steps
            .checked_add(1)
            .ok_or(AnalyzerError::StepOverflow)?;
        let mut moments = self.clone();
        let state = moments.state.update(
            &model.parameters.state,
            &state_gradient,
            self.learning_rate,
            step,
        )?;
        let posterior = moments.posterior.update(
            &model.parameters.posterior,
            &posterior_gradient,
            self.learning_rate,
            step,
        )?;
        model.parameters = AnalyzerParameters { state, posterior };
        moments.steps = step;
        *self = moments;
        Ok(pass.posterior.kl_to_standard_normal())
    }

    pub(crate) fn checkpoint_value(&self) -> Value {
        json!({"learning_rate":self.learning_rate,"steps":self.steps,"state":self.state.value(),"posterior":self.posterior.value()})
    }
    pub(crate) fn from_checkpoint(
        value: &Value,
        model: &NeuralBehaviourAnalyzer,
    ) -> Result<Self, String> {
        exact_keys(value, &["learning_rate", "steps", "state", "posterior"])?;
        let rate = number(&value["learning_rate"])?;
        let mut optimizer = Self::new(model, rate).map_err(|e| e.to_string())?;
        optimizer.steps = unsigned(&value["steps"])?;
        optimizer.state = MatrixMoments::load(&value["state"], &model.parameters.state)?;
        optimizer.posterior =
            MatrixMoments::load(&value["posterior"], &model.parameters.posterior)?;
        Ok(optimizer)
    }
}

/// Explicit seeded standard-normal stream. State is persisted in the checkpoint.
#[derive(Debug, Clone, Copy)]
pub(crate) struct NoiseStream {
    pub(crate) state: u64,
}
impl NoiseStream {
    pub(crate) fn new(seed: u64) -> Result<Self, AnalyzerError> {
        if seed == 0 {
            Err(AnalyzerError::InvalidInitialization)
        } else {
            Ok(Self { state: seed })
        }
    }
    fn uniform(&mut self) -> f64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        ((self.state >> 12) as f64 + 0.5) / (1_u64 << 52) as f64
    }
    pub(crate) fn normal(&mut self) -> f64 {
        let u = self.uniform();
        let v = self.uniform();
        (-2.0 * u.ln()).sqrt() * (2.0 * PI * v).cos()
    }
    pub(crate) fn sample(&mut self, dim: usize) -> Vec<f64> {
        (0..dim).map(|_| self.normal()).collect()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnalyzerError {
    InvalidSupport,
    NoSubjectDecisions,
    InvalidNoise,
    InvalidShape,
    InvalidInitialization,
    NonFiniteOutput,
    NonFiniteGradient,
    InvalidLearningRate,
    OptimizerModelMismatch,
    StepOverflow,
}
impl fmt::Display for AnalyzerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for AnalyzerError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::behavioural::{EVENT_CATEGORICAL_WIDTH, FeaturizedHandData, HandId, Position};
    fn support() -> Vec<FeaturizedHand> {
        let mut categorical = [0; EVENT_CATEGORICAL_WIDTH];
        categorical[CategoricalFeature::IsSubject as usize] = 1;
        categorical[CategoricalFeature::ActorPosition as usize] =
            i64::from(position_id(Position::Button));
        vec![FeaturizedHand::from_data(FeaturizedHandData {
            hand_id: HandId::new("support"),
            subject_position: Position::Button,
            sequence_indexes: vec![0],
            categorical: vec![categorical],
            continuous: vec![[0.0; EVENT_CONTINUOUS_WIDTH]],
            continuous_present: vec![[1; EVENT_CONTINUOUS_WIDTH]],
            decision_time_ms: vec![100],
        })]
    }
    #[test]
    fn posterior_is_finite_stochastic_and_kl_nonnegative() {
        let config = BehaviourLatentConfig::new(3).unwrap();
        let analyzer = NeuralBehaviourAnalyzer::initialize(config, 8, 17).unwrap();
        let posterior = analyzer.infer(&support()).unwrap();
        assert_eq!(posterior.mean().len(), 3);
        assert!(posterior.std().iter().all(|x| *x > 0.0));
        assert!(posterior.kl_to_standard_normal() >= -1e-12);
        assert_ne!(
            posterior.sample(&[1.0, 0.0, 0.0]).unwrap(),
            posterior.mean_profile().unwrap()
        );
    }
    #[test]
    fn analyzer_optimizer_updates_and_is_checkpointable() {
        let config = BehaviourLatentConfig::new(2).unwrap();
        let mut model = NeuralBehaviourAnalyzer::initialize(config, 8, 17).unwrap();
        let mut adam = AnalyzerAdam::new(&model, 0.01).unwrap();
        let before = model.parameters().clone();
        adam.train_step(&mut model, &support(), &[0.3, -0.2], &[0.7, -0.5], 0.01)
            .unwrap();
        assert_ne!(*model.parameters(), before);
        assert_eq!(adam.steps(), 1);
        let loaded = AnalyzerAdam::from_checkpoint(&adam.checkpoint_value(), &model).unwrap();
        assert_eq!(loaded.steps(), 1);
    }
    #[test]
    fn reparameterized_posterior_gradients_match_finite_differences() {
        let model =
            NeuralBehaviourAnalyzer::initialize(BehaviourLatentConfig::new(2).unwrap(), 8, 41)
                .unwrap();
        let epsilon = [0.3, -0.8];
        let latent_gradient = [0.7, -0.5];
        let beta = 0.04;
        let score = |model: &NeuralBehaviourAnalyzer| {
            let posterior = model.infer(&support()).unwrap();
            latent_gradient
                .iter()
                .zip(posterior.mean())
                .zip(posterior.std())
                .zip(epsilon)
                .map(|(((&gradient, &mu), &std), eps)| gradient * (mu + std * eps))
                .sum::<f64>()
                + beta * posterior.kl_to_standard_normal()
        };
        let pass = model.pass(&support()).unwrap();
        let h = 1e-4;
        for index in [0, 2] {
            let change = |direction: f32| {
                let mut changed = model.clone();
                let layer = &model.parameters.posterior;
                let mut bias = layer.biases().to_vec();
                bias[index] += direction;
                changed.parameters.posterior = DenseLayer::new(
                    layer.input_width(),
                    layer.output_width(),
                    layer.weights().to_vec(),
                    bias,
                )
                .unwrap();
                changed
            };
            let numeric = (score(&change(h as f32)) - score(&change(-(h as f32)))) / (2.0 * h);
            let analytic = if index == 0 {
                latent_gradient[0] + beta * pass.posterior.mean[0]
            } else {
                let sigma = pass.posterior.std[0];
                (latent_gradient[0] * sigma * epsilon[0] + beta * (sigma * sigma - 1.0))
                    * LOG_STD_RANGE
                    * (1.0 - pass.raw[2].tanh().powi(2))
            };
            assert!(
                (numeric - analytic).abs() < 0.002,
                "numerical gradient {numeric} vs {analytic}"
            );
        }
    }
    #[test]
    fn without_subject_actions_is_rejected() {
        let model =
            NeuralBehaviourAnalyzer::initialize(BehaviourLatentConfig::new(2).unwrap(), 8, 2)
                .unwrap();
        assert_eq!(
            model.infer(&[]).unwrap_err(),
            AnalyzerError::NoSubjectDecisions
        );
    }
}
