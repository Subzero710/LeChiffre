//! Small CPU MLP with explicit backpropagation and Adam. Float32 parameters;
//! float64 reductions for loss/gradient norms. No Python or tensor runtime.
//! This reference backend targets correctness on small games, not NLHE throughput.
use crate::{game::Observation, memory::Sample, rng::Rng};

pub struct LossItem<'a> {
    pub observation: &'a Observation,
    pub targets: &'a [f32],
    pub weight: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Mlp {
    pub(crate) sizes: Vec<usize>,
    // Each layer: output-major weights followed by biases.
    pub(crate) parameters: Vec<f32>,
}

impl Mlp {
    pub fn zeros(sizes: &[usize]) -> Self {
        assert!(sizes.len() >= 2 && sizes.iter().all(|&n| n > 0));
        let count = sizes.windows(2).map(|s| s[0] * s[1] + s[1]).sum();
        Self {
            sizes: sizes.to_vec(),
            parameters: vec![0.0; count],
        }
    }
    pub fn initialized(sizes: &[usize], rng: &mut Rng) -> Self {
        let mut model = Self::zeros(sizes);
        let mut offset = 0;
        for (layer, dimensions) in sizes.windows(2).enumerate() {
            let [inputs, outputs] = [dimensions[0], dimensions[1]];
            // Uniform He initialization for hidden ReLU layers, Xavier output.
            let bound = if layer + 2 == sizes.len() {
                (6.0 / (inputs + outputs) as f64).sqrt()
            } else {
                (6.0 / inputs as f64).sqrt()
            };
            for w in &mut model.parameters[offset..offset + inputs * outputs] {
                *w = ((2.0 * rng.unit() - 1.0) * bound) as f32;
            }
            offset += inputs * outputs + outputs;
        }
        model
    }
    pub fn sizes(&self) -> &[usize] {
        &self.sizes
    }
    pub fn parameter_count(&self) -> usize {
        self.parameters.len()
    }
    fn activations(&self, input: &[f32]) -> Vec<Vec<f32>> {
        assert_eq!(input.len(), self.sizes[0]);
        let mut activations = vec![input.to_vec()];
        let mut offset = 0;
        for (layer, sizes) in self.sizes.windows(2).enumerate() {
            let (inputs, outputs) = (sizes[0], sizes[1]);
            let previous = activations.last().unwrap();
            let biases = offset + inputs * outputs;
            let mut next = vec![0.0; outputs];
            for (o, value) in next.iter_mut().enumerate() {
                let row = &self.parameters[offset + o * inputs..offset + (o + 1) * inputs];
                *value = self.parameters[biases + o]
                    + row.iter().zip(previous).map(|(w, x)| w * x).sum::<f32>();
                if layer + 2 != self.sizes.len() {
                    *value = value.max(0.0);
                }
            }
            offset = biases + outputs;
            activations.push(next);
        }
        activations
    }
    pub fn predict(&self, input: &[f32]) -> Vec<f32> {
        self.activations(input).pop().unwrap()
    }
    /// Exact gradient of sum_s (t_s / mean_t) * sum_legal(error^2) / batch_size.
    /// `mean_t` is computed on the WHOLE uniform reservoir, so uniform batches
    /// have an unbiased gradient for its iteration-weighted objective.
    pub fn loss_and_gradient(&self, batch: &[&Sample], mean_t: f64) -> (f64, Vec<f32>) {
        assert!(!batch.is_empty() && mean_t.is_finite() && mean_t > 0.0);
        let weighted = batch
            .iter()
            .map(|s| LossItem {
                observation: &s.observation,
                targets: &s.targets,
                weight: s.iteration as f64 / mean_t / batch.len() as f64,
            })
            .collect::<Vec<_>>();
        self.weighted_loss_and_gradient(&weighted)
    }
    pub fn weighted_loss_and_gradient(&self, batch: &[LossItem<'_>]) -> (f64, Vec<f32>) {
        assert!(!batch.is_empty());
        let mut gradient = vec![0.0; self.parameters.len()];
        let mut loss = 0.0;
        for sample in batch {
            let acts = self.activations(&sample.observation.features);
            let output = acts.last().unwrap();
            assert_eq!(output.len(), sample.targets.len());
            assert_eq!(output.len(), sample.observation.legal.len());
            let weight = sample.weight;
            assert!(weight.is_finite() && weight >= 0.0);
            let mut delta = vec![0.0; output.len()];
            for (a, (&predicted, &target)) in output.iter().zip(sample.targets).enumerate() {
                if sample.observation.legal[a] {
                    let error = (predicted - target) as f64;
                    loss += weight * error * error;
                    delta[a] = (2.0 * weight * error) as f32;
                }
            }
            let mut end = self.parameters.len();
            for layer in (0..self.sizes.len() - 1).rev() {
                let (inputs, outputs) = (self.sizes[layer], self.sizes[layer + 1]);
                let biases = end - outputs;
                let offset = biases - inputs * outputs;
                let previous = &acts[layer];
                let mut previous_delta = vec![0.0; inputs];
                for (o, &d) in delta.iter().enumerate() {
                    gradient[biases + o] += d;
                    for (j, &x) in previous.iter().enumerate() {
                        let index = offset + o * inputs + j;
                        gradient[index] += d * x;
                        if layer > 0 {
                            previous_delta[j] += d * self.parameters[index];
                        }
                    }
                }
                if layer > 0 {
                    for (d, &x) in previous_delta.iter_mut().zip(previous) {
                        if x <= 0.0 {
                            *d = 0.0;
                        }
                    }
                }
                delta = previous_delta;
                end = offset;
            }
        }
        (loss, gradient)
    }
}

pub struct Adam {
    first: Vec<f32>,
    second: Vec<f32>,
    step: i32,
}

impl Adam {
    pub fn new(model: &Mlp) -> Self {
        Self {
            first: vec![0.0; model.parameter_count()],
            second: vec![0.0; model.parameter_count()],
            step: 0,
        }
    }
    pub fn update(
        &mut self,
        model: &mut Mlp,
        gradient: &[f32],
        learning_rate: f32,
        max_norm: f32,
    ) -> Result<(), String> {
        if gradient.len() != model.parameter_count() || gradient.iter().any(|g| !g.is_finite()) {
            return Err("invalid or non-finite gradient".into());
        }
        let norm = gradient
            .iter()
            .map(|&g| (g as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        let scale = (max_norm as f64 / norm.max(1e-30)).min(1.0) as f32;
        self.step = self.step.checked_add(1).ok_or("Adam step overflow")?;
        let c1 = 1.0 - 0.9_f32.powi(self.step);
        let c2 = 1.0 - 0.999_f32.powi(self.step);
        for (i, parameter) in model.parameters.iter_mut().enumerate() {
            let g = gradient[i] * scale;
            self.first[i] = 0.9 * self.first[i] + 0.1 * g;
            self.second[i] = 0.999 * self.second[i] + 0.001 * g * g;
            let m = self.first[i] / c1;
            let v = self.second[i] / c2;
            *parameter -= learning_rate * m / (v.sqrt() + 1e-8);
        }
        if model.parameters.iter().any(|x| !x.is_finite()) {
            return Err("non-finite MLP parameters".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::Observation;
    fn sample(x: Vec<f32>, y: Vec<f32>, iteration: u64) -> Sample {
        Sample {
            observation: Observation {
                features: x,
                legal: vec![true; y.len()],
            },
            targets: y,
            iteration,
        }
    }
    #[test]
    fn backprop_matches_finite_differences_with_iteration_weights() {
        let mut model = Mlp::initialized(&[2, 3, 3, 2], &mut Rng::new(91));
        let a = sample(vec![0.7, -0.4], vec![-0.8, 0.9], 1);
        let b = sample(vec![-0.2, 0.8], vec![0.3, -0.4], 3);
        let batch = [&a, &b];
        let (_, gradient) = model.loss_and_gradient(&batch, 2.0);
        for (i, &analytic) in gradient.iter().enumerate() {
            let original = model.parameters[i];
            model.parameters[i] = original + 0.001;
            let plus = model.loss_and_gradient(&batch, 2.0).0;
            model.parameters[i] = original - 0.001;
            let minus = model.loss_and_gradient(&batch, 2.0).0;
            model.parameters[i] = original;
            let numeric = (plus - minus) / 0.002;
            assert!(
                (numeric - analytic as f64).abs() < 0.003,
                "parameter {i}: analytic={analytic}, numeric={numeric}"
            );
        }
    }
    #[test]
    fn illegal_outputs_do_not_contribute_to_loss_or_gradient() {
        let model = Mlp::initialized(&[2, 4, 2], &mut Rng::new(3));
        let mut a = sample(vec![1.0, 0.0], vec![-1.0, 100.0], 2);
        a.observation.legal[1] = false;
        let mut b = a.clone();
        b.targets[1] = -50_000.0;
        assert_eq!(
            model.loss_and_gradient(&[&a], 2.0),
            model.loss_and_gradient(&[&b], 2.0)
        );
    }
    #[test]
    fn adam_learns_signed_targets() {
        let mut model = Mlp::initialized(&[2, 8, 2], &mut Rng::new(44));
        let a = sample(vec![1.0, 0.0], vec![-2.0, 3.0], 1);
        let b = sample(vec![0.0, 1.0], vec![1.0, -1.5], 1);
        let mut adam = Adam::new(&model);
        for _ in 0..400 {
            let (_, gradient) = model.loss_and_gradient(&[&a, &b], 1.0);
            adam.update(&mut model, &gradient, 0.02, 1.0).unwrap();
        }
        assert!(model.loss_and_gradient(&[&a, &b], 1.0).0 < 1e-6);
        assert!(model.predict(&a.observation.features)[0] < -1.9);
    }
    #[test]
    fn grouping_identical_observations_preserves_weighted_mse_gradient() {
        use crate::memory::compact_if_small;
        let model = Mlp::initialized(&[2, 4, 2], &mut Rng::new(5));
        let samples = [
            sample(vec![1.0, 0.0], vec![-2.0, 1.0], 1),
            sample(vec![1.0, 0.0], vec![0.5, -0.3], 3),
            sample(vec![0.0, 1.0], vec![0.8, 0.2], 2),
        ];
        let refs = samples.iter().collect::<Vec<_>>();
        let raw_gradient = model.loss_and_gradient(&refs, 2.0).1;
        let compact = compact_if_small(&samples, 2).unwrap();
        assert_eq!(compact.len(), 2);
        assert!(compact_if_small(&samples, 1).is_none());
        let batch = compact
            .iter()
            .map(|s| LossItem {
                observation: &s.observation,
                targets: &s.targets,
                weight: s.weight,
            })
            .collect::<Vec<_>>();
        let compact_gradient = model.weighted_loss_and_gradient(&batch).1;
        for (a, b) in raw_gradient.iter().zip(compact_gradient) {
            assert!((a - b).abs() < 1e-6);
        }
    }
}
