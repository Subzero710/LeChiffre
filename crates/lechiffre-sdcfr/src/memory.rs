use crate::{game::Observation, rng::Rng};
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq)]
pub struct Sample {
    pub observation: Observation,
    pub targets: Vec<f32>,
    pub iteration: u64,
}

/// Algorithm R: each of the `seen` observations has capacity/seen probability
/// of remaining stored. Iteration weighting is applied only in the MSE.
#[derive(Clone, Debug, PartialEq)]
pub struct Reservoir {
    pub(crate) capacity: usize,
    pub(crate) seen: u64,
    pub(crate) samples: Vec<Sample>,
}

impl Reservoir {
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0);
        Self {
            capacity,
            seen: 0,
            samples: Vec::new(),
        }
    }
    pub fn push(&mut self, sample: Sample, rng: &mut Rng) {
        self.seen = self.seen.checked_add(1).expect("reservoir count overflow");
        if self.samples.len() < self.capacity {
            self.samples.push(sample);
        } else {
            let slot = rng.below(self.seen);
            if slot < self.capacity as u64 {
                self.samples[slot as usize] = sample;
            }
        }
    }
    pub fn seen(&self) -> u64 {
        self.seen
    }
    pub fn samples(&self) -> &[Sample] {
        &self.samples
    }
}

/// Sufficient statistics for repeated IDENTICAL observations. Weighted mean
/// targets preserve the full reservoir's MSE gradient exactly; only an additive
/// target-variance constant disappears from the loss. No information sets merge.
pub struct FitPoint {
    pub observation: Observation,
    pub targets: Vec<f32>,
    pub weight: f64,
}

pub fn compact_if_small(samples: &[Sample], limit: usize) -> Option<Vec<FitPoint>> {
    struct Accumulator {
        observation: Observation,
        sums: Vec<f64>,
        weight: f64,
    }
    let mut groups = BTreeMap::new();
    let mut total_weight = 0.0;
    for sample in samples {
        let key = (
            sample
                .observation
                .features
                .iter()
                .map(|f| f.to_bits())
                .collect::<Vec<_>>(),
            sample.observation.legal.clone(),
        );
        if !groups.contains_key(&key) && groups.len() >= limit {
            return None;
        }
        let entry = groups.entry(key).or_insert_with(|| Accumulator {
            observation: sample.observation.clone(),
            sums: vec![0.0; sample.targets.len()],
            weight: 0.0,
        });
        let weight = sample.iteration as f64;
        for (sum, &target) in entry.sums.iter_mut().zip(&sample.targets) {
            *sum += weight * target as f64;
        }
        entry.weight += weight;
        total_weight += weight;
    }
    Some(
        groups
            .into_values()
            .map(|g| FitPoint {
                observation: g.observation,
                targets: g.sums.into_iter().map(|s| (s / g.weight) as f32).collect(),
                weight: g.weight / total_weight,
            })
            .collect(),
    )
}
