use crate::{game::Observation, neural::Mlp, rng::Rng};

/// Standard regret matching: preserve signed predictions, normalize positive
/// legal regrets, use uniform legal actions when no positive regret exists.
pub fn regret_matching(regrets: &[f32], legal: &[bool]) -> Vec<f32> {
    assert_eq!(regrets.len(), legal.len());
    assert!(regrets.iter().all(|r| r.is_finite()));
    let count = legal.iter().filter(|&&b| b).count();
    assert!(count > 0);
    let sum: f64 = regrets
        .iter()
        .zip(legal)
        .filter(|(_, ok)| **ok)
        .map(|(&r, _)| r.max(0.0) as f64)
        .sum();
    regrets
        .iter()
        .zip(legal)
        .map(|(&r, &ok)| {
            if !ok {
                0.0
            } else if sum > 0.0 {
                (r.max(0.0) as f64 / sum) as f32
            } else {
                1.0 / count as f32
            }
        })
        .collect()
}

#[derive(Clone, Debug, PartialEq)]
pub struct Snapshot {
    pub iteration: u64,
    pub model: Mlp,
}

impl Snapshot {
    pub fn policy(&self, observation: &Observation) -> Vec<f32> {
        regret_matching(
            &self.model.predict(&observation.features),
            &observation.legal,
        )
    }
}

/// Exact behavioral average of the stored neural policies at an information set.
/// own_prefix contains ONLY this player's previous decisions and chosen actions,
/// in order. It contains neither opponent decisions nor chance probabilities.
pub fn average_policy(
    archive: &[Snapshot],
    observation: &Observation,
    own_prefix: &[(Observation, usize)],
) -> Vec<f32> {
    assert!(!archive.is_empty());
    let mut numerator = vec![0.0_f64; observation.legal.len()];
    let mut denominator = 0.0;
    for snapshot in archive {
        let reach = own_prefix
            .iter()
            .map(|(obs, action)| snapshot.policy(obs)[*action] as f64)
            .product::<f64>();
        let weight = snapshot.iteration as f64 * reach;
        let policy = snapshot.policy(observation);
        for (out, p) in numerator.iter_mut().zip(policy) {
            *out += weight * p as f64;
        }
        denominator += weight;
    }
    if denominator == 0.0 {
        return regret_matching(&vec![0.0; observation.legal.len()], &observation.legal);
    }
    numerator
        .into_iter()
        .map(|n| (n / denominator) as f32)
        .collect()
}

/// Pick once at the root of a hand and keep this snapshot for the whole hand.
pub fn sample_snapshot<'a>(archive: &'a [Snapshot], rng: &mut Rng) -> &'a Snapshot {
    assert!(!archive.is_empty());
    let total: f64 = archive.iter().map(|s| s.iteration as f64).sum();
    let threshold = rng.unit() * total;
    let mut cumulative = 0.0;
    for snapshot in archive {
        cumulative += snapshot.iteration as f64;
        if threshold < cumulative {
            return snapshot;
        }
    }
    archive.last().unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::Kuhn;
    fn model(root: [f32; 2], later: [f32; 2]) -> Mlp {
        let mut m = Mlp::zeros(&[7, 2]);
        for action in 0..2 {
            m.parameters[action * 7 + 3] = root[action];
            m.parameters[action * 7 + 6] = later[action];
        }
        m
    }
    #[test]
    fn average_uses_iteration_and_own_reach_weights() {
        let archive = [
            Snapshot {
                iteration: 1,
                model: model([9.0, 1.0], [8.0, 2.0]),
            },
            Snapshot {
                iteration: 3,
                model: model([1.0, 9.0], [2.0, 8.0]),
            },
        ];
        let later = Kuhn::observation_for(1, 3);
        let prefix = [(Kuhn::observation_for(1, 0), 0)];
        let p = average_policy(&archive, &later, &prefix);
        // (1 * .9 * .2 + 3 * .1 * .8) / (1 * .9 + 3 * .1) = .35
        assert!((p[1] - 0.35).abs() < 1e-6);
        assert!((average_policy(&archive, &later, &[])[1] - 0.65).abs() < 1e-6);
    }
    #[test]
    fn trajectory_sampling_uses_iteration_weights() {
        let archive = [
            Snapshot {
                iteration: 1,
                model: model([1.0; 2], [1.0; 2]),
            },
            Snapshot {
                iteration: 3,
                model: model([1.0; 2], [1.0; 2]),
            },
        ];
        let mut rng = Rng::new(4);
        let first = (0..10_000)
            .filter(|_| sample_snapshot(&archive, &mut rng).iteration == 1)
            .count();
        assert!((2300..2700).contains(&first));
    }
}
