//! Paired head-to-head evaluation of fixed information-set policies.
//! Each duplicate has the same cards and random streams, with agents swapped.
//! Standard errors use PAIR means, preserving within-pair correlation.
use crate::{
    exact::{Policy, Tree},
    game::Kuhn,
    leduc::Leduc,
    rng::Rng,
};

#[derive(Clone, Copy, Debug)]
pub enum Baseline {
    Random,
    Passive,
    Aggressive,
}

impl Baseline {
    pub fn policy(self, tree: &Tree) -> Result<Policy, String> {
        let leduc = match tree.game_id() {
            Leduc::GAME_ID => true,
            Kuhn::GAME_ID => false,
            _ => return Err("baseline: unsupported game".into()),
        };
        let mut result = tree.uniform_policy();
        if matches!(self, Self::Random) {
            return Ok(result);
        }
        for (row, info) in result.iter_mut().zip(tree.infosets()) {
            let action = if leduc {
                match self {
                    Self::Passive => 1,
                    Self::Aggressive => {
                        if info.observation.legal[2] {
                            2
                        } else {
                            1
                        }
                    }
                    Self::Random => unreachable!(),
                }
            } else {
                // Kuhn slots: 0 = check/fold, 1 = bet/call.
                let facing_bet =
                    info.observation.features[5] == 1.0 || info.observation.features[6] == 1.0;
                match self {
                    Self::Passive => usize::from(facing_bet),
                    Self::Aggressive => 1,
                    Self::Random => unreachable!(),
                }
            };
            assert!(info.observation.legal[action]);
            row.fill(0.0);
            row[action] = 1.0;
        }
        Ok(result)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MatchReport {
    pub hands: u64,
    pub pairs: u64,
    pub total_chips: f64,
    pub chips_per_hand: f64,
    pub chips_per_100: f64,
    pub ci95_per_100: [f64; 2],
    pub exact_per_100: f64,
    pub mean_by_seat: [f64; 2],
}

#[derive(Default)]
struct PairStats {
    n: u64,
    mean: f64,
    m2: f64,
}
impl PairStats {
    fn add(&mut self, x: f64) {
        self.n += 1;
        let delta = x - self.mean;
        self.mean += delta / self.n as f64;
        self.m2 += delta * (x - self.mean);
    }
    fn standard_error(&self) -> f64 {
        (self.m2.max(0.0) / (self.n - 1) as f64 / self.n as f64).sqrt()
    }
}

/// An even number of hands, at least four. Confidence intervals are approximate
/// normal 95% intervals over independent duplicate pairs (large-sample use).
pub fn duel(
    tree: &Tree,
    hero: &Policy,
    opponent: &Policy,
    hands: u64,
    seed: u64,
) -> Result<MatchReport, String> {
    if hands < 4 || (hands & 1) != 0 {
        return Err("hands must be even and at least 4".into());
    }
    let exact = tree.head_to_head_value(hero, opponent);
    let mut master = Rng::new(seed);
    let mut statistics = PairStats::default();
    let mut totals = [0.0; 2];
    for _ in 0..hands / 2 {
        let seeds = [master.next_u64(), master.next_u64(), master.next_u64()];
        let mut pair = [0.0; 2];
        for seat in 0..2 {
            let mut streams = seeds.map(Rng::new);
            pair[seat] = tree.sample_duel_hand(hero, opponent, seat, &mut streams);
            totals[seat] += pair[seat];
        }
        statistics.add((pair[0] + pair[1]) / 2.0);
    }
    let half_width = 1.959963984540054 * statistics.standard_error() * 100.0;
    let per100 = statistics.mean * 100.0;
    Ok(MatchReport {
        hands,
        pairs: statistics.n,
        total_chips: totals.iter().sum(),
        chips_per_hand: statistics.mean,
        chips_per_100: per100,
        ci95_per_100: [per100 - half_width, per100 + half_width],
        exact_per_100: exact * 100.0,
        mean_by_seat: totals.map(|t| t / statistics.n as f64),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pair_standard_error_uses_pairs_instead_of_individual_hands() {
        let mut stats = PairStats::default();
        for x in [1.0, 2.0, 3.0, 4.0] {
            stats.add(x);
        }
        assert_eq!(stats.mean, 2.5);
        assert!((stats.standard_error() - (5.0_f64 / 12.0).sqrt()).abs() < 1e-12);
    }
}
