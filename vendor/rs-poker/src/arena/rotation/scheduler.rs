use itertools::Itertools;

/// One independent table rotation. The seating stays fixed for the entire
/// block; the dealer button advances once per hand.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RotationJob {
    pub rotation_idx: usize,
    /// Agent indices in physical seat order.
    pub seating: Vec<usize>,
    /// Dealer seat for the first hand in the block.
    pub starting_dealer: usize,
    /// Deterministic seed dedicated to this rotation.
    pub seed: u64,
}

/// All circularly unique seatings for `players_per_table` selected from
/// `num_agents`. Rotating every seat together does not create a new seating
/// because the button will make a complete lap inside the block.
pub fn circular_seatings(num_agents: usize, players_per_table: usize) -> Vec<Vec<usize>> {
    let mut result = Vec::new();
    for group in (0..num_agents).combinations(players_per_table) {
        let anchor = group[0];
        for tail in group[1..]
            .iter()
            .copied()
            .permutations(players_per_table - 1)
        {
            let mut seating = Vec::with_capacity(players_per_table);
            seating.push(anchor);
            seating.extend(tail);
            result.push(seating);
        }
    }
    result
}

/// Build a deterministic schedule that consumes every circular seating once
/// per cycle before any seating repeats. `seed` changes only cycle order and
/// the initial button phase, never which seatings are present.
pub fn build_schedule(
    num_agents: usize,
    players_per_table: usize,
    num_rotations: usize,
    seed: u64,
) -> Vec<RotationJob> {
    let seatings = circular_seatings(num_agents, players_per_table);
    let per_cycle = seatings.len();
    let mut jobs = Vec::with_capacity(num_rotations);
    let mut rotation_idx = 0usize;
    let mut cycle = 0usize;

    while rotation_idx < num_rotations {
        let mut order: Vec<usize> = (0..per_cycle).collect();
        order.sort_by_key(|&seating_idx| {
            (
                mix64(
                    seed ^ (cycle as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                        ^ (seating_idx as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93),
                ),
                seating_idx,
            )
        });

        for seating_idx in order {
            if rotation_idx >= num_rotations {
                break;
            }
            let base_dealer =
                (mix64(seed ^ (seating_idx as u64).wrapping_mul(0xA076_1D64_78BD_642F))
                    % players_per_table as u64) as usize;
            let starting_dealer = (base_dealer + cycle) % players_per_table;
            let job_seed = mix64(seed ^ (rotation_idx as u64).wrapping_mul(0xE703_7ED1_A0B4_28DB));

            jobs.push(RotationJob {
                rotation_idx,
                seating: seatings[seating_idx].clone(),
                starting_dealer,
                seed: job_seed,
            });
            rotation_idx += 1;
        }
        cycle += 1;
    }

    jobs
}

/// Stable SplitMix64-style mixer used for deterministic sub-seeds and ordering.
pub(crate) fn mix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn six_agents_have_expected_circular_seating_counts() {
        let expected = [(2, 15), (3, 40), (4, 90), (5, 144), (6, 120)];
        for (players, count) in expected {
            assert_eq!(circular_seatings(6, players).len(), count);
        }
    }

    #[test]
    fn a_cycle_contains_no_duplicate_seatings() {
        let jobs = build_schedule(6, 6, 120, 42);
        let unique: HashSet<Vec<usize>> = jobs.iter().map(|j| j.seating.clone()).collect();
        assert_eq!(unique.len(), 120);
    }

    #[test]
    fn same_seed_produces_same_schedule() {
        assert_eq!(build_schedule(6, 6, 250, 42), build_schedule(6, 6, 250, 42));
    }

    #[test]
    fn repeated_seating_rotates_starting_button_across_cycles() {
        let seatings = circular_seatings(6, 6).len();
        let jobs = build_schedule(6, 6, seatings * 6, 7);
        let target = jobs[0].seating.clone();
        let dealers: Vec<usize> = jobs
            .iter()
            .filter(|j| j.seating == target)
            .map(|j| j.starting_dealer)
            .collect();
        assert_eq!(dealers.len(), 6);
        assert_eq!(dealers.iter().copied().collect::<HashSet<_>>().len(), 6);
    }
}
