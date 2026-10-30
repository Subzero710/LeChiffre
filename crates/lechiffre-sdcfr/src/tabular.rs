//! Full-tree linear CFR oracle for Kuhn; never used as neural training data.
use crate::{
    evaluation::{PolicyTable, uniform_policy},
    game::{Game, Kuhn, Node},
    strategy::regret_matching,
};

pub fn train(iterations: u64) -> PolicyTable {
    assert!(iterations > 0);
    let mut regrets = [[0.0_f64; 2]; 12];
    let mut average = [[0.0_f64; 2]; 12];
    for iteration in 1..=iterations {
        let mut policy = uniform_policy();
        for i in 0..12 {
            let p = regret_matching(&regrets[i].map(|r| r as f32), &[true, true]);
            policy[i] = [p[0], p[1]];
        }
        // Simultaneous update: keep the same policy for both traversers.
        for player in 0..2 {
            for first in 0..3 {
                for second in 0..3 {
                    if first != second {
                        walk(
                            Kuhn::deal([first, second]),
                            player,
                            [1.0; 2],
                            &policy,
                            &mut regrets,
                            &mut average,
                            iteration as f64 / 6.0,
                        );
                    }
                }
            }
        }
    }
    let mut policy = uniform_policy();
    for i in 0..12 {
        let sum = average[i][0] + average[i][1];
        if sum > 0.0 {
            policy[i] = average[i].map(|p| (p / sum) as f32);
        }
    }
    policy
}

fn walk(
    state: Kuhn,
    traverser: usize,
    reach: [f64; 2],
    policy: &PolicyTable,
    regrets: &mut [[f64; 2]; 12],
    average: &mut [[f64; 2]; 12],
    weight: f64,
) -> f64 {
    let Node::Player(player) = state.node() else {
        return state.payoff(traverser) as f64;
    };
    let info = state.infoset();
    let probabilities = policy[info];
    let mut values = [0.0; 2];
    for action in 0..2 {
        let mut next_reach = reach;
        next_reach[player] *= probabilities[action] as f64;
        values[action] = walk(
            state.child(action),
            traverser,
            next_reach,
            policy,
            regrets,
            average,
            weight,
        );
    }
    let expected = probabilities
        .iter()
        .zip(values)
        .map(|(&p, v)| p as f64 * v)
        .sum();
    if player == traverser {
        for action in 0..2 {
            regrets[info][action] += weight * reach[1 - player] * (values[action] - expected);
            average[info][action] += weight * reach[player] * probabilities[action] as f64;
        }
    }
    expected
}
