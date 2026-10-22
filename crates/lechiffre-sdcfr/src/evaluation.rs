//! Exact Kuhn evaluation. Best responses choose one action per INFORMATION SET,
//! consistently across both possible hidden opponent cards.
use crate::{
    game::{Game, Kuhn, Node},
    strategy::average_policy,
    training::Trainer,
};

pub type PolicyTable = [[f32; 2]; 12];

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Evaluation {
    pub value_player0: f64,
    pub best_response_values: [f64; 2],
    pub nash_conv: f64,
    /// Convention used here: NashConv / 2, in ante chips per hand.
    pub exploitability: f64,
}

pub fn uniform_policy() -> PolicyTable {
    [[0.5; 2]; 12]
}

pub fn average_table(trainer: &Trainer) -> PolicyTable {
    assert_eq!(trainer.game_id, Kuhn::GAME_ID);
    let mut table = uniform_policy();
    for card in 0..3 {
        for history in 0..4 {
            let player = if history == 0 || history == 3 { 0 } else { 1 };
            let obs = Kuhn::observation_for(card, history);
            let own_prefix = if history == 3 {
                vec![(Kuhn::observation_for(card, 0), 0)]
            } else {
                Vec::new()
            };
            let p = average_policy(trainer.archive(player), &obs, &own_prefix);
            table[card as usize * 4 + history as usize] = [p[0], p[1]];
        }
    }
    table
}

/// Exact iteration-weighted external regret of the archived joint policies,
/// divided by total iteration weight. Signed finite-horizon regrets can be
/// negative; their sum equals NashConv. The comparator is a fixed policy.
pub fn sequence_regret(trainer: &Trainer, average_evaluation: &Evaluation) -> [f64; 2] {
    assert_eq!(trainer.game_id, Kuhn::GAME_ID);
    assert!(trainer.completed() > 0);
    let mut actual_value = 0.0;
    let mut total_weight = 0.0;
    for index in 0..trainer.completed() as usize {
        let mut policy = uniform_policy();
        for card in 0..3 {
            for history in 0..4 {
                let player = if history == 0 || history == 3 { 0 } else { 1 };
                let snapshot = &trainer.archive(player)[index];
                let p = snapshot.policy(&Kuhn::observation_for(card, history));
                policy[card as usize * 4 + history as usize] = [p[0], p[1]];
            }
        }
        let weight = trainer.archive(0)[index].iteration as f64;
        actual_value += weight * expected_value(&policy);
        total_weight += weight;
    }
    let actual_value = actual_value / total_weight;
    [
        average_evaluation.best_response_values[0] - actual_value,
        average_evaluation.best_response_values[1] + actual_value,
    ]
}

fn value(state: Kuhn, policy: &PolicyTable) -> f64 {
    if state.node() == Node::Terminal {
        return state.payoff(0) as f64;
    }
    let p = policy[state.infoset()];
    // Renormalize tiny f32 summation error while evaluating in float64.
    let sum = p[0] as f64 + p[1] as f64;
    (p[0] as f64 * value(state.child(0), policy) + p[1] as f64 * value(state.child(1), policy))
        / sum
}

pub fn expected_value(policy: &PolicyTable) -> f64 {
    let mut total = 0.0;
    for first in 0..3 {
        for second in 0..3 {
            if first != second {
                total += value(Kuhn::deal([first, second]), policy) / 6.0;
            }
        }
    }
    total
}

pub fn evaluate(policy: &PolicyTable) -> Evaluation {
    assert!(
        policy
            .iter()
            .all(|p| p.iter().all(|x| x.is_finite() && *x >= 0.0)
                && (p[0] as f64 + p[1] as f64 - 1.0).abs() < 1e-5)
    );
    let mut best = [f64::NEG_INFINITY; 2];
    for (player, best_value) in best.iter_mut().enumerate() {
        // Each player has six information sets, hence 2^6 pure strategies.
        for bits in 0..64 {
            let mut response = *policy;
            for card in 0..3 {
                let histories = if player == 0 { [0, 3] } else { [1, 2] };
                for (j, history) in histories.into_iter().enumerate() {
                    let action = (bits >> (card * 2 + j)) & 1;
                    response[card * 4 + history] =
                        if action == 0 { [1.0, 0.0] } else { [0.0, 1.0] };
                }
            }
            let v = expected_value(&response) * if player == 0 { 1.0 } else { -1.0 };
            *best_value = best_value.max(v);
        }
    }
    let nash_conv = (best[0] + best[1]).max(0.0);
    Evaluation {
        value_player0: expected_value(policy),
        best_response_values: best,
        nash_conv,
        exploitability: nash_conv / 2.0,
    }
}

/// One member (alpha = 1/3) of the analytic Kuhn equilibrium family.
pub fn nash_policy() -> PolicyTable {
    let mut table = uniform_policy();
    let aggressive = [
        [1.0 / 3.0, 1.0 / 3.0, 0.0, 0.0], // J: root, c, b, cb
        [0.0, 0.0, 1.0 / 3.0, 2.0 / 3.0], // Q
        [1.0, 1.0, 1.0, 1.0],             // K
    ];
    for card in 0..3 {
        for history in 0..4 {
            let p = aggressive[card][history];
            table[card * 4 + history] = [1.0 - p, p];
        }
    }
    table
}
