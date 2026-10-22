//! Exact two-player zero-sum evaluation for small, finite, perfect-recall games.
//! Best-response actions are shared across all hidden states of an information set.
use crate::{
    evaluation::Evaluation,
    game::{EnumeratedGame, Node, Observation},
    strategy::regret_matching,
    training::Trainer,
};
use std::collections::BTreeMap;

pub type Policy = Vec<Vec<f64>>;

#[derive(Clone, Debug)]
pub struct InformationSet {
    pub player: usize,
    pub observation: Observation,
    pub own_prefix: Vec<(usize, usize)>,
    pub label: String,
    nodes: Vec<usize>,
}

#[derive(Clone, Debug)]
enum Kind {
    Chance,
    Player(usize),
    Terminal(f64),
}
#[derive(Clone, Debug)]
struct Edge {
    action: usize,
    child: usize,
    probability: f64,
}
#[derive(Clone, Debug)]
struct TreeNode {
    kind: Kind,
    edges: Vec<Edge>,
}

pub struct Tree {
    game_id: String,
    nodes: Vec<TreeNode>,
    infosets: Vec<InformationSet>,
}

pub struct Analysis {
    pub policy: Policy,
    pub evaluation: Evaluation,
    pub external_regret: [f64; 2],
}

type InfoKey = (usize, Vec<u32>, Vec<bool>);
impl Tree {
    pub fn compile<G: EnumeratedGame>(root: &G, game_id: &str) -> Self {
        let mut tree = Self {
            game_id: game_id.into(),
            nodes: Vec::new(),
            infosets: Vec::new(),
        };
        tree.add(root, &[Vec::new(), Vec::new()], &mut BTreeMap::new());
        tree
    }
    pub fn infosets(&self) -> &[InformationSet] {
        &self.infosets
    }
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
    pub fn game_id(&self) -> &str {
        &self.game_id
    }

    /// Hero's expectation against an opponent, averaging both seats equally.
    pub fn head_to_head_value(&self, hero: &Policy, opponent: &Policy) -> f64 {
        self.validate_policy(hero);
        self.validate_policy(opponent);
        let profile = |hero_seat| {
            self.infosets
                .iter()
                .enumerate()
                .map(|(id, info)| {
                    if info.player == hero_seat {
                        hero[id].clone()
                    } else {
                        opponent[id].clone()
                    }
                })
                .collect::<Policy>()
        };
        (self.expected_value(&profile(0)) - self.expected_value(&profile(1))) / 2.0
    }

    /// Streams: chance, seat-0 decisions, seat-1 decisions. Reset all three for
    /// the second hand of a duplicate pair, exchanging the agents' seats.
    pub(crate) fn sample_duel_hand(
        &self,
        hero: &Policy,
        opponent: &Policy,
        hero_seat: usize,
        streams: &mut [crate::rng::Rng; 3],
    ) -> f64 {
        let mut id = 0;
        loop {
            let node = &self.nodes[id];
            let (draw, policy) = match node.kind {
                Kind::Terminal(payoff) => return if hero_seat == 0 { payoff } else { -payoff },
                Kind::Chance => (streams[0].unit(), None),
                Kind::Player(info) => {
                    let player = self.infosets[info].player;
                    let p = if player == hero_seat { hero } else { opponent };
                    (streams[1 + player].unit(), Some(&p[info]))
                }
            };
            let mut cumulative = 0.0;
            let mut chosen = None;
            for edge in &node.edges {
                let probability = policy.map_or(edge.probability, |row| row[edge.action]);
                if probability > 0.0 {
                    chosen = Some(edge.child);
                    cumulative += probability;
                    if draw < cumulative {
                        break;
                    }
                }
            }
            id = chosen.expect("empty sampling distribution");
        }
    }
    fn add<G: EnumeratedGame>(
        &mut self,
        state: &G,
        prefix: &[Vec<(usize, usize)>; 2],
        index: &mut BTreeMap<InfoKey, usize>,
    ) -> usize {
        let node_id = self.nodes.len();
        self.nodes.push(TreeNode {
            kind: Kind::Chance,
            edges: Vec::new(),
        });
        let (kind, edges) = match state.node() {
            Node::Terminal => {
                let payoff = state.payoff(0) as f64;
                assert!((payoff + state.payoff(1) as f64).abs() < 1e-8);
                (Kind::Terminal(payoff), Vec::new())
            }
            Node::Chance => {
                let outcomes = state.chance_outcomes();
                assert!(!outcomes.is_empty());
                assert!((outcomes.iter().map(|(_, p)| p).sum::<f64>() - 1.0).abs() < 1e-10);
                let edges = outcomes
                    .into_iter()
                    .enumerate()
                    .map(|(action, (child, p))| {
                        assert!(p.is_finite() && p > 0.0);
                        Edge {
                            action,
                            child: self.add(&child, prefix, index),
                            probability: p,
                        }
                    })
                    .collect();
                (Kind::Chance, edges)
            }
            Node::Player(player) => {
                assert!(player < 2);
                let observation = state.observation();
                let key = (
                    player,
                    observation.features.iter().map(|x| x.to_bits()).collect(),
                    observation.legal.clone(),
                );
                let info = if let Some(&existing) = index.get(&key) {
                    // This check rejects accidental history truncation or lost recall.
                    assert_eq!(
                        self.infosets[existing].own_prefix, prefix[player],
                        "imperfect recall"
                    );
                    existing
                } else {
                    let id = self.infosets.len();
                    index.insert(key, id);
                    let label = observation
                        .features
                        .iter()
                        .enumerate()
                        .filter(|(_, x)| **x != 0.0)
                        .map(|(i, x)| format!("{i}:{x:.6}"))
                        .collect::<Vec<_>>()
                        .join(";");
                    self.infosets.push(InformationSet {
                        player,
                        observation: observation.clone(),
                        own_prefix: prefix[player].clone(),
                        label,
                        nodes: Vec::new(),
                    });
                    id
                };
                self.infosets[info].nodes.push(node_id);
                let edges = observation
                    .legal
                    .iter()
                    .enumerate()
                    .filter(|(_, ok)| **ok)
                    .map(|(action, _)| {
                        let mut next_prefix = prefix.clone();
                        next_prefix[player].push((info, action));
                        Edge {
                            action,
                            child: self.add(&state.child(action), &next_prefix, index),
                            probability: 1.0,
                        }
                    })
                    .collect();
                (Kind::Player(info), edges)
            }
        };
        self.nodes[node_id] = TreeNode { kind, edges };
        node_id
    }
    pub fn uniform_policy(&self) -> Policy {
        self.infosets
            .iter()
            .map(|i| {
                let n = i.observation.legal.iter().filter(|&&b| b).count() as f64;
                i.observation
                    .legal
                    .iter()
                    .map(|&b| if b { 1.0 / n } else { 0.0 })
                    .collect()
            })
            .collect()
    }
    fn validate_policy(&self, policy: &Policy) {
        assert_eq!(policy.len(), self.infosets.len());
        for (row, info) in policy.iter().zip(&self.infosets) {
            assert_eq!(row.len(), info.observation.legal.len());
            assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-8);
            assert!(
                row.iter()
                    .zip(&info.observation.legal)
                    .all(|(&p, &ok)| p.is_finite() && p >= 0.0 && (ok || p == 0.0))
            );
        }
    }
    pub fn expected_value(&self, policy: &Policy) -> f64 {
        let mut values = vec![0.0; self.nodes.len()];
        for (id, node) in self.nodes.iter().enumerate().rev() {
            values[id] = match node.kind {
                Kind::Terminal(payoff) => payoff,
                Kind::Chance => node
                    .edges
                    .iter()
                    .map(|e| e.probability * values[e.child])
                    .sum(),
                Kind::Player(info) => node
                    .edges
                    .iter()
                    .map(|e| policy[info][e.action] * values[e.child])
                    .sum(),
            };
        }
        values[0]
    }
    pub fn best_response(&self, policy: &Policy, player: usize) -> f64 {
        assert!(player < 2);
        let mut reach = vec![0.0; self.nodes.len()];
        reach[0] = 1.0;
        for (id, node) in self.nodes.iter().enumerate() {
            for edge in &node.edges {
                let factor = match node.kind {
                    Kind::Chance => edge.probability,
                    Kind::Player(info) if self.infosets[info].player != player => {
                        policy[info][edge.action]
                    }
                    _ => 1.0,
                };
                reach[edge.child] = reach[id] * factor;
            }
        }
        self.br_value(
            0,
            player,
            policy,
            &reach,
            &mut vec![None; self.nodes.len()],
            &mut vec![None; self.infosets.len()],
        )
    }
    fn br_value(
        &self,
        id: usize,
        player: usize,
        policy: &Policy,
        reach: &[f64],
        values: &mut [Option<f64>],
        actions: &mut [Option<usize>],
    ) -> f64 {
        if let Some(value) = values[id] {
            return value;
        }
        let node = &self.nodes[id];
        let value = match node.kind {
            Kind::Terminal(payoff) => {
                if player == 0 {
                    payoff
                } else {
                    -payoff
                }
            }
            Kind::Player(info) if self.infosets[info].player == player => {
                let action = if let Some(action) = actions[info] {
                    action
                } else {
                    let mut best = (f64::NEG_INFINITY, 0);
                    for edge in &node.edges {
                        let mut total = 0.0;
                        for &member in &self.infosets[info].nodes {
                            let child = self.nodes[member]
                                .edges
                                .iter()
                                .find(|e| e.action == edge.action)
                                .unwrap()
                                .child;
                            total += reach[member]
                                * self.br_value(child, player, policy, reach, values, actions);
                        }
                        if total > best.0 {
                            best = (total, edge.action);
                        }
                    }
                    actions[info] = Some(best.1);
                    best.1
                };
                let child = node
                    .edges
                    .iter()
                    .find(|e| e.action == action)
                    .unwrap()
                    .child;
                self.br_value(child, player, policy, reach, values, actions)
            }
            _ => node
                .edges
                .iter()
                .map(|edge| {
                    let probability = match node.kind {
                        Kind::Chance => edge.probability,
                        Kind::Player(info) => policy[info][edge.action],
                        _ => unreachable!(),
                    };
                    probability * self.br_value(edge.child, player, policy, reach, values, actions)
                })
                .sum(),
        };
        values[id] = Some(value);
        value
    }
    pub fn evaluate(&self, policy: &Policy) -> Evaluation {
        self.validate_policy(policy);
        let best = [self.best_response(policy, 0), self.best_response(policy, 1)];
        let nash_conv = (best[0] + best[1]).max(0.0);
        Evaluation {
            value_player0: self.expected_value(policy),
            best_response_values: best,
            nash_conv,
            exploitability: nash_conv / 2.0,
        }
    }
    pub fn analyze(&self, trainer: &Trainer) -> Analysis {
        assert_eq!(self.game_id, trainer.game_id);
        assert!(trainer.completed() > 0);
        let mut numerator = self.uniform_policy();
        for row in &mut numerator {
            row.fill(0.0);
        }
        let mut denominator = vec![0.0; self.infosets.len()];
        let mut actual = 0.0;
        let mut total_weight = 0.0;
        for t in 0..trainer.completed() as usize {
            let weight = trainer.archive(0)[t].iteration as f64;
            let profile = self
                .infosets
                .iter()
                .map(|info| {
                    let p = trainer.archive(info.player)[t].policy(&info.observation);
                    let sum = p.iter().map(|&x| x as f64).sum::<f64>();
                    p.into_iter().map(|x| x as f64 / sum).collect::<Vec<_>>()
                })
                .collect::<Policy>();
            actual += weight * self.expected_value(&profile);
            total_weight += weight;
            for (id, info) in self.infosets.iter().enumerate() {
                let own_reach = info
                    .own_prefix
                    .iter()
                    .map(|&(past, action)| profile[past][action])
                    .product::<f64>();
                let w = weight * own_reach;
                for (out, &p) in numerator[id].iter_mut().zip(&profile[id]) {
                    *out += w * p;
                }
                denominator[id] += w;
            }
        }
        let uniform = self.uniform_policy();
        for (id, row) in numerator.iter_mut().enumerate() {
            if denominator[id] > 0.0 {
                for p in row {
                    *p /= denominator[id];
                }
            } else {
                *row = uniform[id].clone();
            }
        }
        let evaluation = self.evaluate(&numerator);
        let mean_actual = actual / total_weight;
        let external_regret = [
            evaluation.best_response_values[0] - mean_actual,
            evaluation.best_response_values[1] + mean_actual,
        ];
        Analysis {
            policy: numerator,
            evaluation,
            external_regret,
        }
    }

    /// Independent full-tree linear CFR oracle. No tabular targets enter SD-CFR.
    pub fn train_tabular(&self, iterations: u64) -> Policy {
        assert!(iterations > 0);
        let mut regrets = self.uniform_policy();
        let mut average = regrets.clone();
        for row in regrets.iter_mut().chain(&mut average) {
            row.fill(0.0);
        }
        for t in 1..=iterations {
            let policy = regrets
                .iter()
                .zip(&self.infosets)
                .map(|(r, i)| {
                    let scale = r.iter().map(|x| x.abs()).fold(1.0_f64, f64::max);
                    let predictions = r.iter().map(|x| (x / scale) as f32).collect::<Vec<_>>();
                    let p = regret_matching(&predictions, &i.observation.legal);
                    let sum = p.iter().map(|&x| x as f64).sum::<f64>();
                    p.into_iter().map(|x| x as f64 / sum).collect()
                })
                .collect::<Policy>();
            for player in 0..2 {
                self.cfr(
                    0,
                    player,
                    [1.0; 2],
                    1.0,
                    &policy,
                    &mut regrets,
                    &mut average,
                    t as f64,
                );
            }
        }
        let uniform = self.uniform_policy();
        for (id, row) in average.iter_mut().enumerate() {
            let sum = row.iter().sum::<f64>();
            if sum > 0.0 {
                for x in row {
                    *x /= sum;
                }
            } else {
                *row = uniform[id].clone();
            }
        }
        average
    }
    #[allow(clippy::too_many_arguments)]
    fn cfr(
        &self,
        id: usize,
        player: usize,
        reach: [f64; 2],
        chance: f64,
        policy: &Policy,
        regrets: &mut Policy,
        average: &mut Policy,
        weight: f64,
    ) -> f64 {
        let node = &self.nodes[id];
        let info = match node.kind {
            Kind::Terminal(payoff) => return if player == 0 { payoff } else { -payoff },
            Kind::Chance => {
                return node
                    .edges
                    .iter()
                    .map(|e| {
                        e.probability
                            * self.cfr(
                                e.child,
                                player,
                                reach,
                                chance * e.probability,
                                policy,
                                regrets,
                                average,
                                weight,
                            )
                    })
                    .sum();
            }
            Kind::Player(info) => info,
        };
        let actor = self.infosets[info].player;
        let mut values = vec![0.0; policy[info].len()];
        for edge in &node.edges {
            let mut next = reach;
            next[actor] *= policy[info][edge.action];
            values[edge.action] = self.cfr(
                edge.child, player, next, chance, policy, regrets, average, weight,
            );
        }
        let expected = policy[info].iter().zip(&values).map(|(p, v)| p * v).sum();
        if actor == player {
            for edge in &node.edges {
                let a = edge.action;
                regrets[info][a] += weight * chance * reach[1 - player] * (values[a] - expected);
                // chance-only normalization is constant across histories in these games.
                average[info][a] += weight * chance * reach[player] * policy[info][a];
            }
        }
        expected
    }
}
