use crate::{
    game::{Game, Node},
    memory::{Reservoir, Sample, compact_if_small},
    neural::{Adam, LossItem, Mlp},
    rng::Rng,
    strategy::{Snapshot, regret_matching},
};
use std::collections::BTreeMap;

type PolicyCache = BTreeMap<(usize, Vec<u32>, Vec<bool>), Vec<f32>>;

#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    pub traversals: usize,
    pub train_steps: usize,
    pub batch_size: usize,
    pub reservoir_capacity: usize,
    pub hidden: usize,
    pub learning_rate: f32,
    pub seed: u64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            traversals: 512,
            train_steps: 256,
            batch_size: 64,
            reservoir_capacity: 50_000,
            hidden: 32,
            learning_rate: 0.005,
            seed: 42,
        }
    }
}

impl Config {
    pub fn validate(&self) -> Result<(), String> {
        if self.traversals == 0
            || self.train_steps == 0
            || self.batch_size == 0
            || self.reservoir_capacity == 0
            || self.hidden == 0
        {
            return Err("traversals, steps, batch, capacity and hidden must be positive".into());
        }
        if !self.learning_rate.is_finite() || self.learning_rate <= 0.0 {
            return Err("learning rate must be finite and positive".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Trainer {
    pub(crate) game_id: String,
    pub(crate) config: Config,
    pub(crate) completed: u64,
    pub(crate) models: [Mlp; 2],
    pub(crate) memory: [Reservoir; 2],
    pub(crate) archive: [Vec<Snapshot>; 2],
    pub(crate) rng: Rng,
}

#[derive(Clone, Debug)]
pub struct TrainingReport {
    pub iteration: u64,
    pub loss: [f64; 2],
    pub samples_seen: [u64; 2],
    pub samples_stored: [usize; 2],
}

impl Trainer {
    pub fn new(
        game_id: &str,
        inputs: usize,
        actions: usize,
        config: Config,
    ) -> Result<Self, String> {
        config.validate()?;
        if inputs == 0 || actions == 0 {
            return Err("empty network dimensions".into());
        }
        let sizes = [inputs, config.hidden, config.hidden, actions];
        Ok(Self {
            game_id: game_id.into(),
            rng: Rng::new(config.seed),
            models: [Mlp::zeros(&sizes), Mlp::zeros(&sizes)],
            memory: [
                Reservoir::new(config.reservoir_capacity),
                Reservoir::new(config.reservoir_capacity),
            ],
            archive: [Vec::new(), Vec::new()],
            config,
            completed: 0,
        })
    }
    pub fn completed(&self) -> u64 {
        self.completed
    }
    pub fn archive(&self, player: usize) -> &[Snapshot] {
        &self.archive[player]
    }
    pub fn config(&self) -> &Config {
        &self.config
    }
    pub fn network_shape(&self) -> &[usize] {
        self.models[0].sizes()
    }
    /// Simultaneous updates: collect both players' targets against one frozen
    /// joint policy, archive precisely that policy, then fit both successors.
    pub fn step<G: Game>(&mut self, root: &G) -> Result<TrainingReport, String> {
        let iteration = self.completed.checked_add(1).ok_or("iteration overflow")?;
        let mut losses = [0.0; 2];
        // Cache only sanitized infosets, shared while BOTH models are frozen.
        let mut cache = PolicyCache::new();
        for player in 0..2 {
            self.archive[player].push(Snapshot {
                iteration,
                model: self.models[player].clone(),
            });
            for _ in 0..self.config.traversals {
                traverse_cached(
                    root,
                    player,
                    &self.models,
                    iteration,
                    &mut self.memory[player],
                    &mut self.rng,
                    &mut cache,
                );
            }
        }
        for (player, loss) in losses.iter_mut().enumerate() {
            let samples = self.memory[player].samples();
            if samples.is_empty() {
                return Err("no traverser samples generated".into());
            }
            // Retrain from scratch on the entire historical reservoir distribution.
            let mut model = Mlp::initialized(self.models[player].sizes(), &mut self.rng);
            let mut optimizer = Adam::new(&model);
            let mean_t =
                samples.iter().map(|s| s.iteration as f64).sum::<f64>() / samples.len() as f64;
            // Up to 512 distinct inputs: exact grouping removes mini-batch noise
            // without changing the regression objective. Larger games use
            // uniform mini-batches instead of building an unbounded table.
            let compact = compact_if_small(samples, 512);
            for _ in 0..self.config.train_steps {
                let (_, gradient) = if let Some(points) = &compact {
                    let batch = points
                        .iter()
                        .map(|s| LossItem {
                            observation: &s.observation,
                            targets: &s.targets,
                            weight: s.weight,
                        })
                        .collect::<Vec<_>>();
                    model.weighted_loss_and_gradient(&batch)
                } else {
                    let batch = (0..self.config.batch_size)
                        .map(|_| &samples[self.rng.below(samples.len() as u64) as usize])
                        .collect::<Vec<_>>();
                    model.loss_and_gradient(&batch, mean_t)
                };
                optimizer.update(&mut model, &gradient, self.config.learning_rate, 1.0)?;
            }
            // A fixed, uniform subset reports loss without consuming RNG state.
            let stride = samples.len().div_ceil(1024);
            let diagnostics = samples.iter().step_by(stride).collect::<Vec<_>>();
            *loss = model.loss_and_gradient(&diagnostics, mean_t).0;
            self.models[player] = model;
        }
        self.completed = iteration;
        Ok(TrainingReport {
            iteration,
            loss: losses,
            samples_seen: [self.memory[0].seen(), self.memory[1].seen()],
            samples_stored: [
                self.memory[0].samples().len(),
                self.memory[1].samples().len(),
            ],
        })
    }
}

/// External-sampling MCCFR: sample chance and opponents, enumerate ALL legal
/// traverser actions to terminal payoffs. Visitation already samples opponent
/// reach, so no additional reach multiplier is applied to stored targets.
pub fn traverse<G: Game>(
    state: &G,
    traverser: usize,
    models: &[Mlp; 2],
    iteration: u64,
    memory: &mut Reservoir,
    rng: &mut Rng,
) -> f32 {
    traverse_cached(
        state,
        traverser,
        models,
        iteration,
        memory,
        rng,
        &mut PolicyCache::new(),
    )
}

fn traverse_cached<G: Game>(
    state: &G,
    traverser: usize,
    models: &[Mlp; 2],
    iteration: u64,
    memory: &mut Reservoir,
    rng: &mut Rng,
    cache: &mut PolicyCache,
) -> f32 {
    match state.node() {
        Node::Terminal => state.payoff(traverser),
        Node::Chance => traverse_cached(
            &state.sample_chance(rng),
            traverser,
            models,
            iteration,
            memory,
            rng,
            cache,
        ),
        Node::Player(player) => {
            assert!(player < 2);
            let observation = state.observation();
            let key = (
                player,
                observation.features.iter().map(|x| x.to_bits()).collect(),
                observation.legal.clone(),
            );
            let policy = cache
                .entry(key)
                .or_insert_with(|| {
                    regret_matching(
                        &models[player].predict(&observation.features),
                        &observation.legal,
                    )
                })
                .clone();
            if player != traverser {
                let action = rng.action(&policy);
                return traverse_cached(
                    &state.child(action),
                    traverser,
                    models,
                    iteration,
                    memory,
                    rng,
                    cache,
                );
            }
            let mut values = vec![0.0; policy.len()];
            for (action, &legal) in observation.legal.iter().enumerate() {
                if legal {
                    values[action] = traverse_cached(
                        &state.child(action),
                        traverser,
                        models,
                        iteration,
                        memory,
                        rng,
                        cache,
                    );
                }
            }
            let expected: f32 = policy.iter().zip(&values).map(|(p, v)| p * v).sum();
            let targets = values
                .iter()
                .zip(&observation.legal)
                .map(|(v, legal)| if *legal { v - expected } else { 0.0 })
                .collect();
            memory.push(
                Sample {
                    observation,
                    targets,
                    iteration,
                },
                rng,
            );
            expected
        }
    }
}
