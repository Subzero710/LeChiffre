use crate::game::{ACTIONS, ALL_IN, CALL, GAME_ID, INPUTS, MIN_RAISE, Nlhe};
use lechiffre_sdcfr::{
    game::{Game, Node},
    rng::Rng,
    strategy::sample_snapshot,
    training::Trainer,
};

#[derive(Clone, Copy)]
pub enum Opponent<'a> {
    Random,
    Passive,
    Aggressive,
    Neural(&'a Trainer),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Report {
    pub hands: u64,
    pub pairs: u64,
    pub total_bb: f64,
    pub bb_per_100: f64,
    pub ci95: [f64; 2],
    pub by_seat: [f64; 2],
}

fn sample_policy(probabilities: &[f32], rng: &mut Rng) -> usize {
    rng.action(probabilities)
}

fn hand(
    hero: &Trainer,
    opponent: Opponent<'_>,
    hero_seat: usize,
    deal: [u8; 9],
    seeds: [u64; 4],
) -> f64 {
    let mut model_rng = [Rng::new(seeds[0]), Rng::new(seeds[1])];
    let mut action_rng = [Rng::new(seeds[2]), Rng::new(seeds[3])];
    // Exactly one archived network per agent and hand, held for the WHOLE hand.
    let hero_model = sample_snapshot(hero.archive(hero_seat), &mut model_rng[hero_seat]);
    let other_model = if let Opponent::Neural(trainer) = opponent {
        Some(sample_snapshot(
            trainer.archive(1 - hero_seat),
            &mut model_rng[1 - hero_seat],
        ))
    } else {
        None
    };
    let mut state = Nlhe::from_deal(deal).unwrap();
    let mut unused_chance = Rng::new(0); // scripted board; never sampled from this RNG
    loop {
        match state.node() {
            Node::Terminal => return state.payoff(hero_seat) as f64,
            Node::Chance => state = state.sample_chance(&mut unused_chance),
            Node::Player(player) => {
                let observation = state.observation();
                let action = if player == hero_seat {
                    sample_policy(&hero_model.policy(&observation), &mut action_rng[player])
                } else if let Some(model) = other_model {
                    sample_policy(&model.policy(&observation), &mut action_rng[player])
                } else {
                    match opponent {
                        Opponent::Passive => CALL,
                        Opponent::Aggressive => {
                            if observation.legal[ALL_IN] {
                                ALL_IN
                            } else if observation.legal[MIN_RAISE] {
                                MIN_RAISE
                            } else {
                                CALL
                            }
                        }
                        Opponent::Random => {
                            let legal = observation
                                .legal
                                .iter()
                                .enumerate()
                                .filter(|(_, ok)| **ok)
                                .map(|(a, _)| a)
                                .collect::<Vec<_>>();
                            legal[action_rng[player].below(legal.len() as u64) as usize]
                        }
                        Opponent::Neural(_) => unreachable!(),
                    }
                };
                state = state.child(action);
            }
        }
    }
}

pub fn duel(
    hero: &Trainer,
    opponent: Opponent<'_>,
    hands: u64,
    seed: u64,
) -> Result<Report, String> {
    validate(hero)?;
    if hero.archive(0).is_empty() || hero.archive(1).is_empty() {
        return Err("empty strategy".into());
    }
    if hands < 4 || (hands & 1) != 0 {
        return Err("hands must be even and at least 4".into());
    }
    // Checkpoint IDs bind rules, action menu and encoder; the loader validates it.
    if let Opponent::Neural(other) = opponent {
        validate(other)?;
        if other.archive(0).is_empty() || other.archive(1).is_empty() {
            return Err("empty opponent strategy".into());
        }
    }
    let mut master = Rng::new(seed);
    let mut mean = 0.0;
    let mut m2 = 0.0;
    let mut totals = [0.0; 2];
    for index in 0..hands / 2 {
        let deal = Nlhe::shuffled_deal(&mut master);
        let seeds = std::array::from_fn(|_| master.next_u64());
        let gains = [
            hand(hero, opponent, 0, deal, seeds),
            hand(hero, opponent, 1, deal, seeds),
        ];
        for seat in 0..2 {
            totals[seat] += gains[seat];
        }
        let pair = (gains[0] + gains[1]) / 2.0;
        let delta = pair - mean;
        mean += delta / (index + 1) as f64;
        m2 += delta * (pair - mean);
    }
    let pairs = hands / 2;
    let width =
        1.959963984540054 * (m2.max(0.0) / (pairs - 1) as f64 / pairs as f64).sqrt() * 100.0;
    Ok(Report {
        hands,
        pairs,
        total_bb: totals.iter().sum(),
        bb_per_100: mean * 100.0,
        ci95: [mean * 100.0 - width, mean * 100.0 + width],
        by_seat: totals.map(|t| t / pairs as f64),
    })
}

pub fn validate(trainer: &Trainer) -> Result<(), String> {
    let shape = trainer.network_shape();
    if trainer.game_id() != GAME_ID || shape[0] != INPUTS || shape.last() != Some(&ACTIONS) {
        return Err("checkpoint belongs to a different game or encoder".into());
    }
    Ok(())
}
