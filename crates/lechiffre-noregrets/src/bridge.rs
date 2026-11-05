use crate::reference::{Reference, hand_config};
use lechiffre_nlhe::{
    game::{ACTIONS, CALL, FOLD, Nlhe},
    matchplay::{Report, validate},
};
use lechiffre_sdcfr::{
    game::{Game, Node},
    rng::Rng,
    strategy::sample_snapshot,
    training::Trainer,
};
use noregrets::{
    abstraction::{AbsAction, Abstraction, TOKEN_STREET_SEP},
    engine::{Hand, PlayerAction},
};
use rand::{SeedableRng, rngs::SmallRng};
use serde::Serialize;

/// rs_poker is suit-major; NoRegrets is rank-major. This fixed bijection
/// preserves suits and ranks; no nearest-action or card approximation.
pub fn card_to_noregrets(card: u8) -> u8 {
    assert!(card < 52);
    (card % 13) * 4 + card / 13
}

pub struct Bridge {
    pub reduced: Nlhe,
    pub native: Hand,
    pub history: Vec<u8>,
}

impl Bridge {
    pub fn from_deal(deal: [u8; 9]) -> Result<Self, String> {
        let reduced = Nlhe::from_deal(deal)?;
        let mut deck = [0; 52];
        for (i, c) in deal.into_iter().enumerate() {
            deck[i] = card_to_noregrets(c);
        }
        let rest: Vec<u8> = (0..52).filter(|c| !deck[..9].contains(c)).collect();
        deck[9..].copy_from_slice(&rest);
        Ok(Self {
            reduced,
            native: Hand::new(&hand_config(), 0, deck),
            history: Vec::new(),
        })
    }

    pub fn finish_chance(&mut self) {
        let mut rng = Rng::new(0); // scripted board, independent of action RNGs
        while self.reduced.node() == Node::Chance {
            self.reduced = self.reduced.sample_chance(&mut rng);
        }
    }

    /// Return exact slot/action pairs, refusing every engine or menu drift.
    pub fn actions(&self, abs: &Abstraction) -> Result<Vec<(usize, AbsAction)>, String> {
        let Node::Player(p) = self.reduced.node() else {
            return Err("not a decision node".into());
        };
        let gs = self.reduced.engine();
        if self.native.is_terminal()
            || p != self.native.to_act()
            || gs.total_pot != self.native.pot() as i64
            || gs.current_round_bet() != self.native.current_bet() as i64
            || gs.round_data.total_raise_count != self.native.n_raises()
        {
            return Err("engine state divergence".into());
        }
        for player in 0..2 {
            if gs.stacks[player] != self.native.stack(player) as i64
                || gs.player_bet[player] != self.native.hand_commit(player) as i64
                || gs.round_data.player_bet[player] != self.native.street_commit(player) as i64
            {
                return Err("engine money divergence".into());
            }
        }
        let board: Vec<u8> = gs
            .board
            .iter()
            .map(|c| card_to_noregrets(u8::from(*c)))
            .collect();
        if board != self.native.board() {
            return Err("engine board divergence".into());
        }
        let targets = self.reduced.action_targets();
        let mut pairs = Vec::new();
        for action in abs.abstract_actions(&self.native) {
            let slot = match abs.concrete(&self.native, action) {
                PlayerAction::Fold => FOLD,
                PlayerAction::CheckCall => CALL,
                PlayerAction::RaiseTo(t) => (2..ACTIONS)
                    .find(|&s| targets[s] == Some(t as i64))
                    .ok_or("NoRegrets proposed a bet outside the reduced menu")?,
            };
            if pairs.iter().any(|(s, _)| *s == slot) || targets[slot].is_none() {
                return Err("duplicate or illegal reference action".into());
            }
            pairs.push((slot, action));
        }
        if pairs.len() != targets.iter().flatten().count() {
            return Err("reference and neural action menus differ".into());
        }
        Ok(pairs)
    }

    pub fn apply(&mut self, slot: usize, abs: &Abstraction) -> Result<(), String> {
        let action = self
            .actions(abs)?
            .into_iter()
            .find(|(s, _)| *s == slot)
            .ok_or("illegal action slot")?
            .1;
        let street = self.native.street();
        let concrete = abs.concrete(&self.native, action);
        self.native.apply(concrete);
        self.history.push(action.token());
        if self.native.street() != street {
            self.history.push(TOKEN_STREET_SEP);
        }
        self.reduced = self.reduced.child(slot);
        self.finish_chance();
        if self.reduced.node() == Node::Terminal {
            if !self.native.is_terminal() {
                return Err("terminal state divergence".into());
            }
            let rewards = self.native.utilities();
            for (p, reward) in rewards.iter().enumerate().take(2) {
                if self.reduced.payoff(p) as f64 != *reward as f64 / 2.0 {
                    return Err("showdown/fold payoff divergence".into());
                }
            }
        } else {
            self.actions(abs)?;
        }
        Ok(())
    }

    pub fn reference_policy(
        &self,
        reference: &Reference,
        rng: &mut SmallRng,
    ) -> Result<([f64; ACTIONS], bool), String> {
        let pairs = self.actions(&reference.policy.abs)?;
        let (actions, probabilities) =
            reference
                .policy
                .blueprint_dist(&self.native, &self.history, rng);
        if actions.len() != pairs.len() || probabilities.len() != pairs.len() {
            return Err("reference distribution dimensions differ".into());
        }
        let bucket = reference.policy.abs.bucket(
            self.native.hole(self.native.to_act()),
            self.native.board(),
            rng,
        );
        let known = match reference.policy.blueprint.get(bucket, &self.history) {
            None => false,
            Some(s) if s.len() != pairs.len() => {
                return Err("blueprint action-vector mismatch".into());
            }
            Some(s) => s.iter().sum::<f32>() > 0.0,
        };
        let mut out = [0.0; ACTIONS];
        for ((slot, expected), (action, probability)) in pairs
            .into_iter()
            .zip(actions.into_iter().zip(probabilities))
        {
            if expected != action || !probability.is_finite() || probability < 0.0 {
                return Err("invalid reference probabilities".into());
            }
            out[slot] = probability;
        }
        if (out.iter().sum::<f64>() - 1.0).abs() > 1e-6 {
            return Err("reference probabilities do not sum to one".into());
        }
        Ok((out, known))
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Coverage {
    pub decisions: [u64; 4],
    pub fallbacks: [u64; 4],
}

pub fn duel(
    hero: &Trainer,
    reference: &Reference,
    hands: u64,
    seed: u64,
) -> Result<(Report, Coverage), String> {
    validate(hero)?;
    reference.validate()?;
    if hands < 4 || !hands.is_multiple_of(2) {
        return Err("hands must be even and at least 4".into());
    }
    if (0..2).any(|p| hero.archive(p).is_empty()) {
        return Err("empty neural strategy".into());
    }
    let mut master = Rng::new(seed);
    let mut totals = [0.0; 2];
    let (mut mean, mut m2) = (0.0, 0.0);
    let mut coverage = Coverage {
        decisions: [0; 4],
        fallbacks: [0; 4],
    };
    for pair in 0..hands / 2 {
        let deal = Nlhe::shuffled_deal(&mut master);
        let seeds: [u64; 5] = std::array::from_fn(|_| master.next_u64());
        let mut pair_gain = 0.0;
        for hero_seat in 0..2 {
            let mut game = Bridge::from_deal(deal)?;
            let mut model_rng = Rng::new(seeds[hero_seat]);
            let model = sample_snapshot(hero.archive(hero_seat), &mut model_rng);
            let mut action_rng = [Rng::new(seeds[2]), Rng::new(seeds[3])];
            let mut bucket_rng = SmallRng::seed_from_u64(seeds[4]);
            game.finish_chance();
            loop {
                match game.reduced.node() {
                    Node::Terminal => break,
                    Node::Chance => unreachable!(),
                    Node::Player(p) => {
                        let slot = if p == hero_seat {
                            action_rng[p].action(&model.policy(&game.reduced.observation()))
                        } else {
                            let (dist, known) =
                                game.reference_policy(reference, &mut bucket_rng)?;
                            let street = game.native.street() as usize;
                            coverage.decisions[street] += 1;
                            coverage.fallbacks[street] += u64::from(!known);
                            let x = action_rng[p].unit();
                            let mut cumulative = 0.0;
                            let mut selected = CALL;
                            for (slot, probability) in dist.into_iter().enumerate() {
                                cumulative += probability;
                                if x < cumulative {
                                    selected = slot;
                                    break;
                                }
                            }
                            selected
                        };
                        game.apply(slot, &reference.policy.abs)?;
                    }
                }
            }
            let gain = game.reduced.payoff(hero_seat) as f64;
            totals[hero_seat] += gain;
            pair_gain += gain / 2.0;
        }
        let delta = pair_gain - mean;
        mean += delta / (pair + 1) as f64;
        m2 += delta * (pair_gain - mean);
    }
    let pairs = hands / 2;
    let width =
        1.959963984540054 * (m2.max(0.0) / (pairs - 1) as f64 / pairs as f64).sqrt() * 100.0;
    Ok((
        Report {
            hands,
            pairs,
            total_bb: totals.iter().sum(),
            bb_per_100: mean * 100.0,
            ci95: [mean * 100.0 - width, mean * 100.0 + width],
            by_seat: totals.map(|x| x / pairs as f64),
        },
        coverage,
    ))
}
