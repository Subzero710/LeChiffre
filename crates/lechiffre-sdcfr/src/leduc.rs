//! Heads-up Leduc, ranks-only observations (suit isomorphism). Six physical
//! cards J1,J2,Q1,Q2,K1,K2; ante 1, bet sizes 2/4, at most two bets per round
//! (an opening bet and one raise). Player 0 starts both rounds; no rake.
use crate::{
    game::{EnumeratedGame, Game, Node, Observation},
    rng::Rng,
};

pub const FOLD: usize = 0;
pub const CALL: usize = 1;
pub const RAISE: usize = 2;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Leduc {
    private: Option<[u8; 2]>,
    board: Option<u8>,
    round: usize,
    actor: usize,
    contributions: [u8; 2],
    street: [u8; 2],
    raises: u8,
    checks: u8,
    history: [Vec<u8>; 2],
    terminal: bool,
    folded: Option<usize>,
}

impl Default for Leduc {
    fn default() -> Self {
        Self {
            private: None,
            board: None,
            round: 0,
            actor: 0,
            contributions: [1; 2],
            street: [0; 2],
            raises: 0,
            checks: 0,
            history: [Vec::new(), Vec::new()],
            terminal: false,
            folded: None,
        }
    }
}

impl Leduc {
    pub const GAME_ID: &'static str = "leduc-v1:ranks-history4x2:ante1-bets2-4-cap2:first0";
    pub const INPUTS: usize = 43;
    pub const ACTIONS: usize = 3;
    pub fn deal(cards: [u8; 2]) -> Self {
        assert!(cards[0] < 6 && cards[1] < 6 && cards[0] != cards[1]);
        Self {
            private: Some(cards),
            ..Self::default()
        }
    }
    pub fn board_card(&self) -> Option<u8> {
        self.board
    }
    pub fn round(&self) -> usize {
        self.round
    }
    pub fn contributions(&self) -> [u8; 2] {
        self.contributions
    }
    pub fn history(&self) -> &[Vec<u8>; 2] {
        &self.history
    }
    pub fn to_call(&self) -> u8 {
        self.street[1 - self.actor] - self.street[self.actor]
    }
    pub fn legal_mask(&self) -> Vec<bool> {
        assert!(matches!(self.node(), Node::Player(_)));
        vec![self.to_call() > 0, true, self.raises < 2]
    }
    pub fn reveal(&self, card: u8) -> Self {
        assert!(
            self.private.is_some() && self.round == 1 && self.board.is_none() && !self.terminal
        );
        assert!(card < 6 && !self.private.unwrap().contains(&card));
        Self {
            board: Some(card),
            ..self.clone()
        }
    }
    fn end_round(&mut self) {
        if self.round == 1 {
            self.terminal = true;
        } else {
            self.round = 1;
            self.actor = 0;
            self.street = [0; 2];
            self.raises = 0;
            self.checks = 0;
        }
    }
}

impl Game for Leduc {
    fn node(&self) -> Node {
        if self.terminal {
            Node::Terminal
        } else if self.private.is_none() || (self.round == 1 && self.board.is_none()) {
            Node::Chance
        } else {
            Node::Player(self.actor)
        }
    }
    fn observation(&self) -> Observation {
        assert!(matches!(self.node(), Node::Player(_)));
        let mut x = vec![0.0; Self::INPUTS];
        x[(self.private.unwrap()[self.actor] / 2) as usize] = 1.0;
        x[3 + self.board.map_or(3, |c| (c / 2) as usize)] = 1.0;
        x[7 + self.actor] = 1.0;
        x[9 + self.round] = 1.0;
        // Eight ordered public action slots, three channels each; no truncation.
        for round in 0..2 {
            assert!(self.history[round].len() <= 4);
            for (slot, &a) in self.history[round].iter().enumerate() {
                x[11 + round * 12 + slot * 3 + a as usize] = 1.0;
            }
        }
        let legal = self.legal_mask();
        for (a, &ok) in legal.iter().enumerate() {
            x[35 + a] = f32::from(ok);
        }
        x[38] = self.contributions.iter().sum::<u8>() as f32 / 26.0;
        x[39] = self.contributions[self.actor] as f32 / 13.0;
        x[40] = self.contributions[1 - self.actor] as f32 / 13.0;
        x[41] = self.to_call() as f32 / 4.0;
        x[42] = self.raises as f32 / 2.0;
        Observation { features: x, legal }
    }
    fn child(&self, action: usize) -> Self {
        assert!(action < 3 && self.legal_mask()[action]);
        let mut next = self.clone();
        next.history[self.round].push(action as u8);
        let player = self.actor;
        match action {
            FOLD => {
                next.terminal = true;
                next.folded = Some(player);
            }
            CALL => {
                let amount = self.to_call();
                next.street[player] += amount;
                next.contributions[player] += amount;
                if amount > 0 || self.checks == 1 {
                    next.end_round();
                } else {
                    next.checks += 1;
                    next.actor = 1 - player;
                }
            }
            RAISE => {
                let amount = self.to_call() + if self.round == 0 { 2 } else { 4 };
                next.street[player] += amount;
                next.contributions[player] += amount;
                next.raises += 1;
                next.checks = 0;
                next.actor = 1 - player;
            }
            _ => unreachable!(),
        }
        next
    }
    fn sample_chance(&self, rng: &mut Rng) -> Self {
        assert_eq!(self.node(), Node::Chance);
        if let Some(private) = self.private {
            let remaining = (0..6).filter(|c| !private.contains(c)).collect::<Vec<_>>();
            self.reveal(remaining[rng.below(4) as usize])
        } else {
            let a = rng.below(6) as u8;
            let mut b = rng.below(5) as u8;
            if b >= a {
                b += 1;
            }
            Self::deal([a, b])
        }
    }
    fn payoff(&self, player: usize) -> f32 {
        assert!(self.terminal && player < 2);
        let winner = if let Some(folded) = self.folded {
            Some(1 - folded)
        } else {
            let ranks = self.private.unwrap().map(|c| c / 2);
            let board = self.board.unwrap() / 2;
            let scores = ranks.map(|rank| if rank == board { 3 + rank } else { rank });
            match scores[0].cmp(&scores[1]) {
                std::cmp::Ordering::Greater => Some(0),
                std::cmp::Ordering::Less => Some(1),
                std::cmp::Ordering::Equal => None,
            }
        };
        match winner {
            Some(w) if w == player => self.contributions[1 - player] as f32,
            Some(_) => -(self.contributions[player] as f32),
            None => 0.0,
        }
    }
}

impl EnumeratedGame for Leduc {
    fn chance_outcomes(&self) -> Vec<(Self, f64)> {
        assert_eq!(self.node(), Node::Chance);
        if let Some(private) = self.private {
            (0..6)
                .filter(|c| !private.contains(c))
                .map(|c| (self.reveal(c), 0.25))
                .collect()
        } else {
            (0..6)
                .flat_map(|a| {
                    (0..6)
                        .filter(move |&b| b != a)
                        .map(move |b| (Self::deal([a, b]), 1.0 / 30.0))
                })
                .collect()
        }
    }
}
