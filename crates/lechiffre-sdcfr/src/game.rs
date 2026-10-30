use crate::rng::Rng;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Node {
    Chance,
    Player(usize),
    Terminal,
}

/// A policy receives only this sanitized information set, never a GameState.
#[derive(Clone, Debug, PartialEq)]
pub struct Observation {
    pub features: Vec<f32>,
    pub legal: Vec<bool>,
}

/// Adapter boundary for future poker engines. States retain the full search state.
/// `child` and `sample_chance` must be side-effect-free on their parent.
pub trait Game: Clone {
    fn node(&self) -> Node;
    fn observation(&self) -> Observation;
    fn child(&self, action: usize) -> Self;
    fn sample_chance(&self, rng: &mut Rng) -> Self;
    fn payoff(&self, player: usize) -> f32;
}

/// Optional exact chance enumeration for small validation games only.
pub trait EnumeratedGame: Game {
    fn chance_outcomes(&self) -> Vec<(Self, f64)>;
}

impl EnumeratedGame for Kuhn {
    fn chance_outcomes(&self) -> Vec<(Self, f64)> {
        assert_eq!(self.node(), Node::Chance);
        (0..3)
            .flat_map(|a| {
                (0..3)
                    .filter(move |&b| b != a)
                    .map(move |b| (Self::deal([a, b]), 1.0 / 6.0))
            })
            .collect()
    }
}

/// Classic two-player Kuhn: J/Q/K, ante 1 each, one possible bet of 1.
/// Public history: "", "c", "b", "cb" at decisions. Action 0 = check/fold;
/// action 1 = bet/call. These are Kuhn action slots, not Hold'em bet amounts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Kuhn {
    cards: Option<[u8; 2]>,
    history: u8,
}

impl Kuhn {
    pub const GAME_ID: &'static str = "kuhn-v1:private3-history4:passive-aggressive";
    pub const INPUTS: usize = 7;
    pub const ACTIONS: usize = 2;
    pub fn deal(cards: [u8; 2]) -> Self {
        assert!(cards[0] < 3 && cards[1] < 3 && cards[0] != cards[1]);
        Self {
            cards: Some(cards),
            history: 0,
        }
    }
    pub fn at(cards: [u8; 2], history: u8) -> Self {
        assert!(history < 4);
        Self {
            history,
            ..Self::deal(cards)
        }
    }
    pub fn history(&self) -> u8 {
        self.history
    }
    pub fn infoset(&self) -> usize {
        let Node::Player(player) = self.node() else {
            panic!("not a decision");
        };
        self.cards.unwrap()[player] as usize * 4 + self.history as usize
    }
    pub fn public_history(&self) -> &'static str {
        ["", "c", "b", "cb", "cc", "bf", "bc", "cbf", "cbc"][self.history as usize]
    }
    pub fn observation_for(card: u8, history: u8) -> Observation {
        assert!(card < 3 && history < 4);
        let mut features = vec![0.0; Self::INPUTS];
        features[card as usize] = 1.0;
        features[3 + history as usize] = 1.0;
        Observation {
            features,
            legal: vec![true, true],
        }
    }
}

impl Game for Kuhn {
    fn node(&self) -> Node {
        if self.cards.is_none() {
            return Node::Chance;
        }
        match self.history {
            0 | 3 => Node::Player(0),
            1 | 2 => Node::Player(1),
            _ => Node::Terminal,
        }
    }
    fn observation(&self) -> Observation {
        let Node::Player(p) = self.node() else {
            panic!("not a decision");
        };
        Self::observation_for(self.cards.unwrap()[p], self.history)
    }
    fn child(&self, action: usize) -> Self {
        assert!(action < 2);
        let next = match (self.history, action) {
            (0, 0) => 1,
            (0, 1) => 2,
            (1, 0) => 4,
            (1, 1) => 3,
            (2, 0) => 5,
            (2, 1) => 6,
            (3, 0) => 7,
            (3, 1) => 8,
            _ => panic!("action at a non-decision node"),
        };
        Self {
            history: next,
            ..*self
        }
    }
    fn sample_chance(&self, rng: &mut Rng) -> Self {
        assert_eq!(self.node(), Node::Chance);
        let first = rng.below(3) as u8;
        let mut second = rng.below(2) as u8;
        if second >= first {
            second += 1;
        }
        Self::deal([first, second])
    }
    fn payoff(&self, player: usize) -> f32 {
        assert!(player < 2 && self.node() == Node::Terminal);
        let cards = self.cards.unwrap();
        let result = match self.history {
            5 => 1.0,  // player 1 folds
            7 => -1.0, // player 0 folds
            4 | 6 | 8 => {
                let stake = if self.history == 4 { 1.0 } else { 2.0 };
                if cards[0] > cards[1] { stake } else { -stake }
            }
            _ => unreachable!(),
        };
        if player == 0 { result } else { -result }
    }
}
