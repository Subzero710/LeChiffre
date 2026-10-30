//! Clonable reduced NLHE search state. rs_poker owns bet validation, money,
//! turn bookkeeping, cards and showdown ranks. This adapter owns chance events,
//! public action history and the intentionally reduced action menu.
use lechiffre_sdcfr::{
    game::{Game, Node, Observation},
    rng::Rng,
};
use rs_poker::{
    arena::{
        GameState, GameStateBuilder,
        action::AgentAction,
        game_state::{Round, RoundData},
    },
    core::{Card, Hand, Rankable},
};

pub const FOLD: usize = 0;
pub const CALL: usize = 1;
pub const MIN_RAISE: usize = 2;
pub const HALF_POT: usize = 3;
pub const POT: usize = 4;
pub const ALL_IN: usize = 5;
pub const ACTIONS: usize = 6;
pub const INPUTS: usize = 312;
pub const BIG_BLIND: i64 = 2;
pub const STACK: i64 = 20;
pub const GAME_ID: &str =
    "nlhe-hu-v1:52:stack20-blinds2-1:button0:cap1:history3x4:inputs312-actions6";
const FULL_DECK: u64 = (1_u64 << 52) - 1;

#[derive(Clone, Debug, PartialEq)]
pub struct PublicAction {
    pub slot: usize,
    pub amount: i64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Nlhe {
    engine: GameState,
    remaining: u64,
    private_dealt: bool,
    runout: bool,
    finished: bool,
    history: [Vec<PublicAction>; 4],
    // Evaluation-only full deal; never enters the observation.
    scripted_board: Option<[u8; 5]>,
}

impl Default for Nlhe {
    fn default() -> Self {
        let engine = GameStateBuilder::new()
            .stacks([STACK; 2])
            .blinds(BIG_BLIND, 1)
            .dealer_idx(0)
            .max_raises_per_round(None)
            .build()
            .unwrap();
        Self {
            engine,
            remaining: FULL_DECK,
            private_dealt: false,
            runout: false,
            finished: false,
            history: std::array::from_fn(|_| Vec::new()),
            scripted_board: None,
        }
    }
}

impl Nlhe {
    pub fn engine(&self) -> &GameState {
        &self.engine
    }
    pub fn history(&self) -> &[Vec<PublicAction>; 4] {
        &self.history
    }
    pub fn remaining_cards(&self) -> u32 {
        self.remaining.count_ones()
    }
    /// Four private cards (two per seat), then flop/turn/river. Deterministic
    /// deals are useful for duplicate matches and tests; no future card is public.
    pub fn from_deal(deal: [u8; 9]) -> Result<Self, String> {
        let mut used = 0_u64;
        for card in deal {
            if card >= 52 || used & (1_u64 << card) != 0 {
                return Err("invalid or duplicate deal card".into());
            }
            used |= 1_u64 << card;
        }
        let mut state = Self::default();
        state.deal_private(deal[..4].try_into().unwrap());
        state.scripted_board = Some(deal[4..].try_into().unwrap());
        Ok(state)
    }
    pub fn shuffled_deal(rng: &mut Rng) -> [u8; 9] {
        let mut mask = FULL_DECK;
        std::array::from_fn(|_| Self::draw(&mut mask, rng))
    }
    fn draw(mask: &mut u64, rng: &mut Rng) -> u8 {
        let mut offset = rng.below(mask.count_ones() as u64);
        let mut rest = *mask;
        while offset > 0 {
            rest &= rest - 1;
            offset -= 1;
        }
        let card = rest.trailing_zeros() as u8;
        *mask &= !(1_u64 << card);
        card
    }
    fn take(&mut self, card: u8) {
        assert!(card < 52 && self.remaining & (1_u64 << card) != 0);
        self.remaining &= !(1_u64 << card);
    }
    fn deal_private(&mut self, cards: [u8; 4]) {
        assert!(!self.private_dealt);
        for card in cards {
            self.take(card);
        }
        for player in 0..2 {
            self.engine.hands[player] = Hand::new_with_cards(vec![
                Card::from(cards[2 * player]),
                Card::from(cards[2 * player + 1]),
            ]);
        }
        self.engine.round = Round::Preflop;
        self.engine.round_data = RoundData::new(2, BIG_BLIND, self.engine.player_active, 0);
        self.engine.do_bet(1, true).unwrap();
        self.engine.do_bet(BIG_BLIND, true).unwrap();
        self.engine.sb_posted = true;
        self.engine.bb_posted = true;
        self.private_dealt = true;
    }
    pub fn street(&self) -> usize {
        match self.engine.round {
            Round::Preflop => 0,
            Round::DealFlop | Round::Flop => 1,
            Round::DealTurn | Round::Turn => 2,
            Round::DealRiver | Round::River => 3,
            _ => self.engine.board.len().saturating_sub(2).min(3),
        }
    }
    pub fn to_call(&self) -> i64 {
        (self.engine.current_round_bet() - self.engine.current_round_current_player_bet())
            .min(self.engine.current_player_stack())
    }
    /// Concrete targets are TOTAL contribution on this street (rs_poker's Bet
    /// convention). Duplicate sizes are masked; all-in keeps its own output.
    pub fn action_targets(&self) -> [Option<i64>; ACTIONS] {
        assert!(matches!(self.node(), Node::Player(_)));
        let gs = &self.engine;
        let mut targets = [None; ACTIONS];
        let current = gs.current_round_bet();
        let maximum = gs.current_round_current_player_bet() + gs.current_player_stack();
        if self.to_call() > 0 {
            targets[FOLD] = Some(0);
        }
        targets[CALL] = Some(current.min(maximum));
        // One voluntary opening bet/raise per street in this reduced game.
        if gs.round_data.total_raise_count >= 1 || maximum <= current {
            return targets;
        }
        let minimum = gs.current_round_min_raise();
        let after_call = gs.total_pot + self.to_call();
        let candidates = [
            current + minimum,
            current + minimum.max((after_call + 1) / 2),
            current + minimum.max(after_call),
        ];
        for (slot, target) in (MIN_RAISE..=POT).zip(candidates) {
            if target < maximum
                && !targets[CALL..slot].contains(&Some(target))
                && gs.validate_bet_amount(target).is_ok()
            {
                targets[slot] = Some(target);
            }
        }
        if gs.validate_bet_amount(maximum).is_ok() {
            targets[ALL_IN] = Some(maximum);
        }
        targets
    }
    pub fn agent_action(&self, slot: usize) -> Result<AgentAction, String> {
        let target = self
            .action_targets()
            .get(slot)
            .copied()
            .flatten()
            .ok_or("illegal reduced action")?;
        Ok(match slot {
            FOLD => AgentAction::Fold,
            CALL => AgentAction::Call,
            ALL_IN => AgentAction::AllIn,
            _ => AgentAction::Bet(target),
        })
    }
    fn finish_fold(&mut self, winner: usize) {
        let pot = self.engine.total_pot;
        self.engine.award(winner, pot);
        self.engine.total_pot = 0;
        self.engine.complete();
        self.finished = true;
    }
    fn showdown(&mut self) {
        assert_eq!(self.engine.board.len(), 5);
        let matched = self.engine.player_bet[0].min(self.engine.player_bet[1]);
        for player in 0..2 {
            let refund = self.engine.player_bet[player] - matched;
            if refund > 0 {
                self.engine.return_uncalled_bet(player, refund);
            }
        }
        let pot = self.engine.total_pot;
        let ranks = self
            .engine
            .hands
            .iter()
            .map(|h| h.rank())
            .collect::<Vec<_>>();
        match ranks[0].cmp(&ranks[1]) {
            std::cmp::Ordering::Greater => self.engine.award(0, pot),
            std::cmp::Ordering::Less => self.engine.award(1, pot),
            std::cmp::Ordering::Equal => {
                assert_eq!(pot % 2, 0);
                self.engine.award(0, pot / 2);
                self.engine.award(1, pot / 2);
            }
        }
        self.engine.total_pot = 0;
        self.engine.complete();
        self.finished = true;
    }
    fn advance_after_action(&mut self) {
        if self.engine.round_data.needs_action.count() != 0 {
            return;
        }
        if self.engine.round == Round::River {
            self.showdown();
        } else {
            self.runout = self.engine.num_all_in_players() > 0;
            self.engine.advance_round(); // explicit chance node before the next street
        }
    }
}

impl Game for Nlhe {
    fn node(&self) -> Node {
        if self.finished {
            Node::Terminal
        } else if !self.private_dealt
            || matches!(
                self.engine.round,
                Round::DealFlop | Round::DealTurn | Round::DealRiver
            )
        {
            Node::Chance
        } else {
            Node::Player(self.engine.to_act_idx())
        }
    }
    fn observation(&self) -> Observation {
        let Node::Player(player) = self.node() else {
            panic!("not a decision");
        };
        let mut x = vec![0.0; INPUTS];
        for card in self.engine.hands[player].iter() {
            if !self.engine.board.contains(&card) {
                x[u8::from(card) as usize] = 1.0;
            }
        }
        for (index, &card) in self.engine.board.iter().enumerate() {
            let channel = if index < 3 {
                1
            } else if index == 3 {
                2
            } else {
                3
            };
            x[channel * 52 + u8::from(card) as usize] = 1.0;
        }
        x[208 + self.street()] = 1.0;
        x[212 + usize::from(player == self.engine.dealer_idx)] = 1.0;
        let gs = &self.engine;
        let numeric = [
            gs.total_pot as f32 / (2 * STACK) as f32,
            gs.stacks[player] as f32 / STACK as f32,
            gs.stacks[1 - player] as f32 / STACK as f32,
            gs.current_round_player_bet(player) as f32 / STACK as f32,
            gs.current_round_player_bet(1 - player) as f32 / STACK as f32,
            self.to_call() as f32 / STACK as f32,
            gs.current_round_min_raise() as f32 / STACK as f32,
            gs.round_data.total_raise_count as f32,
        ];
        x[214..222].copy_from_slice(&numeric);
        let legal = self
            .action_targets()
            .map(|target| target.is_some())
            .to_vec();
        for (a, &ok) in legal.iter().enumerate() {
            x[222 + a] = if ok { 1.0 } else { 0.0 };
        }
        for (street, history) in self.history.iter().enumerate() {
            assert!(history.len() <= 3, "history bound violated");
            for (index, action) in history.iter().enumerate() {
                let offset = 228 + (street * 3 + index) * 7;
                x[offset + action.slot] = 1.0;
                x[offset + 6] = action.amount as f32 / STACK as f32;
            }
        }
        Observation { features: x, legal }
    }
    fn child(&self, action: usize) -> Self {
        let target = self
            .action_targets()
            .get(action)
            .copied()
            .flatten()
            .expect("illegal reduced action");
        let mut next = self.clone();
        let actor = self.engine.to_act_idx();
        let street = self.street();
        next.history[street].push(PublicAction {
            slot: action,
            amount: target,
        });
        assert!(next.history[street].len() <= 3);
        if action == FOLD {
            next.engine.fold();
            next.finish_fold(1 - actor);
        } else {
            next.engine.do_bet(target, false).unwrap();
            next.advance_after_action();
        }
        next
    }
    fn sample_chance(&self, rng: &mut Rng) -> Self {
        assert_eq!(self.node(), Node::Chance);
        let mut next = self.clone();
        if !next.private_dealt {
            let cards = std::array::from_fn(|_| Self::draw(&mut next.remaining, rng));
            // deal_private also removes the cards; restore their bits first.
            for card in cards {
                next.remaining |= 1_u64 << card;
            }
            next.deal_private(cards);
            return next;
        }
        loop {
            let count = if next.engine.round == Round::DealFlop {
                3
            } else {
                1
            };
            for _ in 0..count {
                let index = next.engine.board.len();
                let card = if let Some(board) = next.scripted_board {
                    let c = board[index];
                    next.take(c);
                    c
                } else {
                    Self::draw(&mut next.remaining, rng)
                };
                let card = Card::from(card);
                next.engine.board.push(card);
                for hand in &mut next.engine.hands {
                    assert!(hand.insert(card));
                }
            }
            next.engine.advance_round();
            if !next.runout {
                return next;
            }
            if next.engine.board.len() == 5 {
                next.showdown();
                return next;
            }
            next.engine.advance_round();
        }
    }
    /// Terminal utility in BIG BLINDS, keeping targets comparable to bb/100.
    fn payoff(&self, player: usize) -> f32 {
        assert!(self.finished && player < 2);
        self.engine.player_reward(player) as f32 / BIG_BLIND as f32
    }
}
