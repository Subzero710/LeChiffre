//! Compatibility API for NoRegrets, backed exclusively by rs_poker.
//! Betting rules, action order, dealing, side pots and payouts belong to
//! HoldemSimulation. This module only translates cards/actions and samples
//! hidden information for the original NoRegrets traversals and resolvers.
use crate::cards::Card;
use futures_executor::block_on;
use rand::{rngs::SmallRng, Rng};
use rs_poker::{
    arena::{
        action::AgentAction, game_state::Round, GameState, GameStateBuilder, HoldemSimulation,
        HoldemSimulationBuilder,
    },
    core::{Card as RsCard, Hand as RsHand},
};
pub const MAX_PLAYERS: usize = 6;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Street {
    Preflop = 0,
    Flop = 1,
    Turn = 2,
    River = 3,
}
impl Street {
    pub fn next(self) -> Self {
        match self {
            Self::Preflop => Self::Flop,
            Self::Flop => Self::Turn,
            Self::Turn | Self::River => Self::River,
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlayerAction {
    Fold,
    CheckCall,
    RaiseTo(u32),
}
#[derive(Debug, Clone)]
pub struct HandConfig {
    pub num_players: usize,
    pub stack: u32,
    pub sb: u32,
    pub bb: u32,
}
impl Default for HandConfig {
    fn default() -> Self {
        Self {
            num_players: 6,
            stack: 10_000,
            sb: 50,
            bb: 100,
        }
    }
}

/// NoRegrets: rank*4+suit (c,d,h,s). rs_poker: suit*13+rank (s,c,h,d).
pub fn to_rs_card(c: Card) -> RsCard {
    assert!(c < 52);
    RsCard::from([1, 3, 2, 0][(c % 4) as usize] * 13 + c / 4)
}
pub fn from_rs_card(c: RsCard) -> Card {
    let c = u8::from(c);
    (c % 13) * 4 + [3, 0, 2, 1][(c / 13) as usize]
}

#[derive(Debug, Clone)]
pub struct Hand {
    state: GameState,
    // Card views in NoRegrets' encoding; no separate betting state.
    holes: [[Card; 2]; MAX_PLAYERS],
    board: [Card; 5],
    runout: [Card; 5],
}
impl Hand {
    pub fn new(cfg: &HandConfig, button: usize, deck: [Card; 52]) -> Self {
        Self::new_with_stacks(cfg, button, deck, &vec![cfg.stack; cfg.num_players])
    }
    pub fn new_with_stacks(
        cfg: &HandConfig,
        button: usize,
        deck: [Card; 52],
        stacks: &[u32],
    ) -> Self {
        assert!((2..=MAX_PLAYERS).contains(&cfg.num_players));
        assert_eq!(stacks.len(), cfg.num_players);
        assert!(button < cfg.num_players);
        let mut holes = [[0; 2]; MAX_PLAYERS];
        let hands: Vec<_> = (0..cfg.num_players)
            .map(|p| {
                holes[p] = [deck[2 * p], deck[2 * p + 1]];
                RsHand::new_with_cards(holes[p].map(to_rs_card).to_vec())
            })
            .collect();
        // Holes come from the original NoRegrets chance sampler.
        // rs_poker posts the blinds and handles short stacks and action order.
        let mut state = GameStateBuilder::new()
            .stacks(stacks.iter().map(|&x| i64::from(x)).collect::<Vec<_>>())
            .blinds(i64::from(cfg.bb), i64::from(cfg.sb))
            .dealer_idx(button)
            .hands(hands)
            .round(Round::Preflop)
            .max_raises_per_round(None)
            .build()
            .expect("valid NoRegrets hand config");
        state.sb_posted = false;
        state.bb_posted = false;
        let mut hand = Self {
            state,
            holes,
            board: [0; 5],
            runout: deck[2 * cfg.num_players..2 * cfg.num_players + 5]
                .try_into()
                .unwrap(),
        };
        let mut sim = hand.simulation();
        block_on(sim.advance_to_decision());
        hand.update(sim.game_state);
        hand
    }
    fn simulation(&self) -> HoldemSimulation {
        let mut sim = HoldemSimulationBuilder::default()
            .game_state(self.state.clone())
            .build_with_rng(rs_poker::arena::seeded_rng(Some(0)))
            .expect("valid native snapshot");
        sim.set_deal_order(
            self.runout[self.board().len()..]
                .iter()
                .copied()
                .map(to_rs_card),
        )
        .expect("valid sampled runout");
        sim
    }
    fn update(&mut self, state: GameState) {
        for (i, c) in state.board.iter().enumerate() {
            self.board[i] = from_rs_card(*c);
        }
        self.state = state;
    }
    pub fn apply(&mut self, action: PlayerAction) {
        assert!(!self.is_terminal(), "action on terminal hand");
        let action = match action {
            PlayerAction::Fold => AgentAction::Fold,
            PlayerAction::CheckCall => AgentAction::Call,
            PlayerAction::RaiseTo(t) => match self.raise_bounds() {
                Some((lo, hi)) => AgentAction::Bet(i64::from(t.clamp(lo, hi))),
                None => AgentAction::Call,
            },
        };
        let mut sim = self.simulation();
        block_on(async {
            sim.run_agent_action(action).await;
            sim.advance_to_decision().await;
        });
        self.update(sim.game_state);
    }
    pub fn utilities(&self) -> [i64; MAX_PLAYERS] {
        assert!(self.is_terminal());
        let mut result = [0; MAX_PLAYERS];
        for (p, r) in result.iter_mut().enumerate().take(self.num_players()) {
            *r = self.state.player_reward(p);
        }
        result
    }
    pub fn resample_hidden(&mut self, keep: Option<usize>, rng: &mut SmallRng) {
        let mut want = [None; MAX_PLAYERS];
        if let Some(p) = keep {
            want[p] = Some(self.hole(p));
        }
        self.resample_hidden_with(&want, rng);
    }
    pub fn resample_hidden_with(
        &mut self,
        want: &[Option<[Card; 2]>; MAX_PLAYERS],
        rng: &mut SmallRng,
    ) {
        let mut used = [false; 52];
        for &c in self.board() {
            used[c as usize] = true;
        }
        for cards in want.iter().take(self.num_players()).flatten() {
            for &c in cards {
                assert!(!used[c as usize], "requested card already used");
                used[c as usize] = true;
            }
        }
        let mut unseen: Vec<Card> = (0..52).filter(|&c| !used[c as usize]).collect();
        let need = 2 * want[..self.num_players()]
            .iter()
            .filter(|w| w.is_none())
            .count()
            + 5
            - self.board().len();
        for k in 0..need {
            let j = rng.random_range(k..unseen.len());
            unseen.swap(k, j);
        }
        let mut next = 0;
        for (p, w) in want.iter().enumerate().take(self.num_players()) {
            self.holes[p] = w.unwrap_or_else(|| {
                next += 2;
                [unseen[next - 2], unseen[next - 1]]
            });
            self.state.hands[p] = RsHand::new_with_cards(
                self.holes[p]
                    .iter()
                    .copied()
                    .map(to_rs_card)
                    .chain(self.state.board.iter().copied())
                    .collect(),
            );
        }
        for i in self.board().len()..5 {
            self.runout[i] = unseen[next];
            next += 1;
        }
    }
    /// Synchronize an externally observed board, queuing cards not dealt yet.
    /// Preserve hero's cards and replace conflicting opponent placeholders.
    /// rs_poker still controls when cards become public and awards the pots.
    pub fn set_public_board(&mut self, cards: &[Card], hero: usize) -> Result<(), String> {
        let dealt = self.board().len();
        if hero >= self.num_players()
            || !matches!(cards.len(), 0 | 3 | 4 | 5)
            || cards.len() < dealt
        {
            return Err("invalid observed board or hero seat".into());
        }
        let mut used = [false; 52];
        for &c in cards.iter().chain(self.holes[hero].iter()) {
            if c >= 52 || used[c as usize] {
                return Err("observed board conflicts with known cards".into());
            }
            used[c as usize] = true;
        }
        let mut holes = self.holes;
        for (p, hole) in holes.iter_mut().enumerate().take(self.num_players()) {
            if p == hero {
                continue;
            }
            for c in hole {
                if used[*c as usize] {
                    *c = (0..52).find(|&v| !used[v as usize]).unwrap();
                }
                used[*c as usize] = true;
            }
        }
        let mut runout = self.runout;
        runout[..cards.len()].copy_from_slice(cards);
        for c in &mut runout[cards.len()..] {
            if used[*c as usize] {
                *c = (0..52).find(|&v| !used[v as usize]).unwrap();
            }
            used[*c as usize] = true;
        }
        self.holes = holes;
        self.runout = runout;
        self.board[..dealt].copy_from_slice(&cards[..dealt]);
        self.state.board = cards[..dealt].iter().copied().map(to_rs_card).collect();
        for (p, hole) in holes.iter().enumerate().take(self.num_players()) {
            self.state.hands[p] = RsHand::new_with_cards(
                hole.iter()
                    .copied()
                    .map(to_rs_card)
                    .chain(self.state.board.iter().copied())
                    .collect(),
            );
        }
        Ok(())
    }
    /// Upstream range-vector chance override: not for engine showdowns.
    pub fn force_board_card(&mut self, idx: usize, c: Card) {
        assert!(idx < self.board().len());
        self.board[idx] = c;
        self.state.board[idx] = to_rs_card(c);
    }
    pub fn num_players(&self) -> usize {
        self.state.num_players
    }
    pub fn button(&self) -> usize {
        self.state.dealer_idx
    }
    pub fn to_act(&self) -> usize {
        self.state.to_act_idx()
    }
    pub fn street(&self) -> Street {
        match self.state.board.len() {
            0 => Street::Preflop,
            3 => Street::Flop,
            4 => Street::Turn,
            _ => Street::River,
        }
    }
    pub fn board(&self) -> &[Card] {
        &self.board[..self.state.board.len()]
    }
    pub fn hole(&self, p: usize) -> [Card; 2] {
        self.holes[p]
    }
    pub fn is_terminal(&self) -> bool {
        self.state.round == Round::Complete
    }
    pub fn current_bet(&self) -> u32 {
        self.state.current_round_bet().try_into().unwrap()
    }
    pub fn street_commit(&self, p: usize) -> u32 {
        self.state.current_round_player_bet(p).try_into().unwrap()
    }
    pub fn hand_commit(&self, p: usize) -> u32 {
        self.state.player_bet[p].try_into().unwrap()
    }
    pub fn stack(&self, p: usize) -> u32 {
        self.state.stacks[p].try_into().unwrap()
    }
    pub fn folded(&self, p: usize) -> bool {
        !self.state.player_active.get(p) && !self.state.player_all_in.get(p)
    }
    pub fn all_in(&self, p: usize) -> bool {
        self.state.player_all_in.get(p)
    }
    pub fn n_raises(&self) -> u8 {
        self.state.round_data.total_raise_count
    }
    pub fn big_blind(&self) -> u32 {
        self.state.big_blind.try_into().unwrap()
    }
    pub fn pot(&self) -> u32 {
        self.state.total_pot.try_into().unwrap()
    }
    pub fn to_call(&self) -> u32 {
        self.current_bet()
            .saturating_sub(self.street_commit(self.to_act()))
    }
    pub fn can_check(&self) -> bool {
        self.to_call() == 0
    }
    pub fn live_count(&self) -> usize {
        (self.state.player_active | self.state.player_all_in).count()
    }
    pub fn raise_bounds(&self) -> Option<(u32, u32)> {
        if self.is_terminal() {
            return None;
        }
        let max = self.state.current_round_current_player_bet() + self.state.current_player_stack();
        if max <= self.state.current_round_bet() || self.state.validate_bet_amount(max).is_err() {
            return None;
        }
        let min = (self.state.current_round_bet() + self.state.current_round_min_raise()).min(max);
        Some((min.try_into().unwrap(), max.try_into().unwrap()))
    }
}
