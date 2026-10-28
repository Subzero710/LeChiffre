use async_trait::async_trait;
use tracing::{instrument, trace};

use crate::{
    arena::{GameState, action::AgentAction, money::apply_ratio},
    core::Hand,
    holdem::MonteCarloGame,
};

use super::Agent;

/// Default Monte Carlo sample count used by [`EquityAgent`].
pub const DEFAULT_EQUITY_ITERATIONS: usize = 1_000;

/// Default normalized equity-edge thresholds.
pub const DEFAULT_EQUITY_SMALL_EDGE: f32 = 0.15;
pub const DEFAULT_EQUITY_MEDIUM_EDGE: f32 = 0.35;
pub const DEFAULT_EQUITY_LARGE_EDGE: f32 = 0.60;

/// Default pot-fraction raise sizes selected at each equity edge tier.
pub const DEFAULT_EQUITY_SMALL_BET_POT: f32 = 0.33;
pub const DEFAULT_EQUITY_MEDIUM_BET_POT: f32 = 0.66;
pub const DEFAULT_EQUITY_LARGE_BET_POT: f32 = 1.00;

/// An equity-driven baseline agent.
///
/// The agent hides every live opponent's hole cards, estimates showdown equity
/// against uniformly random unknown hands with [`MonteCarloGame`], then compares
/// that equity to a context-dependent baseline:
///
/// - facing a bet: immediate pot odds;
/// - with a free check: equal-share equity among live contenders.
///
/// Equity above that baseline is normalized to an `equity_edge` in `[0, 1]`.
/// Small edges call/check, while larger edges deterministically select small,
/// medium, or large pot-fraction raises. This intentionally does not model fold
/// equity, opponent ranges, future action, or bluffing: bet size is driven only
/// by estimated showdown equity.
#[derive(Debug, Clone)]
pub struct EquityAgent {
    name: String,
    iterations: usize,
    small_edge: f32,
    medium_edge: f32,
    large_edge: f32,
    small_bet_pot: f32,
    medium_bet_pot: f32,
    large_bet_pot: f32,
}

impl EquityAgent {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        name: impl Into<String>,
        iterations: usize,
        small_edge: f32,
        medium_edge: f32,
        large_edge: f32,
        small_bet_pot: f32,
        medium_bet_pot: f32,
        large_bet_pot: f32,
    ) -> Self {
        assert!(iterations > 0, "EquityAgent iterations must be > 0");
        assert!(
            small_edge.is_finite()
                && medium_edge.is_finite()
                && large_edge.is_finite()
                && 0.0 <= small_edge
                && small_edge < medium_edge
                && medium_edge < large_edge
                && large_edge <= 1.0,
            "EquityAgent edge thresholds must satisfy 0 <= small < medium < large <= 1"
        );
        assert!(
            small_bet_pot.is_finite()
                && medium_bet_pot.is_finite()
                && large_bet_pot.is_finite()
                && small_bet_pot > 0.0
                && small_bet_pot <= medium_bet_pot
                && medium_bet_pot <= large_bet_pot,
            "EquityAgent bet fractions must satisfy 0 < small <= medium <= large"
        );

        Self {
            name: name.into(),
            iterations,
            small_edge,
            medium_edge,
            large_edge,
            small_bet_pot,
            medium_bet_pot,
            large_bet_pot,
        }
    }

    pub fn iterations(&self) -> usize {
        self.iterations
    }

    /// Build the Monte Carlo participants from the acting player's perspective.
    /// Folded seats are excluded entirely; live opponents keep only the public
    /// board, so `MonteCarloGame` samples their missing hole cards uniformly.
    /// Returns `(hands, hero_simulation_index)`.
    fn simulation_hands(&self, game_state: &GameState) -> (Vec<Hand>, usize) {
        let mut unknown_hand = Hand::new();
        unknown_hand.extend(game_state.board.iter().cloned());

        let hero_seat = game_state.to_act_idx();
        let mut hands = Vec::with_capacity(game_state.num_players);
        let mut hero_sim_idx = 0;

        for (seat, hand) in game_state.hands.iter().enumerate() {
            if seat == hero_seat {
                hero_sim_idx = hands.len();
                hands.push(hand.clone());
            } else if game_state.player_active.get(seat) || game_state.player_all_in.get(seat) {
                hands.push(unknown_hand.clone());
            }
        }

        (hands, hero_sim_idx)
    }

    fn call_cost(&self, game_state: &GameState) -> crate::arena::Chips {
        let amount_to_match = game_state
            .current_round_bet()
            .saturating_sub(game_state.current_round_current_player_bet());
        amount_to_match.min(game_state.current_player_stack())
    }

    /// Equity required before an aggressive edge exists.
    ///
    /// Facing a bet this is the immediate pot-odds threshold. With a free check
    /// it is the equal-share equity of one player among all live contenders.
    fn required_equity(&self, game_state: &GameState) -> f32 {
        let call_cost = self.call_cost(game_state);
        if call_cost > 0 {
            let final_pot_if_called = game_state.total_pot.saturating_add(call_cost);
            if final_pot_if_called > 0 {
                return call_cost as f32 / final_pot_if_called as f32;
            }
            return 1.0;
        }

        let contenders = game_state.num_active_players() + game_state.num_all_in_players();
        1.0 / contenders.max(1) as f32
    }

    fn normalized_edge(equity: f32, required_equity: f32) -> f32 {
        if equity <= required_equity {
            return 0.0;
        }

        let remaining = 1.0 - required_equity;
        if remaining <= f32::EPSILON {
            0.0
        } else {
            ((equity - required_equity) / remaining).clamp(0.0, 1.0)
        }
    }

    fn passive_action(&self, game_state: &GameState) -> AgentAction {
        if self.call_cost(game_state) > 0 {
            AgentAction::Call
        } else {
            AgentAction::Bet(game_state.current_round_bet())
        }
    }

    /// Turn a pot fraction into a legal raise-to amount. If betting is capped,
    /// not reopened, or otherwise invalid, fall back to call/check rather than
    /// emitting an illegal action.
    fn raise_for_fraction(&self, game_state: &GameState, pot_fraction: f32) -> AgentAction {
        let fallback = self.passive_action(game_state);
        if game_state.is_raise_capped() {
            return fallback;
        }

        let current_bet = game_state.current_round_bet();
        let player_bet = game_state.current_round_current_player_bet();
        let stack = game_state.current_player_stack();
        let all_in_amount = player_bet.saturating_add(stack);

        // A player whose all-in only reaches the current bet can call, not raise.
        if all_in_amount <= current_bet {
            return fallback;
        }

        let min_raise_total = current_bet.saturating_add(game_state.current_round_min_raise());
        let desired_raise = apply_ratio(game_state.total_pot, pot_fraction);
        let desired_total = current_bet
            .saturating_add(desired_raise)
            .max(min_raise_total);
        let target = desired_total.min(all_in_amount);

        if target <= current_bet || game_state.validate_bet_amount(target).is_err() {
            return fallback;
        }

        if target == all_in_amount {
            AgentAction::AllIn
        } else {
            AgentAction::Bet(target)
        }
    }

    fn action_for_equity(&self, game_state: &GameState, equity: f32) -> AgentAction {
        let required_equity = self.required_equity(game_state);
        let call_cost = self.call_cost(game_state);

        // Below the immediate threshold we never invest extra chips. A free
        // action becomes a check rather than a fold.
        if equity < required_equity {
            return if call_cost > 0 {
                AgentAction::Fold
            } else {
                AgentAction::Bet(game_state.current_round_bet())
            };
        }

        let edge = Self::normalized_edge(equity, required_equity);
        if edge < self.small_edge {
            self.passive_action(game_state)
        } else if edge < self.medium_edge {
            self.raise_for_fraction(game_state, self.small_bet_pot)
        } else if edge < self.large_edge {
            self.raise_for_fraction(game_state, self.medium_bet_pot)
        } else {
            self.raise_for_fraction(game_state, self.large_bet_pot)
        }
    }

    fn monte_carlo_action(
        &self,
        game_state: &GameState,
        mut monte: MonteCarloGame,
        hero_sim_idx: usize,
    ) -> AgentAction {
        let equities = monte.estimate_equity(self.iterations);
        let equity = equities.get(hero_sim_idx).copied().unwrap_or(0.0);
        let required_equity = self.required_equity(game_state);
        let edge = Self::normalized_edge(equity, required_equity);
        let action = self.action_for_equity(game_state, equity);

        trace!(
            ?action,
            equity,
            required_equity,
            edge,
            iterations = self.iterations,
            total_pot = game_state.total_pot,
            current_bet = game_state.current_round_bet(),
            already_bet = game_state.current_round_current_player_bet(),
            "EquityAgent decision"
        );

        action
    }
}

impl Default for EquityAgent {
    fn default() -> Self {
        Self::new(
            "EquityAgent",
            DEFAULT_EQUITY_ITERATIONS,
            DEFAULT_EQUITY_SMALL_EDGE,
            DEFAULT_EQUITY_MEDIUM_EDGE,
            DEFAULT_EQUITY_LARGE_EDGE,
            DEFAULT_EQUITY_SMALL_BET_POT,
            DEFAULT_EQUITY_MEDIUM_BET_POT,
            DEFAULT_EQUITY_LARGE_BET_POT,
        )
    }
}

#[async_trait]
impl Agent for EquityAgent {
    #[instrument(level = "trace", skip(self, game_state), fields(agent_name = %self.name))]
    async fn act(&mut self, _id: u128, game_state: &GameState) -> AgentAction {
        let (hands, hero_sim_idx) = self.simulation_hands(game_state);
        match MonteCarloGame::new(hands) {
            Ok(monte) => self.monte_carlo_action(game_state, monte, hero_sim_idx),
            Err(_) => {
                // Fail closed: do not put extra chips in when equity cannot be
                // estimated, but still check when checking is free.
                let current_bet = game_state.current_round_bet();
                let already_bet = game_state.current_round_current_player_bet();
                if current_bet > already_bet {
                    AgentAction::Fold
                } else {
                    AgentAction::Bet(current_bet)
                }
            }
        }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::GameStateBuilder;

    fn agent() -> EquityAgent {
        EquityAgent::default()
    }

    #[test]
    fn name_and_iterations_are_exposed() {
        let agent = EquityAgent::new("Equity-10k", 10_000, 0.15, 0.35, 0.60, 0.33, 0.66, 1.0);
        assert_eq!(agent.name(), "Equity-10k");
        assert_eq!(agent.iterations(), 10_000);
    }

    #[test]
    fn simulation_hands_preserve_hero_hide_live_opponents_and_drop_folded_seats() {
        let agent = agent();
        let mut game_state = GameStateBuilder::new()
            .stacks(vec![100, 100, 100])
            .blinds(10, 5)
            .build()
            .unwrap();

        let cards: Vec<_> = crate::core::Deck::default().into_iter().take(9).collect();
        let hero = game_state.to_act_idx();
        let villain = (hero + 1) % 3;
        let folded = (hero + 2) % 3;

        game_state.hands[hero].insert(cards[0]);
        game_state.hands[hero].insert(cards[1]);
        game_state.hands[villain].insert(cards[2]);
        game_state.hands[villain].insert(cards[3]);
        game_state.hands[folded].insert(cards[4]);
        game_state.hands[folded].insert(cards[5]);
        game_state.board.extend([cards[6], cards[7], cards[8]]);
        game_state.player_active.disable(folded);

        let (hands, hero_sim_idx) = agent.simulation_hands(&game_state);
        assert_eq!(hands.len(), 2);
        assert_eq!(hands[hero_sim_idx], game_state.hands[hero]);
        let villain_sim_idx = 1 - hero_sim_idx;
        assert_eq!(hands[villain_sim_idx].count(), 3);
        assert!(!hands[villain_sim_idx].contains(&cards[2]));
        assert!(!hands[villain_sim_idx].contains(&cards[3]));
    }

    #[test]
    fn folds_below_pot_odds_and_calls_at_small_positive_edge() {
        let agent = agent();
        let mut game_state = GameStateBuilder::new()
            .stacks(vec![90, 80])
            .player_bet(vec![10, 20])
            .blinds(10, 5)
            .build()
            .unwrap();

        game_state.round_data.to_act_idx = 0;
        game_state.round_data.bet = 20;
        game_state.round_data.player_bet[0] = 10;
        game_state.round_data.player_bet[1] = 20;
        game_state.total_pot = 30;

        // Calling costs 10 into a final pot of 40 => required equity 25%.
        assert_eq!(
            agent.action_for_equity(&game_state, 0.249),
            AgentAction::Fold
        );
        assert_eq!(
            agent.action_for_equity(&game_state, 0.30),
            AgentAction::Call
        );
    }

    #[test]
    fn free_action_uses_equal_share_equity_as_aggression_baseline() {
        let agent = agent();
        let mut game_state = GameStateBuilder::new()
            .stacks(vec![100, 100])
            .blinds(10, 5)
            .build()
            .unwrap();

        game_state.round_data.bet = 10;
        let hero = game_state.to_act_idx();
        game_state.round_data.player_bet[hero] = 10;
        game_state.total_pot = 20;

        // Heads-up fair share is 50%; 49% checks, 55% has only a 10% normalized
        // edge and also checks, 60% crosses the 15% small-bet threshold.
        assert_eq!(
            agent.action_for_equity(&game_state, 0.49),
            AgentAction::Bet(10)
        );
        assert_eq!(
            agent.action_for_equity(&game_state, 0.55),
            AgentAction::Bet(10)
        );
        assert_eq!(
            agent.action_for_equity(&game_state, 0.60),
            AgentAction::Bet(20)
        );
    }

    #[test]
    fn equity_edge_selects_small_medium_and_large_sizings() {
        let agent = agent();
        let mut game_state = GameStateBuilder::new()
            .stacks(vec![200, 200])
            .blinds(2, 1)
            .round(crate::arena::game_state::Round::Flop)
            .build()
            .unwrap();
        game_state.round_data.to_act_idx = 0;
        game_state.round_data.bet = 0;
        game_state.round_data.player_bet[0] = 0;
        game_state.total_pot = 100;

        // Heads-up free action => required equity 50%.
        // 60% equity => edge 20% => 33% pot.
        assert_eq!(
            agent.action_for_equity(&game_state, 0.60),
            AgentAction::Bet(33)
        );
        // 70% equity => edge 40% => 66% pot.
        assert_eq!(
            agent.action_for_equity(&game_state, 0.70),
            AgentAction::Bet(66)
        );
        // 85% equity => edge 70% => 100% pot.
        assert_eq!(
            agent.action_for_equity(&game_state, 0.85),
            AgentAction::Bet(100)
        );
    }

    #[test]
    fn sizing_is_clamped_to_min_raise_and_stack() {
        let agent = agent();
        let mut game_state = GameStateBuilder::new()
            .stacks(vec![25, 200])
            .blinds(10, 5)
            .round(crate::arena::game_state::Round::Flop)
            .build()
            .unwrap();
        game_state.round_data.to_act_idx = 0;
        game_state.round_data.bet = 10;
        game_state.round_data.min_raise = 10;
        game_state.round_data.player_bet[0] = 10;
        game_state.total_pot = 20;

        // Strong edge asks for a pot-sized raise to 30, which is legal here.
        assert_eq!(
            agent.action_for_equity(&game_state, 0.90),
            AgentAction::Bet(30)
        );

        game_state.stacks[0] = 15;
        // Same request now reaches Hero's 25 total and becomes all-in.
        assert_eq!(
            agent.action_for_equity(&game_state, 0.90),
            AgentAction::AllIn
        );
    }

    #[test]
    fn raise_cap_falls_back_to_check_or_call() {
        let agent = agent();
        let mut game_state = GameStateBuilder::new()
            .stacks(vec![200, 200])
            .blinds(10, 5)
            .round(crate::arena::game_state::Round::Flop)
            .build()
            .unwrap();
        game_state.round_data.to_act_idx = 0;
        game_state.round_data.bet = 20;
        game_state.round_data.player_bet[0] = 10;
        game_state.round_data.total_raise_count = 3;
        game_state.total_pot = 40;
        game_state.max_raises_per_round = Some(3);

        assert_eq!(
            agent.action_for_equity(&game_state, 0.95),
            AgentAction::Call
        );
    }

    #[test]
    fn short_stack_uses_actual_call_cost_for_pot_odds() {
        let agent = agent();
        let mut game_state = GameStateBuilder::new()
            .stacks(vec![5, 80])
            .player_bet(vec![10, 20])
            .blinds(10, 5)
            .build()
            .unwrap();

        game_state.round_data.to_act_idx = 0;
        game_state.round_data.bet = 20;
        game_state.round_data.player_bet[0] = 10;
        game_state.round_data.player_bet[1] = 20;
        game_state.total_pot = 30;

        // Nominal gap is 10, but Hero has only 5. Required equity = 5 / 35.
        assert_eq!(
            agent.action_for_equity(&game_state, 0.14),
            AgentAction::Fold
        );
        assert_eq!(
            agent.action_for_equity(&game_state, 0.15),
            AgentAction::Call
        );
    }
}
