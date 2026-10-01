use async_trait::async_trait;
use tracing::{instrument, trace};

use crate::{
    arena::{GameState, action::AgentAction},
    core::Hand,
    holdem::MonteCarloGame,
};

use super::Agent;

/// Default Monte Carlo sample count used by [`EquityAgent`].
pub const DEFAULT_EQUITY_ITERATIONS: usize = 1_000;

/// A deliberately small equity-only baseline agent.
///
/// The agent hides every opponent's hole cards, estimates showdown equity
/// against uniformly random unknown hands with [`MonteCarloGame`], and compares
/// that equity with the immediate pot odds of calling.
///
/// It never value-bets or raises. When checking is free it checks; when facing a
/// bet it calls iff `equity >= call_cost / (pot + call_cost)`, otherwise it
/// folds. This isolates raw equity information from action randomization and
/// bet-sizing policy.
#[derive(Debug, Clone)]
pub struct EquityAgent {
    name: String,
    iterations: usize,
}

impl EquityAgent {
    pub fn new(name: impl Into<String>, iterations: usize) -> Self {
        assert!(iterations > 0, "EquityAgent iterations must be > 0");
        Self {
            name: name.into(),
            iterations,
        }
    }

    pub fn iterations(&self) -> usize {
        self.iterations
    }

    /// Keep the acting player's real cards while replacing every opponent hand
    /// with the public board only. `MonteCarloGame` fills the missing private
    /// cards and remaining community cards during simulation.
    fn clean_hands(&self, game_state: &GameState) -> Vec<Hand> {
        let mut unknown_hand = Hand::new();
        unknown_hand.extend(game_state.board.iter().cloned());

        let to_act_idx = game_state.to_act_idx();
        game_state
            .hands
            .iter()
            .enumerate()
            .map(|(hand_idx, hand)| {
                if hand_idx == to_act_idx {
                    hand.clone()
                } else {
                    unknown_hand.clone()
                }
            })
            .collect()
    }

    fn action_for_equity(&self, game_state: &GameState, equity: f32) -> AgentAction {
        let current_bet = game_state.current_round_bet();
        let already_bet = game_state.current_round_current_player_bet();
        let amount_to_match = current_bet.saturating_sub(already_bet);

        // A short stack can call only with the chips it has left, so use the
        // actual incremental cost rather than the nominal amount to match.
        let call_cost = amount_to_match.min(game_state.current_player_stack());

        if call_cost <= 0 {
            return AgentAction::Bet(current_bet);
        }

        let final_pot_if_called = game_state.total_pot.saturating_add(call_cost);
        if final_pot_if_called <= 0 {
            return AgentAction::Fold;
        }

        let pot_odds = call_cost as f32 / final_pot_if_called as f32;
        if equity >= pot_odds {
            AgentAction::Call
        } else {
            AgentAction::Fold
        }
    }

    fn monte_carlo_action(&self, game_state: &GameState, mut monte: MonteCarloGame) -> AgentAction {
        let equities = monte.estimate_equity(self.iterations);
        let to_act_idx = game_state.to_act_idx();
        let equity = equities.get(to_act_idx).copied().unwrap_or(0.0);
        let action = self.action_for_equity(game_state, equity);

        trace!(
            ?action,
            equity,
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
        Self::new("EquityAgent", DEFAULT_EQUITY_ITERATIONS)
    }
}

#[async_trait]
impl Agent for EquityAgent {
    #[instrument(level = "trace", skip(self, game_state), fields(agent_name = %self.name))]
    async fn act(&mut self, _id: u128, game_state: &GameState) -> AgentAction {
        let clean_hands = self.clean_hands(game_state);
        match MonteCarloGame::new(clean_hands) {
            Ok(monte) => self.monte_carlo_action(game_state, monte),
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

    #[test]
    fn name_and_iterations_are_exposed() {
        let agent = EquityAgent::new("Equity-10k", 10_000);
        assert_eq!(agent.name(), "Equity-10k");
        assert_eq!(agent.iterations(), 10_000);
    }

    #[test]
    fn clean_hands_preserves_hero_and_hides_opponents() {
        let agent = EquityAgent::new("Equity", 100);
        let mut game_state = GameStateBuilder::new()
            .stacks(vec![100, 100])
            .blinds(10, 5)
            .build()
            .unwrap();

        let cards: Vec<_> = crate::core::Deck::default().into_iter().take(7).collect();
        let hero = game_state.to_act_idx();
        let villain = 1 - hero;

        game_state.hands[hero].insert(cards[0]);
        game_state.hands[hero].insert(cards[1]);
        game_state.hands[villain].insert(cards[2]);
        game_state.hands[villain].insert(cards[3]);
        game_state.board.extend([cards[4], cards[5], cards[6]]);

        let clean = agent.clean_hands(&game_state);
        assert_eq!(clean[hero], game_state.hands[hero]);
        assert_eq!(clean[villain].count(), 3);
        assert!(!clean[villain].contains(&cards[2]));
        assert!(!clean[villain].contains(&cards[3]));
        for board_card in &game_state.board {
            assert!(clean[villain].contains(board_card));
        }
    }

    #[test]
    fn calls_when_equity_meets_pot_odds_and_folds_when_it_does_not() {
        let agent = EquityAgent::new("Equity", 100);
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

        // Calling costs 10 into a final pot of 40 => break-even equity 25%.
        assert_eq!(agent.action_for_equity(&game_state, 0.25), AgentAction::Call);
        assert_eq!(agent.action_for_equity(&game_state, 0.249), AgentAction::Fold);
    }

    #[test]
    fn checks_when_no_call_is_required() {
        let agent = EquityAgent::new("Equity", 100);
        let mut game_state = GameStateBuilder::new()
            .stacks(vec![100, 100])
            .blinds(10, 5)
            .build()
            .unwrap();

        game_state.round_data.bet = 10;
        let hero = game_state.to_act_idx();
        game_state.round_data.player_bet[hero] = 10;

        assert_eq!(
            agent.action_for_equity(&game_state, 0.0),
            AgentAction::Bet(10)
        );
    }

    #[test]
    fn short_stack_uses_actual_call_cost_for_pot_odds() {
        let agent = EquityAgent::new("Equity", 100);
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

        // Nominal gap is 10, but Hero has only 5. Break-even = 5 / 35.
        assert_eq!(agent.action_for_equity(&game_state, 0.15), AgentAction::Call);
        assert_eq!(agent.action_for_equity(&game_state, 0.14), AgentAction::Fold);
    }
}
