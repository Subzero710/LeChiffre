//! Compatibility agent retained from LeChiffre before the EquityAgent merge.
//! Original rs_poker implementation, not an alias for EquityAgent.
use super::Agent;
use crate::{
    arena::{
        Chips,
        action::AgentAction,
        game_state::{GameState, Round},
        money::apply_ratio,
    },
    core::Hand,
    holdem::MonteCarloGame,
};
use async_trait::async_trait;
use rand::{RngExt, SeedableRng, rng, rngs::SmallRng};
use tracing::{instrument, trace};

/// This is an `Agent` implementation that chooses random actions in some
/// relation to the value of the pot. It assumes that it's up against totally
/// random cards for each hand then estimates the value of the pot for what
/// range of values to bet.
///
/// The percent_call is the percent that the agent will not bet even though it
/// values the pot above the current bet or 0 if it's the first to act.
#[derive(Debug, Clone)]
pub struct RandomPotControlAgent {
    name: String,
    percent_call: Vec<f64>,
    rng: SmallRng,
}

impl RandomPotControlAgent {
    pub fn new(name: impl Into<String>, percent_call: Vec<f64>) -> Self {
        Self::with_rng(name, percent_call, SmallRng::from_rng(&mut rng()))
    }

    pub fn new_with_seed(name: impl Into<String>, percent_call: Vec<f64>, seed: u64) -> Self {
        Self::with_rng(name, percent_call, SmallRng::seed_from_u64(seed))
    }

    fn with_rng(name: impl Into<String>, percent_call: Vec<f64>, rng: SmallRng) -> Self {
        Self {
            name: name.into(),
            percent_call,
            rng,
        }
    }

    fn expected_pot(&self, game_state: &GameState) -> Chips {
        if game_state.round == Round::Preflop {
            (3 * game_state.big_blind).max(game_state.total_pot)
        } else {
            game_state.total_pot
        }
    }

    fn clean_hands(&self, game_state: &GameState) -> Vec<Hand> {
        let mut default_hand = Hand::new();
        // Copy the board into the default hand
        default_hand.extend(game_state.board.iter().cloned());

        let to_act_idx = game_state.to_act_idx();
        game_state
            .hands
            .clone()
            .into_iter()
            .enumerate()
            .map(|(hand_idx, hand)| {
                if hand_idx == to_act_idx {
                    hand
                } else {
                    default_hand
                }
            })
            .collect()
    }

    fn monte_carlo_based_action(
        &mut self,
        game_state: &GameState,
        mut monte: MonteCarloGame,
    ) -> AgentAction {
        // We play some trickery to make sure that someone will call before there's
        // money in the pot
        let expected_pot = self.expected_pot(game_state);
        // run the monte carlo simulation a lot of times to see who would win with the
        // knowledge that we have. Keeping in mind that we have no information and are
        // actively guessing no hand ranges at all. So this is likely a horrible way to
        // estimate hand strength
        //
        // Then truncate the values to f32.
        let values: Vec<f32> = monte.estimate_equity(1_000).into_iter().collect();
        let to_act_idx = game_state.to_act_idx();

        // How much do I actually value the pot right now?
        let my_value = apply_ratio(expected_pot, *values.get(to_act_idx).unwrap_or(&0.0_f32));

        // What have we already put into the pot for the round?
        let bet_already = game_state.current_round_player_bet(to_act_idx);
        // How much total is required to continue
        let to_call = game_state.current_round_bet();
        // What more is needed from us
        let needed = to_call - bet_already;

        // If we don't value the pot at what's required then just bail out.
        // But only fold if there's actually something to call (otherwise check)
        if my_value < needed && needed > 0 {
            AgentAction::Fold
        } else if needed <= 0 {
            // Nothing to call - just check
            AgentAction::Bet(to_call)
        } else {
            self.random_action(game_state, my_value)
        }
    }

    fn random_action(&mut self, game_state: &GameState, max_value: Chips) -> AgentAction {
        // Use the number of bets to determine the call percentage
        let round_data = &game_state.round_data;
        let raise_count = round_data.total_raise_count;

        let call_idx = raise_count.min((self.percent_call.len() - 1) as u8) as usize;
        let percent_call = self.percent_call.get(call_idx).map_or_else(|| 1.0, |v| *v);

        // Check player's stack to determine valid bet range
        let player_stack = game_state.current_player_stack();
        let player_bet_this_round = game_state.current_round_current_player_bet();
        let max_total_bet = player_bet_this_round + player_stack;

        if self.rng.random_bool(percent_call) {
            AgentAction::Bet(round_data.bet)
        } else {
            // Even though this is a random action try not to under min raise
            let min_raise = round_data.min_raise;
            // The minimum valid raise amount
            let min_raise_total = round_data.bet + min_raise;

            // Check if we can even afford the min raise
            if max_total_bet < min_raise_total {
                // Can't afford min raise - either go all-in or call
                if max_total_bet > round_data.bet {
                    // All-in is our only raise option
                    AgentAction::AllIn
                } else {
                    // Just call
                    AgentAction::Bet(round_data.bet)
                }
            } else {
                // We can afford to raise - pick a random amount
                let high = max_value
                    .max(min_raise_total + min_raise)
                    .min(max_total_bet);
                let bet_value = self
                    .rng
                    .random_range(min_raise_total..high.max(min_raise_total + 1));

                AgentAction::Bet(bet_value)
            }
        }
    }
}

#[async_trait]
impl Agent for RandomPotControlAgent {
    #[instrument(level = "trace", skip(self, game_state), fields(agent_name = %self.name))]
    async fn act(&mut self, _id: u128, game_state: &GameState) -> AgentAction {
        // We don't want to cheat.
        // So replace all the hands but our own
        let clean_hands = self.clean_hands(game_state);
        // Now check if we can simulate that
        let action = if let Ok(monte) = MonteCarloGame::new(clean_hands) {
            self.monte_carlo_based_action(game_state, monte)
        } else {
            // If we can't do monte carlo, check if we can fold or need to check
            let to_call = game_state.current_round_bet();
            let bet_already = game_state.current_round_current_player_bet();
            let needed = to_call - bet_already;
            if needed > 0 {
                AgentAction::Fold
            } else {
                // Nothing to call - just check
                AgentAction::Bet(to_call)
            }
        };

        trace!(?action, "RandomPotControlAgent decision");
        action
    }

    fn name(&self) -> &str {
        &self.name
    }
}
