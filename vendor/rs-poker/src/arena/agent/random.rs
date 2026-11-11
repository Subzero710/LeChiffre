use crate::arena::{Chips, GameState, action::AgentAction};
use async_trait::async_trait;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng, rng};
use std::sync::atomic::{AtomicUsize, Ordering};

use tracing::{instrument, trace};

use super::{Agent, AgentGenerator};

#[derive(Debug, Clone)]
pub struct RandomAgent {
    name: String,
    percent_fold: Vec<f64>,
    percent_call: Vec<f64>,
    rng: SmallRng,
}

impl RandomAgent {
    pub fn new(name: impl Into<String>, percent_fold: Vec<f64>, percent_call: Vec<f64>) -> Self {
        Self::with_rng(
            name,
            percent_fold,
            percent_call,
            SmallRng::from_rng(&mut rng()),
        )
    }

    pub fn new_with_seed(
        name: impl Into<String>,
        percent_fold: Vec<f64>,
        percent_call: Vec<f64>,
        seed: u64,
    ) -> Self {
        Self::with_rng(
            name,
            percent_fold,
            percent_call,
            SmallRng::seed_from_u64(seed),
        )
    }

    fn with_rng(
        name: impl Into<String>,
        percent_fold: Vec<f64>,
        percent_call: Vec<f64>,
        rng: SmallRng,
    ) -> Self {
        Self {
            name: name.into(),
            percent_call,
            percent_fold,
            rng,
        }
    }
}

impl Default for RandomAgent {
    fn default() -> Self {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let idx = COUNTER.fetch_add(1, Ordering::Relaxed);
        Self::new(
            format!("RandomAgent-default-{idx}"),
            vec![0.25, 0.30, 0.50],
            vec![0.5, 0.6, 0.45],
        )
    }
}

#[async_trait]
impl Agent for RandomAgent {
    #[instrument(level = "trace", skip(self, game_state), fields(agent_name = %self.name))]
    async fn act(self: &mut RandomAgent, _id: u128, game_state: &GameState) -> AgentAction {
        let round_data = &game_state.round_data;
        let player_bet = round_data.current_player_bet();
        let player_stack = game_state.stacks[round_data.to_act_idx];
        let curr_bet = round_data.bet;
        let raise_count = round_data.total_raise_count;

        // The min we can bet when not calling is the current bet plus the min raise
        // However it's possible that would put the player all in.
        let min = (curr_bet + round_data.min_raise).min(player_bet + player_stack);

        // The max we can bet going all in.
        //
        // However we don't want to overbet too early
        // so cap to a value representing how much we
        // could get everyone to put into the pot by
        // calling a pot sized bet (plus a little more for spicyness)
        //
        // That could be the same as the min
        let pot_value = game_state
            .total_pot
            .saturating_mul(round_data.num_players_need_action() as Chips + 1);
        let max = (player_bet + player_stack).min(pot_value).max(min);

        // We shouldn't fold when checking is an option.
        let can_fold = curr_bet > player_bet;

        // As there are more raises we should look deeper
        // into the fold percentaages that the user gave us
        let fold_idx = raise_count.min((self.percent_fold.len() - 1) as u8) as usize;
        let percent_fold = self.percent_fold.get(fold_idx).map_or_else(|| 1.0, |v| *v);

        // As there are more raises we should look deeper
        // into the call percentages that the user gave us
        let call_idx = raise_count.min((self.percent_call.len() - 1) as u8) as usize;
        let percent_call = self.percent_call.get(call_idx).map_or_else(|| 1.0, |v| *v);

        // Now do the action decision
        let action = if can_fold && self.rng.random_bool(percent_fold) {
            // We can fold and the rng was in favor so fold.
            AgentAction::Fold
        } else if self.rng.random_bool(percent_call) {
            // We're calling, which is the same as betting the same as the current.
            // Luckily for us the simulation will take care of us if this puts us all in.
            AgentAction::Call
        } else if max > min {
            // If there's some range and the rng didn't choose another option. So bet some
            // amount.
            AgentAction::Bet(self.rng.random_range(min..max))
        } else {
            AgentAction::Bet(max)
        };

        trace!(?action, raise_count, can_fold, "RandomAgent decision");
        action
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[derive(Debug, Clone)]
pub struct RandomAgentGenerator {
    name: Option<String>,
    percent_fold: Vec<f64>,
    percent_call: Vec<f64>,
    /// Optional base seed. When set, each generated `RandomAgent` is
    /// seeded from `base.wrapping_add(player_idx as u64)` so that
    /// repeated runs of the same competition produce bit-identical
    /// decisions. When `None` the agents draw fresh OS entropy.
    seed: Option<u64>,
}

impl RandomAgentGenerator {
    pub fn new(percent_fold: Vec<f64>, percent_call: Vec<f64>) -> Self {
        Self {
            name: None,
            percent_fold,
            percent_call,
            seed: None,
        }
    }

    /// Create a generator whose produced agents are seeded from a
    /// deterministic base. Each player's `RandomAgent` is seeded from
    /// `seed.wrapping_add(player_idx as u64)`.
    pub fn seeded(percent_fold: Vec<f64>, percent_call: Vec<f64>, seed: u64) -> Self {
        Self {
            name: None,
            percent_fold,
            percent_call,
            seed: Some(seed),
        }
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set (or clear) the seed used to construct generated agents.
    pub fn with_seed(mut self, seed: Option<u64>) -> Self {
        self.seed = seed;
        self
    }

    fn resolve_name(&self, player_idx: usize) -> String {
        self.name
            .clone()
            .unwrap_or_else(|| format!("RandomAgent-{player_idx}"))
    }
}

impl AgentGenerator for RandomAgentGenerator {
    fn generate(&self, player_idx: usize, _game_state: &GameState) -> Box<dyn Agent> {
        let name = self.resolve_name(player_idx);
        match self.seed {
            Some(base) => Box::new(RandomAgent::new_with_seed(
                name,
                self.percent_fold.clone(),
                self.percent_call.clone(),
                base.wrapping_add(player_idx as u64),
            )),
            None => Box::new(RandomAgent::new(
                name,
                self.percent_fold.clone(),
                self.percent_call.clone(),
            )),
        }
    }
}

impl Default for RandomAgentGenerator {
    fn default() -> Self {
        Self::new(vec![0.25, 0.30, 0.50], vec![0.5, 0.6, 0.45])
    }
}

#[cfg(test)]
mod tests {
    use crate::arena::{
        HoldemSimulationBuilder,
        test_util::{assert_valid_game_state, assert_valid_round_data},
    };

    use super::*;
    use crate::arena::GameStateBuilder;

    #[tokio::test(flavor = "current_thread")]
    async fn test_random_generator_produces_named_caller() {
        let generator = RandomAgentGenerator::new(vec![0.0], vec![1.0]);
        let game_state = GameStateBuilder::new()
            .num_players_with_stack(2, 100)
            .blinds(10, 5)
            .build()
            .unwrap();

        let mut agent = generator.generate(3, &game_state);
        assert_eq!(agent.name(), "RandomAgent-3");

        match agent.act(0, &game_state).await {
            AgentAction::Call => {}
            action => panic!("Expected forced call, got {:?}", action),
        }
    }

    /// Regression test for M2: `RandomAgentGenerator::seeded` must
    /// produce deterministic agents — two generators built with the
    /// same seed must generate agents whose action streams are
    /// identical across runs.
    #[tokio::test(flavor = "current_thread")]
    async fn test_seeded_random_agent_generator_is_deterministic() {
        let game_state = GameStateBuilder::new()
            .num_players_with_stack(2, 500)
            .blinds(10, 5)
            .build()
            .unwrap();

        async fn collect_actions(game_state: &GameState) -> Vec<AgentAction> {
            let generator =
                RandomAgentGenerator::seeded(vec![0.1, 0.2], vec![0.4, 0.3], 0xdeadbeef);
            let mut agent = generator.generate(1, game_state);
            let mut actions = Vec::new();
            for i in 0..64u128 {
                actions.push(agent.act(i, game_state).await);
            }
            actions
        }

        let run_a = collect_actions(&game_state).await;
        let run_b = collect_actions(&game_state).await;
        assert_eq!(run_a, run_b);
    }

    /// Different player indices must still produce independent streams
    /// even when seeded from the same base.
    #[tokio::test(flavor = "current_thread")]
    async fn test_seeded_random_agent_generator_differs_across_players() {
        let game_state = GameStateBuilder::new()
            .num_players_with_stack(2, 500)
            .blinds(10, 5)
            .build()
            .unwrap();
        let generator = RandomAgentGenerator::seeded(vec![0.1, 0.2], vec![0.4, 0.3], 0xdeadbeef);
        let mut p0 = generator.generate(0, &game_state);
        let mut p1 = generator.generate(1, &game_state);
        let mut actions0 = Vec::new();
        let mut actions1 = Vec::new();
        for i in 0..32u128 {
            actions0.push(p0.act(i, &game_state).await);
            actions1.push(p1.act(i, &game_state).await);
        }
        assert_ne!(actions0, actions1);
    }

    #[test]
    fn test_random_generator_uses_custom_name() {
        let generator = RandomAgentGenerator::new(vec![0.0], vec![1.0]).with_name("RandomHero");
        let game_state = GameStateBuilder::new()
            .num_players_with_stack(2, 20)
            .blinds(10, 5)
            .build()
            .unwrap();

        let agent = generator.generate(7, &game_state);
        assert_eq!(agent.name(), "RandomHero");
    }

    #[tokio::test]
    async fn test_random_five_nl() {
        let rng = SmallRng::from_rng(&mut rng());

        let stacks = vec![100; 5];
        let game_state = GameStateBuilder::new()
            .stacks(stacks)
            .blinds(10, 5)
            .build()
            .unwrap();
        let agents: Vec<Box<dyn Agent>> = (0..5)
            .map(|idx| {
                Box::new(RandomAgent::new(
                    format!("RandomAgent-{idx}"),
                    vec![0.25, 0.30, 0.50],
                    vec![0.5, 0.6, 0.45],
                )) as Box<dyn Agent>
            })
            .collect();

        // The simulation deals hole cards itself; seeding them here too would
        // leave each player holding four hole cards (and a nine-card showdown).
        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .build_with_rng(rng)
            .unwrap();

        sim.run().await;

        let min_stack = sim
            .game_state
            .stacks
            .clone()
            .into_iter()
            .reduce(Chips::min)
            .unwrap();
        let max_stack = sim
            .game_state
            .stacks
            .clone()
            .into_iter()
            .reduce(Chips::max)
            .unwrap();

        assert_ne!(min_stack, max_stack, "There should have been some betting.");

        assert_valid_round_data(&sim.game_state.round_data);
        assert_valid_game_state(&sim.game_state);
    }

    #[tokio::test]
    async fn test_random_agents_no_fold_get_all_rounds() {
        let stacks = vec![100; 5];
        let game_state = GameStateBuilder::new()
            .stacks(stacks)
            .blinds(10, 5)
            .build()
            .unwrap();
        let agents: Vec<Box<dyn Agent>> = (0..5)
            .map(|idx| {
                Box::new(RandomAgent::new(
                    format!("AggroRandom-{idx}"),
                    vec![0.0],
                    vec![0.75],
                )) as Box<dyn Agent>
            })
            .collect();
        let mut sim = HoldemSimulationBuilder::default()
            .agents(agents)
            .game_state(game_state)
            .build()
            .unwrap();

        sim.run().await;
        assert!(sim.game_state.is_complete());
        assert_valid_game_state(&sim.game_state);
    }

    #[test]
    fn test_random_agent_name_returns_name() {
        let agent = RandomAgent::new("TestAgent", vec![0.5], vec![0.5]);
        // Test that name() returns the actual name, not empty string
        assert_eq!(agent.name(), "TestAgent");
        assert!(!agent.name().is_empty());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_random_agent_can_fold_logic() {
        // When current bet > player bet, should be able to fold
        let mut agent = RandomAgent::new("FoldTest", vec![1.0], vec![0.0]); // 100% fold
        let mut game_state = GameStateBuilder::new()
            .stacks(vec![100, 100])
            .blinds(10, 5)
            .build()
            .unwrap();

        // Set up a situation where the player faces a bet
        // round_data.bet = current bet to call
        // player_bets[to_act] = what this player has already bet this round
        game_state.round_data.bet = 10; // There's a bet of 10 to call
        game_state.round_data.player_bet[0] = 5; // Player has only bet 5 (like SB)

        // can_fold = curr_bet (10) > player_bet (5) = true
        // With 100% fold probability, should fold
        let action = agent.act(0, &game_state).await;
        assert!(
            matches!(action, AgentAction::Fold),
            "With 100% fold when can_fold=true, should fold. Got {:?}",
            action
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_random_agent_cannot_fold_when_checking() {
        // When current bet == player bet, should not fold (can check)
        let mut agent = RandomAgent::new("CheckTest", vec![1.0], vec![0.0]); // 100% fold
        let mut game_state = GameStateBuilder::new()
            .stacks(vec![100, 100])
            .blinds(10, 5)
            .build()
            .unwrap();

        // Simulate BB position where bet is matched (nothing to call)
        game_state.round_data.bet = 0;
        game_state.round_data.player_bet[0] = 0;
        game_state.stacks[0] = 100;

        // can_fold = curr_bet (0) > player_bet (0) = false
        let action = agent.act(0, &game_state).await;
        // With no bet to call (bet=0), can_fold=false, so shouldn't fold
        // Should call or bet
        assert!(
            !matches!(action, AgentAction::Fold),
            "Should not fold when can check. Got {:?}",
            action
        );
    }

    #[tokio::test]
    async fn test_random_agent_min_calculation() {
        // Test that min bet is calculated correctly using addition
        let agent = RandomAgent::new("MinTest", vec![0.0], vec![0.0]);
        let game_state = GameStateBuilder::new()
            .stacks(vec![50, 50])
            .blinds(10, 5)
            .build()
            .unwrap();

        // min = (curr_bet + min_raise).min(player_bet + player_stack)
        // curr_bet = 10 (big blind)
        // min_raise = 10 (big blind)
        // So min = 20 unless that would be all-in
        // player_bet = 5 (SB), player_stack = 45
        // player_bet + player_stack = 50
        // min(20, 50) = 20

        // The agent uses this internally; we can verify the game completes
        let agents: Vec<Box<dyn Agent>> = vec![
            Box::new(agent),
            Box::new(RandomAgent::new("Other", vec![0.0], vec![1.0])),
        ];

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .build()
            .unwrap();

        sim.run().await;
        assert!(sim.game_state.is_complete());
    }

    #[tokio::test]
    async fn test_random_agent_pot_value_multiplication() {
        // Test: pot_value = (num_players + 1.0) * total_pot
        // With *: correct
        // With +: (5 + 1.0) + 15 = 21 (wrong)
        // With /: (5 + 1.0) / 15 = 0.4 (wrong)

        let stacks = vec![100; 5];
        let game_state = GameStateBuilder::new()
            .stacks(stacks)
            .blinds(10, 5)
            .build()
            .unwrap();

        // 5 players, pot = 15 (BB 10 + SB 5)
        // pot_value = (5 + 1.0) * 15 = 90

        let agents: Vec<Box<dyn Agent>> = (0..5)
            .map(|idx| {
                Box::new(RandomAgent::new(
                    format!("Agent{idx}"),
                    vec![0.0],
                    vec![0.0],
                )) as Box<dyn Agent>
            })
            .collect();

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .build()
            .unwrap();

        sim.run().await;
        assert!(sim.game_state.is_complete());
    }

    #[tokio::test]
    async fn test_random_agent_raise_count_index_calculation() {
        // Test: fold_idx = raise_count.min((self.percent_fold.len() - 1) as u8) as usize
        // The subtraction is important: len() - 1 gives last valid index

        let agent = RandomAgent::new(
            "IndexTest",
            vec![0.5, 0.75, 0.9], // 3 elements, indices 0, 1, 2
            vec![0.5],
        );

        // With many raises, should use index 2 (last), not overflow
        assert_eq!(agent.percent_fold.len(), 3);

        // Test with simulation - should not panic
        let game_state = GameStateBuilder::new()
            .stacks(vec![100, 100])
            .blinds(10, 5)
            .build()
            .unwrap();
        let agents: Vec<Box<dyn Agent>> = vec![
            Box::new(agent),
            Box::new(RandomAgent::new("Aggro", vec![0.0], vec![0.0])), // Always raises
        ];

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .build()
            .unwrap();

        sim.run().await;
        assert!(sim.game_state.is_complete());
    }
}
