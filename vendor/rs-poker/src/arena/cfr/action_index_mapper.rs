use crate::arena::Chips;

use crate::arena::{GameState, action::AgentAction};

/// Total number of action indices in the fixed mapping scheme.
/// - Index 0: Fold
/// - Index 1: Call/Check
/// - Indices 2-13: Raises (spread from min to max using logarithmic distribution)
/// - Index 14: All-in
/// - Index 15: Reserved
///
/// 16 indices is sufficient for typical poker action generators which produce
/// 4-8 distinct bet sizes. The 12 raise slots provide ~38% log-space resolution,
/// enough to distinguish standard pot-fraction bets (0.33x, 0.67x, 1.0x, 1.5x).
/// This compact layout reduces regret matcher memory by ~69% compared to 52 indices.
pub const NUM_ACTION_INDICES: usize = 16;

/// Index for fold action.
pub const ACTION_IDX_FOLD: usize = 0;

/// Index for call/check action.
pub const ACTION_IDX_CALL: usize = 1;

/// First index for raise actions.
pub const ACTION_IDX_RAISE_MIN: usize = 2;

/// Last index for raise actions.
pub const ACTION_IDX_RAISE_MAX: usize = 13;

/// Index for all-in action.
pub const ACTION_IDX_ALL_IN: usize = 14;

/// Configuration for the ActionIndexMapper.
///
/// Defines the effective range of bet amounts that will be mapped to raise indices.
/// Amounts outside this range will be clamped to the boundary indices.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ActionIndexMapperConfig {
    /// Minimum bet amount for the mapping range (typically big blind).
    pub min_bet: Chips,
    /// Maximum bet amount for the mapping range (typically second largest stack).
    pub max_bet: Chips,
}

impl ActionIndexMapperConfig {
    /// Create a new configuration with the specified bet range.
    pub fn new(min_bet: Chips, max_bet: Chips) -> Self {
        Self { min_bet, max_bet }
    }

    /// Compute the effective range from a game state.
    ///
    /// Returns `(min_bet, max_bet)` where:
    /// - `min_bet` is the big blind
    /// - `max_bet` is the second largest stack (or largest if only one player)
    ///
    /// The second largest stack is used because that's the maximum effective bet
    /// that can be called in a heads-up or multi-way pot.
    pub fn from_game_state(game_state: &GameState) -> Self {
        let (min_bet, max_bet) = compute_effective_range(game_state);
        Self::new(min_bet, max_bet)
    }
}

/// Compute the effective bet range from a game state.
///
/// Returns `(min_bet, max_bet)` where:
/// - `min_bet` is the big blind
/// - `max_bet` is the second largest stack (or largest stack if only one player)
pub fn compute_effective_range(game_state: &GameState) -> (Chips, Chips) {
    let min_bet = game_state.big_blind;

    // Find the two largest starting stacks to determine effective stack
    let mut stacks: Vec<Chips> = game_state.starting_stacks.to_vec();
    stacks.sort_by(|a, b| b.cmp(a));

    let max_bet = if stacks.len() >= 2 {
        // Second largest stack is the effective stack
        stacks[1]
    } else if !stacks.is_empty() {
        // Single player - use their stack
        stacks[0]
    } else {
        panic!("action mapper requires at least one stack")
    };

    // Ensure max_bet is at least min_bet to avoid division issues
    let max_bet = max_bet.max(min_bet.saturating_mul(2));

    (min_bet, max_bet)
}

/// Maps poker actions to fixed indices for the CFR tree.
///
/// This mapper uses a fixed absolute-amount mapping that spreads raises
/// across indices 2-13 based on the actual bet amount using logarithmic
/// distribution.
///
/// ## Index Layout (16 total)
///
/// | Index | Action |
/// |-------|--------|
/// | 0 | Fold |
/// | 1 | Call/Check |
/// | 2-13 | Raises (spread from min to max using log scale) |
/// | 14 | All-in |
/// | 15 | Reserved |
///
/// The logarithmic distribution is used because poker bet sizing often
/// grows exponentially (e.g., 3x, 9x, 27x patterns).
#[derive(Debug, Clone)]
pub struct ActionIndexMapper {
    config: ActionIndexMapperConfig,
}

impl ActionIndexMapper {
    /// Create a new mapper with the given configuration.
    pub fn new(config: ActionIndexMapperConfig) -> Self {
        Self { config }
    }

    /// Create a new mapper from a game state.
    ///
    /// The effective range is computed from the big blind and stacks.
    pub fn from_game_state(game_state: &GameState) -> Self {
        Self::new(ActionIndexMapperConfig::from_game_state(game_state))
    }

    /// Get the configuration for this mapper.
    pub fn config(&self) -> &ActionIndexMapperConfig {
        &self.config
    }

    /// Map an action to its index in the children array.
    ///
    /// # Arguments
    ///
    /// * `action` - The action to map
    /// * `game_state` - The current game state (used to determine all-in amount)
    ///
    /// # Returns
    ///
    /// The index (0-51) for this action
    pub fn action_to_idx(&self, action: &AgentAction, game_state: &GameState) -> usize {
        self.action_to_idx_raw(
            action,
            game_state.current_round_bet(),
            game_state.current_round_current_player_bet(),
            game_state.current_player_stack(),
        )
    }

    /// Map an action to its index using raw state values.
    ///
    /// This is useful when you have the relevant state values but not a full GameState,
    /// such as when reconstructing pre-action state from a payload.
    ///
    /// # Arguments
    ///
    /// * `action` - The action to map
    /// * `current_round_bet` - The current bet to call in this round
    /// * `current_player_bet` - How much the current player has already bet this round
    /// * `current_player_stack` - The current player's remaining stack
    ///
    /// # Returns
    ///
    /// The index (0-51) for this action
    pub fn action_to_idx_raw(
        &self,
        action: &AgentAction,
        current_round_bet: Chips,
        current_player_bet: Chips,
        current_player_stack: Chips,
    ) -> usize {
        match action {
            AgentAction::Fold => ACTION_IDX_FOLD,
            AgentAction::Call => ACTION_IDX_CALL,
            AgentAction::AllIn => ACTION_IDX_ALL_IN,
            AgentAction::Bet(amount) => {
                // Check if this is a call (matches current bet) FIRST.
                // This is important because when call amount == all-in amount
                // (e.g., calling uses entire stack), we want to treat it as a call
                // not an all-in raise.
                if *amount == current_round_bet {
                    return ACTION_IDX_CALL;
                }

                // Check if this bet is actually an all-in
                let all_in_amount = current_player_bet + current_player_stack;

                // Treat an exact all-in amount as the all-in action.
                if *amount == all_in_amount {
                    return ACTION_IDX_ALL_IN;
                }

                // Otherwise, map the bet amount to an index using log scale
                self.bet_to_index(*amount)
            }
        }
    }

    /// Map a bet amount to an index using logarithmic distribution.
    ///
    /// Bets <= min_bet map to ACTION_IDX_RAISE_MIN.
    /// Bets >= max_bet map to ACTION_IDX_RAISE_MAX.
    /// Bets in between are distributed logarithmically across the raise range.
    fn bet_to_index(&self, bet: Chips) -> usize {
        let min_bet = self.config.min_bet;
        let max_bet = self.config.max_bet;

        if bet <= min_bet {
            return ACTION_IDX_RAISE_MIN;
        }
        if bet >= max_bet {
            return ACTION_IDX_RAISE_MAX;
        }

        // Use logarithmic interpolation
        let log_min = (min_bet as f64).ln();
        let log_max = (max_bet as f64).ln();
        let log_bet = (bet as f64).ln();

        // Compute fraction in log space
        let fraction = (log_bet - log_min) / (log_max - log_min);

        // Map to raise index range
        let num_slots = (ACTION_IDX_RAISE_MAX - ACTION_IDX_RAISE_MIN) as f64;
        let index = ACTION_IDX_RAISE_MIN + (fraction * num_slots).round() as usize;

        // Clamp to valid range
        index.clamp(ACTION_IDX_RAISE_MIN, ACTION_IDX_RAISE_MAX)
    }

    /// Map an index back to an approximate bet amount.
    ///
    /// This is the inverse of `bet_to_index` and is useful for debugging
    /// or for generating representative bet amounts for each index. Interior
    /// amounts round to the nearest cent with ties up. Endpoints and the final
    /// bounds use integer comparisons, preserving them above f64's exact range.
    pub fn index_to_bet(&self, index: usize) -> Option<Chips> {
        match index {
            ACTION_IDX_FOLD | ACTION_IDX_CALL | ACTION_IDX_ALL_IN => None,
            idx if (ACTION_IDX_RAISE_MIN..=ACTION_IDX_RAISE_MAX).contains(&idx) => {
                let min_bet = self.config.min_bet;
                let max_bet = self.config.max_bet;

                if idx == ACTION_IDX_RAISE_MIN {
                    return Some(min_bet);
                }
                if idx == ACTION_IDX_RAISE_MAX {
                    return Some(max_bet);
                }

                let log_min = (min_bet as f64).ln();
                let log_max = (max_bet as f64).ln();

                // Compute fraction from index
                let num_slots = (ACTION_IDX_RAISE_MAX - ACTION_IDX_RAISE_MIN) as f64;
                let fraction = (idx - ACTION_IDX_RAISE_MIN) as f64 / num_slots;

                // Convert back from log space
                let log_bet = log_min + fraction * (log_max - log_min);
                let representative = log_bet.exp().round() as Chips;
                Some(representative.clamp(min_bet, max_bet))
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::GameStateBuilder;

    fn create_test_game_state() -> GameState {
        GameStateBuilder::new()
            .num_players_with_stack(2, 100)
            .blinds(10, 5)
            .build()
            .unwrap()
    }

    fn create_mapper() -> ActionIndexMapper {
        ActionIndexMapper::new(ActionIndexMapperConfig::new(10, 100))
    }

    // === Basic action mapping tests ===

    #[test]
    fn test_fold_always_maps_to_zero() {
        let mapper = create_mapper();
        let game_state = create_test_game_state();

        assert_eq!(
            mapper.action_to_idx(&AgentAction::Fold, &game_state),
            ACTION_IDX_FOLD
        );
    }

    #[test]
    fn test_call_always_maps_to_one() {
        let mapper = create_mapper();
        let game_state = create_test_game_state();

        assert_eq!(
            mapper.action_to_idx(&AgentAction::Call, &game_state),
            ACTION_IDX_CALL
        );
    }

    #[test]
    fn test_all_in_always_maps_to_all_in_idx() {
        let mapper = create_mapper();
        let game_state = create_test_game_state();

        assert_eq!(
            mapper.action_to_idx(&AgentAction::AllIn, &game_state),
            ACTION_IDX_ALL_IN
        );
    }

    #[test]
    fn test_bet_matching_current_bet_maps_to_call() {
        let mapper = create_mapper();
        let game_state = create_test_game_state();

        let current_bet = game_state.current_round_bet();
        assert_eq!(
            mapper.action_to_idx(&AgentAction::Bet(current_bet), &game_state),
            ACTION_IDX_CALL
        );
    }

    #[test]
    fn test_bet_matching_all_in_maps_to_all_in_idx() {
        let mapper = create_mapper();
        let game_state = create_test_game_state();

        let all_in_amount =
            game_state.current_round_current_player_bet() + game_state.current_player_stack();
        assert_eq!(
            mapper.action_to_idx(&AgentAction::Bet(all_in_amount), &game_state),
            ACTION_IDX_ALL_IN
        );
    }

    // === Raise mapping tests ===

    #[test]
    fn test_raises_spread_across_indices() {
        let mapper = ActionIndexMapper::new(ActionIndexMapperConfig::new(10, 1000));
        let game_state = create_test_game_state();

        // Small raise should map to lower index
        let small_raise_idx = mapper.action_to_idx(&AgentAction::Bet(20), &game_state);
        assert!((ACTION_IDX_RAISE_MIN..=ACTION_IDX_RAISE_MAX).contains(&small_raise_idx));

        // Large raise should map to higher index
        let large_raise_idx = mapper.action_to_idx(&AgentAction::Bet(500), &game_state);
        assert!((ACTION_IDX_RAISE_MIN..=ACTION_IDX_RAISE_MAX).contains(&large_raise_idx));

        // Large raise should have higher index than small raise
        assert!(large_raise_idx > small_raise_idx);
    }

    #[test]
    fn test_min_bet_maps_to_raise_min() {
        let mapper = ActionIndexMapper::new(ActionIndexMapperConfig::new(10, 100));
        let game_state = create_test_game_state();

        let idx = mapper.action_to_idx(&AgentAction::Bet(10), &game_state);
        assert_eq!(idx, ACTION_IDX_RAISE_MIN);
    }

    #[test]
    fn test_bet_below_min_maps_to_raise_min() {
        let mapper = ActionIndexMapper::new(ActionIndexMapperConfig::new(10, 100));
        let game_state = create_test_game_state();

        let idx = mapper.action_to_idx(&AgentAction::Bet(5), &game_state);
        assert_eq!(idx, ACTION_IDX_RAISE_MIN);
    }

    #[test]
    fn test_bet_at_max_maps_to_raise_max() {
        // Use a custom game state where 100 is not the all-in amount
        let game_state = GameStateBuilder::new()
            .num_players_with_stack(2, 200)
            .blinds(10, 5)
            .build()
            .unwrap();

        let mapper = ActionIndexMapper::new(ActionIndexMapperConfig::new(10, 100));

        let idx = mapper.action_to_idx(&AgentAction::Bet(100), &game_state);
        assert_eq!(idx, ACTION_IDX_RAISE_MAX);
    }

    #[test]
    fn test_bet_above_max_maps_to_raise_max() {
        // Use a custom game state where 150 is not the all-in amount
        let game_state = GameStateBuilder::new()
            .num_players_with_stack(2, 200)
            .blinds(10, 5)
            .build()
            .unwrap();

        let mapper = ActionIndexMapper::new(ActionIndexMapperConfig::new(10, 100));

        let idx = mapper.action_to_idx(&AgentAction::Bet(150), &game_state);
        // This should map to raise_max since 150 > 100 but < all-in (200)
        assert_eq!(idx, ACTION_IDX_RAISE_MAX);
    }

    // === Logarithmic distribution tests ===

    #[test]
    fn test_log_distribution_midpoint() {
        let mapper = ActionIndexMapper::new(ActionIndexMapperConfig::new(10, 1000));

        // Geometric mean of 10 and 1000 is sqrt(10 * 1000) = 100
        let midpoint_idx = mapper.bet_to_index(100);

        // Should be roughly in the middle of the range (index 26)
        let mid_idx = (ACTION_IDX_RAISE_MIN + ACTION_IDX_RAISE_MAX) / 2;
        assert!(
            (midpoint_idx as i32 - mid_idx as i32).abs() <= 1,
            "Geometric mean should map to middle index, got {} expected ~{}",
            midpoint_idx,
            mid_idx
        );
    }

    #[test]
    fn test_index_to_bet_roundtrip() {
        let mapper = ActionIndexMapper::new(ActionIndexMapperConfig::new(10, 1000));

        // Test that index_to_bet is a reasonable inverse of bet_to_index
        for idx in ACTION_IDX_RAISE_MIN..=ACTION_IDX_RAISE_MAX {
            let bet = mapper.index_to_bet(idx).unwrap();
            let recovered_idx = mapper.bet_to_index(bet);

            // Should round-trip to same index
            assert_eq!(
                idx, recovered_idx,
                "Index {} -> bet {} -> index {}",
                idx, bet, recovered_idx
            );
        }
    }

    #[test]
    fn representative_bets_preserve_large_integer_bounds() {
        for (min_bet, max_bet) in [
            (9_007_199_254_740_993, 9_007_199_254_741_101),
            (Chips::MAX - 1_000, Chips::MAX - 1),
        ] {
            let mapper = ActionIndexMapper::new(ActionIndexMapperConfig::new(min_bet, max_bet));
            assert_eq!(mapper.index_to_bet(ACTION_IDX_RAISE_MIN), Some(min_bet));
            assert_eq!(mapper.index_to_bet(ACTION_IDX_RAISE_MAX), Some(max_bet));
            for idx in ACTION_IDX_RAISE_MIN..=ACTION_IDX_RAISE_MAX {
                let bet = mapper.index_to_bet(idx).unwrap();
                assert!((min_bet..=max_bet).contains(&bet));
            }
        }
    }

    #[test]
    fn effective_range_doubling_stays_within_the_ledger() {
        let game_state = GameStateBuilder::new()
            .stacks([1, 1])
            .blinds(Chips::MAX, Chips::MAX / 2)
            .build()
            .unwrap();
        assert_eq!(compute_effective_range(&game_state), (Chips::MAX, Chips::MAX));
    }

    // === Configuration tests ===

    #[test]
    fn test_compute_effective_range() {
        let game_state = GameStateBuilder::new()
            .stacks(vec![100, 200, 150])
            .blinds(10, 5)
            .build()
            .unwrap();

        let (min_bet, max_bet) = compute_effective_range(&game_state);

        // min_bet should be big blind
        assert_eq!(min_bet, 10);

        // max_bet should be second largest stack (150)
        assert_eq!(max_bet, 150);
    }

    #[test]
    fn test_compute_effective_range_two_players() {
        let game_state = GameStateBuilder::new()
            .stacks(vec![100, 200])
            .blinds(10, 5)
            .build()
            .unwrap();

        let (min_bet, max_bet) = compute_effective_range(&game_state);

        assert_eq!(min_bet, 10);
        // Second largest is 100
        assert_eq!(max_bet, 100);
    }

    #[test]
    fn test_config_from_game_state() {
        let game_state = GameStateBuilder::new()
            .stacks(vec![100, 200])
            .blinds(10, 5)
            .build()
            .unwrap();

        let config = ActionIndexMapperConfig::from_game_state(&game_state);

        assert_eq!(config.min_bet, 10);
        assert_eq!(config.max_bet, 100);
    }

    #[test]
    fn test_mapper_from_game_state() {
        let game_state = GameStateBuilder::new()
            .stacks(vec![100, 200])
            .blinds(10, 5)
            .build()
            .unwrap();

        let mapper = ActionIndexMapper::from_game_state(&game_state);

        assert_eq!(mapper.config().min_bet, 10);
        assert_eq!(mapper.config().max_bet, 100);
    }

    // === Edge case tests ===

    #[test]
    fn test_small_pot_edge_case() {
        // Very small pot where all bets might cluster at low indices
        let game_state = GameStateBuilder::new()
            .stacks(vec![10, 10])
            .blinds(2, 1)
            .build()
            .unwrap();

        let mapper = ActionIndexMapper::from_game_state(&game_state);

        // Should still produce valid indices
        let idx = mapper.action_to_idx(&AgentAction::Bet(5), &game_state);
        assert!((ACTION_IDX_RAISE_MIN..=ACTION_IDX_RAISE_MAX).contains(&idx));
    }

    #[test]
    fn test_all_in_close_to_min_raise() {
        // Edge case where all-in is very close to min raise
        let game_state = GameStateBuilder::new()
            .stacks(vec![12, 100])
            .blinds(10, 5)
            .build()
            .unwrap();

        let mapper = ActionIndexMapper::from_game_state(&game_state);

        // All-in should still map to all-in index
        assert_eq!(
            mapper.action_to_idx(&AgentAction::AllIn, &game_state),
            ACTION_IDX_ALL_IN
        );

        // Bet exactly equal to all-in amount should also map to all-in
        let all_in_amount =
            game_state.current_round_current_player_bet() + game_state.current_player_stack();
        assert_eq!(
            mapper.action_to_idx(&AgentAction::Bet(all_in_amount), &game_state),
            ACTION_IDX_ALL_IN
        );
    }

    #[test]
    fn test_index_to_bet_returns_none_for_special_indices() {
        let mapper = create_mapper();

        assert!(mapper.index_to_bet(ACTION_IDX_FOLD).is_none());
        assert!(mapper.index_to_bet(ACTION_IDX_CALL).is_none());
        assert!(mapper.index_to_bet(ACTION_IDX_ALL_IN).is_none());
    }

    #[test]
    fn test_index_to_bet_returns_none_for_out_of_range() {
        let mapper = create_mapper();

        assert!(mapper.index_to_bet(16).is_none());
        assert!(mapper.index_to_bet(100).is_none());
    }

    #[test]
    fn test_num_action_indices_constant() {
        assert_eq!(NUM_ACTION_INDICES, 16);
    }
}
