//! OHH Converter - Converts arena Actions and GameState to Open Hand History format
//!
//! This module provides the core structure for converting rs-poker arena simulations
//! into the standardized Open Hand History (OHH) JSON format (v1.4.7).

use std::collections::HashMap;

use crate::Chips;
use chrono::{DateTime, Utc};

use crate::{
    arena::errors::OHHConversionError,
    arena::game_state::GameState,
    core::Card,
    open_hand_history::{
        ActionObj, BetLimitObj, BetType, GameType, HandHistory, PlayerObj, PlayerWinsObj, PotObj,
        RoundObj,
    },
};

/// Configuration for the OHH converter
#[derive(Debug, Clone)]
pub struct ConverterConfig {
    pub site_name: String,
    pub network_name: String,
    pub currency: String,
}

impl Default for ConverterConfig {
    fn default() -> Self {
        Self {
            site_name: "rs_poker".to_string(),
            network_name: "rs_poker_arena".to_string(),
            currency: "USD".to_string(),
        }
    }
}

#[derive(Debug)]
pub struct HandHistoryBuilder {
    // Configuration
    game_id: Option<u128>,
    site_name: String,
    network_name: String,
    currency: String,

    // Game setup (from initial GameState)
    dealer_idx: usize,
    table_size: usize,
    big_blind: Chips,
    small_blind: Chips,
    ante: Chips,
    start_time: DateTime<Utc>,

    // Players tracking
    players: Vec<PlayerObj>,
    player_cards: HashMap<usize, Vec<Card>>, // Track dealt cards per player
    player_stacks: Vec<Chips>,

    // Rounds tracking
    rounds: Vec<RoundObj>,
    current_round_id: u64,
    current_street: Option<String>,
    current_round_actions: Vec<ActionObj>,
    current_round_cards: Vec<Card>, // Community cards for this round
    total_board_cards: usize,

    // Action numbering (resets each round)
    action_number: u64,

    // Betting round tracking
    round_bet_state: RoundBetState,

    // Pots (calculated at end)
    pots: Vec<PotObj>,
    pending_pot_expected_total: Option<Chips>,
    pending_pot_awarded: Chips,
    pending_pot_rake: Chips,
    observed_rake_collected: Chips,
    pending_pot_player_wins: Vec<PlayerWinsObj>,
}

#[derive(Debug, Default)]
struct RoundBetState {
    contributions: Vec<Chips>,
    current_max: Chips,
}

impl RoundBetState {
    fn reset(&mut self) {
        self.contributions.clear();
        self.current_max = 0;
    }

    fn ensure_capacity(&mut self, idx: usize) {
        if self.contributions.len() <= idx {
            self.contributions.resize(idx + 1, 0);
        }
    }

    fn committed(&self, idx: usize) -> Chips {
        self.contributions.get(idx).copied().unwrap_or(0)
    }

    #[cfg(test)]
    fn current_max(&self) -> Chips {
        self.current_max
    }

    fn record(&mut self, idx: usize, amount: Chips) -> Chips {
        if (amount == 0) || amount < 0 {
            return self.committed(idx);
        }
        self.ensure_capacity(idx);
        self.contributions[idx] += amount;
        if self.contributions[idx] > self.current_max {
            self.current_max = self.contributions[idx];
        }
        self.contributions[idx]
    }
}

impl HandHistoryBuilder {
    /// Creates a new HandHistoryBuilder with the given configuration
    pub fn new(config: ConverterConfig) -> Self {
        Self {
            game_id: None,
            site_name: config.site_name,
            network_name: config.network_name,
            currency: config.currency,

            // These will be set when init_from_game_state is called
            dealer_idx: 0,
            table_size: 0,
            big_blind: 0,
            small_blind: 0,
            ante: 0,
            start_time: Utc::now(),

            players: Vec::new(),
            player_cards: HashMap::new(),
            player_stacks: Vec::new(),

            rounds: Vec::new(),
            current_round_id: 0,
            current_street: None,
            current_round_actions: Vec::new(),
            current_round_cards: Vec::new(),
            total_board_cards: 0,

            action_number: 1,

            round_bet_state: RoundBetState::default(),

            pots: Vec::new(),
            pending_pot_expected_total: None,
            pending_pot_awarded: 0,
            pending_pot_rake: 0,
            observed_rake_collected: 0,
            pending_pot_player_wins: Vec::new(),
        }
    }

    /// Initialize the builder from the first GameState
    /// This sets the game_id and initializes game parameters from the GameState
    pub fn init_from_game_state(&mut self, game_id: u128, game_state: &GameState) {
        self.game_id = Some(game_id);
        self.start_time = Utc::now();

        // Extract game parameters from GameState
        self.dealer_idx = game_state.dealer_idx;
        self.table_size = game_state.stacks.len();
        self.big_blind = game_state.big_blind;
        self.small_blind = game_state.small_blind;
        self.ante = game_state.ante;

        // Initialize players from GameState
        self.players = game_state
            .stacks
            .iter()
            .enumerate()
            .map(|(idx, &stack)| {
                let is_sitting_out = stack == 0;
                PlayerObj {
                    id: idx as u64,
                    seat: idx as u64 + 1, // Seats are 1-indexed
                    name: format!("Player{}", idx + 1),
                    display: None,
                    starting_stack: stack,
                    player_bounty: None,
                    is_sitting_out: Some(is_sitting_out),
                }
            })
            .collect();
        self.player_stacks = game_state.stacks.to_vec();
    }

    /// Start a new round with the given street name
    pub fn start_round(&mut self, street: String) {
        // Finish the current round if one is in progress
        self.finish_current_round();

        // Start new round
        self.current_round_id += 1;
        self.current_street = Some(street);
        self.current_round_actions.clear();
        self.current_round_cards.clear();
        self.action_number = 1; // OHH action numbers are 1-based within each round
        self.round_bet_state.reset();
    }

    fn is_betting_street_name(street: &str) -> bool {
        matches!(street, "Preflop" | "Flop" | "Turn" | "River")
    }

    fn is_in_betting_round(&self) -> bool {
        self.current_street
            .as_deref()
            .map(Self::is_betting_street_name)
            .unwrap_or(false)
    }

    /// Add an action to the current round
    pub fn add_action_to_round(&mut self, action: ActionObj) {
        self.current_round_actions.push(action);
    }

    /// Add a community card to the current round
    pub fn add_community_card(&mut self, card: Card) {
        self.current_round_cards.push(card);
    }

    fn register_contribution(&mut self, idx: usize, amount: Chips) {
        if self.is_in_betting_round() {
            self.round_bet_state.record(idx, amount);
        }
    }

    fn ensure_round_for_community_card(&mut self) -> Result<(), OHHConversionError> {
        let target_street = match self.total_board_cards {
            0..=2 => "Flop",
            3 => "Turn",
            4 => "River",
            _ => {
                return Err(OHHConversionError::InconsistentState(
                    "Holdem board cannot contain more than five cards".to_string(),
                ));
            }
        };

        if self.current_street.as_deref() != Some(target_street) {
            self.start_round(target_street.to_string());
        }

        Ok(())
    }

    /// Get the next action number and increment the counter
    pub fn next_action_number(&mut self) -> u64 {
        let current = self.action_number;
        self.action_number += 1;
        current
    }

    /// Finish the current round and move it to the rounds collection
    pub fn finish_current_round(&mut self) {
        if let Some(street) = &self.current_street {
            let round = RoundObj {
                id: self.current_round_id,
                street: street.clone(),
                cards: if self.current_round_cards.is_empty() {
                    None
                } else {
                    Some(self.current_round_cards.clone())
                },
                actions: self.current_round_actions.clone(),
            };

            self.rounds.push(round);
        }
    }

    fn push_played_payload(
        &mut self,
        payload: &crate::arena::action::PlayedActionPayload,
    ) -> Result<(), OHHConversionError> {
        let amount = self.calculate_action_amount(payload)?;
        let is_all_in = payload.player_stack == 0;

        let ohh_action = self.determine_ohh_action(payload, amount);
        self.register_contribution(payload.idx, amount);
        let action_obj = ActionObj {
            action_number: self.next_action_number(),
            player_id: payload.idx as u64,
            action: ohh_action,
            amount,
            is_allin: is_all_in,
            cards: None,
        };
        self.add_action_to_round(action_obj);
        Ok(())
    }

    fn determine_ohh_action(
        &self,
        payload: &crate::arena::action::PlayedActionPayload,
        amount: Chips,
    ) -> crate::open_hand_history::Action {
        use crate::arena::action::AgentAction;
        use crate::open_hand_history::Action;

        if matches!(payload.action, AgentAction::Fold) {
            return Action::Fold;
        }

        if amount == 0 {
            return Action::Check;
        }
        if payload.final_player_bet <= payload.starting_bet {
            return Action::Call;
        }
        if payload.round == crate::arena::game_state::Round::Preflop || payload.starting_bet > 0 {
            Action::Raise
        } else {
            Action::Bet
        }
    }

    fn ensure_pending_pot(&mut self, total_pot: Chips) {
        let needs_new = match self.pending_pot_expected_total {
            None => true,
            Some(expected) => {
                !(expected == total_pot)
                    || (self.pending_pot_awarded == expected)
                    || self.pending_pot_awarded >= expected
            }
        };

        if needs_new {
            self.flush_pending_pot();
            self.pending_pot_expected_total = Some(total_pot);
        }
    }

    fn flush_pending_pot(&mut self) {
        if let Some(expected) = self.pending_pot_expected_total.take() {
            if !self.pending_pot_player_wins.is_empty() {
                let pot = PotObj {
                    number: self.pots.len() as u64,
                    amount: expected,
                    rake: if self.pending_pot_rake > 0 {
                        Some(self.pending_pot_rake)
                    } else {
                        None
                    },
                    jackpot: None,
                    player_wins: std::mem::take(&mut self.pending_pot_player_wins),
                };
                self.pots.push(pot);
            } else {
                self.pending_pot_player_wins.clear();
            }
        }
        self.pending_pot_awarded = 0;
        self.pending_pot_rake = 0;
    }

    /// Records an arena action and converts it to OHH format
    pub fn record_action(
        &mut self,
        game_id: u128,
        action: &crate::arena::action::Action,
        game_state: &GameState,
    ) -> Result<(), OHHConversionError> {
        // Initialize builder on first action
        if self.game_id.is_none() {
            self.init_from_game_state(game_id, game_state);
        }

        if self.game_id != Some(game_id) {
            return Err(OHHConversionError::InconsistentState(
                "actions from different game IDs".into(),
            ));
        }
        match action {
            crate::arena::action::Action::GameStart(payload) => {
                self.big_blind = payload.big_blind;
                self.small_blind = payload.small_blind;
                self.ante = payload.ante;
            }

            crate::arena::action::Action::PlayerSit(payload) => {
                if let Some(name) = &payload.name
                    && let Some(player) = self.players.get_mut(payload.idx)
                {
                    player.name = name.clone();
                    if player.display.is_none() {
                        player.display = Some(name.clone());
                    }
                }
            }

            crate::arena::action::Action::DealStartingHand(payload) => {
                // Store card for this player
                self.player_cards
                    .entry(payload.idx)
                    .or_default()
                    .push(payload.card);

                // If this completes dealing to all players (2 cards each in hold'em)
                if self.player_cards.len()
                    == self
                        .players
                        .iter()
                        .filter(|p| p.is_sitting_out != Some(true))
                        .count()
                    && self.player_cards.values().all(|cards| cards.len() == 2)
                {
                    // Collect the actions to add (to avoid borrow checker issues)
                    // Sort by player index to ensure consistent ordering
                    let mut player_cards_sorted: Vec<_> = self.player_cards.iter().collect();
                    player_cards_sorted.sort_by_key(|(player_idx, _)| *player_idx);

                    let actions_to_add: Vec<ActionObj> = player_cards_sorted
                        .iter()
                        .map(|(player_idx, cards)| ActionObj {
                            action_number: 0, // Will be set when adding to round
                            player_id: **player_idx as u64,
                            action: crate::open_hand_history::Action::DealtCards,
                            amount: 0,
                            is_allin: false,
                            cards: Some((*cards).clone()),
                        })
                        .collect();

                    // Add "Dealt Cards" actions for all players
                    for mut action_obj in actions_to_add {
                        action_obj.action_number = self.next_action_number();
                        self.add_action_to_round(action_obj);
                    }
                }
            }

            crate::arena::action::Action::RoundAdvance(round) => {
                if let Some(street) = Self::map_arena_round_to_street(round) {
                    let is_new_round = self
                        .current_street
                        .as_ref()
                        .map(|current| current != &street)
                        .unwrap_or(true);
                    if is_new_round {
                        self.start_round(street);
                    }
                }
            }

            crate::arena::action::Action::PlayedAction(payload) => {
                self.push_played_payload(payload)?;
            }

            crate::arena::action::Action::FailedAction(payload) => {
                // Convert the corrected result action instead of the failed attempt
                self.push_played_payload(&payload.result)?;
            }

            crate::arena::action::Action::ForcedBet(payload) => {
                if self.current_street.is_none() {
                    // Antes occur before any betting street is officially started.
                    // Attach them to the preflop round so they are preserved.
                    self.start_round("Preflop".to_string());
                }
                let (ohh_action, _configured_amount) = match payload.forced_bet_type {
                    crate::arena::action::ForcedBetType::Ante => {
                        (crate::open_hand_history::Action::PostAnte, self.ante)
                    }
                    crate::arena::action::ForcedBetType::SmallBlind => (
                        crate::open_hand_history::Action::PostSmallBlind,
                        self.small_blind,
                    ),
                    crate::arena::action::ForcedBetType::BigBlind => (
                        crate::open_hand_history::Action::PostBigBlind,
                        self.big_blind,
                    ),
                };

                let player_idx = payload.idx;
                let previous = self.player_stacks.get(player_idx).copied().ok_or_else(|| {
                    OHHConversionError::MissingData("unknown forced-bet player".into())
                })?;
                let amount = payload.bet;
                if amount < 0 || amount > previous || payload.player_stack != previous - amount {
                    return Err(OHHConversionError::InconsistentState(
                        "forced bet/stack disagreement".into(),
                    ));
                }
                self.player_stacks[player_idx] = payload.player_stack;
                let is_allin = payload.player_stack == 0;

                let action_obj = ActionObj {
                    action_number: self.next_action_number(),
                    player_id: payload.idx as u64,
                    action: ohh_action,
                    amount,
                    is_allin,
                    cards: None,
                };
                self.register_contribution(payload.idx, amount);
                self.add_action_to_round(action_obj);
            }

            crate::arena::action::Action::DealCommunity(card) => {
                self.ensure_round_for_community_card()?;
                self.add_community_card(*card);
                self.total_board_cards += 1;
            }

            crate::arena::action::Action::ReturnUncalledBet(payload) => {
                let stack = self.player_stacks.get_mut(payload.idx).ok_or_else(|| {
                    OHHConversionError::MissingData("unknown refund player".into())
                })?;
                if payload.amount <= 0 {
                    return Err(OHHConversionError::InconsistentState(
                        "nonpositive refund".into(),
                    ));
                }
                *stack = stack.checked_add(payload.amount).ok_or_else(|| {
                    OHHConversionError::InconsistentState("refund overflow".into())
                })?;
                // OHH's published action vocabulary has no return-bet action.
                // Returns are inferred from contributions and gross pots by replay;
                // they must never be serialized as winnings or external added chips.
            }

            crate::arena::action::Action::Award(payload) => {
                if payload.award_amount < 0
                    || payload.total_pot < 0
                    || payload.award_amount > payload.total_pot
                {
                    return Err(OHHConversionError::InconsistentState(
                        "invalid pot award".into(),
                    ));
                }
                let stack = self.player_stacks.get_mut(payload.idx).ok_or_else(|| {
                    OHHConversionError::MissingData("unknown awarded player".into())
                })?;
                *stack = stack.checked_add(payload.award_amount).ok_or_else(|| {
                    OHHConversionError::InconsistentState("award overflow".into())
                })?;
                self.ensure_pending_pot(payload.total_pot);

                // Rake is applied once before a pot is awarded. Track the delta
                // of GameState's cumulative rake so ties see the same pot rake
                // without double-counting it. This also lets equal-sized adjacent
                // side pots flush independently.
                let rake_delta = game_state
                    .rake_collected
                    .checked_sub(self.observed_rake_collected)
                    .filter(|&delta| delta >= 0)
                    .ok_or_else(|| {
                        OHHConversionError::InconsistentState("cumulative rake decreased".into())
                    })?;
                if rake_delta > 0 {
                    self.pending_pot_rake += rake_delta;
                    self.observed_rake_collected = game_state.rake_collected;
                }

                let win = PlayerWinsObj {
                    player_id: payload.idx as u64,
                    win_amount: payload.award_amount,
                    cashout_amount: None,
                    cashout_fee: None,
                    bonus_amount: None,
                    contributed_rake: None,
                };

                self.pending_pot_player_wins.push(win);
                self.pending_pot_awarded += payload.award_amount;

                if let Some(expected) = self.pending_pot_expected_total
                    && self.pending_pot_awarded + self.pending_pot_rake >= expected
                {
                    self.flush_pending_pot();
                }

                // If showing cards at showdown, add ShowsCards action
                if let (Some(_hand), Some(_rank)) = (payload.hand, payload.rank)
                    && let Some(player_cards) = self.player_cards.get(&payload.idx).cloned()
                {
                    let action_obj = ActionObj {
                        action_number: self.next_action_number(),
                        player_id: payload.idx as u64,
                        action: crate::open_hand_history::Action::ShowsCards,
                        amount: 0,
                        is_allin: false,
                        cards: Some(player_cards),
                    };
                    self.add_action_to_round(action_obj);
                }
            }
        }

        Ok(())
    }

    /// Calculate the amount for this action (difference from starting to final bet)
    fn calculate_action_amount(
        &mut self,
        payload: &crate::arena::action::PlayedActionPayload,
    ) -> Result<Chips, OHHConversionError> {
        let previous = self
            .player_stacks
            .get(payload.idx)
            .copied()
            .ok_or_else(|| OHHConversionError::MissingData("unknown acting player".into()))?;
        let amount = payload
            .final_player_bet
            .checked_sub(payload.starting_player_bet)
            .filter(|&amount| amount >= 0 && amount <= previous)
            .ok_or_else(|| {
                OHHConversionError::InconsistentState("invalid exact bet delta".into())
            })?;
        if payload.player_stack != previous - amount {
            return Err(OHHConversionError::InconsistentState(
                "bet delta/remaining stack disagreement".into(),
            ));
        }
        self.player_stacks[payload.idx] = payload.player_stack;
        Ok(amount)
    }

    /// Map arena Round to OHH street name
    fn map_arena_round_to_street(round: &crate::arena::game_state::Round) -> Option<String> {
        use crate::arena::game_state::Round;
        match round {
            Round::Ante | Round::Preflop => Some("Preflop".to_string()),
            Round::Flop => Some("Flop".to_string()),
            Round::Turn => Some("Turn".to_string()),
            Round::River => Some("River".to_string()),
            Round::Showdown => Some("Showdown".to_string()),
            Round::DealPreflop
            | Round::DealFlop
            | Round::DealTurn
            | Round::DealRiver
            | Round::Starting
            | Round::Complete => None,
        }
    }

    /// Build the final HandHistory from accumulated state
    pub fn build(mut self) -> Result<HandHistory, OHHConversionError> {
        // Finish any open round
        self.finish_current_round();
        self.flush_pending_pot();

        let game_id = self.game_id.ok_or(OHHConversionError::NotInitialized)?;
        for pot in &self.pots {
            let wins = pot
                .player_wins
                .iter()
                .try_fold(0_i64, |sum, win| sum.checked_add(win.win_amount))
                .ok_or_else(|| {
                    OHHConversionError::InconsistentState("pot winnings overflow".into())
                })?;
            if wins.checked_add(pot.rake.unwrap_or(0)) != Some(pot.amount) {
                return Err(OHHConversionError::InconsistentState(
                    "gross pot differs from winnings plus rake".into(),
                ));
            }
        }

        Ok(HandHistory {
            spec_version: "1.4.7".to_string(),
            site_name: self.site_name,
            network_name: self.network_name,
            internal_version: env!("CARGO_PKG_VERSION").to_string(),
            tournament: false,
            tournament_info: None,
            game_number: game_id.to_string(),
            start_date_utc: Some(self.start_time),
            table_name: format!("Table {}", game_id),
            table_handle: None,
            table_skin: None,
            game_type: GameType::Holdem,
            bet_limit: Some(BetLimitObj {
                bet_type: BetType::NoLimit,
                bet_cap: 0,
            }),
            table_size: self.table_size as u64,
            currency: self.currency,
            dealer_seat: self.dealer_idx as u64 + 1, // Convert to 1-indexed
            small_blind_amount: self.small_blind,
            big_blind_amount: self.big_blind,
            ante_amount: self.ante,
            hero_player_id: None, // No hero in simulations
            players: self.players,
            rounds: self.rounds,
            pots: self.pots,
            tournament_bounties: None,
        })
    }
}

#[cfg(all(test, feature = "arena"))]
mod tests {
    use super::*;
    use crate::arena::GameStateBuilder;
    use crate::arena::action::{AgentAction, ForcedBetPayload, ForcedBetType, PlayedActionPayload};
    use crate::arena::game_state::Round;
    use crate::core::PlayerBitSet;

    fn create_test_game_state() -> GameState {
        use crate::arena::game_state::RoundData;
        use crate::core::Hand;

        let stacks = vec![1000, 1500, 800];
        let hands = vec![Hand::default(), Hand::default(), Hand::default()];
        let player_bet = vec![0, 0, 0];
        let mut player_active = PlayerBitSet::new(3);
        player_active.enable(0);
        player_active.enable(1);
        player_active.enable(2);

        let round_data = RoundData::new(3, 2, player_active, 0);

        GameStateBuilder::new()
            .round(Round::Starting)
            .round_data(round_data)
            .hands(hands)
            .stacks(stacks)
            .player_bet(player_bet)
            .big_blind(2)
            .small_blind(1)
            .dealer_idx(1)
            .build()
            .unwrap()
    }

    fn record_forced_bet_action(
        builder: &mut HandHistoryBuilder,
        game_id: u128,
        game_state: &GameState,
        remaining: &mut [Chips],
        idx: usize,
        amount: Chips,
        bet_type: ForcedBetType,
    ) {
        remaining[idx] -= amount;
        let payload = ForcedBetPayload {
            bet: amount,
            player_stack: remaining[idx],
            idx,
            forced_bet_type: bet_type,
        };
        builder
            .record_action(
                game_id,
                &crate::arena::action::Action::ForcedBet(payload),
                game_state,
            )
            .unwrap();
    }

    fn active_players(num_players: usize) -> PlayerBitSet {
        let mut bitset = PlayerBitSet::new(num_players);
        for idx in 0..num_players {
            bitset.enable(idx);
        }
        bitset
    }

    struct MultiPlayerActionParams {
        agent_action: AgentAction,
        idx: usize,
        round: Round,
        player_stack: Chips,
        starting_player_bet: Chips,
        final_player_bet: Chips,
        starting_bet: Chips,
        final_bet: Chips,
        num_players: usize,
    }

    fn create_multi_player_action(params: MultiPlayerActionParams) -> crate::arena::action::Action {
        let MultiPlayerActionParams {
            agent_action,
            idx,
            round,
            player_stack,
            starting_player_bet,
            final_player_bet,
            starting_bet,
            final_bet,
            num_players,
        } = params;

        crate::arena::action::Action::PlayedAction(PlayedActionPayload {
            action: agent_action,
            idx,
            round,
            player_stack,
            starting_pot: 0,
            final_pot: 0,
            starting_bet,
            final_bet,
            starting_min_raise: 2,
            final_min_raise: 2,
            starting_player_bet,
            final_player_bet,
            players_active: active_players(num_players),
            players_all_in: PlayerBitSet::new(num_players),
        })
    }

    #[test]
    fn test_builder_new_with_default_config() {
        let builder = HandHistoryBuilder::new(ConverterConfig::default());

        assert_eq!(builder.site_name, "rs_poker");
        assert_eq!(builder.network_name, "rs_poker_arena");
        assert_eq!(builder.currency, "USD");
        assert!(builder.game_id.is_none());
        assert_eq!(builder.action_number, 1);
    }

    #[test]
    fn test_builder_new_with_custom_config() {
        let config = ConverterConfig {
            site_name: "CustomSite".to_string(),
            network_name: "CustomNetwork".to_string(),
            currency: "EUR".to_string(),
        };
        let builder = HandHistoryBuilder::new(config);

        assert_eq!(builder.site_name, "CustomSite");
        assert_eq!(builder.network_name, "CustomNetwork");
        assert_eq!(builder.currency, "EUR");
    }

    #[test]
    fn test_init_from_game_state() {
        let mut builder = HandHistoryBuilder::new(ConverterConfig::default());
        let game_state = create_test_game_state();
        let game_id = 12345u128;

        builder.init_from_game_state(game_id, &game_state);

        assert_eq!(builder.game_id, Some(game_id));
        assert_eq!(builder.dealer_idx, 1);
        assert_eq!(builder.table_size, 3);
        assert_eq!(builder.players.len(), 3);

        // Check player setup
        assert_eq!(builder.players[0].id, 0);
        assert_eq!(builder.players[0].seat, 1);
        assert_eq!(builder.players[0].starting_stack, 1000);
        assert_eq!(builder.players[0].name, "Player1");

        assert_eq!(builder.players[1].id, 1);
        assert_eq!(builder.players[1].seat, 2);
        assert_eq!(builder.players[1].starting_stack, 1500);
    }

    #[test]
    fn test_build_without_init_fails() {
        let builder = HandHistoryBuilder::new(ConverterConfig::default());

        let result = builder.build();
        assert!(matches!(result, Err(OHHConversionError::NotInitialized)));
    }

    #[test]
    fn test_round_management() {
        let mut builder = HandHistoryBuilder::new(ConverterConfig::default());
        let game_state = create_test_game_state();
        builder.init_from_game_state(12345, &game_state);

        // Start first round
        builder.start_round("Preflop".to_string());
        assert_eq!(builder.current_round_id, 1);
        assert_eq!(builder.current_street, Some("Preflop".to_string()));
        assert_eq!(builder.action_number, 1);

        // Start second round
        builder.start_round("Flop".to_string());
        assert_eq!(builder.current_round_id, 2);
        assert_eq!(builder.current_street, Some("Flop".to_string()));
        assert_eq!(builder.action_number, 1); // Should reset
        assert_eq!(builder.rounds.len(), 1); // Previous round should be finished
        assert_eq!(builder.rounds[0].street, "Preflop");
    }

    #[test]
    fn test_action_numbering() {
        let mut builder = HandHistoryBuilder::new(ConverterConfig::default());

        assert_eq!(builder.next_action_number(), 1);
        assert_eq!(builder.next_action_number(), 2);
        assert_eq!(builder.next_action_number(), 3);

        // Starting new round resets numbering
        builder.start_round("Preflop".to_string());
        assert_eq!(builder.action_number, 1);
        assert_eq!(builder.next_action_number(), 1);
        assert_eq!(builder.next_action_number(), 2);
    }

    #[test]
    fn test_community_cards() {
        use crate::core::{Card, Suit, Value};

        let mut builder = HandHistoryBuilder::new(ConverterConfig::default());
        let game_state = create_test_game_state();
        builder.init_from_game_state(12345, &game_state);

        builder.start_round("Flop".to_string());

        let card1 = Card::new(Value::Ace, Suit::Spade);
        let card2 = Card::new(Value::King, Suit::Heart);

        builder.add_community_card(card1);
        builder.add_community_card(card2);

        assert_eq!(builder.current_round_cards.len(), 2);
        assert_eq!(builder.current_round_cards[0], card1);
        assert_eq!(builder.current_round_cards[1], card2);
    }

    #[test]
    fn test_build_success() {
        let mut builder = HandHistoryBuilder::new(ConverterConfig::default());
        let game_state = create_test_game_state();
        builder.init_from_game_state(12345, &game_state);

        builder.start_round("Preflop".to_string());

        let hand_history = builder.build().unwrap();

        assert_eq!(hand_history.spec_version, "1.4.7");
        assert_eq!(hand_history.game_number, "12345");
        assert_eq!(hand_history.table_size, 3);
        assert_eq!(hand_history.dealer_seat, 2); // dealer_idx 1 -> seat 2 (1-indexed)
        assert_eq!(hand_history.players.len(), 3);
        assert_eq!(hand_history.rounds.len(), 1);
        assert_eq!(hand_history.rounds[0].street, "Preflop");
        assert!(!hand_history.tournament);
        assert_eq!(hand_history.game_type, GameType::Holdem);
    }

    // Helper function for creating test actions
    fn create_game_start_action() -> crate::arena::action::Action {
        crate::arena::action::Action::GameStart(crate::arena::action::GameStartPayload {
            ante: 0,
            small_blind: 1,
            big_blind: 2,
        })
    }

    fn create_player_sit_action(
        idx: usize,
        stack: Chips,
        name: Option<&str>,
    ) -> crate::arena::action::Action {
        crate::arena::action::Action::PlayerSit(crate::arena::action::PlayerSitPayload {
            idx,
            player_stack: stack,
            name: name.map(|n| n.to_string()),
        })
    }

    fn create_deal_starting_hand_action(idx: usize, card: Card) -> crate::arena::action::Action {
        crate::arena::action::Action::DealStartingHand(
            crate::arena::action::DealStartingHandPayload { card, idx },
        )
    }

    fn create_forced_bet_action(
        idx: usize,
        amount: Chips,
        bet_type: crate::arena::action::ForcedBetType,
    ) -> crate::arena::action::Action {
        crate::arena::action::Action::ForcedBet(crate::arena::action::ForcedBetPayload {
            bet: amount,
            player_stack: [1000, 1500, 800][idx] - amount,
            idx,
            forced_bet_type: bet_type,
        })
    }

    fn create_played_action(
        agent_action: crate::arena::action::AgentAction,
        idx: usize,
        start_bet: Chips,
        final_bet: Chips,
        stack: Chips,
    ) -> crate::arena::action::Action {
        create_played_action_with_table(agent_action, idx, start_bet, final_bet, stack, 0, 0)
    }

    fn create_played_action_with_table(
        agent_action: crate::arena::action::AgentAction,
        idx: usize,
        start_bet: Chips,
        final_bet: Chips,
        stack: Chips,
        starting_table_bet: Chips,
        final_table_bet: Chips,
    ) -> crate::arena::action::Action {
        use crate::core::PlayerBitSet;
        let mut player_active = PlayerBitSet::new(3);
        player_active.enable(0);
        player_active.enable(1);
        player_active.enable(2);

        crate::arena::action::Action::PlayedAction(crate::arena::action::PlayedActionPayload {
            action: agent_action,
            idx,
            round: crate::arena::game_state::Round::Preflop,
            player_stack: stack,
            starting_pot: 0,
            final_pot: 0,
            starting_bet: starting_table_bet,
            final_bet: final_table_bet,
            starting_min_raise: 2,
            final_min_raise: 2,
            starting_player_bet: start_bet,
            final_player_bet: final_bet,
            players_active: player_active,
            players_all_in: PlayerBitSet::new(3),
        })
    }

    #[test]
    fn test_record_action_game_start() {
        let mut builder = HandHistoryBuilder::new(ConverterConfig::default());
        let game_state = create_test_game_state();
        let action = create_game_start_action();

        builder.record_action(12345, &action, &game_state).unwrap();

        assert_eq!(builder.big_blind, 2);
        assert_eq!(builder.small_blind, 1);
        assert_eq!(builder.ante, 0);
        assert_eq!(builder.game_id, Some(12345));
    }

    #[test]
    fn test_record_action_player_sit() {
        let mut builder = HandHistoryBuilder::new(ConverterConfig::default());
        let game_state = create_test_game_state();
        let action = create_player_sit_action(0, 1000, Some("HeroZero"));

        builder.record_action(12345, &action, &game_state).unwrap();

        assert_eq!(builder.game_id, Some(12345));
        assert_eq!(builder.players.len(), 3);
        assert_eq!(builder.players[0].name, "HeroZero");
        assert_eq!(builder.players[0].display.as_deref(), Some("HeroZero"));
    }

    #[test]
    fn test_record_action_deal_starting_hand() {
        use crate::core::{Card, Suit, Value};

        let mut builder = HandHistoryBuilder::new(ConverterConfig::default());
        let game_state = create_test_game_state();

        // Initialize first
        builder.init_from_game_state(12345, &game_state);
        builder.start_round("Preflop".to_string());

        // Deal 2 cards to each of 3 players (6 total cards)
        let cards = [
            (0, Card::new(Value::Ace, Suit::Spade)),
            (0, Card::new(Value::King, Suit::Heart)),
            (1, Card::new(Value::Queen, Suit::Diamond)),
            (1, Card::new(Value::Jack, Suit::Club)),
            (2, Card::new(Value::Ten, Suit::Spade)),
        ];

        // Deal first 4 cards (not complete yet)
        for (idx, card) in &cards[0..4] {
            let action = create_deal_starting_hand_action(*idx, *card);
            builder.record_action(12345, &action, &game_state).unwrap();
        }

        // Should not have any dealt cards actions yet
        assert_eq!(builder.current_round_actions.len(), 0);

        // Deal the 5th card - still not complete
        let action = create_deal_starting_hand_action(2, cards[4].1);
        builder.record_action(12345, &action, &game_state).unwrap();
        assert_eq!(builder.current_round_actions.len(), 0);

        // Deal the final card - should trigger dealt cards actions
        let final_card = Card::new(Value::Nine, Suit::Heart);
        let action = create_deal_starting_hand_action(2, final_card);
        builder.record_action(12345, &action, &game_state).unwrap();

        // Now should have 3 dealt cards actions
        assert_eq!(builder.current_round_actions.len(), 3);
        for i in 0..3 {
            assert_eq!(
                builder.current_round_actions[i].action,
                crate::open_hand_history::Action::DealtCards
            );
            assert_eq!(builder.current_round_actions[i].player_id, i as u64);
            assert_eq!(builder.current_round_actions[i].amount, 0);
            assert!(!builder.current_round_actions[i].is_allin);
            assert!(builder.current_round_actions[i].cards.is_some());
            assert_eq!(
                builder.current_round_actions[i]
                    .cards
                    .as_ref()
                    .unwrap()
                    .len(),
                2
            );
        }
    }

    #[test]
    fn test_record_action_round_advance() {
        let mut builder = HandHistoryBuilder::new(ConverterConfig::default());
        let game_state = create_test_game_state();
        builder.init_from_game_state(12345, &game_state);

        let action =
            crate::arena::action::Action::RoundAdvance(crate::arena::game_state::Round::Flop);
        builder.record_action(12345, &action, &game_state).unwrap();

        assert_eq!(builder.current_round_id, 1);
        assert_eq!(builder.current_street, Some("Flop".to_string()));

        // Test that Deal rounds are skipped
        let action =
            crate::arena::action::Action::RoundAdvance(crate::arena::game_state::Round::DealTurn);
        builder.record_action(12345, &action, &game_state).unwrap();

        // Should still be on Flop since DealTurn is skipped
        assert_eq!(builder.current_round_id, 1);
        assert_eq!(builder.current_street, Some("Flop".to_string()));
    }

    #[test]
    fn test_record_action_fold() {
        let mut builder = HandHistoryBuilder::new(ConverterConfig::default());
        let game_state = create_test_game_state();
        builder.init_from_game_state(12345, &game_state);
        builder.start_round("Preflop".to_string());

        let action = create_played_action(crate::arena::action::AgentAction::Fold, 0, 0, 0, 1000);
        builder.record_action(12345, &action, &game_state).unwrap();

        assert_eq!(builder.current_round_actions.len(), 1);
        assert_eq!(
            builder.current_round_actions[0].action,
            crate::open_hand_history::Action::Fold
        );
        assert_eq!(builder.current_round_actions[0].player_id, 0);
        assert_eq!(builder.current_round_actions[0].amount, 0);
        assert!(!builder.current_round_actions[0].is_allin);
    }

    #[test]
    fn test_record_action_call_as_check() {
        let mut builder = HandHistoryBuilder::new(ConverterConfig::default());
        let game_state = create_test_game_state();
        builder.init_from_game_state(12345, &game_state);
        builder.start_round("Preflop".to_string());

        // Call with 0 amount should be check
        let action = create_played_action(crate::arena::action::AgentAction::Call, 0, 0, 0, 1000);
        builder.record_action(12345, &action, &game_state).unwrap();

        assert_eq!(builder.current_round_actions.len(), 1);
        assert_eq!(
            builder.current_round_actions[0].action,
            crate::open_hand_history::Action::Check
        );
        assert_eq!(builder.current_round_actions[0].amount, 0);
    }

    #[test]
    fn test_record_action_call_with_amount() {
        let mut builder = HandHistoryBuilder::new(ConverterConfig::default());
        let game_state = create_test_game_state();
        builder.init_from_game_state(12345, &game_state);
        builder.start_round("Preflop".to_string());

        // Create an opening bet so the subsequent action is facing chips.
        let opening_bet =
            create_played_action(crate::arena::action::AgentAction::Bet(10), 1, 0, 10, 1490);
        builder
            .record_action(12345, &opening_bet, &game_state)
            .unwrap();

        // Call with amount should be call
        let action = create_played_action_with_table(
            crate::arena::action::AgentAction::Call,
            0,
            0,
            10,
            990,
            10,
            10,
        );
        builder.record_action(12345, &action, &game_state).unwrap();

        assert_eq!(builder.current_round_actions.len(), 2);
        let last_action = builder.current_round_actions.last().unwrap();
        assert_eq!(last_action.action, crate::open_hand_history::Action::Call);
        assert_eq!(last_action.amount, 10);
        assert!(!last_action.is_allin);
    }

    #[test]
    fn test_rejects_bet_delta_and_stack_disagreement() {
        let mut builder = HandHistoryBuilder::new(ConverterConfig::default());
        let game_state = create_test_game_state();
        builder.init_from_game_state(12345, &game_state);
        builder.start_round("Preflop".to_string());

        // Simulate a player with only 750 chips left whose table bet delta only reports 700.
        builder.player_stacks[0] = 750;
        let action = create_played_action_with_table(
            crate::arena::action::AgentAction::Call,
            0,
            0,
            700,
            0,
            0,
            700,
        );

        assert!(matches!(
            builder.record_action(12345, &action, &game_state),
            Err(OHHConversionError::InconsistentState(_))
        ));
        assert!(builder.current_round_actions.is_empty());
        assert_eq!(builder.player_stacks[0], 750);
    }

    #[test]
    fn test_record_action_bet_first_to_act() {
        let mut builder = HandHistoryBuilder::new(ConverterConfig::default());
        let game_state = create_test_game_state();
        builder.init_from_game_state(12345, &game_state);
        builder.start_round("Preflop".to_string());

        // OHH labels a preflop opening action as Raise.
        let action =
            create_played_action(crate::arena::action::AgentAction::Bet(50), 0, 0, 50, 950);
        builder.record_action(12345, &action, &game_state).unwrap();

        assert_eq!(builder.current_round_actions.len(), 1);
        assert_eq!(
            builder.current_round_actions[0].action,
            crate::open_hand_history::Action::Raise
        );
        assert_eq!(builder.current_round_actions[0].amount, 50);
    }

    #[test]
    fn test_short_all_in_is_classified_as_call() {
        use crate::arena::action::{AgentAction, PlayedActionPayload};
        use crate::open_hand_history::Action;

        let mut builder = HandHistoryBuilder::new(ConverterConfig::default());
        let game_state = create_test_game_state();
        builder.init_from_game_state(12345, &game_state);
        builder.start_round("Preflop".to_string());

        // Simulate blinds and an existing raise that sets the live bet.
        builder.round_bet_state.record(1, 1000);
        builder.round_bet_state.record(0, 400);

        let payload = PlayedActionPayload {
            action: AgentAction::AllIn,
            idx: 0,
            round: Round::Preflop,
            player_stack: 0,
            starting_pot: 0,
            final_pot: 0,
            starting_bet: 1000,
            final_bet: 1000,
            starting_min_raise: 2,
            final_min_raise: 2,
            starting_player_bet: 400,
            final_player_bet: 900,
            players_active: active_players(game_state.num_players),
            players_all_in: PlayerBitSet::new(game_state.num_players),
        };

        let classified = builder.determine_ohh_action(&payload, 500);
        assert_eq!(classified, Action::Call);
    }

    #[test]
    fn test_raise_when_creating_new_live_bet() {
        use crate::arena::action::{AgentAction, PlayedActionPayload};
        use crate::open_hand_history::Action;

        let mut builder = HandHistoryBuilder::new(ConverterConfig::default());
        let game_state = create_test_game_state();
        builder.init_from_game_state(12345, &game_state);
        builder.start_round("Flop".to_string());

        builder.round_bet_state.record(2, 150);
        builder.round_bet_state.record(0, 150);

        let payload = PlayedActionPayload {
            action: AgentAction::Bet(50),
            idx: 0,
            round: Round::Flop,
            player_stack: 0,
            starting_pot: 0,
            final_pot: 0,
            starting_bet: 150,
            final_bet: 350,
            starting_min_raise: 2,
            final_min_raise: 2,
            starting_player_bet: 150,
            final_player_bet: 350,
            players_active: active_players(game_state.num_players),
            players_all_in: PlayerBitSet::new(game_state.num_players),
        };

        let classified = builder.determine_ohh_action(&payload, 200);
        assert_eq!(classified, Action::Raise);
    }

    #[test]
    fn test_small_blind_completion_is_call() {
        use crate::arena::action::{
            AgentAction, ForcedBetType, GameStartPayload, PlayedActionPayload,
        };
        use crate::arena::game_state::Round;

        let sb = 315_632;
        let bb = 366_832;
        let ante = 78_908;
        let call_delta = bb - sb;
        let starting_stack = 100_000_000;
        let stacks = vec![starting_stack, starting_stack];
        let dealer_idx = 1;
        let game_state = GameStateBuilder::new()
            .stacks(stacks.clone())
            .blinds(bb, sb)
            .ante(ante)
            .dealer_idx(dealer_idx)
            .build()
            .unwrap();

        let mut builder = HandHistoryBuilder::new(ConverterConfig::default());
        let game_id = 4242u128;

        let game_start = crate::arena::action::Action::GameStart(GameStartPayload {
            ante,
            small_blind: sb,
            big_blind: bb,
        });
        builder
            .record_action(game_id, &game_start, &game_state)
            .unwrap();

        let mut remaining = stacks.clone();
        record_forced_bet_action(
            &mut builder,
            game_id,
            &game_state,
            &mut remaining,
            0,
            ante,
            ForcedBetType::Ante,
        );
        record_forced_bet_action(
            &mut builder,
            game_id,
            &game_state,
            &mut remaining,
            1,
            ante,
            ForcedBetType::Ante,
        );
        record_forced_bet_action(
            &mut builder,
            game_id,
            &game_state,
            &mut remaining,
            1,
            sb,
            ForcedBetType::SmallBlind,
        );
        record_forced_bet_action(
            &mut builder,
            game_id,
            &game_state,
            &mut remaining,
            0,
            bb,
            ForcedBetType::BigBlind,
        );

        let committed_sb = builder.round_bet_state.committed(1);
        let current_max = builder.round_bet_state.current_max();
        assert!((committed_sb - (ante + sb)) == 0);
        assert!((current_max - (ante + bb)) == 0);

        // Small blind completes to the big blind amount
        use crate::core::PlayerBitSet;
        let mut players_active = PlayerBitSet::new(2);
        players_active.enable(0);
        players_active.enable(1);

        let completion = crate::arena::action::Action::PlayedAction(PlayedActionPayload {
            action: AgentAction::Call,
            idx: 1,
            round: Round::Preflop,
            player_stack: remaining[1] - call_delta,
            starting_pot: 0,
            final_pot: 0,
            starting_bet: bb,
            final_bet: bb,
            starting_min_raise: bb * 2,
            final_min_raise: bb * 2,
            starting_player_bet: sb,
            final_player_bet: sb + call_delta,
            players_active,
            players_all_in: PlayerBitSet::new(2),
        });

        builder
            .record_action(game_id, &completion, &game_state)
            .unwrap();

        let last_action = builder.current_round_actions.last().unwrap();
        assert_eq!(last_action.action, crate::open_hand_history::Action::Call);
        assert_eq!(last_action.amount, call_delta);
    }

    #[test]
    fn test_small_blind_completion_matches_close_big_blind() {
        use crate::arena::action::{
            AgentAction, ForcedBetType, GameStartPayload, PlayedActionPayload,
        };
        use crate::arena::game_state::Round;

        let sb = 315_632;
        let bb = 315_676;
        let ante = 140_216;
        let call_delta = bb - sb;
        let starting_stack = 100_000_000;
        let stacks = vec![starting_stack, starting_stack];
        let dealer_idx = 0;
        let game_state = GameStateBuilder::new()
            .stacks(stacks.clone())
            .blinds(bb, sb)
            .ante(ante)
            .dealer_idx(dealer_idx)
            .build()
            .unwrap();

        let mut builder = HandHistoryBuilder::new(ConverterConfig::default());
        let game_id = 9001u128;

        let game_start = crate::arena::action::Action::GameStart(GameStartPayload {
            ante,
            small_blind: sb,
            big_blind: bb,
        });
        builder
            .record_action(game_id, &game_start, &game_state)
            .unwrap();

        let mut remaining = stacks.clone();
        record_forced_bet_action(
            &mut builder,
            game_id,
            &game_state,
            &mut remaining,
            0,
            ante,
            ForcedBetType::Ante,
        );
        record_forced_bet_action(
            &mut builder,
            game_id,
            &game_state,
            &mut remaining,
            1,
            ante,
            ForcedBetType::Ante,
        );
        record_forced_bet_action(
            &mut builder,
            game_id,
            &game_state,
            &mut remaining,
            0,
            sb,
            ForcedBetType::SmallBlind,
        );
        record_forced_bet_action(
            &mut builder,
            game_id,
            &game_state,
            &mut remaining,
            1,
            bb,
            ForcedBetType::BigBlind,
        );

        let committed_sb = builder.round_bet_state.committed(0);
        let current_max = builder.round_bet_state.current_max();
        assert!((committed_sb - (ante + sb)) == 0);
        assert!((current_max - (ante + bb)) == 0);

        use crate::core::PlayerBitSet;
        let mut players_active = PlayerBitSet::new(2);
        players_active.enable(0);
        players_active.enable(1);

        let completion = crate::arena::action::Action::PlayedAction(PlayedActionPayload {
            action: AgentAction::Call,
            idx: 0,
            round: Round::Preflop,
            player_stack: remaining[0] - call_delta,
            starting_pot: 0,
            final_pot: 0,
            starting_bet: bb,
            final_bet: bb,
            starting_min_raise: bb * 2,
            final_min_raise: bb * 2,
            starting_player_bet: sb,
            final_player_bet: sb + call_delta,
            players_active,
            players_all_in: PlayerBitSet::new(2),
        });

        builder
            .record_action(game_id, &completion, &game_state)
            .unwrap();

        let last_action = builder.current_round_actions.last().unwrap();
        assert_eq!(last_action.action, crate::open_hand_history::Action::Call);
        assert_eq!(last_action.amount, call_delta);
    }

    #[test]
    fn test_small_blind_with_one_cent_stack_action_is_recorded() {
        // This test verifies that actions from players with small (but not all-in) stacks
        // are recorded. Players with stack > 0 are not all-in.
        let num_players = 6;
        let sb = 300;
        let bb = 300;
        let ante = 0;
        // Use a value > 0 so player is not considered all-in
        let small_remaining = 1;
        let stacks = vec![
            100_000_000,
            100_000_000,
            100_000_000,
            100_000_000,
            100_000_000,
            sb + small_remaining,
        ];
        let dealer_idx = 4; // Player 4 is the dealer, so player 5 posts the small blind
        let game_state = GameStateBuilder::new()
            .stacks(stacks.clone())
            .blinds(bb, sb)
            .ante(ante)
            .dealer_idx(dealer_idx)
            .build()
            .unwrap();

        let mut builder = HandHistoryBuilder::new(ConverterConfig::default());
        let game_id = 7777u128;

        builder
            .record_action(
                game_id,
                &crate::arena::action::Action::GameStart(crate::arena::action::GameStartPayload {
                    ante,
                    small_blind: sb,
                    big_blind: bb,
                }),
                &game_state,
            )
            .unwrap();

        let mut remaining = stacks.clone();
        let sb_player_idx = num_players - 1;

        record_forced_bet_action(
            &mut builder,
            game_id,
            &game_state,
            &mut remaining,
            sb_player_idx,
            sb,
            ForcedBetType::SmallBlind,
        );
        record_forced_bet_action(
            &mut builder,
            game_id,
            &game_state,
            &mut remaining,
            0,
            bb,
            ForcedBetType::BigBlind,
        );

        for (idx, &stack) in remaining.iter().enumerate().take(sb_player_idx).skip(1) {
            let fold = create_multi_player_action(MultiPlayerActionParams {
                agent_action: AgentAction::Fold,
                idx,
                round: Round::Preflop,
                player_stack: stack,
                starting_player_bet: 0,
                final_player_bet: 0,
                starting_bet: bb,
                final_bet: bb,
                num_players,
            });
            builder.record_action(game_id, &fold, &game_state).unwrap();
        }

        // SB player has small_remaining stack left (> 0). Action should be recorded.
        let sb_check = create_multi_player_action(MultiPlayerActionParams {
            agent_action: AgentAction::Call,
            idx: sb_player_idx,
            round: Round::Preflop,
            player_stack: remaining[sb_player_idx],
            starting_player_bet: sb,
            final_player_bet: sb,
            starting_bet: bb,
            final_bet: bb,
            num_players,
        });
        builder
            .record_action(game_id, &sb_check, &game_state)
            .unwrap();

        let bb_check = create_multi_player_action(MultiPlayerActionParams {
            agent_action: AgentAction::Call,
            idx: 0,
            round: Round::Preflop,
            player_stack: remaining[0],
            starting_player_bet: bb,
            final_player_bet: bb,
            starting_bet: bb,
            final_bet: bb,
            num_players,
        });
        builder
            .record_action(game_id, &bb_check, &game_state)
            .unwrap();

        let small_blind_checks = builder
            .current_round_actions
            .iter()
            .filter(|action| {
                action.player_id == sb_player_idx as u64
                    && matches!(action.action, crate::open_hand_history::Action::Check)
            })
            .count();

        // The action is recorded because the player has stack > 0.
        // Players with small but meaningful stacks should have their actions recorded.
        assert_eq!(
            small_blind_checks, 1,
            "Small blind with small stack (> 0) should have their action recorded"
        );
    }

    #[test]
    fn test_small_blind_completion_is_exactly_one_cent() {
        use crate::arena::action::{AgentAction, ForcedBetType, GameStartPayload};
        use crate::arena::game_state::Round;

        let ante = 306_033;
        let sb = 315_632;
        let bb = 315_633;
        let completion = 1;
        let num_players = 6;
        let stacks = vec![100_000_000; num_players];
        let dealer_idx = num_players - 1;
        let game_state = GameStateBuilder::new()
            .stacks(stacks.clone())
            .blinds(bb, sb)
            .ante(ante)
            .dealer_idx(dealer_idx)
            .build()
            .unwrap();

        let mut builder = HandHistoryBuilder::new(ConverterConfig::default());
        let game_id = 11_337u128;

        let game_start = crate::arena::action::Action::GameStart(GameStartPayload {
            ante,
            small_blind: sb,
            big_blind: bb,
        });
        builder
            .record_action(game_id, &game_start, &game_state)
            .unwrap();

        let mut remaining = stacks.clone();
        for idx in 0..num_players {
            record_forced_bet_action(
                &mut builder,
                game_id,
                &game_state,
                &mut remaining,
                idx,
                ante,
                ForcedBetType::Ante,
            );
        }

        record_forced_bet_action(
            &mut builder,
            game_id,
            &game_state,
            &mut remaining,
            0,
            sb,
            ForcedBetType::SmallBlind,
        );
        record_forced_bet_action(
            &mut builder,
            game_id,
            &game_state,
            &mut remaining,
            1,
            bb,
            ForcedBetType::BigBlind,
        );

        let mut enqueue_call = |player_idx: usize| {
            remaining[player_idx] -= bb;
            let call = create_multi_player_action(MultiPlayerActionParams {
                agent_action: AgentAction::Call,
                idx: player_idx,
                round: Round::Preflop,
                player_stack: remaining[player_idx],
                starting_player_bet: 0,
                final_player_bet: bb,
                starting_bet: bb,
                final_bet: bb,
                num_players,
            });
            builder.record_action(game_id, &call, &game_state).unwrap();
        };

        enqueue_call(2);
        enqueue_call(3);

        for (idx, &stack) in remaining.iter().enumerate().take(num_players).skip(4) {
            let fold = create_multi_player_action(MultiPlayerActionParams {
                agent_action: AgentAction::Fold,
                idx,
                round: Round::Preflop,
                player_stack: stack,
                starting_player_bet: 0,
                final_player_bet: 0,
                starting_bet: bb,
                final_bet: bb,
                num_players,
            });
            builder.record_action(game_id, &fold, &game_state).unwrap();
        }

        remaining[0] -= completion;
        let sb_completion = create_multi_player_action(MultiPlayerActionParams {
            agent_action: AgentAction::Call,
            idx: 0,
            round: Round::Preflop,
            player_stack: remaining[0],
            starting_player_bet: sb,
            final_player_bet: sb + completion,
            starting_bet: bb,
            final_bet: bb,
            num_players,
        });
        builder
            .record_action(game_id, &sb_completion, &game_state)
            .unwrap();

        let last_action = builder.current_round_actions.last().unwrap();
        assert_eq!(last_action.action, crate::open_hand_history::Action::Call);
        assert!((last_action.amount - completion) == 0);
    }

    #[test]
    fn test_record_action_bet_after_bet_is_raise() {
        let mut builder = HandHistoryBuilder::new(ConverterConfig::default());
        let game_state = create_test_game_state();
        builder.init_from_game_state(12345, &game_state);
        builder.start_round("Preflop".to_string());

        // First bet
        let action1 = create_played_action_with_table(
            crate::arena::action::AgentAction::Bet(50),
            0,
            0,
            50,
            950,
            0,
            50,
        );
        builder.record_action(12345, &action1, &game_state).unwrap();

        // Second bet should be raise
        let action2 = create_played_action_with_table(
            crate::arena::action::AgentAction::Bet(100),
            1,
            0,
            100,
            1400,
            50,
            100,
        );
        builder.record_action(12345, &action2, &game_state).unwrap();

        assert_eq!(builder.current_round_actions.len(), 2);
        assert_eq!(
            builder.current_round_actions[1].action,
            crate::open_hand_history::Action::Raise
        );
        assert_eq!(builder.current_round_actions[1].amount, 100);
    }

    #[test]
    fn test_record_action_all_in_detection() {
        let mut builder = HandHistoryBuilder::new(ConverterConfig::default());
        let game_state = create_test_game_state();
        builder.init_from_game_state(12345, &game_state);
        builder.start_round("Preflop".to_string());

        // Player goes all-in (stack becomes 0)
        let action = create_played_action(crate::arena::action::AgentAction::AllIn, 0, 0, 1000, 0);
        builder.record_action(12345, &action, &game_state).unwrap();

        assert_eq!(builder.current_round_actions.len(), 1);
        assert!(builder.current_round_actions[0].is_allin);
        assert_eq!(builder.current_round_actions[0].amount, 1000);
    }

    #[test]
    fn test_record_action_all_in_call_vs_raise() {
        let mut builder = HandHistoryBuilder::new(ConverterConfig::default());
        let game_state = create_test_game_state();
        builder.init_from_game_state(12345, &game_state);
        builder.start_round("Preflop".to_string());
        let opening_bet = create_played_action_with_table(
            crate::arena::action::AgentAction::Bet(50),
            0,
            0,
            50,
            950,
            0,
            50,
        );
        builder
            .record_action(12345, &opening_bet, &game_state)
            .unwrap();

        builder.player_stacks[1] = 50;
        builder.players[1].starting_stack = 50;

        let call_all_in = create_played_action_with_table(
            crate::arena::action::AgentAction::AllIn,
            1,
            0,
            50,
            0,
            50,
            50,
        );
        builder
            .record_action(12345, &call_all_in, &game_state)
            .unwrap();

        assert_eq!(
            builder.current_round_actions.last().unwrap().action,
            crate::open_hand_history::Action::Call
        );

        let mut builder_raise = HandHistoryBuilder::new(ConverterConfig::default());
        builder_raise.init_from_game_state(67890, &game_state);
        builder_raise.start_round("Preflop".to_string());
        builder_raise
            .record_action(67890, &opening_bet, &game_state)
            .unwrap();

        builder_raise.player_stacks[1] = 120;
        builder_raise.players[1].starting_stack = 120;

        let raise_all_in = create_played_action_with_table(
            crate::arena::action::AgentAction::AllIn,
            1,
            0,
            120,
            0,
            50,
            120,
        );
        builder_raise
            .record_action(67890, &raise_all_in, &game_state)
            .unwrap();

        assert_eq!(
            builder_raise.current_round_actions.last().unwrap().action,
            crate::open_hand_history::Action::Raise
        );
    }

    #[test]
    fn test_rejects_action_amount_above_stack() {
        let mut builder = HandHistoryBuilder::new(ConverterConfig::default());
        let game_state = create_test_game_state();
        builder.init_from_game_state(12345, &game_state);
        builder.start_round("Preflop".to_string());

        builder.player_stacks[0] = 100;
        builder.players[0].starting_stack = 100;

        let action = create_played_action_with_table(
            crate::arena::action::AgentAction::Call,
            0,
            0,
            1_000_000,
            0,
            1_000_000,
            1_000_000,
        );
        assert!(matches!(
            builder.record_action(12345, &action, &game_state),
            Err(OHHConversionError::InconsistentState(_))
        ));
        assert!(builder.current_round_actions.is_empty());
        assert_eq!(builder.player_stacks[0], 100);
    }

    #[test]
    fn test_rejects_oversized_exact_bet_delta() {
        let mut builder = HandHistoryBuilder::new(ConverterConfig::default());
        let game_state = create_test_game_state();

        builder
            .record_action(999, &create_game_start_action(), &game_state)
            .unwrap();
        builder
            .record_action(
                999,
                &crate::arena::action::Action::RoundAdvance(
                    crate::arena::game_state::Round::Preflop,
                ),
                &game_state,
            )
            .unwrap();

        let oversized_bet =
            create_played_action(crate::arena::action::AgentAction::Bet(1000), 0, 0, 1001, 0);

        assert!(matches!(
            builder.record_action(999, &oversized_bet, &game_state),
            Err(OHHConversionError::InconsistentState(_))
        ));
        assert!(builder.current_round_actions.is_empty());
        assert_eq!(builder.player_stacks[0], 1000);
    }

    #[test]
    fn test_short_stack_raise_classified_as_call() {
        let mut builder = HandHistoryBuilder::new(ConverterConfig::default());
        let game_state = create_test_game_state();

        builder
            .record_action(1, &create_game_start_action(), &game_state)
            .unwrap();
        builder
            .record_action(
                1,
                &create_forced_bet_action(0, 1, crate::arena::action::ForcedBetType::SmallBlind),
                &game_state,
            )
            .unwrap();
        builder
            .record_action(
                1,
                &create_forced_bet_action(1, 2, crate::arena::action::ForcedBetType::BigBlind),
                &game_state,
            )
            .unwrap();

        let big_raise = create_played_action_with_table(
            crate::arena::action::AgentAction::Bet(1000),
            1,
            2,
            1000,
            500,
            2,
            1000,
        );
        builder.record_action(1, &big_raise, &game_state).unwrap();

        builder.player_stacks[2] = 40;
        builder.players[2].starting_stack = 40;

        let attempted_raise = create_played_action_with_table(
            crate::arena::action::AgentAction::Bet(2000),
            2,
            0,
            40,
            0,
            1000,
            1000,
        );

        builder
            .record_action(1, &attempted_raise, &game_state)
            .unwrap();

        let last_action = builder.current_round_actions.last().unwrap();
        assert_eq!(last_action.action, crate::open_hand_history::Action::Call);
        assert_eq!(last_action.amount, 40);
        assert!(last_action.is_allin);
    }

    #[test]
    fn test_big_blind_short_all_in_treated_as_raise() {
        let mut builder = HandHistoryBuilder::new(ConverterConfig::default());
        let game_state = create_test_game_state();

        builder
            .record_action(42, &create_game_start_action(), &game_state)
            .unwrap();
        builder
            .record_action(
                42,
                &create_forced_bet_action(1, 1, crate::arena::action::ForcedBetType::SmallBlind),
                &game_state,
            )
            .unwrap();
        builder
            .record_action(
                42,
                &create_forced_bet_action(2, 2, crate::arena::action::ForcedBetType::BigBlind),
                &game_state,
            )
            .unwrap();

        builder.player_stacks[2] = 1;
        builder.players[2].starting_stack = 3;

        let bb_all_in = create_played_action_with_table(
            crate::arena::action::AgentAction::AllIn,
            2,
            2,
            3,
            0,
            2,
            3,
        );

        builder.record_action(42, &bb_all_in, &game_state).unwrap();

        let last_action = builder.current_round_actions.last().unwrap();
        assert_eq!(last_action.action, crate::open_hand_history::Action::Raise);
        assert!(last_action.is_allin);
        assert!((last_action.amount - 1) == 0);
    }

    #[test]
    fn test_small_blind_call_is_not_raise() {
        use crate::arena::action::{AgentAction, PlayedActionPayload};
        use crate::open_hand_history::Action;

        let mut builder = HandHistoryBuilder::new(ConverterConfig::default());
        let game_state = create_test_game_state();
        builder.init_from_game_state(314, &game_state);
        builder.start_round("Preflop".to_string());

        // Simulate antes plus blind postings for two seats.
        builder.round_bet_state.record(0, 315_600);
        builder.round_bet_state.record(0, 315_632);
        builder.round_bet_state.record(1, 315_600);
        builder.round_bet_state.record(1, 315_676);

        let payload = PlayedActionPayload {
            action: AgentAction::Call,
            idx: 0,
            round: Round::Preflop,
            player_stack: 0,
            starting_pot: 0,
            final_pot: 0,
            starting_bet: 315_676,
            final_bet: 315_676,
            starting_min_raise: 2,
            final_min_raise: 2,
            starting_player_bet: 315_632,
            final_player_bet: 315_676,
            players_active: active_players(game_state.num_players),
            players_all_in: PlayerBitSet::new(game_state.num_players),
        };

        let classified = builder.determine_ohh_action(&payload, 44);
        assert_eq!(classified, Action::Call);
    }

    #[test]
    fn test_large_pot_call_not_misclassified_as_raise() {
        let mut builder = HandHistoryBuilder::new(ConverterConfig::default());
        let game_state = create_test_game_state();

        builder
            .record_action(7, &create_game_start_action(), &game_state)
            .unwrap();
        builder
            .record_action(
                7,
                &crate::arena::action::Action::RoundAdvance(
                    crate::arena::game_state::Round::Preflop,
                ),
                &game_state,
            )
            .unwrap();

        builder.player_stacks[1] = 200_000_000;
        builder.players[1].starting_stack = 200_000_000;
        builder.player_stacks[2] = 200_000_000;
        builder.players[2].starting_stack = 200_000_000;

        let big_raise = create_played_action_with_table(
            crate::arena::action::AgentAction::Bet(99_996_850),
            1,
            2,
            99_996_850,
            100_003_152,
            2,
            99_996_850,
        );
        builder.record_action(7, &big_raise, &game_state).unwrap();

        let near_call = create_played_action_with_table(
            crate::arena::action::AgentAction::Call,
            2,
            952_902,
            99_996_850,
            100_956_052,
            99_996_850,
            99_996_850,
        );
        builder.record_action(7, &near_call, &game_state).unwrap();

        let last_action = builder.current_round_actions.last().unwrap();
        assert_eq!(last_action.action, crate::open_hand_history::Action::Call);
    }

    #[test]
    fn test_record_action_forced_bets() {
        let mut builder = HandHistoryBuilder::new(ConverterConfig::default());
        let game_state = create_test_game_state();
        builder.init_from_game_state(12345, &game_state);
        builder.start_round("Preflop".to_string());

        // Test small blind
        let action =
            create_forced_bet_action(0, 1, crate::arena::action::ForcedBetType::SmallBlind);
        builder.record_action(12345, &action, &game_state).unwrap();

        assert_eq!(builder.current_round_actions.len(), 1);
        assert_eq!(
            builder.current_round_actions[0].action,
            crate::open_hand_history::Action::PostSmallBlind
        );
        assert_eq!(builder.current_round_actions[0].amount, 1);

        // Test big blind
        let action = create_forced_bet_action(1, 2, crate::arena::action::ForcedBetType::BigBlind);
        builder.record_action(12345, &action, &game_state).unwrap();

        assert_eq!(builder.current_round_actions.len(), 2);
        assert_eq!(
            builder.current_round_actions[1].action,
            crate::open_hand_history::Action::PostBigBlind
        );
        assert_eq!(builder.current_round_actions[1].amount, 2);

        // Test ante
        let action = create_forced_bet_action(2, 1, crate::arena::action::ForcedBetType::Ante);
        builder.record_action(12345, &action, &game_state).unwrap();

        assert_eq!(builder.current_round_actions.len(), 3);
        assert_eq!(
            builder.current_round_actions[2].action,
            crate::open_hand_history::Action::PostAnte
        );
        assert_eq!(builder.current_round_actions[2].amount, 1);
    }

    #[test]
    fn test_ante_forced_bets_before_round_are_preserved() {
        let mut builder = HandHistoryBuilder::new(ConverterConfig::default());
        let game_state = create_test_game_state();
        builder.init_from_game_state(12345, &game_state);

        let game_start =
            crate::arena::action::Action::GameStart(crate::arena::action::GameStartPayload {
                small_blind: 1,
                big_blind: 2,
                ante: 1,
            });
        builder
            .record_action(12345, &game_start, &game_state)
            .unwrap();

        let ante_action = create_forced_bet_action(0, 1, crate::arena::action::ForcedBetType::Ante);
        builder
            .record_action(12345, &ante_action, &game_state)
            .unwrap();

        assert_eq!(builder.current_street.as_deref(), Some("Preflop"));
        assert_eq!(builder.current_round_id, 1);
        assert_eq!(builder.current_round_actions.len(), 1);
        assert_eq!(
            builder.current_round_actions[0].action,
            crate::open_hand_history::Action::PostAnte
        );
    }

    #[test]
    fn test_record_action_deal_community() {
        use crate::core::{Card, Suit, Value};

        let mut builder = HandHistoryBuilder::new(ConverterConfig::default());
        let game_state = create_test_game_state();
        builder.init_from_game_state(12345, &game_state);
        builder.start_round("Flop".to_string());

        let card = Card::new(Value::Ace, Suit::Spade);
        let action = crate::arena::action::Action::DealCommunity(card);
        builder.record_action(12345, &action, &game_state).unwrap();

        assert_eq!(builder.current_round_cards.len(), 1);
        assert_eq!(builder.current_round_cards[0], card);
    }

    #[test]
    fn test_deal_community_starts_flop_when_round_not_advanced() {
        use crate::core::{Card, Suit, Value};

        let mut builder = HandHistoryBuilder::new(ConverterConfig::default());
        let game_state = create_test_game_state();
        builder.init_from_game_state(12345, &game_state);
        builder.start_round("Preflop".to_string());

        // Ensure there is at least one preflop action recorded
        let sb = create_forced_bet_action(0, 1, crate::arena::action::ForcedBetType::SmallBlind);
        builder.record_action(12345, &sb, &game_state).unwrap();

        // Engine can enter DealFlop before Flop; ensure this doesn't keep cards on Preflop
        let deal_flop_round =
            crate::arena::action::Action::RoundAdvance(crate::arena::game_state::Round::DealFlop);
        builder
            .record_action(12345, &deal_flop_round, &game_state)
            .unwrap();

        let card = Card::new(Value::Ten, Suit::Spade);
        let action = crate::arena::action::Action::DealCommunity(card);
        builder.record_action(12345, &action, &game_state).unwrap();

        assert_eq!(builder.rounds.len(), 1);
        assert_eq!(builder.rounds[0].street, "Preflop");
        assert!(builder.rounds[0].cards.is_none());
        assert_eq!(builder.current_street.as_deref(), Some("Flop"));
        assert_eq!(builder.current_round_cards, vec![card]);
    }

    #[test]
    fn test_record_action_award() {
        use crate::core::{Card, FlatHand, Hand, Rankable, Suit, Value};

        let mut builder = HandHistoryBuilder::new(ConverterConfig::default());
        let game_state = create_test_game_state();
        builder.init_from_game_state(12345, &game_state);
        builder.start_round("Showdown".to_string());

        // Add cards to player 0 so ShowsCards action can be created
        let card1 = Card::new(Value::Ace, Suit::Spade);
        let card2 = Card::new(Value::Ace, Suit::Heart);
        builder.player_cards.insert(0, vec![card1, card2]);

        let hand = Hand::new_with_cards(vec![card1, card2]);
        // Placeholder rank: any one-pair value works for this payload test.
        let rank = FlatHand::new_from_str("AsAh2c3d4s").unwrap().rank();

        let action = crate::arena::action::Action::Award(crate::arena::action::AwardPayload {
            total_pot: 100,
            award_amount: 100,
            rank: Some(rank),
            hand: Some(hand),
            idx: 0,
        });

        builder.record_action(12345, &action, &game_state).unwrap();

        // Check pot creation
        assert_eq!(builder.pots.len(), 1);
        assert_eq!(builder.pots[0].number, 0);
        assert_eq!(builder.pots[0].amount, 100);
        assert_eq!(builder.pots[0].player_wins.len(), 1);
        assert_eq!(builder.pots[0].player_wins[0].player_id, 0);
        assert_eq!(builder.pots[0].player_wins[0].win_amount, 100);

        // Check ShowsCards action was added
        assert_eq!(builder.current_round_actions.len(), 1);
        assert_eq!(
            builder.current_round_actions[0].action,
            crate::open_hand_history::Action::ShowsCards
        );
        assert_eq!(builder.current_round_actions[0].player_id, 0);
    }

    #[test]
    fn test_record_action_award_split_pot() {
        let mut builder = HandHistoryBuilder::new(ConverterConfig::default());
        let game_state = create_test_game_state();
        builder.init_from_game_state(12345, &game_state);
        builder.start_round("Showdown".to_string());

        let award_one = crate::arena::action::Action::Award(crate::arena::action::AwardPayload {
            total_pot: 100,
            award_amount: 60,
            rank: None,
            hand: None,
            idx: 0,
        });
        builder
            .record_action(12345, &award_one, &game_state)
            .unwrap();

        assert!(builder.pots.is_empty());

        let award_two = crate::arena::action::Action::Award(crate::arena::action::AwardPayload {
            total_pot: 100,
            award_amount: 40,
            rank: None,
            hand: None,
            idx: 1,
        });
        builder
            .record_action(12345, &award_two, &game_state)
            .unwrap();

        assert_eq!(builder.pots.len(), 1);
        assert_eq!(builder.pots[0].player_wins.len(), 2);
        assert!(
            (builder.pots[0]
                .player_wins
                .iter()
                .map(|w| w.win_amount)
                .sum::<Chips>()
                - 100)
                == 0
        );
    }

    #[test]
    fn test_record_action_award_multiple_pots() {
        let mut builder = HandHistoryBuilder::new(ConverterConfig::default());
        let game_state = create_test_game_state();
        builder.init_from_game_state(12345, &game_state);
        builder.start_round("Showdown".to_string());

        let first_pot = crate::arena::action::Action::Award(crate::arena::action::AwardPayload {
            total_pot: 50,
            award_amount: 50,
            rank: None,
            hand: None,
            idx: 0,
        });
        builder
            .record_action(12345, &first_pot, &game_state)
            .unwrap();

        let second_pot = crate::arena::action::Action::Award(crate::arena::action::AwardPayload {
            total_pot: 30,
            award_amount: 30,
            rank: None,
            hand: None,
            idx: 1,
        });
        builder
            .record_action(12345, &second_pot, &game_state)
            .unwrap();

        assert_eq!(builder.pots.len(), 2);
        assert_eq!(builder.pots[0].amount, 50);
        assert_eq!(builder.pots[1].amount, 30);
    }

    #[test]
    fn test_street_mapping() {
        assert_eq!(
            HandHistoryBuilder::map_arena_round_to_street(
                &crate::arena::game_state::Round::Preflop
            ),
            Some("Preflop".to_string())
        );
        assert_eq!(
            HandHistoryBuilder::map_arena_round_to_street(&crate::arena::game_state::Round::Flop),
            Some("Flop".to_string())
        );
        assert_eq!(
            HandHistoryBuilder::map_arena_round_to_street(&crate::arena::game_state::Round::Turn),
            Some("Turn".to_string())
        );
        assert_eq!(
            HandHistoryBuilder::map_arena_round_to_street(&crate::arena::game_state::Round::River),
            Some("River".to_string())
        );
        assert_eq!(
            HandHistoryBuilder::map_arena_round_to_street(
                &crate::arena::game_state::Round::Showdown
            ),
            Some("Showdown".to_string())
        );

        // Deal rounds should be skipped
        assert_eq!(
            HandHistoryBuilder::map_arena_round_to_street(
                &crate::arena::game_state::Round::DealPreflop
            ),
            None
        );
        assert_eq!(
            HandHistoryBuilder::map_arena_round_to_street(
                &crate::arena::game_state::Round::DealFlop
            ),
            None
        );
        assert_eq!(
            HandHistoryBuilder::map_arena_round_to_street(
                &crate::arena::game_state::Round::Starting
            ),
            None
        );
    }

    #[test]
    fn test_integration_full_hand_sequence() {
        use crate::core::{Card, Suit, Value};

        let mut builder = HandHistoryBuilder::new(ConverterConfig::default());
        let game_state = create_test_game_state();

        // Game setup
        let game_start = create_game_start_action();
        builder
            .record_action(12345, &game_start, &game_state)
            .unwrap();

        // Round advance to preflop
        let round_advance =
            crate::arena::action::Action::RoundAdvance(crate::arena::game_state::Round::Preflop);
        builder
            .record_action(12345, &round_advance, &game_state)
            .unwrap();

        // Deal cards to complete dealing
        let cards = [
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::King, Suit::Heart),
            Card::new(Value::Queen, Suit::Diamond),
            Card::new(Value::Jack, Suit::Club),
            Card::new(Value::Ten, Suit::Spade),
            Card::new(Value::Nine, Suit::Heart),
        ];

        for (i, card) in cards.iter().enumerate() {
            let action = create_deal_starting_hand_action(i % 3, *card);
            builder.record_action(12345, &action, &game_state).unwrap();
        }

        // Force bets
        let sb_action =
            create_forced_bet_action(1, 1, crate::arena::action::ForcedBetType::SmallBlind);
        builder
            .record_action(12345, &sb_action, &game_state)
            .unwrap();

        let bb_action =
            create_forced_bet_action(2, 2, crate::arena::action::ForcedBetType::BigBlind);
        builder
            .record_action(12345, &bb_action, &game_state)
            .unwrap();

        // Play actions
        let call_action =
            create_played_action(crate::arena::action::AgentAction::Call, 0, 0, 2, 998);
        builder
            .record_action(12345, &call_action, &game_state)
            .unwrap();

        let fold_action =
            create_played_action(crate::arena::action::AgentAction::Fold, 1, 1, 1, 1499);
        builder
            .record_action(12345, &fold_action, &game_state)
            .unwrap();

        // Award pot
        let award_action =
            crate::arena::action::Action::Award(crate::arena::action::AwardPayload {
                total_pot: 5,
                award_amount: 5,
                rank: None,
                hand: None,
                idx: 2,
            });
        builder
            .record_action(12345, &award_action, &game_state)
            .unwrap();

        // Build final hand history
        let hand_history = builder.build().unwrap();

        // Verify structure
        assert_eq!(hand_history.spec_version, "1.4.7");
        assert_eq!(hand_history.game_number, "12345");
        assert_eq!(hand_history.small_blind_amount, 1);
        assert_eq!(hand_history.big_blind_amount, 2);
        assert_eq!(hand_history.rounds.len(), 1);
        assert_eq!(hand_history.rounds[0].street, "Preflop");

        // Should have: 3 dealt cards, 1 small blind, 1 big blind, 1 call, 1 fold actions
        assert_eq!(hand_history.rounds[0].actions.len(), 7);

        // Verify pots
        assert_eq!(hand_history.pots.len(), 1);
        assert_eq!(hand_history.pots[0].amount, 5);
    }

    /// Regression test: Ensure all actions from the arena are recorded for players
    /// with small-but-valid stacks. Players with stack > 0
    /// should have their actions recorded.
    #[test]
    fn test_check_action_recorded_for_tiny_stack() {
        let mut builder = HandHistoryBuilder::new(ConverterConfig::default());
        let game_state = create_test_game_state();

        builder.init_from_game_state(12345, &game_state);
        builder.start_round("Flop".to_string());

        // Use a small stack that's greater than 0
        // to verify actions are recorded for small-but-valid stacks
        let small_stack = 1;
        builder.player_stacks[0] = small_stack;

        // Create a check action where player has a small stack remaining
        let check_action = create_played_action(
            crate::arena::action::AgentAction::Call,
            0,
            0,
            0,
            small_stack,
        );
        builder
            .record_action(12345, &check_action, &game_state)
            .unwrap();

        // The check action MUST be recorded since player is not all-in
        assert_eq!(
            builder.current_round_actions.len(),
            1,
            "Actions for players with stack > 0 should be recorded"
        );
        assert_eq!(
            builder.current_round_actions[0].action,
            crate::open_hand_history::Action::Check
        );
        assert_eq!(builder.current_round_actions[0].player_id, 0);
    }

    /// Ensure bet actions are recorded for players with small (but meaningful) stacks.
    #[test]
    fn test_bet_action_recorded_for_small_stack() {
        let mut builder = HandHistoryBuilder::new(ConverterConfig::default());
        let game_state = create_test_game_state();

        builder.init_from_game_state(12345, &game_state);
        builder.start_round("Flop".to_string());

        // Use a small stack that's greater than 0
        let small_stack = 1;
        builder.player_stacks[0] = small_stack;

        // Player bets their entire (small) stack
        let bet_action = create_played_action(
            crate::arena::action::AgentAction::Bet(small_stack),
            0,
            0,
            small_stack,
            0, // After betting, they have 0 remaining
        );
        builder
            .record_action(12345, &bet_action, &game_state)
            .unwrap();

        // Player with stack > 0 can bet - record it
        assert_eq!(
            builder.current_round_actions.len(),
            1,
            "Players with meaningful stacks (> 0) can bet"
        );
    }

    /// Verifies that RoundBetState::reset clears both the max bet and all
    /// player contributions back to zero.
    #[test]
    fn test_round_bet_state_reset() {
        let mut state = RoundBetState::default();

        state.record(0, 50);
        state.record(1, 100);

        assert_eq!(state.current_max(), 100);
        assert_eq!(state.committed(0), 50);
        assert_eq!(state.committed(1), 100);

        state.reset();

        assert_eq!(state.current_max(), 0);
        assert_eq!(state.committed(0), 0);
        assert_eq!(state.committed(1), 0);
    }

    /// Verifies that RoundBetState::record correctly tracks the maximum bet:
    /// - First bet sets the max
    /// - Smaller or equal bets do not change the max
    /// - Only strictly larger bets update the max
    #[test]
    fn test_round_bet_state_record_max() {
        let mut state = RoundBetState::default();

        state.record(0, 50);
        assert_eq!(state.current_max(), 50);

        state.record(1, 30);
        assert_eq!(state.current_max(), 50);

        state.record(2, 50);
        assert_eq!(state.current_max(), 50);

        state.record(3, 100);
        assert_eq!(state.current_max(), 100);
    }

    /// Verifies that multiple bets from the same player accumulate correctly,
    /// and the max reflects the player's total contribution.
    #[test]
    fn test_round_bet_state_accumulates() {
        let mut state = RoundBetState::default();

        state.record(0, 10);
        assert_eq!(state.committed(0), 10);

        state.record(0, 20);
        assert_eq!(state.committed(0), 30);

        state.record(0, 5);
        assert_eq!(state.committed(0), 35);

        assert_eq!(state.current_max(), 35);
    }

    /// Verifies that RoundBetState::record ignores zero and negative amounts,
    /// leaving the player's committed amount unchanged.
    #[test]
    fn test_round_bet_state_ignores_invalid() {
        let mut state = RoundBetState::default();

        state.record(0, 50);
        assert_eq!(state.committed(0), 50);

        state.record(0, 0);
        assert_eq!(state.committed(0), 50);

        state.record(0, -10);
        assert_eq!(state.committed(0), 50);
    }

    /// Verifies that active_players creates a bitset with all players enabled,
    /// and the count matches the requested number of players.
    #[test]
    fn test_active_players_returns_enabled_bitset() {
        let bitset = active_players(3);

        assert!(bitset.get(0), "Player 0 should be active");
        assert!(bitset.get(1), "Player 1 should be active");
        assert!(bitset.get(2), "Player 2 should be active");

        assert_eq!(bitset.count(), 3, "Should have 3 active players");
    }

    /// Verifies that is_in_betting_round returns the correct value:
    /// - false when no street is set
    /// - false for Showdown
    /// - true for Preflop, Flop, Turn, and River
    #[test]
    fn test_is_in_betting_round_returns_correct_value() {
        let mut builder = HandHistoryBuilder::new(ConverterConfig::default());

        assert!(
            !builder.is_in_betting_round(),
            "Should return false when no street is set"
        );

        builder.current_street = Some("Showdown".to_string());
        assert!(
            !builder.is_in_betting_round(),
            "Should return false for Showdown"
        );

        builder.current_street = Some("Preflop".to_string());
        assert!(
            builder.is_in_betting_round(),
            "Should return true for Preflop"
        );

        builder.current_street = Some("Flop".to_string());
        assert!(builder.is_in_betting_round(), "Should return true for Flop");

        builder.current_street = Some("Turn".to_string());
        assert!(builder.is_in_betting_round(), "Should return true for Turn");

        builder.current_street = Some("River".to_string());
        assert!(
            builder.is_in_betting_round(),
            "Should return true for River"
        );
    }

    #[test]
    fn test_ensure_pending_pot_logic() {
        let mut builder = HandHistoryBuilder::new(ConverterConfig::default());

        builder.ensure_pending_pot(100);
        assert_eq!(builder.pending_pot_expected_total, Some(100));

        builder.ensure_pending_pot(100);
        assert_eq!(builder.pending_pot_expected_total, Some(100));

        builder.ensure_pending_pot(200);
        assert_eq!(builder.pending_pot_expected_total, Some(200));
    }

    /// Verifies that determine_ohh_action correctly identifies a Call action
    /// when a player matches the current bet.
    #[test]
    fn test_determine_ohh_action_with_payload() {
        use crate::arena::action::{AgentAction, PlayedActionPayload};

        let mut builder = HandHistoryBuilder::new(ConverterConfig::default());
        builder.big_blind = 2;
        builder.current_street = Some("Preflop".to_string());

        builder.round_bet_state.record(0, 10);

        let payload = PlayedActionPayload {
            action: AgentAction::Call,
            idx: 1,
            round: Round::Preflop,
            player_stack: 90,
            starting_pot: 10,
            final_pot: 20,
            starting_bet: 10,
            final_bet: 10,
            starting_min_raise: 2,
            final_min_raise: 2,
            starting_player_bet: 0,
            final_player_bet: 10,
            players_active: active_players(2),
            players_all_in: PlayerBitSet::new(2),
        };

        let action = builder.determine_ohh_action(&payload, 10);

        assert!(
            matches!(action, crate::open_hand_history::Action::Call),
            "Should be Call when matching current bet, got {:?}",
            action
        );
    }

    /// Verifies that determine_ohh_action returns Fold for AgentAction::Fold.
    #[test]
    fn test_determine_ohh_action_fold() {
        use crate::arena::action::{AgentAction, PlayedActionPayload};

        let builder = HandHistoryBuilder::new(ConverterConfig::default());

        let payload = PlayedActionPayload {
            action: AgentAction::Fold,
            idx: 0,
            round: Round::Preflop,
            player_stack: 100,
            starting_pot: 0,
            final_pot: 0,
            starting_bet: 0,
            final_bet: 0,
            starting_min_raise: 2,
            final_min_raise: 2,
            starting_player_bet: 0,
            final_player_bet: 0,
            players_active: active_players(2),
            players_all_in: PlayerBitSet::new(2),
        };

        let action = builder.determine_ohh_action(&payload, 0);

        assert!(
            matches!(action, crate::open_hand_history::Action::Fold),
            "Should be Fold for AgentAction::Fold, got {:?}",
            action
        );
    }

    /// Verifies that determine_ohh_action returns Check for a zero amount
    /// with no facing bet.
    #[test]
    fn test_determine_ohh_action_check() {
        use crate::arena::action::{AgentAction, PlayedActionPayload};

        let mut builder = HandHistoryBuilder::new(ConverterConfig::default());
        builder.current_street = Some("Preflop".to_string());

        let payload = PlayedActionPayload {
            action: AgentAction::Call, // Big blind checking
            idx: 0,
            round: Round::Preflop,
            player_stack: 100,
            starting_pot: 0,
            final_pot: 0,
            starting_bet: 0,
            final_bet: 0,
            starting_min_raise: 2,
            final_min_raise: 2,
            starting_player_bet: 0,
            final_player_bet: 0,
            players_active: active_players(2),
            players_all_in: PlayerBitSet::new(2),
        };

        let action = builder.determine_ohh_action(&payload, 0);

        assert!(
            matches!(action, crate::open_hand_history::Action::Check),
            "Should be Check for zero amount with no facing bet, got {:?}",
            action
        );
    }

    #[test]
    fn test_award_records_gross_pot_and_rake_separately() {
        let mut builder = HandHistoryBuilder::new(ConverterConfig::default());
        let mut game_state = create_test_game_state();
        builder.init_from_game_state(777, &game_state);

        // Simulate the state seen by historians after a 10-chip gross pot had
        // 1 chip raked and 9 chips awarded.
        game_state.rake_collected = 1;
        let action = crate::arena::action::Action::Award(crate::arena::action::AwardPayload {
            idx: 0,
            total_pot: 10,
            award_amount: 9,
            rank: None,
            hand: None,
        });
        builder.record_action(777, &action, &game_state).unwrap();

        let hand = builder.build().unwrap();
        assert_eq!(hand.pots.len(), 1);
        assert_eq!(hand.pots[0].amount, 10);
        assert_eq!(hand.pots[0].rake, Some(1));
        assert_eq!(hand.pots[0].player_wins[0].win_amount, 9);
    }
}
