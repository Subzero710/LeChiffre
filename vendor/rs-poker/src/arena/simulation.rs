use smallvec::SmallVec;
use std::fmt;

use tracing::{Level, event, instrument};

use crate::arena::action::{FailedActionPayload, PlayedActionPayload};
use crate::arena::game_state::Round;
use crate::core::{Card, Deck, Rankable};

use super::action::{
    Action, AgentAction, AwardPayload, DealStartingHandPayload, ForcedBetPayload, GameStartPayload,
    PlayerSitPayload,
};

use super::Agent;
use super::GameState;
use super::historian::Historian;

/// # Description
///
/// This code is implementing a version of Texas Hold'em poker. It is a
/// simulation of the game that can be played with computer agents. The game
/// progresses through a number of rounds: Starting,
/// Preflop,
/// Flop,
/// Turn,
/// River, and
/// Showdown.
///
/// The simulation creates a deck of cards, shuffles it, and deals cards to the
/// players. The players then take turns making bets, raising or folding until
/// the round is complete. Then, the game moves to the next round, and the
/// process repeats. At the end of the game, the player with the best hand wins.
///
/// The simulation is designed to be used with agents that can make decisions
/// based on the game state. The `HoldemSimulation` struct keeps track of the
/// game state, the deck, and the actions taken in the game.
///
/// The `run` method can be used to run the entire game
///
/// # Behavior
///
/// - Any agent bet that is an over bet will silently turn into an all in. That
///   is to say if an agent has 100 in their stack and bet `100_000_000` that
///   will be accepted and will be equivilant to bet `100`
/// - Any bet that `GameState` rules as being impossible, those that turn into
///   [`rs-poker::arena::errors::GameStateError`] will instead be turned into
///   fold.
/// - It's expected that you have the same number of agents as you have chip
///   stacks in the game state. If players are not active, you can use the
///   `FoldingAgent` as a stand in and set the active bit to false.
pub struct HoldemSimulation {
    /// A randomly generated ID to represent the simulation.
    pub id: u128,
    pub agents: Vec<Box<dyn Agent>>,
    pub game_state: GameState,
    pub deck: Deck,
    pub historians: Vec<Box<dyn Historian>>,
    pub panic_on_historian_error: bool,
    /// The RNG this simulation deals from. Owned so `run()` can be `async` and
    /// spawned without borrowing an external `&mut R` across await points.
    pub rng: Box<dyn rand::Rng + Send>,
}

impl HoldemSimulation {
    pub fn more_rounds(&self) -> bool {
        !matches!(self.game_state.round, Round::Complete)
    }
    /// Returns the number of poker agents participating in this simulation.
    pub fn num_agents(&self) -> usize {
        self.agents.len()
    }

    /// Run the simulation all the way to completion. This will mutate the
    /// current state.
    ///
    /// The span is attached via `#[instrument]` (not a manual `span.enter()`
    /// guard) so it is correctly entered/exited around every `.await`
    /// suspension point on a multi-thread runtime.
    #[instrument(
        level = "debug",
        skip(self),
        fields(id = self.id, num_players = self.game_state.num_players)
    )]
    pub async fn run(&mut self) {
        while self.more_rounds() {
            if (self.game_state.player_active | self.game_state.player_all_in).count() == 1
                && !matches!(
                    self.game_state.round,
                    Round::Starting | Round::Ante | Round::DealPreflop
                )
            {
                self.distribute_pots(false).await;
                self.end_game().await;
                break;
            }
            self.run_round().await;
        }
    }

    pub async fn run_round(&mut self) {
        match self.game_state.round {
            // Dealing the user hand is dealt with as its own round
            // in order to use the per round active bit set
            // for iterating players
            Round::Starting => self.start().await,
            Round::Ante => self.ante().await,

            Round::DealPreflop => self.deal_preflop().await,
            Round::Preflop => self.preflop().await,

            Round::DealFlop => self.deal_flop().await,
            Round::Flop => self.flop().await,

            Round::DealTurn => self.deal_turn().await,
            Round::Turn => self.turn().await,

            Round::DealRiver => self.deal_river().await,
            Round::River => self.river().await,

            Round::Showdown => self.showdown().await,

            // There's nothing left to do to this.
            Round::Complete => (),
        }
    }

    async fn start(&mut self) {
        // Add an action to record the ante, sb and bb
        // This should allow recreating starting game state
        // together with PlayerSit actions.
        self.record_action(Action::GameStart(GameStartPayload {
            ante: self.game_state.ante,
            small_blind: self.game_state.small_blind,
            big_blind: self.game_state.big_blind,
        }))
        .await;

        while self.game_state.current_round_num_active_players() > 0 {
            let idx = self.game_state.to_act_idx();

            // Add an action that records starting stack for each player
            // Starting with to the left of the dealer
            // and ending with dealer button.
            self.record_action(Action::PlayerSit(PlayerSitPayload {
                player_stack: self.game_state.stacks[idx],
                idx,
                name: Some(self.agents[idx].name().to_string()),
            }))
            .await;

            // set the active bit on the player to false.
            // This allows us to not deal to players that
            // are sitting out, while also going in the same
            // order of dealing
            self.game_state.round_data.needs_action.disable(idx);

            self.game_state.round_data.advance_action();
        }

        // We're done with the non-betting dealing only round
        self.advance_round().await;
    }

    async fn ante(&mut self) {
        let ante = self.game_state.ante;
        if ante > 0 {
            // Force the ante from each active player.
            while self.game_state.current_round_num_active_players() > 0 {
                let idx = self.game_state.to_act_idx();

                let actual_ante = self.game_state.do_bet(ante, true).unwrap();
                self.record_action(Action::ForcedBet(ForcedBetPayload {
                    bet: actual_ante,
                    idx,
                    player_stack: self.game_state.stacks[idx],
                    forced_bet_type: super::action::ForcedBetType::Ante,
                }))
                .await;

                self.game_state.round_data.needs_action.disable(idx);
            }
        }
        self.advance_round().await;
    }

    async fn deal_preflop(&mut self) {
        // We deal the cards before advancing the round
        // This allows us to use the round active bitset
        while self.game_state.current_round_num_active_players() > 0 {
            let idx = self.game_state.to_act_idx();

            self.deal_player_cards(2).await;

            // This allows us to not deal to players that
            // are sitting out, while also going in the same
            // order of dealing
            self.game_state.round_data.needs_action.disable(idx);

            self.game_state.round_data.advance_action();
        }
        self.advance_round().await
    }

    async fn preflop(&mut self) {
        // Forced seats are determined by dealt players, including players
        // already all-in from the ante. Such a player can post a zero blind.
        let dealt = self.game_state.player_active | self.game_state.player_all_in;
        let next_dealt = |seat: usize| {
            (1..=self.game_state.num_players)
                .map(|offset| (seat + offset) % self.game_state.num_players)
                .find(|&idx| dealt.get(idx))
                .expect("a dealt blind seat")
        };
        let sb_seat = if dealt.count() == 2 && dealt.get(self.game_state.dealer_idx) {
            self.game_state.dealer_idx
        } else {
            next_dealt(self.game_state.dealer_idx)
        };
        let bb_seat = next_dealt(sb_seat);
        // Force the small blind and the big blind.
        if !self.game_state.sb_posted {
            let sb = self.game_state.small_blind;
            let sb_idx = sb_seat;
            self.game_state.round_data.to_act_idx = sb_idx;
            let actual_sb = self.game_state.do_bet(sb, true).unwrap();
            self.game_state.sb_posted = true;

            self.record_action(Action::ForcedBet(ForcedBetPayload {
                bet: actual_sb,
                idx: sb_idx,
                forced_bet_type: super::action::ForcedBetType::SmallBlind,
                player_stack: self.game_state.stacks[sb_idx],
            }))
            .await;
        }

        if !self.game_state.bb_posted {
            let bb = self.game_state.big_blind;
            let bb_idx = bb_seat;
            self.game_state.round_data.to_act_idx = bb_idx;
            let actual_bb = self.game_state.do_bet(bb, true).unwrap();
            self.game_state.bb_posted = true;
            // A short big blind does not lower the opening amount owed.
            self.game_state.round_data.bet = self.game_state.round_data.bet.max(bb);
            self.record_action(Action::ForcedBet(ForcedBetPayload {
                bet: actual_bb,
                idx: bb_idx,
                forced_bet_type: super::action::ForcedBetType::BigBlind,
                player_stack: self.game_state.stacks[bb_idx],
            }))
            .await;
        }

        self.run_betting_round().await;
        self.advance_round().await;
    }

    async fn deal_flop(&mut self) {
        self.deal_comunity_cards(3).await;
        self.advance_round().await;
    }

    async fn flop(&mut self) {
        self.run_betting_round().await;
        self.advance_round().await;
    }

    async fn deal_turn(&mut self) {
        self.deal_comunity_cards(1).await;
        self.advance_round().await;
    }

    async fn turn(&mut self) {
        self.run_betting_round().await;
        self.advance_round().await;
    }

    async fn deal_river(&mut self) {
        self.deal_comunity_cards(1).await;
        self.advance_round().await;
    }
    async fn river(&mut self) {
        self.run_betting_round().await;
        self.advance_round().await;
    }

    async fn showdown(&mut self) {
        self.distribute_pots(true).await;
        self.end_game().await;
    }

    async fn distribute_pots(&mut self, showdown: bool) {
        let awards = crate::arena::pot::planned_awards(&self.game_state);
        let mut last_pot = None;
        for award in awards {
            if award.refund {
                self.game_state.return_uncalled_bet(award.idx, award.amount);
                self.record_action(Action::ReturnUncalledBet(
                    crate::arena::action::UncalledBetPayload {
                        idx: award.idx,
                        amount: award.amount,
                    },
                ))
                .await;
                continue;
            }
            if last_pot != Some(award.pot_index) {
                self.game_state.rake_collected += award.rake;
                last_pot = Some(award.pot_index);
            }
            self.game_state.award(award.idx, award.amount);
            self.record_action(Action::Award(AwardPayload {
                idx: award.idx,
                total_pot: award.gross,
                award_amount: award.amount,
                rank: showdown.then(|| self.game_state.hands[award.idx].rank()),
                hand: showdown.then_some(self.game_state.hands[award.idx]),
            }))
            .await;
        }
        assert_eq!(
            self.game_state
                .player_winnings
                .iter()
                .sum::<crate::arena::Chips>()
                + self.game_state.rake_collected,
            self.game_state.total_pot
        );
    }

    async fn deal_player_cards(&mut self, num_cards: usize) {
        let new_hand = self.deal_cards(num_cards);
        for c in &new_hand {
            self.record_action(Action::DealStartingHand(DealStartingHandPayload {
                card: *c,
                idx: self.game_state.to_act_idx(),
            }))
            .await;
        }

        let idx = self.game_state.to_act_idx();
        self.game_state.hands[idx].extend(new_hand);
    }

    async fn deal_comunity_cards(&mut self, num_cards: usize) {
        let community_cards = self.deal_cards(num_cards);
        for c in &community_cards {
            self.record_action(Action::DealCommunity(*c)).await;
        }
        // Add all the cards to the hands as well.
        for h in &mut self.game_state.hands {
            h.extend(community_cards.to_owned());
        }
        // Move the community_cards into the game_state board.
        self.game_state.board.extend(community_cards);
    }

    /// Pull num_cards from the deck and return them. A single deal is small (≤5:
    /// flop, turn, river, or 2 hole cards), so an inline `SmallVec` keeps it off
    /// the heap.
    fn deal_cards(&mut self, num_cards: usize) -> SmallVec<[Card; 5]> {
        // Reborrow the boxed RNG as a sized `&mut` reference so it satisfies
        // the `R: Rng` (Sized) bound on `Deck::deal`.
        let deck = &mut self.deck;
        let mut rng = &mut *self.rng;
        let mut cards: SmallVec<[Card; 5]> = (0..num_cards)
            .map(|_| deck.deal(&mut rng).unwrap())
            .collect();

        // Keep the cards sorted in min to max order
        // this keeps the number of permutations down since
        // its now AsKsKd is the same as KdAsKs after sorting.
        cards.sort();

        cards
    }

    /// This runs betting for the round to completion. It will run until
    /// everyone has acted or until the round has been completed because no one
    /// can act anymore.
    ///
    /// When fewer than 2 players are active (the rest are all-in or folded),
    /// we skip the round since the lone active player has no one to bet against.
    /// However, if there's an unmatched bet (e.g., an all-in or a blind that
    /// hasn't been called), active players must still fold or call.
    async fn run_betting_round(&mut self) {
        let current_round = self.game_state.round;
        if self.game_state.player_active.count() < 2 {
            let has_unmatched_bet = self.game_state.player_active.ones().any(|idx| {
                self.game_state.round_data.player_bet[idx] < self.game_state.round_data.bet
            });
            if !has_unmatched_bet {
                return;
            }
        }
        while self.needs_action() && self.game_state.round == current_round {
            self.run_single_agent().await;
        }
    }

    fn needs_action(&self) -> bool {
        let active_players = self.game_state.player_active;

        let players_needing_action = self.game_state.round_data.needs_action;

        let active_players_needing_action = active_players & players_needing_action;
        !active_players_needing_action.empty()
    }

    /// Run the next agent in the game state to act.
    async fn run_single_agent(&mut self) {
        let idx = self.game_state.to_act_idx();
        let action = self.agents[idx].act(self.id, &self.game_state).await;

        event!(Level::TRACE, ?action, idx, "run_agent");
        self.run_agent_action(action).await;
    }

    /// Given the action that an agent wants to take, this function will
    /// determine if the action is valid and then apply it to the game state.
    /// If the action is invalid, the agent will be forced to fold.
    pub async fn run_agent_action(&mut self, agent_action: AgentAction) {
        event!(Level::TRACE, ?agent_action, "run_agent_action");

        let idx = self.game_state.to_act_idx();
        let starting_bet = self.game_state.current_round_bet();
        let starting_player_bet = self.game_state.current_round_player_bet(idx);
        let starting_min_raise = self.game_state.current_round_min_raise();
        let starting_pot = self.game_state.total_pot;

        match agent_action {
            AgentAction::Fold => {
                // It plays hell on verifying games if there are players that quit when there's
                // no money in being asked for. For now do exact equality, but
                // this might be a place we should use approxiate compaison
                // crate.
                if starting_player_bet == starting_bet {
                    event!(Level::WARN, "fold_error");

                    let new_action = AgentAction::Bet(starting_bet);

                    self.game_state.do_bet(starting_bet, false).unwrap();
                    self.record_action(Action::FailedAction(FailedActionPayload {
                        action: agent_action,
                        result: PlayedActionPayload {
                            action: new_action,
                            player_stack: self.game_state.stacks[idx],
                            idx,
                            round: self.game_state.round,
                            starting_bet,
                            final_bet: starting_bet,
                            starting_min_raise,
                            final_min_raise: self.game_state.current_round_min_raise(),
                            starting_player_bet,
                            final_player_bet: starting_player_bet,
                            players_active: self.game_state.player_active,
                            players_all_in: self.game_state.player_all_in,
                            starting_pot,
                            final_pot: self.game_state.total_pot,
                        },
                    }))
                    .await;
                } else {
                    self.record_action(Action::PlayedAction(PlayedActionPayload {
                        action: agent_action,
                        player_stack: self.game_state.stacks[idx],
                        idx,
                        round: self.game_state.round,
                        // None of the bets move if this is a fold
                        starting_bet,
                        final_bet: starting_bet,
                        starting_min_raise,
                        final_min_raise: self.game_state.current_round_min_raise(),
                        starting_player_bet,
                        final_player_bet: starting_player_bet,
                        players_active: self.game_state.player_active,
                        players_all_in: self.game_state.player_all_in,
                        starting_pot,
                        final_pot: self.game_state.total_pot,
                    }))
                    .await;
                    self.player_fold().await;
                }
            }
            AgentAction::Bet(bet_amount) => {
                // Check if this raise would exceed the max raises per round limit
                let is_raise = bet_amount > starting_bet;
                let is_raise_capped = self.game_state.is_raise_capped();

                // All-in is always allowed regardless of raise cap
                // A bet is all-in if it uses the player's entire remaining stack
                let stack = self.game_state.current_player_stack();
                let is_all_in = bet_amount >= starting_player_bet + stack;

                // If this is a raise but raises are capped (and not all-in), convert to call
                let effective_bet_amount = if is_raise && is_raise_capped && !is_all_in {
                    starting_bet
                } else {
                    bet_amount
                };

                let bet_result = self.game_state.do_bet(effective_bet_amount, false);

                match bet_result {
                    Err(error) => {
                        // If the agent failed to give us a good bet then they lose this round.
                        //
                        // We emit the error as an event with detailed context for debugging.
                        // Assume that game_state.do_bet() will have changed nothing since it
                        // errored out Add an action that shows the user was
                        // force folded. Actually fold the user
                        event!(
                            Level::WARN,
                            ?error,
                            agent_name = %self.agents[idx].name(),
                            bet_amount = bet_amount,
                            current_bet = starting_bet,
                            player_bet = starting_player_bet,
                            player_stack = self.game_state.stacks[idx],
                            min_raise = starting_min_raise,
                            "bet_error"
                        );

                        // Record this errant action
                        self.record_action(Action::FailedAction(FailedActionPayload {
                            action: agent_action,
                            result: PlayedActionPayload {
                                action: AgentAction::Fold,
                                player_stack: self.game_state.stacks[idx],
                                idx,
                                round: self.game_state.round,
                                starting_bet,
                                final_bet: starting_bet,
                                starting_min_raise,
                                final_min_raise: self.game_state.current_round_min_raise(),
                                starting_player_bet,
                                final_player_bet: starting_player_bet,
                                players_active: self.game_state.player_active,
                                players_all_in: self.game_state.player_all_in,
                                // What's the pot worth
                                starting_pot,
                                final_pot: self.game_state.total_pot,
                            },
                        }))
                        .await;

                        // Actually fold the user
                        self.player_fold().await;
                    }
                    Ok(_added) => {
                        let player_bet = self.game_state.current_round_player_bet(idx);

                        let new_action = match agent_action {
                            AgentAction::Bet(_) => AgentAction::Bet(player_bet),
                            AgentAction::Call => AgentAction::Call,
                            AgentAction::Fold => AgentAction::Fold,
                            AgentAction::AllIn => AgentAction::AllIn,
                        };

                        let payload = PlayedActionPayload {
                            action: new_action,
                            player_stack: self.game_state.stacks[idx],
                            idx,
                            round: self.game_state.round,
                            starting_bet,
                            final_bet: self.game_state.current_round_bet(),
                            starting_min_raise,
                            final_min_raise: self.game_state.current_round_min_raise(),
                            starting_player_bet,
                            final_player_bet: player_bet,
                            players_active: self.game_state.player_active,
                            players_all_in: self.game_state.player_all_in,
                            starting_pot,
                            final_pot: self.game_state.total_pot,
                        };

                        // If the raise was capped (and not all-in), record as failed action
                        if is_raise && is_raise_capped && !is_all_in {
                            event!(
                                Level::DEBUG,
                                agent_name = %self.agents[idx].name(),
                                original_bet = bet_amount,
                                effective_bet = effective_bet_amount,
                                raise_count = self.game_state.round_data.total_raise_count,
                                "raise_capped_to_call"
                            );
                            self.record_action(Action::FailedAction(FailedActionPayload {
                                action: agent_action,
                                result: payload,
                            }))
                            .await;
                        } else {
                            self.record_action(Action::PlayedAction(payload)).await;
                        }
                    }
                }
            }
            AgentAction::Call => {
                let call_amount = self.game_state.current_round_bet();
                let bet_result = self.game_state.do_bet(call_amount, false);

                match bet_result {
                    Err(error) => {
                        // If the agent failed to give us a good bet then they lose this round.
                        //
                        // We emit the error as an event with detailed context for debugging.
                        // Assume that game_state.do_bet() will have changed nothing since it
                        // errored out Add an action that shows the user was
                        // force folded. Actually fold the user
                        event!(
                            Level::WARN,
                            ?error,
                            agent_name = %self.agents[idx].name(),
                            current_bet = starting_bet,
                            player_bet = starting_player_bet,
                            player_stack = self.game_state.stacks[idx],
                            min_raise = starting_min_raise,
                            "bet_error"
                        );

                        // Record this errant action
                        self.record_action(Action::FailedAction(FailedActionPayload {
                            action: agent_action,
                            result: PlayedActionPayload {
                                action: AgentAction::Fold,
                                player_stack: self.game_state.stacks[idx],
                                idx,
                                round: self.game_state.round,
                                starting_bet,
                                final_bet: starting_bet,
                                starting_min_raise,
                                final_min_raise: self.game_state.current_round_min_raise(),
                                starting_player_bet,
                                final_player_bet: starting_player_bet,
                                players_active: self.game_state.player_active,
                                players_all_in: self.game_state.player_all_in,
                                // What's the pot worth
                                starting_pot,
                                final_pot: self.game_state.total_pot,
                            },
                        }))
                        .await;

                        // Actually fold the user
                        self.player_fold().await;
                    }
                    Ok(_added) => {
                        let player_bet = self.game_state.current_round_player_bet(idx);

                        let new_action = match agent_action {
                            AgentAction::Call => AgentAction::Call,
                            AgentAction::Bet(_) => AgentAction::Bet(player_bet),
                            AgentAction::Fold => AgentAction::Fold,
                            AgentAction::AllIn => AgentAction::AllIn,
                        };
                        // If the game_state.do_bet function returned Ok then
                        // the state is already changed so record the action as played.
                        self.record_action(Action::PlayedAction(PlayedActionPayload {
                            action: new_action,
                            player_stack: self.game_state.stacks[idx],
                            idx,
                            round: self.game_state.round,
                            starting_bet,
                            final_bet: self.game_state.current_round_bet(),
                            starting_min_raise,
                            final_min_raise: self.game_state.current_round_min_raise(),
                            starting_player_bet,
                            final_player_bet: player_bet,
                            // Keep track of who's in a
                            players_active: self.game_state.player_active,
                            players_all_in: self.game_state.player_all_in,
                            // What's the pot worth
                            starting_pot,
                            final_pot: self.game_state.total_pot,
                        }))
                        .await;
                    }
                }
            }
            AgentAction::AllIn => {
                let all_in_amount = self.game_state.current_round_current_player_bet()
                    + self.game_state.current_player_stack();
                let bet_result = self.game_state.do_bet(all_in_amount, false);

                match bet_result {
                    Err(error) => {
                        // If the agent failed to give us a good bet then they lose this round.
                        //
                        // We emit the error as an event with detailed context for debugging.
                        // Assume that game_state.do_bet() will have changed nothing since it
                        // errored out Add an action that shows the user was
                        // force folded. Actually fold the user
                        event!(
                            Level::WARN,
                            ?error,
                            agent_name = %self.agents[idx].name(),
                            current_bet = starting_bet,
                            player_bet = starting_player_bet,
                            player_stack = self.game_state.stacks[idx],
                            min_raise = starting_min_raise,
                            "bet_error"
                        );

                        // Record this errant action
                        self.record_action(Action::FailedAction(FailedActionPayload {
                            action: agent_action,
                            result: PlayedActionPayload {
                                action: AgentAction::Fold,
                                player_stack: self.game_state.stacks[idx],
                                idx,
                                round: self.game_state.round,
                                starting_bet,
                                final_bet: starting_bet,
                                starting_min_raise,
                                final_min_raise: self.game_state.current_round_min_raise(),
                                starting_player_bet,
                                final_player_bet: starting_player_bet,
                                players_active: self.game_state.player_active,
                                players_all_in: self.game_state.player_all_in,
                                // What's the pot worth
                                starting_pot,
                                final_pot: self.game_state.total_pot,
                            },
                        }))
                        .await;

                        // Actually fold the user
                        self.player_fold().await;
                    }
                    Ok(_added) => {
                        let player_bet = self.game_state.current_round_player_bet(idx);

                        let new_action = match agent_action {
                            AgentAction::Bet(_) => AgentAction::Bet(player_bet),
                            AgentAction::Fold => AgentAction::Fold,
                            AgentAction::Call => AgentAction::Call,
                            AgentAction::AllIn => AgentAction::AllIn,
                        };
                        // If the game_state.do_bet function returned Ok then
                        // the state is already changed so record the action as played.
                        self.record_action(Action::PlayedAction(PlayedActionPayload {
                            action: new_action,
                            player_stack: self.game_state.stacks[idx],
                            idx,
                            round: self.game_state.round,
                            starting_bet,
                            final_bet: self.game_state.current_round_bet(),
                            starting_min_raise,
                            final_min_raise: self.game_state.current_round_min_raise(),
                            starting_player_bet,
                            final_player_bet: player_bet,
                            // Keep track of who's in a
                            players_active: self.game_state.player_active,
                            players_all_in: self.game_state.player_all_in,

                            // What's the pot worth
                            starting_pot,
                            final_pot: self.game_state.total_pot,
                        }))
                        .await;
                    }
                }
            }
        }
    }

    #[instrument(level = "trace", skip(self))]
    async fn player_fold(&mut self) {
        self.game_state.fold();
        let left = self.game_state.player_active | self.game_state.player_all_in;

        // If there's only one person left then they win.
        // If there's no one left, and one person went all in they win.
        //
        if left.count() <= 1 {
            if !left.empty() {
                self.distribute_pots(false).await;
            }

            self.end_game().await;
        }
    }

    #[instrument(level = "trace", skip(self))]
    async fn end_game(&mut self) {
        let current_round = self.game_state.round;
        self.game_state.complete();
        if current_round != self.game_state.round {
            self.record_action(Action::RoundAdvance(self.game_state.round))
                .await;
        }
    }

    #[instrument(level = "trace", skip(self))]
    async fn advance_round(&mut self) {
        let current_round = self.game_state.round;
        self.game_state.advance_round();
        if self.game_state.round != current_round {
            self.record_action(Action::RoundAdvance(self.game_state.round))
                .await;
        }
    }

    // Make sure all modifications to game_state are complete before calling
    // `record_action`. Critical for deterministic replays.
    async fn record_action(&mut self, action: Action) {
        event!(Level::TRACE, action = ?action, "add_action");
        let panic_on_error = self.panic_on_historian_error;
        let id = self.id;

        // Walk historians in order; drop any that error. Index walk (not
        // retain_mut) because the body awaits.
        let mut i = 0;
        while i < self.historians.len() {
            let result = {
                let game_state = &self.game_state;
                self.historians[i]
                    .record_action(id, game_state, &action)
                    .await
            };
            match result {
                Ok(()) => i += 1,
                Err(error) => {
                    event!(Level::ERROR, ?error, "historian_error");
                    if panic_on_error {
                        panic!(
                            "Historian error {}\naction={:?}\ngame_state = {:?}",
                            error, action, self.game_state
                        );
                    }
                    self.historians.remove(i);
                }
            }
        }
    }
}

impl fmt::Debug for HoldemSimulation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HoldemSimulation")
            .field("game_state", &self.game_state)
            .field("deck", &self.deck.len())
            .field("historians", &self.historians.len())
            .field("agents", &self.agents.len())
            .field("panic_on_historian_error", &self.panic_on_historian_error)
            .finish()
    }
}
