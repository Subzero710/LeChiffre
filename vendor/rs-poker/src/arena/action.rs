use crate::arena::Chips;
use crate::core::{Card, Hand, PlayerBitSet, Rank};

use super::game_state::Round;

/// Represents an action that an agent can take in a game.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
pub enum AgentAction {
    /// Folds the current hand.
    Fold,
    /// Matches the current bet.
    Call,
    /// Bets the specified amount of money.
    Bet(Chips),
    /// Go all-in
    AllIn,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
/// The game has started.
pub struct GameStartPayload {
    pub ante: Chips,
    pub small_blind: Chips,
    pub big_blind: Chips,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PlayerSitPayload {
    pub idx: usize,
    pub player_stack: Chips,
    /// Optional agent name reported by the simulator so historians can preserve it.
    pub name: Option<String>,
}

/// Each player is dealt a card. This is the payload for the event.
#[derive(Debug, Clone, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DealStartingHandPayload {
    pub card: Card,
    pub idx: usize,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ForcedBetType {
    Ante,
    SmallBlind,
    BigBlind,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ForcedBetPayload {
    /// The actual amount that left the player's stack for the forced bet.
    /// When the player's stack is smaller than the nominal blind/ante, this
    /// is the (smaller) amount that was actually taken — not the requested
    /// nominal amount.
    pub bet: Chips,
    pub player_stack: Chips,
    pub idx: usize,
    pub forced_bet_type: ForcedBetType,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PlayedActionPayload {
    // The tried Action
    pub action: AgentAction,

    pub idx: usize,
    pub round: Round,
    pub player_stack: Chips,

    pub starting_pot: Chips,
    pub final_pot: Chips,

    pub starting_bet: Chips,
    pub final_bet: Chips,

    pub starting_min_raise: Chips,
    pub final_min_raise: Chips,

    pub starting_player_bet: Chips,
    pub final_player_bet: Chips,

    pub players_active: PlayerBitSet,
    pub players_all_in: PlayerBitSet,
}

impl PlayedActionPayload {
    pub fn raise_amount(&self) -> Chips {
        self.final_bet - self.starting_bet
    }
}

/// A player tried to play an action and failed
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FailedActionPayload {
    // The tried Action
    pub action: AgentAction,
    // The result action
    pub result: PlayedActionPayload,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AwardPayload {
    pub total_pot: Chips,
    pub award_amount: Chips,
    pub rank: Option<Rank>,
    pub hand: Option<Hand>,
    pub idx: usize,
}

/// Unmatched money returned to its contributor; this is neither a win nor rake.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct UncalledBetPayload {
    pub idx: usize,
    pub amount: Chips,
}

/// Represents an action that can happen in a game.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Action {
    GameStart(GameStartPayload),
    PlayerSit(PlayerSitPayload),
    DealStartingHand(DealStartingHandPayload),
    /// The round has advanced.
    RoundAdvance(Round),
    /// A player has played an action.
    PlayedAction(PlayedActionPayload),
    /// The player tried and failed to take some action.
    /// If the action failed then there is no PlayedAction event coming.
    ///
    /// Players can fail to fold when there's no money being wagered.
    /// Players can fail to bet when they bet an illegal amount.
    FailedAction(FailedActionPayload),

    /// A player/agent was forced to make a bet.
    ForcedBet(ForcedBetPayload),
    /// A community card has been dealt.
    DealCommunity(Card),
    /// There was some pot given to a player
    Award(AwardPayload),
    ReturnUncalledBet(UncalledBetPayload),
}

#[cfg(test)]
mod tests {
    use crate::core::PlayerBitSet;

    use super::*;

    #[test]
    fn test_bet() {
        let a = AgentAction::Bet(100);
        assert_eq!(AgentAction::Bet(100), a);
    }

    /// Verifies raise_amount correctly calculates the increase from starting to final bet.
    #[test]
    fn test_raise_amount_calculation() {
        let payload = PlayedActionPayload {
            action: AgentAction::Bet(100),
            idx: 0,
            round: Round::Preflop,
            player_stack: 500,
            starting_pot: 15,
            final_pot: 115,
            starting_bet: 10,
            final_bet: 30, // Raise from 10 to 30
            starting_min_raise: 10,
            final_min_raise: 20,
            starting_player_bet: 0,
            final_player_bet: 30,
            players_active: PlayerBitSet::new(2),
            players_all_in: PlayerBitSet::default(),
        };

        // raise_amount = final_bet - starting_bet = 30 - 10 = 20
        assert_eq!(payload.raise_amount(), 20);
    }

    /// Verifies raise_amount with a raise from 25 to 75 (50 raise amount).
    #[test]
    fn test_raise_amount_different_values() {
        let payload = PlayedActionPayload {
            action: AgentAction::Bet(50),
            idx: 0,
            round: Round::Flop,
            player_stack: 200,
            starting_pot: 50,
            final_pot: 100,
            starting_bet: 25,
            final_bet: 75,
            starting_min_raise: 25,
            final_min_raise: 50,
            starting_player_bet: 0,
            final_player_bet: 75,
            players_active: PlayerBitSet::new(3),
            players_all_in: PlayerBitSet::default(),
        };

        assert_eq!(payload.raise_amount(), 50);
    }

    /// Test raise_amount with zero raise (check scenario).
    #[test]
    fn test_raise_amount_no_raise() {
        let payload = PlayedActionPayload {
            action: AgentAction::Bet(10),
            idx: 1,
            round: Round::Preflop,
            player_stack: 100,
            starting_pot: 15,
            final_pot: 25,
            starting_bet: 10,
            final_bet: 10, // No raise - just called
            starting_min_raise: 10,
            final_min_raise: 10,
            starting_player_bet: 0,
            final_player_bet: 10,
            players_active: PlayerBitSet::new(2),
            players_all_in: PlayerBitSet::default(),
        };

        // No raise was made (just called), so raise_amount is 0
        assert_eq!(payload.raise_amount(), 0);
    }
}
