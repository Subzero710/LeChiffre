use std::collections::HashSet;
use std::fmt;

/// Betting street visible to every player.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Street {
    Preflop,
    Flop,
    Turn,
    River,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Rank {
    Two,
    Three,
    Four,
    Five,
    Six,
    Seven,
    Eight,
    Nine,
    Ten,
    Jack,
    Queen,
    King,
    Ace,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Suit {
    Clubs,
    Diamonds,
    Hearts,
    Spades,
}

/// A community card visible to every player.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PublicCard {
    pub rank: Rank,
    pub suit: Suit,
}

/// Semantic action observed at the table, with check/call and bet/raise kept distinct.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ActionKind {
    Fold,
    Check,
    Call,
    Bet,
    Raise,
    AllIn,
}

/// Public poker state immediately before a player makes a decision.
/// Monetary values are expressed in big blinds.
#[derive(Debug, Clone, PartialEq)]
pub struct DecisionContext {
    pub street: Street,
    pub board: Vec<PublicCard>,

    pub actor_seat: usize,
    pub dealer_seat: usize,
    pub players_at_table: usize,
    pub players_active: usize,

    pub pot_bb: f32,
    pub actor_stack_bb: f32,
    pub effective_stack_bb: f32,
    pub actor_street_commitment_bb: f32,
    pub current_bet_bb: f32,
    pub to_call_bb: f32,
    pub min_raise_to_bb: Option<f32>,

    /// Number of voluntary actions that increased the street bet before this
    /// decision (open bet/raise/3-bet/etc.). Preflop, 2 means the player is
    /// facing a 3-bet after an open raise.
    pub bet_or_raise_count_this_street: u8,
}

impl DecisionContext {
    pub fn validate(&self) -> Result<(), BehaviourEventError> {
        if self.players_at_table < 2 {
            return Err(BehaviourEventError::InvalidPlayerCount);
        }
        if self.players_active == 0 || self.players_active > self.players_at_table {
            return Err(BehaviourEventError::InvalidActivePlayerCount);
        }
        if self.actor_seat >= self.players_at_table || self.dealer_seat >= self.players_at_table {
            return Err(BehaviourEventError::InvalidSeat);
        }

        let expected_board_len = match self.street {
            Street::Preflop => 0,
            Street::Flop => 3,
            Street::Turn => 4,
            Street::River => 5,
        };
        if self.board.len() != expected_board_len {
            return Err(BehaviourEventError::InvalidBoardLength {
                street: self.street,
                expected: expected_board_len,
                actual: self.board.len(),
            });
        }

        let unique_cards: HashSet<_> = self.board.iter().copied().collect();
        if unique_cards.len() != self.board.len() {
            return Err(BehaviourEventError::DuplicateBoardCard);
        }

        validate_non_negative("pot_bb", self.pot_bb)?;
        validate_non_negative("actor_stack_bb", self.actor_stack_bb)?;
        validate_non_negative("effective_stack_bb", self.effective_stack_bb)?;
        validate_non_negative(
            "actor_street_commitment_bb",
            self.actor_street_commitment_bb,
        )?;
        validate_non_negative("current_bet_bb", self.current_bet_bb)?;
        validate_non_negative("to_call_bb", self.to_call_bb)?;

        if let Some(value) = self.min_raise_to_bb {
            validate_non_negative("min_raise_to_bb", value)?;
        }

        // Keep redundant betting fields internally consistent.
        let expected_to_call =
            (self.current_bet_bb - self.actor_street_commitment_bb).max(0.0);
        if !approx_eq(self.to_call_bb, expected_to_call) {
            return Err(BehaviourEventError::InconsistentToCall {
                expected: expected_to_call,
                actual: self.to_call_bb,
            });
        }

        Ok(())
    }
}

/// Behaviour that is actually observable when the player answers a decision.
#[derive(Debug, Clone, PartialEq)]
pub struct ObservedAction {
    pub kind: ActionKind,

    /// Chips newly committed by this action, in big blinds.
    /// Zero for fold/check.
    pub amount_added_bb: f32,

    /// Player's total commitment on the current street after the action.
    /// This remains meaningful for fold/check because a player may already have
    /// chips committed on that street (for example the big blind checking).
    pub street_total_after_bb: f32,

    /// Wall-clock decision latency once the action was available, in ms.
    /// `None` means the data source did not expose timing information.
    pub decision_time_ms: Option<u32>,
}

impl ObservedAction {
    fn validate_against(&self, context: &DecisionContext) -> Result<(), BehaviourEventError> {
        validate_non_negative("amount_added_bb", self.amount_added_bb)?;
        validate_non_negative("street_total_after_bb", self.street_total_after_bb)?;

        if matches!(self.kind, ActionKind::Fold | ActionKind::Check)
            && !approx_eq(self.amount_added_bb, 0.0)
        {
            return Err(BehaviourEventError::UnexpectedAmountForAction(self.kind));
        }

        let expected_total = context.actor_street_commitment_bb + self.amount_added_bb;
        if !approx_eq(self.street_total_after_bb, expected_total) {
            return Err(BehaviourEventError::InconsistentStreetTotal {
                expected: expected_total,
                actual: self.street_total_after_bb,
            });
        }

        Ok(())
    }
}

/// One behavioural observation: public state before the decision and the observed response.
#[derive(Debug, Clone, PartialEq)]
pub struct BehaviourEvent {
    pub hand_id: u128,
    pub sequence_index: u32,
    pub context: DecisionContext,
    pub action: ObservedAction,
}

impl BehaviourEvent {
    pub fn new(
        hand_id: u128,
        sequence_index: u32,
        context: DecisionContext,
        action: ObservedAction,
    ) -> Result<Self, BehaviourEventError> {
        context.validate()?;
        action.validate_against(&context)?;

        Ok(Self {
            hand_id,
            sequence_index,
            context,
            action,
        })
    }

    pub fn actor_seat(&self) -> usize {
        self.context.actor_seat
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum BehaviourEventError {
    InvalidPlayerCount,
    InvalidActivePlayerCount,
    InvalidSeat,
    InvalidBoardLength {
        street: Street,
        expected: usize,
        actual: usize,
    },
    DuplicateBoardCard,
    InvalidNumber { field: &'static str },
    NegativeNumber { field: &'static str },
    UnexpectedAmountForAction(ActionKind),
    InconsistentToCall { expected: f32, actual: f32 },
    InconsistentStreetTotal { expected: f32, actual: f32 },
}

impl fmt::Display for BehaviourEventError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPlayerCount => write!(f, "players_at_table must be at least 2"),
            Self::InvalidActivePlayerCount => write!(
                f,
                "players_active must be between 1 and players_at_table"
            ),
            Self::InvalidSeat => write!(f, "actor/dealer seat is outside the table"),
            Self::InvalidBoardLength {
                street,
                expected,
                actual,
            } => write!(
                f,
                "{street:?} requires {expected} public cards, got {actual}"
            ),
            Self::DuplicateBoardCard => write!(f, "board contains a duplicate card"),
            Self::InvalidNumber { field } => write!(f, "{field} must be finite"),
            Self::NegativeNumber { field } => write!(f, "{field} must be non-negative"),
            Self::UnexpectedAmountForAction(action) => {
                write!(f, "{action:?} cannot add chips")
            }
            Self::InconsistentToCall { expected, actual } => {
                write!(f, "to_call_bb is {actual}, expected {expected}")
            }
            Self::InconsistentStreetTotal { expected, actual } => write!(
                f,
                "street_total_after_bb is {actual}, expected {expected}"
            ),
        }
    }
}

impl std::error::Error for BehaviourEventError {}

fn validate_non_negative(field: &'static str, value: f32) -> Result<(), BehaviourEventError> {
    if !value.is_finite() {
        return Err(BehaviourEventError::InvalidNumber { field });
    }
    if value < 0.0 {
        return Err(BehaviourEventError::NegativeNumber { field });
    }
    Ok(())
}

fn approx_eq(left: f32, right: f32) -> bool {
    (left - right).abs() <= 1e-4
}

#[cfg(test)]
mod tests {
    use super::*;

    fn context() -> DecisionContext {
        DecisionContext {
            street: Street::Preflop,
            board: vec![],
            actor_seat: 0,
            dealer_seat: 1,
            players_at_table: 2,
            players_active: 2,
            pot_bb: 2.5,
            actor_stack_bb: 99.0,
            effective_stack_bb: 99.0,
            actor_street_commitment_bb: 0.5,
            current_bet_bb: 2.0,
            to_call_bb: 1.5,
            min_raise_to_bb: Some(4.0),
            bet_or_raise_count_this_street: 1,
        }
    }

    #[test]
    fn accepts_valid_call_event() {
        let event = BehaviourEvent::new(
            7,
            0,
            context(),
            ObservedAction {
                kind: ActionKind::Call,
                amount_added_bb: 1.5,
                street_total_after_bb: 2.0,
                decision_time_ms: Some(18_400),
            },
        )
        .unwrap();

        assert_eq!(event.actor_seat(), 0);
        assert_eq!(event.action.decision_time_ms, Some(18_400));
    }

    #[test]
    fn rejects_nan_context_values() {
        let mut ctx = context();
        ctx.pot_bb = f32::NAN;

        assert_eq!(
            ctx.validate(),
            Err(BehaviourEventError::InvalidNumber { field: "pot_bb" })
        );
    }

    #[test]
    fn rejects_inconsistent_to_call() {
        let mut ctx = context();
        ctx.to_call_bb = 0.25;

        assert!(matches!(
            ctx.validate(),
            Err(BehaviourEventError::InconsistentToCall { .. })
        ));
    }

    #[test]
    fn check_keeps_existing_street_commitment() {
        let mut ctx = context();
        ctx.actor_street_commitment_bb = 2.0;
        ctx.current_bet_bb = 2.0;
        ctx.to_call_bb = 0.0;

        let event = BehaviourEvent::new(
            7,
            0,
            ctx,
            ObservedAction {
                kind: ActionKind::Check,
                amount_added_bb: 0.0,
                street_total_after_bb: 2.0,
                decision_time_ms: Some(250),
            },
        );

        assert!(event.is_ok());
    }
}
