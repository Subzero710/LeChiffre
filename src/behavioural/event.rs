use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct HandId(String);

impl HandId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn into_string(self) -> String {
        self.0
    }
}

impl From<String> for HandId {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl From<&str> for HandId {
    fn from(value: &str) -> Self {
        Self(value.to_owned())
    }
}

impl fmt::Display for HandId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

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

        Ok(())
    }
}

/// Behaviour that is actually observable when the player answers a decision.
#[derive(Debug, Clone, PartialEq)]
pub struct ObservedAction {
    pub kind: ActionKind,
    pub is_all_in: bool,

    /// Chips newly committed by this action, in big blinds.
    pub amount_added_bb: f32,

    /// Player's total commitment on the current street after the action.
    pub street_total_after_bb: f32,

    /// Wall-clock decision latency once the action was available, in ms.
    /// `None` means the data source did not expose timing information.
    pub decision_time_ms: Option<u32>,
}

impl ObservedAction {
    fn validate(&self) -> Result<(), BehaviourEventError> {
        validate_non_negative("amount_added_bb", self.amount_added_bb)?;
        validate_non_negative("street_total_after_bb", self.street_total_after_bb)?;
        Ok(())
    }
}

/// One behavioural observation: public state before the decision and the observed response.
#[derive(Debug, Clone, PartialEq)]
pub struct BehaviourEvent {
    pub hand_id: HandId,
    pub sequence_index: u32,
    pub context: DecisionContext,
    pub action: ObservedAction,
}

impl BehaviourEvent {
    pub fn new(
        hand_id: impl Into<HandId>,
        sequence_index: u32,
        context: DecisionContext,
        action: ObservedAction,
    ) -> Result<Self, BehaviourEventError> {
        context.validate()?;
        action.validate()?;

        Ok(Self {
            hand_id: hand_id.into(),
            sequence_index,
            context,
            action,
        })
    }

    pub fn actor_seat(&self) -> usize {
        self.context.actor_seat
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BehaviourEventError {
    InvalidNumber { field: &'static str },
    NegativeNumber { field: &'static str },
}

impl fmt::Display for BehaviourEventError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidNumber { field } => write!(f, "{field} must be finite"),
            Self::NegativeNumber { field } => write!(f, "{field} must be non-negative"),
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
    fn accepts_well_formed_event() {
        let event = BehaviourEvent::new(
            "hand-7",
            0,
            context(),
            ObservedAction {
                kind: ActionKind::Call,
                is_all_in: false,
                amount_added_bb: 1.5,
                street_total_after_bb: 2.0,
                decision_time_ms: Some(18_400),
            },
        )
        .unwrap();

        assert_eq!(event.hand_id.as_str(), "hand-7");
        assert_eq!(event.actor_seat(), 0);
        assert_eq!(event.action.decision_time_ms, Some(18_400));
    }

    #[test]
    fn accepts_missing_timing() {
        let event = BehaviourEvent::new(
            "hand-7",
            0,
            context(),
            ObservedAction {
                kind: ActionKind::Call,
                is_all_in: false,
                amount_added_bb: 1.5,
                street_total_after_bb: 2.0,
                decision_time_ms: None,
            },
        );

        assert!(event.is_ok());
    }

    #[test]
    fn does_not_revalidate_poker_state_or_action_semantics() {
        let event = BehaviourEvent::new(
            "hand-7",
            0,
            DecisionContext {
                street: Street::River,
                board: vec![],
                actor_seat: 99,
                dealer_seat: 88,
                players_at_table: 1,
                players_active: 0,
                pot_bb: 2.5,
                actor_stack_bb: 1.0,
                effective_stack_bb: 50.0,
                actor_street_commitment_bb: 20.0,
                current_bet_bb: 1.0,
                to_call_bb: 17.0,
                min_raise_to_bb: Some(0.0),
                bet_or_raise_count_this_street: 255,
            },
            ObservedAction {
                kind: ActionKind::Check,
                is_all_in: true,
                amount_added_bb: 3.0,
                street_total_after_bb: 42.0,
                decision_time_ms: None,
            },
        );

        assert!(event.is_ok());
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
    fn rejects_negative_action_amount() {
        let event = BehaviourEvent::new(
            "hand-7",
            0,
            context(),
            ObservedAction {
                kind: ActionKind::Fold,
                is_all_in: false,
                amount_added_bb: -1.0,
                street_total_after_bb: 0.0,
                decision_time_ms: None,
            },
        );

        assert_eq!(
            event,
            Err(BehaviourEventError::NegativeNumber {
                field: "amount_added_bb"
            })
        );
    }
}
