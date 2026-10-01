//! Exact rake arithmetic. Room schedules live in a separate data layer.
pub mod schedule;

use super::money::Chips;
use thiserror::Error;

/// A validated rational rate between zero and one (inclusive).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RakeRate {
    numerator: u32,
    denominator: u32,
}
impl RakeRate {
    pub fn new(numerator: u32, denominator: u32) -> Result<Self, RakeConfigError> {
        if denominator == 0 || numerator > denominator {
            return Err(RakeConfigError::InvalidRate {
                numerator,
                denominator,
            });
        }
        Ok(Self {
            numerator,
            denominator,
        })
    }
    pub const fn numerator(self) -> u32 {
        self.numerator
    }
    pub const fn denominator(self) -> u32 {
        self.denominator
    }
    pub const fn zero() -> Self {
        Self {
            numerator: 0,
            denominator: 1,
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RakeRounding {
    HalfToEven,
    Floor,
    Ceil,
}
impl RakeRounding {
    fn round(self, numerator: i128, denominator: i128) -> i128 {
        let quotient = numerator / denominator;
        let remainder = numerator % denominator;
        let up = match self {
            Self::Floor => false,
            Self::Ceil => remainder != 0,
            Self::HalfToEven => {
                remainder * 2 > denominator || (remainder * 2 == denominator && quotient % 2 != 0)
            }
        };
        quotient + i128::from(up)
    }
}
/// One configuration consumed by the engine, with a cap shared by the hand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RakeConfig {
    pub rate: RakeRate,
    /// None means unlimited; negative caps are invalid.
    pub cap: Option<Chips>,
    /// Baseline no-flop-no-drop rule. Some rooms (currently GG) make an
    /// explicit exception for preflop pots that reach a 3-bet or higher.
    pub no_flop_no_drop: bool,
    /// When true, a preflop 3-bet+ pot is rake-eligible even though
    /// `no_flop_no_drop` is enabled.
    pub preflop_three_bet_rake: bool,
    pub rounding: RakeRounding,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum RakeConfigError {
    #[error("invalid rational rake rate {numerator}/{denominator}")]
    InvalidRate { numerator: u32, denominator: u32 },
    #[error("rake cap must be non-negative, got {0}")]
    InvalidCap(Chips),
}
impl RakeConfig {
    pub fn new(
        rate: RakeRate,
        cap: Option<Chips>,
        no_flop_no_drop: bool,
        rounding: RakeRounding,
    ) -> Result<Self, RakeConfigError> {
        let config = Self {
            rate,
            cap,
            no_flop_no_drop,
            preflop_three_bet_rake: false,
            rounding,
        };
        config.validate()?;
        Ok(config)
    }
    /// Enable a room-specific exception that rakes preflop pots once the
    /// betting reaches a 3-bet or higher.
    pub const fn with_preflop_three_bet_rake(mut self, enabled: bool) -> Self {
        self.preflop_three_bet_rake = enabled;
        self
    }

    pub fn validate(&self) -> Result<(), RakeConfigError> {
        if let Some(cap) = self.cap
            && cap < 0
        {
            return Err(RakeConfigError::InvalidCap(cap));
        }
        Ok(())
    }
    pub const fn none() -> Self {
        Self {
            rate: RakeRate::zero(),
            cap: None,
            no_flop_no_drop: false,
            preflop_three_bet_rake: false,
            rounding: RakeRounding::HalfToEven,
        }
    }
    /// Gross slice -> rational amount -> rounding -> remaining hand cap.
    /// Multiplication is widened to i128; no money crosses a float boundary.
    /// This compatibility wrapper assumes the hand did not reach a preflop
    /// 3-bet; the arena uses `calculate_with_hand_context` below.
    pub fn calculate(
        &self,
        pot_amount: Chips,
        flop_dealt: bool,
        already_collected: Chips,
    ) -> Chips {
        self.calculate_with_hand_context(pot_amount, flop_dealt, false, already_collected)
    }

    /// Calculate rake with the room-relevant preflop betting context.
    pub fn calculate_with_hand_context(
        &self,
        pot_amount: Chips,
        flop_dealt: bool,
        preflop_three_bet_or_higher: bool,
        already_collected: Chips,
    ) -> Chips {
        assert!(pot_amount >= 0 && already_collected >= 0);
        self.validate().expect("invalid rake configuration");
        let preflop_exception =
            self.preflop_three_bet_rake && preflop_three_bet_or_higher;
        if self.no_flop_no_drop && !flop_dealt && !preflop_exception {
            return 0;
        }
        let amount = self.rounding.round(
            i128::from(pot_amount) * i128::from(self.rate.numerator),
            i128::from(self.rate.denominator),
        ) as Chips;
        let remaining_cap = self.cap.map_or(Chips::MAX, |cap| {
            cap.saturating_sub(already_collected).max(0)
        });
        amount.min(remaining_cap).min(pot_amount)
    }
}
impl Default for RakeConfig {
    fn default() -> Self {
        Self::none()
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn config(n: u32, d: u32, rounding: RakeRounding) -> RakeConfig {
        RakeConfig::new(RakeRate::new(n, d).unwrap(), None, false, rounding).unwrap()
    }
    #[test]
    fn exact_percentages_and_wide_multiplication() {
        assert_eq!(RakeConfig::none().calculate(100, true, 0), 0);
        assert_eq!(
            config(5, 100, RakeRounding::Floor).calculate(100, true, 0),
            5
        );
        assert_eq!(
            config(45, 1000, RakeRounding::Floor).calculate(2000, true, 0),
            90
        );
        assert_eq!(
            config(55, 1000, RakeRounding::Floor).calculate(2000, true, 0),
            110
        );
        assert_eq!(
            config(u32::MAX, u32::MAX, RakeRounding::Ceil).calculate(Chips::MAX, true, 0),
            Chips::MAX
        );
    }
    #[test]
    fn quotient_remainder_rounding() {
        assert_eq!(config(1, 2, RakeRounding::Floor).calculate(3, true, 0), 1);
        assert_eq!(config(1, 2, RakeRounding::Ceil).calculate(3, true, 0), 2);
        let even = config(1, 2, RakeRounding::HalfToEven);
        assert_eq!(even.calculate(1, true, 0), 0);
        assert_eq!(even.calculate(3, true, 0), 2);
        assert_eq!(even.calculate(5, true, 0), 2);
        assert_eq!(even.calculate(7, true, 0), 4);
        assert_eq!(
            config(1, 3, RakeRounding::HalfToEven).calculate(4, true, 0),
            1
        );
        assert_eq!(
            config(1, 3, RakeRounding::HalfToEven).calculate(5, true, 0),
            2
        );
    }
    #[test]
    fn shared_cap_and_no_flop_no_drop() {
        let c = RakeConfig::new(
            RakeRate::new(1, 10).unwrap(),
            Some(6),
            true,
            RakeRounding::HalfToEven,
        )
        .unwrap();
        assert_eq!(c.calculate(40, false, 0), 0);
        assert_eq!(c.calculate(40, true, 0), 4);
        assert_eq!(c.calculate(40, true, 4), 2);
        assert_eq!(c.calculate(40, true, 6), 0);
        assert_eq!(c.calculate(40, true, 10), 0);
        for pot in 0..500 {
            for rounding in [
                RakeRounding::Floor,
                RakeRounding::Ceil,
                RakeRounding::HalfToEven,
            ] {
                let rake = config(99, 100, rounding).calculate(pot, true, 0);
                assert!((0..=pot).contains(&rake));
            }
        }
    }
    #[test]
    fn preflop_three_bet_exception_is_explicit() {
        let c = RakeConfig::new(
            RakeRate::new(5, 100).unwrap(),
            None,
            true,
            RakeRounding::Ceil,
        )
        .unwrap()
        .with_preflop_three_bet_rake(true);

        assert_eq!(c.calculate_with_hand_context(100, false, false, 0), 0);
        assert_eq!(c.calculate_with_hand_context(100, false, true, 0), 5);
        assert_eq!(c.calculate_with_hand_context(21, false, true, 0), 2);
        assert_eq!(c.calculate_with_hand_context(21, true, false, 0), 2);
    }

    #[test]
    fn invalid_configs_are_rejected() {
        assert!(RakeRate::new(1, 0).is_err());
        assert!(RakeRate::new(101, 100).is_err());
        assert!(RakeConfig::new(RakeRate::zero(), Some(-1), false, RakeRounding::Floor).is_err());
    }
}
