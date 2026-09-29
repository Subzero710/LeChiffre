use thiserror::Error;

/// Configuration for rake collected from a completed poker hand.
///
/// The rake is expressed as a percentage of each awarded pot slice, with a
/// cap shared by the whole hand. By default rake is disabled.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RakeConfig {
    /// Fraction of the pot collected as rake. `0.05` means 5%.
    pub percentage: f32,
    /// Maximum total rake collected during a single hand.
    pub cap: f32,
    /// If true, no rake is collected unless at least a flop was dealt.
    pub no_flop_no_drop: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum RakeConfigError {
    #[error("rake percentage must be between 0 and 1, got {0}")]
    InvalidPercentage(f32),
    #[error("rake cap must be non-negative, got {0}")]
    InvalidCap(f32),
}

impl RakeConfig {
    /// Create a validated rake configuration.
    pub fn new(
        percentage: f32,
        cap: f32,
        no_flop_no_drop: bool,
    ) -> Result<Self, RakeConfigError> {
        if !percentage.is_finite() || !(0.0..=1.0).contains(&percentage) {
            return Err(RakeConfigError::InvalidPercentage(percentage));
        }
        if cap.is_nan() || cap < 0.0 {
            return Err(RakeConfigError::InvalidCap(cap));
        }

        Ok(Self {
            percentage,
            cap,
            no_flop_no_drop,
        })
    }

    /// A zero-rake configuration.
    pub const fn none() -> Self {
        Self {
            percentage: 0.0,
            cap: f32::INFINITY,
            no_flop_no_drop: false,
        }
    }

    /// Return the rake to collect from `pot_amount`.
    ///
    /// `already_collected` is the amount already raked earlier in the same
    /// hand, so the cap is global to the hand rather than reset for side pots.
    pub fn calculate(
        &self,
        pot_amount: f32,
        flop_dealt: bool,
        already_collected: f32,
    ) -> f32 {
        if pot_amount <= 0.0 || self.percentage == 0.0 {
            return 0.0;
        }
        if self.no_flop_no_drop && !flop_dealt {
            return 0.0;
        }

        let remaining_cap = (self.cap - already_collected).max(0.0);
        (pot_amount * self.percentage)
            .min(remaining_cap)
            .min(pot_amount)
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

    #[test]
    fn default_collects_no_rake() {
        let rake = RakeConfig::default();
        assert_eq!(rake.calculate(100.0, true, 0.0), 0.0);
    }

    #[test]
    fn percentage_is_applied() {
        let rake = RakeConfig::new(0.05, f32::INFINITY, false).unwrap();
        assert_eq!(rake.calculate(100.0, true, 0.0), 5.0);
    }

    #[test]
    fn cap_is_shared_across_the_hand() {
        let rake = RakeConfig::new(0.10, 6.0, false).unwrap();
        assert_eq!(rake.calculate(40.0, true, 0.0), 4.0);
        assert_eq!(rake.calculate(40.0, true, 4.0), 2.0);
        assert_eq!(rake.calculate(40.0, true, 6.0), 0.0);
    }

    #[test]
    fn no_flop_no_drop_skips_preflop_fold() {
        let rake = RakeConfig::new(0.05, 10.0, true).unwrap();
        assert_eq!(rake.calculate(100.0, false, 0.0), 0.0);
        assert_eq!(rake.calculate(100.0, true, 0.0), 5.0);
    }

    #[test]
    fn invalid_configs_are_rejected() {
        assert!(RakeConfig::new(-0.01, 1.0, false).is_err());
        assert!(RakeConfig::new(1.01, 1.0, false).is_err());
        assert!(RakeConfig::new(0.05, -1.0, false).is_err());
    }
}
