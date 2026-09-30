//! Published NLHE cash rake data, researched 2026-09-30. Prices are cents.
//! This module does not infer missing room policies or apply currency conversion.
use super::{RakeConfig, RakeRate, RakeRounding};
use crate::Chips;
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Platform {
    PokerStars,
    CoinPoker,
    GGPoker,
}
/// Explicit table product: two dealt players at a regular table is not a HU table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TableFormat {
    Regular,
    HeadsUp,
    SixMax,
    NineMax,
    FastFold,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RakeContext {
    pub platform: Platform,
    pub small_blind: Chips,
    pub big_blind: Chips,
    pub dealt_players: usize,
    pub table_format: TableFormat,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RakeSchedule {
    pub rate: RakeRate,
    pub cap: Chips,
    pub no_flop_no_drop: Option<bool>,
    pub rounding: Option<RakeRounding>,
    pub source: &'static str,
    /// Base percentage rake only. CoinPoker splash fees and GG promotional drops
    /// are distinct charges and must be handled explicitly by an importer.
    pub limitations: &'static str,
}
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ScheduleError {
    #[error("invalid dealt-player count {0}")]
    InvalidPlayers(usize),
    #[error("unsupported stakes or table product: {0:?}")]
    Unsupported(RakeContext),
    #[error("official {policy} policy is unverified; supply an explicit policy")]
    UnverifiedPolicy { policy: &'static str },
}
impl RakeSchedule {
    /// Explicit overrides fill unverified fields; verified policies cannot be
    /// silently replaced. No default rounding/NFND policy is invented.
    pub fn resolve(
        self,
        rounding: Option<RakeRounding>,
        no_flop_no_drop: Option<bool>,
    ) -> Result<RakeConfig, ScheduleError> {
        let rounding = self
            .rounding
            .or(rounding)
            .ok_or(ScheduleError::UnverifiedPolicy {
                policy: "rake rounding",
            })?;
        let no_flop_no_drop =
            self.no_flop_no_drop
                .or(no_flop_no_drop)
                .ok_or(ScheduleError::UnverifiedPolicy {
                    policy: "no-flop-no-drop",
                })?;
        Ok(
            RakeConfig::new(self.rate, Some(self.cap), no_flop_no_drop, rounding)
                .expect("validated published schedule"),
        )
    }
}
// SB, BB, rate numerator / 1000, cap by 2 / 3-4 / 5+ dealt players.
const STARS: &[(Chips, Chips, u32, [Chips; 3])] = &[
    (1, 2, 50, [100, 100, 100]),
    (2, 5, 50, [100, 100, 100]),
    (5, 10, 50, [100, 100, 100]),
    (10, 25, 45, [50, 100, 200]),
    (25, 50, 50, [75, 75, 200]),
    (50, 100, 50, [100, 100, 250]),
    (100, 200, 50, [125, 125, 275]),
    (200, 400, 50, [150, 150, 300]),
    (250, 500, 50, [150, 150, 300]),
    (300, 600, 50, [150, 150, 350]),
    (500, 1000, 45, [150, 150, 300]),
    (1000, 2000, 45, [175, 175, 300]),
    (2500, 5000, 45, [225, 200, 300]),
    (5000, 10000, 45, [250, 300, 500]),
];
const COIN: &[(Chips, Chips, [Chips; 3], Chips)] = &[
    (1, 2, [5, 12, 20], 6),
    (2, 5, [13, 30, 50], 15),
    (5, 10, [25, 60, 100], 30),
    (10, 25, [50, 120, 200], 60),
    (25, 50, [100, 240, 400], 120),
    (50, 100, [125, 300, 500], 150),
    (100, 200, [150, 360, 600], 180),
    (200, 500, [200, 500, 800], 240),
];
// GG's six-max caps differ for 3 versus 4 players. Nine-max microstakes also
// have distinct caps; the caller must identify the actual table product.
const GG: &[(Chips, Chips, [Chips; 4], [Chips; 4])] = &[
    (1, 2, [5, 10, 15, 20], [8, 15, 23, 30]),
    (2, 5, [13, 25, 38, 50], [19, 38, 56, 75]),
    (5, 10, [25, 50, 75, 100], [38, 75, 113, 150]),
    (10, 25, [50, 100, 150, 200], [63, 125, 188, 250]),
    (25, 50, [100, 200, 300, 400], [100, 200, 300, 400]),
    (50, 100, [125, 250, 375, 500], [125, 250, 375, 500]),
    (100, 200, [150, 300, 450, 600], [150, 300, 450, 600]),
    (200, 500, [200, 400, 600, 800], [200, 400, 600, 800]),
    (500, 1000, [250, 500, 750, 1000], [250, 500, 750, 1000]),
];
pub fn rake_schedule_for(c: RakeContext) -> Result<RakeSchedule, ScheduleError> {
    if !(2..=9).contains(&c.dealt_players) {
        return Err(ScheduleError::InvalidPlayers(c.dealt_players));
    }
    let bucket = if c.dealt_players == 2 {
        0
    } else if c.dealt_players <= 4 {
        1
    } else {
        2
    };
    let unsupported = || ScheduleError::Unsupported(c);
    match c.platform {
        Platform::PokerStars => {
            if c.table_format != TableFormat::Regular {
                return Err(unsupported());
            }
            let row = STARS
                .iter()
                .find(|r| (r.0, r.1) == (c.small_blind, c.big_blind))
                .ok_or_else(unsupported)?;
            Ok(RakeSchedule {
                rate: RakeRate::new(row.2, 1000).unwrap(),
                cap: row.3[bucket],
                no_flop_no_drop: Some(true),
                rounding: Some(RakeRounding::HalfToEven),
                source: "https://www.pokerstars.com/poker/room/rake/",
                limitations: "USD regular NLHE cash; excludes Zoom, other currencies and unlisted stakes",
            })
        }
        Platform::CoinPoker => {
            if !matches!(c.table_format, TableFormat::Regular | TableFormat::HeadsUp)
                || (c.table_format == TableFormat::HeadsUp && c.dealt_players != 2)
            {
                return Err(unsupported());
            }
            let row = COIN
                .iter()
                .find(|r| (r.0, r.1) == (c.small_blind, c.big_blind))
                .ok_or_else(unsupported)?;
            Ok(RakeSchedule {
                rate: RakeRate::new(5, 100).unwrap(),
                cap: if c.table_format == TableFormat::HeadsUp {
                    row.3
                } else {
                    row.2[bucket]
                },
                no_flop_no_drop: None,
                rounding: None,
                source: "https://coinpoker.com/rake/",
                limitations: "Published dollar-denominated base fee; excludes splash fee (0.1BB) and cash drops; rounding and NFND unverified",
            })
        }
        Platform::GGPoker => {
            let max = match c.table_format {
                TableFormat::SixMax => 6,
                TableFormat::NineMax => 9,
                _ => return Err(unsupported()),
            };
            if c.dealt_players > max {
                return Err(unsupported());
            }
            let row = GG
                .iter()
                .find(|r| (r.0, r.1) == (c.small_blind, c.big_blind))
                .ok_or_else(unsupported)?;
            let bucket = (c.dealt_players - 2).min(3);
            let caps = if c.table_format == TableFormat::SixMax {
                row.2
            } else {
                row.3
            };
            Ok(RakeSchedule {
                rate: RakeRate::new(5, 100).unwrap(),
                cap: caps[bucket],
                no_flop_no_drop: None,
                rounding: None,
                source: "https://legal.ggpoker.com/poker-games/texas-holdem/",
                limitations: "USD base NLHE rake only; excludes Rush & Cash, jackpot/promotional drops; NFND and rounding unverified; nine-max antes must be supplied separately",
            })
        }
    }
}
pub fn rake_config_for(c: RakeContext) -> Result<RakeConfig, ScheduleError> {
    rake_schedule_for(c)?.resolve(None, None)
}
#[cfg(test)]
mod tests {
    use super::*;
    fn context(
        platform: Platform,
        table_format: TableFormat,
        sb: Chips,
        bb: Chips,
        players: usize,
    ) -> RakeContext {
        RakeContext {
            platform,
            table_format,
            small_blind: sb,
            big_blind: bb,
            dealt_players: players,
        }
    }
    #[test]
    fn official_stars_examples_and_bankers_rounding() {
        let c = rake_config_for(context(
            Platform::PokerStars,
            TableFormat::Regular,
            10,
            25,
            5,
        ))
        .unwrap();
        assert_eq!(c.rate, RakeRate::new(45, 1000).unwrap());
        assert_eq!(c.cap, Some(200));
        assert_eq!(c.calculate(100, true, 0), 4);
        assert_eq!(c.calculate(300, true, 0), 14);
        assert_eq!(c.calculate(10000, false, 0), 0);
        assert_eq!(
            rake_config_for(context(
                Platform::PokerStars,
                TableFormat::Regular,
                25,
                50,
                2
            ))
            .unwrap()
            .cap,
            Some(75)
        );
    }
    #[test]
    fn products_and_dealt_player_caps_are_explicit() {
        assert_eq!(
            rake_schedule_for(context(Platform::CoinPoker, TableFormat::Regular, 2, 5, 2))
                .unwrap()
                .cap,
            13
        );
        assert_eq!(
            rake_schedule_for(context(Platform::CoinPoker, TableFormat::HeadsUp, 2, 5, 2))
                .unwrap()
                .cap,
            15
        );
        assert_eq!(
            rake_schedule_for(context(Platform::GGPoker, TableFormat::SixMax, 2, 5, 3))
                .unwrap()
                .cap,
            25
        );
        assert_eq!(
            rake_schedule_for(context(Platform::GGPoker, TableFormat::SixMax, 2, 5, 4))
                .unwrap()
                .cap,
            38
        );
        assert_eq!(
            rake_schedule_for(context(Platform::GGPoker, TableFormat::NineMax, 2, 5, 5))
                .unwrap()
                .cap,
            75
        );
    }
    #[test]
    fn unverified_and_unsupported_policies_never_default() {
        for platform in [Platform::CoinPoker, Platform::GGPoker] {
            let format = if platform == Platform::GGPoker {
                TableFormat::SixMax
            } else {
                TableFormat::Regular
            };
            let c = context(platform, format, 2, 5, 2);
            assert!(matches!(
                rake_config_for(c),
                Err(ScheduleError::UnverifiedPolicy { .. })
            ));
            assert!(
                rake_schedule_for(c)
                    .unwrap()
                    .resolve(Some(RakeRounding::Floor), Some(true))
                    .is_ok()
            );
        }
        assert!(
            rake_config_for(context(
                Platform::PokerStars,
                TableFormat::FastFold,
                1,
                2,
                6
            ))
            .is_err()
        );
        assert!(
            rake_config_for(context(Platform::PokerStars, TableFormat::Regular, 3, 7, 2)).is_err()
        );
    }
}
