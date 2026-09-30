//! Published NLHE cash rake data, researched 2026-09-30. Currency amounts are
//! expressed in their smallest supported unit (cents for USD/EUR/GBP/USDT).
//! This module does not infer exchange rates or missing room policies.
use super::{RakeConfig, RakeRate, RakeRounding};
use crate::Chips;
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Platform {
    PokerStars,
    CoinPoker,
    GGPoker,
}

/// Currency/denomination of the table whose schedule is being requested.
///
/// The schedule never treats two currencies as interchangeable merely because
/// their displayed values are similar. CoinPoker cash-game schedules are keyed
/// as USDT because the room currently presents USDT poker while displaying stakes in dollars.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Currency {
    Usd,
    Eur,
    Gbp,
    Usdt,
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
    pub currency: Currency,
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
    #[error("unsupported stakes, currency, or table product: {0:?}")]
    Unsupported(RakeContext),
    #[error("conflicting published rake data: {detail}")]
    ConflictingPublishedData { detail: &'static str },
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
        let no_flop_no_drop = self.no_flop_no_drop.or(no_flop_no_drop).ok_or(
            ScheduleError::UnverifiedPolicy {
                policy: "no-flop-no-drop",
            },
        )?;
        Ok(
            RakeConfig::new(self.rate, Some(self.cap), no_flop_no_drop, rounding)
                .expect("validated published schedule"),
        )
    }
}

// SB, BB, rake rate in basis points (1 bp = 0.01%), cap by 2 / 3-4 / 5+ dealt players.
const STARS_USD: &[(Chips, Chips, u32, [Chips; 3])] = &[
    (1, 2, 500, [100, 100, 100]),
    (2, 5, 500, [100, 100, 100]),
    (5, 10, 500, [100, 100, 100]),
    (10, 25, 450, [50, 100, 200]),
    (25, 50, 500, [75, 75, 200]),
    (50, 100, 500, [100, 100, 250]),
    (100, 200, 500, [125, 125, 275]),
    (200, 400, 500, [150, 150, 300]),
    (250, 500, 500, [150, 150, 300]),
    (300, 600, 500, [150, 150, 350]),
    (500, 1000, 450, [150, 150, 300]),
    (1000, 2000, 450, [175, 175, 300]),
    (2500, 5000, 450, [225, 200, 300]),
    (5000, 10000, 450, [250, 300, 500]),
];

const STARS_USD_ZOOM_MICRO: &[(Chips, Chips, u32, [Chips; 3])] = &[
    (1, 2, 350, [30, 30, 30]),
    (2, 5, 415, [50, 50, 100]),
    (5, 10, 450, [50, 100, 150]),
];

const STARS_EUR: &[(Chips, Chips, u32, [Chips; 3])] = &[
    (1, 2, 350, [25, 25, 25]),
    (2, 5, 415, [50, 50, 75]),
    (5, 10, 450, [50, 75, 125]),
    (10, 25, 450, [50, 75, 150]),
    (25, 50, 500, [70, 70, 180]),
    (50, 100, 500, [90, 90, 225]),
    (100, 200, 500, [115, 115, 250]),
    (200, 400, 500, [135, 135, 275]),
    (250, 500, 500, [135, 135, 275]),
    (300, 600, 500, [135, 135, 320]),
    (500, 1000, 450, [135, 135, 275]),
];

const STARS_GBP: &[(Chips, Chips, u32, [Chips; 3])] = &[
    (1, 2, 350, [20, 20, 20]),
    (2, 5, 415, [30, 30, 60]),
    (5, 10, 450, [30, 60, 100]),
    (10, 25, 450, [30, 60, 125]),
    (25, 50, 500, [50, 50, 140]),
    (50, 100, 500, [70, 70, 140]),
    (100, 200, 500, [90, 90, 195]),
];

// CoinPoker regular-table caps by 2 / 3-4 / 5+ dealt players.
const COIN_REGULAR: &[(Chips, Chips, [Chips; 3])] = &[
    (1, 2, [5, 12, 20]),
    (2, 5, [13, 30, 50]),
    (5, 10, [25, 60, 100]),
    (10, 25, [50, 120, 200]),
    (25, 50, [100, 240, 400]),
    (50, 100, [125, 300, 500]),
    (100, 200, [150, 360, 600]),
    (200, 500, [200, 500, 800]),
    (500, 1000, [250, 600, 1000]),
    (1000, 2000, [380, 900, 1500]),
    (2500, 5000, [500, 1200, 2000]),
];

const COIN_HEADS_UP: &[(Chips, Chips, Chips)] = &[
    (1, 2, 6),
    (2, 5, 15),
    (5, 10, 30),
    (10, 25, 60),
    (25, 50, 120),
    (50, 100, 150),
    (100, 200, 180),
    (200, 500, 240),
    (500, 1000, 300),
    (1000, 2000, 460),
    (2500, 5000, 600),
];

// GG caps by 2 / 3 / 4 / 5+ dealt players.
const GG_SIX_MAX: &[(Chips, Chips, [Chips; 4])] = &[
    (1, 2, [5, 10, 15, 20]),
    (2, 5, [13, 25, 38, 50]),
    (5, 10, [25, 50, 75, 100]),
    (10, 25, [50, 100, 150, 200]),
    (25, 50, [100, 200, 300, 400]),
    (50, 100, [125, 250, 375, 500]),
    (100, 200, [150, 300, 450, 600]),
    (200, 500, [200, 400, 600, 800]),
    (500, 1000, [250, 500, 750, 1000]),
    // Published as 0.188 / 0.375 / 0.563 / 0.75 BB at $10/$20.
    (1000, 2000, [376, 750, 1126, 1500]),
];

const GG_NINE_MAX: &[(Chips, Chips, [Chips; 4])] = &[
    (1, 2, [8, 15, 23, 30]),
    (2, 5, [19, 38, 56, 75]),
    (5, 10, [38, 75, 113, 150]),
    (10, 25, [63, 125, 188, 250]),
    (25, 50, [100, 200, 300, 400]),
    (50, 100, [125, 250, 375, 500]),
    (100, 200, [150, 300, 450, 600]),
    (200, 500, [200, 400, 600, 800]),
    (500, 1000, [250, 500, 750, 1000]),
];

fn bucket_three(players: usize) -> usize {
    if players == 2 {
        0
    } else if players <= 4 {
        1
    } else {
        2
    }
}

fn exact_ratio(value: Chips, numerator: Chips, denominator: Chips) -> Option<Chips> {
    let product = i128::from(value).checked_mul(i128::from(numerator))?;
    let denominator = i128::from(denominator);
    if product % denominator != 0 {
        return None;
    }
    Chips::try_from(product / denominator).ok()
}

fn standard_two_to_one_stakes(c: RakeContext) -> bool {
    c.small_blind > 0
        && c.big_blind > 0
        && c.small_blind.checked_mul(2) == Some(c.big_blind)
}

fn stars_schedule(c: RakeContext) -> Result<RakeSchedule, ScheduleError> {
    if !matches!(c.table_format, TableFormat::Regular | TableFormat::FastFold) {
        return Err(ScheduleError::Unsupported(c));
    }
    let bucket = bucket_three(c.dealt_players);
    let row = match c.currency {
        Currency::Usd => {
            if c.table_format == TableFormat::FastFold {
                STARS_USD_ZOOM_MICRO
                    .iter()
                    .find(|r| (r.0, r.1) == (c.small_blind, c.big_blind))
                    .copied()
                    .or_else(|| {
                        STARS_USD
                            .iter()
                            .find(|r| (r.0, r.1) == (c.small_blind, c.big_blind))
                            .copied()
                    })
            } else {
                STARS_USD
                    .iter()
                    .find(|r| (r.0, r.1) == (c.small_blind, c.big_blind))
                    .copied()
            }
            .or_else(|| {
                (c.big_blind >= 20_000 && standard_two_to_one_stakes(c))
                    .then_some((c.small_blind, c.big_blind, 450, [300, 500, 500]))
            })
        }
        Currency::Eur => STARS_EUR
            .iter()
            .find(|r| (r.0, r.1) == (c.small_blind, c.big_blind))
            .copied(),
        Currency::Gbp => STARS_GBP
            .iter()
            .find(|r| (r.0, r.1) == (c.small_blind, c.big_blind))
            .copied(),
        Currency::Usdt => None,
    }
    .ok_or(ScheduleError::Unsupported(c))?;

    Ok(RakeSchedule {
        rate: RakeRate::new(row.2, 10_000).expect("published PokerStars rate"),
        cap: row.3[bucket],
        no_flop_no_drop: Some(true),
        rounding: Some(RakeRounding::HalfToEven),
        source: "https://www.pokerstars.com/poker/room/rake/",
        limitations: "Published NLHE cash rows only; non-USD caps can change after quarterly review",
    })
}

fn coin_schedule(c: RakeContext) -> Result<RakeSchedule, ScheduleError> {
    if c.currency != Currency::Usdt
        || !matches!(c.table_format, TableFormat::Regular | TableFormat::HeadsUp)
        || (c.table_format == TableFormat::HeadsUp && c.dealt_players != 2)
    {
        return Err(ScheduleError::Unsupported(c));
    }

    if c.table_format == TableFormat::Regular
        && (c.small_blind, c.big_blind) == (200, 500)
        && matches!(c.dealt_players, 3 | 4)
    {
        return Err(ScheduleError::ConflictingPublishedData {
            detail: "CoinPoker official localized rake pages disagree on the $2/$5 3-4 player cap ($4.80 vs $5.00)",
        });
    }

    let cap = if c.table_format == TableFormat::HeadsUp {
        COIN_HEADS_UP
            .iter()
            .find(|r| (r.0, r.1) == (c.small_blind, c.big_blind))
            .map(|r| r.2)
            .or_else(|| {
                if c.big_blind >= 10_000 && standard_two_to_one_stakes(c) {
                    exact_ratio(c.big_blind, 1, 2)
                } else {
                    None
                }
            })
    } else {
        let bucket = bucket_three(c.dealt_players);
        COIN_REGULAR
            .iter()
            .find(|r| (r.0, r.1) == (c.small_blind, c.big_blind))
            .map(|r| r.2[bucket])
            .or_else(|| {
                if !standard_two_to_one_stakes(c) || c.big_blind < 10_000 {
                    return None;
                }
                if matches!(
                    (c.small_blind, c.big_blind),
                    (250_000, 500_000) | (500_000, 1_000_000)
                ) {
                    return exact_ratio(c.big_blind, 2, 5);
                }
                let (numerator, denominator) = if bucket == 0 { (1, 2) } else { (3, 5) };
                exact_ratio(c.big_blind, numerator, denominator)
            })
    }
    .ok_or(ScheduleError::Unsupported(c))?;

    Ok(RakeSchedule {
        rate: RakeRate::new(5, 100).expect("published CoinPoker rate"),
        cap,
        no_flop_no_drop: None,
        rounding: None,
        source: "https://coinpoker.com/rake/",
        limitations: "USDT NLHE base rake; excludes splash fees/cash drops; exact base-rake rounding and no-flop-no-drop remain unverified",
    })
}

fn gg_schedule(c: RakeContext) -> Result<RakeSchedule, ScheduleError> {
    if c.currency != Currency::Usd {
        return Err(ScheduleError::Unsupported(c));
    }
    let (max_players, rows) = match c.table_format {
        TableFormat::SixMax => (6, GG_SIX_MAX),
        TableFormat::NineMax => (9, GG_NINE_MAX),
        _ => return Err(ScheduleError::Unsupported(c)),
    };
    if c.dealt_players > max_players {
        return Err(ScheduleError::Unsupported(c));
    }
    let row = rows
        .iter()
        .find(|r| (r.0, r.1) == (c.small_blind, c.big_blind))
        .ok_or(ScheduleError::Unsupported(c))?;
    let bucket = (c.dealt_players - 2).min(3);
    Ok(RakeSchedule {
        rate: RakeRate::new(5, 100).expect("published GGPoker rate"),
        cap: row.2[bucket],
        no_flop_no_drop: None,
        rounding: None,
        source: "https://legal.ggpoker.com/poker-games/texas-holdem/",
        limitations: "USD base NLHE rake only; excludes Rush & Cash, jackpot/promotional drops; exact NFND and rounding policy unverified; nine-max antes are outside the rake key",
    })
}

pub fn rake_schedule_for(c: RakeContext) -> Result<RakeSchedule, ScheduleError> {
    if !(2..=9).contains(&c.dealt_players) {
        return Err(ScheduleError::InvalidPlayers(c.dealt_players));
    }
    match c.platform {
        Platform::PokerStars => stars_schedule(c),
        Platform::CoinPoker => coin_schedule(c),
        Platform::GGPoker => gg_schedule(c),
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
        currency: Currency,
        table_format: TableFormat,
        sb: Chips,
        bb: Chips,
        players: usize,
    ) -> RakeContext {
        RakeContext {
            platform,
            currency,
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
            Currency::Usd,
            TableFormat::Regular,
            10,
            25,
            5,
        ))
        .unwrap();
        assert_eq!(c.rate, RakeRate::new(450, 10_000).unwrap());
        assert_eq!(c.cap, Some(200));
        assert_eq!(c.calculate(100, true, 0), 4);
        assert_eq!(c.calculate(300, true, 0), 14);
        assert_eq!(c.calculate(10000, false, 0), 0);

        assert_eq!(
            rake_config_for(context(
                Platform::PokerStars,
                Currency::Usd,
                TableFormat::Regular,
                25,
                50,
                2,
            ))
            .unwrap()
            .cap,
            Some(75)
        );
    }

    #[test]
    fn pokerstars_currency_and_zoom_rows_are_explicit() {
        let zoom = rake_config_for(context(
            Platform::PokerStars,
            Currency::Usd,
            TableFormat::FastFold,
            1,
            2,
            6,
        ))
        .unwrap();
        assert_eq!(zoom.rate, RakeRate::new(350, 10_000).unwrap());
        assert_eq!(zoom.cap, Some(30));

        let zoom_non_micro = rake_config_for(context(
            Platform::PokerStars,
            Currency::Usd,
            TableFormat::FastFold,
            10,
            25,
            5,
        ))
        .unwrap();
        assert_eq!(zoom_non_micro.rate, RakeRate::new(450, 10_000).unwrap());
        assert_eq!(zoom_non_micro.cap, Some(200));

        let eur = rake_config_for(context(
            Platform::PokerStars,
            Currency::Eur,
            TableFormat::Regular,
            1,
            2,
            5,
        ))
        .unwrap();
        assert_eq!(eur.rate, RakeRate::new(350, 10_000).unwrap());
        assert_eq!(eur.cap, Some(25));

        let eur_415 = rake_config_for(context(
            Platform::PokerStars,
            Currency::Eur,
            TableFormat::Regular,
            2,
            5,
            5,
        ))
        .unwrap();
        assert_eq!(eur_415.rate, RakeRate::new(415, 10_000).unwrap());
        assert_eq!(eur_415.calculate(10_000, true, 0), 75); // 4.15% capped at €0.75

        let gbp = rake_config_for(context(
            Platform::PokerStars,
            Currency::Gbp,
            TableFormat::FastFold,
            100,
            200,
            5,
        ))
        .unwrap();
        assert_eq!(gbp.cap, Some(195));

        let high = rake_config_for(context(
            Platform::PokerStars,
            Currency::Usd,
            TableFormat::Regular,
            20_000,
            40_000,
            4,
        ))
        .unwrap();
        assert_eq!(high.cap, Some(500));
    }

    #[test]
    fn coinpoker_caps_high_stakes_and_conflicts_are_explicit() {
        assert!(matches!(
            rake_schedule_for(context(
                Platform::CoinPoker,
                Currency::Usdt,
                TableFormat::Regular,
                200,
                500,
                3,
            )),
            Err(ScheduleError::ConflictingPublishedData { .. })
        ));

        let regular = rake_schedule_for(context(
            Platform::CoinPoker,
            Currency::Usdt,
            TableFormat::Regular,
            100,
            200,
            3,
        ))
        .unwrap();
        assert_eq!(regular.cap, 360);
        assert_eq!(regular.no_flop_no_drop, None);

        assert_eq!(
            rake_schedule_for(context(
                Platform::CoinPoker,
                Currency::Usdt,
                TableFormat::HeadsUp,
                2,
                5,
                2,
            ))
            .unwrap()
            .cap,
            15
        );

        assert_eq!(
            rake_schedule_for(context(
                Platform::CoinPoker,
                Currency::Usdt,
                TableFormat::Regular,
                5000,
                10_000,
                5,
            ))
            .unwrap()
            .cap,
            6000
        );
        assert_eq!(
            rake_schedule_for(context(
                Platform::CoinPoker,
                Currency::Usdt,
                TableFormat::Regular,
                250_000,
                500_000,
                3,
            ))
            .unwrap()
            .cap,
            200_000
        );
    }

    #[test]
    fn ggpoker_products_and_dealt_player_caps_are_explicit() {
        assert_eq!(
            rake_schedule_for(context(
                Platform::GGPoker,
                Currency::Usd,
                TableFormat::SixMax,
                2,
                5,
                3,
            ))
            .unwrap()
            .cap,
            25
        );
        assert_eq!(
            rake_schedule_for(context(
                Platform::GGPoker,
                Currency::Usd,
                TableFormat::SixMax,
                1000,
                2000,
                4,
            ))
            .unwrap()
            .cap,
            1126
        );
        assert_eq!(
            rake_schedule_for(context(
                Platform::GGPoker,
                Currency::Usd,
                TableFormat::NineMax,
                2,
                5,
                5,
            ))
            .unwrap()
            .cap,
            75
        );
    }

    #[test]
    fn currency_and_unverified_policies_never_default() {
        assert!(
            rake_schedule_for(context(
                Platform::PokerStars,
                Currency::Usdt,
                TableFormat::Regular,
                1,
                2,
                6,
            ))
            .is_err()
        );
        assert!(
            rake_schedule_for(context(
                Platform::CoinPoker,
                Currency::Usd,
                TableFormat::Regular,
                2,
                5,
                2,
            ))
            .is_err()
        );

        let coin = context(
            Platform::CoinPoker,
            Currency::Usdt,
            TableFormat::Regular,
            2,
            5,
            2,
        );
        assert!(matches!(
            rake_config_for(coin),
            Err(ScheduleError::UnverifiedPolicy {
                policy: "rake rounding"
            })
        ));
        assert!(
            rake_schedule_for(coin)
                .unwrap()
                .resolve(Some(RakeRounding::Floor), Some(true))
                .is_ok()
        );

        let gg = context(
            Platform::GGPoker,
            Currency::Usd,
            TableFormat::SixMax,
            2,
            5,
            2,
        );
        assert!(matches!(
            rake_config_for(gg),
            Err(ScheduleError::UnverifiedPolicy { .. })
        ));
        assert!(
            rake_schedule_for(gg)
                .unwrap()
                .resolve(Some(RakeRounding::Floor), Some(true))
                .is_ok()
        );

        assert!(
            rake_config_for(context(
                Platform::PokerStars,
                Currency::Usd,
                TableFormat::Regular,
                3,
                7,
                2,
            ))
            .is_err()
        );
    }
}
