use rs_poker::Chips;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use clap::{Args, ValueEnum};
use rs_poker::arena::comparison::{ArenaComparison, ComparisonBuilder, PermutationResult};
use rs_poker::arena::rake::schedule::{
    Currency, Platform, RakeContext, ScheduleError, TableFormat, rake_config_for,
};
use rs_poker::arena::RakeConfig;

use crate::tui::app::{self, App};
use crate::tui::event::{EventHandler, SimError, SimMessage};
use crate::tui::hand_store::HandStore;
use crate::tui::state::{GameResult, SeatStats, ending_round_from_stats};
use crate::tui::{TuiFlags, run_blocking_tui_loop};

#[derive(Debug, thiserror::Error)]
pub enum CompareError {
    #[error(transparent)]
    Comparison(#[from] rs_poker::arena::comparison::ComparisonError),
    #[error("TUI error: {0}")]
    TuiError(#[from] std::io::Error),
    #[error(transparent)]
    RakeSchedule(#[from] ScheduleError),
    #[error("--rake-currency and --rake-format require --rake to be a room preset")]
    RakeOptionsWithoutPreset,
}

#[derive(Debug, Clone, Copy, Default, ValueEnum)]
enum RakePreset {
    #[default]
    None,
    Coinpoker,
    Pokerstars,
    Ggpoker,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum RakeCurrencyArg {
    Usd,
    Eur,
    Gbp,
    Usdt,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum RakeFormatArg {
    Regular,
    HeadsUp,
    SixMax,
    NineMax,
    FastFold,
}

impl From<RakeCurrencyArg> for Currency {
    fn from(value: RakeCurrencyArg) -> Self {
        match value {
            RakeCurrencyArg::Usd => Currency::Usd,
            RakeCurrencyArg::Eur => Currency::Eur,
            RakeCurrencyArg::Gbp => Currency::Gbp,
            RakeCurrencyArg::Usdt => Currency::Usdt,
        }
    }
}

impl From<RakeFormatArg> for TableFormat {
    fn from(value: RakeFormatArg) -> Self {
        match value {
            RakeFormatArg::Regular => TableFormat::Regular,
            RakeFormatArg::HeadsUp => TableFormat::HeadsUp,
            RakeFormatArg::SixMax => TableFormat::SixMax,
            RakeFormatArg::NineMax => TableFormat::NineMax,
            RakeFormatArg::FastFold => TableFormat::FastFold,
        }
    }
}

#[derive(Args, Debug)]
#[command(
    about = "Compare poker agents across all possible matchups and positions",
    long_about = "Evaluates poker agents by running all permutations of seat arrangements,\n\
                  tracking detailed per-agent statistics to determine which agents perform best."
)]
pub struct CompareArgs {
    /// Directory containing agent JSON config files
    agents_dir: PathBuf,

    /// Number of unique game states to test
    #[arg(short = 'n', long = "num-games", default_value_t = 1000)]
    num_games: usize,

    /// Number of players per table (must be >= 2 and <= number of agents)
    #[arg(short = 'p', long = "players", default_value_t = 3)]
    players_per_table: usize,

    /// Big blind amount in cents
    #[arg(long = "big-blind", default_value_t = 10)]
    big_blind: Chips,

    /// Small blind amount in cents
    #[arg(long = "small-blind", default_value_t = 5)]
    small_blind: Chips,

    /// Minimum starting stack in big blinds
    #[arg(long = "min-stack-bb", default_value_t = 100.0)]
    min_stack_bb: f32,

    /// Maximum starting stack in big blinds
    #[arg(long = "max-stack-bb", default_value_t = 100.0)]
    max_stack_bb: f32,

    /// Optional directory to save game history and results
    #[arg(short = 'o', long = "output-dir")]
    output_dir: Option<PathBuf>,

    /// Optional random seed for reproducibility
    #[arg(short = 's', long = "seed")]
    seed: Option<u64>,

    /// Rake schedule preset. Defaults to no rake.
    #[arg(long = "rake", value_enum, default_value = "none")]
    rake: RakePreset,

    /// Override the preset currency (defaults: CoinPoker=USDT, others=USD).
    #[arg(long = "rake-currency", value_enum)]
    rake_currency: Option<RakeCurrencyArg>,

    /// Override the table product used to resolve the room rake schedule.
    /// Auto defaults: CoinPoker HU for -p 2 else regular; PokerStars regular;
    /// GGPoker six-max for <=6 players else nine-max.
    #[arg(long = "rake-format", value_enum)]
    rake_format: Option<RakeFormatArg>,

    #[command(flatten)]
    tui: TuiFlags,
}

fn resolve_rake(args: &CompareArgs) -> Result<RakeConfig, CompareError> {
    if matches!(args.rake, RakePreset::None) {
        if args.rake_currency.is_some() || args.rake_format.is_some() {
            return Err(CompareError::RakeOptionsWithoutPreset);
        }
        return Ok(RakeConfig::none());
    }

    let (platform, default_currency, default_format) = match args.rake {
        RakePreset::None => unreachable!(),
        RakePreset::Coinpoker => (
            Platform::CoinPoker,
            Currency::Usdt,
            if args.players_per_table == 2 {
                TableFormat::HeadsUp
            } else {
                TableFormat::Regular
            },
        ),
        RakePreset::Pokerstars => (Platform::PokerStars, Currency::Usd, TableFormat::Regular),
        RakePreset::Ggpoker => (
            Platform::GGPoker,
            Currency::Usd,
            if args.players_per_table <= 6 {
                TableFormat::SixMax
            } else {
                TableFormat::NineMax
            },
        ),
    };

    let context = RakeContext {
        platform,
        currency: args.rake_currency.map(Into::into).unwrap_or(default_currency),
        small_blind: args.small_blind,
        big_blind: args.big_blind,
        dealt_players: args.players_per_table,
        table_format: args.rake_format.map(Into::into).unwrap_or(default_format),
    };

    Ok(rake_config_for(context)?)
}

fn build_comparison(
    args: &CompareArgs,
    default_budget: &rs_poker::arena::cfr::BudgetConfig,
) -> Result<ArenaComparison, CompareError> {
    let rake = resolve_rake(args)?;
    let mut builder = ComparisonBuilder::new()
        .num_games(args.num_games)
        .players_per_table(args.players_per_table)
        .big_blind(args.big_blind)
        .small_blind(args.small_blind)
        .min_stack_bb(args.min_stack_bb)
        .max_stack_bb(args.max_stack_bb)
        .rake(rake)
        .load_agents_from_dir(&args.agents_dir)?
        .fill_default_budget(default_budget);

    if let Some(seed) = args.seed {
        builder = builder.seed(seed);
    }

    if let Some(ref output_dir) = args.output_dir {
        builder = builder.output_dir(output_dir);
    }

    Ok(builder.build()?)
}

/// Convert a PermutationResult into a GameResult for the TUI.
fn perm_to_game_result(perm: PermutationResult, big_blind: Chips) -> GameResult {
    let num_players = perm.agent_names.len();
    let ending_round = ending_round_from_stats(&perm.stats, num_players);
    let profits: Vec<Chips> = (0..num_players)
        .map(|i| perm.stats.total_profit[i])
        .collect();
    let seat_stats: Vec<SeatStats> = (0..num_players)
        .map(|i| SeatStats::from_storage(&perm.stats, i))
        .collect();
    // perm.stats (with its 40+ Vecs) is dropped here, on the comparison thread

    GameResult {
        agent_names: perm.agent_names,
        profits,
        ending_round,
        seat_stats,
        big_blind,
    }
}

/// Run the comparison as a background tokio task, sending results over a channel.
///
/// `cancel` is a shared cancellation flag. The callback checks it after
/// each permutation and returns `Break` to stop the comparison early,
/// so the worker stops allocating CFR trees as soon as the TUI quits.
///
/// The comparison itself runs in a nested `tokio::spawn` so a panic in CFR
/// exploration surfaces as a `JoinError` (reported as `SimError::Panic`)
/// rather than aborting the process.
async fn run_comparison_background(
    comparison: ArenaComparison,
    tx: std::sync::mpsc::SyncSender<SimMessage<GameResult>>,
    hand_store: HandStore,
    ohh_path: Option<PathBuf>,
    big_blind: Chips,
    cancel: Arc<AtomicBool>,
) {
    let run_tx = tx.clone();
    let join = tokio::spawn(async move {
        let mut prev_file_size: u64 = 0;
        comparison
            .run_with_cancellable_callback(|perm| {
                // Track byte offsets for on-demand hand loading
                if let Some(ref path) = ohh_path {
                    let current_size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
                    if current_size > prev_file_size {
                        hand_store.push_offset(prev_file_size);
                    }
                    prev_file_size = current_size;
                }

                let game_result = perm_to_game_result(perm, big_blind);
                // If send fails the TUI has quit — mark cancelled so we stop
                // as soon as the current permutation ends rather than
                // allocating the next CFR tree.
                if run_tx.send(SimMessage::GameResult(game_result)).is_err() {
                    cancel.store(true, Ordering::Release);
                }

                if cancel.load(Ordering::Acquire) {
                    std::ops::ControlFlow::Break(())
                } else {
                    std::ops::ControlFlow::Continue(())
                }
            })
            .await
    })
    .await;

    match join {
        Ok(Ok(_)) => {
            let _ = tx.send(SimMessage::Completed);
        }
        Ok(Err(e)) => {
            let _ = tx.send(SimMessage::Error(SimError::ComparisonFailed { source: e }));
        }
        Err(_) => {
            let _ = tx.send(SimMessage::Error(SimError::Panic));
        }
    }
}

/// Run comparison with the TUI dashboard.
///
/// The comparison runs as a tokio task on the multi-thread runtime (whose
/// worker threads carry a large stack for deep CFR recursion), sending results
/// over a sync channel. The blocking ratatui render loop runs concurrently on a
/// dedicated blocking thread via `spawn_blocking`.
async fn run_comparison_with_tui(
    comparison: ArenaComparison,
    big_blind: Chips,
) -> Result<(), CompareError> {
    let total_games = comparison.total_games();

    // Extract OHH path before moving comparison into background task
    let ohh_path = comparison
        .config()
        .output_dir
        .as_ref()
        .map(|dir| dir.join("hands.jsonl"));
    let hand_store = match ohh_path {
        Some(ref p) => HandStore::new(p.clone()),
        None => HandStore::none(),
    };

    let (tx, rx) = std::sync::mpsc::sync_channel::<SimMessage<GameResult>>(1024);

    // Shared cancellation flag: set when the TUI exits so the background
    // comparison stops allocating new CFR trees.
    let cancel = Arc::new(AtomicBool::new(false));

    // Spawn the comparison as a background tokio task. CFR exploration recurses
    // deeply; the runtime's worker threads are built with a large stack (see
    // `main`) so this no longer needs a hand-rolled OS thread.
    let bg_hand_store = hand_store.clone();
    let bg_cancel = Arc::clone(&cancel);
    let comparison_handle = tokio::spawn(async move {
        run_comparison_background(
            comparison,
            tx,
            bg_hand_store,
            ohh_path,
            big_blind,
            bg_cancel,
        )
        .await;
    });

    // The TUI render loop is blocking (crossterm poll + terminal draw); the
    // shared helper runs it on a blocking thread and joins the background task.
    // On TUI exit we set the cancel flag so the worker stops allocating new CFR
    // trees, then wait for it to finish writing — this keeps any caller-owned
    // temp directory alive until the OHH historian is done with it.
    run_blocking_tui_loop(
        move || {
            let handler = EventHandler::new(rx, Duration::from_millis(33));
            let mut tui_app = App::new(Some(total_games));
            tui_app.hand_store = hand_store;
            app::run_app(&mut tui_app, &handler)
        },
        comparison_handle,
        || cancel.store(true, Ordering::Release),
    )
    .await?;

    Ok(())
}

pub async fn run(
    mut args: CompareArgs,
    default_budget: &rs_poker::arena::cfr::BudgetConfig,
) -> Result<(), CompareError> {
    let use_tui = args.tui.should_use_tui();

    // When using the TUI without an explicit output dir, use a temp dir
    // so OHH hands are always written and game detail view works.
    let _temp_dir = if args.output_dir.is_none() && use_tui {
        let tmp = tempfile::TempDir::new()?;
        args.output_dir = Some(tmp.path().to_path_buf());
        Some(tmp)
    } else {
        None
    };

    let comparison = build_comparison(&args, default_budget)?;

    if use_tui {
        comparison.print_configuration_summary();
        run_comparison_with_tui(comparison, args.big_blind).await
    } else {
        // Print configuration summary
        comparison.print_configuration_summary();

        // Run simulations
        println!("Starting simulations...");
        let result = comparison.run().await?;
        println!("\nCompleted all {} game states!", result.config().num_games);

        // Print results
        println!("{}", result.to_markdown());

        // Save to files if output directory specified
        if let Some(ref output_dir) = args.output_dir {
            result.save_to_dir(output_dir)?;
            println!("Results saved to:");
            println!("  - {}", output_dir.join("results.json").display());
            println!("  - {}", output_dir.join("results.md").display());
            println!("  - {}", output_dir.join("hands.jsonl").display());
        }

        Ok(())
    }
}

#[cfg(test)]
mod rake_cli_tests {
    use super::*;

    fn args(preset: RakePreset, players: usize) -> CompareArgs {
        CompareArgs {
            agents_dir: PathBuf::from("agents"),
            num_games: 1,
            players_per_table: players,
            big_blind: 10,
            small_blind: 5,
            min_stack_bb: 100.0,
            max_stack_bb: 100.0,
            output_dir: None,
            seed: Some(42),
            rake: preset,
            rake_currency: None,
            rake_format: None,
            tui: TuiFlags {
                force_tui: false,
                no_tui: false,
            },
        }
    }

    #[test]
    fn no_rake_is_the_default_economy() {
        assert_eq!(resolve_rake(&args(RakePreset::None, 2)).unwrap(), RakeConfig::none());
    }

    #[test]
    fn coinpoker_heads_up_defaults_to_usdt_hu_schedule() {
        let rake = resolve_rake(&args(RakePreset::Coinpoker, 2)).unwrap();
        assert_eq!(rake.rate.numerator(), 5);
        assert_eq!(rake.rate.denominator(), 100);
        assert_eq!(rake.cap, Some(30));
        assert!(rake.no_flop_no_drop);
    }

    #[test]
    fn pokerstars_defaults_to_usd_regular_schedule() {
        let rake = resolve_rake(&args(RakePreset::Pokerstars, 2)).unwrap();
        assert_eq!(rake.rate.numerator(), 500);
        assert_eq!(rake.rate.denominator(), 10_000);
        assert_eq!(rake.cap, Some(100));
    }

    #[test]
    fn ggpoker_defaults_to_six_max_for_two_players() {
        let rake = resolve_rake(&args(RakePreset::Ggpoker, 2)).unwrap();
        assert_eq!(rake.rate.numerator(), 5);
        assert_eq!(rake.rate.denominator(), 100);
        assert_eq!(rake.cap, Some(25));
        assert!(rake.preflop_three_bet_rake);
    }
}
