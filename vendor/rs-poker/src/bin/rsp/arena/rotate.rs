use rs_poker::Chips;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use clap::{Args, ValueEnum};
use rs_poker::arena::RakeConfig;
use rs_poker::arena::rake::schedule::{
    Currency, Platform, RakeContext, ScheduleError, TableFormat, rake_config_for,
};
use rs_poker::arena::rotation::{ArenaRotation, RotationBuilder, RotationHandResult};

use crate::tui::app::{self, App};
use crate::tui::event::{EventHandler, SimError, SimMessage};
use crate::tui::hand_store::HandStore;
use crate::tui::state::{GameResult, SeatStats, ending_round_from_stats};
use crate::tui::{TuiFlags, run_blocking_tui_loop};

#[derive(Debug, thiserror::Error)]
pub enum RotateError {
    #[error(transparent)]
    Rotation(#[from] rs_poker::arena::rotation::RotationError),
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
    about = "Benchmark poker agents in persistent-stack table rotations",
    long_about = "Seats agents at a table for one complete dealer-button rotation, preserves stacks between hands, rebuys busted players, and runs independent table rotations concurrently."
)]
pub struct RotateArgs {
    /// Directory containing agent JSON config files.
    agents_dir: PathBuf,

    /// Number of independent complete table rotations.
    #[arg(
        short = 'n',
        long = "rotations",
        alias = "num-games",
        default_value_t = 1000
    )]
    num_rotations: usize,

    /// Number of players per table.
    #[arg(short = 'p', long = "players", default_value_t = 3)]
    players_per_table: usize,

    /// Big blind amount in cents.
    #[arg(long = "big-blind", default_value_t = 10)]
    big_blind: Chips,

    /// Small blind amount in cents.
    #[arg(long = "small-blind", default_value_t = 5)]
    small_blind: Chips,

    /// Initial minimum stack in big blinds at the start of each rotation.
    #[arg(long = "min-stack-bb", default_value_t = 100.0)]
    min_stack_bb: f32,

    /// Initial maximum stack in big blinds at the start of each rotation.
    #[arg(long = "max-stack-bb", default_value_t = 100.0)]
    max_stack_bb: f32,

    /// Buy-in after a player busts. The injected chips are not counted as profit.
    #[arg(long = "rebuy-to-bb", default_value_t = 100.0)]
    rebuy_to_bb: f32,

    /// Concurrent table rotations. Defaults to roughly 80% of available parallelism.
    #[arg(short = 'j', long = "jobs")]
    jobs: Option<usize>,

    /// Optional directory to save game history and results.
    #[arg(short = 'o', long = "output-dir")]
    output_dir: Option<PathBuf>,

    /// Random seed for reproducible scheduling/deals.
    #[arg(short = 's', long = "seed")]
    seed: Option<u64>,

    /// Rake schedule preset. Defaults to no rake.
    #[arg(long = "rake", value_enum, default_value = "none")]
    rake: RakePreset,

    /// Override preset currency (CoinPoker defaults to USDT).
    #[arg(long = "rake-currency", value_enum)]
    rake_currency: Option<RakeCurrencyArg>,

    /// Override table product used to resolve room rake.
    #[arg(long = "rake-format", value_enum)]
    rake_format: Option<RakeFormatArg>,

    #[command(flatten)]
    tui: TuiFlags,
}

fn resolve_rake(args: &RotateArgs) -> Result<RakeConfig, RotateError> {
    if matches!(args.rake, RakePreset::None) {
        if args.rake_currency.is_some() || args.rake_format.is_some() {
            return Err(RotateError::RakeOptionsWithoutPreset);
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

    Ok(rake_config_for(RakeContext {
        platform,
        currency: args
            .rake_currency
            .map(Into::into)
            .unwrap_or(default_currency),
        small_blind: args.small_blind,
        big_blind: args.big_blind,
        dealt_players: args.players_per_table,
        table_format: args.rake_format.map(Into::into).unwrap_or(default_format),
    })?)
}

fn build_rotation(
    args: &RotateArgs,
    default_budget: &rs_poker::arena::cfr::BudgetConfig,
) -> Result<ArenaRotation, RotateError> {
    let mut builder = RotationBuilder::new()
        .num_rotations(args.num_rotations)
        .players_per_table(args.players_per_table)
        .big_blind(args.big_blind)
        .small_blind(args.small_blind)
        .min_stack_bb(args.min_stack_bb)
        .max_stack_bb(args.max_stack_bb)
        .rebuy_to_bb(args.rebuy_to_bb)
        .rake(resolve_rake(args)?)
        .load_agents_from_dir(&args.agents_dir)?
        .fill_default_budget(default_budget);

    if let Some(seed) = args.seed {
        builder = builder.seed(seed);
    }
    if let Some(jobs) = args.jobs {
        builder = builder.jobs(jobs);
    }
    if let Some(ref output_dir) = args.output_dir {
        builder = builder.output_dir(output_dir);
    }
    Ok(builder.build()?)
}

fn hand_to_game_result(hand: RotationHandResult, big_blind: Chips) -> GameResult {
    let num_players = hand.agent_names.len();
    let ending_round = ending_round_from_stats(&hand.stats, num_players);
    let profits = (0..num_players)
        .map(|i| hand.stats.total_profit[i])
        .collect();
    let seat_stats = (0..num_players)
        .map(|i| SeatStats::from_storage(&hand.stats, i))
        .collect();

    GameResult {
        agent_names: hand.agent_names,
        profits,
        ending_round,
        seat_stats,
        big_blind,
    }
}

async fn run_rotation_background(
    rotation: ArenaRotation,
    tx: std::sync::mpsc::SyncSender<SimMessage<GameResult>>,
    hand_store: HandStore,
    ohh_path: Option<PathBuf>,
    big_blind: Chips,
    cancel: Arc<AtomicBool>,
) {
    let run_tx = tx.clone();
    let join = tokio::spawn(async move {
        let mut prev_file_size: u64 = 0;
        rotation
            .run_with_cancellable_callback(|hand| {
                if let Some(ref path) = ohh_path {
                    let current_size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
                    if current_size > prev_file_size {
                        hand_store.push_offset(prev_file_size);
                    }
                    prev_file_size = current_size;
                }

                if run_tx
                    .send(SimMessage::GameResult(hand_to_game_result(hand, big_blind)))
                    .is_err()
                {
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
            let _ = tx.send(SimMessage::Error(SimError::RotationFailed { source: e }));
        }
        Err(_) => {
            let _ = tx.send(SimMessage::Error(SimError::Panic));
        }
    }
}

async fn run_with_tui(rotation: ArenaRotation, big_blind: Chips) -> Result<(), RotateError> {
    let total_hands = rotation.total_hands();
    let ohh_path = rotation
        .config()
        .output_dir
        .as_ref()
        .map(|dir| dir.join("hands.jsonl"));
    let hand_store = match ohh_path {
        Some(ref path) => HandStore::new(path.clone()),
        None => HandStore::none(),
    };

    let (tx, rx) = std::sync::mpsc::sync_channel::<SimMessage<GameResult>>(1024);
    let cancel = Arc::new(AtomicBool::new(false));
    let bg_cancel = Arc::clone(&cancel);
    let bg_store = hand_store.clone();
    let handle = tokio::spawn(async move {
        run_rotation_background(rotation, tx, bg_store, ohh_path, big_blind, bg_cancel).await;
    });

    run_blocking_tui_loop(
        move || {
            let handler = EventHandler::new(rx, Duration::from_millis(33));
            let mut app = App::new(Some(total_hands));
            app.hand_store = hand_store;
            app::run_app(&mut app, &handler)
        },
        handle,
        || cancel.store(true, Ordering::Release),
    )
    .await?;

    Ok(())
}

pub async fn run(
    mut args: RotateArgs,
    default_budget: &rs_poker::arena::cfr::BudgetConfig,
) -> Result<(), RotateError> {
    let use_tui = args.tui.should_use_tui();

    let _temp_dir = if args.output_dir.is_none() && use_tui {
        let tmp = tempfile::TempDir::new()?;
        args.output_dir = Some(tmp.path().to_path_buf());
        Some(tmp)
    } else {
        None
    };

    let rotation = build_rotation(&args, default_budget)?;
    rotation.print_configuration_summary();

    if use_tui {
        run_with_tui(rotation, args.big_blind).await
    } else {
        println!("Starting rotations...");
        let result = rotation.run().await?;
        println!(
            "\nCompleted {} rotations / {} hands.",
            result.rotations_run(),
            result.hands_run()
        );
        println!("{}", result.to_markdown());

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
