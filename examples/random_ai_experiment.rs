use rand::{SeedableRng, rngs::StdRng};
use rs_poker::arena::{
    Agent, GameStateBuilder, HoldemSimulationBuilder, agent::RandomAgent,
};
use std::{error::Error, fs, path::Path};

const STARTING_STACK: u64 = 100;
const MATCH_COUNT: usize = 40;

/// Results collected from a batch of heads-up games.
#[derive(Debug, Default, PartialEq)]
struct ExperimentResults {
    games: usize,
    baseline_wins: usize,
    cautious_wins: usize,
    ties: usize,
    baseline_total_chips: u64,
    cautious_total_chips: u64,
}

impl ExperimentResults {
    fn record(&mut self, stacks: [u64; 2]) {
        self.games += 1;
        self.baseline_total_chips += stacks[0];
        self.cautious_total_chips += stacks[1];

        match stacks[0].cmp(&stacks[1]) {
            std::cmp::Ordering::Greater => self.baseline_wins += 1,
            std::cmp::Ordering::Less => self.cautious_wins += 1,
            std::cmp::Ordering::Equal => self.ties += 1,
        }
    }

    fn baseline_win_rate(&self) -> f64 {
        percentage(self.baseline_wins, self.games)
    }

    fn cautious_win_rate(&self) -> f64 {
        percentage(self.cautious_wins, self.games)
    }

    fn average_stacks(&self) -> [f64; 2] {
        if self.games == 0 {
            return [0.0, 0.0];
        }

        [
            self.baseline_total_chips as f64 / self.games as f64,
            self.cautious_total_chips as f64 / self.games as f64,
        ]
    }
}

fn percentage(part: usize, whole: usize) -> f64 {
    if whole == 0 {
        0.0
    } else {
        part as f64 * 100.0 / whole as f64
    }
}

/// Play one game with deterministic seeds for both agents and the deck.
async fn play_game(seed: u64, dealer_idx: usize) -> [u64; 2] {
    let game_state = GameStateBuilder::new()
        .num_players_with_stack(2, STARTING_STACK as i64)
        .blinds(2, 1)
        .dealer_idx(dealer_idx)
        .max_raises_per_round(Some(3))
        .build()
        .expect("the two-player game state should be valid");

    // The baseline uses the example probabilities from the existing main program.
    let baseline = RandomAgent::new_with_seed(
        "Baseline Random",
        vec![0.25, 0.30, 0.50],
        vec![0.50, 0.60, 0.45],
        seed.wrapping_add(1),
    );

    // This second randomized profile folds more often and calls less often.
    let cautious = RandomAgent::new_with_seed(
        "Cautious Random",
        vec![0.65, 0.75, 0.85],
        vec![0.35, 0.30, 0.25],
        seed.wrapping_add(2),
    );

    let agents: Vec<Box<dyn Agent>> = vec![Box::new(baseline), Box::new(cautious)];
    let mut simulation = HoldemSimulationBuilder::default()
        .game_state(game_state)
        .agents(agents)
        .build_with_rng(StdRng::seed_from_u64(seed))
        .expect("the simulation should be valid");

    simulation.run().await;
    [
        simulation.game_state.stacks[0] as u64,
        simulation.game_state.stacks[1] as u64,
    ]
}

/// Run multiple games, alternating the dealer to reduce position bias.
async fn run_experiment(game_count: usize) -> ExperimentResults {
    let mut results = ExperimentResults::default();

    for game_index in 0..game_count {
        let dealer_idx = game_index % 2;
        let stacks = play_game(game_index as u64 + 100, dealer_idx).await;
        results.record(stacks);
    }

    results
}

/// Create a standalone SVG with win-rate and average-stack bar charts.
fn render_svg(results: &ExperimentResults) -> String {
    let average_stacks = results.average_stacks();
    let max_average = average_stacks[0].max(average_stacks[1]).max(1.0);
    let baseline_stack_width = average_stacks[0] / max_average * 400.0;
    let cautious_stack_width = average_stacks[1] / max_average * 400.0;

    format!(
        r##"<svg xmlns="http://www.w3.org/2000/svg" width="760" height="340" viewBox="0 0 760 340">
  <rect width="760" height="340" fill="#f7f7f2" />
  <text x="32" y="38" font-family="sans-serif" font-size="23" font-weight="bold" fill="#202820">Random AI Experiment Results</text>
  <text x="32" y="68" font-family="sans-serif" font-size="14" fill="#596359">{games} reproducible heads-up games | {ties} ties</text>
  <text x="32" y="112" font-family="sans-serif" font-size="16" font-weight="bold" fill="#202820">Win rate</text>
  <text x="32" y="147" font-family="sans-serif" font-size="14" fill="#202820">Baseline Random</text>
  <rect x="220" y="130" width="{baseline_win_width:.1}" height="24" fill="#237a57" />
  <text x="{baseline_win_label_x:.1}" y="147" font-family="sans-serif" font-size="14" fill="#202820">{baseline_win_rate:.1}%</text>
  <text x="32" y="185" font-family="sans-serif" font-size="14" fill="#202820">Cautious Random</text>
  <rect x="220" y="168" width="{cautious_win_width:.1}" height="24" fill="#d2783e" />
  <text x="{cautious_win_label_x:.1}" y="185" font-family="sans-serif" font-size="14" fill="#202820">{cautious_win_rate:.1}%</text>
  <text x="32" y="235" font-family="sans-serif" font-size="16" font-weight="bold" fill="#202820">Average final stack (starting stack: 100)</text>
  <text x="32" y="270" font-family="sans-serif" font-size="14" fill="#202820">Baseline Random</text>
  <rect x="220" y="253" width="{baseline_stack_width:.1}" height="24" fill="#237a57" />
  <text x="{baseline_stack_label_x:.1}" y="270" font-family="sans-serif" font-size="14" fill="#202820">{baseline_average:.1}</text>
  <text x="32" y="308" font-family="sans-serif" font-size="14" fill="#202820">Cautious Random</text>
  <rect x="220" y="291" width="{cautious_stack_width:.1}" height="24" fill="#d2783e" />
  <text x="{cautious_stack_label_x:.1}" y="308" font-family="sans-serif" font-size="14" fill="#202820">{cautious_average:.1}</text>
</svg>
"##,
        games = results.games,
        ties = results.ties,
        baseline_win_width = results.baseline_win_rate() * 4.0,
        baseline_win_label_x = 228.0 + results.baseline_win_rate() * 4.0,
        baseline_win_rate = results.baseline_win_rate(),
        cautious_win_width = results.cautious_win_rate() * 4.0,
        cautious_win_label_x = 228.0 + results.cautious_win_rate() * 4.0,
        cautious_win_rate = results.cautious_win_rate(),
        baseline_stack_width = baseline_stack_width,
        baseline_stack_label_x = 228.0 + baseline_stack_width,
        baseline_average = average_stacks[0],
        cautious_stack_width = cautious_stack_width,
        cautious_stack_label_x = 228.0 + cautious_stack_width,
        cautious_average = average_stacks[1],
    )
}

fn print_results(results: &ExperimentResults, chart_path: &Path) {
    let average_stacks = results.average_stacks();

    println!("Games played: {}", results.games);
    println!(
        "Baseline Random: {} wins ({:.1}%), average final stack {:.1}",
        results.baseline_wins,
        results.baseline_win_rate(),
        average_stacks[0]
    );
    println!(
        "Cautious Random: {} wins ({:.1}%), average final stack {:.1}",
        results.cautious_wins,
        results.cautious_win_rate(),
        average_stacks[1]
    );
    println!("Ties: {}", results.ties);
    println!("SVG chart saved to {}", chart_path.display());
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let results = run_experiment(MATCH_COUNT).await;
    let chart_path = Path::new("target/random_ai_results.svg");

    if let Some(parent) = chart_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(chart_path, render_svg(&results))?;
    print_results(&results, chart_path);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn same_seed_produces_same_final_stacks() {
        let first_run = play_game(42, 0).await;
        let second_run = play_game(42, 0).await;

        assert_eq!(first_run, second_run);
    }

    #[tokio::test]
    async fn chips_are_conserved_in_each_game() {
        for seed in [7, 11, 21] {
            let final_stacks = play_game(seed, seed as usize % 2).await;
            assert_eq!(final_stacks.iter().sum::<u64>(), STARTING_STACK * 2);
        }
    }

    #[tokio::test]
    async fn experiment_records_every_requested_game() {
        let results = run_experiment(6).await;

        assert_eq!(results.games, 6);
        assert_eq!(results.baseline_wins + results.cautious_wins + results.ties, 6);
        assert_eq!(results.baseline_total_chips + results.cautious_total_chips, 6 * STARTING_STACK * 2);
    }

    #[test]
    fn svg_includes_chart_titles_and_result_labels() {
        let results = ExperimentResults {
            games: 10,
            baseline_wins: 4,
            cautious_wins: 5,
            ties: 1,
            baseline_total_chips: 980,
            cautious_total_chips: 1020,
        };

        let svg = render_svg(&results);

        assert!(svg.contains("Random AI Experiment Results"));
        assert!(svg.contains("Baseline Random"));
        assert!(svg.contains("Cautious Random"));
        assert!(svg.contains("Average final stack"));
    }
}