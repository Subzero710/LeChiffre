## Poker AI Project
AI agents trained to play poker.
The project uses the RS_Poker library in Rust.

## AI Types
- RandomAgent (plays randomly; this is the baseline)
- EquityAgent 
- FoldingAgent (always chooses the "Fold" action)
- CallingAgent (always chooses the "Call" action)
- CFRSimpleAgent
- CFRBasicAgent

## Random AI Experiment
Run a reproducible comparison between two randomized strategies and generate an SVG chart:

```sh
cargo run --example random_ai_experiment
```

The chart is written to `target/random_ai_results.svg`. Run the experiment's tests with:

```sh
cargo test --example random_ai_experiment
```

See [TEST_BRANCH_GUIDE.txt](TEST_BRANCH_GUIDE.txt) for an explanation of the files and results.
