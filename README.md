## Le Chiffre Poker AI

A Rust project for experimenting with poker-playing agents using the `rs_poker` library.

## Agent Strategies

### Implemented in this project

- **Random agent**: The current executable runs a heads-up simulation between two seeded `rs_poker` `RandomAgent`s. It provides a simple baseline for checking that a simulation runs.
- **LeChiffreIA**: A custom prototype that selects between two heuristic policies. Its GTO-inspired policy bets when there is no current bet and otherwise calls. Its exploitative policy applies a larger bet or raise. The choice is randomized, with the exploitative policy becoming more likely after a decision-count threshold. This is not a trained or mathematically solved GTO strategy: opponent profiling and model integration are placeholders, and the policy does not calculate hand equity.

### Available as library benchmark agents

`rs_poker` also provides simple agents that can be used as opponents when building a simulation:

- **Folding agent**: Folds whenever possible.
- **Calling agent**: Calls whenever possible.
- **Random agent**: Selects actions according to configured probabilities.

These are useful for basic comparisons, not advanced poker strategies.

### Planned strategies

Equity-based and Counterfactual Regret Minimization (CFR) agents are possible future strategies, but `EquityAgent`, `CFRSimpleAgent`, and `CFRBasicAgent` are not implemented in this project yet.

## Run a Simulation

Run the current heads-up random-agent simulation:

```bash
cargo run
```

The program reports when the simulation completes and prints the final chip stacks. The agents use fixed seeds so their random action streams are reproducible.

## Testing

Run the Rust test suite with:

```bash
cargo test
```

Check that the project compiles without running it:

```bash
cargo check
```

To try a strategy against another agent, add both agents to the `agents` list in `src/main.rs`, then run `cargo run`. For more reliable comparisons, run multiple simulations with different seeds and compare results over many hands rather than drawing conclusions from a single game.
