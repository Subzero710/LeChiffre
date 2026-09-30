#![no_main]

extern crate arbitrary;
extern crate libfuzzer_sys;
extern crate rand;
extern crate rs_poker;

use rand::{rngs::StdRng, SeedableRng};
use std::sync::atomic::{AtomicUsize, Ordering};

use rs_poker::arena::{
    action::AgentAction,
    agent::VecReplayAgent,
    historian::{self, OpenHandHistoryVecHistorian},
    test_util::assert_valid_game_state,
    test_util::assert_valid_round_data,
    Agent, GameStateBuilder, HoldemSimulation, HoldemSimulationBuilder,
};
use rs_poker::open_hand_history::{
    assert_open_hand_history_matches_game_state, assert_valid_open_hand_history,
};

use libfuzzer_sys::fuzz_target;
use rs_poker::Chips;

#[derive(Debug, Clone, arbitrary::Arbitrary)]
struct PlayerInput {
    pub stack: Chips,
    pub actions: Vec<AgentAction>,
}

#[derive(Debug, Clone, arbitrary::Arbitrary)]
struct MultiInput {
    pub players: Vec<PlayerInput>,
    pub sb: Chips,
    pub bb: Chips,
    pub ante: Chips,
    pub dealer_idx: usize,
    pub seed: u64,
}

fn build_agent(actions: Vec<AgentAction>) -> Box<dyn Agent> {
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    let idx = COUNTER.fetch_add(1, Ordering::Relaxed);
    Box::<VecReplayAgent>::new(VecReplayAgent::new(
        format!("multi-replay-agent-{idx}"),
        actions,
    ))
}

fn input_good(input: &MultiInput) -> bool {
    if !(2..=16).contains(&input.players.len()) {
        return false;
    }
    if input
        .players
        .iter()
        .any(|p| !(0..=100_000_000).contains(&p.stack))
    {
        return false;
    }
    if input.ante < 0
        || input.sb < input.ante
        || input.sb < 0
        || input.bb < input.sb
        || !(1..=100_000_000).contains(&input.bb)
    {
        return false;
    }
    let min_stack = input
        .players
        .iter()
        .map(|p| p.stack)
        .min()
        .expect("nonempty validated player list");
    if input
        .bb
        .checked_add(input.ante)
        .is_none_or(|required| required > min_stack)
    {
        return false;
    }

    // All bet actions are valid
    for player in &input.players {
        for action in &player.actions {
            if let AgentAction::Bet(bet) = action {
                if *bet < 0 || *bet < input.bb {
                    return false;
                }
            }
        }
    }

    true
}

fuzz_target!(|input: MultiInput| {
    let sb = input.sb;
    let bb = input.bb;
    let ante = input.ante;

    if !input_good(&input) {
        return;
    }

    let stacks: Vec<Chips> = input.players.iter().map(|pi| pi.stack).collect();

    let agents: Vec<Box<dyn Agent>> = input
        .players
        .into_iter()
        .map(|pi| build_agent(pi.actions))
        .collect();

    let open_hand_hist = Box::new(OpenHandHistoryVecHistorian::new());
    let hand_storage = open_hand_hist.get_storage();
    let historians: Vec<Box<dyn historian::Historian>> = vec![open_hand_hist];

    // Create the game state using the builder
    // Notice that dealer_idx is sanitized to ensure it's in the proper range here
    // rather than with the rest of the safety checks.
    let game_state = match GameStateBuilder::new()
        .stacks(stacks)
        .blinds(bb, sb)
        .ante(ante)
        .dealer_idx(input.dealer_idx % agents.len())
        .build()
    {
        Ok(gs) => gs,
        Err(_) => return, // Invalid input, skip
    };
    // Do the thing
    let mut sim: HoldemSimulation = HoldemSimulationBuilder::default()
        .game_state(game_state)
        .agents(agents)
        .historians(historians)
        .build_with_rng(StdRng::seed_from_u64(input.seed))
        .unwrap();

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(sim.run());

    // for _record in records.borrow().iter() {
    //     // println!("{:?}", record.action);
    // }
    assert_valid_round_data(&sim.game_state.round_data);
    assert_valid_game_state(&sim.game_state);

    let hands = hand_storage.lock().unwrap();
    assert!(!hands.is_empty());
    for hand in hands.iter() {
        if std::env::var_os("DUMP_HAND").is_some() {
            println!("{hand:#?}");
        }
        assert_valid_open_hand_history(hand);
        assert_open_hand_history_matches_game_state(hand, &sim.game_state);
    }
});
