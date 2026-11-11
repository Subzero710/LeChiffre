use lechiffre_noregrets::{
    rs_agent::{AgentReference, NoRegretsAgent},
    rs_arena,
};
use rs_poker::arena::{
    Agent,
    action::{Action, GameStartPayload},
};
use std::sync::{Arc, OnceLock};

fn reference() -> Arc<AgentReference> {
    static R: OnceLock<Arc<AgentReference>> = OnceLock::new();
    R.get_or_init(|| Arc::new(AgentReference::train(256, 42, 20).unwrap()))
        .clone()
}

#[tokio::test]
async fn actual_existing_agents_run_inside_the_real_simulator() {
    for name in [
        "random",
        "calling",
        "all_in",
        "folding",
        "random_pot_control",
        "cfr_basic",
        "cfr_simple",
    ] {
        let r = rs_arena::duel(
            reference(),
            name,
            rs_arena::opponent_config(name).unwrap(),
            8,
            11,
        )
        .await
        .unwrap();
        assert!(r.noregrets.errors.is_empty(), "{name}");
        assert!(r.noregrets.decisions > 0, "{name}");
        assert!((r.bb_per_100 - r.total_bb / 8.0 * 100.0).abs() < 1e-9);
    }
}

#[tokio::test]
async fn normal_agent_trait_and_hidden_card_independence() {
    let mut gs = rs_poker::arena::GameStateBuilder::new()
        .stacks([20, 20])
        .blinds(2, 1)
        .dealer_idx(0)
        .max_raises_per_round(None)
        .build()
        .unwrap();
    gs.advance_round(); // Starting -> Ante
    gs.advance_round(); // Ante -> DealPreflop
    gs.advance_round(); // DealPreflop -> Preflop
    gs.do_bet(1, true).unwrap();
    gs.do_bet(2, true).unwrap();
    gs.sb_posted = true;
    gs.bb_posted = true;
    gs.hands[0] = rs_poker::core::Hand::new_with_cards(vec![0u8.into(), 1u8.into()]);
    gs.hands[1] = rs_poker::core::Hand::new_with_cards(vec![2u8.into(), 3u8.into()]);
    let mut other = gs.clone();
    other.hands[1] = rs_poker::core::Hand::new_with_cards(vec![30u8.into(), 31u8.into()]);
    let mut a = NoRegretsAgent::new(reference(), 9);
    let mut b = NoRegretsAgent::new(reference(), 9);
    let event = Action::GameStart(GameStartPayload {
        ante: 0,
        small_blind: 1,
        big_blind: 2,
    });
    a.historian()
        .unwrap()
        .record_action(7, &gs, &event)
        .await
        .unwrap();
    b.historian()
        .unwrap()
        .record_action(7, &other, &event)
        .await
        .unwrap();
    assert_eq!(
        a.try_action(7, &gs).unwrap(),
        b.try_action(7, &other).unwrap()
    );
    let boxed: Box<dyn Agent> = Box::new(a);
    assert_eq!(boxed.name(), "NoRegrets");
    let mut wrong = NoRegretsAgent::new(reference(), 9);
    assert!(wrong.try_action(7, &gs).is_err());
    let mut capped = gs.clone();
    capped.max_raises_per_round = Some(1);
    assert!(b.try_action(7, &capped).is_err());
    // Canonicalization must also work when the real table's button is seat 1.
    for dealer in 0..2 {
        let agent = NoRegretsAgent::new(reference(), 19);
        let stats = agent.stats();
        let gs = rs_poker::arena::GameStateBuilder::new()
            .stacks([20, 20])
            .blinds(2, 1)
            .dealer_idx(dealer)
            .max_raises_per_round(None)
            .build()
            .unwrap();
        let other = rs_poker::arena::agent::ConfigAgentBuilder::new(
            rs_arena::opponent_config("random").unwrap(),
        )
        .unwrap()
        .player_idx(1)
        .rng_seed(21)
        .build();
        let agents: Vec<Box<dyn Agent>> = vec![Box::new(agent), other];
        let mut sim = rs_poker::arena::HoldemSimulationBuilder::default()
            .game_state(gs)
            .agents(agents)
            .build_with_rng(rs_poker::arena::seeded_rng(Some(23)))
            .unwrap();
        sim.run().await;
        assert!(stats.lock().unwrap().errors.is_empty());
    }
}

#[tokio::test]
async fn seeded_random_matches_are_reproducible_and_metadata_separates_modes() {
    let r = reference();
    let a = rs_arena::duel(
        r.clone(),
        "random",
        rs_arena::opponent_config("random").unwrap(),
        20,
        43,
    )
    .await
    .unwrap();
    let b = rs_arena::duel(
        r,
        "random",
        rs_arena::opponent_config("random").unwrap(),
        20,
        43,
    )
    .await
    .unwrap();
    assert_eq!(a.total_bb, b.total_bb);
    assert_eq!(a.ci95, b.ci95);
    assert!(
        AgentReference::load(std::path::Path::new(
            "../../references/noregrets-hu10bb-10000-seed42"
        ))
        .is_err()
    );
    for n in ["cfr_basic", "cfr_simple"] {
        let v = serde_json::to_value(rs_arena::opponent_config(n).unwrap()).unwrap();
        assert_eq!(v["hand_estimator"], "uniform");
    }
}
