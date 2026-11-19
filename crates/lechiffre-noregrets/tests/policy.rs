use lechiffre_noregrets::{
    rs_agent::{AgentReference, NoRegretsAgent},
    table_arena::PreparedTable,
};
use rs_poker::{
    arena::{
        Agent, GameStateBuilder, HoldemSimulationBuilder,
        action::{Action, GameStartPayload},
    },
    core::{Card, Hand},
};
use serde_json::json;
use std::{fs, path::PathBuf, sync::Arc};
fn temp(label: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("noregrets-rsp-{label}-{}", std::process::id()));
    if p.exists() {
        fs::remove_dir_all(&p).unwrap();
    }
    fs::create_dir_all(&p).unwrap();
    p
}

#[test]
fn a_failed_replacement_restores_the_previous_blueprint() {
    let root = temp("rollback");
    let model = AgentReference::train_for_players(2, 99, 6, 2).unwrap();
    model.save(&root).unwrap();
    let before = fs::read(root.join("blueprint.bin")).unwrap();
    let metadata = fs::read(root.join("agent-reference.json")).unwrap();
    fs::remove_file(root.join("agent-reference.json")).unwrap();
    fs::create_dir(root.join("agent-reference.json")).unwrap();
    fs::write(root.join("agent-reference.json/keep.txt"), "keep").unwrap();
    assert!(model.save(&root).is_err());
    assert_eq!(fs::read(root.join("blueprint.bin")).unwrap(), before);
    assert_eq!(
        fs::read_to_string(root.join("agent-reference.json/keep.txt")).unwrap(),
        "keep"
    );
    fs::remove_dir_all(root.join("agent-reference.json")).unwrap();
    fs::write(root.join("agent-reference.json"), metadata).unwrap();
    assert_eq!(AgentReference::load(&root).unwrap().metadata.seed, 99);
    fs::remove_dir_all(root).unwrap();
}
#[test]
fn original_noregrets_trainer_produces_normalized_blueprints() {
    let a = AgentReference::train_for_players(3, 42, 6, 2).unwrap();
    assert!(!a.policy.blueprint.strategies.is_empty());
    assert_eq!(a.policy.blueprint.iterations, 3);
    assert_eq!(
        a.metadata.source,
        lechiffre_noregrets::reference::UPSTREAM_COMMIT
    );
    assert_eq!(a.metadata.engine, "rs_poker");
    for probs in a.policy.blueprint.strategies.values() {
        assert!(probs.iter().all(|p| p.is_finite() && *p >= 0.0));
        assert!((probs.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    }
}
#[tokio::test]
async fn other_players_private_cards_do_not_change_the_blueprint_decision() {
    let model = Arc::new(AgentReference::train_for_players(2, 1, 6, 2).unwrap());
    let gs = GameStateBuilder::new()
        .stacks([6; 2])
        .blinds(2, 1)
        .max_raises_per_round(None)
        .build()
        .unwrap();
    let mut sim = HoldemSimulationBuilder::default()
        .game_state(gs)
        .build_with_rng(rs_poker::arena::seeded_rng(Some(2)))
        .unwrap();
    assert!(sim.advance_to_decision().await);
    let gs = sim.game_state;
    let other = 1 - gs.to_act_idx();
    let mut changed = gs.clone();
    let known: Vec<_> = gs.hands[gs.to_act_idx()].iter().collect();
    let replacement: Vec<_> = (0u8..52)
        .map(Card::from)
        .filter(|c| !known.contains(c) && !gs.hands[other].contains(c))
        .take(2)
        .collect();
    changed.hands[other] = Hand::new_with_cards(replacement);
    let mut a = NoRegretsAgent::new(model.clone(), 17);
    let mut b = NoRegretsAgent::new(model, 17);
    let start = Action::GameStart(GameStartPayload {
        ante: 0,
        small_blind: 1,
        big_blind: 2,
    });
    a.historian()
        .unwrap()
        .record_action(123, &gs, &start)
        .await
        .unwrap();
    b.historian()
        .unwrap()
        .record_action(123, &changed, &start)
        .await
        .unwrap();
    assert_eq!(
        a.try_action(123, &gs).unwrap(),
        b.try_action(123, &changed).unwrap()
    );
}
#[test]
fn model_replacement_preserves_personal_files_and_refuses_legacy_engine() {
    let root = temp("overwrite");
    let a = AgentReference::train_for_players(2, 7, 6, 2).unwrap();
    a.save(&root).unwrap();
    let first = AgentReference::load(&root)
        .unwrap()
        .metadata
        .blueprint_sha256;
    fs::write(root.join("personal.txt"), "keep").unwrap();
    let b = AgentReference::train_for_players(3, 8, 6, 2).unwrap();
    b.save(&root).unwrap();
    let loaded = AgentReference::load(&root).unwrap();
    assert_ne!(first, loaded.metadata.blueprint_sha256);
    assert_eq!(loaded.metadata.seed, 8);
    assert_eq!(
        fs::read_to_string(root.join("personal.txt")).unwrap(),
        "keep"
    );
    assert!(AgentReference::train_for_players(0, 9, 6, 2).is_err());
    assert_eq!(AgentReference::load(&root).unwrap().metadata.seed, 8);
    let mut m: serde_json::Value =
        serde_json::from_slice(&fs::read(root.join("agent-reference.json")).unwrap()).unwrap();
    m["format"] = json!("noregrets-rs-agent-unlimited-v2");
    fs::write(
        root.join("agent-reference.json"),
        serde_json::to_vec(&m).unwrap(),
    )
    .unwrap();
    assert!(
        AgentReference::load(&root)
            .err()
            .unwrap()
            .contains("reentrainer")
    );
    fs::remove_dir_all(root).unwrap();
}
#[tokio::test]
async fn actual_noregrets_plays_existing_agents_at_two_three_and_six_players() {
    for n in [2, 3, 6] {
        let root = temp(&format!("table{n}"));
        AgentReference::train_for_players(2, 3, 6, n)
            .unwrap()
            .save(&root)
            .unwrap();
        let mut profiles = vec![json!({"type":"noregrets","reference":root,"name":"Learned"})];
        let kinds = ["calling", "equity", "folding", "random", "calling"];
        profiles.extend(kinds[..n - 1].iter().map(|k| json!({"type":k})));
        let table = PreparedTable::new(profiles, None, None).unwrap();
        let report = table.run(2 * n as u64, 17).await.unwrap();
        assert_eq!(report.seats.iter().map(|s| s.total_bb).sum::<f64>(), 0.0);
        assert!(
            report.seats[0]
                .noregrets
                .as_ref()
                .unwrap()
                .errors
                .is_empty()
        );
        fs::remove_dir_all(root).unwrap();
    }
}
#[test]
fn a_model_refuses_incompatible_game_rules() {
    let model = Arc::new(AgentReference::train_for_players(1, 1, 6, 2).unwrap());
    let mut agent = NoRegretsAgent::new(model, 1);
    let gs = GameStateBuilder::new()
        .stacks([200; 2])
        .blinds(2, 1)
        .max_raises_per_round(None)
        .build()
        .unwrap();
    assert!(agent.try_action(1, &gs).err().unwrap().contains("matching"));
}
