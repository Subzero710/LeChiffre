//! Regression coverage for the merge of Generated agents and Equity.
use lechiffre_noregrets::{
    arena_config::built_in_reference, comparison::PreparedComparison, table_arena::PreparedTable,
};
use rs_poker::arena::{
    Agent, GameState,
    agent::{AgentConfig, AgentGenerator, AllInAgent},
    rotation::RotationBuilder,
};
use serde_json::json;
use std::sync::Arc;

#[test]
fn two_calling_files_make_six_separate_profiles() {
    let root = std::env::temp_dir().join(format!("lechiffre-six-profiles-{}", std::process::id()));
    if root.exists() {
        std::fs::remove_dir_all(&root).unwrap();
    }
    std::fs::create_dir_all(&root).unwrap();
    for (file, config) in [
        (
            "noregrets.json",
            json!({"type":"noregrets","name":"Blueprint"}),
        ),
        ("equity.json", equity()),
        ("calling.json", json!({"type":"calling","name":"Calling"})),
        ("folding.json", json!({"type":"folding"})),
        ("random.json", json!({"type":"random"})),
    ] {
        std::fs::write(root.join(file), serde_json::to_vec(&config).unwrap()).unwrap();
    }
    assert!(PreparedComparison::from_directory(&root, 6, 1, None, None, 123, None).is_err());
    std::fs::copy(root.join("calling.json"), root.join("calling_2.json")).unwrap();
    let prepared = PreparedComparison::from_directory(&root, 6, 1, None, None, 123, None).unwrap();
    assert_eq!(prepared.arena.num_agents(), 6);
    assert_eq!(prepared.arena.total_permutations(), 720);
    let names: std::collections::HashSet<_> =
        prepared.arena.agents().iter().map(|(n, _)| n).collect();
    assert_eq!(names.len(), 6);
    assert_eq!(
        prepared
            .profiles
            .iter()
            .filter(|p| p["type"] == "calling")
            .count(),
        2
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[derive(Debug)]
struct Generator;
impl AgentGenerator for Generator {
    fn generate(&self, _player: usize, _state: &GameState) -> Box<dyn Agent> {
        Box::new(AllInAgent::new("generated-test"))
    }
}

fn equity() -> serde_json::Value {
    json!({"type":"equity", "iterations":1000,
        "small_edge":0.15,"medium_edge":0.35,"large_edge":0.60,
        "small_bet_pot":0.33,"medium_bet_pot":0.66,"large_bet_pot":1.0,
        "name":"EquityAgent"})
}

#[test]
fn rotation_accepts_names_of_generated_and_equity_agents() {
    let rotation = RotationBuilder::new()
        .players_per_table(3)
        .add_agent_config(AgentConfig::Generated {
            name: Some("Generated-blueprint".into()),
            generator: Arc::new(Generator),
        })
        .add_agent_config(serde_json::from_value(equity()).unwrap())
        .add_agent_config(
            serde_json::from_value(json!({
                "type":"calling","name":"Calling"
            }))
            .unwrap(),
        )
        .build()
        .unwrap();
    let names: Vec<_> = rotation
        .agents()
        .iter()
        .map(|(name, _)| name.as_str())
        .collect();
    assert_eq!(names, ["Generated-blueprint", "EquityAgent", "Calling"]);
}

#[tokio::test]
async fn equity_and_two_callers_play_with_noregrets_at_six_seats() {
    let table = PreparedTable::new(
        vec![
            json!({"type":"noregrets","name":"Blueprint"}),
            equity(),
            json!({"type":"calling"}),
            json!({"type":"calling"}),
            json!({"type":"random"}),
            json!({"type":"all_in"}),
        ],
        None,
        built_in_reference(6).as_deref(),
    )
    .unwrap();
    let report = table.run(12, 42).await.unwrap();
    assert_eq!(report.hands, 12);
    assert!(report.seats.iter().any(|s| s.agent == "EquityAgent"));
    assert!(report.seats.iter().any(|s| s.noregrets.is_some()));
    assert_eq!(report.seats.iter().map(|s| s.total_bb).sum::<f64>(), 0.0);
}

#[tokio::test]
async fn native_comparison_runs_equity_against_generated_blueprint() {
    let prepared = PreparedComparison::new(
        vec![json!({"type":"noregrets","name":"Blueprint"}), equity()],
        2,
        2,
        None,
        None,
        91,
        None,
    )
    .unwrap();
    let report = prepared.arena.run().await.unwrap();
    assert_eq!(report.total_games(), 4);
    for name in ["Blueprint", "EquityAgent"] {
        let stats = report.get_agent_stats(name).unwrap();
        assert_eq!(stats.total_games, 4);
        assert!(stats.position_stats.iter().all(|p| p.games_played == 2));
    }
    assert_eq!(
        report
            .all_stats()
            .values()
            .map(|s| s.total_profit)
            .sum::<i64>(),
        0
    );
}
