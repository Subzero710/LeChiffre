use lechiffre_noregrets::{
    arena_config::load_profile_files, comparison::PreparedComparison, rs_agent::AgentReference,
};
use serde_json::json;
use std::{fs, path::PathBuf};
fn temp() -> PathBuf {
    let p = std::env::temp_dir().join(format!("noregrets-selection-{}", std::process::id()));
    if p.exists() {
        fs::remove_dir_all(&p).unwrap();
    }
    fs::create_dir_all(&p).unwrap();
    p
}
#[tokio::test]
async fn a_shared_profile_can_fill_two_seats_without_copied_configurations() {
    let root = temp();
    let model = root.join("model");
    AgentReference::train_for_players(2, 2, 6, 3)
        .unwrap()
        .save(&model)
        .unwrap();
    let caller = root.join("calling.json");
    let learned = root.join("noregrets.json");
    fs::write(&caller, r#"{"type":"calling","name":"Calling"}"#).unwrap();
    fs::write(
        &learned,
        serde_json::to_vec(&json!({"type":"noregrets","name":"Learned","reference":model}))
            .unwrap(),
    )
    .unwrap();
    let profiles = load_profile_files(&[learned, caller.clone(), caller]).unwrap();
    let comparison = PreparedComparison::new(profiles, 3, 1, None, None, 7, None).unwrap();
    assert_eq!(comparison.arena.num_agents(), 3);
    assert_eq!(comparison.arena.total_games(), 6);
    let result = comparison.arena.run().await.unwrap();
    assert_eq!(result.agent_names().len(), 3);
    for s in result.all_stats().values() {
        assert_eq!(s.total_games, 6);
        assert!(s.position_stats.iter().all(|p| p.games_played == 2));
    }
    fs::remove_dir_all(root).unwrap();
}
