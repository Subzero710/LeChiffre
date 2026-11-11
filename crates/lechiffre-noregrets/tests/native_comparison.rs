use lechiffre_noregrets::{comparison::PreparedComparison, rs_agent::AgentReference};
use serde_json::{Value, json};
use std::{
    collections::{HashMap, HashSet},
    fs,
    path::PathBuf,
};

fn temp(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("lechiffre-native-{name}-{}", std::process::id()));
    if path.exists() {
        fs::remove_dir_all(&path).unwrap();
    }
    fs::create_dir_all(&path).unwrap();
    path
}

#[tokio::test]
async fn trained_versions_use_every_seat_order_and_native_statistics() {
    let root = temp("versions");
    let mut profiles = Vec::new();
    for (name, iterations, seed) in [("v1", 32, 17), ("v2", 64, 18)] {
        let model = root.join(name);
        AgentReference::train_for_players(iterations, seed, 20, 3)
            .unwrap()
            .save(&model)
            .unwrap();
        profiles.push(json!({"type":"noregrets","name":name,"reference":model}));
    }
    profiles.push(json!({"type":"calling","name":"caller"}));
    let prepared =
        PreparedComparison::new(profiles.clone(), 3, 2, None, None, 123, Some(&root)).unwrap();
    assert_eq!(prepared.arena.total_permutations(), 6);
    assert_eq!(prepared.arena.total_games(), 12);
    prepared.save_manifest(&root).unwrap();
    let manifest: Value =
        serde_json::from_slice(&fs::read(root.join("comparison.json")).unwrap()).unwrap();
    let a = &manifest["profiles"][0]["metadata"];
    let b = &manifest["profiles"][1]["metadata"];
    assert_eq!(a["iterations"], 32);
    assert_eq!(b["iterations"], 64);
    assert_ne!(a["blueprint_sha256"], b["blueprint_sha256"]);
    let mut orders = HashMap::<Vec<String>, usize>::new();
    let result = prepared
        .arena
        .run_with_callback(|hand| {
            assert_eq!((0..3).map(|p| hand.stats.total_profit[p]).sum::<i64>(), 0);
            *orders.entry(hand.agent_names).or_default() += 1;
        })
        .await
        .unwrap();
    assert_eq!(orders.len(), 6);
    assert!(orders.values().all(|&n| n == 2));
    for name in ["v1", "v2", "caller"] {
        let stats = result.get_agent_stats(name).unwrap();
        assert_eq!(stats.total_games, 12);
        assert_eq!(stats.position_stats.len(), 3);
        assert!(stats.position_stats.iter().all(|s| s.games_played == 4));
    }
    assert_eq!(
        result
            .all_stats()
            .values()
            .map(|s| s.total_profit)
            .sum::<i64>(),
        0
    );
    let history = fs::read_to_string(root.join("hands.jsonl")).unwrap();
    assert_eq!(history.lines().filter(|l| !l.trim().is_empty()).count(), 12);
    let history_orders: HashSet<Vec<String>> = history
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|line| {
            let v: Value = serde_json::from_str(line).unwrap();
            v["ohh"]["players"]
                .as_array()
                .unwrap()
                .iter()
                .map(|p| p["name"].as_str().unwrap().to_owned())
                .collect()
        })
        .collect();
    assert_eq!(history_orders.len(), 6);
    result.save_to_dir(&root).unwrap();
    assert!(root.join("results.json").exists());
    assert!(PreparedComparison::new(profiles, 2, 1, None, None, 123, None).is_err());
    fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn more_profiles_than_seats_and_seeded_replay() {
    // Seven profiles are legal even though each table has only two seats.
    let profiles: Vec<_> = (0..7)
        .map(|i| json!({"type":"random","name":format!("random-{i}")}))
        .collect();
    let run = || PreparedComparison::new(profiles.clone(), 2, 1, None, None, 99, None).unwrap();
    let a = run();
    assert_eq!(a.arena.total_permutations(), 42);
    let a = a.arena.run().await.unwrap();
    let b = run().arena.run().await.unwrap();
    for name in a.agent_names() {
        let stats = a.get_agent_stats(name).unwrap();
        assert_eq!(stats.total_games, 12);
        assert!(stats.position_stats.iter().all(|s| s.games_played == 6));
        assert_eq!(
            serde_json::to_value(stats).unwrap(),
            serde_json::to_value(b.get_agent_stats(name).unwrap()).unwrap()
        );
    }
}

#[test]
fn bad_models_and_table_settings_fail_before_simulation() {
    let configs = || {
        vec![
            json!({"type":"noregrets","name":"default"}),
            json!({"type":"calling"}),
        ]
    };
    assert!(PreparedComparison::new(configs(), 2, 1, Some(40), None, 1, None).is_err());
    assert!(PreparedComparison::new(configs(), 3, 1, None, None, 1, None).is_err());
    assert!(PreparedComparison::new(configs(), 2, 0, None, None, 1, None).is_err());
    let mut missing = configs();
    missing[0]["reference"] = json!("does-not-exist");
    assert!(PreparedComparison::new(missing, 2, 1, None, None, 1, None).is_err());
}
