use lechiffre_noregrets::{
    arena_config::{load_directory, prepare_directory},
    rs_agent::AgentReference,
};
use serde_json::{Value, json};
use std::{
    fs,
    path::{Path, PathBuf},
};

fn temp(name: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("lechiffre-arena-{name}-{}", std::process::id()));
    if p.exists() {
        fs::remove_dir_all(&p).unwrap();
    }
    fs::create_dir_all(&p).unwrap();
    p
}
fn write(p: &Path, v: Value) {
    fs::write(p, serde_json::to_vec_pretty(&v).unwrap()).unwrap();
}

#[tokio::test]
async fn different_blueprints_keep_separate_names_models_and_history() {
    let root = temp("versions");
    let configs = root.join("agents");
    fs::create_dir(&configs).unwrap();
    for (name, iterations, seed) in [("v1", 64, 7), ("v2", 128, 8)] {
        let model = root.join(name);
        AgentReference::train_for_players(iterations, seed, 20, 6)
            .unwrap()
            .save(&model)
            .unwrap();
        write(
            &configs.join(format!("{name}.json")),
            json!({"type":"noregrets","name":name,"reference":model}),
        );
    }
    for kind in ["random", "calling", "all_in", "random_pot_control"] {
        write(&configs.join(format!("{kind}.json")), json!({"type":kind}));
    }
    let table = prepare_directory(&configs, None, None).unwrap();
    let history = root.join("hands.jsonl");
    let mut count = 0;
    let report = table
        .run_observed(24, 123, Some(&history), |hand| {
            count += 1;
            assert_eq!(hand.number, count);
            assert_eq!(hand.profits.iter().sum::<i64>(), 0);
            assert!(hand.names.contains(&"v1".into()) && hand.names.contains(&"v2".into()));
            true
        })
        .await
        .unwrap()
        .unwrap();
    assert_eq!(count, 24);
    let v1 = report.seats.iter().find(|s| s.agent == "v1").unwrap();
    let v2 = report.seats.iter().find(|s| s.agent == "v2").unwrap();
    assert_eq!(v1.config["metadata"]["iterations"], 64);
    assert_eq!(v2.config["metadata"]["iterations"], 128);
    assert_ne!(
        v1.config["metadata"]["blueprint_sha256"],
        v2.config["metadata"]["blueprint_sha256"]
    );
    let lines = fs::read_to_string(history).unwrap();
    assert_eq!(
        lines.lines().filter(|line| !line.trim().is_empty()).count(),
        24
    );
    for line in lines.lines().filter(|line| !line.trim().is_empty()) {
        let v: Value = serde_json::from_str(line).unwrap();
        let players = v["ohh"]["players"].as_array().unwrap();
        let names: Vec<_> = players
            .iter()
            .map(|p| p["name"].as_str().unwrap())
            .collect();
        assert!(names.contains(&"v1") && names.contains(&"v2"));
    }
    fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn stock_only_default_model_and_cancelled_runs() {
    let root = temp("stock");
    write(
        &root.join("01.json"),
        json!({"type":"calling","name":"Caller"}),
    );
    write(
        &root.join("02.json"),
        json!({"type":"calling","name":"Caller"}),
    );
    fs::write(root.join("ignored.json.example"), "not JSON").unwrap();
    assert_eq!(load_directory(&root).unwrap().len(), 2);
    let table = prepare_directory(&root, None, None).unwrap();
    let report = table.run(8, 5).await.unwrap();
    assert_ne!(report.seats[0].agent, report.seats[1].agent);
    let cancelled = table
        .run_observed(8, 5, Some(&root.join("hands.jsonl")), |_| false)
        .await
        .unwrap();
    assert!(cancelled.is_none());
    write(
        &root.join("01.json"),
        json!({"type":"noregrets","name":"Default-HU"}),
    );
    let report = prepare_directory(&root, None, None)
        .unwrap()
        .run(4, 8)
        .await
        .unwrap();
    assert!(report.seats[0].noregrets.is_some());
    write(
        &root.join("01.json"),
        json!({"type":"noregrets","reference":"does-not-exist"}),
    );
    assert!(prepare_directory(&root, None, None).is_err());
    fs::remove_dir_all(root).unwrap();
}
