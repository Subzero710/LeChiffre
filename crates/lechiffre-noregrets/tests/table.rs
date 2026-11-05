use lechiffre_noregrets::{rs_agent::AgentReference, table_arena::PreparedTable};
use serde_json::{Value, json};
use std::sync::OnceLock;

fn six_reference() -> std::path::PathBuf {
    static P: OnceLock<std::path::PathBuf> = OnceLock::new();
    P.get_or_init(|| {
        let path = std::env::temp_dir().join(format!("lechiffre-six-table-{}", std::process::id()));
        if path.exists() {
            std::fs::remove_dir_all(&path).unwrap();
        }
        AgentReference::train_for_players(128, 42, 20, 6)
            .unwrap()
            .save(&path)
            .unwrap();
        path
    })
    .clone()
}

#[tokio::test]
async fn six_players_with_separate_seat_choices_and_multiple_noregrets() {
    let path = six_reference();
    let configs = vec![
        json!({"type":"noregrets","reference":path}),
        json!({"type":"random"}),
        json!({"type":"calling"}),
        json!({"type":"noregrets","reference":path}),
        json!({"type":"all_in"}),
        json!({"type":"folding"}),
    ];
    let table = PreparedTable::new(configs.clone(), None, None).unwrap();
    let r = table.run(120, 17).await.unwrap();
    assert_eq!(r.players, 6);
    assert_eq!(r.cycles, 20);
    assert_eq!(r.seats.len(), 6);
    assert!((r.seats.iter().map(|s| s.total_bb).sum::<f64>()).abs() < 1e-9);
    assert_eq!(r.seats[1].agent, "random");
    assert_eq!(r.seats[4].agent, "all_in");
    for p in [0, 3] {
        assert!(r.seats[p].noregrets.as_ref().unwrap().decisions > 0);
    }
    let other = PreparedTable::new(configs, None, None)
        .unwrap()
        .run(120, 17)
        .await
        .unwrap();
    for p in 0..6 {
        assert_eq!(r.seats[p].total_bb, other.seats[p].total_bb);
        assert_eq!(r.seats[p].ci95, other.seats[p].ci95);
    }
}

#[tokio::test]
async fn existing_agents_work_at_every_supported_table_size() {
    for n in 2..=6 {
        let types = [
            "random",
            "calling",
            "all_in",
            "folding",
            "random_pot_control",
            "cfr_basic",
        ];
        let values: Vec<Value> = types[..n].iter().map(|t| json!({"type":t})).collect();
        let table = PreparedTable::new(values, Some(20), None).unwrap();
        let r = table.run(4 * n as u64, 21).await.unwrap();
        assert_eq!(r.players, n);
        assert_eq!(r.seats.iter().map(|s| s.total_bb).sum::<f64>(), 0.0);
    }
    let configs = [
        "random",
        "calling",
        "all_in",
        "folding",
        "cfr_basic",
        "cfr_simple",
    ]
    .iter()
    .map(|t| json!({"type":t}))
    .collect();
    let r = PreparedTable::new(configs, Some(20), None)
        .unwrap()
        .run(24, 5)
        .await
        .unwrap();
    assert_eq!(r.seats.iter().map(|s| s.total_bb).sum::<f64>(), 0.0);
}

#[tokio::test]
async fn noregrets_works_with_three_four_and_five_players() {
    for n in 3..=5 {
        let path = std::env::temp_dir().join(format!("lechiffre-multi-{n}-{}", std::process::id()));
        if path.exists() {
            std::fs::remove_dir_all(&path).unwrap();
        }
        AgentReference::train_for_players(64, 21, 20, n)
            .unwrap()
            .save(&path)
            .unwrap();
        let mut configs = vec![json!({"type":"random"}); n];
        configs[n - 1] = json!({"type":"noregrets","reference":path});
        let r = PreparedTable::new(configs, None, None)
            .unwrap()
            .run(12 * n as u64, 11)
            .await
            .unwrap();
        assert!(r.seats[n - 1].noregrets.as_ref().unwrap().decisions > 0);
        assert_eq!(r.seats.iter().map(|s| s.total_bb).sum::<f64>(), 0.0);
        std::fs::remove_dir_all(path).unwrap();
    }
}

#[tokio::test]
async fn reject_wrong_models_and_unbalanced_or_empty_tables() {
    let path = six_reference();
    assert!(
        PreparedTable::new(
            vec![
                json!({"type":"noregrets","reference":path}),
                json!({"type":"random"})
            ],
            None,
            None
        )
        .is_err()
    );
    assert!(PreparedTable::new(vec![], None, None).is_err());
    assert!(PreparedTable::new(vec![json!({"type":"calling"}); 7], None, None).is_err());
    let table = PreparedTable::new(vec![json!({"type":"calling"}); 6], None, None).unwrap();
    assert!(table.run(13, 1).await.is_err());
    assert!(table.run(6, 1).await.is_err());
    let old = AgentReference::load(std::path::Path::new(
        "../../references/noregrets-hu10bb-agent-10000-seed42",
    ))
    .unwrap();
    assert_eq!(old.metadata.num_players, 2);
}
