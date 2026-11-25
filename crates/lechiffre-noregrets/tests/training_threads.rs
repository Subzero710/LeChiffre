use lechiffre_noregrets::{rs_agent::AgentReference, table_arena::PreparedTable};
use serde_json::json;
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    sync::atomic::{AtomicUsize, Ordering},
};

struct Scratch(PathBuf);

impl Scratch {
    fn new(label: &str) -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "lechiffre-threads-{label}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[tokio::test]
async fn parallel_training_produces_a_loadable_playable_blueprint() {
    let root = Scratch::new("parallel");
    let model = AgentReference::train_for_players_with_threads(1024, 42, 6, 2, 4).unwrap();
    assert_eq!(model.metadata.training_threads, Some(4));
    assert_eq!(model.policy.blueprint.iterations, 1024);
    assert!(!model.policy.blueprint.strategies.is_empty());
    for probabilities in model.policy.blueprint.strategies.values() {
        assert!(probabilities.iter().all(|p| p.is_finite() && *p >= 0.0));
        assert!((probabilities.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    }
    model.save(root.path()).unwrap();
    assert_eq!(
        AgentReference::load(root.path())
            .unwrap()
            .metadata
            .training_threads,
        Some(4)
    );
    let table = PreparedTable::new(
        vec![
            json!({"type":"noregrets","reference":root.path()}),
            json!({"type":"calling"}),
        ],
        None,
        None,
    )
    .unwrap();
    let report = table.run(12, 123).await.unwrap();
    assert_eq!(
        report.seats.iter().map(|seat| seat.total_bb).sum::<f64>(),
        0.0
    );
    assert!(
        report.seats[0]
            .noregrets
            .as_ref()
            .unwrap()
            .errors
            .is_empty()
    );
}

#[test]
fn one_thread_is_reproducible_and_the_existing_api_still_works() {
    let first = AgentReference::train_for_players(512, 73, 6, 2).unwrap();
    let second = AgentReference::train_for_players_with_threads(512, 73, 6, 2, 1).unwrap();
    assert_eq!(first.metadata.training_threads, Some(1));
    assert_eq!(
        first.policy.blueprint.strategies,
        second.policy.blueprint.strategies
    );
    assert!(AgentReference::train_for_players_with_threads(1, 42, 6, 2, 0).is_err());
}

#[test]
fn old_model_metadata_remains_compatible() {
    let root = Scratch::new("legacy");
    AgentReference::train_for_players(2, 42, 6, 2)
        .unwrap()
        .save(root.path())
        .unwrap();
    let path = root.path().join("agent-reference.json");
    let mut metadata: serde_json::Value =
        serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    metadata.as_object_mut().unwrap().remove("training_threads");
    fs::write(path, serde_json::to_vec(&metadata).unwrap()).unwrap();
    assert_eq!(
        AgentReference::load(root.path())
            .unwrap()
            .metadata
            .training_threads,
        None
    );
}

#[test]
fn cli_uses_the_requested_threads_and_rejects_zero_before_replacing_a_model() {
    let root = Scratch::new("cli");
    let binary = env!("CARGO_BIN_EXE_lechiffre-noregrets");
    let output = Command::new(binary)
        .args([
            "train-agent",
            "--players",
            "3",
            "--stack",
            "6",
            "--iterations",
            "769",
            "--threads",
            "2",
            "--seed",
            "42",
            "--out",
        ])
        .arg(root.path())
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let model = AgentReference::load(root.path()).unwrap();
    assert_eq!(model.metadata.training_threads, Some(2));
    assert_eq!(model.metadata.iterations, 769);
    assert_eq!(model.metadata.num_players, 3);
    let metadata = fs::read(root.path().join("agent-reference.json")).unwrap();
    let blueprint = fs::read(root.path().join("blueprint.bin")).unwrap();
    let failed = Command::new(binary)
        .args(["train-agent", "--threads", "0", "--out"])
        .arg(root.path())
        .output()
        .unwrap();
    assert!(!failed.status.success());
    assert!(String::from_utf8_lossy(&failed.stderr).contains("--threads"));
    assert_eq!(
        fs::read(root.path().join("agent-reference.json")).unwrap(),
        metadata
    );
    assert_eq!(
        fs::read(root.path().join("blueprint.bin")).unwrap(),
        blueprint
    );
}

#[test]
fn cli_selects_available_parallelism_when_threads_are_omitted() {
    let root = Scratch::new("auto");
    let output = Command::new(env!("CARGO_BIN_EXE_lechiffre-noregrets"))
        .args([
            "train-agent",
            "--iterations",
            "1",
            "--players",
            "2",
            "--stack",
            "6",
            "--out",
        ])
        .arg(root.path())
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        AgentReference::load(root.path())
            .unwrap()
            .metadata
            .training_threads,
        Some(std::thread::available_parallelism().map_or(1, |threads| threads.get()))
    );
}
