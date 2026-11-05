use lechiffre_nlhe::game::GAME_ID;
use noregrets::{
    abstraction::{AbsConfig, Abstraction, Centroids, MenuShape},
    bot::Policy,
    cfr::{Blueprint, TrainConfig, Trainer},
    engine::HandConfig,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{fs, path::Path, sync::Arc};

pub const UPSTREAM_COMMIT: &str = "757f7692738069522195d2b486eec60a8b010c0c";

pub fn hand_config() -> HandConfig {
    HandConfig {
        num_players: 2,
        stack: 20,
        sb: 1,
        bb: 2,
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Metadata {
    pub format: u32,
    pub game_id: String,
    pub upstream_commit: String,
    pub mode: String,
    pub iterations: u64,
    pub seed: u64,
    pub threads: usize,
    pub centroid_samples: usize,
    pub blueprint_sha256: String,
}

pub struct Reference {
    pub policy: Policy,
    pub metadata: Metadata,
}

pub struct Settings {
    pub iterations: u64,
    pub seed: u64,
    pub threads: usize,
    pub buckets: u16,
    pub equity_rollouts: u32,
    pub centroid_samples: usize,
}

impl Settings {
    pub fn validate(&self) -> Result<(), String> {
        if self.iterations == 0
            || self.threads == 0
            || self.buckets == 0
            || self.equity_rollouts == 0
        {
            return Err("iterations, threads, buckets and equity-rollouts must be positive".into());
        }
        if self.centroid_samples > 0 && self.centroid_samples < self.buckets as usize {
            return Err("centroid-samples must be zero or at least buckets".into());
        }
        Ok(())
    }
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Calls the actual upstream Trainer; only its action menu is adapted.
pub fn train(settings: &Settings) -> Result<Reference, String> {
    settings.validate()?;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(settings.threads)
        .build()
        .map_err(|e| e.to_string())?;
    pool.install(|| {
        let cfg = AbsConfig {
            menu: MenuShape::LeChiffreReduced,
            postflop_buckets: settings.buckets,
            equity_rollouts: settings.equity_rollouts,
            dist_runouts: 12,
            runout_rollouts: settings.equity_rollouts,
            cache_cap: 200_000,
        };
        let centroids = (settings.centroid_samples > 0)
            .then(|| Centroids::train(&cfg, settings.centroid_samples, settings.seed));
        let abs = Arc::new(Abstraction::with_centroids(cfg, centroids));
        let cfg = TrainConfig {
            hand: hand_config(),
            seed: settings.seed,
            // Pruning thresholds of the 100-BB upstream game are not scaled to
            // these 20-chip hands. Disable pruning for this controlled reference.
            prune_after: u64::MAX,
            ..TrainConfig::default()
        };
        let trainer = Trainer::new(abs.clone(), cfg);
        trainer.run(settings.iterations, &|done| {
            eprintln!("NoRegrets : {done}/{} parcours", settings.iterations)
        });
        let blueprint = trainer.to_blueprint();
        let reference = Reference {
            metadata: Metadata {
                format: 1,
                game_id: GAME_ID.into(),
                upstream_commit: UPSTREAM_COMMIT.into(),
                mode: "blueprint-only-reduced-no-pruning".into(),
                iterations: blueprint.iterations,
                seed: settings.seed,
                threads: settings.threads,
                centroid_samples: settings.centroid_samples,
                blueprint_sha256: String::new(),
            },
            policy: Policy::new(blueprint, abs),
        };
        reference.validate()?;
        Ok(reference)
    })
}

impl Reference {
    pub fn validate(&self) -> Result<(), String> {
        let m = &self.metadata;
        let b = &self.policy.blueprint;
        if m.format != 1
            || m.game_id != GAME_ID
            || m.upstream_commit != UPSTREAM_COMMIT
            || m.mode != "blueprint-only-reduced-no-pruning"
            || m.iterations == 0
            || m.iterations != b.iterations
            || m.threads == 0
            || b.num_players != 2
            || !matches!(b.abs_cfg.menu, MenuShape::LeChiffreReduced)
            || b.abs_cfg.postflop_buckets == 0
            || b.abs_cfg.equity_rollouts == 0
        {
            return Err("reference incompatible with the LeChiffre reduced game".into());
        }
        if b.strategies
            .values()
            .any(|s| s.is_empty() || s.iter().any(|p| !p.is_finite() || *p < 0.0))
        {
            return Err("invalid blueprint probabilities".into());
        }
        Ok(())
    }

    pub fn save(&self, directory: &Path) -> Result<(), String> {
        self.validate()?;
        if let Some(parent) = directory.parent().filter(|p| !p.as_os_str().is_empty()) {
            fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        fs::create_dir(directory).map_err(|e| e.to_string())?;
        let path = directory.join("blueprint.bin");
        self.policy
            .blueprint
            .save(path.to_str().ok_or("invalid UTF-8 path")?)
            .map_err(|e| e.to_string())?;
        let mut metadata = self.metadata.clone();
        metadata.blueprint_sha256 = digest(&fs::read(path).map_err(|e| e.to_string())?);
        let json = serde_json::to_vec_pretty(&metadata).map_err(|e| e.to_string())?;
        fs::write(directory.join("reference.json"), json).map_err(|e| e.to_string())
    }

    pub fn load(directory: &Path) -> Result<Self, String> {
        let metadata: Metadata = serde_json::from_slice(
            &fs::read(directory.join("reference.json")).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        let path = directory.join("blueprint.bin");
        let bytes = fs::read(&path).map_err(|e| e.to_string())?;
        if digest(&bytes) != metadata.blueprint_sha256 {
            return Err("blueprint SHA-256 does not match reference.json".into());
        }
        let blueprint = Blueprint::load(path.to_str().ok_or("invalid UTF-8 path")?)
            .map_err(|e| e.to_string())?;
        let abs = Arc::new(Abstraction::with_centroids(
            blueprint.abs_cfg.clone(),
            blueprint.centroids.clone(),
        ));
        let reference = Self {
            policy: Policy::new(blueprint, abs),
            metadata,
        };
        reference.validate()?;
        Ok(reference)
    }
}
