//! NoRegrets profiles plugged into rs_poker's native seat-permutation runner.
use crate::{
    arena_config::{built_in_reference, load_profile_pool},
    rs_agent::{AgentReference, NoRegretsAgent},
    table_arena::{Bot, PreparedTable},
};
use rs_poker::arena::{
    Agent, GameState,
    agent::{AgentConfig, AgentGenerator},
    comparison::{ArenaComparison, ComparisonBuilder},
};
use serde_json::{Value, json};
use std::{fmt, fs, path::Path, sync::Arc};

struct BlueprintGenerator {
    name: String,
    reference: Arc<AgentReference>,
}

impl fmt::Debug for BlueprintGenerator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BlueprintGenerator")
            .field("name", &self.name)
            .field("metadata", &self.reference.metadata)
            .finish()
    }
}

impl AgentGenerator for BlueprintGenerator {
    fn generate(&self, player: usize, state: &GameState) -> Box<dyn Agent> {
        self.generate_seeded(player, state, Some(0))
    }

    fn generate_seeded(
        &self,
        _player: usize,
        _state: &GameState,
        seed: Option<u64>,
    ) -> Box<dyn Agent> {
        Box::new(
            NoRegretsAgent::new(self.reference.clone(), seed.unwrap_or(0))
                .with_name(self.name.clone()),
        )
    }
}

/// Native comparison plus the effective JSON/model provenance saved alongside it.
pub struct PreparedComparison {
    pub arena: ArenaComparison,
    pub profiles: Vec<Value>,
    pub stack: u32,
}

impl PreparedComparison {
    pub fn from_directory(
        directory: &Path,
        players: usize,
        num_games: usize,
        stack: Option<u32>,
        reference: Option<&Path>,
        seed: u64,
        output: Option<&Path>,
    ) -> Result<Self, String> {
        Self::new(
            load_profile_pool(directory)?,
            players,
            num_games,
            stack,
            reference,
            seed,
            output,
        )
    }

    pub fn new(
        configs: Vec<Value>,
        players: usize,
        num_games: usize,
        stack: Option<u32>,
        reference: Option<&Path>,
        seed: u64,
        output: Option<&Path>,
    ) -> Result<Self, String> {
        if !(2..=6).contains(&players) || configs.len() < players || num_games == 0 {
            return Err("choose 2..6 seats, at least that many profiles, and -n > 0".into());
        }
        // Guard factorial growth before invoking the upstream permutation count.
        let permutations = (0..players)
            .try_fold(1usize, |p, i| p.checked_mul(configs.len() - i))
            .ok_or("permutation count overflows this platform")?;
        permutations
            .checked_mul(num_games)
            .ok_or("total game count overflows this platform")?;
        let default_model = built_in_reference(players);
        let prepared = PreparedTable::prepare_profiles(
            configs,
            players,
            stack,
            reference.or(default_model.as_deref()),
        )?;
        let stack = prepared.stack;
        let names: std::collections::HashSet<_> = prepared.seats.iter().map(|s| &s.label).collect();
        if names.len() != prepared.seats.len() {
            return Err("profile names must be distinct; set a unique name in each JSON".into());
        }
        if stack > i32::MAX as u32 {
            return Err("stack exceeds the supported integer range".into());
        }
        let stack_bb = stack as f32 / 2.0;
        let mut builder = ComparisonBuilder::new()
            .players_per_table(players)
            .num_games(num_games)
            .big_blind(2)
            .small_blind(1)
            .min_stack_bb(stack_bb)
            .max_stack_bb(stack_bb)
            .ante(0)
            .seed(seed);
        if let Some(output) = output {
            builder = builder.output_dir(output);
        }
        let mut profiles = Vec::new();
        for seat in prepared.seats {
            let config = match seat.bot {
                Bot::Existing(config) => config,
                Bot::NoRegrets(reference) => AgentConfig::Generated {
                    name: Some(seat.label.clone()),
                    generator: Arc::new(BlueprintGenerator {
                        name: seat.label.clone(),
                        reference,
                    }),
                },
            };
            builder = builder.add_agent(seat.label, config);
            profiles.push(seat.config);
        }
        let arena = builder
            .build()
            .map_err(|e| e.to_string())?
            .with_max_raises_per_round(None);
        if arena.config().min_stack() != stack as i64 {
            return Err(
                "stack cannot be represented exactly by the native comparison settings".into(),
            );
        }
        Ok(Self {
            arena,
            profiles,
            stack,
        })
    }

    /// Keep effective configs and model hashes without serializing live factories.
    pub fn save_manifest(&self, output: &Path) -> Result<(), String> {
        let cfg = self.arena.config();
        let value = json!({
            "runner": "rs_poker::arena::comparison::ArenaComparison",
            "players": cfg.players_per_table,
            "num_game_states": cfg.num_games,
            "permutations_per_state": self.arena.total_permutations(),
            "planned_total_hands": self.arena.total_games(),
            "seed": cfg.seed,
            "stack": self.stack,
            "big_blind": 2, "small_blind": 1, "ante": 0,
            "max_raises_per_round": null,
            "profiles": self.profiles,
            "exact_exploitability": null,
            "note": "Native seat permutations; future deals use independent RNG substreams. Native statistics; no exact exploitability or paired confidence interval."
        });
        fs::write(
            output.join("comparison.json"),
            serde_json::to_vec_pretty(&value).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())
    }
}
