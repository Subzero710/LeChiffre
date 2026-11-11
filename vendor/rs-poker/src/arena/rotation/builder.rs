use crate::arena::Chips;
use crate::arena::RakeConfig;
use crate::arena::agent::AgentConfig;
use crate::arena::cfr::BudgetConfig;
use std::path::{Path, PathBuf};

use super::config::RotationConfig;
use super::error::{Result, RotationError};
use super::runner::ArenaRotation;

/// Builder for the rotation-based benchmark. This is intentionally separate
/// from `comparison::ComparisonBuilder`: the legacy comparison remains a
/// single-hand exhaustive-permutation benchmark.
#[derive(Debug, Default)]
pub struct RotationBuilder {
    agents: Vec<(String, AgentConfig)>,
    num_rotations: Option<usize>,
    players_per_table: Option<usize>,
    big_blind: Option<Chips>,
    small_blind: Option<Chips>,
    min_stack_bb: Option<f32>,
    max_stack_bb: Option<f32>,
    ante: Option<Chips>,
    output_dir: Option<PathBuf>,
    seed: Option<u64>,
    rake: Option<RakeConfig>,
    rebuy_to_bb: Option<f32>,
    jobs: Option<usize>,
}

impl RotationBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn num_rotations(mut self, num_rotations: usize) -> Self {
        self.num_rotations = Some(num_rotations);
        self
    }

    pub fn players_per_table(mut self, players_per_table: usize) -> Self {
        self.players_per_table = Some(players_per_table);
        self
    }

    pub fn big_blind(mut self, big_blind: Chips) -> Self {
        self.big_blind = Some(big_blind);
        self
    }

    pub fn small_blind(mut self, small_blind: Chips) -> Self {
        self.small_blind = Some(small_blind);
        self
    }

    pub fn min_stack_bb(mut self, min_stack_bb: f32) -> Self {
        self.min_stack_bb = Some(min_stack_bb);
        self
    }

    pub fn max_stack_bb(mut self, max_stack_bb: f32) -> Self {
        self.max_stack_bb = Some(max_stack_bb);
        self
    }

    pub fn ante(mut self, ante: Chips) -> Self {
        self.ante = Some(ante);
        self
    }

    pub fn output_dir<P: AsRef<Path>>(mut self, output_dir: P) -> Self {
        self.output_dir = Some(output_dir.as_ref().to_path_buf());
        self
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    pub fn rake(mut self, rake: RakeConfig) -> Self {
        self.rake = Some(rake);
        self
    }

    pub fn rebuy_to_bb(mut self, rebuy_to_bb: f32) -> Self {
        self.rebuy_to_bb = Some(rebuy_to_bb);
        self
    }

    pub fn jobs(mut self, jobs: usize) -> Self {
        self.jobs = Some(jobs);
        self
    }

    pub fn add_agent(mut self, name: String, config: AgentConfig) -> Self {
        self.agents.push((name, config));
        self
    }

    pub fn add_agent_config(mut self, config: AgentConfig) -> Self {
        let name = get_agent_name(&config, &format!("Agent{}", self.agents.len()));
        self.agents.push((name, config));
        self
    }

    pub fn add_agents(mut self, agents: Vec<(String, AgentConfig)>) -> Self {
        self.agents.extend(agents);
        self
    }

    pub fn load_agents_from_dir<P: AsRef<Path>>(mut self, dir: P) -> Result<Self> {
        self.agents.extend(load_agents_from_dir(dir.as_ref())?);
        Ok(self)
    }

    /// Fill missing CFR budgets exactly like the legacy comparison builder.
    /// Explicit per-agent budgets remain untouched.
    pub fn fill_default_budget(mut self, default: &BudgetConfig) -> Self {
        for (_, cfg) in &mut self.agents {
            cfg.fill_default_budget(default);
        }
        self
    }

    pub fn build(self) -> Result<ArenaRotation> {
        if self.agents.is_empty() {
            return Err(RotationError::MissingConfig(
                "No agents configured. Use add_agent(), add_agent_config(), or load_agents_from_dir()"
                    .to_string(),
            ));
        }

        let config = RotationConfig {
            num_rotations: self.num_rotations.unwrap_or(1000),
            players_per_table: self.players_per_table.unwrap_or(3),
            big_blind: self.big_blind.unwrap_or(10),
            small_blind: self.small_blind.unwrap_or(5),
            min_stack_bb: self.min_stack_bb.unwrap_or(100.0),
            max_stack_bb: self.max_stack_bb.unwrap_or(100.0),
            ante: self.ante.unwrap_or(0),
            output_dir: self.output_dir,
            seed: self.seed,
            rake: self.rake.unwrap_or_default(),
            rebuy_to_bb: self.rebuy_to_bb.unwrap_or(100.0),
            jobs: self.jobs,
        };

        config.validate(self.agents.len())?;
        for (_, agent_config) in &self.agents {
            agent_config.validate()?;
        }

        Ok(ArenaRotation::new(config, self.agents))
    }
}

fn get_agent_name(config: &AgentConfig, fallback_name: &str) -> String {
    match config {
        AgentConfig::AllIn { name, .. }
        | AgentConfig::Calling { name, .. }
        | AgentConfig::Folding { name, .. }
        | AgentConfig::Random { name, .. }
        | AgentConfig::Equity { name, .. }
        | AgentConfig::CfrBasic { name, .. }
        | AgentConfig::CfrSimple { name, .. }
        | AgentConfig::CfrConfigurable { name, .. }
        | AgentConfig::CfrPreflopChart { name, .. } => {
            name.clone().unwrap_or_else(|| fallback_name.to_string())
        }
    }
}

fn load_agents_from_dir(dir: &Path) -> Result<Vec<(String, AgentConfig)>> {
    if !dir.exists() {
        return Err(RotationError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("Directory does not exist: {}", dir.display()),
        )));
    }
    if !dir.is_dir() {
        return Err(RotationError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("Path is not a directory: {}", dir.display()),
        )));
    }

    let mut agents = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.extension().and_then(|s| s.to_str()) != Some("json") {
            continue;
        }
        let fallback_name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();
        match load_agent_config(&path) {
            Ok(config) => {
                let agent_name = get_agent_name(&config, &fallback_name);
                agents.push((agent_name, config));
            }
            Err(e) => tracing::warn!("Skipping invalid config file {}: {}", path.display(), e),
        }
    }

    if agents.is_empty() {
        return Err(RotationError::NoAgentsFound(dir.display().to_string()));
    }
    agents.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(agents)
}

fn load_agent_config(path: &Path) -> Result<AgentConfig> {
    let contents = std::fs::read_to_string(path)?;
    let config: AgentConfig =
        serde_json::from_str(&contents).map_err(|source| RotationError::ParseConfig {
            path: path.display().to_string(),
            source,
        })?;
    config.validate()?;
    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_cash_benchmark_friendly() {
        let rotation = RotationBuilder::new()
            .add_agent_config(AgentConfig::Folding { name: None })
            .add_agent_config(AgentConfig::Calling { name: None })
            .add_agent_config(AgentConfig::AllIn { name: None })
            .build()
            .unwrap();
        assert_eq!(rotation.config().num_rotations, 1000);
        assert_eq!(rotation.config().rebuy_to_bb, 100.0);
    }
}
