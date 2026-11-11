use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::arena::Chips;
use crate::arena::comparison::AgentStats;

use super::config::RotationConfig;
use super::error::Result;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RebuyStats {
    pub count: usize,
    pub chips: Chips,
}

#[derive(Debug, Clone)]
pub struct RotationComparisonResult {
    agent_names: Vec<String>,
    agent_stats: HashMap<String, AgentStats>,
    rebuy_stats: HashMap<String, RebuyStats>,
    config: RotationConfig,
    unique_seatings: usize,
    rotations_run: usize,
    hands_run: usize,
}

impl RotationComparisonResult {
    pub(crate) fn new(
        agent_names: Vec<String>,
        agent_stats: HashMap<String, AgentStats>,
        rebuy_stats: HashMap<String, RebuyStats>,
        config: RotationConfig,
        unique_seatings: usize,
        rotations_run: usize,
        hands_run: usize,
    ) -> Self {
        Self {
            agent_names,
            agent_stats,
            rebuy_stats,
            config,
            unique_seatings,
            rotations_run,
            hands_run,
        }
    }

    pub fn agent_names(&self) -> &[String] {
        &self.agent_names
    }

    pub fn get_agent_stats(&self, name: &str) -> Option<&AgentStats> {
        self.agent_stats.get(name)
    }

    pub fn all_stats(&self) -> &HashMap<String, AgentStats> {
        &self.agent_stats
    }

    pub fn rebuy_stats(&self) -> &HashMap<String, RebuyStats> {
        &self.rebuy_stats
    }

    pub fn config(&self) -> &RotationConfig {
        &self.config
    }

    pub fn unique_seatings(&self) -> usize {
        self.unique_seatings
    }

    pub fn rotations_run(&self) -> usize {
        self.rotations_run
    }

    pub fn hands_run(&self) -> usize {
        self.hands_run
    }

    pub fn get_rankings(&self) -> Vec<(&String, &AgentStats)> {
        let mut rankings: Vec<_> = self.agent_stats.iter().collect();
        rankings.sort_by(|a, b| {
            b.1.profit_per_game
                .partial_cmp(&a.1.profit_per_game)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        rankings
    }

    pub fn to_markdown(&self) -> String {
        let mut out = String::new();
        out.push_str("# Rotation Comparison Results\n\n");
        out.push_str("## Configuration\n\n");
        out.push_str(&format!(
            "- **Agents Tested**: {}\n",
            self.agent_names.len()
        ));
        out.push_str(&format!(
            "- **Players per Table**: {}\n",
            self.config.players_per_table
        ));
        out.push_str(&format!("- **Table Rotations**: {}\n", self.rotations_run));
        out.push_str(&format!("- **Hands Simulated**: {}\n", self.hands_run));
        out.push_str(&format!(
            "- **Unique Circular Seatings**: {}\n",
            self.unique_seatings
        ));
        if self.unique_seatings > 0 {
            out.push_str(&format!(
                "- **Complete Seating Cycles**: {}\n",
                self.rotations_run / self.unique_seatings
            ));
            out.push_str(&format!(
                "- **Partial Seating Cycle**: {} / {}\n",
                self.rotations_run % self.unique_seatings,
                self.unique_seatings
            ));
        }
        out.push_str(&format!("- **Workers**: {}\n", self.config.resolved_jobs()));
        out.push_str(&format!(
            "- **Rebuy on Bust**: {:.1} BB\n",
            self.config.rebuy_to_bb
        ));
        out.push_str(&format!("- **Big Blind**: {}\n", self.config.big_blind));
        out.push_str(&format!("- **Small Blind**: {}\n", self.config.small_blind));
        if let Some(seed) = self.config.seed {
            out.push_str(&format!("- **Random Seed**: {}\n", seed));
        }
        out.push('\n');

        out.push_str("## Rankings\n\n");
        out.push_str("| Agent | bb/100 | Total bb | Hands | Rebuys | Rebuy bb |\n");
        out.push_str("|---|---:|---:|---:|---:|---:|\n");
        for (name, stats) in self.get_rankings() {
            let rebuy = self.rebuy_stats.get(name).cloned().unwrap_or_default();
            out.push_str(&format!(
                "| {} | {:+.2} | {:+.2} | {} | {} | {:.2} |\n",
                name,
                stats.profit_per_100_hands / self.config.big_blind as f32,
                stats.total_profit as f32 / self.config.big_blind as f32,
                stats.total_games,
                rebuy.count,
                rebuy.chips as f32 / self.config.big_blind as f32,
            ));
        }
        out.push('\n');

        out.push_str("## Position Performance\n\n");
        for (name, stats) in self.get_rankings() {
            out.push_str(&format!("### {}\n\n", name));
            out.push_str("| Position | Profit/Hand | Hands |\n");
            out.push_str("|---|---:|---:|\n");
            for p in &stats.position_stats {
                out.push_str(&format!(
                    "| {} | {:+.2} bb | {} |\n",
                    position_name(self.config.players_per_table, p.seat_index),
                    p.profit_per_game / self.config.big_blind as f32,
                    p.games_played,
                ));
            }
            out.push('\n');
        }

        out
    }

    pub fn to_json(&self) -> Result<String> {
        #[derive(Serialize)]
        struct JsonResult<'a> {
            agent_stats: &'a HashMap<String, AgentStats>,
            rebuy_stats: &'a HashMap<String, RebuyStats>,
            rotations_run: usize,
            hands_run: usize,
            unique_seatings: usize,
        }

        Ok(serde_json::to_string_pretty(&JsonResult {
            agent_stats: &self.agent_stats,
            rebuy_stats: &self.rebuy_stats,
            rotations_run: self.rotations_run,
            hands_run: self.hands_run,
            unique_seatings: self.unique_seatings,
        })?)
    }

    pub fn save_to_dir(&self, output_dir: &Path) -> Result<()> {
        std::fs::create_dir_all(output_dir)?;
        std::fs::write(output_dir.join("results.json"), self.to_json()?)?;
        std::fs::write(output_dir.join("results.md"), self.to_markdown())?;
        Ok(())
    }
}

pub fn position_name(players: usize, position: usize) -> String {
    let label = match (players, position) {
        (2, 0) => "BTN/SB",
        (2, 1) => "BB",
        (_, 0) => "BTN",
        (_, 1) => "SB",
        (_, 2) => "BB",
        (3, _) => return format!("P{position}"),
        (4, 3) => "CO",
        (5, 3) => "UTG",
        (5, 4) => "CO",
        (6, 3) => "UTG",
        (6, 4) => "HJ",
        (6, 5) => "CO",
        _ => return format!("P{position}"),
    };
    label.to_string()
}
