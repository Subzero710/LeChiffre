//! Shared agent catalog. A profile may be selected more than once, without copies.
use crate::table_arena::PreparedTable;
use clap::Args;
use serde_json::Value;
use std::{
    fs,
    path::{Path, PathBuf},
};

#[derive(Args)]
pub struct AgentSelection {
    /// Existing directory of JSON profiles (optional when --agents is used).
    #[arg(
        value_name = "AGENTS_DIR",
        required_unless_present = "agents",
        conflicts_with = "agents"
    )]
    pub agents_dir: Option<PathBuf>,
    /// Select profiles directly from agents/. Repeating a path selects another player.
    #[arg(long,num_args=2..,value_delimiter=',',conflicts_with="agents_dir")]
    pub agents: Vec<PathBuf>,
}
impl AgentSelection {
    pub fn load(&self) -> Result<Vec<Value>, String> {
        if let Some(dir) = &self.agents_dir {
            load_profile_pool(dir)
        } else {
            load_profile_files(&self.agents)
        }
    }
}
pub fn default_reference() -> PathBuf {
    PathBuf::from("runs/noregrets")
}
pub fn load_profile_files(paths: &[PathBuf]) -> Result<Vec<Value>, String> {
    if paths.len() < 2 {
        return Err("select at least two agent profiles".into());
    }
    paths
        .iter()
        .map(|path| {
            let bytes = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
            let bytes = bytes.strip_prefix(&[0xef, 0xbb, 0xbf]).unwrap_or(&bytes);
            let mut value: Value =
                serde_json::from_slice(bytes).map_err(|e| format!("{}: {e}", path.display()))?;
            if !value.is_object() {
                return Err(format!("{}: expected an agent object", path.display()));
            }
            if value.get("name").is_none() {
                value["name"] = Value::String(
                    path.file_stem()
                        .ok_or("profile file name")?
                        .to_string_lossy()
                        .into_owned(),
                );
            }
            Ok(value)
        })
        .collect()
}
pub fn load_profile_pool(path: &Path) -> Result<Vec<Value>, String> {
    let mut files = fs::read_dir(path)
        .map_err(|e| format!("{}: {e}", path.display()))?
        .map(|x| x.map(|v| v.path()))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())?;
    files.retain(|p| p.is_file() && p.extension().is_some_and(|x| x == "json"));
    files.sort();
    load_profile_files(&files)
}
pub fn load_directory(path: &Path) -> Result<Vec<Value>, String> {
    let v = load_profile_pool(path)?;
    if v.len() > 6 {
        return Err(
            "a fixed table has at most six players; use --agents to select profiles".into(),
        );
    }
    Ok(v)
}
pub fn prepare_directory(
    path: &Path,
    stack: Option<u32>,
    reference: Option<&Path>,
) -> Result<PreparedTable, String> {
    PreparedTable::new(load_directory(path)?, stack, reference)
}
