//! JSON-per-agent tables, compatible with the stock rs_poker config files.
use crate::table_arena::PreparedTable;
use serde_json::Value;
use std::{
    fs,
    path::{Path, PathBuf},
};

pub fn load_directory(path: &Path) -> Result<Vec<Value>, String> {
    let mut files = fs::read_dir(path)
        .map_err(|e| format!("{}: {e}", path.display()))?
        .map(|e| e.map(|e| e.path()))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())?;
    files.retain(|p| p.is_file() && p.extension().is_some_and(|e| e == "json"));
    files.sort();
    if !(2..=6).contains(&files.len()) {
        return Err(format!(
            "select 2 to 6 JSON files in {}; found {}",
            path.display(),
            files.len()
        ));
    }
    files
        .into_iter()
        .map(|file| {
            let mut value: Value =
                serde_json::from_slice(&fs::read(&file).map_err(|e| e.to_string())?)
                    .map_err(|e| format!("{}: {e}", file.display()))?;
            if !value.is_object() {
                return Err(format!("{}: expected an agent JSON object", file.display()));
            }
            if value.get("name").is_none() {
                value["name"] =
                    Value::String(file.file_stem().unwrap().to_string_lossy().into_owned());
            }
            Ok(value)
        })
        .collect()
}

/// The supplied 2-player and 6-player bootstrap models remain separate.
pub fn built_in_reference(players: usize) -> Option<PathBuf> {
    let name = match players {
        2 => "noregrets-hu10bb-agent-10000-seed42",
        6 => "noregrets-6max-10bb-agent-10000-seed42",
        _ => return None,
    };
    Some(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../references")
            .join(name),
    )
}

pub fn prepare_directory(
    path: &Path,
    stack: Option<u32>,
    reference: Option<&Path>,
) -> Result<PreparedTable, String> {
    let values = load_directory(path)?;
    let built_in = built_in_reference(values.len());
    let reference = reference.or(built_in.as_deref());
    PreparedTable::new(values, stack, reference)
}
