//! Immutable exact NLHE legal masks and chip bounds, sealed into the dataset directory.
//!
//! This sidecar is generated from the same rs-poker replay used by the original
//! importer. It contains no inferred legal actions and no player identifiers.
//! All retained decisions are checked against the Parquet observations before
//! publication and on every subsequent load. No training path reads raw sources.
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::Path;

use serde_json::{Value, json};

use super::BehaviourParquetDataset;
use super::training_data::decode_decision;
use crate::behavioural::checkpoint::{exact_keys, unsigned};
use crate::behavioural::{HandId, LegalActionSpace, Position, WagerBounds};

pub const EXACT_LEGALITY_FILE_NAME: &str = "exact_legality.v1.json";
const FORMAT: &str = "lechiffre.parquet-exact-legality.v1";

#[derive(Debug)]
pub struct ExactLegalityError(String);

impl ExactLegalityError {
    fn invalid(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}
impl fmt::Display for ExactLegalityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for ExactLegalityError {}

#[derive(Debug)]
pub(super) struct ExactLegalityIndex {
    pub(super) by_decision: HashMap<(HandId, u32), LegalActionSpace>,
    pub(super) fingerprint: String,
}

fn fingerprint(bytes: &[u8]) -> String {
    let mut hash = 0x6c62_272e_07bb_0142_62b8_2175_6295_c58d_u128;
    const PRIME: u128 = 0x0000_0000_0100_0000_0000_0000_0000_013b;
    for byte in bytes {
        hash = (hash ^ u128::from(*byte)).wrapping_mul(PRIME);
    }
    format!("{hash:032x}")
}

fn encode_bounds(bounds: Option<WagerBounds>) -> Value {
    bounds.map_or(Value::Null, |b| {
        json!({
            "min_added_chips": b.min_added_chips,
            "max_added_chips": b.max_added_chips,
            "chip_unit": b.chip_unit,
            "big_blind_chips": b.big_blind_chips
        })
    })
}

fn encode(hand: &HandId, sequence: u32, legal: LegalActionSpace) -> Value {
    json!({
        "hand_id": hand.as_str(),
        "sequence_index": sequence,
        "fold": legal.fold,
        "check": legal.check,
        "call": legal.call,
        "bet": encode_bounds(legal.bet),
        "raise": encode_bounds(legal.raise)
    })
}

fn decode_bounds(v: &Value) -> Result<Option<WagerBounds>, ExactLegalityError> {
    if v.is_null() {
        return Ok(None);
    }
    exact_keys(
        v,
        &[
            "min_added_chips",
            "max_added_chips",
            "chip_unit",
            "big_blind_chips",
        ],
    )
    .map_err(ExactLegalityError::invalid)?;
    let integer = |key: &str| {
        v[key]
            .as_i64()
            .ok_or_else(|| ExactLegalityError::invalid(format!("invalid exact chip bound {key}")))
    };
    let bounds = WagerBounds {
        min_added_chips: integer("min_added_chips")?,
        max_added_chips: integer("max_added_chips")?,
        chip_unit: integer("chip_unit")?,
        big_blind_chips: integer("big_blind_chips")?,
    };
    if bounds.min_added_chips <= 0
        || bounds.max_added_chips < bounds.min_added_chips
        || bounds.chip_unit <= 0
        || bounds.big_blind_chips <= 0
    {
        return Err(ExactLegalityError::invalid("invalid exact wager limits"));
    }
    Ok(Some(bounds))
}

fn decode(v: &Value, hand: &HandId, sequence: u32) -> Result<LegalActionSpace, ExactLegalityError> {
    exact_keys(
        v,
        &[
            "hand_id",
            "sequence_index",
            "fold",
            "check",
            "call",
            "bet",
            "raise",
        ],
    )
    .map_err(ExactLegalityError::invalid)?;
    if v["hand_id"].as_str() != Some(hand.as_str())
        || unsigned(&v["sequence_index"]).map_err(ExactLegalityError::invalid)?
            != u64::from(sequence)
    {
        return Err(ExactLegalityError::invalid(format!(
            "exact legality row does not match {hand}:{sequence}"
        )));
    }
    let boolean = |key: &str| {
        v[key]
            .as_bool()
            .ok_or_else(|| ExactLegalityError::invalid(format!("invalid exact legal mask {key}")))
    };
    Ok(LegalActionSpace {
        fold: boolean("fold")?,
        check: boolean("check")?,
        call: boolean("call")?,
        bet: decode_bounds(&v["bet"])?,
        raise: decode_bounds(&v["raise"])?,
    })
}

fn subject_positions(
    dataset: &BehaviourParquetDataset,
) -> Result<Vec<Position>, ExactLegalityError> {
    let mut positions = vec![None; dataset.hands.len()];
    for memberships in dataset.subjects.values() {
        for m in memberships {
            positions[m.hand_slot].get_or_insert(m.subject_position);
        }
    }
    positions
        .into_iter()
        .map(|value| {
            value.ok_or_else(|| {
                ExactLegalityError::invalid("retained hand has no subject membership")
            })
        })
        .collect()
}
type ValidatedLegalityRows = (Vec<Value>, HashMap<(HandId, u32), LegalActionSpace>);
fn validated_rows(
    dataset: &BehaviourParquetDataset,
    source: Option<&HashMap<(HandId, u32), LegalActionSpace>>,
    persisted: Option<&[Value]>,
) -> Result<ValidatedLegalityRows, ExactLegalityError> {
    let positions = subject_positions(dataset)?;
    let mut rows = Vec::with_capacity(dataset.event_count);
    let mut by_decision = HashMap::with_capacity(dataset.event_count);
    for (slot, hand) in dataset.hands.iter().enumerate() {
        let encoded = hand.materialize(positions[slot]);
        for (row, &sequence) in hand.sequence_indexes.iter().enumerate() {
            let key = (hand.hand_id.clone(), sequence);
            let legal = if let Some(source) = source {
                *source.get(&key).ok_or_else(|| {
                    ExactLegalityError::invalid(format!(
                        "missing exact replay legal actions for {}:{}",
                        key.0, key.1
                    ))
                })?
            } else {
                let value = persisted
                    .and_then(|entries| entries.get(rows.len()))
                    .ok_or_else(|| {
                        ExactLegalityError::invalid("exact legality rows are incomplete")
                    })?;
                decode(value, &key.0, sequence)?
            };
            decode_decision(&encoded, row)
                .map_err(|err| ExactLegalityError::invalid(err.to_string()))?
                .validate_legality(legal)
                .map_err(|err| {
                    ExactLegalityError::invalid(format!(
                        "exact legality incompatible with retained observation {}:{}: {err}",
                        key.0, sequence
                    ))
                })?;
            if by_decision.insert(key.clone(), legal).is_some() {
                return Err(ExactLegalityError::invalid(
                    "duplicated retained legal decision",
                ));
            }
            rows.push(encode(&key.0, sequence, legal));
        }
    }
    if rows.len() != dataset.event_count
        || persisted.is_some_and(|entries| entries.len() != rows.len())
    {
        return Err(ExactLegalityError::invalid(
            "exact legality decision count mismatch",
        ));
    }
    if let Some(source) = source {
        // Extra source entries for rejected or timing-dropped hands are expected;
        // extra decisions belonging to a *retained* hand are not.
        let retained = dataset
            .hands
            .iter()
            .map(|h| h.hand_id.clone())
            .collect::<HashSet<_>>();
        for key in source.keys() {
            if retained.contains(&key.0) && !by_decision.contains_key(key) {
                return Err(ExactLegalityError::invalid(format!(
                    "source replay has an extra decision for retained hand {}:{}",
                    key.0, key.1
                )));
            }
        }
    }
    Ok((rows, by_decision))
}

impl BehaviourParquetDataset {
    /// One-time source replay materialization. Publishes only after validating
    /// *every* retained decision. Never replaces an existing sidecar.
    pub fn attach_exact_legality(
        directory: impl AsRef<Path>,
        replay: &HashMap<(HandId, u32), LegalActionSpace>,
    ) -> Result<usize, ExactLegalityError> {
        let directory = directory.as_ref();
        let path = directory.join(EXACT_LEGALITY_FILE_NAME);
        if path.exists() {
            return Err(ExactLegalityError::invalid(format!(
                "exact legality already exists: {}",
                path.display()
            )));
        }
        let dataset =
            Self::open(directory).map_err(|err| ExactLegalityError::invalid(err.to_string()))?;
        if dataset.exact_legality.is_some() {
            return Err(ExactLegalityError::invalid(
                "dataset already contains exact legality",
            ));
        }
        let (rows, _) = validated_rows(&dataset, Some(replay), None)?;
        let bytes = serde_json::to_vec(&rows)
            .map_err(|err| ExactLegalityError::invalid(err.to_string()))?;
        let record = json!({
            "format":FORMAT,
            "dataset_fingerprint":dataset.dataset_fingerprint(),
            "legality_fingerprint":fingerprint(&bytes),
            "decisions":rows
        });
        let temporary = directory.join(format!(
            ".{EXACT_LEGALITY_FILE_NAME}.{}.inprogress",
            std::process::id()
        ));
        let result = (|| -> Result<(), ExactLegalityError> {
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temporary)
                .map_err(|err| ExactLegalityError::invalid(err.to_string()))?;
            serde_json::to_writer(&mut file, &record)
                .map_err(|err| ExactLegalityError::invalid(err.to_string()))?;
            file.write_all(b"\n")
                .map_err(|err| ExactLegalityError::invalid(err.to_string()))?;
            file.sync_all()
                .map_err(|err| ExactLegalityError::invalid(err.to_string()))?;
            fs::hard_link(&temporary, &path)
                .map_err(|err| ExactLegalityError::invalid(err.to_string()))?;
            Ok(())
        })();
        let _ = fs::remove_file(&temporary);
        result?;
        Ok(dataset.event_count)
    }

    /// Loaded once, validated in full, and used exclusively by every A/G job.
    pub fn exact_legal_actions(
        &self,
        hand: &HandId,
        sequence: u32,
    ) -> Result<LegalActionSpace, ExactLegalityError> {
        self.exact_legality.as_ref().ok_or_else(|| ExactLegalityError::invalid(
            "dataset lacks exact_legality.v1.json: attach the original replay evidence once before training"))?
            .by_decision.get(&(hand.clone(), sequence)).copied().ok_or_else(||
                ExactLegalityError::invalid(format!("missing retained exact legal decision {hand}:{sequence}")))
    }

    pub fn legality_fingerprint(&self) -> Result<&str, ExactLegalityError> {
        self.exact_legality
            .as_ref()
            .map(|idx| idx.fingerprint.as_str())
            .ok_or_else(|| {
                ExactLegalityError::invalid(
                    "dataset lacks exact_legality.v1.json: run --attach-behaviour-legality",
                )
            })
    }
}

pub(super) fn read_index(
    directory: &Path,
    dataset: &BehaviourParquetDataset,
) -> Result<Option<ExactLegalityIndex>, ExactLegalityError> {
    let path = directory.join(EXACT_LEGALITY_FILE_NAME);
    let file = match File::open(&path) {
        Ok(file) => file,
        Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(err) => {
            return Err(ExactLegalityError::invalid(format!(
                "{}: {err}",
                path.display()
            )));
        }
    };
    let value: Value = serde_json::from_reader(file)
        .map_err(|err| ExactLegalityError::invalid(err.to_string()))?;
    exact_keys(
        &value,
        &[
            "format",
            "dataset_fingerprint",
            "legality_fingerprint",
            "decisions",
        ],
    )
    .map_err(ExactLegalityError::invalid)?;
    if value["format"].as_str() != Some(FORMAT)
        || value["dataset_fingerprint"].as_str() != Some(dataset.dataset_fingerprint().as_str())
    {
        return Err(ExactLegalityError::invalid(
            "exact legality format or dataset fingerprint mismatch",
        ));
    }
    let entries = value["decisions"]
        .as_array()
        .ok_or_else(|| ExactLegalityError::invalid("exact legality decisions must be an array"))?;
    let bytes =
        serde_json::to_vec(entries).map_err(|err| ExactLegalityError::invalid(err.to_string()))?;
    let digest = fingerprint(&bytes);
    if value["legality_fingerprint"].as_str() != Some(digest.as_str()) {
        return Err(ExactLegalityError::invalid(
            "exact legality checksum mismatch",
        ));
    }
    let (_, by_decision) = validated_rows(dataset, None, Some(entries))?;
    Ok(Some(ExactLegalityIndex {
        by_decision,
        fingerprint: digest,
    }))
}
