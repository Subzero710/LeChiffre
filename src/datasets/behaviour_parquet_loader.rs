use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use std::fs::File;
use std::io;
use std::path::{Path, PathBuf};

use arrow_array::{
    Array, BooleanArray, Float32Array, RecordBatch, StringArray, UInt8Array, UInt32Array,
    UInt64Array,
};
use arrow_schema::{ArrowError, SchemaRef};
use parquet::arrow::arrow_reader::{ParquetRecordBatchReader, ParquetRecordBatchReaderBuilder};
use parquet::errors::ParquetError;
use parquet::file::metadata::ParquetMetaData;

use crate::behavioural::codebook::{
    ACTION_CODEBOOK, POSITION_CODEBOOK, RANK_CODEBOOK, STREET_CODEBOOK, SUIT_CODEBOOK,
    action_from_id, position_from_id, position_id, rank_from_id, street_from_id, suit_from_id,
};
use crate::behavioural::{
    CategoricalFeature, ContinuousFeature, EVENT_CATEGORICAL_WIDTH, EVENT_CONTINUOUS_WIDTH,
    FeaturizedHand, FeaturizedHandData, HandId, MAX_BOARD_CARDS, Position,
};

use super::behaviour_parquet::{
    BEHAVIOUR_PARQUET_SCHEMA_VERSION, BehaviourParquetDatasetSummary, EVENTS_FILE_NAME,
    SUBJECTS_FILE_NAME, SUCCESS_FILE_NAME, events_schema, subjects_schema,
};

const READ_BATCH_ROWS: usize = 8_192;
const HAND_INDEX_SEMANTICS: &str =
    "monotonic_input_order_metadata_not_model_feature; gaps_correspond_to_dropped_hands";
const SUBJECT_ID_SEMANTICS: &str = "pseudonymous_grouping_metadata_not_model_feature";

/// Fully validated in-memory view of a committed behavioural Parquet dataset.
///
/// Event rows remain unique in memory. Subjects only store references to those
/// hands; a subject-specific [`LoadedSubjectHistory`] is materialized on demand.
#[derive(Debug)]
pub struct BehaviourParquetDataset {
    summary: BehaviourParquetDatasetSummary,
    hands: Vec<PersistedHand>,
    subjects: BTreeMap<u64, Vec<SubjectMembership>>,
    event_count: usize,
}

impl BehaviourParquetDataset {
    pub fn open(root: impl AsRef<Path>) -> Result<Self, BehaviourParquetLoadError> {
        let root = root.as_ref();
        let success_path = root.join(SUCCESS_FILE_NAME);
        if !success_path.is_file() {
            return Err(BehaviourParquetLoadError::MissingSuccessMarker { path: success_path });
        }

        let events_path = root.join(EVENTS_FILE_NAME);
        let subjects_path = root.join(SUBJECTS_FILE_NAME);
        let (summary, hands, event_count) = read_events(&events_path)?;
        let subjects = read_subjects(&subjects_path, summary, &hands)?;

        Ok(Self {
            summary,
            hands,
            subjects,
            event_count,
        })
    }

    pub fn summary(&self) -> BehaviourParquetDatasetSummary {
        self.summary
    }

    pub fn hand_count(&self) -> usize {
        self.hands.len()
    }

    pub fn event_count(&self) -> usize {
        self.event_count
    }

    pub fn subject_count(&self) -> usize {
        self.subjects.len()
    }

    pub fn subject_ids(&self) -> impl Iterator<Item = u64> + '_ {
        self.subjects.keys().copied()
    }

    pub fn subject_hand_count(&self, subject_id: u64) -> Option<usize> {
        self.subjects.get(&subject_id).map(Vec::len)
    }

    /// Lazily materializes retained hands for one subject, one hand at a time.
    ///
    /// This avoids cloning an entire multi-hand history when the training layer
    /// can consume hands incrementally.
    pub fn subject_hands(
        &self,
        subject_id: u64,
    ) -> Option<impl ExactSizeIterator<Item = FeaturizedHand> + '_> {
        let memberships = self.subjects.get(&subject_id)?;
        Some(memberships.iter().map(move |membership| {
            self.hands[membership.hand_slot].materialize(membership.subject_position)
        }))
    }

    /// Builds the exact retained model-ready history for one pseudonymous subject.
    ///
    /// `subject_id` is used only to select membership rows. It is never copied
    /// into the returned features. `IsSubject` is derived from the persisted
    /// actor position and the subject's position in each hand.
    ///
    /// The persisted format stores dropped-hand counts only at dataset scope,
    /// not per subject. This type therefore exposes only retained hands instead
    /// of fabricating a per-subject dropped-hand count.
    pub fn subject_history(&self, subject_id: u64) -> Option<LoadedSubjectHistory> {
        let hands = self.subject_hands(subject_id)?.collect();
        Some(LoadedSubjectHistory { subject_id, hands })
    }
}

/// Retained model-ready hands for one pseudonymous subject.
///
/// This deliberately does not expose a per-subject missing-timing drop count:
/// the normalized Parquet format does not persist enough information to derive
/// that value exactly.
#[derive(Debug, Clone, PartialEq)]
pub struct LoadedSubjectHistory {
    subject_id: u64,
    hands: Vec<FeaturizedHand>,
}

impl LoadedSubjectHistory {
    pub fn subject_id(&self) -> u64 {
        self.subject_id
    }

    pub fn hands(&self) -> &[FeaturizedHand] {
        &self.hands
    }

    pub fn len(&self) -> usize {
        self.hands.len()
    }

    pub fn is_empty(&self) -> bool {
        self.hands.is_empty()
    }

    pub fn total_events(&self) -> usize {
        self.hands.iter().map(FeaturizedHand::sequence_len).sum()
    }
}

#[derive(Debug)]
struct PersistedHand {
    hand_id: HandId,
    hand_index: u64,
    players_at_table: u8,
    sequence_indexes: Vec<u32>,
    categorical_without_subject: Vec<[i64; 4]>,
    continuous: Vec<[f32; EVENT_CONTINUOUS_WIDTH]>,
    continuous_present: Vec<[u8; EVENT_CONTINUOUS_WIDTH]>,
    decision_time_ms: Vec<u32>,
    board_rank: Vec<[i64; MAX_BOARD_CARDS]>,
    board_suit: Vec<[i64; MAX_BOARD_CARDS]>,
    board_present: Vec<[u8; MAX_BOARD_CARDS]>,
}

impl PersistedHand {
    fn materialize(&self, subject_position: Position) -> FeaturizedHand {
        let subject_position_id = i64::from(position_id(subject_position));
        let categorical = self
            .categorical_without_subject
            .iter()
            .map(|base| {
                let mut row = [0_i64; EVENT_CATEGORICAL_WIDTH];
                row[CategoricalFeature::Street as usize] = base[0];
                row[CategoricalFeature::ActorPosition as usize] = base[1];
                row[CategoricalFeature::Action as usize] = base[2];
                row[CategoricalFeature::IsSubject as usize] =
                    if base[1] == subject_position_id { 1 } else { 0 };
                row[CategoricalFeature::IsAllIn as usize] = base[3];
                row
            })
            .collect();

        FeaturizedHand::from_data(FeaturizedHandData {
            hand_id: self.hand_id.clone(),
            subject_position,
            sequence_indexes: self.sequence_indexes.clone(),
            categorical,
            continuous: self.continuous.clone(),
            continuous_present: self.continuous_present.clone(),
            decision_time_ms: self.decision_time_ms.clone(),
            board_rank: self.board_rank.clone(),
            board_suit: self.board_suit.clone(),
            board_present: self.board_present.clone(),
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct SubjectMembership {
    hand_slot: usize,
    subject_position: Position,
}

#[derive(Debug)]
struct HandBuilder {
    hand_id: HandId,
    hand_index: u64,
    players_at_table: u8,
    sequence_indexes: Vec<u32>,
    categorical_without_subject: Vec<[i64; 4]>,
    continuous: Vec<[f32; EVENT_CONTINUOUS_WIDTH]>,
    continuous_present: Vec<[u8; EVENT_CONTINUOUS_WIDTH]>,
    decision_time_ms: Vec<u32>,
    board_rank: Vec<[i64; MAX_BOARD_CARDS]>,
    board_suit: Vec<[i64; MAX_BOARD_CARDS]>,
    board_present: Vec<[u8; MAX_BOARD_CARDS]>,
}

impl HandBuilder {
    fn new(hand_id: HandId, hand_index: u64, players_at_table: u8) -> Self {
        Self {
            hand_id,
            hand_index,
            players_at_table,
            sequence_indexes: Vec::new(),
            categorical_without_subject: Vec::new(),
            continuous: Vec::new(),
            continuous_present: Vec::new(),
            decision_time_ms: Vec::new(),
            board_rank: Vec::new(),
            board_suit: Vec::new(),
            board_present: Vec::new(),
        }
    }

    fn push(&mut self, row: EventRow) -> Result<(), BehaviourParquetLoadError> {
        if row.hand_id != self.hand_id {
            return Err(BehaviourParquetLoadError::HandIdentityChangedWithinIndex {
                hand_index: self.hand_index,
                expected: self.hand_id.clone(),
                observed: row.hand_id,
            });
        }
        if row.players_at_table != self.players_at_table {
            return Err(BehaviourParquetLoadError::PlayerCountChangedWithinHand {
                hand_id: self.hand_id.clone(),
                expected: self.players_at_table,
                observed: row.players_at_table,
            });
        }
        let expected_sequence_index = u32::try_from(self.sequence_indexes.len()).map_err(|_| {
            BehaviourParquetLoadError::SequenceLengthOverflow {
                hand_id: self.hand_id.clone(),
            }
        })?;
        if row.sequence_index != expected_sequence_index {
            return Err(BehaviourParquetLoadError::NonContiguousSequence {
                hand_id: self.hand_id.clone(),
                expected: expected_sequence_index,
                observed: row.sequence_index,
            });
        }

        self.sequence_indexes.push(row.sequence_index);
        self.categorical_without_subject
            .push(row.categorical_without_subject);
        self.continuous.push(row.continuous);
        self.continuous_present.push(row.continuous_present);
        self.decision_time_ms.push(row.decision_time_ms);
        self.board_rank.push(row.board_rank);
        self.board_suit.push(row.board_suit);
        self.board_present.push(row.board_present);
        Ok(())
    }

    fn finish(self) -> PersistedHand {
        PersistedHand {
            hand_id: self.hand_id,
            hand_index: self.hand_index,
            players_at_table: self.players_at_table,
            sequence_indexes: self.sequence_indexes,
            categorical_without_subject: self.categorical_without_subject,
            continuous: self.continuous,
            continuous_present: self.continuous_present,
            decision_time_ms: self.decision_time_ms,
            board_rank: self.board_rank,
            board_suit: self.board_suit,
            board_present: self.board_present,
        }
    }
}

struct EventRow {
    hand_id: HandId,
    sequence_index: u32,
    players_at_table: u8,
    categorical_without_subject: [i64; 4],
    continuous: [f32; EVENT_CONTINUOUS_WIDTH],
    continuous_present: [u8; EVENT_CONTINUOUS_WIDTH],
    decision_time_ms: u32,
    board_rank: [i64; MAX_BOARD_CARDS],
    board_suit: [i64; MAX_BOARD_CARDS],
    board_present: [u8; MAX_BOARD_CARDS],
}

fn read_events(
    path: &Path,
) -> Result<(BehaviourParquetDatasetSummary, Vec<PersistedHand>, usize), BehaviourParquetLoadError>
{
    let (mut reader, summary) = open_table_reader(path, "events", events_schema())?;
    let mut hands = Vec::new();
    let mut current: Option<HandBuilder> = None;
    let mut seen_hand_ids = HashSet::new();
    let mut last_started_hand_index: Option<u64> = None;
    let mut event_count = 0usize;

    for batch in &mut reader {
        let batch = batch?;
        let columns = EventColumns::new(&batch)?;
        for row_index in 0..batch.num_rows() {
            let hand_index = columns.hand_index.value(row_index);
            if hand_index >= summary.hands_seen {
                return Err(BehaviourParquetLoadError::HandIndexOutsideSeenRange {
                    hand_index,
                    hands_seen: summary.hands_seen,
                });
            }

            let hand_id = HandId::new(columns.hand_id.value(row_index));
            let starts_new_hand = current
                .as_ref()
                .is_none_or(|builder| builder.hand_index != hand_index);
            if starts_new_hand {
                if let Some(previous) = current.take() {
                    hands.push(previous.finish());
                }
                if last_started_hand_index.is_some_and(|previous| hand_index <= previous) {
                    return Err(BehaviourParquetLoadError::NonMonotonicHandIndex {
                        previous: last_started_hand_index,
                        observed: hand_index,
                    });
                }
                if !seen_hand_ids.insert(hand_id.clone()) {
                    return Err(BehaviourParquetLoadError::DuplicateHandId { hand_id });
                }
                last_started_hand_index = Some(hand_index);
                current = Some(HandBuilder::new(
                    hand_id.clone(),
                    hand_index,
                    columns.players_at_table.value(row_index),
                ));
            }

            let event = columns.event_row(row_index, hand_id)?;
            current
                .as_mut()
                .ok_or(BehaviourParquetLoadError::InternalMissingCurrentHand)?
                .push(event)?;
            event_count = event_count
                .checked_add(1)
                .ok_or(BehaviourParquetLoadError::CountOverflow { field: "events" })?;
        }
    }
    if let Some(previous) = current {
        hands.push(previous.finish());
    }

    validate_loaded_counts(summary, hands.len(), event_count)?;
    Ok((summary, hands, event_count))
}

fn read_subjects(
    path: &Path,
    expected_summary: BehaviourParquetDatasetSummary,
    hands: &[PersistedHand],
) -> Result<BTreeMap<u64, Vec<SubjectMembership>>, BehaviourParquetLoadError> {
    let (mut reader, summary) = open_table_reader(path, "subjects", subjects_schema())?;
    if summary != expected_summary {
        return Err(BehaviourParquetLoadError::CrossTableSummaryMismatch {
            events: expected_summary,
            subjects: summary,
        });
    }

    let hand_by_index: HashMap<u64, usize> = hands
        .iter()
        .enumerate()
        .map(|(slot, hand)| (hand.hand_index, slot))
        .collect();
    let mut memberships_by_hand = vec![Vec::<(u64, Position)>::new(); hands.len()];
    let mut subjects = BTreeMap::<u64, Vec<SubjectMembership>>::new();
    let mut subject_row_count = 0usize;

    for batch in &mut reader {
        let batch = batch?;
        let columns = SubjectColumns::new(&batch)?;
        for row_index in 0..batch.num_rows() {
            let subject_id = columns.subject_id.value(row_index);
            let hand_index = columns.hand_index.value(row_index);
            let hand_slot = *hand_by_index
                .get(&hand_index)
                .ok_or(BehaviourParquetLoadError::SubjectReferencesUnknownHand { hand_index })?;
            let expected_hand_id = &hands[hand_slot].hand_id;
            let observed_hand_id = columns.hand_id.value(row_index);
            if expected_hand_id.as_str() != observed_hand_id {
                return Err(BehaviourParquetLoadError::SubjectHandIdMismatch {
                    hand_index,
                    expected: expected_hand_id.clone(),
                    observed: HandId::new(observed_hand_id),
                });
            }
            let position_code = columns.subject_position.value(row_index);
            let subject_position =
                position_from_id(position_code).ok_or(BehaviourParquetLoadError::InvalidCode {
                    table: "subjects",
                    column: "subject_position",
                    value: position_code,
                })?;

            let memberships = &mut memberships_by_hand[hand_slot];
            if memberships.iter().any(|(id, _)| *id == subject_id) {
                return Err(BehaviourParquetLoadError::DuplicateSubjectInHand {
                    hand_id: expected_hand_id.clone(),
                    subject_id,
                });
            }
            if memberships
                .iter()
                .any(|(_, position)| *position == subject_position)
            {
                return Err(BehaviourParquetLoadError::DuplicatePositionInHand {
                    hand_id: expected_hand_id.clone(),
                    position: subject_position,
                });
            }
            memberships.push((subject_id, subject_position));
            subjects
                .entry(subject_id)
                .or_default()
                .push(SubjectMembership {
                    hand_slot,
                    subject_position,
                });
            subject_row_count = subject_row_count.checked_add(1).ok_or(
                BehaviourParquetLoadError::CountOverflow {
                    field: "subject rows",
                },
            )?;
        }
    }

    let observed_subject_rows =
        u64::try_from(subject_row_count).map_err(|_| BehaviourParquetLoadError::CountOverflow {
            field: "subject rows",
        })?;
    if observed_subject_rows != summary.subject_hand_rows_written {
        return Err(BehaviourParquetLoadError::SummaryCountMismatch {
            field: "subject_hand_rows_written",
            expected: summary.subject_hand_rows_written,
            observed: observed_subject_rows,
        });
    }

    for (hand_slot, memberships) in memberships_by_hand.iter().enumerate() {
        let hand = &hands[hand_slot];
        if memberships.len() != usize::from(hand.players_at_table) {
            return Err(BehaviourParquetLoadError::SubjectCountMismatch {
                hand_id: hand.hand_id.clone(),
                expected: hand.players_at_table,
                observed: memberships.len(),
            });
        }
        for base in &hand.categorical_without_subject {
            let actor_code = u8::try_from(base[1]).map_err(|_| {
                BehaviourParquetLoadError::InvalidInternalActorPosition {
                    hand_id: hand.hand_id.clone(),
                    value: base[1],
                }
            })?;
            let actor_position = position_from_id(actor_code).ok_or(
                BehaviourParquetLoadError::InvalidInternalActorPosition {
                    hand_id: hand.hand_id.clone(),
                    value: base[1],
                },
            )?;
            if !memberships
                .iter()
                .any(|(_, position)| *position == actor_position)
            {
                return Err(BehaviourParquetLoadError::ActorMissingFromSubjects {
                    hand_id: hand.hand_id.clone(),
                    position: actor_position,
                });
            }
        }
    }

    for memberships in subjects.values_mut() {
        memberships.sort_unstable_by_key(|membership| hands[membership.hand_slot].hand_index);
    }

    Ok(subjects)
}

fn validate_loaded_counts(
    summary: BehaviourParquetDatasetSummary,
    hand_count: usize,
    event_count: usize,
) -> Result<(), BehaviourParquetLoadError> {
    let observed_hands = u64::try_from(hand_count)
        .map_err(|_| BehaviourParquetLoadError::CountOverflow { field: "hands" })?;
    let observed_events = u64::try_from(event_count)
        .map_err(|_| BehaviourParquetLoadError::CountOverflow { field: "events" })?;
    if observed_hands != summary.hands_written {
        return Err(BehaviourParquetLoadError::SummaryCountMismatch {
            field: "hands_written",
            expected: summary.hands_written,
            observed: observed_hands,
        });
    }
    if observed_events != summary.events_written {
        return Err(BehaviourParquetLoadError::SummaryCountMismatch {
            field: "events_written",
            expected: summary.events_written,
            observed: observed_events,
        });
    }
    Ok(())
}

fn open_table_reader(
    path: &Path,
    table: &'static str,
    expected_schema: SchemaRef,
) -> Result<(ParquetRecordBatchReader, BehaviourParquetDatasetSummary), BehaviourParquetLoadError> {
    let file = File::open(path).map_err(|source| BehaviourParquetLoadError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    if builder.schema().as_ref() != expected_schema.as_ref() {
        return Err(BehaviourParquetLoadError::SchemaMismatch {
            table,
            path: path.to_path_buf(),
        });
    }
    let summary = validate_metadata(path, table, builder.metadata())?;
    let file_rows = u64::try_from(builder.metadata().file_metadata().num_rows()).map_err(|_| {
        BehaviourParquetLoadError::InvalidFileRowCount {
            table,
            rows: builder.metadata().file_metadata().num_rows(),
        }
    })?;
    let expected_rows = match table {
        "events" => summary.events_written,
        "subjects" => summary.subject_hand_rows_written,
        _ => {
            return Err(BehaviourParquetLoadError::InternalUnknownTable { table });
        }
    };
    if file_rows != expected_rows {
        return Err(BehaviourParquetLoadError::SummaryCountMismatch {
            field: if table == "events" {
                "events_written"
            } else {
                "subject_hand_rows_written"
            },
            expected: expected_rows,
            observed: file_rows,
        });
    }

    let reader = builder.with_batch_size(READ_BATCH_ROWS).build()?;
    Ok((reader, summary))
}

fn validate_metadata(
    path: &Path,
    table: &'static str,
    metadata: &ParquetMetaData,
) -> Result<BehaviourParquetDatasetSummary, BehaviourParquetLoadError> {
    let entries = metadata
        .file_metadata()
        .key_value_metadata()
        .ok_or_else(|| BehaviourParquetLoadError::MissingMetadata {
            path: path.to_path_buf(),
            key: "<all>",
        })?;
    let mut values = HashMap::<String, String>::with_capacity(entries.len());
    for entry in entries {
        let value = entry.value.as_ref().ok_or_else(|| {
            BehaviourParquetLoadError::MissingMetadataValue {
                path: path.to_path_buf(),
                key: entry.key.clone(),
            }
        })?;
        if values.insert(entry.key.clone(), value.clone()).is_some() {
            return Err(BehaviourParquetLoadError::DuplicateMetadataKey {
                path: path.to_path_buf(),
                key: entry.key.clone(),
            });
        }
    }

    require_metadata(
        path,
        &values,
        "lechiffre.behaviour_schema_version",
        &BEHAVIOUR_PARQUET_SCHEMA_VERSION.to_string(),
    )?;
    require_metadata(path, &values, "lechiffre.table", table)?;
    require_metadata(
        path,
        &values,
        "lechiffre.position_codebook",
        POSITION_CODEBOOK,
    )?;
    require_metadata(
        path,
        &values,
        "lechiffre.hand_index_semantics",
        HAND_INDEX_SEMANTICS,
    )?;

    match table {
        "events" => {
            require_metadata(path, &values, "lechiffre.street_codebook", STREET_CODEBOOK)?;
            require_metadata(path, &values, "lechiffre.action_codebook", ACTION_CODEBOOK)?;
            require_metadata(path, &values, "lechiffre.rank_codebook", RANK_CODEBOOK)?;
            require_metadata(path, &values, "lechiffre.suit_codebook", SUIT_CODEBOOK)?;
        }
        "subjects" => {
            require_metadata(
                path,
                &values,
                "lechiffre.subject_id_semantics",
                SUBJECT_ID_SEMANTICS,
            )?;
        }
        _ => return Err(BehaviourParquetLoadError::InternalUnknownTable { table }),
    }

    let summary = BehaviourParquetDatasetSummary {
        hands_seen: parse_u64_metadata(path, &values, "lechiffre.hands_seen")?,
        hands_written: parse_u64_metadata(path, &values, "lechiffre.hands_written")?,
        events_written: parse_u64_metadata(path, &values, "lechiffre.events_written")?,
        subject_hand_rows_written: parse_u64_metadata(
            path,
            &values,
            "lechiffre.subject_hand_rows_written",
        )?,
        dropped_hands_missing_timing: parse_u64_metadata(
            path,
            &values,
            "lechiffre.dropped_hands_missing_timing",
        )?,
    };
    let reconstructed_seen = summary
        .hands_written
        .checked_add(summary.dropped_hands_missing_timing)
        .ok_or(BehaviourParquetLoadError::CountOverflow {
            field: "hands_seen",
        })?;
    if reconstructed_seen != summary.hands_seen {
        return Err(BehaviourParquetLoadError::InvalidSummaryInvariant {
            hands_seen: summary.hands_seen,
            hands_written: summary.hands_written,
            dropped_hands_missing_timing: summary.dropped_hands_missing_timing,
        });
    }
    Ok(summary)
}

fn require_metadata(
    path: &Path,
    values: &HashMap<String, String>,
    key: &'static str,
    expected: &str,
) -> Result<(), BehaviourParquetLoadError> {
    let observed = values
        .get(key)
        .ok_or_else(|| BehaviourParquetLoadError::MissingMetadata {
            path: path.to_path_buf(),
            key,
        })?;
    if observed != expected {
        return Err(BehaviourParquetLoadError::MetadataMismatch {
            path: path.to_path_buf(),
            key,
            expected: expected.to_string(),
            observed: observed.clone(),
        });
    }
    Ok(())
}

fn parse_u64_metadata(
    path: &Path,
    values: &HashMap<String, String>,
    key: &'static str,
) -> Result<u64, BehaviourParquetLoadError> {
    let value = values
        .get(key)
        .ok_or_else(|| BehaviourParquetLoadError::MissingMetadata {
            path: path.to_path_buf(),
            key,
        })?;
    value
        .parse::<u64>()
        .map_err(|_| BehaviourParquetLoadError::InvalidIntegerMetadata {
            path: path.to_path_buf(),
            key,
            value: value.clone(),
        })
}

struct EventColumns<'a> {
    hand_id: &'a StringArray,
    hand_index: &'a UInt64Array,
    sequence_index: &'a UInt32Array,
    street: &'a UInt8Array,
    actor_position: &'a UInt8Array,
    action: &'a UInt8Array,
    is_all_in: &'a BooleanArray,
    players_at_table: &'a UInt8Array,
    players_active: &'a UInt8Array,
    pot_bb: &'a Float32Array,
    actor_stack_bb: &'a Float32Array,
    effective_stack_bb: &'a Float32Array,
    actor_street_commitment_bb: &'a Float32Array,
    current_bet_bb: &'a Float32Array,
    to_call_bb: &'a Float32Array,
    min_raise_to_bb: &'a Float32Array,
    bet_or_raise_count_this_street: &'a UInt8Array,
    amount_added_bb: &'a Float32Array,
    street_total_after_bb: &'a Float32Array,
    decision_time_ms: &'a UInt32Array,
    board_rank: Vec<&'a UInt8Array>,
    board_suit: Vec<&'a UInt8Array>,
}

impl<'a> EventColumns<'a> {
    fn new(batch: &'a RecordBatch) -> Result<Self, BehaviourParquetLoadError> {
        let hand_id = typed_column::<StringArray>(batch, "events", "hand_id")?;
        let hand_index = typed_column::<UInt64Array>(batch, "events", "hand_index")?;
        let sequence_index = typed_column::<UInt32Array>(batch, "events", "sequence_index")?;
        let street = typed_column::<UInt8Array>(batch, "events", "street")?;
        let actor_position = typed_column::<UInt8Array>(batch, "events", "actor_position")?;
        let action = typed_column::<UInt8Array>(batch, "events", "action")?;
        let is_all_in = typed_column::<BooleanArray>(batch, "events", "is_all_in")?;
        let players_at_table = typed_column::<UInt8Array>(batch, "events", "players_at_table")?;
        let players_active = typed_column::<UInt8Array>(batch, "events", "players_active")?;
        let pot_bb = typed_column::<Float32Array>(batch, "events", "pot_bb")?;
        let actor_stack_bb = typed_column::<Float32Array>(batch, "events", "actor_stack_bb")?;
        let effective_stack_bb =
            typed_column::<Float32Array>(batch, "events", "effective_stack_bb")?;
        let actor_street_commitment_bb =
            typed_column::<Float32Array>(batch, "events", "actor_street_commitment_bb")?;
        let current_bet_bb = typed_column::<Float32Array>(batch, "events", "current_bet_bb")?;
        let to_call_bb = typed_column::<Float32Array>(batch, "events", "to_call_bb")?;
        let min_raise_to_bb = typed_column::<Float32Array>(batch, "events", "min_raise_to_bb")?;
        let bet_or_raise_count_this_street =
            typed_column::<UInt8Array>(batch, "events", "bet_or_raise_count_this_street")?;
        let amount_added_bb = typed_column::<Float32Array>(batch, "events", "amount_added_bb")?;
        let street_total_after_bb =
            typed_column::<Float32Array>(batch, "events", "street_total_after_bb")?;
        let decision_time_ms = typed_column::<UInt32Array>(batch, "events", "decision_time_ms")?;

        for (name, array) in [
            ("hand_id", hand_id as &dyn Array),
            ("hand_index", hand_index as &dyn Array),
            ("sequence_index", sequence_index as &dyn Array),
            ("street", street as &dyn Array),
            ("actor_position", actor_position as &dyn Array),
            ("action", action as &dyn Array),
            ("is_all_in", is_all_in as &dyn Array),
            ("players_at_table", players_at_table as &dyn Array),
            ("players_active", players_active as &dyn Array),
            ("pot_bb", pot_bb as &dyn Array),
            ("actor_stack_bb", actor_stack_bb as &dyn Array),
            ("effective_stack_bb", effective_stack_bb as &dyn Array),
            (
                "actor_street_commitment_bb",
                actor_street_commitment_bb as &dyn Array,
            ),
            ("current_bet_bb", current_bet_bb as &dyn Array),
            ("to_call_bb", to_call_bb as &dyn Array),
            (
                "bet_or_raise_count_this_street",
                bet_or_raise_count_this_street as &dyn Array,
            ),
            ("amount_added_bb", amount_added_bb as &dyn Array),
            ("street_total_after_bb", street_total_after_bb as &dyn Array),
            ("decision_time_ms", decision_time_ms as &dyn Array),
        ] {
            ensure_non_nullable("events", name, array)?;
        }

        let mut board_rank = Vec::with_capacity(MAX_BOARD_CARDS);
        let mut board_suit = Vec::with_capacity(MAX_BOARD_CARDS);
        for index in 0..MAX_BOARD_CARDS {
            board_rank.push(typed_column::<UInt8Array>(
                batch,
                "events",
                &format!("board_{index}_rank"),
            )?);
            board_suit.push(typed_column::<UInt8Array>(
                batch,
                "events",
                &format!("board_{index}_suit"),
            )?);
        }

        Ok(Self {
            hand_id,
            hand_index,
            sequence_index,
            street,
            actor_position,
            action,
            is_all_in,
            players_at_table,
            players_active,
            pot_bb,
            actor_stack_bb,
            effective_stack_bb,
            actor_street_commitment_bb,
            current_bet_bb,
            to_call_bb,
            min_raise_to_bb,
            bet_or_raise_count_this_street,
            amount_added_bb,
            street_total_after_bb,
            decision_time_ms,
            board_rank,
            board_suit,
        })
    }

    fn event_row(
        &self,
        row: usize,
        hand_id: HandId,
    ) -> Result<EventRow, BehaviourParquetLoadError> {
        let street = self.street.value(row);
        if street_from_id(street).is_none() {
            return Err(invalid_code("events", "street", street));
        }
        let actor_position = self.actor_position.value(row);
        if position_from_id(actor_position).is_none() {
            return Err(invalid_code("events", "actor_position", actor_position));
        }
        let action = self.action.value(row);
        if action_from_id(action).is_none() {
            return Err(invalid_code("events", "action", action));
        }

        let players_at_table = self.players_at_table.value(row);
        if !(2..=6).contains(&players_at_table) {
            return Err(BehaviourParquetLoadError::InvalidPlayersAtTable {
                hand_id: hand_id.clone(),
                value: players_at_table,
            });
        }
        let players_active = self.players_active.value(row);
        if players_active > players_at_table {
            return Err(BehaviourParquetLoadError::InvalidPlayersActive {
                hand_id: hand_id.clone(),
                active: players_active,
                table: players_at_table,
            });
        }

        let mut continuous = [0.0_f32; EVENT_CONTINUOUS_WIDTH];
        let mut continuous_present = [1_u8; EVENT_CONTINUOUS_WIDTH];
        continuous[ContinuousFeature::PlayersAtTable as usize] = f32::from(players_at_table);
        continuous[ContinuousFeature::PlayersActive as usize] = f32::from(players_active);
        continuous[ContinuousFeature::PotBb as usize] =
            checked_non_negative_f32(&hand_id, "pot_bb", self.pot_bb.value(row))?;
        continuous[ContinuousFeature::ActorStackBb as usize] =
            checked_non_negative_f32(&hand_id, "actor_stack_bb", self.actor_stack_bb.value(row))?;
        continuous[ContinuousFeature::EffectiveStackBb as usize] = checked_non_negative_f32(
            &hand_id,
            "effective_stack_bb",
            self.effective_stack_bb.value(row),
        )?;
        continuous[ContinuousFeature::ActorStreetCommitmentBb as usize] = checked_non_negative_f32(
            &hand_id,
            "actor_street_commitment_bb",
            self.actor_street_commitment_bb.value(row),
        )?;
        continuous[ContinuousFeature::CurrentBetBb as usize] =
            checked_non_negative_f32(&hand_id, "current_bet_bb", self.current_bet_bb.value(row))?;
        continuous[ContinuousFeature::ToCallBb as usize] =
            checked_non_negative_f32(&hand_id, "to_call_bb", self.to_call_bb.value(row))?;
        if self.min_raise_to_bb.is_null(row) {
            continuous_present[ContinuousFeature::MinRaiseToBb as usize] = 0;
        } else {
            continuous[ContinuousFeature::MinRaiseToBb as usize] = checked_non_negative_f32(
                &hand_id,
                "min_raise_to_bb",
                self.min_raise_to_bb.value(row),
            )?;
        }
        continuous[ContinuousFeature::BetOrRaiseCountThisStreet as usize] =
            f32::from(self.bet_or_raise_count_this_street.value(row));
        continuous[ContinuousFeature::AmountAddedBb as usize] =
            checked_non_negative_f32(&hand_id, "amount_added_bb", self.amount_added_bb.value(row))?;
        continuous[ContinuousFeature::StreetTotalAfterBb as usize] = checked_non_negative_f32(
            &hand_id,
            "street_total_after_bb",
            self.street_total_after_bb.value(row),
        )?;
        let decision_time_ms = self.decision_time_ms.value(row);
        continuous[ContinuousFeature::DecisionTimeMs as usize] = decision_time_ms as f32;

        let mut board_rank = [0_i64; MAX_BOARD_CARDS];
        let mut board_suit = [0_i64; MAX_BOARD_CARDS];
        let mut board_present = [0_u8; MAX_BOARD_CARDS];
        let mut encountered_absent_slot = false;
        for index in 0..MAX_BOARD_CARDS {
            let rank = self.board_rank[index];
            let suit = self.board_suit[index];
            match (rank.is_null(row), suit.is_null(row)) {
                (true, true) => encountered_absent_slot = true,
                (false, false) => {
                    if encountered_absent_slot {
                        return Err(BehaviourParquetLoadError::NonContiguousBoard {
                            hand_id: hand_id.clone(),
                            slot: index,
                        });
                    }
                    let rank_value = rank.value(row);
                    if rank_from_id(rank_value).is_none() {
                        return Err(invalid_code("events", "board_rank", rank_value));
                    }
                    let suit_value = suit.value(row);
                    if suit_from_id(suit_value).is_none() {
                        return Err(invalid_code("events", "board_suit", suit_value));
                    }
                    board_rank[index] = i64::from(rank_value);
                    board_suit[index] = i64::from(suit_value);
                    board_present[index] = 1;
                }
                _ => {
                    return Err(BehaviourParquetLoadError::BoardRankSuitNullMismatch {
                        hand_id: hand_id.clone(),
                        slot: index,
                    });
                }
            }
        }

        Ok(EventRow {
            hand_id,
            sequence_index: self.sequence_index.value(row),
            players_at_table,
            categorical_without_subject: [
                i64::from(street),
                i64::from(actor_position),
                i64::from(action),
                if self.is_all_in.value(row) { 1 } else { 0 },
            ],
            continuous,
            continuous_present,
            decision_time_ms,
            board_rank,
            board_suit,
            board_present,
        })
    }
}

struct SubjectColumns<'a> {
    subject_id: &'a UInt64Array,
    hand_id: &'a StringArray,
    hand_index: &'a UInt64Array,
    subject_position: &'a UInt8Array,
}

impl<'a> SubjectColumns<'a> {
    fn new(batch: &'a RecordBatch) -> Result<Self, BehaviourParquetLoadError> {
        let subject_id = typed_column::<UInt64Array>(batch, "subjects", "subject_id")?;
        let hand_id = typed_column::<StringArray>(batch, "subjects", "hand_id")?;
        let hand_index = typed_column::<UInt64Array>(batch, "subjects", "hand_index")?;
        let subject_position = typed_column::<UInt8Array>(batch, "subjects", "subject_position")?;
        for (name, array) in [
            ("subject_id", subject_id as &dyn Array),
            ("hand_id", hand_id as &dyn Array),
            ("hand_index", hand_index as &dyn Array),
            ("subject_position", subject_position as &dyn Array),
        ] {
            ensure_non_nullable("subjects", name, array)?;
        }
        Ok(Self {
            subject_id,
            hand_id,
            hand_index,
            subject_position,
        })
    }
}

fn typed_column<'a, T: Array + 'static>(
    batch: &'a RecordBatch,
    table: &'static str,
    column: &str,
) -> Result<&'a T, BehaviourParquetLoadError> {
    let index =
        batch
            .schema()
            .index_of(column)
            .map_err(|_| BehaviourParquetLoadError::MissingColumn {
                table,
                column: column.to_string(),
            })?;
    batch
        .column(index)
        .as_any()
        .downcast_ref::<T>()
        .ok_or_else(|| BehaviourParquetLoadError::ColumnTypeMismatch {
            table,
            column: column.to_string(),
        })
}

fn ensure_non_nullable(
    table: &'static str,
    column: &'static str,
    array: &dyn Array,
) -> Result<(), BehaviourParquetLoadError> {
    if array.null_count() != 0 {
        return Err(BehaviourParquetLoadError::UnexpectedNull { table, column });
    }
    Ok(())
}

fn checked_non_negative_f32(
    hand_id: &HandId,
    field: &'static str,
    value: f32,
) -> Result<f32, BehaviourParquetLoadError> {
    if !value.is_finite() || value < 0.0 {
        return Err(BehaviourParquetLoadError::InvalidFloat {
            hand_id: hand_id.clone(),
            field,
            value,
        });
    }
    Ok(value)
}

fn invalid_code(table: &'static str, column: &'static str, value: u8) -> BehaviourParquetLoadError {
    BehaviourParquetLoadError::InvalidCode {
        table,
        column,
        value,
    }
}

#[derive(Debug)]
pub enum BehaviourParquetLoadError {
    Io {
        path: PathBuf,
        source: io::Error,
    },
    Arrow(ArrowError),
    Parquet(ParquetError),
    MissingSuccessMarker {
        path: PathBuf,
    },
    SchemaMismatch {
        table: &'static str,
        path: PathBuf,
    },
    MissingMetadata {
        path: PathBuf,
        key: &'static str,
    },
    MissingMetadataValue {
        path: PathBuf,
        key: String,
    },
    DuplicateMetadataKey {
        path: PathBuf,
        key: String,
    },
    MetadataMismatch {
        path: PathBuf,
        key: &'static str,
        expected: String,
        observed: String,
    },
    InvalidIntegerMetadata {
        path: PathBuf,
        key: &'static str,
        value: String,
    },
    InvalidSummaryInvariant {
        hands_seen: u64,
        hands_written: u64,
        dropped_hands_missing_timing: u64,
    },
    InvalidFileRowCount {
        table: &'static str,
        rows: i64,
    },
    SummaryCountMismatch {
        field: &'static str,
        expected: u64,
        observed: u64,
    },
    CrossTableSummaryMismatch {
        events: BehaviourParquetDatasetSummary,
        subjects: BehaviourParquetDatasetSummary,
    },
    MissingColumn {
        table: &'static str,
        column: String,
    },
    ColumnTypeMismatch {
        table: &'static str,
        column: String,
    },
    UnexpectedNull {
        table: &'static str,
        column: &'static str,
    },
    InvalidCode {
        table: &'static str,
        column: &'static str,
        value: u8,
    },
    InvalidFloat {
        hand_id: HandId,
        field: &'static str,
        value: f32,
    },
    InvalidPlayersAtTable {
        hand_id: HandId,
        value: u8,
    },
    InvalidPlayersActive {
        hand_id: HandId,
        active: u8,
        table: u8,
    },
    BoardRankSuitNullMismatch {
        hand_id: HandId,
        slot: usize,
    },
    NonContiguousBoard {
        hand_id: HandId,
        slot: usize,
    },
    HandIndexOutsideSeenRange {
        hand_index: u64,
        hands_seen: u64,
    },
    NonMonotonicHandIndex {
        previous: Option<u64>,
        observed: u64,
    },
    DuplicateHandId {
        hand_id: HandId,
    },
    HandIdentityChangedWithinIndex {
        hand_index: u64,
        expected: HandId,
        observed: HandId,
    },
    PlayerCountChangedWithinHand {
        hand_id: HandId,
        expected: u8,
        observed: u8,
    },
    NonContiguousSequence {
        hand_id: HandId,
        expected: u32,
        observed: u32,
    },
    SequenceLengthOverflow {
        hand_id: HandId,
    },
    SubjectReferencesUnknownHand {
        hand_index: u64,
    },
    SubjectHandIdMismatch {
        hand_index: u64,
        expected: HandId,
        observed: HandId,
    },
    DuplicateSubjectInHand {
        hand_id: HandId,
        subject_id: u64,
    },
    DuplicatePositionInHand {
        hand_id: HandId,
        position: Position,
    },
    SubjectCountMismatch {
        hand_id: HandId,
        expected: u8,
        observed: usize,
    },
    ActorMissingFromSubjects {
        hand_id: HandId,
        position: Position,
    },
    InvalidInternalActorPosition {
        hand_id: HandId,
        value: i64,
    },
    CountOverflow {
        field: &'static str,
    },
    InternalMissingCurrentHand,
    InternalUnknownTable {
        table: &'static str,
    },
}

impl fmt::Display for BehaviourParquetLoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, source } => write!(f, "failed to read {}: {source}", path.display()),
            Self::Arrow(error) => write!(f, "Arrow decoding failed: {error}"),
            Self::Parquet(error) => write!(f, "Parquet decoding failed: {error}"),
            Self::MissingSuccessMarker { path } => {
                write!(
                    f,
                    "behaviour dataset is not committed: missing {}",
                    path.display()
                )
            }
            Self::SchemaMismatch { table, path } => {
                write!(
                    f,
                    "{table} schema does not match LeChiffre v1 at {}",
                    path.display()
                )
            }
            Self::MissingMetadata { path, key } => {
                write!(f, "missing Parquet metadata {key} in {}", path.display())
            }
            Self::MissingMetadataValue { path, key } => write!(
                f,
                "Parquet metadata {key} has no value in {}",
                path.display()
            ),
            Self::DuplicateMetadataKey { path, key } => write!(
                f,
                "Parquet metadata key {key} is duplicated in {}",
                path.display()
            ),
            Self::MetadataMismatch {
                path,
                key,
                expected,
                observed,
            } => write!(
                f,
                "Parquet metadata {key} mismatch in {}: expected {expected}, got {observed}",
                path.display()
            ),
            Self::InvalidIntegerMetadata { path, key, value } => write!(
                f,
                "Parquet metadata {key} in {} is not a valid u64: {value}",
                path.display()
            ),
            Self::InvalidSummaryInvariant {
                hands_seen,
                hands_written,
                dropped_hands_missing_timing,
            } => write!(
                f,
                "invalid dataset summary: hands_seen={hands_seen}, hands_written={hands_written}, \
                 dropped_hands_missing_timing={dropped_hands_missing_timing}"
            ),
            Self::InvalidFileRowCount { table, rows } => {
                write!(
                    f,
                    "{table} Parquet footer contains invalid row count {rows}"
                )
            }
            Self::SummaryCountMismatch {
                field,
                expected,
                observed,
            } => write!(
                f,
                "dataset count {field} mismatch: metadata says {expected}, observed {observed}"
            ),
            Self::CrossTableSummaryMismatch { events, subjects } => write!(
                f,
                "events/subjects dataset summaries differ: events={events:?}, subjects={subjects:?}"
            ),
            Self::MissingColumn { table, column } => {
                write!(f, "missing column {column} in {table} record batch")
            }
            Self::ColumnTypeMismatch { table, column } => {
                write!(f, "column {column} has the wrong Arrow type in {table}")
            }
            Self::UnexpectedNull { table, column } => {
                write!(f, "non-nullable {table}.{column} contains NULL")
            }
            Self::InvalidCode {
                table,
                column,
                value,
            } => write!(f, "invalid code {value} in {table}.{column}"),
            Self::InvalidFloat {
                hand_id,
                field,
                value,
            } => write!(f, "hand {hand_id} contains invalid {field} value {value}"),
            Self::InvalidPlayersAtTable { hand_id, value } => {
                write!(f, "hand {hand_id} has invalid players_at_table {value}")
            }
            Self::InvalidPlayersActive {
                hand_id,
                active,
                table,
            } => write!(
                f,
                "hand {hand_id} has {active} active players with only {table} at the table"
            ),
            Self::BoardRankSuitNullMismatch { hand_id, slot } => write!(
                f,
                "hand {hand_id} board slot {slot} has rank/suit NULL mismatch"
            ),
            Self::NonContiguousBoard { hand_id, slot } => write!(
                f,
                "hand {hand_id} board slot {slot} is present after an absent slot"
            ),
            Self::HandIndexOutsideSeenRange {
                hand_index,
                hands_seen,
            } => write!(
                f,
                "hand_index {hand_index} is outside metadata hands_seen range 0..{hands_seen}"
            ),
            Self::NonMonotonicHandIndex { previous, observed } => write!(
                f,
                "event hand_index order is not strictly increasing between hands: \
                 previous={previous:?}, observed={observed}"
            ),
            Self::DuplicateHandId { hand_id } => {
                write!(f, "duplicate hand id {hand_id} in events dataset")
            }
            Self::HandIdentityChangedWithinIndex {
                hand_index,
                expected,
                observed,
            } => write!(
                f,
                "hand_index {hand_index} changes identity from {expected} to {observed}"
            ),
            Self::PlayerCountChangedWithinHand {
                hand_id,
                expected,
                observed,
            } => write!(
                f,
                "hand {hand_id} changes players_at_table from {expected} to {observed}"
            ),
            Self::NonContiguousSequence {
                hand_id,
                expected,
                observed,
            } => write!(
                f,
                "hand {hand_id} has non-contiguous sequence index: \
                 expected {expected}, got {observed}"
            ),
            Self::SequenceLengthOverflow { hand_id } => {
                write!(f, "hand {hand_id} contains more than u32::MAX events")
            }
            Self::SubjectReferencesUnknownHand { hand_index } => write!(
                f,
                "subjects dataset references unknown retained hand_index {hand_index}"
            ),
            Self::SubjectHandIdMismatch {
                hand_index,
                expected,
                observed,
            } => write!(
                f,
                "subjects hand_index {hand_index} names {observed}, events name it {expected}"
            ),
            Self::DuplicateSubjectInHand {
                hand_id,
                subject_id,
            } => write!(
                f,
                "hand {hand_id} contains subject {subject_id} more than once"
            ),
            Self::DuplicatePositionInHand { hand_id, position } => {
                write!(
                    f,
                    "hand {hand_id} contains subject position {position:?} more than once"
                )
            }
            Self::SubjectCountMismatch {
                hand_id,
                expected,
                observed,
            } => write!(
                f,
                "hand {hand_id} has {observed} subject rows but players_at_table is {expected}"
            ),
            Self::ActorMissingFromSubjects { hand_id, position } => write!(
                f,
                "hand {hand_id} has actor position {position:?} without a matching subject row"
            ),
            Self::InvalidInternalActorPosition { hand_id, value } => write!(
                f,
                "hand {hand_id} contains invalid internal actor position code {value}"
            ),
            Self::CountOverflow { field } => write!(f, "dataset {field} count overflow"),
            Self::InternalMissingCurrentHand => {
                write!(
                    f,
                    "internal loader error: no current hand while decoding event"
                )
            }
            Self::InternalUnknownTable { table } => {
                write!(f, "internal loader error: unknown table {table}")
            }
        }
    }
}

impl std::error::Error for BehaviourParquetLoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Arrow(error) => Some(error),
            Self::Parquet(error) => Some(error),
            _ => None,
        }
    }
}

impl From<ArrowError> for BehaviourParquetLoadError {
    fn from(error: ArrowError) -> Self {
        Self::Arrow(error)
    }
}

impl From<ParquetError> for BehaviourParquetLoadError {
    fn from(error: ParquetError) -> Self {
        Self::Parquet(error)
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;
    use crate::adapters::{AdaptedHand, ObservedPlayer, PlayerId};
    use crate::behavioural::{
        ActionKind, BehaviourEvent, BehaviourSequence, DecisionContext, ObservedAction, PublicCard,
        Rank, Street, Suit,
    };
    use crate::datasets::behaviour_parquet::BehaviourParquetDatasetWriter;

    static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn temp_dataset_dir(name: &str) -> PathBuf {
        let id = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "lechiffre-loader-{name}-{}-{id}",
            std::process::id()
        ))
    }

    fn subject_ids() -> HashMap<PlayerId, u64> {
        HashMap::from([
            (PlayerId::new("test:alice"), 17),
            (PlayerId::new("test:bob"), 42),
        ])
    }

    fn make_hand(hand_id: &str, first_timing: Option<u32>) -> AdaptedHand {
        let mut sequence = BehaviourSequence::new(hand_id);
        sequence
            .push(
                BehaviourEvent::new(
                    hand_id,
                    0,
                    DecisionContext {
                        street: Street::Preflop,
                        board: vec![],
                        actor_position: Position::ButtonSmallBlind,
                        players_at_table: 2,
                        players_active: 2,
                        pot_bb: 1.5,
                        actor_stack_bb: 99.5,
                        effective_stack_bb: 99.0,
                        actor_street_commitment_bb: 0.5,
                        current_bet_bb: 1.0,
                        to_call_bb: 0.5,
                        min_raise_to_bb: Some(2.0),
                        bet_or_raise_count_this_street: 0,
                    },
                    ObservedAction {
                        kind: ActionKind::Call,
                        is_all_in: false,
                        amount_added_bb: 0.5,
                        street_total_after_bb: 1.0,
                        decision_time_ms: first_timing,
                    },
                )
                .unwrap(),
            )
            .unwrap();
        sequence
            .push(
                BehaviourEvent::new(
                    hand_id,
                    1,
                    DecisionContext {
                        street: Street::Flop,
                        board: vec![
                            PublicCard {
                                rank: Rank::King,
                                suit: Suit::Spades,
                            },
                            PublicCard {
                                rank: Rank::Eight,
                                suit: Suit::Hearts,
                            },
                            PublicCard {
                                rank: Rank::Three,
                                suit: Suit::Clubs,
                            },
                        ],
                        actor_position: Position::BigBlind,
                        players_at_table: 2,
                        players_active: 2,
                        pot_bb: 2.0,
                        actor_stack_bb: 99.0,
                        effective_stack_bb: 99.0,
                        actor_street_commitment_bb: 0.0,
                        current_bet_bb: 0.0,
                        to_call_bb: 0.0,
                        min_raise_to_bb: None,
                        bet_or_raise_count_this_street: 0,
                    },
                    ObservedAction {
                        kind: ActionKind::Check,
                        is_all_in: false,
                        amount_added_bb: 0.0,
                        street_total_after_bb: 0.0,
                        decision_time_ms: Some(320),
                    },
                )
                .unwrap(),
            )
            .unwrap();

        AdaptedHand::new(
            sequence,
            vec![
                ObservedPlayer {
                    player_id: PlayerId::new("test:alice"),
                    position: Position::ButtonSmallBlind,
                },
                ObservedPlayer {
                    player_id: PlayerId::new("test:bob"),
                    position: Position::BigBlind,
                },
            ],
        )
    }

    #[test]
    fn loads_unique_events_and_materializes_subject_specific_histories() {
        let dir = temp_dataset_dir("roundtrip");
        let exact_timing = 16_777_217_u32;
        let hand = make_hand("hand-a", Some(exact_timing));
        let mut writer = BehaviourParquetDatasetWriter::create(&dir).unwrap();
        writer.write_hands(&[hand], &subject_ids()).unwrap();
        writer.close().unwrap();

        let dataset = BehaviourParquetDataset::open(&dir).unwrap();
        assert_eq!(dataset.hand_count(), 1);
        assert_eq!(dataset.event_count(), 2);
        assert_eq!(dataset.subject_count(), 2);
        assert_eq!(dataset.subject_ids().collect::<Vec<_>>(), vec![17, 42]);
        assert_eq!(dataset.subject_hand_count(17), Some(1));
        assert_eq!(dataset.subject_hand_count(999), None);

        let alice = dataset.subject_history(17).unwrap();
        let bob = dataset.subject_history(42).unwrap();
        assert_eq!(alice.subject_id(), 17);
        assert_eq!(bob.subject_id(), 42);
        assert_eq!(alice.len(), 1);
        assert_eq!(bob.len(), 1);
        assert_eq!(alice.total_events(), 2);
        assert_eq!(dataset.subject_hands(17).unwrap().len(), 1);
        assert_eq!(alice.hands()[0].hand_id().as_str(), "hand-a");
        assert_eq!(alice.hands()[0].decision_time_ms(), &[exact_timing, 320]);
        assert_eq!(bob.hands()[0].decision_time_ms(), &[exact_timing, 320]);
        assert_eq!(
            alice.hands()[0].categorical()[0][CategoricalFeature::IsSubject as usize],
            1
        );
        assert_eq!(
            alice.hands()[0].categorical()[1][CategoricalFeature::IsSubject as usize],
            0
        );
        assert_eq!(
            bob.hands()[0].categorical()[0][CategoricalFeature::IsSubject as usize],
            0
        );
        assert_eq!(
            bob.hands()[0].categorical()[1][CategoricalFeature::IsSubject as usize],
            1
        );
        assert_eq!(
            alice.hands()[0].continuous_present()[1][ContinuousFeature::MinRaiseToBb as usize],
            0
        );
        assert_eq!(alice.hands()[0].board_rank()[1], [13, 8, 3, 0, 0]);
        assert_eq!(alice.hands()[0].board_present()[1], [1, 1, 1, 0, 0]);
        assert!(dataset.subject_history(999).is_none());

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn preserves_retained_hand_order_across_dropped_timing_gaps() {
        let dir = temp_dataset_dir("order-gaps");
        let hands = vec![
            make_hand("hand-a", Some(100)),
            make_hand("hand-dropped", None),
            make_hand("hand-c", Some(300)),
        ];
        let mut writer = BehaviourParquetDatasetWriter::create(&dir).unwrap();
        writer.write_hands(&hands, &subject_ids()).unwrap();
        writer.close().unwrap();

        let dataset = BehaviourParquetDataset::open(&dir).unwrap();
        assert_eq!(dataset.summary().hands_seen, 3);
        assert_eq!(dataset.summary().hands_written, 2);
        assert_eq!(dataset.summary().dropped_hands_missing_timing, 1);
        let history = dataset.subject_history(17).unwrap();
        assert_eq!(history.len(), 2);
        assert_eq!(history.hands()[0].hand_id().as_str(), "hand-a");
        assert_eq!(history.hands()[1].hand_id().as_str(), "hand-c");

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn rejects_dataset_without_commit_marker() {
        let dir = temp_dataset_dir("missing-success");
        let hand = make_hand("hand-a", Some(100));
        let mut writer = BehaviourParquetDatasetWriter::create(&dir).unwrap();
        writer.write_hands(&[hand], &subject_ids()).unwrap();
        writer.close().unwrap();
        fs::remove_file(dir.join(SUCCESS_FILE_NAME)).unwrap();

        assert!(matches!(
            BehaviourParquetDataset::open(&dir),
            Err(BehaviourParquetLoadError::MissingSuccessMarker { .. })
        ));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn rejects_cross_table_schema_substitution() {
        let dir = temp_dataset_dir("schema-substitution");
        let hand = make_hand("hand-a", Some(100));
        let mut writer = BehaviourParquetDatasetWriter::create(&dir).unwrap();
        writer.write_hands(&[hand], &subject_ids()).unwrap();
        writer.close().unwrap();
        fs::copy(dir.join(EVENTS_FILE_NAME), dir.join(SUBJECTS_FILE_NAME)).unwrap();

        assert!(matches!(
            BehaviourParquetDataset::open(&dir),
            Err(BehaviourParquetLoadError::SchemaMismatch {
                table: "subjects",
                ..
            })
        ));
        fs::remove_dir_all(dir).unwrap();
    }
}
