use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use arrow_array::{
    ArrayRef, BooleanArray, Float32Array, RecordBatch, StringArray, UInt8Array, UInt32Array,
    UInt64Array,
};
use arrow_schema::{ArrowError, DataType, Field, Schema, SchemaRef};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::errors::ParquetError;
use parquet::file::metadata::KeyValue;
use parquet::file::properties::WriterProperties;

use crate::adapters::{AdaptedHand, PlayerId};
use crate::behavioural::codebook::{
    ACTION_CODEBOOK, POSITION_CODEBOOK, STREET_CODEBOOK, action_id, position_id, street_id,
};
use crate::behavioural::{BehaviourEvent, EVENT_CONTINUOUS_WIDTH, HandId};
use crate::behavioural::{continuous_feature_name, encode_continuous};

pub const EVENTS_FILE_NAME: &str = "events.parquet";
pub const SUBJECTS_FILE_NAME: &str = "subjects.parquet";
pub const SUCCESS_FILE_NAME: &str = "_SUCCESS";

static WRITER_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Writes a normalized behavioural training dataset as two Parquet tables.
///
/// `events.parquet` stores every observed decision exactly once. `subjects.parquet`
/// stores only the pseudonymous subject-to-hand relation. The subject marker used
/// by the model is therefore derived later from `actor_position == subject_position`
/// instead of duplicating a complete hand once per player.
pub struct BehaviourParquetDatasetWriter {
    events_path: PathBuf,
    subjects_path: PathBuf,
    success_path: PathBuf,
    events_in_progress_path: PathBuf,
    subjects_in_progress_path: PathBuf,
    events_writer: Option<ArrowWriter<File>>,
    subjects_writer: Option<ArrowWriter<File>>,
    seen_hand_ids: HashSet<HandId>,
    player_to_subject: HashMap<PlayerId, u64>,
    subject_to_player: HashMap<u64, PlayerId>,
    summary: BehaviourParquetDatasetSummary,
    next_hand_index: u64,
    poisoned: bool,
    events_final_linked: bool,
    subjects_final_linked: bool,
    success_created: bool,
    committed: bool,
}

impl BehaviourParquetDatasetWriter {
    pub fn create(root: impl AsRef<Path>) -> Result<Self, BehaviourParquetDatasetError> {
        let root = root.as_ref().to_path_buf();
        fs::create_dir_all(&root)?;

        let events_path = root.join(EVENTS_FILE_NAME);
        let subjects_path = root.join(SUBJECTS_FILE_NAME);
        let success_path = root.join(SUCCESS_FILE_NAME);
        let writer_id = WRITER_COUNTER.fetch_add(1, Ordering::Relaxed);
        let in_progress_suffix = format!("{}.{}", std::process::id(), writer_id);
        let events_in_progress_path =
            root.join(format!(".events.parquet.{in_progress_suffix}.inprogress"));
        let subjects_in_progress_path =
            root.join(format!(".subjects.parquet.{in_progress_suffix}.inprogress"));

        for path in [&events_path, &subjects_path, &success_path] {
            if path.exists() {
                return Err(BehaviourParquetDatasetError::OutputAlreadyExists {
                    path: path.clone(),
                });
            }
        }

        let result: Result<Self, BehaviourParquetDatasetError> = (|| {
            let events_file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&events_in_progress_path)?;
            let subjects_file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&subjects_in_progress_path)?;

            let events_writer = ArrowWriter::try_new(
                events_file,
                events_schema(),
                Some(writer_properties("events")),
            )?;
            let subjects_writer = ArrowWriter::try_new(
                subjects_file,
                subjects_schema(),
                Some(writer_properties("subjects")),
            )?;

            Ok(Self {
                events_path: events_path.clone(),
                subjects_path: subjects_path.clone(),
                success_path: success_path.clone(),
                events_in_progress_path: events_in_progress_path.clone(),
                subjects_in_progress_path: subjects_in_progress_path.clone(),
                events_writer: Some(events_writer),
                subjects_writer: Some(subjects_writer),
                seen_hand_ids: HashSet::new(),
                player_to_subject: HashMap::new(),
                subject_to_player: HashMap::new(),
                summary: BehaviourParquetDatasetSummary::default(),
                next_hand_index: 0,
                poisoned: false,
                events_final_linked: false,
                subjects_final_linked: false,
                success_created: false,
                committed: false,
            })
        })();

        if result.is_err() {
            let _ = fs::remove_file(&events_in_progress_path);
            let _ = fs::remove_file(&subjects_in_progress_path);
        }

        result
    }

    /// Writes unique adapted hands and their subject membership.
    ///
    /// `subject_ids` is ingestion metadata. It must provide exactly one stable
    /// pseudonymous `u64` for every observed player in every retained hand.
    /// Player names/IDs are never persisted in either Parquet table.
    ///
    /// A hand with any missing `decision_time_ms` is dropped as a whole. All
    /// other validation failures are hard errors; no values are guessed or
    /// imputed.
    pub fn write_hands(
        &mut self,
        hands: &[AdaptedHand],
        subject_ids: &HashMap<PlayerId, u64>,
    ) -> Result<BehaviourParquetWriteReport, BehaviourParquetDatasetError> {
        self.ensure_writable()?;
        self.preflight_subject_ids(subject_ids)?;

        let batch_len = u64::try_from(hands.len())
            .map_err(|_| BehaviourParquetDatasetError::HandIndexOverflow)?;
        let next_hand_index = self
            .next_hand_index
            .checked_add(batch_len)
            .ok_or(BehaviourParquetDatasetError::HandIndexOverflow)?;

        let mut batch_hand_ids = HashSet::with_capacity(hands.len());
        let mut retained = Vec::with_capacity(hands.len());
        let mut dropped_hand_ids_missing_timing = Vec::new();

        for (offset, hand) in hands.iter().enumerate() {
            let offset = u64::try_from(offset)
                .map_err(|_| BehaviourParquetDatasetError::HandIndexOverflow)?;
            let hand_index = self
                .next_hand_index
                .checked_add(offset)
                .ok_or(BehaviourParquetDatasetError::HandIndexOverflow)?;
            let hand_id = hand.sequence().hand_id().clone();
            if self.seen_hand_ids.contains(&hand_id) || !batch_hand_ids.insert(hand_id.clone()) {
                return Err(BehaviourParquetDatasetError::DuplicateHandId { hand_id });
            }
            if hand.sequence().is_empty() {
                return Err(BehaviourParquetDatasetError::EmptyHand { hand_id });
            }

            validate_hand_structure(hand)?;

            if hand
                .sequence()
                .events()
                .iter()
                .any(|event| event.action.decision_time_ms.is_none())
            {
                dropped_hand_ids_missing_timing.push(hand_id.clone());
                continue;
            }

            for player in hand.players() {
                let subject_id = *subject_ids.get(&player.player_id).ok_or_else(|| {
                    BehaviourParquetDatasetError::MissingSubjectId {
                        player_id: player.player_id.clone(),
                        hand_id: hand_id.clone(),
                    }
                })?;
                self.validate_registered_subject(&player.player_id, subject_id)?;
            }

            retained.push((hand, hand_index));
        }

        let events_batch = match events_record_batch(&retained) {
            Ok(batch) => batch,
            Err(error) => {
                self.poisoned = true;
                return Err(error);
            }
        };
        let subjects_batch = match subjects_record_batch(&retained, subject_ids) {
            Ok(batch) => batch,
            Err(error) => {
                self.poisoned = true;
                return Err(error);
            }
        };

        let events_written = events_batch.num_rows();
        let subject_hand_rows_written = subjects_batch.num_rows();
        let hands_written_u64 = u64::try_from(retained.len()).map_err(|_| {
            BehaviourParquetDatasetError::CounterOverflow {
                field: "hands_written",
            }
        })?;
        let events_written_u64 = u64::try_from(events_written).map_err(|_| {
            BehaviourParquetDatasetError::CounterOverflow {
                field: "events_written",
            }
        })?;
        let subject_rows_u64 = u64::try_from(subject_hand_rows_written).map_err(|_| {
            BehaviourParquetDatasetError::CounterOverflow {
                field: "subject_hand_rows_written",
            }
        })?;
        let dropped_u64 = u64::try_from(dropped_hand_ids_missing_timing.len()).map_err(|_| {
            BehaviourParquetDatasetError::CounterOverflow {
                field: "dropped_hands_missing_timing",
            }
        })?;
        let next_summary = BehaviourParquetDatasetSummary {
            hands_seen: checked_counter_add(self.summary.hands_seen, batch_len, "hands_seen")?,
            hands_written: checked_counter_add(
                self.summary.hands_written,
                hands_written_u64,
                "hands_written",
            )?,
            events_written: checked_counter_add(
                self.summary.events_written,
                events_written_u64,
                "events_written",
            )?,
            subject_hand_rows_written: checked_counter_add(
                self.summary.subject_hand_rows_written,
                subject_rows_u64,
                "subject_hand_rows_written",
            )?,
            dropped_hands_missing_timing: checked_counter_add(
                self.summary.dropped_hands_missing_timing,
                dropped_u64,
                "dropped_hands_missing_timing",
            )?,
        };

        if events_batch.num_rows() > 0
            && let Err(error) = self
                .events_writer
                .as_mut()
                .ok_or(BehaviourParquetDatasetError::WriterClosed)?
                .write(&events_batch)
        {
            self.poisoned = true;
            return Err(error.into());
        }
        if subjects_batch.num_rows() > 0
            && let Err(error) = self
                .subjects_writer
                .as_mut()
                .ok_or(BehaviourParquetDatasetError::WriterClosed)?
                .write(&subjects_batch)
        {
            self.poisoned = true;
            return Err(error.into());
        }

        for hand_id in batch_hand_ids {
            self.seen_hand_ids.insert(hand_id);
        }
        for (player_id, &subject_id) in subject_ids {
            self.player_to_subject
                .entry(player_id.clone())
                .or_insert(subject_id);
            self.subject_to_player
                .entry(subject_id)
                .or_insert_with(|| player_id.clone());
        }

        self.summary = next_summary;
        self.next_hand_index = next_hand_index;

        Ok(BehaviourParquetWriteReport {
            hands_seen: hands.len(),
            hands_written: retained.len(),
            events_written,
            subject_hand_rows_written,
            dropped_hand_ids_missing_timing,
        })
    }

    pub fn summary(&self) -> BehaviourParquetDatasetSummary {
        self.summary
    }

    /// Finalizes both Parquet footers and commits the dataset.
    ///
    /// Files are written under unique `.inprogress` names and linked into their
    /// final names only after both writers close successfully. `_SUCCESS` is
    /// created last, so consumers can reject directories without it as incomplete.
    pub fn close(mut self) -> Result<BehaviourParquetDatasetSummary, BehaviourParquetDatasetError> {
        self.ensure_writable()?;
        append_final_metadata(
            self.events_writer
                .as_mut()
                .ok_or(BehaviourParquetDatasetError::WriterClosed)?,
            self.summary,
        );
        append_final_metadata(
            self.subjects_writer
                .as_mut()
                .ok_or(BehaviourParquetDatasetError::WriterClosed)?,
            self.summary,
        );

        if let Err(error) = finish_and_sync(
            self.events_writer
                .as_mut()
                .ok_or(BehaviourParquetDatasetError::WriterClosed)?,
        ) {
            self.poisoned = true;
            return Err(error);
        }
        if let Err(error) = finish_and_sync(
            self.subjects_writer
                .as_mut()
                .ok_or(BehaviourParquetDatasetError::WriterClosed)?,
        ) {
            self.poisoned = true;
            return Err(error);
        }
        self.events_writer.take();
        self.subjects_writer.take();

        fs::hard_link(&self.events_in_progress_path, &self.events_path)?;
        self.events_final_linked = true;
        fs::hard_link(&self.subjects_in_progress_path, &self.subjects_path)?;
        self.subjects_final_linked = true;
        fs::remove_file(&self.events_in_progress_path)?;
        fs::remove_file(&self.subjects_in_progress_path)?;
        let success_file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&self.success_path)?;
        self.success_created = true;
        success_file.sync_all()?;

        self.committed = true;
        Ok(self.summary)
    }

    fn ensure_writable(&self) -> Result<(), BehaviourParquetDatasetError> {
        if self.poisoned {
            return Err(BehaviourParquetDatasetError::WriterPoisoned);
        }
        if self.events_writer.is_none() || self.subjects_writer.is_none() {
            return Err(BehaviourParquetDatasetError::WriterClosed);
        }
        Ok(())
    }

    fn preflight_subject_ids(
        &self,
        subject_ids: &HashMap<PlayerId, u64>,
    ) -> Result<(), BehaviourParquetDatasetError> {
        let mut reverse = HashMap::<u64, &PlayerId>::with_capacity(subject_ids.len());
        for (player_id, &subject_id) in subject_ids {
            if let Some(existing) = reverse.insert(subject_id, player_id)
                && existing != player_id
            {
                return Err(BehaviourParquetDatasetError::DuplicateSubjectId {
                    subject_id,
                    first_player_id: (*existing).clone(),
                    second_player_id: player_id.clone(),
                });
            }
            self.validate_registered_subject(player_id, subject_id)?;
        }
        Ok(())
    }

    fn validate_registered_subject(
        &self,
        player_id: &PlayerId,
        subject_id: u64,
    ) -> Result<(), BehaviourParquetDatasetError> {
        if let Some(&registered) = self.player_to_subject.get(player_id)
            && registered != subject_id
        {
            return Err(BehaviourParquetDatasetError::PlayerRemapped {
                player_id: player_id.clone(),
                registered_subject_id: registered,
                supplied_subject_id: subject_id,
            });
        }
        if let Some(registered_player) = self.subject_to_player.get(&subject_id)
            && registered_player != player_id
        {
            return Err(BehaviourParquetDatasetError::DuplicateSubjectId {
                subject_id,
                first_player_id: (*registered_player).clone(),
                second_player_id: player_id.clone(),
            });
        }
        Ok(())
    }

    fn cleanup_uncommitted_files(&self) {
        let _ = fs::remove_file(&self.events_in_progress_path);
        let _ = fs::remove_file(&self.subjects_in_progress_path);
        if self.events_final_linked {
            let _ = fs::remove_file(&self.events_path);
        }
        if self.subjects_final_linked {
            let _ = fs::remove_file(&self.subjects_path);
        }
        if self.success_created {
            let _ = fs::remove_file(&self.success_path);
        }
    }
}

impl Drop for BehaviourParquetDatasetWriter {
    fn drop(&mut self) {
        if self.committed {
            return;
        }
        self.events_writer.take();
        self.subjects_writer.take();
        self.cleanup_uncommitted_files();
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BehaviourParquetDatasetSummary {
    pub hands_seen: u64,
    pub hands_written: u64,
    pub events_written: u64,
    pub subject_hand_rows_written: u64,
    pub dropped_hands_missing_timing: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BehaviourParquetWriteReport {
    pub hands_seen: usize,
    pub hands_written: usize,
    pub events_written: usize,
    pub subject_hand_rows_written: usize,
    pub dropped_hand_ids_missing_timing: Vec<HandId>,
}

fn finish_and_sync(writer: &mut ArrowWriter<File>) -> Result<(), BehaviourParquetDatasetError> {
    writer.finish()?;
    writer.inner().sync_all()?;
    Ok(())
}

fn writer_properties(table: &str) -> WriterProperties {
    let mut metadata = vec![
        KeyValue::new("lechiffre.table".to_string(), table.to_string()),
        KeyValue::new(
            "lechiffre.position_codebook".to_string(),
            POSITION_CODEBOOK.to_string(),
        ),
        KeyValue::new(
            "lechiffre.hand_index_semantics".to_string(),
            "monotonic_input_order_metadata_not_model_feature; gaps_correspond_to_dropped_hands"
                .to_string(),
        ),
    ];

    if table == "events" {
        metadata.extend([
            KeyValue::new(
                "lechiffre.street_codebook".to_string(),
                STREET_CODEBOOK.to_string(),
            ),
            KeyValue::new(
                "lechiffre.action_codebook".to_string(),
                ACTION_CODEBOOK.to_string(),
            ),
        ]);
    } else {
        metadata.push(KeyValue::new(
            "lechiffre.subject_id_semantics".to_string(),
            "pseudonymous_grouping_metadata_not_model_feature".to_string(),
        ));
    }

    WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .set_created_by(format!("LeChiffre version {}", env!("CARGO_PKG_VERSION")))
        .set_key_value_metadata(Some(metadata))
        .build()
}

fn append_final_metadata(writer: &mut ArrowWriter<File>, summary: BehaviourParquetDatasetSummary) {
    for (key, value) in [
        ("lechiffre.hands_seen", summary.hands_seen),
        ("lechiffre.hands_written", summary.hands_written),
        ("lechiffre.events_written", summary.events_written),
        (
            "lechiffre.subject_hand_rows_written",
            summary.subject_hand_rows_written,
        ),
        (
            "lechiffre.dropped_hands_missing_timing",
            summary.dropped_hands_missing_timing,
        ),
    ] {
        writer.append_key_value_metadata(KeyValue::new(key.to_string(), value.to_string()));
    }
}

pub(crate) fn events_schema() -> SchemaRef {
    let mut fields = vec![
        Field::new("hand_id", DataType::Utf8, false),
        Field::new("hand_index", DataType::UInt64, false),
        Field::new("sequence_index", DataType::UInt32, false),
        Field::new("street", DataType::UInt8, false),
        Field::new("actor_position", DataType::UInt8, false),
        Field::new("action", DataType::UInt8, false),
        Field::new("is_all_in", DataType::Boolean, false),
        Field::new("players_at_table", DataType::UInt8, false),
        Field::new("decision_time_ms", DataType::UInt32, false),
    ];
    for index in 0..EVENT_CONTINUOUS_WIDTH {
        fields.push(Field::new(
            continuous_feature_name(index),
            DataType::Float32,
            true,
        ));
    }
    Arc::new(Schema::new(fields))
}

pub(crate) fn subjects_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("subject_id", DataType::UInt64, false),
        Field::new("hand_id", DataType::Utf8, false),
        Field::new("hand_index", DataType::UInt64, false),
        Field::new("subject_position", DataType::UInt8, false),
    ]))
}

fn event_rows<'a, 'h>(
    hands: &'a [(&'h AdaptedHand, u64)],
) -> impl Iterator<Item = (&'h BehaviourEvent, u64)> + 'a
where
    'h: 'a,
{
    hands.iter().flat_map(|(hand, hand_index)| {
        hand.sequence()
            .events()
            .iter()
            .map(move |event| (event, *hand_index))
    })
}

fn events_record_batch(
    hands: &[(&AdaptedHand, u64)],
) -> Result<RecordBatch, BehaviourParquetDatasetError> {
    let row_count = hands.iter().try_fold(0usize, |total, (hand, _)| {
        total
            .checked_add(hand.sequence().len())
            .ok_or(BehaviourParquetDatasetError::RowCountOverflow)
    })?;
    let players_at_table = event_rows(hands)
        .map(|(event, _)| checked_u8("players_at_table", event.context.players.len(), event))
        .collect::<Result<Vec<_>, _>>()?;
    let decision_time_ms = event_rows(hands)
        .map(|(event, _)| event.action.decision_time_ms)
        .collect::<Option<Vec<_>>>()
        .ok_or(BehaviourParquetDatasetError::InternalMissingTiming)?;
    let mut columns: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from_iter_values(
            event_rows(hands).map(|(e, _)| e.hand_id.as_str()),
        )),
        Arc::new(UInt64Array::from_iter_values(
            event_rows(hands).map(|(_, i)| i),
        )),
        Arc::new(UInt32Array::from_iter_values(
            event_rows(hands).map(|(e, _)| e.sequence_index),
        )),
        Arc::new(UInt8Array::from_iter_values(
            event_rows(hands).map(|(e, _)| street_id(e.context.street)),
        )),
        Arc::new(UInt8Array::from_iter_values(
            event_rows(hands).map(|(e, _)| position_id(e.context.actor_position)),
        )),
        Arc::new(UInt8Array::from_iter_values(
            event_rows(hands).map(|(e, _)| action_id(e.action.kind)),
        )),
        Arc::new(BooleanArray::from(
            event_rows(hands)
                .map(|(e, _)| e.action.is_all_in)
                .collect::<Vec<_>>(),
        )),
        Arc::new(UInt8Array::from(players_at_table)),
        Arc::new(UInt32Array::from(decision_time_ms)),
    ];
    let encoded = event_rows(hands)
        .map(|(event, _)| encode_continuous(event))
        .collect::<Vec<_>>();
    for feature in 0..EVENT_CONTINUOUS_WIDTH {
        let values = encoded
            .iter()
            .map(|(values, mask)| {
                if mask[feature] == 1 {
                    Some(values[feature])
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        columns.push(Arc::new(Float32Array::from(values)));
    }
    let batch = RecordBatch::try_new(events_schema(), columns)?;
    if batch.num_rows() != row_count {
        return Err(BehaviourParquetDatasetError::InternalRowCountMismatch {
            expected: row_count,
            observed: batch.num_rows(),
        });
    }
    Ok(batch)
}

fn subjects_record_batch(
    hands: &[(&AdaptedHand, u64)],
    subject_ids: &HashMap<PlayerId, u64>,
) -> Result<RecordBatch, BehaviourParquetDatasetError> {
    let row_count = hands.iter().try_fold(0usize, |total, (hand, _)| {
        total
            .checked_add(hand.players().len())
            .ok_or(BehaviourParquetDatasetError::RowCountOverflow)
    })?;

    let ids = hands
        .iter()
        .flat_map(|(hand, _)| hand.players().iter())
        .map(|player| {
            subject_ids.get(&player.player_id).copied().ok_or_else(|| {
                BehaviourParquetDatasetError::InternalMissingSubjectId {
                    player_id: player.player_id.clone(),
                }
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    let columns: Vec<ArrayRef> = vec![
        Arc::new(UInt64Array::from(ids)),
        Arc::new(StringArray::from_iter_values(hands.iter().flat_map(
            |(hand, _)| {
                std::iter::repeat_n(hand.sequence().hand_id().as_str(), hand.players().len())
            },
        ))),
        Arc::new(UInt64Array::from_iter_values(hands.iter().flat_map(
            |(hand, hand_index)| std::iter::repeat_n(*hand_index, hand.players().len()),
        ))),
        Arc::new(UInt8Array::from_iter_values(hands.iter().flat_map(
            |(hand, _)| {
                hand.players()
                    .iter()
                    .map(|player| position_id(player.position))
            },
        ))),
    ];

    let batch = RecordBatch::try_new(subjects_schema(), columns)?;
    if batch.num_rows() != row_count {
        return Err(BehaviourParquetDatasetError::InternalRowCountMismatch {
            expected: row_count,
            observed: batch.num_rows(),
        });
    }
    Ok(batch)
}

fn validate_hand_structure(hand: &AdaptedHand) -> Result<(), BehaviourParquetDatasetError> {
    let hand_id = hand.sequence().hand_id().clone();
    if !(2..=6).contains(&hand.players().len()) {
        return Err(BehaviourParquetDatasetError::UnsupportedPlayerCount {
            hand_id,
            players: hand.players().len(),
        });
    }

    let mut player_ids = HashSet::with_capacity(hand.players().len());
    let mut positions = HashSet::with_capacity(hand.players().len());
    for player in hand.players() {
        if !player_ids.insert(player.player_id.clone()) {
            return Err(BehaviourParquetDatasetError::DuplicatePlayerInHand {
                hand_id: hand.sequence().hand_id().clone(),
                player_id: player.player_id.clone(),
            });
        }
        if !positions.insert(player.position) {
            return Err(BehaviourParquetDatasetError::DuplicatePositionInHand {
                hand_id: hand.sequence().hand_id().clone(),
                position: player.position,
            });
        }
    }

    for event in hand.sequence().events() {
        event
            .context
            .validate()
            .map_err(BehaviourParquetDatasetError::InvalidEvent)?;
        event
            .action
            .validate_with_context(&event.context)
            .map_err(BehaviourParquetDatasetError::InvalidEvent)?;
        if event.context.players.len() != hand.players().len() {
            return Err(BehaviourParquetDatasetError::PlayerCountMismatch {
                hand_id: hand.sequence().hand_id().clone(),
                sequence_index: event.sequence_index,
                observed_players: hand.players().len(),
                context_players: event.context.players.len(),
            });
        }
        if !positions.contains(&event.context.actor_position) {
            return Err(BehaviourParquetDatasetError::UnknownActorPosition {
                hand_id: hand.sequence().hand_id().clone(),
                sequence_index: event.sequence_index,
                position: event.context.actor_position,
            });
        }
        for state in &event.context.players {
            if !positions.contains(&state.position) {
                return Err(BehaviourParquetDatasetError::UnknownActorPosition {
                    hand_id: hand.sequence().hand_id().clone(),
                    sequence_index: event.sequence_index,
                    position: state.position,
                });
            }
        }
        checked_u8("players_at_table", event.context.players.len(), event)?;
    }

    Ok(())
}

fn checked_counter_add(
    current: u64,
    delta: u64,
    field: &'static str,
) -> Result<u64, BehaviourParquetDatasetError> {
    current
        .checked_add(delta)
        .ok_or(BehaviourParquetDatasetError::CounterOverflow { field })
}

fn checked_u8(
    field: &'static str,
    value: usize,
    event: &BehaviourEvent,
) -> Result<u8, BehaviourParquetDatasetError> {
    u8::try_from(value).map_err(|_| BehaviourParquetDatasetError::IntegerOverflow {
        hand_id: event.hand_id.clone(),
        sequence_index: event.sequence_index,
        field,
        value,
    })
}

#[derive(Debug)]
pub enum BehaviourParquetDatasetError {
    Io(io::Error),
    Arrow(ArrowError),
    Parquet(ParquetError),
    OutputAlreadyExists {
        path: PathBuf,
    },
    WriterPoisoned,
    WriterClosed,
    HandIndexOverflow,
    RowCountOverflow,
    CounterOverflow {
        field: &'static str,
    },
    InternalMissingTiming,
    InvalidEvent(crate::behavioural::BehaviourEventError),
    InternalMissingSubjectId {
        player_id: PlayerId,
    },
    InternalRowCountMismatch {
        expected: usize,
        observed: usize,
    },
    DuplicateHandId {
        hand_id: HandId,
    },
    EmptyHand {
        hand_id: HandId,
    },
    MissingSubjectId {
        player_id: PlayerId,
        hand_id: HandId,
    },
    DuplicateSubjectId {
        subject_id: u64,
        first_player_id: PlayerId,
        second_player_id: PlayerId,
    },
    PlayerRemapped {
        player_id: PlayerId,
        registered_subject_id: u64,
        supplied_subject_id: u64,
    },
    UnsupportedPlayerCount {
        hand_id: HandId,
        players: usize,
    },
    DuplicatePlayerInHand {
        hand_id: HandId,
        player_id: PlayerId,
    },
    DuplicatePositionInHand {
        hand_id: HandId,
        position: crate::behavioural::Position,
    },
    PlayerCountMismatch {
        hand_id: HandId,
        sequence_index: u32,
        observed_players: usize,
        context_players: usize,
    },
    UnknownActorPosition {
        hand_id: HandId,
        sequence_index: u32,
        position: crate::behavioural::Position,
    },
    IntegerOverflow {
        hand_id: HandId,
        sequence_index: u32,
        field: &'static str,
        value: usize,
    },
}

impl fmt::Display for BehaviourParquetDatasetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(f, "Parquet dataset I/O failed: {error}"),
            Self::Arrow(error) => write!(f, "Arrow record construction failed: {error}"),
            Self::Parquet(error) => write!(f, "Parquet writing failed: {error}"),
            Self::OutputAlreadyExists { path } => {
                write!(
                    f,
                    "refusing to overwrite existing dataset path {}",
                    path.display()
                )
            }
            Self::WriterPoisoned => write!(
                f,
                "Parquet dataset writer is poisoned after a write failure"
            ),
            Self::WriterClosed => write!(f, "Parquet dataset writer is already closed"),
            Self::HandIndexOverflow => write!(f, "dataset hand index overflow"),
            Self::RowCountOverflow => write!(f, "dataset row count overflow"),
            Self::CounterOverflow { field } => {
                write!(f, "dataset counter {field} overflow")
            }
            Self::InvalidEvent(error) => write!(f, "invalid behavioural event: {error}"),
            Self::InternalMissingTiming => write!(
                f,
                "internal error: retained hand lost required decision timing during batch construction"
            ),
            Self::InternalMissingSubjectId { player_id } => write!(
                f,
                "internal error: retained player {player_id} lost its subject id during batch construction"
            ),
            Self::InternalRowCountMismatch { expected, observed } => write!(
                f,
                "internal error: Arrow batch row count mismatch, expected {expected}, got {observed}"
            ),
            Self::DuplicateHandId { hand_id } => write!(f, "duplicate hand id {hand_id}"),
            Self::EmptyHand { hand_id } => write!(f, "cannot persist empty hand {hand_id}"),
            Self::MissingSubjectId { player_id, hand_id } => write!(
                f,
                "missing pseudonymous subject id for player {player_id} in hand {hand_id}"
            ),
            Self::DuplicateSubjectId {
                subject_id,
                first_player_id,
                second_player_id,
            } => write!(
                f,
                "subject id {subject_id} is assigned to both {first_player_id} and {second_player_id}"
            ),
            Self::PlayerRemapped {
                player_id,
                registered_subject_id,
                supplied_subject_id,
            } => write!(
                f,
                "player {player_id} was already mapped to subject {registered_subject_id}, not {supplied_subject_id}"
            ),
            Self::UnsupportedPlayerCount { hand_id, players } => write!(
                f,
                "hand {hand_id} has {players} players; behavioural dataset supports only 2..=6-max"
            ),
            Self::DuplicatePlayerInHand { hand_id, player_id } => {
                write!(
                    f,
                    "hand {hand_id} contains duplicate player identity {player_id}"
                )
            }
            Self::DuplicatePositionInHand { hand_id, position } => {
                write!(f, "hand {hand_id} contains duplicate position {position:?}")
            }
            Self::PlayerCountMismatch {
                hand_id,
                sequence_index,
                observed_players,
                context_players,
            } => write!(
                f,
                "hand {hand_id} event {sequence_index} reports {context_players} table players but adapter exposes {observed_players}"
            ),
            Self::UnknownActorPosition {
                hand_id,
                sequence_index,
                position,
            } => write!(
                f,
                "hand {hand_id} event {sequence_index} actor position {position:?} is not present in the adapted player list"
            ),
            Self::IntegerOverflow {
                hand_id,
                sequence_index,
                field,
                value,
            } => write!(
                f,
                "hand {hand_id} event {sequence_index} field {field} value {value} does not fit the Parquet UInt8 schema"
            ),
        }
    }
}

impl std::error::Error for BehaviourParquetDatasetError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Arrow(error) => Some(error),
            Self::Parquet(error) => Some(error),
            Self::InvalidEvent(error) => Some(error),
            _ => None,
        }
    }
}

impl From<io::Error> for BehaviourParquetDatasetError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<ArrowError> for BehaviourParquetDatasetError {
    fn from(error: ArrowError) -> Self {
        Self::Arrow(error)
    }
}

impl From<ParquetError> for BehaviourParquetDatasetError {
    fn from(error: ParquetError) -> Self {
        Self::Parquet(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapters::ObservedPlayer;
    use crate::behavioural::{
        ActionHistory, ActionKind, BehaviourEvent, BehaviourSequence, DecisionContext, ExactAction,
        PlayerState, PlayerStatus, Position, Street,
    };
    use arrow_array::UInt32Array;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn temp_dataset_dir(name: &str) -> PathBuf {
        let id = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "lechiffre-parquet-{name}-{}-{id}",
            std::process::id()
        ))
    }
    fn make_hand(hand_id: &str, timing: Option<u32>) -> AdaptedHand {
        let mut history = ActionHistory::new();
        let mut sequence = BehaviourSequence::new(hand_id);
        for (
            index,
            (street, position, kind, pot, stack, street_bet, hand_bet, highest, added, time),
        ) in [
            (
                Street::Preflop,
                Position::ButtonSmallBlind,
                ActionKind::Call,
                1.5,
                99.5,
                0.5,
                0.5,
                1.0,
                0.5,
                timing,
            ),
            (
                Street::Flop,
                Position::BigBlind,
                ActionKind::Check,
                2.0,
                99.0,
                0.0,
                1.0,
                0.0,
                0.0,
                Some(320),
            ),
        ]
        .into_iter()
        .enumerate()
        {
            let other = if position == Position::ButtonSmallBlind {
                Position::BigBlind
            } else {
                Position::ButtonSmallBlind
            };
            let mut ctx = DecisionContext {
                street,
                actor_position: position,
                pot_bb: pot,
                players: vec![
                    PlayerState {
                        position,
                        status: PlayerStatus::Active,
                        stack_remaining_bb: stack,
                        committed_this_street_bb: street_bet,
                        committed_this_hand_bb: hand_bet,
                    },
                    PlayerState {
                        position: other,
                        status: PlayerStatus::Active,
                        stack_remaining_bb: 99.0,
                        committed_this_street_bb: highest,
                        committed_this_hand_bb: 1.0,
                    },
                ],
                current_bet_bb: highest,
                to_call_bb: (highest - street_bet).max(0.0),
                min_raise_to_bb: Some(2.0),
                players_yet_to_act: 1,
                bets_and_raises_this_street: 0,
                calls_this_street: 0,
                folds_this_street: 0,
                actor_has_acted_this_street: false,
                actor_aggression_streak_before: 0,
                table_aggression_streak_before: 0,
                last_aggressor: None,
                preflop_last_aggressor_position: None,
            };
            history.prepare(&mut ctx);
            let response = history
                .record(
                    &ctx,
                    ExactAction {
                        kind,
                        amount_added_bb: added,
                        highest_bet_after_bb: highest,
                        is_all_in: false,
                        decision_time_ms: time,
                    },
                )
                .unwrap();
            sequence
                .push(BehaviourEvent::new(hand_id, index as u32, ctx, response).unwrap())
                .unwrap();
        }
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
    fn subject_ids() -> HashMap<PlayerId, u64> {
        HashMap::from([
            (PlayerId::new("test:alice"), 17),
            (PlayerId::new("test:bob"), 42),
        ])
    }
    #[test]
    fn roundtrip_uses_single_observation_contract() {
        let dir = temp_dataset_dir("roundtrip");
        let hand = make_hand("hand-a", Some(16_777_217));
        let mut writer = BehaviourParquetDatasetWriter::create(&dir).unwrap();
        let report = writer.write_hands(&[hand], &subject_ids()).unwrap();
        writer.close().unwrap();
        assert_eq!(report.events_written, 2);
        let loaded =
            crate::datasets::behaviour_parquet_loader::BehaviourParquetDataset::open(&dir).unwrap();
        assert_eq!(loaded.hand_count(), 1);
        assert_eq!(loaded.event_count(), 2);
        let subject = loaded.subject_history(17).unwrap();
        assert_eq!(subject.hands().len(), 1);
        assert_eq!(subject.hands()[0].sequence_len(), 2);
        let reader = ParquetRecordBatchReaderBuilder::try_new(
            File::open(dir.join(EVENTS_FILE_NAME)).unwrap(),
        )
        .unwrap()
        .build()
        .unwrap();
        let batch = reader.map(|b| b.unwrap()).next().unwrap();
        assert_eq!(batch.num_columns(), 9 + EVENT_CONTINUOUS_WIDTH);
        assert!(
            batch
                .schema()
                .field_with_name("actor_stack_remaining_bb")
                .is_ok()
        );
        assert!(batch.schema().field_with_name("board_0_rank").is_err());
        let times = batch
            .column(8)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();
        assert_eq!(times.value(0), 16_777_217);
        fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn missing_timing_drops_entire_hand() {
        let dir = temp_dataset_dir("timing");
        let mut writer = BehaviourParquetDatasetWriter::create(&dir).unwrap();
        let report = writer
            .write_hands(&[make_hand("hand-no-time", None)], &subject_ids())
            .unwrap();
        assert_eq!(report.hands_written, 0);
        writer.close().unwrap();
        fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn duplicate_hand_is_rejected() {
        let dir = temp_dataset_dir("duplicate");
        let mut writer = BehaviourParquetDatasetWriter::create(&dir).unwrap();
        let hand = make_hand("hand-dup", Some(1));
        writer
            .write_hands(std::slice::from_ref(&hand), &subject_ids())
            .unwrap();
        assert!(matches!(
            writer.write_hands(&[hand], &subject_ids()),
            Err(BehaviourParquetDatasetError::DuplicateHandId { .. })
        ));
        drop(writer);
        fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn retained_hand_order_skips_timing_gaps_without_renumbering() {
        let dir = temp_dataset_dir("ordering");
        let mut writer = BehaviourParquetDatasetWriter::create(&dir).unwrap();
        let hands = [
            make_hand("first", Some(100)),
            make_hand("missing", None),
            make_hand("third", Some(300)),
        ];
        let report = writer.write_hands(&hands, &subject_ids()).unwrap();
        assert_eq!(report.hands_written, 2);
        assert_eq!(
            report.dropped_hand_ids_missing_timing,
            vec![HandId::new("missing")]
        );
        writer.close().unwrap();
        let loaded =
            crate::datasets::behaviour_parquet_loader::BehaviourParquetDataset::open(&dir).unwrap();
        let hands = loaded.subject_history(17).unwrap();
        assert_eq!(hands.hands().len(), 2);
        assert_eq!(hands.hands()[0].hand_id().as_str(), "first");
        assert_eq!(hands.hands()[1].hand_id().as_str(), "third");
        fs::remove_dir_all(dir).unwrap();
    }
}
