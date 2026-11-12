//! Deterministic ratio-based partitioning of complete retained hands.
//!
//! `hand_index` is ingestion order, NOT a verified wall-clock timestamp.
//! Each persisted hand, including every one of its decisions and subjects,
//! belongs to exactly one partition. Nothing is split by event row.
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::fmt;
use std::path::Path;

use super::{BehaviourParquetDataset, SPLIT_BASIS_POINTS};

pub const HAND_SPLIT_ALGORITHM_VERSION: u16 = 1;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HandPartitionEntry {
    pub hand_id: String,
    pub hand_index: u64,
    pub decision_count: u64,
    pub players_at_table: u8,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IngestionHandSplit {
    train: Vec<HandPartitionEntry>,
    validation: Vec<HandPartitionEntry>,
    test: Vec<HandPartitionEntry>,
    first_validation_hand_index: u64,
    first_test_hand_index: u64,
    decision_counts: [u64; 3],
    subject_hand_rows: [u64; 3],
    subject_hand_counts: BTreeMap<u64, [u64; 3]>,
}

impl IngestionHandSplit {
    /// Stable snapshot of the original v1 JSON schema, including its historical
    /// integration-warning field. Existing v1 manifests need no regeneration.
    pub fn manifest_value(&self, dataset: &BehaviourParquetDataset) -> Value {
        let to_entries = |hands: &[HandPartitionEntry]| {
            hands
                .iter()
                .map(|entry| {
                    json!({
                        "hand_id": entry.hand_id,
                        "hand_index": entry.hand_index,
                        "decisions": entry.decision_count,
                        "players_at_table": entry.players_at_table,
                    })
                })
                .collect::<Vec<_>>()
        };
        let subject_counts = self
            .subject_hand_counts
            .iter()
            .map(|(&subject_id, &counts)| {
                json!({"subject_id": subject_id, "train_hands": counts[0],
                   "validation_hands": counts[1], "test_hands": counts[2]})
            })
            .collect::<Vec<_>>();
        let stats = dataset.summary();
        json!({
            "format": "lechiffre.ingestion-hand-split.v1",
            "algorithm_version": HAND_SPLIT_ALGORITHM_VERSION,
            "ordering": "persisted_hand_index_ingestion_order_not_verified_wall_clock_chronology",
            "ratios_basis_points": {"train": 7000, "validation": 1500, "test": 1500},
            "rounding": "floor_train_and_validation_remainder_to_test",
            "dataset": {
                "hands": stats.hands_written, "decisions": stats.events_written,
                "subject_hand_rows": stats.subject_hand_rows_written,
                "fingerprint_fnv128_noncryptographic": dataset.dataset_fingerprint(),
            },
            "boundaries": {
                "first_validation_hand_index": self.first_validation_hand_index,
                "first_test_hand_index": self.first_test_hand_index,
            },
            "hierarchical_training_integration": "not_compatible_query_partition_is_used_for_training",
            "partitions": {
                "train": {"hands": self.train.len(), "decisions": self.decision_counts[0],
                          "subject_hand_rows": self.subject_hand_rows[0], "entries": to_entries(&self.train)},
                "validation": {"hands": self.validation.len(), "decisions": self.decision_counts[1],
                               "subject_hand_rows": self.subject_hand_rows[1], "entries": to_entries(&self.validation)},
                "test": {"hands": self.test.len(), "decisions": self.decision_counts[2],
                           "subject_hand_rows": self.subject_hand_rows[2], "entries": to_entries(&self.test)},
            },
            "subjects": subject_counts,
        })
    }

    /// Every field of the supplied manifest must match the committed Parquet.
    /// No defaults, ignored entries, re-partitioning or silent migration.
    pub fn read_manifest(
        dataset: &BehaviourParquetDataset,
        path: impl AsRef<Path>,
    ) -> Result<Self, HandSplitError> {
        let file = std::fs::File::open(path)
            .map_err(|e| HandSplitError::InvalidManifest(e.to_string()))?;
        let observed: Value = serde_json::from_reader(file)
            .map_err(|e| HandSplitError::InvalidManifest(e.to_string()))?;
        let expected = dataset.split_hands_by_ingestion(7_000, 1_500, 1_500)?;
        if observed != expected.manifest_value(dataset) {
            return Err(HandSplitError::InvalidManifest(
                "manifest differs from the complete canonical v1 split for this Parquet".into(),
            ));
        }
        Ok(expected)
    }

    pub fn manifest_fingerprint(&self, dataset: &BehaviourParquetDataset) -> String {
        let payload = self.manifest_value(dataset).to_string();
        let mut hash = 0x6c62_272e_07bb_0142_62b8_2175_6295_c58d_u128;
        const PRIME: u128 = 0x0000_0000_0100_0000_0000_0000_0000_013b;
        for byte in payload.bytes() {
            hash = (hash ^ u128::from(byte)).wrapping_mul(PRIME);
        }
        format!("{hash:032x}")
    }

    pub fn contains_train(&self, hand_index: u64) -> bool {
        self.train
            .binary_search_by_key(&hand_index, |hand| hand.hand_index)
            .is_ok()
    }

    pub fn contains_evaluation(&self, partition: EvaluationPartition, hand_index: u64) -> bool {
        let entries = match partition {
            EvaluationPartition::Validation => &self.validation,
            EvaluationPartition::Test => &self.test,
        };
        entries
            .binary_search_by_key(&hand_index, |hand| hand.hand_index)
            .is_ok()
    }

    pub fn train(&self) -> &[HandPartitionEntry] {
        &self.train
    }

    pub fn validation(&self) -> &[HandPartitionEntry] {
        &self.validation
    }

    pub fn test(&self) -> &[HandPartitionEntry] {
        &self.test
    }

    /// Earliest retained validation hand's original (possibly gapped) index.
    pub fn first_validation_hand_index(&self) -> u64 {
        self.first_validation_hand_index
    }

    /// Earliest retained test hand's original (possibly gapped) index.
    pub fn first_test_hand_index(&self) -> u64 {
        self.first_test_hand_index
    }

    /// Event counts for train, validation and test, in that order.
    pub fn decision_counts(&self) -> [u64; 3] {
        self.decision_counts
    }

    /// Subject-to-hand membership row counts in the same order.
    pub fn subject_hand_rows(&self) -> [u64; 3] {
        self.subject_hand_rows
    }

    /// Number of retained hands available per subject in each partition.
    pub fn subject_hand_counts(&self) -> &BTreeMap<u64, [u64; 3]> {
        &self.subject_hand_counts
    }
}

/// Proof of an exact v1 manifest/Parquet match. There is no public unchecked
/// constructor; training code accepts this verified object, not a raw split.
#[derive(Debug, Clone)]
pub struct DatasetPartitions {
    split: IngestionHandSplit,
    dataset_fingerprint: String,
    manifest_fingerprint: String,
}

impl DatasetPartitions {
    pub fn read_manifest(
        dataset: &BehaviourParquetDataset,
        path: impl AsRef<Path>,
    ) -> Result<Self, HandSplitError> {
        let split = IngestionHandSplit::read_manifest(dataset, path)?;
        Ok(Self {
            dataset_fingerprint: dataset.dataset_fingerprint(),
            manifest_fingerprint: split.manifest_fingerprint(dataset),
            split,
        })
    }

    pub fn verify_dataset(&self, dataset: &BehaviourParquetDataset) -> Result<(), HandSplitError> {
        if self.dataset_fingerprint != dataset.dataset_fingerprint()
            || self.split != dataset.split_hands_by_ingestion(7_000, 1_500, 1_500)?
        {
            return Err(HandSplitError::InvalidManifest(
                "verified partitions do not belong to this dataset".into(),
            ));
        }
        Ok(())
    }

    pub fn contains_train(&self, index: u64) -> bool {
        self.split.contains_train(index)
    }

    pub fn contains_evaluation(&self, partition: EvaluationPartition, index: u64) -> bool {
        self.split.contains_evaluation(partition, index)
    }

    pub fn manifest_fingerprint(&self) -> &str {
        &self.manifest_fingerprint
    }

    #[cfg(test)]
    pub(crate) fn canonical_for_test(dataset: &BehaviourParquetDataset) -> Self {
        let split = dataset
            .split_hands_by_ingestion(7_000, 1_500, 1_500)
            .unwrap();
        Self {
            dataset_fingerprint: dataset.dataset_fingerprint(),
            manifest_fingerprint: split.manifest_fingerprint(dataset),
            split,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvaluationPartition {
    Validation,
    Test,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HandSplitError {
    InvalidManifest(String),
    InvalidRatios {
        train_bps: u16,
        validation_bps: u16,
        test_bps: u16,
    },
    EmptyPartition {
        hands: usize,
        train: usize,
        validation: usize,
        test: usize,
    },
    CountOverflow,
    EventCountMismatch {
        expected: u64,
        observed: u64,
    },
    SubjectRowsMismatch {
        expected: u64,
        observed: u64,
    },
}

impl fmt::Display for HandSplitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidManifest(message) => write!(f, "invalid split manifest: {message}"),
            Self::InvalidRatios {
                train_bps,
                validation_bps,
                test_bps,
            } => write!(
                f,
                "hand split needs three positive ratios totaling {SPLIT_BASIS_POINTS} bps; got {train_bps}/{validation_bps}/{test_bps}"
            ),
            Self::EmptyPartition {
                hands,
                train,
                validation,
                test,
            } => write!(
                f,
                "hand split cannot create three nonempty partitions from {hands} hands: {train}/{validation}/{test}"
            ),
            Self::CountOverflow => write!(f, "hand split count overflow"),
            Self::EventCountMismatch { expected, observed } => write!(
                f,
                "hand split event count mismatch: expected {expected}, observed {observed}"
            ),
            Self::SubjectRowsMismatch { expected, observed } => write!(
                f,
                "hand split subject rows mismatch: expected {expected}, observed {observed}"
            ),
        }
    }
}

impl std::error::Error for HandSplitError {}

impl BehaviourParquetDataset {
    /// Splits *retained complete hands* in persisted ingestion order.
    ///
    /// Both leading partition sizes use integer floor. Any remainder goes to
    /// the test partition. This is intentionally independent of decision counts.
    /// The ordering is not claimed to be chronological without source-time proof.
    pub fn split_hands_by_ingestion(
        &self,
        train_bps: u16,
        validation_bps: u16,
        test_bps: u16,
    ) -> Result<IngestionHandSplit, HandSplitError> {
        let total = u32::from(train_bps) + u32::from(validation_bps) + u32::from(test_bps);
        if train_bps == 0
            || validation_bps == 0
            || test_bps == 0
            || total != u32::from(SPLIT_BASIS_POINTS)
        {
            return Err(HandSplitError::InvalidRatios {
                train_bps,
                validation_bps,
                test_bps,
            });
        }

        let n = self.hands.len();
        let train_n = (n as u128 * u128::from(train_bps) / u128::from(SPLIT_BASIS_POINTS)) as usize;
        let validation_n =
            (n as u128 * u128::from(validation_bps) / u128::from(SPLIT_BASIS_POINTS)) as usize;
        let test_n = n - train_n - validation_n;
        if train_n == 0 || validation_n == 0 || test_n == 0 {
            return Err(HandSplitError::EmptyPartition {
                hands: n,
                train: train_n,
                validation: validation_n,
                test: test_n,
            });
        }

        // These are original, non-renumbered indexes, even when source hands
        // were rejected and gaps remain in the persisted sequence.
        let first_validation_hand_index = self.hands[train_n].hand_index;
        let first_test_hand_index = self.hands[train_n + validation_n].hand_index;
        let mut train = Vec::with_capacity(train_n);
        let mut validation = Vec::with_capacity(validation_n);
        let mut test = Vec::with_capacity(test_n);
        let mut decision_counts = [0_u64; 3];

        for (slot, hand) in self.hands.iter().enumerate() {
            let partition = if slot < train_n {
                0
            } else if slot < train_n + validation_n {
                1
            } else {
                2
            };
            let decision_count = u64::try_from(hand.sequence_indexes.len())
                .map_err(|_| HandSplitError::CountOverflow)?;
            decision_counts[partition] = decision_counts[partition]
                .checked_add(decision_count)
                .ok_or(HandSplitError::CountOverflow)?;
            let entry = HandPartitionEntry {
                hand_id: hand.hand_id.as_str().to_owned(),
                hand_index: hand.hand_index,
                decision_count,
                players_at_table: hand.players_at_table,
            };
            match partition {
                0 => train.push(entry),
                1 => validation.push(entry),
                _ => test.push(entry),
            }
        }

        let observed_events = decision_counts.iter().try_fold(0_u64, |sum, n| {
            sum.checked_add(*n).ok_or(HandSplitError::CountOverflow)
        })?;
        if observed_events != self.summary.events_written {
            return Err(HandSplitError::EventCountMismatch {
                expected: self.summary.events_written,
                observed: observed_events,
            });
        }

        let mut subject_hand_counts = BTreeMap::new();
        let mut subject_hand_rows = [0_u64; 3];
        for (&subject_id, memberships) in &self.subjects {
            let mut per_partition = [0_u64; 3];
            for membership in memberships {
                let slot = membership.hand_slot;
                let partition = if slot < train_n {
                    0
                } else if slot < train_n + validation_n {
                    1
                } else {
                    2
                };
                per_partition[partition] = per_partition[partition]
                    .checked_add(1)
                    .ok_or(HandSplitError::CountOverflow)?;
                subject_hand_rows[partition] = subject_hand_rows[partition]
                    .checked_add(1)
                    .ok_or(HandSplitError::CountOverflow)?;
            }
            subject_hand_counts.insert(subject_id, per_partition);
        }
        let observed_rows = subject_hand_rows.iter().try_fold(0_u64, |sum, n| {
            sum.checked_add(*n).ok_or(HandSplitError::CountOverflow)
        })?;
        if observed_rows != self.summary.subject_hand_rows_written {
            return Err(HandSplitError::SubjectRowsMismatch {
                expected: self.summary.subject_hand_rows_written,
                observed: observed_rows,
            });
        }

        Ok(IngestionHandSplit {
            train,
            validation,
            test,
            first_validation_hand_index,
            first_test_hand_index,
            decision_counts,
            subject_hand_rows,
            subject_hand_counts,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::{PersistedHand, SubjectMembership};
    use super::*;
    use crate::behavioural::{EVENT_CONTINUOUS_WIDTH, HandId, Position};
    use crate::datasets::behaviour_parquet::BehaviourParquetDatasetSummary;
    use std::collections::HashSet;

    fn fixture(n: usize) -> BehaviourParquetDataset {
        let hands = (0..n)
            .map(|i| PersistedHand {
                hand_id: HandId::new(format!("hand-{i}")),
                hand_index: (i * 2) as u64, // deliberate gaps, as in dropped source hands
                players_at_table: 2,
                sequence_indexes: vec![0, 1],
                categorical_without_subject: vec![[0, 4, 2, 0], [0, 6, 1, 0]],
                continuous: vec![[0.0; EVENT_CONTINUOUS_WIDTH]; 2],
                continuous_present: vec![[1; EVENT_CONTINUOUS_WIDTH]; 2],
                decision_time_ms: vec![0, 123], // zero-ms action stays in its hand
            })
            .collect::<Vec<_>>();
        let subjects = BTreeMap::from([
            (
                11_u64,
                (0..n)
                    .map(|hand_slot| SubjectMembership {
                        hand_slot,
                        subject_position: Position::ButtonSmallBlind,
                    })
                    .collect::<Vec<_>>(),
            ),
            (
                22_u64,
                (0..n)
                    .map(|hand_slot| SubjectMembership {
                        hand_slot,
                        subject_position: Position::BigBlind,
                    })
                    .collect::<Vec<_>>(),
            ),
        ]);
        BehaviourParquetDataset {
            summary: BehaviourParquetDatasetSummary {
                hands_seen: (n * 2) as u64,
                hands_written: n as u64,
                events_written: (n * 2) as u64,
                subject_hand_rows_written: (n * 2) as u64,
                dropped_hands_missing_timing: n as u64,
                dropped_hands_invalid: 0,
            },
            hands,
            subjects,
            event_count: n * 2,
        }
    }

    #[test]
    fn splits_complete_hands_and_all_subjects_without_leakage() {
        let ds = fixture(20);
        let result = ds.split_hands_by_ingestion(7_000, 1_500, 1_500).unwrap();
        assert_eq!(
            (
                result.train().len(),
                result.validation().len(),
                result.test().len()
            ),
            (14, 3, 3)
        );
        assert_eq!(result.decision_counts(), [28, 6, 6]);
        assert_eq!(result.subject_hand_rows(), [28, 6, 6]);
        assert_eq!(result.first_validation_hand_index(), 28);
        assert_eq!(result.first_test_hand_index(), 34);
        assert_eq!(result.subject_hand_counts()[&11], [14, 3, 3]);
        assert_eq!(result.subject_hand_counts()[&22], [14, 3, 3]);
        let all = result
            .train()
            .iter()
            .chain(result.validation())
            .chain(result.test())
            .collect::<Vec<_>>();
        assert_eq!(all.len(), 20);
        let ids = all
            .iter()
            .map(|e| e.hand_id.as_str())
            .collect::<HashSet<_>>();
        assert_eq!(ids.len(), 20);
        assert!(all.iter().all(|e| e.decision_count == 2));
        assert_eq!(
            ds.split_hands_by_ingestion(7_000, 1_500, 1_500).unwrap(),
            result
        );
    }

    #[test]
    fn exactly_matches_17611_hand_partition_sizes() {
        let ds = fixture(17_611);
        let result = ds.split_hands_by_ingestion(7_000, 1_500, 1_500).unwrap();
        assert_eq!(
            (
                result.train().len(),
                result.validation().len(),
                result.test().len()
            ),
            (12_327, 2_641, 2_643)
        );
        assert_eq!(result.decision_counts().iter().sum::<u64>(), 35_222);
    }

    #[test]
    fn rejects_invalid_ratios_and_unusable_small_partitions() {
        let ds = fixture(5);
        assert!(matches!(
            ds.split_hands_by_ingestion(7_000, 1_500, 0),
            Err(HandSplitError::InvalidRatios { .. })
        ));
        assert!(matches!(
            ds.split_hands_by_ingestion(7_000, 1_500, 1_499),
            Err(HandSplitError::InvalidRatios { .. })
        ));
        assert!(matches!(
            ds.split_hands_by_ingestion(7_000, 1_500, 1_500),
            Err(HandSplitError::EmptyPartition { .. })
        ));
    }

    #[test]
    fn strict_manifest_accepts_original_v1_and_rejects_tampering() {
        let ds = fixture(20);
        let split = ds.split_hands_by_ingestion(7_000, 1_500, 1_500).unwrap();
        let path = std::env::temp_dir().join(format!(
            "lechiffre-manifest-test-{}.json",
            std::process::id()
        ));
        let canonical = split.manifest_value(&ds);
        std::fs::write(&path, serde_json::to_vec(&canonical).unwrap()).unwrap();
        assert_eq!(
            IngestionHandSplit::read_manifest(&ds, &path).unwrap(),
            split
        );
        let verified = DatasetPartitions::read_manifest(&ds, &path).unwrap();
        assert!(verified.verify_dataset(&ds).is_ok());
        assert_eq!(
            verified.manifest_fingerprint(),
            split.manifest_fingerprint(&ds)
        );
        let mut tampered = canonical.clone();
        tampered["partitions"]["validation"]["entries"][0]["hand_id"] = json!("corrupt");
        std::fs::write(&path, serde_json::to_vec(&tampered).unwrap()).unwrap();
        assert!(IngestionHandSplit::read_manifest(&ds, &path).is_err());
        assert!(DatasetPartitions::read_manifest(&ds, &path).is_err());
        let mut incomplete = canonical;
        incomplete.as_object_mut().unwrap().remove("subjects");
        std::fs::write(&path, serde_json::to_vec(&incomplete).unwrap()).unwrap();
        assert!(IngestionHandSplit::read_manifest(&ds, &path).is_err());
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn global_partitions_are_disjoint_and_test_is_not_validation() {
        let ds = fixture(20);
        let split = ds.split_hands_by_ingestion(7_000, 1_500, 1_500).unwrap();
        assert!(split.contains_train(0));
        assert!(!split.contains_evaluation(EvaluationPartition::Validation, 0));
        assert!(!split.contains_train(28));
        assert!(split.contains_evaluation(EvaluationPartition::Validation, 28));
        assert!(!split.contains_evaluation(EvaluationPartition::Test, 28));
        assert!(split.contains_evaluation(EvaluationPartition::Test, 34));
    }

    #[test]
    fn refuses_corrupt_event_or_membership_totals() {
        let mut ds = fixture(20);
        ds.summary.events_written += 1;
        assert!(matches!(
            ds.split_hands_by_ingestion(7_000, 1_500, 1_500),
            Err(HandSplitError::EventCountMismatch { .. })
        ));
        ds.summary.events_written -= 1;
        ds.summary.subject_hand_rows_written += 1;
        assert!(matches!(
            ds.split_hands_by_ingestion(7_000, 1_500, 1_500),
            Err(HandSplitError::SubjectRowsMismatch { .. })
        ));
    }
}
