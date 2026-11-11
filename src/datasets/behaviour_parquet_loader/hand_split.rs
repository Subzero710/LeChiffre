//! Deterministic ratio-based partitioning of complete retained hands.
//!
//! `hand_index` is ingestion order, NOT a verified wall-clock timestamp.
//! Each persisted hand, including every one of its decisions and subjects,
//! belongs to exactly one partition. Nothing is split by event row.
use std::collections::BTreeMap;
use std::fmt;

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HandSplitError {
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
                f, "hand split event count mismatch: expected {expected}, observed {observed}"
            ),
            Self::SubjectRowsMismatch { expected, observed } => write!(
                f, "hand split subject rows mismatch: expected {expected}, observed {observed}"
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
        let train_n =
            (n as u128 * u128::from(train_bps) / u128::from(SPLIT_BASIS_POINTS)) as usize;
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
    use super::*;
    use crate::behavioural::{HandId, Position, EVENT_CONTINUOUS_WIDTH};
    use crate::datasets::behaviour_parquet::BehaviourParquetDatasetSummary;
    use super::super::{PersistedHand, SubjectMembership};
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
            (11_u64, (0..n).map(|hand_slot| SubjectMembership {
                hand_slot,
                subject_position: Position::ButtonSmallBlind,
            }).collect::<Vec<_>>()),
            (22_u64, (0..n).map(|hand_slot| SubjectMembership {
                hand_slot,
                subject_position: Position::BigBlind,
            }).collect::<Vec<_>>()),
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
        let ids = all.iter().map(|e| e.hand_id.as_str()).collect::<HashSet<_>>();
        assert_eq!(ids.len(), 20);
        assert!(all.iter().all(|e| e.decision_count == 2));
        assert_eq!(
            ds.split_hands_by_ingestion(7_000, 1_500, 1_500)
                .unwrap(),
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
