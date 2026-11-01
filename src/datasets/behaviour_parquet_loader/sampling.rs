use std::fmt;

use crate::behavioural::FeaturizedHand;

use super::BehaviourParquetDataset;

/// Version of the deterministic support/query hand sampling algorithm.
/// Persist this alongside training configuration when reproducibility matters.
pub const SUPPORT_QUERY_SAMPLING_ALGORITHM_VERSION: u16 = 1;

/// Exact support/query hand counts for one behavioural training episode.
///
/// Counts are explicit instead of inferred from ratios so the training layer
/// controls the amount of evidence available to the Analyzer and the amount of
/// held-out supervision independently.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SupportQueryConfig {
    support_hands: usize,
    query_hands: usize,
    seed: u64,
}

impl SupportQueryConfig {
    pub fn new(
        support_hands: usize,
        query_hands: usize,
        seed: u64,
    ) -> Result<Self, SupportQueryConfigError> {
        if support_hands == 0 {
            return Err(SupportQueryConfigError::ZeroSupportHands);
        }
        if query_hands == 0 {
            return Err(SupportQueryConfigError::ZeroQueryHands);
        }
        support_hands
            .checked_add(query_hands)
            .ok_or(SupportQueryConfigError::HandCountOverflow)?;

        Ok(Self {
            support_hands,
            query_hands,
            seed,
        })
    }

    pub fn support_hands(&self) -> usize {
        self.support_hands
    }

    pub fn query_hands(&self) -> usize {
        self.query_hands
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    pub fn total_hands(&self) -> usize {
        self.support_hands + self.query_hands
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SupportQueryConfigError {
    ZeroSupportHands,
    ZeroQueryHands,
    HandCountOverflow,
}

impl fmt::Display for SupportQueryConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroSupportHands => {
                write!(f, "support/query sampling requires support_hands > 0")
            }
            Self::ZeroQueryHands => write!(f, "support/query sampling requires query_hands > 0"),
            Self::HandCountOverflow => write!(f, "support/query hand-count sum overflow"),
        }
    }
}

impl std::error::Error for SupportQueryConfigError {}

/// One subject-conditioned training episode.
///
/// `subject_id` remains metadata used for grouping and sampling. It is not
/// copied into any [`FeaturizedHand`] feature. Support and query contain
/// distinct retained hands sampled without replacement.
#[derive(Debug, Clone, PartialEq)]
pub struct SupportQuerySample {
    subject_id: u64,
    support: Vec<FeaturizedHand>,
    query: Vec<FeaturizedHand>,
}

impl SupportQuerySample {
    pub fn subject_id(&self) -> u64 {
        self.subject_id
    }

    pub fn support(&self) -> &[FeaturizedHand] {
        &self.support
    }

    pub fn query(&self) -> &[FeaturizedHand] {
        &self.query
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SupportQuerySampleError {
    UnknownSubject {
        subject_id: u64,
    },
    InsufficientHands {
        subject_id: u64,
        required: usize,
        available: usize,
    },
}

impl fmt::Display for SupportQuerySampleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownSubject { subject_id } => {
                write!(f, "unknown behavioural subject_id {subject_id}")
            }
            Self::InsufficientHands {
                subject_id,
                required,
                available,
            } => write!(
                f,
                "subject_id {subject_id} has {available} retained hands; support/query sampling requires {required}"
            ),
        }
    }
}

impl std::error::Error for SupportQuerySampleError {}

impl BehaviourParquetDataset {
    /// Deterministically samples disjoint support and query hands for one subject.
    ///
    /// Selection is without replacement and keyed only by persisted metadata:
    /// `(seed, subject_id, hand_index)`. Selected hands are returned in their
    /// original dataset order inside each set so hand order is never randomized
    /// after membership has been chosen.
    pub fn sample_support_query(
        &self,
        subject_id: u64,
        config: SupportQueryConfig,
    ) -> Result<SupportQuerySample, SupportQuerySampleError> {
        let memberships = self
            .subjects
            .get(&subject_id)
            .ok_or(SupportQuerySampleError::UnknownSubject { subject_id })?;

        let required = config.total_hands();
        if memberships.len() < required {
            return Err(SupportQuerySampleError::InsufficientHands {
                subject_id,
                required,
                available: memberships.len(),
            });
        }

        let mut ranked = memberships
            .iter()
            .enumerate()
            .map(|(membership_index, membership)| {
                let hand_index = self.hands[membership.hand_slot].hand_index;
                (
                    sampling_key(config.seed, subject_id, hand_index),
                    hand_index,
                    membership_index,
                )
            })
            .collect::<Vec<_>>();
        ranked.sort_unstable();

        let mut support_indexes = ranked[..config.support_hands]
            .iter()
            .map(|entry| entry.2)
            .collect::<Vec<_>>();
        let mut query_indexes = ranked[config.support_hands..required]
            .iter()
            .map(|entry| entry.2)
            .collect::<Vec<_>>();

        // Membership rows are already sorted by persisted hand_index. Sorting
        // these indexes therefore restores the original dataset order within
        // support and query after deterministic membership selection.
        support_indexes.sort_unstable();
        query_indexes.sort_unstable();

        let materialize = |membership_index: usize| {
            let membership = memberships[membership_index];
            self.hands[membership.hand_slot].materialize(membership.subject_position)
        };

        let support = support_indexes.into_iter().map(materialize).collect();
        let query = query_indexes.into_iter().map(materialize).collect();

        Ok(SupportQuerySample {
            subject_id,
            support,
            query,
        })
    }
}

fn sampling_key(seed: u64, subject_id: u64, hand_index: u64) -> u64 {
    splitmix64(seed ^ splitmix64(subject_id) ^ splitmix64(hand_index))
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, HashSet};

    use crate::behavioural::{EVENT_CONTINUOUS_WIDTH, HandId, MAX_BOARD_CARDS, Position};
    use crate::datasets::behaviour_parquet::BehaviourParquetDatasetSummary;

    use super::super::{PersistedHand, SubjectMembership};
    use super::*;

    fn hand(hand_index: u64) -> PersistedHand {
        PersistedHand {
            hand_id: HandId::new(format!("h{hand_index}")),
            hand_index,
            players_at_table: 2,
            sequence_indexes: vec![0],
            categorical_without_subject: vec![[0, 4, 2, 0]],
            continuous: vec![[0.0; EVENT_CONTINUOUS_WIDTH]],
            continuous_present: vec![[1; EVENT_CONTINUOUS_WIDTH]],
            decision_time_ms: vec![100],
            board_rank: vec![[0; MAX_BOARD_CARDS]],
            board_suit: vec![[0; MAX_BOARD_CARDS]],
            board_present: vec![[0; MAX_BOARD_CARDS]],
        }
    }

    fn dataset(hand_count: usize) -> BehaviourParquetDataset {
        let hands = (0..hand_count)
            .map(|index| hand(u64::try_from(index).unwrap()))
            .collect::<Vec<_>>();
        let memberships = (0..hand_count)
            .map(|hand_slot| SubjectMembership {
                hand_slot,
                subject_position: Position::ButtonSmallBlind,
            })
            .collect::<Vec<_>>();

        BehaviourParquetDataset {
            summary: BehaviourParquetDatasetSummary {
                hands_seen: u64::try_from(hand_count).unwrap(),
                hands_written: u64::try_from(hand_count).unwrap(),
                events_written: u64::try_from(hand_count).unwrap(),
                subject_hand_rows_written: u64::try_from(hand_count).unwrap(),
                dropped_hands_missing_timing: 0,
            },
            hands,
            subjects: BTreeMap::from([(42, memberships)]),
            event_count: hand_count,
        }
    }

    fn hand_ids(hands: &[FeaturizedHand]) -> Vec<&str> {
        hands.iter().map(|hand| hand.hand_id().as_str()).collect()
    }

    #[test]
    fn config_requires_nonzero_support_and_query() {
        assert_eq!(
            SupportQueryConfig::new(0, 1, 7),
            Err(SupportQueryConfigError::ZeroSupportHands)
        );
        assert_eq!(
            SupportQueryConfig::new(1, 0, 7),
            Err(SupportQueryConfigError::ZeroQueryHands)
        );
    }

    #[test]
    fn sampling_is_deterministic_disjoint_and_exact() {
        let dataset = dataset(8);
        let config = SupportQueryConfig::new(3, 2, 0x1234).unwrap();

        let first = dataset.sample_support_query(42, config).unwrap();
        let second = dataset.sample_support_query(42, config).unwrap();

        assert_eq!(first, second);
        assert_eq!(first.subject_id(), 42);
        assert_eq!(first.support().len(), 3);
        assert_eq!(first.query().len(), 2);

        let support = hand_ids(first.support())
            .into_iter()
            .collect::<HashSet<_>>();
        let query = hand_ids(first.query()).into_iter().collect::<HashSet<_>>();
        assert!(support.is_disjoint(&query));
        assert_eq!(support.len() + query.len(), 5);
    }

    #[test]
    fn sampled_sets_preserve_original_hand_order() {
        let dataset = dataset(12);
        let config = SupportQueryConfig::new(5, 4, 99).unwrap();
        let sample = dataset.sample_support_query(42, config).unwrap();

        for hands in [sample.support(), sample.query()] {
            let indexes = hands
                .iter()
                .map(|hand| {
                    hand.hand_id()
                        .as_str()
                        .strip_prefix('h')
                        .unwrap()
                        .parse::<u64>()
                        .unwrap()
                })
                .collect::<Vec<_>>();
            assert!(indexes.windows(2).all(|pair| pair[0] < pair[1]));
        }
    }

    #[test]
    fn sampling_rejects_unknown_subject() {
        let dataset = dataset(4);
        let config = SupportQueryConfig::new(2, 1, 1).unwrap();

        assert_eq!(
            dataset.sample_support_query(999, config),
            Err(SupportQuerySampleError::UnknownSubject { subject_id: 999 })
        );
    }

    #[test]
    fn sampling_rejects_insufficient_hands_without_reuse() {
        let dataset = dataset(4);
        let config = SupportQueryConfig::new(3, 2, 1).unwrap();

        assert_eq!(
            dataset.sample_support_query(42, config),
            Err(SupportQuerySampleError::InsufficientHands {
                subject_id: 42,
                required: 5,
                available: 4,
            })
        );
    }
}
