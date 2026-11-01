use std::collections::{BTreeMap, HashMap};
use std::fmt;

use crate::behavioural::ContinuousFeature;

use super::BehaviourParquetDataset;

/// Split ratios are expressed in basis points to keep dataset partitioning exact
/// and independent of floating-point rounding.
pub const SPLIT_BASIS_POINTS: u16 = 10_000;

/// Version of the deterministic connected-component split algorithm.
/// Persist this alongside training configuration when reproducibility matters.
pub const SUBJECT_SPLIT_ALGORITHM_VERSION: u16 = 1;

/// Summary of an integer-valued empirical distribution.
///
/// Percentiles use the nearest-rank definition over the sorted observations.
#[derive(Debug, Clone, PartialEq)]
pub struct IntegerDistributionSummary {
    pub count: u64,
    pub min: Option<u64>,
    pub max: Option<u64>,
    pub mean: Option<f64>,
    pub p50: Option<u64>,
    pub p90: Option<u64>,
    pub p95: Option<u64>,
    pub p99: Option<u64>,
}

/// Summary of a finite non-negative floating-point empirical distribution.
///
/// Percentiles use the nearest-rank definition over the sorted observations.
#[derive(Debug, Clone, PartialEq)]
pub struct FloatDistributionSummary {
    pub count: u64,
    pub min: Option<f32>,
    pub max: Option<f32>,
    pub mean: Option<f64>,
    pub p50: Option<f32>,
    pub p90: Option<f32>,
    pub p95: Option<f32>,
    pub p99: Option<f32>,
}

/// Statistics derived from the validated normalized behavioural dataset.
///
/// Event distributions count each persisted event exactly once. They are not
/// multiplied by the number of subjects who participated in the hand.
#[derive(Debug, Clone, PartialEq)]
pub struct BehaviourDatasetStatistics {
    pub hands_seen: u64,
    pub retained_hands: u64,
    pub dropped_hands_missing_timing: u64,
    pub unique_events: u64,
    pub subject_hand_rows: u64,
    pub subjects: u64,
    pub connected_subject_components: u64,
    pub hands_per_subject: IntegerDistributionSummary,
    pub events_per_hand: IntegerDistributionSummary,
    pub subjects_per_component: IntegerDistributionSummary,
    pub decision_time_ms: IntegerDistributionSummary,
    pub pot_bb: FloatDistributionSummary,
    pub actor_stack_bb: FloatDistributionSummary,
    pub effective_stack_bb: FloatDistributionSummary,
    pub amount_added_bb: FloatDistributionSummary,
    /// Preflop, flop, turn, river in the persisted street codebook order.
    pub street_counts: [u64; 4],
    /// Fold, check, call, bet, raise in the persisted action codebook order.
    pub action_counts: [u64; 5],
    /// UTG, HJ, CO, BTN, BTN/SB, SB, BB in the persisted position codebook order.
    pub actor_position_counts: [u64; 7],
    /// Indexed by table size. Slots 0 and 1 are always zero for supported data.
    pub table_size_hand_counts: [u64; 7],
    pub all_in_events: u64,
    pub min_raise_observed_events: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DatasetPartition {
    Train,
    Validation,
    Test,
}

/// Exact configured split thresholds in basis points.
///
/// All three ratios must sum to [`SPLIT_BASIS_POINTS`]. Connected components are
/// indivisible, so the realized subject counts can differ from these target
/// proportions. The seed affects only deterministic split assignment; it is
/// never exposed as a model feature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubjectSplitConfig {
    train_bps: u16,
    validation_bps: u16,
    test_bps: u16,
    seed: u64,
}

impl SubjectSplitConfig {
    pub fn new(
        train_bps: u16,
        validation_bps: u16,
        test_bps: u16,
        seed: u64,
    ) -> Result<Self, SubjectSplitConfigError> {
        let total = u32::from(train_bps) + u32::from(validation_bps) + u32::from(test_bps);
        if total != u32::from(SPLIT_BASIS_POINTS) {
            return Err(SubjectSplitConfigError::RatiosMustSumToBasisPoints {
                train_bps,
                validation_bps,
                test_bps,
            });
        }
        Ok(Self {
            train_bps,
            validation_bps,
            test_bps,
            seed,
        })
    }

    pub fn train_bps(&self) -> u16 {
        self.train_bps
    }

    pub fn validation_bps(&self) -> u16 {
        self.validation_bps
    }

    pub fn test_bps(&self) -> u16 {
        self.test_bps
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubjectSplitConfigError {
    RatiosMustSumToBasisPoints {
        train_bps: u16,
        validation_bps: u16,
        test_bps: u16,
    },
}

impl fmt::Display for SubjectSplitConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RatiosMustSumToBasisPoints {
                train_bps,
                validation_bps,
                test_bps,
            } => write!(
                f,
                "subject split ratios must sum to {SPLIT_BASIS_POINTS} basis points, got train={train_bps}, validation={validation_bps}, test={test_bps}"
            ),
        }
    }
}

impl std::error::Error for SubjectSplitConfigError {}

/// Leakage-safe partition of pseudonymous subjects.
///
/// Subjects connected through one or more retained hands are assigned together,
/// so the same subject and the same retained hand can never cross partitions.
/// This does not claim session-level isolation because the normalized dataset
/// does not persist a generic session identifier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubjectDatasetSplit {
    train_subject_ids: Vec<u64>,
    validation_subject_ids: Vec<u64>,
    test_subject_ids: Vec<u64>,
    connected_components: usize,
}

impl SubjectDatasetSplit {
    pub fn train_subject_ids(&self) -> &[u64] {
        &self.train_subject_ids
    }

    pub fn validation_subject_ids(&self) -> &[u64] {
        &self.validation_subject_ids
    }

    pub fn test_subject_ids(&self) -> &[u64] {
        &self.test_subject_ids
    }

    pub fn connected_components(&self) -> usize {
        self.connected_components
    }

    pub fn partition_of(&self, subject_id: u64) -> Option<DatasetPartition> {
        if self.train_subject_ids.binary_search(&subject_id).is_ok() {
            return Some(DatasetPartition::Train);
        }
        if self
            .validation_subject_ids
            .binary_search(&subject_id)
            .is_ok()
        {
            return Some(DatasetPartition::Validation);
        }
        if self.test_subject_ids.binary_search(&subject_id).is_ok() {
            return Some(DatasetPartition::Test);
        }
        None
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BehaviourDatasetAnalysisError {
    CountOverflow { field: &'static str },
    InvalidInternalCategorical { feature: &'static str, value: i64 },
    InvalidInternalPresenceMask { feature: &'static str, value: u8 },
}

impl fmt::Display for BehaviourDatasetAnalysisError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CountOverflow { field } => write!(f, "dataset analysis {field} count overflow"),
            Self::InvalidInternalCategorical { feature, value } => write!(
                f,
                "dataset analysis found invalid internal categorical {feature} value {value}"
            ),
            Self::InvalidInternalPresenceMask { feature, value } => write!(
                f,
                "dataset analysis found invalid internal presence mask {feature} value {value}"
            ),
        }
    }
}

impl std::error::Error for BehaviourDatasetAnalysisError {}

impl BehaviourParquetDataset {
    /// Computes exact corpus statistics from unique retained event rows.
    pub fn statistics(&self) -> Result<BehaviourDatasetStatistics, BehaviourDatasetAnalysisError> {
        let components = cooccurrence_components(self);
        let hands_per_subject = self
            .subjects
            .values()
            .map(|memberships| usize_to_u64(memberships.len(), "hands per subject"))
            .collect::<Result<Vec<_>, _>>()?;
        let events_per_hand = self
            .hands
            .iter()
            .map(|hand| usize_to_u64(hand.sequence_indexes.len(), "events per hand"))
            .collect::<Result<Vec<_>, _>>()?;
        let subjects_per_component = components
            .iter()
            .map(|component| usize_to_u64(component.len(), "subjects per component"))
            .collect::<Result<Vec<_>, _>>()?;

        let mut decision_time_ms = Vec::with_capacity(self.event_count);
        let mut pot_bb = Vec::with_capacity(self.event_count);
        let mut actor_stack_bb = Vec::with_capacity(self.event_count);
        let mut effective_stack_bb = Vec::with_capacity(self.event_count);
        let mut amount_added_bb = Vec::with_capacity(self.event_count);
        let mut street_counts = [0_u64; 4];
        let mut action_counts = [0_u64; 5];
        let mut actor_position_counts = [0_u64; 7];
        let mut table_size_hand_counts = [0_u64; 7];
        let mut all_in_events = 0_u64;
        let mut min_raise_observed_events = 0_u64;

        for hand in &self.hands {
            let table_size = usize::from(hand.players_at_table);
            table_size_hand_counts[table_size] =
                checked_increment(table_size_hand_counts[table_size], "table size hand counts")?;

            for (event_index, base) in hand.categorical_without_subject.iter().enumerate() {
                let street = checked_categorical_index(base[0], street_counts.len(), "street")?;
                street_counts[street] = checked_increment(street_counts[street], "street counts")?;

                let position = checked_categorical_index(
                    base[1],
                    actor_position_counts.len(),
                    "actor_position",
                )?;
                actor_position_counts[position] =
                    checked_increment(actor_position_counts[position], "position counts")?;

                let action = checked_categorical_index(base[2], action_counts.len(), "action")?;
                action_counts[action] = checked_increment(action_counts[action], "action counts")?;

                match base[3] {
                    0 => {}
                    1 => all_in_events = checked_increment(all_in_events, "all-in events")?,
                    value => {
                        return Err(BehaviourDatasetAnalysisError::InvalidInternalCategorical {
                            feature: "is_all_in",
                            value,
                        });
                    }
                }

                let present =
                    hand.continuous_present[event_index][ContinuousFeature::MinRaiseToBb as usize];
                match present {
                    0 => {}
                    1 => {
                        min_raise_observed_events = checked_increment(
                            min_raise_observed_events,
                            "observed min-raise events",
                        )?;
                    }
                    value => {
                        return Err(BehaviourDatasetAnalysisError::InvalidInternalPresenceMask {
                            feature: "min_raise_to_bb",
                            value,
                        });
                    }
                }

                decision_time_ms.push(u64::from(hand.decision_time_ms[event_index]));
                let continuous = hand.continuous[event_index];
                pot_bb.push(continuous[ContinuousFeature::PotBb as usize]);
                actor_stack_bb.push(continuous[ContinuousFeature::ActorStackBb as usize]);
                effective_stack_bb.push(continuous[ContinuousFeature::EffectiveStackBb as usize]);
                amount_added_bb.push(continuous[ContinuousFeature::AmountAddedBb as usize]);
            }
        }

        Ok(BehaviourDatasetStatistics {
            hands_seen: self.summary.hands_seen,
            retained_hands: self.summary.hands_written,
            dropped_hands_missing_timing: self.summary.dropped_hands_missing_timing,
            unique_events: self.summary.events_written,
            subject_hand_rows: self.summary.subject_hand_rows_written,
            subjects: usize_to_u64(self.subjects.len(), "subjects")?,
            connected_subject_components: usize_to_u64(
                components.len(),
                "connected subject components",
            )?,
            hands_per_subject: summarize_u64(hands_per_subject)?,
            events_per_hand: summarize_u64(events_per_hand)?,
            subjects_per_component: summarize_u64(subjects_per_component)?,
            decision_time_ms: summarize_u64(decision_time_ms)?,
            pot_bb: summarize_f32(pot_bb)?,
            actor_stack_bb: summarize_f32(actor_stack_bb)?,
            effective_stack_bb: summarize_f32(effective_stack_bb)?,
            amount_added_bb: summarize_f32(amount_added_bb)?,
            street_counts,
            action_counts,
            actor_position_counts,
            table_size_hand_counts,
            all_in_events,
            min_raise_observed_events,
        })
    }

    /// Deterministically partitions connected subject components.
    ///
    /// Every pair of subjects linked transitively through retained hands stays in
    /// the same partition. Therefore no subject and no retained hand is shared
    /// between train, validation, and test. Generic session-level isolation is not
    /// asserted because the persisted schema has no generic session identifier.
    pub fn split_subjects(&self, config: SubjectSplitConfig) -> SubjectDatasetSplit {
        let mut train_subject_ids = Vec::new();
        let mut validation_subject_ids = Vec::new();
        let mut test_subject_ids = Vec::new();
        let components = cooccurrence_components(self);
        let train_end = u64::from(config.train_bps);
        let validation_end = train_end + u64::from(config.validation_bps);

        for component in &components {
            let bucket = component_bucket(component, config.seed);
            let destination = if bucket < train_end {
                &mut train_subject_ids
            } else if bucket < validation_end {
                &mut validation_subject_ids
            } else {
                &mut test_subject_ids
            };
            destination.extend(component.iter().copied());
        }

        train_subject_ids.sort_unstable();
        validation_subject_ids.sort_unstable();
        test_subject_ids.sort_unstable();

        SubjectDatasetSplit {
            train_subject_ids,
            validation_subject_ids,
            test_subject_ids,
            connected_components: components.len(),
        }
    }
}

fn checked_categorical_index(
    value: i64,
    width: usize,
    feature: &'static str,
) -> Result<usize, BehaviourDatasetAnalysisError> {
    let index = usize::try_from(value).map_err(|_| {
        BehaviourDatasetAnalysisError::InvalidInternalCategorical { feature, value }
    })?;
    if index >= width {
        return Err(BehaviourDatasetAnalysisError::InvalidInternalCategorical { feature, value });
    }
    Ok(index)
}

fn checked_increment(
    value: u64,
    field: &'static str,
) -> Result<u64, BehaviourDatasetAnalysisError> {
    value
        .checked_add(1)
        .ok_or(BehaviourDatasetAnalysisError::CountOverflow { field })
}

fn usize_to_u64(value: usize, field: &'static str) -> Result<u64, BehaviourDatasetAnalysisError> {
    u64::try_from(value).map_err(|_| BehaviourDatasetAnalysisError::CountOverflow { field })
}

fn summarize_u64(
    mut values: Vec<u64>,
) -> Result<IntegerDistributionSummary, BehaviourDatasetAnalysisError> {
    if values.is_empty() {
        return Ok(IntegerDistributionSummary {
            count: 0,
            min: None,
            max: None,
            mean: None,
            p50: None,
            p90: None,
            p95: None,
            p99: None,
        });
    }

    values.sort_unstable();
    let count = usize_to_u64(values.len(), "integer distribution")?;
    let sum = values.iter().try_fold(0_u128, |total, &value| {
        total
            .checked_add(u128::from(value))
            .ok_or(BehaviourDatasetAnalysisError::CountOverflow {
                field: "integer distribution sum",
            })
    })?;

    Ok(IntegerDistributionSummary {
        count,
        min: values.first().copied(),
        max: values.last().copied(),
        mean: Some(sum as f64 / count as f64),
        p50: nearest_rank_u64(&values, 50),
        p90: nearest_rank_u64(&values, 90),
        p95: nearest_rank_u64(&values, 95),
        p99: nearest_rank_u64(&values, 99),
    })
}

fn summarize_f32(
    mut values: Vec<f32>,
) -> Result<FloatDistributionSummary, BehaviourDatasetAnalysisError> {
    if values.is_empty() {
        return Ok(FloatDistributionSummary {
            count: 0,
            min: None,
            max: None,
            mean: None,
            p50: None,
            p90: None,
            p95: None,
            p99: None,
        });
    }

    values.sort_unstable_by(f32::total_cmp);
    let count = usize_to_u64(values.len(), "float distribution")?;
    let sum = values.iter().map(|&value| f64::from(value)).sum::<f64>();

    Ok(FloatDistributionSummary {
        count,
        min: values.first().copied(),
        max: values.last().copied(),
        mean: Some(sum / count as f64),
        p50: nearest_rank_f32(&values, 50),
        p90: nearest_rank_f32(&values, 90),
        p95: nearest_rank_f32(&values, 95),
        p99: nearest_rank_f32(&values, 99),
    })
}

fn nearest_rank_index(len: usize, percentile: u8) -> Option<usize> {
    if len == 0 {
        return None;
    }
    let len = len as u128;
    let percentile = u128::from(percentile);
    let rank = len
        .checked_mul(percentile)?
        .checked_add(99)?
        .checked_div(100)?;
    let index = rank.saturating_sub(1);
    usize::try_from(index).ok()
}

fn nearest_rank_u64(values: &[u64], percentile: u8) -> Option<u64> {
    values
        .get(nearest_rank_index(values.len(), percentile)?)
        .copied()
}

fn nearest_rank_f32(values: &[f32], percentile: u8) -> Option<f32> {
    values
        .get(nearest_rank_index(values.len(), percentile)?)
        .copied()
}

fn cooccurrence_components(dataset: &BehaviourParquetDataset) -> Vec<Vec<u64>> {
    let subject_ids = dataset.subjects.keys().copied().collect::<Vec<_>>();
    if subject_ids.is_empty() {
        return Vec::new();
    }

    let subject_index = subject_ids
        .iter()
        .enumerate()
        .map(|(index, &subject_id)| (subject_id, index))
        .collect::<HashMap<_, _>>();
    let mut subjects_by_hand = vec![Vec::<usize>::new(); dataset.hands.len()];
    for (&subject_id, memberships) in &dataset.subjects {
        let index = subject_index[&subject_id];
        for membership in memberships {
            subjects_by_hand[membership.hand_slot].push(index);
        }
    }

    let mut union_find = UnionFind::new(subject_ids.len());
    for hand_subjects in subjects_by_hand {
        if let Some((&first, rest)) = hand_subjects.split_first() {
            for &other in rest {
                union_find.union(first, other);
            }
        }
    }

    let mut by_root = BTreeMap::<usize, Vec<u64>>::new();
    for (index, subject_id) in subject_ids.into_iter().enumerate() {
        let root = union_find.find(index);
        by_root.entry(root).or_default().push(subject_id);
    }
    let mut components = by_root.into_values().collect::<Vec<_>>();
    for component in &mut components {
        component.sort_unstable();
    }
    components.sort_unstable_by_key(|component| component[0]);
    components
}

fn component_bucket(component: &[u64], seed: u64) -> u64 {
    let mut state = splitmix64(seed);
    for &subject_id in component {
        state = splitmix64(state ^ splitmix64(subject_id));
    }
    state % u64::from(SPLIT_BASIS_POINTS)
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[derive(Debug)]
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl UnionFind {
    fn new(len: usize) -> Self {
        Self {
            parent: (0..len).collect(),
            rank: vec![0; len],
        }
    }

    fn find(&mut self, index: usize) -> usize {
        if self.parent[index] != index {
            self.parent[index] = self.find(self.parent[index]);
        }
        self.parent[index]
    }

    fn union(&mut self, left: usize, right: usize) {
        let left_root = self.find(left);
        let right_root = self.find(right);
        if left_root == right_root {
            return;
        }

        match self.rank[left_root].cmp(&self.rank[right_root]) {
            std::cmp::Ordering::Less => self.parent[left_root] = right_root,
            std::cmp::Ordering::Greater => self.parent[right_root] = left_root,
            std::cmp::Ordering::Equal => {
                self.parent[right_root] = left_root;
                self.rank[left_root] = self.rank[left_root].saturating_add(1);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::behavioural::{ContinuousFeature, HandId, Position};
    use crate::datasets::behaviour_parquet::BehaviourParquetDatasetSummary;

    use super::super::{PersistedHand, SubjectMembership};

    fn hand(
        hand_id: &str,
        hand_index: u64,
        players_at_table: u8,
        street: i64,
        position: i64,
        action: i64,
        timing: u32,
        pot_bb: f32,
    ) -> PersistedHand {
        let mut continuous = [0.0_f32; crate::behavioural::EVENT_CONTINUOUS_WIDTH];
        continuous[ContinuousFeature::PotBb as usize] = pot_bb;
        continuous[ContinuousFeature::ActorStackBb as usize] = 100.0;
        continuous[ContinuousFeature::EffectiveStackBb as usize] = 80.0;
        continuous[ContinuousFeature::AmountAddedBb as usize] = 2.0;
        continuous[ContinuousFeature::DecisionTimeMs as usize] = timing as f32;

        PersistedHand {
            hand_id: HandId::new(hand_id),
            hand_index,
            players_at_table,
            sequence_indexes: vec![0],
            categorical_without_subject: vec![[street, position, action, 0]],
            continuous: vec![continuous],
            continuous_present: vec![[1; crate::behavioural::EVENT_CONTINUOUS_WIDTH]],
            decision_time_ms: vec![timing],
            board_rank: vec![[0; crate::behavioural::MAX_BOARD_CARDS]],
            board_suit: vec![[0; crate::behavioural::MAX_BOARD_CARDS]],
            board_present: vec![[0; crate::behavioural::MAX_BOARD_CARDS]],
        }
    }

    fn dataset_for_stats() -> BehaviourParquetDataset {
        let hands = vec![
            hand("h0", 0, 2, 0, 4, 2, 100, 1.5),
            hand("h1", 1, 2, 1, 6, 1, 300, 4.0),
        ];
        let subjects = BTreeMap::from([
            (
                10,
                vec![
                    SubjectMembership {
                        hand_slot: 0,
                        subject_position: Position::ButtonSmallBlind,
                    },
                    SubjectMembership {
                        hand_slot: 1,
                        subject_position: Position::ButtonSmallBlind,
                    },
                ],
            ),
            (
                20,
                vec![
                    SubjectMembership {
                        hand_slot: 0,
                        subject_position: Position::BigBlind,
                    },
                    SubjectMembership {
                        hand_slot: 1,
                        subject_position: Position::BigBlind,
                    },
                ],
            ),
        ]);
        BehaviourParquetDataset {
            summary: BehaviourParquetDatasetSummary {
                hands_seen: 3,
                hands_written: 2,
                events_written: 2,
                subject_hand_rows_written: 4,
                dropped_hands_missing_timing: 1,
            },
            hands,
            subjects,
            event_count: 2,
        }
    }

    #[test]
    fn statistics_count_unique_events_instead_of_subject_materializations() {
        let stats = dataset_for_stats().statistics().unwrap();
        assert_eq!(stats.hands_seen, 3);
        assert_eq!(stats.retained_hands, 2);
        assert_eq!(stats.dropped_hands_missing_timing, 1);
        assert_eq!(stats.unique_events, 2);
        assert_eq!(stats.subjects, 2);
        assert_eq!(stats.subject_hand_rows, 4);
        assert_eq!(stats.connected_subject_components, 1);
        assert_eq!(stats.hands_per_subject.count, 2);
        assert_eq!(stats.hands_per_subject.min, Some(2));
        assert_eq!(stats.events_per_hand.p50, Some(1));
        assert_eq!(stats.subjects_per_component.max, Some(2));
        assert_eq!(stats.decision_time_ms.min, Some(100));
        assert_eq!(stats.decision_time_ms.max, Some(300));
        assert_eq!(stats.decision_time_ms.p50, Some(100));
        assert_eq!(stats.decision_time_ms.p90, Some(300));
        assert_eq!(stats.pot_bb.min, Some(1.5));
        assert_eq!(stats.pot_bb.max, Some(4.0));
        assert_eq!(stats.street_counts, [1, 1, 0, 0]);
        assert_eq!(stats.action_counts, [0, 1, 1, 0, 0]);
        assert_eq!(stats.actor_position_counts, [0, 0, 0, 0, 1, 0, 1]);
        assert_eq!(stats.table_size_hand_counts[2], 2);
    }

    fn dataset_for_split() -> BehaviourParquetDataset {
        let hands = vec![
            hand("h0", 0, 2, 0, 4, 2, 100, 1.5),
            hand("h1", 1, 2, 0, 6, 1, 100, 1.5),
            hand("h2", 2, 2, 0, 4, 2, 100, 1.5),
            hand("h3", 3, 2, 0, 4, 2, 100, 1.5),
        ];
        let subjects = BTreeMap::from([
            (
                1,
                vec![SubjectMembership {
                    hand_slot: 0,
                    subject_position: Position::ButtonSmallBlind,
                }],
            ),
            (
                2,
                vec![
                    SubjectMembership {
                        hand_slot: 0,
                        subject_position: Position::BigBlind,
                    },
                    SubjectMembership {
                        hand_slot: 1,
                        subject_position: Position::ButtonSmallBlind,
                    },
                ],
            ),
            (
                3,
                vec![SubjectMembership {
                    hand_slot: 1,
                    subject_position: Position::BigBlind,
                }],
            ),
            (
                4,
                vec![SubjectMembership {
                    hand_slot: 2,
                    subject_position: Position::ButtonSmallBlind,
                }],
            ),
            (
                5,
                vec![SubjectMembership {
                    hand_slot: 2,
                    subject_position: Position::BigBlind,
                }],
            ),
            (
                6,
                vec![SubjectMembership {
                    hand_slot: 3,
                    subject_position: Position::ButtonSmallBlind,
                }],
            ),
            (
                7,
                vec![SubjectMembership {
                    hand_slot: 3,
                    subject_position: Position::BigBlind,
                }],
            ),
        ]);
        BehaviourParquetDataset {
            summary: BehaviourParquetDatasetSummary {
                hands_seen: 4,
                hands_written: 4,
                events_written: 4,
                subject_hand_rows_written: 8,
                dropped_hands_missing_timing: 0,
            },
            hands,
            subjects,
            event_count: 4,
        }
    }

    #[test]
    fn split_keeps_transitively_connected_subjects_together() {
        let dataset = dataset_for_split();
        let config = SubjectSplitConfig::new(8_000, 1_000, 1_000, 7).unwrap();
        let split = dataset.split_subjects(config);

        assert_eq!(split.connected_components(), 3);
        assert_eq!(split.partition_of(1), split.partition_of(2));
        assert_eq!(split.partition_of(2), split.partition_of(3));
        assert_eq!(split.partition_of(4), split.partition_of(5));
        assert_eq!(split.partition_of(6), split.partition_of(7));
        assert!(split.partition_of(999).is_none());

        for hand_slot in 0..dataset.hands.len() {
            let partitions = dataset
                .subjects
                .iter()
                .filter(|(_, memberships)| {
                    memberships
                        .iter()
                        .any(|membership| membership.hand_slot == hand_slot)
                })
                .map(|(&subject_id, _)| split.partition_of(subject_id).unwrap())
                .collect::<Vec<_>>();
            assert!(partitions.windows(2).all(|pair| pair[0] == pair[1]));
        }

        let assigned = split.train_subject_ids().len()
            + split.validation_subject_ids().len()
            + split.test_subject_ids().len();
        assert_eq!(assigned, 7);
    }

    #[test]
    fn split_is_deterministic_for_same_seed() {
        let dataset = dataset_for_split();
        let config = SubjectSplitConfig::new(7_000, 2_000, 1_000, 0x1234).unwrap();
        assert_eq!(
            dataset.split_subjects(config),
            dataset.split_subjects(config)
        );
    }

    #[test]
    fn split_config_requires_exact_basis_point_total() {
        assert_eq!(
            SubjectSplitConfig::new(8_000, 1_000, 999, 0),
            Err(SubjectSplitConfigError::RatiosMustSumToBasisPoints {
                train_bps: 8_000,
                validation_bps: 1_000,
                test_bps: 999,
            })
        );
    }
}
