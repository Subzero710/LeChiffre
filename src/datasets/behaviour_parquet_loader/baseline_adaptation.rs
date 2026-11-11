//! Chronological Z1 personalization using **only** retained hands preceding an
//! explicit exclusive global hand-index cutoff. Future observations cannot be
//! read by this path, including for another subject in the same table hand.

use super::training_data::{decode_decision, lookup, subject_decisions};
use super::{BehaviourParquetDataset, ExactLegalityProvider, TrainingDataError};
use crate::behavioural::{NeuralBehaviourGenerator, PlayerBaselineAdapter, PlayerBaselineState};

impl BehaviourParquetDataset {
    /// Calculate the subject's Z1 from strictly earlier retained hands.
    ///
    /// `before_hand_index` is an exclusive, dataset-global chronological boundary.
    /// Only the selected player's actual voluntary decisions count as evidence.
    /// Hands without such decisions do not increment `observed_hands`.
    /// No weights of the supplied Generator (or the Analyzer) are modified.
    /// Every used decision must have an exact replay-sourced legal space.
    pub fn adapt_baseline_before<P: ExactLegalityProvider>(
        &self,
        subject_id: u64,
        before_hand_index: u64,
        generator: &NeuralBehaviourGenerator,
        adapter: &PlayerBaselineAdapter,
        provider: &mut P,
    ) -> Result<PlayerBaselineState, TrainingDataError> {
        let mut state = adapter.initialize_player();
        self.advance_baseline_before(
            subject_id,
            before_hand_index,
            generator,
            adapter,
            &mut state,
            provider,
        )?;
        Ok(state)
    }

    /// Advance an existing persisted Z1 without rereading already covered hands.
    /// Fully atomic: an error in any new hand leaves the caller's state unchanged.
    pub fn advance_baseline_before<P: ExactLegalityProvider>(
        &self,
        subject_id: u64,
        before_hand_index: u64,
        generator: &NeuralBehaviourGenerator,
        adapter: &PlayerBaselineAdapter,
        state: &mut PlayerBaselineState,
        provider: &mut P,
    ) -> Result<(), TrainingDataError> {
        adapter
            .validate_player_state(state)
            .map_err(TrainingDataError::Baseline)?;
        let memberships = self
            .subjects
            .get(&subject_id)
            .ok_or(TrainingDataError::UnknownSubject { subject_id })?;
        let start = state.processed_before_hand_index().unwrap_or(0);
        if before_hand_index < start {
            return Err(TrainingDataError::NonIncreasingCutoff);
        }
        let mut earlier = memberships
            .iter()
            .map(|membership| {
                (
                    &self.hands[membership.hand_slot],
                    membership.subject_position,
                )
            })
            .filter(|(hand, _)| hand.hand_index >= start && hand.hand_index < before_hand_index)
            .collect::<Vec<_>>();
        earlier.sort_unstable_by_key(|(hand, _)| hand.hand_index);
        let mut working = state.clone();
        for (hand, position) in earlier {
            let observed = hand.materialize(position);
            let mut decisions = Vec::new();
            for row in subject_decisions(&observed) {
                decisions.push((
                    decode_decision(&observed, row)?,
                    lookup(provider, subject_id, &observed, row)?,
                ));
            }
            if !decisions.is_empty() {
                adapter
                    .adapt_hand(&mut working, generator, hand.hand_index, &decisions)
                    .map_err(TrainingDataError::Baseline)?;
            }
        }
        working
            .mark_processed_before(before_hand_index)
            .map_err(TrainingDataError::Baseline)?;
        *state = working;
        Ok(())
    }
}
