//! Chronological Z1 personalization using **only** retained hands preceding an
//! explicit exclusive global hand-index cutoff. Future observations cannot be
//! read by this path, including for another subject in the same table hand.

use super::generator_training::{decode_decision, lookup, subject_decisions};
use super::{BehaviourParquetDataset, ExactLegalityProvider, GeneratorTrainingError};
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
    ) -> Result<PlayerBaselineState, GeneratorTrainingError> {
        let memberships = self
            .subjects
            .get(&subject_id)
            .ok_or(GeneratorTrainingError::UnknownSubject { subject_id })?;
        let mut earlier = memberships
            .iter()
            .map(|membership| {
                (
                    &self.hands[membership.hand_slot],
                    membership.subject_position,
                )
            })
            .filter(|(hand, _)| hand.hand_index < before_hand_index)
            .collect::<Vec<_>>();
        earlier.sort_unstable_by_key(|(hand, _)| hand.hand_index);
        let mut state = adapter.initialize_player();
        for (hand, position) in earlier {
            let observed = hand.materialize(position);
            let mut decisions = Vec::new();
            for row in subject_decisions(&observed) {
                let observation = decode_decision(&observed, row)?;
                let legal = lookup(provider, subject_id, &observed, row)?;
                decisions.push((observation, legal));
            }
            if decisions.is_empty() {
                continue;
            }
            adapter.adapt_hand(&mut state, generator, hand.hand_index, &decisions)?;
        }
        Ok(state)
    }
}
