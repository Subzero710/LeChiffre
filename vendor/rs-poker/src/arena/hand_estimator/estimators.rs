//! Built-in estimators: known hands (default) and uniform-random.

use async_trait::async_trait;

use crate::arena::GameState;
use crate::core::{Card, CardBitSet};

use super::{
    GameLog, HandDistribution, HandDistributionEstimator, HoleCombo, OpponentRanges,
    WeightedCombos, all_hole_combos,
};

/// True when `seat` still holds hidden cards in the hand (active or all-in).
fn seat_in_hand(game_state: &GameState, seat: usize) -> bool {
    game_state.player_active.get(seat) || game_state.player_all_in.get(seat)
}

/// The two hole cards for `seat`: the seat's hand minus the shared board.
fn hole_cards(game_state: &GameState, seat: usize, board: &CardBitSet) -> Option<HoleCombo> {
    let hole: Vec<Card> = game_state.hands[seat]
        .iter()
        .filter(|c| !board.contains(*c))
        .collect();
    if hole.len() == 2 {
        Some(HoleCombo::new(hole[0], hole[1]))
    } else {
        None
    }
}

/// Estimator that returns each opponent's true hand as a point mass. This is
/// the default; with it, `sample_world` reproduces the real hands exactly, so
/// CFR behavior is identical to the pre-estimator engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct KnownHandsEstimator;

#[async_trait]
impl HandDistributionEstimator for KnownHandsEstimator {
    async fn estimate(
        &self,
        game_state: &GameState,
        _history: Option<&GameLog<'_>>,
    ) -> OpponentRanges {
        let perspective_idx = game_state.to_act_idx();
        let board: CardBitSet = game_state.board.iter().copied().collect();
        let per_seat = (0..game_state.num_players)
            .map(|seat| {
                if seat == perspective_idx || !seat_in_hand(game_state, seat) {
                    None
                } else {
                    hole_cards(game_state, seat, &board).map(HandDistribution::PointMass)
                }
            })
            .collect();
        OpponentRanges::new(per_seat)
    }
}

/// Estimator that assigns each opponent a uniform distribution over all combos
/// (dead-card filtering happens at sample time). The no-ML "play against a
/// range" baseline and the plumbing's test fixture. Folded seats are also
/// resampled: their real private cards must not become known dead cards.
#[derive(Debug, Clone, Copy, Default)]
pub struct UniformRandomEstimator;

#[async_trait]
impl HandDistributionEstimator for UniformRandomEstimator {
    async fn estimate(
        &self,
        game_state: &GameState,
        _history: Option<&GameLog<'_>>,
    ) -> OpponentRanges {
        let perspective_idx = game_state.to_act_idx();
        let weights: Vec<(HoleCombo, f32)> =
            all_hole_combos().into_iter().map(|c| (c, 1.0)).collect();
        let per_seat = (0..game_state.num_players)
            .map(|seat| {
                if seat == perspective_idx || game_state.hands[seat].count() < 2 {
                    None
                } else {
                    Some(HandDistribution::Weighted(WeightedCombos {
                        weights: weights.clone(),
                    }))
                }
            })
            .collect();
        OpponentRanges::new(per_seat)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::GameStateBuilder;
    use crate::core::Hand;

    fn two_player_state() -> GameState {
        // Two players, each dealt distinct hole cards, no board.
        let mut gs = GameStateBuilder::default()
            .num_players_with_stack(2, 100)
            .big_blind(2)
            .build()
            .unwrap();
        gs.hands[0] = Hand::new_with_cards(vec![Card::from(0), Card::from(1)]);
        gs.hands[1] = Hand::new_with_cards(vec![Card::from(2), Card::from(3)]);
        gs
    }

    #[tokio::test]
    async fn returns_point_mass_for_opponent_only() {
        let mut gs = two_player_state();
        gs.round_data.to_act_idx = 0;
        let ranges = KnownHandsEstimator.estimate(&gs, None).await;
        assert!(ranges.get(0).is_none(), "acting seat must be None");
        match ranges.get(1) {
            Some(HandDistribution::PointMass(c)) => {
                assert_eq!(*c, HoleCombo::new(Card::from(2), Card::from(3)));
            }
            other => panic!("expected point mass, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn uniform_returns_weighted_for_opponent_only() {
        let mut gs = two_player_state();
        gs.round_data.to_act_idx = 1;
        let ranges = UniformRandomEstimator.estimate(&gs, None).await;
        assert!(ranges.get(1).is_none(), "acting seat must be None");
        match ranges.get(0) {
            Some(HandDistribution::Weighted(w)) => assert_eq!(w.weights.len(), 1326),
            other => panic!("expected weighted, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn uniform_world_does_not_use_real_folded_cards_as_blockers() {
        use super::super::sample_world;
        use rand::{SeedableRng, rngs::StdRng};
        let mut a = GameStateBuilder::new()
            .num_players_with_stack(3, 100)
            .blinds(2, 1)
            .build()
            .unwrap();
        a.round_data.to_act_idx = 0;
        for p in 0..3 {
            a.hands[p] = Hand::new_with_cards(vec![
                Card::from((2 * p) as u8),
                Card::from((2 * p + 1) as u8),
            ]);
        }
        a.player_active.disable(2);
        let mut b = a.clone();
        b.hands[2] = Hand::new_with_cards(vec![Card::from(7), Card::from(8)]);
        let ar = UniformRandomEstimator.estimate(&a, None).await;
        let br = UniformRandomEstimator.estimate(&b, None).await;
        assert!(ar.get(2).is_some(), "folded private cards remain unknown");
        for seed in 0..8 {
            let aw = sample_world(&ar, &a, &mut StdRng::seed_from_u64(seed));
            let bw = sample_world(&br, &b, &mut StdRng::seed_from_u64(seed));
            assert_eq!(aw.hands, bw.hands);
        }
    }
}
