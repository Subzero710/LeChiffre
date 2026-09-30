//! Shared main/side-pot planning and settlement for simulation and CFR.
use super::{Chips, GameState, game_state::MAX_PLAYERS, money::split_pot};
use crate::core::{PlayerBitSet, Rank, Rankable};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PotSlice {
    pub gross: Chips,
    pub net: Chips,
    pub rake: Chips,
    pub eligible: PlayerBitSet,
    /// A single contributor's unmatched excess is returned without rake.
    pub refund_to: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PotAward {
    pub pot_index: usize,
    pub idx: usize,
    pub gross: Chips,
    pub amount: Chips,
    pub rake: Chips,
    pub refund: bool,
}

/// Construct pots in contribution-level order, preserving folded contributions.
/// All pot amounts, rates, caps and remainders are evaluated before CFR converts
/// a completed monetary reward into a float.
pub fn plan_pots(gs: &GameState, flop_dealt: bool) -> Vec<PotSlice> {
    assert_eq!(
        gs.player_bet.iter().sum::<Chips>(),
        gs.total_pot,
        "pot ledger mismatch"
    );
    let contenders = gs.player_active | gs.player_all_in;
    let mut levels: Vec<Chips> = gs.player_bet.iter().copied().filter(|&v| v > 0).collect();
    levels.sort_unstable();
    levels.dedup();
    let mut pots: Vec<PotSlice> = Vec::with_capacity(levels.len());
    let mut previous = 0;
    for level in levels {
        let contributors: Vec<usize> = gs
            .player_bet
            .iter()
            .enumerate()
            .filter_map(|(idx, &bet)| (bet >= level).then_some(idx))
            .collect();
        let gross = (level - previous) * contributors.len() as Chips;
        let refund_to = (contributors.len() == 1).then(|| contributors[0]);
        let mut eligible = PlayerBitSet::default();
        for &idx in &contributors {
            if contenders.get(idx) {
                eligible.enable(idx);
            }
        }
        assert!(
            refund_to.is_some() || !eligible.empty(),
            "pot has no eligible player"
        );
        if refund_to.is_none()
            && pots
                .last()
                .is_some_and(|p| p.refund_to.is_none() && p.eligible == eligible)
        {
            pots.last_mut().unwrap().gross += gross;
        } else {
            pots.push(PotSlice {
                gross,
                net: gross,
                rake: 0,
                eligible,
                refund_to,
            });
        }
        previous = level;
    }
    let mut collected = gs.rake_collected;
    for pot in &mut pots {
        pot.rake = if pot.refund_to.is_some() {
            0
        } else {
            gs.rake.calculate(pot.gross, flop_dealt, collected)
        };
        pot.net = pot.gross - pot.rake;
        collected += pot.rake;
    }
    assert_eq!(pots.iter().map(|p| p.gross).sum::<Chips>(), gs.total_pot);
    pots
}

pub(crate) fn winners_for(pot: &PotSlice, ranks: &[Option<Rank>; MAX_PLAYERS]) -> PlayerBitSet {
    let best = pot.eligible.ones().filter_map(|idx| ranks[idx]).max();
    let mut winners = PlayerBitSet::default();
    for idx in pot.eligible.ones() {
        if ranks[idx] == best {
            winners.enable(idx);
        }
    }
    winners
}

/// Exact share for a single player without allocating on enumerated CFR boards.
pub(crate) fn player_share(
    amount: Chips,
    winners: PlayerBitSet,
    gs: &GameState,
    idx: usize,
) -> Chips {
    if !winners.get(idx) {
        return 0;
    }
    let count = winners.count() as Chips;
    let distance = |seat: usize| (seat + gs.num_players - gs.dealer_idx - 1) % gs.num_players;
    let position = winners
        .ones()
        .filter(|&seat| distance(seat) < distance(idx))
        .count() as Chips;
    amount / count + Chips::from(position < amount % count)
}

pub(crate) fn winnings_for_ranks(
    pots: &[PotSlice],
    ranks: &[Option<Rank>; MAX_PLAYERS],
    gs: &GameState,
    idx: usize,
) -> Chips {
    pots.iter()
        .map(|pot| match pot.refund_to {
            Some(owner) => {
                if idx == owner {
                    pot.net
                } else {
                    0
                }
            }
            None => player_share(pot.net, winners_for(pot, ranks), gs, idx),
        })
        .sum()
}

/// Settle all pots using the same rules in the full and fast-forward engines.
/// `total_pot` retains its historical gross amount, as expected by historians.
pub fn planned_awards(gs: &GameState) -> Vec<PotAward> {
    let pots = plan_pots(gs, gs.board.len() >= 3);
    let contenders = gs.player_active | gs.player_all_in;
    let mut ranks = [None; MAX_PLAYERS];
    if contenders.count() > 1 {
        for idx in contenders.ones() {
            ranks[idx] = Some(gs.hands[idx].rank());
        }
    }
    let mut awards = Vec::new();
    for (pot_index, pot) in pots.into_iter().enumerate() {
        let splits = if let Some(idx) = pot.refund_to {
            vec![(idx, pot.net)]
        } else {
            let winners: Vec<_> = winners_for(&pot, &ranks).ones().collect();
            split_pot(pot.net, &winners, gs.dealer_idx, gs.num_players)
        };
        for (idx, amount) in splits {
            awards.push(PotAward {
                pot_index,
                idx,
                gross: pot.gross,
                amount,
                rake: pot.rake,
                refund: pot.refund_to.is_some(),
            });
        }
    }
    awards
}

pub fn settle_pots(gs: &mut GameState) -> Vec<PotAward> {
    let awards = planned_awards(gs);
    let mut last_pot = None;
    for award in &awards {
        if award.refund {
            gs.return_uncalled_bet(award.idx, award.amount);
        } else {
            if last_pot != Some(award.pot_index) {
                gs.rake_collected += award.rake;
                last_pot = Some(award.pot_index);
            }
            gs.award(award.idx, award.amount);
        }
    }
    assert_eq!(
        gs.player_winnings.iter().sum::<Chips>() + gs.rake_collected,
        gs.total_pot
    );
    awards
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::{GameStateBuilder, RakeConfig, RakeRate, RakeRounding, game_state::Round};
    use crate::core::Hand;
    fn tied_state(bets: &[Chips], stacks: &[Chips], dealer: usize, rake: RakeConfig) -> GameState {
        let board = Hand::new_from_str("AsKsQsJsTs").unwrap();
        let holes = ["2c2d", "3c3d", "4c4d", "5c5d"];
        let hands = (0..bets.len())
            .map(|i| {
                let mut hand = Hand::new_from_str(holes[i]).unwrap();
                for card in board.iter() {
                    hand.insert(card);
                }
                hand
            })
            .collect::<Vec<_>>();
        GameStateBuilder::new()
            .round(Round::Showdown)
            .blinds(2, 1)
            .dealer_idx(dealer)
            .stacks(stacks)
            .player_bet(bets)
            .hands(hands)
            .board(board.iter().collect::<Vec<_>>())
            .rake(rake)
            .build()
            .unwrap()
    }
    #[test]
    fn side_pots_ties_shared_cap_and_uncalled_returns_conserve() {
        let rake = RakeConfig::new(
            RakeRate::new(5, 100).unwrap(),
            Some(25),
            true,
            RakeRounding::Floor,
        )
        .unwrap();
        let mut gs = tied_state(&[100, 200, 200, 300], &[0, 0, 0, 500], 3, rake);
        let pots = plan_pots(&gs, true);
        assert_eq!(
            pots.iter()
                .map(|p| (p.gross, p.rake, p.refund_to))
                .collect::<Vec<_>>(),
            vec![(400, 20, None), (300, 5, None), (100, 0, Some(3))]
        );
        settle_pots(&mut gs);
        assert_eq!(gs.stacks.as_slice(), &[95, 194, 193, 793]);
        assert_eq!(gs.player_winnings.as_slice(), &[95, 194, 193, 193]);
        assert_eq!(gs.total_pot, 700);
        assert_eq!(gs.rake_collected, 25);
        assert_eq!(
            gs.starting_stacks.iter().sum::<Chips>(),
            gs.stacks.iter().sum::<Chips>() + gs.rake_collected
        );
    }
    #[test]
    fn folded_contributions_merge_rake_rounding_and_odd_cent() {
        let rake = RakeConfig::new(
            RakeRate::new(45, 1000).unwrap(),
            None,
            true,
            RakeRounding::HalfToEven,
        )
        .unwrap();
        let mut gs = tied_state(&[333, 333, 333, 2], &[0, 0, 0, 100], 0, rake);
        gs.player_active.disable(3);
        let pots = plan_pots(&gs, true);
        assert_eq!(pots.len(), 1);
        assert_eq!(pots[0].gross, 1001);
        assert_eq!(pots[0].rake, 45);
        settle_pots(&mut gs);
        assert_eq!(gs.player_winnings.as_slice(), &[318, 319, 319, 0]);
        assert_eq!(
            gs.starting_stacks.iter().sum::<Chips>(),
            gs.stacks.iter().sum::<Chips>() + 45
        );
    }
    #[test]
    fn no_flop_no_drop_uncontested_hand_and_one_cent_cap() {
        let rake = RakeConfig::new(
            RakeRate::new(5, 100).unwrap(),
            Some(1),
            true,
            RakeRounding::Ceil,
        )
        .unwrap();
        let mut gs = GameStateBuilder::new()
            .stacks([90, 90])
            .player_bet([10, 10])
            .blinds(2, 1)
            .round(Round::Preflop)
            .rake(rake)
            .build()
            .unwrap();
        gs.player_active.disable(1);
        settle_pots(&mut gs);
        assert_eq!(gs.player_winnings.as_slice(), &[20, 0]);
        assert_eq!(gs.rake_collected, 0);
        let mut gs = tied_state(&[333, 333, 333, 2], &[0, 0, 0, 100], 0, rake);
        gs.player_active.disable(3);
        settle_pots(&mut gs);
        assert_eq!(gs.player_winnings.as_slice(), &[333, 334, 333, 0]);
        assert_eq!(gs.rake_collected, 1);
    }
}
