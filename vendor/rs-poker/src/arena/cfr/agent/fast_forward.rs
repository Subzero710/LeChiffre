use crate::arena::{
    Chips,
    game_state::MAX_PLAYERS,
    pot::{PotSlice, plan_pots, winnings_for_ranks},
};
use rand::Rng;

use crate::arena::{GameState, action::AgentAction, game_state::Round};
use crate::core::{Card, Deck, PlayerBitSet, Rankable, SevenCardAccum, Suit, Value};

/// Upper bound on simultaneous contenders. `PlayerBitSet` is `u16`-backed, so a
/// table seats at most 16 players; contenders never exceed that. Lets the
/// showdown enumeration tally bases on the stack instead of in a heap `Vec`.
const MAX_CONTENDERS: usize = 16;

/// Upper bound on cards left in the deck (a full deck is 52). Lets board
/// enumeration collect the remaining cards into a stack buffer.
const DECK_LEN: usize = 52;

// -----------------------------------------------------------------------------
// Fast-forward helpers
//
// These free functions implement the cheap reward path used by
// `CFRAgent::compute_reward_fast_forward`. They mutate a cloned `GameState`
// directly: apply the candidate action, play out the rest of the hand
// assuming every further action is a check/call, and distribute a single pot.
// -----------------------------------------------------------------------------

/// Apply a validated generated action; illegal actions cannot change the reward silently.
pub(super) fn fast_forward_apply_action(gs: &mut GameState, action: &AgentAction) {
    let amount = match action {
        AgentAction::Fold => {
            gs.fold();
            return;
        }
        AgentAction::Call => gs.current_round_bet(),
        AgentAction::Bet(amount) => *amount,
        AgentAction::AllIn => gs.current_round_current_player_bet() + gs.current_player_stack(),
    };
    gs.do_bet(amount, false)
        .expect("illegal CFR fast-forward action");
}

/// Walk the game state forward through any remaining rounds. Betting rounds
/// are settled by having every still-needing-action player call; deal rounds
/// are settled by drawing fresh community cards from the remaining deck.
pub(super) fn fast_forward_run_to_showdown<R: Rng>(gs: &mut GameState, rng: &mut R) {
    let mut deck = fast_forward_remaining_deck(gs);
    loop {
        // If at most one player can contest the pot, further play is moot —
        // skip straight to the pot distribution step.
        let contenders = gs.player_active.count() + gs.player_all_in.count();
        if contenders <= 1 {
            return;
        }
        match gs.round {
            Round::Showdown | Round::Complete => return,
            Round::Starting | Round::Ante | Round::DealPreflop => gs.advance_round(),
            Round::DealFlop => {
                fast_forward_deal_community_cards(gs, &mut deck, 3, rng);
                gs.advance_round();
            }
            Round::DealTurn | Round::DealRiver => {
                fast_forward_deal_community_cards(gs, &mut deck, 1, rng);
                gs.advance_round();
            }
            Round::Preflop | Round::Flop | Round::Turn | Round::River => {
                fast_forward_everyone_calls(gs);
                gs.advance_round();
            }
        }
    }
}

/// Have every player whose `needs_action` bit is still set call the current
/// bet. Players who cannot cover the call still put in what they have and
/// are marked all-in by `do_bet`.
fn fast_forward_everyone_calls(gs: &mut GameState) {
    // Safety cap: at most one call per seat per round. `do_bet` disables the
    // to-act player's `needs_action` bit, so this loop must terminate in at
    // most `num_players` iterations.
    for _ in 0..gs.num_players {
        if gs.round_data.num_players_need_action() == 0 {
            break;
        }
        let to_match = gs.current_round_bet();
        gs.do_bet(to_match, false)
            .expect("illegal fast-forward call");
    }
}

/// Build a deck of cards that haven't been dealt yet by removing every known
/// card from a fresh 52-card deck. Each player's hand already contains the
/// shared board cards, so iterating hands covers the board implicitly.
fn fast_forward_remaining_deck(gs: &GameState) -> Deck {
    let mut deck = Deck::default();
    for hand in &gs.hands {
        for card in hand.iter() {
            deck.remove(&card);
        }
    }
    deck
}

/// Draw `num_cards` from the deck and add them to the board and to every
/// player's hand, mirroring what `HoldemSimulation::deal_comunity_cards` does.
fn fast_forward_deal_community_cards<R: Rng>(
    gs: &mut GameState,
    deck: &mut Deck,
    num_cards: usize,
    rng: &mut R,
) {
    for _ in 0..num_cards {
        let card = deck
            .deal(rng)
            .expect("valid NLHE deck must have the remaining community cards");
        gs.board.push(card);
        for hand in gs.hands.iter_mut() {
            hand.insert(card);
        }
    }
}

/// Settle exactly the same main/side pots, rake and odd chips as the full simulation.
pub(super) fn fast_forward_distribute_pot(gs: &mut GameState) {
    crate::arena::pot::settle_pots(gs);
}

/// Advance the game state through all remaining betting rounds (everyone
/// calls/checks) until a deal round or showdown is reached. This separates
/// the deterministic betting from the stochastic card dealing, allowing
/// the caller to enumerate board completions instead of sampling.
pub(super) fn fast_forward_advance_betting(gs: &mut GameState) {
    // Safety cap: at most 8 round advances to prevent infinite loops.
    for _ in 0..8 {
        match gs.round {
            // Stop at deal rounds — the caller will enumerate cards.
            Round::DealFlop | Round::DealTurn | Round::DealRiver => return,
            // Stop at terminal states.
            Round::Showdown | Round::Complete => return,
            // Skip non-betting advance rounds.
            Round::Starting | Round::Ante | Round::DealPreflop => gs.advance_round(),
            // Betting rounds: everyone calls, then advance.
            Round::Preflop | Round::Flop | Round::Turn | Round::River => {
                fast_forward_everyone_calls(gs);
                gs.advance_round();
            }
        }
    }
}

/// Reward for `player_idx` when at most one player can still contest the pot.
///
/// Called after `fast_forward_advance_betting` has moved bets into the pot.
/// Returns `None` when two or more players remain, signalling that a real
/// showdown (enumeration or sampling) is still required. The single-contender
/// branch includes `gs.stacks[player_idx]` because chips have already moved
/// from stacks into the pot.
fn fast_forward_uncontested_reward(
    gs: &GameState,
    contenders: PlayerBitSet,
    player_idx: usize,
) -> Option<f32> {
    if contenders.count() > 1 {
        return None;
    }
    let pots = plan_pots(gs, gs.board.len() >= 3);
    let mut ranks = [None; MAX_PLAYERS];
    for idx in contenders.ones() {
        ranks[idx] = Some(gs.hands[idx].rank());
    }
    let winnings = winnings_for_ranks(&pots, &ranks, gs, player_idx);
    Some((gs.stacks[player_idx] + winnings - gs.starting_stacks[player_idx]) as f32)
}

/// Enumerate all possible board completions and compute the exact expected
/// reward for `player_idx`.
///
/// This replaces the random-sample approach in `fast_forward_run_to_showdown`
/// with deterministic enumeration when the number of remaining cards is small
/// enough (0, 1, or 2 cards). The result is zero variance in the reward
/// signal, which dramatically improves CFR convergence quality.
///
/// # Arguments
///
/// * `gs` - Game state positioned at a deal round (or showdown) after all
///   betting is resolved. The `total_pot` must already reflect all bets.
/// * `player_idx` - The player whose reward we compute.
/// * `cards_needed` - Number of community cards still to be dealt (0, 1, or 2).
pub(super) fn fast_forward_enumerate_showdowns(
    gs: &GameState,
    player_idx: usize,
    cards_needed: usize,
) -> f32 {
    assert!(
        cards_needed <= 2,
        "enumeration accepts at most two remaining cards"
    );
    let contenders = gs.player_active | gs.player_all_in;

    // No contenders (everyone folded; pot already awarded) or a single
    // contender (wins regardless of board) needs no enumeration.
    if let Some(reward) = fast_forward_uncontested_reward(gs, contenders, player_idx) {
        return reward;
    }

    let gross_pot = gs.total_pot;
    if gross_pot <= 0 {
        return gs.player_reward(player_idx) as f32;
    }
    // If three cards still need to be dealt, every enumerated showdown reaches
    // a flop. Otherwise the current board already tells us whether a flop exists.
    let flop_dealt = gs.board.len() >= 3 || cards_needed >= 3;
    let pots = plan_pots(gs, flop_dealt);

    if cards_needed == 0 {
        // Board is complete — just evaluate the showdown.
        return evaluate_showdown_reward(gs, &contenders, &pots, player_idx);
    }

    // Collect the remaining deck into a stack buffer for indexed access.
    let deck = fast_forward_remaining_deck(gs);
    let mut card_buf = [Card::new(Value::Two, Suit::Spade); DECK_LEN];
    let mut rn = 0;
    for c in deck.iter() {
        card_buf[rn] = c;
        rn += 1;
    }
    let remaining = &card_buf[..rn];

    let starting_stack = gs.starting_stacks[player_idx];
    // After fast_forward_advance_betting, chips have moved from stacks into
    // the pot. `evaluate_with_extra_cards` returns only the player's share of
    // the pot (or 0), so the net reward is:
    //   remaining_stack + pot_share - starting_stack
    // The remaining_stack term accounts for the chips the player kept — without
    // it the reward would be off by exactly the unbet portion of their stack.
    let remaining_stack = gs.stacks[player_idx];
    let mut total_reward = 0.0f64;
    let mut count = 0u32;

    // Tally each contender's fixed hole+board cards once; every runout below
    // only folds in the 1-2 enumerated cards on a cheap copy of the tally.
    let mut acc_buf = [(0usize, SevenCardAccum::new()); MAX_CONTENDERS];
    let base = contender_accums(gs, &contenders, &mut acc_buf);
    if cards_needed == 1 {
        // Enumerate single card (river).
        for &card in remaining {
            let reward = combo_reward::<1>(base, player_idx, &pots, gs, [card]);
            total_reward += (remaining_stack + reward - starting_stack) as f64;
            count += 1;
        }
    } else {
        // cards_needed == 2: enumerate all unordered pairs (turn + river).
        // Card order doesn't matter for hand evaluation, so visit each once.
        for i in 0..remaining.len() {
            for j in (i + 1)..remaining.len() {
                let reward =
                    combo_reward::<2>(base, player_idx, &pots, gs, [remaining[i], remaining[j]]);
                total_reward += (remaining_stack + reward - starting_stack) as f64;
                count += 1;
            }
        }
    }

    (total_reward / f64::from(count)) as f32
}

/// Number of random flop samples to draw when 3 community cards remain.
/// For each sampled flop, all turn+river combinations are enumerated
/// exhaustively (~C(44,2) ≈ 946 evals per flop). This hybrid approach
/// gives much lower variance than a single random runout at modest cost.
pub(super) const FLOP_SAMPLES: usize = 3;

/// Sample random flops and enumerate all turn+river completions for each.
///
/// When 3 community cards remain (pre-flop fast-forward), full enumeration
/// costs C(47,3) ≈ 16K evaluations — too expensive per action. Instead we
/// sample `FLOP_SAMPLES` random flop combinations and for each one
/// exhaustively enumerate all C(remaining,2) turn+river pairs. This
/// eliminates variance from 2 of the 3 unknown cards while keeping cost
/// at roughly `FLOP_SAMPLES × 1000` evaluations.
pub(super) fn fast_forward_sample_flop_enumerate_runout<R: Rng>(
    gs: &GameState,
    player_idx: usize,
    rng: &mut R,
) -> f32 {
    fast_forward_sample_flop_enumerate_runout_n(gs, player_idx, rng, FLOP_SAMPLES)
}

/// Inner implementation parameterized by sample count for benchmarking.
pub(super) fn fast_forward_sample_flop_enumerate_runout_n<R: Rng>(
    gs: &GameState,
    player_idx: usize,
    rng: &mut R,
    num_samples: usize,
) -> f32 {
    assert!(
        num_samples > 0,
        "flop expectation requires at least one sample"
    );
    let contenders = gs.player_active | gs.player_all_in;

    // No contenders (everyone folded) or a single contender (wins regardless of
    // board) needs no sampling.
    if let Some(reward) = fast_forward_uncontested_reward(gs, contenders, player_idx) {
        return reward;
    }

    let gross_pot = gs.total_pot;
    if gross_pot <= 0 {
        return gs.player_reward(player_idx) as f32;
    }
    // This path explicitly samples a flop, so no-flop-no-drop never suppresses
    // rake for the resulting showdown.
    let pots = plan_pots(gs, true);

    let mut deck = fast_forward_remaining_deck(gs);
    let starting_stack = gs.starting_stacks[player_idx];
    // See comment in fast_forward_enumerate_showdowns — remaining_stack
    // accounts for unbet chips after fast_forward_advance_betting.
    let remaining_stack = gs.stacks[player_idx];
    let mut total_reward = 0.0f64;
    let mut total_count = 0u64;

    let mut acc_buf = [(0usize, SevenCardAccum::new()); MAX_CONTENDERS];
    let mut card_buf = [Card::new(Value::Two, Suit::Spade); DECK_LEN];
    for _ in 0..num_samples {
        // Deal 3 random flop cards from the deck.
        let mut flop = [Card::new(Value::Two, Suit::Spade); 3];
        let mut fc = 0;
        while fc < 3 {
            match deck.deal(rng) {
                Some(c) => {
                    flop[fc] = c;
                    fc += 1;
                }
                // Not enough cards — shouldn't happen in practice.
                None => break,
            }
        }
        if fc < 3 {
            break;
        }

        // Tally each contender's hole+flop cards once for this sampled flop.
        // No GameState clone is needed: `combo_reward` only reads the per-hand
        // tally, so we fold the hole cards and the flop straight into the
        // accumulator and skip the (allocation-heavy) board/hand mutation.
        let mut n = 0;
        for idx in contenders.ones() {
            let mut acc = SevenCardAccum::new();
            for c in gs.hands[idx].iter() {
                acc.add(c);
            }
            for &c in &flop {
                acc.add(c);
            }
            acc_buf[n] = (idx, acc);
            n += 1;
        }
        let base = &acc_buf[..n];

        // The flop cards were just drawn from `deck`, so it already holds
        // exactly the cards available for the turn/river. Collect them into a
        // stack buffer and enumerate all turn+river completions on top.
        let mut rn = 0;
        for c in deck.iter() {
            card_buf[rn] = c;
            rn += 1;
        }
        let remaining = &card_buf[..rn];
        for i in 0..remaining.len() {
            for j in (i + 1)..remaining.len() {
                let reward =
                    combo_reward::<2>(base, player_idx, &pots, gs, [remaining[i], remaining[j]]);
                total_reward += (remaining_stack + reward - starting_stack) as f64;
                total_count += 1;
            }
        }

        // Put flop cards back in the deck for the next sample.
        for &card in &flop {
            deck.insert(card);
        }
    }

    if total_count == 0 {
        return gs.player_reward(player_idx) as f32;
    }

    (total_reward / total_count as f64) as f32
}

/// Precompute per-contender ranking state for the fixed hole+board cards.
///
/// Each contender's `gs.hands[idx]` already holds its hole cards plus every
/// community card dealt so far. Those cards are constant across a board
/// enumeration, so we tally them once into a [`SevenCardAccum`]; each
/// enumerated runout then only folds its 1-2 extra cards into a cheap `Copy`
/// of the tally rather than re-iterating the whole hand.
fn contender_accums<'b>(
    gs: &GameState,
    contenders: &PlayerBitSet,
    buf: &'b mut [(usize, SevenCardAccum); MAX_CONTENDERS],
) -> &'b [(usize, SevenCardAccum)] {
    let mut n = 0;
    for idx in contenders.ones() {
        let mut acc = SevenCardAccum::new();
        for c in gs.hands[idx].iter() {
            acc.add(c);
        }
        buf[n] = (idx, acc);
        n += 1;
    }
    &buf[..n]
}

/// Reward for `player_idx` on one board completion, given precomputed base
/// accumulators. Folds `extra` into each contender's tally, finds the best
/// rank, and returns `player_idx`'s share of `pot` (0 if not a winner).
///
/// `N` is the number of extra cards to fold in (0, 1, or 2). Making it a
/// const generic monomorphizes the inner `for` over a fixed-length array, so
/// the 1- and 2-card runout loops unroll the `add` calls and skip the slice
/// bounds checks the old `&[Card]` signature paid on every contender.
///
/// Equivalent to [`find_winners`] over hands extended with `extra` followed by
/// an even split of `pot`, but it allocates nothing and re-ranks only the
/// varying cards.
#[inline]
fn combo_reward<const N: usize>(
    base: &[(usize, SevenCardAccum)],
    player_idx: usize,
    pots: &[PotSlice],
    gs: &GameState,
    extra: [crate::core::Card; N],
) -> Chips {
    let mut ranks = [None; MAX_PLAYERS];
    for &(idx, base_acc) in base {
        let mut acc = base_acc;
        for card in extra {
            acc.add(card);
        }
        ranks[idx] = Some(acc.rank());
    }
    winnings_for_ranks(pots, &ranks, gs, player_idx)
}

/// Evaluate showdown with the current board (no extra cards).
/// Returns `remaining_stack + pot_share - starting_stack` to account for
/// chips already moved from stacks into the pot by `fast_forward_advance_betting`.
fn evaluate_showdown_reward(
    gs: &GameState,
    contenders: &PlayerBitSet,
    pots: &[PotSlice],
    player_idx: usize,
) -> f32 {
    let mut acc_buf = [(0usize, SevenCardAccum::new()); MAX_CONTENDERS];
    let base = contender_accums(gs, contenders, &mut acc_buf);
    let reward = combo_reward::<0>(base, player_idx, &pots, gs, []);
    (gs.stacks[player_idx] + reward - gs.starting_stacks[player_idx]) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::{GameStateBuilder, RakeConfig};

    #[test]
    fn distribute_pot_applies_rake_in_fast_forward() {
        let rake = RakeConfig::new(
            crate::arena::RakeRate::new(10, 100).unwrap(),
            None,
            false,
            crate::arena::RakeRounding::HalfToEven,
        )
        .unwrap();
        let mut gs = GameStateBuilder::new()
            .stacks(vec![100, 100])
            .big_blind(2)
            .rake(rake)
            .build()
            .unwrap();

        gs.stacks[0] = 90;
        gs.stacks[1] = 90;
        gs.total_pot = 20;
        gs.player_bet = vec![10, 10].into();
        gs.player_active.disable(1);

        fast_forward_distribute_pot(&mut gs);

        assert_eq!(gs.rake_collected, 2);
        assert_eq!(gs.stacks[0], 108);
        assert_eq!(gs.total_pot, 20);
    }
}
