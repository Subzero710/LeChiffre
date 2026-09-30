//! Exact poker money. One chip is one cent at currency boundaries.
//!
//! Abstract-chip tests keep their original magnitudes. Ratios and statistical
//! values remain floating point; only generated sizes cross this boundary.

pub use crate::Chips;

/// Apply an IEEE-754 sizing ratio, rounding to the nearest cent with ties up.
///
/// The binary value of the supplied `f32` is multiplied using integer arithmetic.
/// This preserves large chip amounts and gives every action generator the same
/// deterministic rounding. Sizes above the ledger range saturate at `Chips::MAX`;
/// callers still cap the result to the player's stack. Invalid ratios are errors
/// in sizing configuration and are rejected, rather than producing a bet.
pub fn apply_ratio(amount: Chips, ratio: f32) -> Chips {
    assert!(amount >= 0, "sizing amount must be non-negative");
    assert!(ratio.is_finite() && ratio >= 0.0, "invalid sizing ratio");
    let bits = ratio.to_bits();
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = (bits & 0x7fffff) | if exponent == 0 { 0 } else { 1 << 23 };
    let shift = if exponent == 0 { -149 } else { exponent - 150 };
    let product = i128::from(amount) * i128::from(mantissa);
    let rounded = if shift >= 0 {
        1_i128
            .checked_shl(shift as u32)
            .and_then(|scale| product.checked_mul(scale))
            .unwrap_or(i128::MAX)
    } else if -shift >= 127 {
        0
    } else {
        let divisor = 1_i128 << -shift;
        let quotient = product / divisor;
        let remainder = product % divisor;
        quotient + i128::from(remainder >= divisor - remainder)
    };
    rounded.min(i128::from(Chips::MAX)) as Chips
}

/// Split a pot in whole cents, giving odd chips clockwise from the button.
///
/// This follows the board-game rule in Poker TDA rule 21(A): the first winning
/// seat left of the button receives the first odd chip. Further remainder chips
/// proceed clockwise among winners. Seat indices must follow physical seat order.
/// Source: https://www.pokertda.com/view-poker-tda-rules/
pub fn split_pot(
    amount: Chips,
    winners: &[usize],
    dealer_idx: usize,
    num_players: usize,
) -> Vec<(usize, Chips)> {
    assert!(amount >= 0 && !winners.is_empty());
    assert!(dealer_idx < num_players);
    let mut ordered = winners.to_vec();
    assert!(ordered.iter().all(|&idx| idx < num_players));
    ordered.sort_unstable_by_key(|&idx| (idx + num_players - dealer_idx - 1) % num_players);
    assert!(ordered.windows(2).all(|pair| pair[0] != pair[1]));
    let share = amount / ordered.len() as Chips;
    let remainder = amount % ordered.len() as Chips;
    ordered
        .into_iter()
        .enumerate()
        .map(|(position, idx)| (idx, share + Chips::from((position as Chips) < remainder)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_ledger_and_ratio_rounding() {
        assert_eq!(12_345 + 1 - 12_346, 0);
        assert_eq!(apply_ratio(101, 0.5), 51);
        assert_eq!(apply_ratio(100, 0.5), 50);
        assert_eq!(
            apply_ratio(9_007_199_254_740_993, 1.0),
            9_007_199_254_740_993
        );
        assert_eq!(apply_ratio(Chips::MAX, 2.0), Chips::MAX);
        assert_eq!(apply_ratio(Chips::MAX, f32::from_bits(1)), 0);
    }

    #[test]
    fn split_single_two_and_three_way() {
        assert_eq!(split_pot(1001, &[2], 0, 4), vec![(2, 1001)]);
        assert_eq!(split_pot(1001, &[0, 2], 0, 4), vec![(2, 501), (0, 500)]);
        assert_eq!(
            split_pot(1001, &[0, 1, 2], 0, 3),
            vec![(1, 334), (2, 334), (0, 333)]
        );
        assert_eq!(split_pot(1, &[0, 1, 2], 2, 3), vec![(0, 1), (1, 0), (2, 0)]);
    }
}
