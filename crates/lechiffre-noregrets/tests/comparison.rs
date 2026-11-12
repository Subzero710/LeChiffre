//! Native card interoperability checks; no retired reduced-game dependency.
use lechiffre_noregrets::bridge::card_to_noregrets;

#[test]
fn cards_preserve_rank_suit_and_are_bijective() {
    let mut seen = [false; 52];
    for card in 0..52 {
        let n = card_to_noregrets(card);
        assert_eq!(n / 4, card % 13);
        assert_eq!(n % 4, card / 13);
        assert!(!seen[n as usize]);
        seen[n as usize] = true;
    }
}
