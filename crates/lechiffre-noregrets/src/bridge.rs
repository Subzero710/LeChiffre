//! Card encoding adapter; suit names are preserved across both APIs.
pub fn card_to_noregrets(card: u8) -> u8 {
    noregrets::engine::from_rs_card(rs_poker::core::Card::from(card))
}
