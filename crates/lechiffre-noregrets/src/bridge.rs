//! Card encoding adapter between rs_poker and NoRegrets.
//! rs_poker uses suit-major IDs; NoRegrets uses rank-major IDs.
pub fn card_to_noregrets(card: u8) -> u8 {
    (card % 13) * 4 + card / 13
}
