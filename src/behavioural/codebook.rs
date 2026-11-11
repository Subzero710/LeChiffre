use super::event::{ActionKind, Position, Street};

pub(crate) const STREET_CODEBOOK: &str = "0=preflop,1=flop,2=turn,3=river";
pub(crate) const POSITION_CODEBOOK: &str =
    "0=utg,1=hijack,2=cutoff,3=button,4=button_small_blind,5=small_blind,6=big_blind";
pub(crate) const ACTION_CODEBOOK: &str = "0=fold,1=check,2=call,3=bet,4=raise";

pub(crate) fn street_id(street: Street) -> u8 {
    match street {
        Street::Preflop => 0,
        Street::Flop => 1,
        Street::Turn => 2,
        Street::River => 3,
    }
}

pub(crate) fn street_from_id(id: u8) -> Option<Street> {
    match id {
        0 => Some(Street::Preflop),
        1 => Some(Street::Flop),
        2 => Some(Street::Turn),
        3 => Some(Street::River),
        _ => None,
    }
}

pub(crate) fn position_id(position: Position) -> u8 {
    match position {
        Position::UnderTheGun => 0,
        Position::Hijack => 1,
        Position::Cutoff => 2,
        Position::Button => 3,
        Position::ButtonSmallBlind => 4,
        Position::SmallBlind => 5,
        Position::BigBlind => 6,
    }
}

pub(crate) fn position_from_id(id: u8) -> Option<Position> {
    match id {
        0 => Some(Position::UnderTheGun),
        1 => Some(Position::Hijack),
        2 => Some(Position::Cutoff),
        3 => Some(Position::Button),
        4 => Some(Position::ButtonSmallBlind),
        5 => Some(Position::SmallBlind),
        6 => Some(Position::BigBlind),
        _ => None,
    }
}

pub(crate) fn action_id(action: ActionKind) -> u8 {
    match action {
        ActionKind::Fold => 0,
        ActionKind::Check => 1,
        ActionKind::Call => 2,
        ActionKind::Bet => 3,
        ActionKind::Raise => 4,
    }
}

pub(crate) fn action_from_id(id: u8) -> Option<ActionKind> {
    match id {
        0 => Some(ActionKind::Fold),
        1 => Some(ActionKind::Check),
        2 => Some(ActionKind::Call),
        3 => Some(ActionKind::Bet),
        4 => Some(ActionKind::Raise),
        _ => None,
    }
}
