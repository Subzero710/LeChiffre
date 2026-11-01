use super::event::{ActionKind, Position, Rank, Street, Suit};

pub(crate) const STREET_CODEBOOK: &str = "0=preflop,1=flop,2=turn,3=river";
pub(crate) const POSITION_CODEBOOK: &str =
    "0=utg,1=hijack,2=cutoff,3=button,4=button_small_blind,5=small_blind,6=big_blind";
pub(crate) const ACTION_CODEBOOK: &str = "0=fold,1=check,2=call,3=bet,4=raise";
pub(crate) const RANK_CODEBOOK: &str = "2=two,3=three,4=four,5=five,6=six,7=seven,8=eight,9=nine,10=ten,11=jack,12=queen,13=king,14=ace";
pub(crate) const SUIT_CODEBOOK: &str = "1=clubs,2=diamonds,3=hearts,4=spades";

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

pub(crate) fn rank_id(rank: Rank) -> u8 {
    match rank {
        Rank::Two => 2,
        Rank::Three => 3,
        Rank::Four => 4,
        Rank::Five => 5,
        Rank::Six => 6,
        Rank::Seven => 7,
        Rank::Eight => 8,
        Rank::Nine => 9,
        Rank::Ten => 10,
        Rank::Jack => 11,
        Rank::Queen => 12,
        Rank::King => 13,
        Rank::Ace => 14,
    }
}

pub(crate) fn rank_from_id(id: u8) -> Option<Rank> {
    match id {
        2 => Some(Rank::Two),
        3 => Some(Rank::Three),
        4 => Some(Rank::Four),
        5 => Some(Rank::Five),
        6 => Some(Rank::Six),
        7 => Some(Rank::Seven),
        8 => Some(Rank::Eight),
        9 => Some(Rank::Nine),
        10 => Some(Rank::Ten),
        11 => Some(Rank::Jack),
        12 => Some(Rank::Queen),
        13 => Some(Rank::King),
        14 => Some(Rank::Ace),
        _ => None,
    }
}

pub(crate) fn suit_id(suit: Suit) -> u8 {
    match suit {
        Suit::Clubs => 1,
        Suit::Diamonds => 2,
        Suit::Hearts => 3,
        Suit::Spades => 4,
    }
}

pub(crate) fn suit_from_id(id: u8) -> Option<Suit> {
    match id {
        1 => Some(Suit::Clubs),
        2 => Some(Suit::Diamonds),
        3 => Some(Suit::Hearts),
        4 => Some(Suit::Spades),
        _ => None,
    }
}
