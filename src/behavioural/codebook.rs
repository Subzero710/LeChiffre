use super::event::{ActionKind, Position, Rank, Street, Suit};

pub(crate) const STREET_CODEBOOK: &str = "0=preflop,1=flop,2=turn,3=river";
pub(crate) const POSITION_CODEBOOK: &str =
    "0=utg,1=hijack,2=cutoff,3=button,4=button_small_blind,5=small_blind,6=big_blind";
pub(crate) const ACTION_CODEBOOK: &str = "0=fold,1=check,2=call,3=bet,4=raise";
pub(crate) const RANK_CODEBOOK: &str =
    "2=two,3=three,4=four,5=five,6=six,7=seven,8=eight,9=nine,10=ten,11=jack,12=queen,13=king,14=ace";
pub(crate) const SUIT_CODEBOOK: &str = "1=clubs,2=diamonds,3=hearts,4=spades";

pub(crate) fn street_id(street: Street) -> u8 {
    match street {
        Street::Preflop => 0,
        Street::Flop => 1,
        Street::Turn => 2,
        Street::River => 3,
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

pub(crate) fn action_id(action: ActionKind) -> u8 {
    match action {
        ActionKind::Fold => 0,
        ActionKind::Check => 1,
        ActionKind::Call => 2,
        ActionKind::Bet => 3,
        ActionKind::Raise => 4,
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

pub(crate) fn suit_id(suit: Suit) -> u8 {
    match suit {
        Suit::Clubs => 1,
        Suit::Diamonds => 2,
        Suit::Hearts => 3,
        Suit::Spades => 4,
    }
}
