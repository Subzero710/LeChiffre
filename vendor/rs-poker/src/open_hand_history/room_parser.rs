//! Strict English, single-board NLHE cash text -> canonical OHH.
//! Formats and source fixtures are documented in docs/MONEY_AND_HISTORY.md.
use super::*;
use crate::{Chips, core::Card};
use chrono::{FixedOffset, NaiveDateTime, TimeZone, Utc};
use regex::Regex;
use std::{
    collections::{BTreeMap, HashMap},
    io::{BufRead, Cursor},
    sync::LazyLock,
};
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Room {
    PokerStars,
    CoinPoker,
    GGPoker,
}
impl Room {
    fn prefix(self) -> &'static str {
        match self {
            Self::PokerStars => "PokerStars Hand #",
            Self::CoinPoker => "CoinPoker Hand #",
            Self::GGPoker => "Poker Hand #",
        }
    }
    fn name(self) -> &'static str {
        match self {
            Self::PokerStars => "PokerStars",
            Self::CoinPoker => "CoinPoker",
            Self::GGPoker => "GGPoker",
        }
    }
}
#[derive(Debug, Clone, Copy, Default)]
pub struct ParserOptions {
    /// Required when an export omits its timezone. Never assume local or UTC.
    pub utc_offset: Option<FixedOffset>,
}
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("{room:?} hand {hand_id:?}, line {line}: {kind}: {text}")]
pub struct HandHistoryParseError {
    pub room: Room,
    pub hand_id: Option<String>,
    pub line: usize,
    pub text: String,
    pub kind: String,
}
pub trait HandHistoryParser {
    fn parse(&self, input: &str) -> Result<Vec<HandHistory>, HandHistoryParseError>;
}
macro_rules! room_parser {
    ($name:ident,$room:ident) => {
        #[derive(Debug, Clone, Copy, Default)]
        pub struct $name {
            pub options: ParserOptions,
        }
        impl HandHistoryParser for $name {
            fn parse(&self, input: &str) -> Result<Vec<HandHistory>, HandHistoryParseError> {
                RoomHandReader::new(Cursor::new(input), Room::$room, self.options).collect()
            }
        }
    };
}
room_parser!(PokerStarsParser, PokerStars);
room_parser!(CoinPokerParser, CoinPoker);
room_parser!(GGPokerParser, GGPoker);

/// Reads one hand at a time; the reader never buffers a database of histories.
/// A malformed hand is returned as an error and is not silently skipped.
pub struct RoomHandReader<R> {
    reader: R,
    room: Room,
    options: ParserOptions,
    pending: Option<(usize, String)>,
    line: usize,
    finished: bool,
}
impl<R: BufRead> RoomHandReader<R> {
    pub fn new(reader: R, room: Room, options: ParserOptions) -> Self {
        Self {
            reader,
            room,
            options,
            pending: None,
            line: 0,
            finished: false,
        }
    }
}
impl<R: BufRead> Iterator for RoomHandReader<R> {
    type Item = Result<HandHistory, HandHistoryParseError>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }
        let mut lines = Vec::new();
        if let Some(line) = self.pending.take() {
            lines.push(line);
        }
        loop {
            let mut text = String::new();
            match self.reader.read_line(&mut text) {
                Ok(0) => {
                    self.finished = true;
                    break;
                }
                Err(e) => {
                    self.finished = true;
                    return Some(Err(error(
                        self.room,
                        None,
                        self.line,
                        "",
                        &format!("I/O: {e}"),
                    )));
                }
                Ok(_) => {
                    self.line += 1;
                }
            }
            let text = text.trim().trim_start_matches('\u{feff}').to_string();
            if text.is_empty() {
                continue;
            }
            if text.starts_with(self.room.prefix()) && !lines.is_empty() {
                self.pending = Some((self.line, text));
                break;
            }
            if lines.is_empty() && !text.starts_with(self.room.prefix()) {
                self.finished = true;
                return Some(Err(error(
                    self.room,
                    None,
                    self.line,
                    &text,
                    "expected room hand header",
                )));
            }
            lines.push((self.line, text));
            if lines.len() > 20000 {
                self.finished = true;
                return Some(Err(error(
                    self.room,
                    None,
                    self.line,
                    "",
                    "hand exceeds 20,000-line limit",
                )));
            }
        }
        if lines.is_empty() {
            None
        } else {
            Some(parse_hand(self.room, self.options, &lines))
        }
    }
}
fn error(
    room: Room,
    id: Option<&str>,
    line: usize,
    text: &str,
    kind: &str,
) -> HandHistoryParseError {
    HandHistoryParseError {
        room,
        hand_id: id.map(str::to_string),
        line,
        text: text.into(),
        kind: kind.into(),
    }
}
fn re(pattern: &str) -> Regex {
    Regex::new(pattern).expect("constant parser regex")
}
static HEADER: LazyLock<Regex> = LazyLock::new(|| {
    re(
        r"^(?:PokerStars Hand|CoinPoker Hand|Poker Hand) #([^:]+):\s+(?:Hold'em No Limit|NLH)\s+\(([^)]+)\)\s*(?:-\s*)?(\d{4}/\d{2}/\d{2})\s+(\d{1,2}:\d{2}:\d{2})(?:\s+([A-Z]+))?(?:\s+\[.*\])?$",
    )
});
static TABLE: LazyLock<Regex> =
    LazyLock::new(|| re(r"^Table '(.+)' (\d+)-max Seat #(\d+) is the button$"));
static SEAT: LazyLock<Regex> = LazyLock::new(|| {
    re(r"^Seat (\d+): (.+) \(([^ )]+) in chips\)( is sitting out| out of hand)?$")
});
static RAISE: LazyLock<Regex> = LazyLock::new(|| re(r"^raises ([^ ]+) to ([^ ]+)$"));
static COLLECT: LazyLock<Regex> =
    LazyLock::new(|| re(r"^(.+) collected ([^ ]+) from (pot|main pot|side pot(?:-\d+)?)$"));
static RETURN: LazyLock<Regex> =
    LazyLock::new(|| re(r"^Uncalled bet \(([^)]+)\) returned to (.+)$"));

fn money(text: &str) -> Result<Chips, String> {
    let t = text.trim().trim_start_matches(['$', '₮']);
    let amount = amount::parse_currency(t).map_err(|e| e.to_string())?;
    if amount < 0 {
        Err("negative currency amount".into())
    } else {
        Ok(amount)
    }
}
fn cards(text: &str) -> Result<Vec<Card>, String> {
    text.split_whitespace()
        .map(|t| Card::try_from(t).map_err(|e| format!("invalid card: {e}")))
        .collect()
}
fn bracket_groups(text: &str) -> Result<Vec<Vec<Card>>, String> {
    let mut groups = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find('[') {
        let tail = &rest[start + 1..];
        let end = tail.find(']').ok_or("unclosed card list")?;
        groups.push(cards(tail[..end].trim())?);
        rest = &tail[end + 1..];
    }
    Ok(groups)
}
fn parse_hand(
    room: Room,
    options: ParserOptions,
    lines: &[(usize, String)],
) -> Result<HandHistory, HandHistoryParseError> {
    let (first, header) = &lines[0];
    let fail = |line: usize, text: &str, kind: &str| error(room, None, line, text, kind);
    let h = HEADER.captures(header).ok_or_else(|| {
        fail(
            *first,
            header,
            "unsupported cash NLHE header (tournaments/variants are rejected)",
        )
    })?;
    let id = h[1].to_string();
    let fail = |line: usize, text: &str, kind: &str| error(room, Some(&id), line, text, kind);
    let zone = match h.get(5).map(|v| v.as_str()) {
        Some("UTC" | "GMT") => FixedOffset::east_opt(0),
        Some("PDT") => FixedOffset::west_opt(7 * 3600),
        Some("PST") => FixedOffset::west_opt(8 * 3600),
        Some("EDT") => FixedOffset::west_opt(4 * 3600),
        Some("EST") => FixedOffset::west_opt(5 * 3600),
        None => options.utc_offset,
        _ => {
            return Err(fail(
                *first,
                header,
                "unverified timezone; use an export with an explicit UTC/fixed-offset zone",
            ));
        }
    }
    .ok_or_else(|| {
        fail(
            *first,
            header,
            "timezone missing; supply ParserOptions.utc_offset",
        )
    })?;
    let date = NaiveDateTime::parse_from_str(&format!("{} {}", &h[3], &h[4]), "%Y/%m/%d %H:%M:%S")
        .map_err(|e| fail(*first, header, &format!("invalid timestamp: {e}")))?;
    let start_date_utc = zone
        .from_local_datetime(&date)
        .single()
        .map(|d| d.with_timezone(&Utc));
    let stake = h[2].trim_end_matches(" USD");
    let stakes: Vec<&str> = stake.split('/').collect();
    if !(2..=3).contains(&stakes.len()) {
        return Err(fail(*first, header, "unsupported stake/ante syntax"));
    }
    let currency = if room == Room::CoinPoker
        && (stakes[0].starts_with('$') || stakes[0].starts_with('₮'))
    {
        // CoinPoker's published tables display dollar-denominated stakes while
        // the poker platform uses USDT as its primary game currency. Normalize
        // either export symbol to USDT without performing an exchange-rate conversion.
        "USDT"
    } else if stakes[0].starts_with('$') {
        "USD"
    } else {
        return Err(fail(*first, header, "unverified currency"));
    };
    let small_blind_amount = money(stakes[0]).map_err(|e| fail(*first, header, &e))?;
    let big_blind_amount = money(stakes[1]).map_err(|e| fail(*first, header, &e))?;
    let ante_amount = if stakes.len() == 3 {
        money(stakes[2]).map_err(|e| fail(*first, header, &e))?
    } else {
        0
    };
    let mut hh = HandHistory {
        spec_version: "1.4.7".into(),
        site_name: room.name().into(),
        network_name: room.name().into(),
        internal_version: "rs-poker-cents-1".into(),
        tournament: false,
        tournament_info: None,
        game_number: id.clone(),
        start_date_utc,
        table_name: String::new(),
        table_handle: None,
        table_skin: None,
        game_type: GameType::Holdem,
        bet_limit: Some(BetLimitObj {
            bet_type: BetType::NoLimit,
            bet_cap: 0,
        }),
        table_size: 0,
        currency: currency.into(),
        dealer_seat: 0,
        small_blind_amount,
        big_blind_amount,
        ante_amount,
        hero_player_id: None,
        players: Vec::new(),
        rounds: vec![RoundObj {
            id: 0,
            street: "Preflop".into(),
            cards: None,
            actions: Vec::new(),
        }],
        pots: Vec::new(),
        tournament_bounties: None,
    };
    let mut names = HashMap::new();
    let mut stacks = Vec::new();
    let mut contributions = Vec::new();
    let mut round_bets = Vec::new();
    let mut contenders = Vec::new();
    let mut collected: BTreeMap<u64, Vec<PlayerWinsObj>> = BTreeMap::new();
    let mut summary = false;
    let mut total = None;
    let mut rake = None;
    let mut board: Vec<Card> = Vec::new();
    let mut seq = 0;
    for (line, text) in &lines[1..] {
        let err = |kind: &str| fail(*line, text, kind);
        if let Some(t) = TABLE.captures(text) {
            if hh.table_size != 0 {
                return Err(err("duplicate table line"));
            }
            hh.table_name = t[1].into();
            hh.table_size = t[2].parse().map_err(|_| err("invalid table size"))?;
            hh.dealer_seat = t[3].parse().map_err(|_| err("invalid button seat"))?;
            continue;
        }
        if !summary && let Some(s) = SEAT.captures(text) {
            if !hh.rounds[0].actions.is_empty() {
                return Err(err("seat after betting began"));
            }
            let seat: u64 = s[1].parse().map_err(|_| err("invalid seat"))?;
            if names.contains_key(&s[2]) || hh.players.iter().any(|p| p.seat == seat) {
                return Err(err("duplicate player/seat"));
            }
            let stack = money(&s[3]).map_err(|e| err(&e))?;
            let idx = hh.players.len();
            names.insert(s[2].to_string(), idx);
            hh.players.push(PlayerObj {
                id: seat,
                seat,
                name: s[2].into(),
                display: None,
                starting_stack: stack,
                player_bounty: None,
                is_sitting_out: Some(s.get(4).is_some()),
            });
            stacks.push(stack);
            contributions.push(0);
            round_bets.push(0);
            contenders.push(s.get(4).is_none());
            continue;
        }
        if text == "*** SUMMARY ***" {
            summary = true;
            continue;
        }
        if text == "*** HOLE CARDS ***" {
            continue;
        }
        if text == "*** SHOW DOWN ***" || text == "*** SHOWDOWN ***" {
            hh.rounds.push(RoundObj {
                id: hh.rounds.len() as u64,
                street: "Showdown".into(),
                cards: None,
                actions: Vec::new(),
            });
            continue;
        }
        if text.starts_with("*** FLOP ***")
            || text.starts_with("*** TURN ***")
            || text.starts_with("*** RIVER ***")
        {
            let street = if text.starts_with("*** FLOP") {
                "Flop"
            } else if text.starts_with("*** TURN") {
                "Turn"
            } else {
                "River"
            };
            let groups = bracket_groups(text).map_err(|e| err(&e))?;
            let count = match street {
                "Flop" => 3,
                "Turn" => 4,
                _ => 5,
            };
            let all: Vec<Card> = groups.into_iter().flatten().collect();
            if all.len() != count || !all.starts_with(&board) {
                return Err(err("malformed or inconsistent street board"));
            }
            let new = all[board.len()..].to_vec();
            board = all;
            round_bets.fill(0);
            hh.rounds.push(RoundObj {
                id: hh.rounds.len() as u64,
                street: street.into(),
                cards: Some(new),
                actions: Vec::new(),
            });
            continue;
        }
        if text.starts_with("Total pot ") {
            if !summary || total.is_some() {
                return Err(err("duplicate/out-of-order summary pot"));
            }
            for (i, part) in text.split('|').enumerate() {
                let part = part.trim();
                if i == 0 {
                    total =
                        Some(money(part.trim_start_matches("Total pot ")).map_err(|e| err(&e))?);
                } else if let Some(v) = part.strip_prefix("Rake ") {
                    rake = Some(money(v).map_err(|e| err(&e))?);
                } else {
                    let (label, v) = part.rsplit_once(' ').ok_or_else(|| err("malformed fee"))?;
                    if !["Jackpot", "Bingo", "Fortune", "Tax", "Splash Fee"].contains(&label)
                        || money(v).map_err(|e| err(&e))? != 0
                    {
                        return Err(err(
                            "unsupported nonzero extra fee or unknown monetary summary field",
                        ));
                    }
                }
            }
            continue;
        }
        if let Some(b) = text.strip_prefix("Board ") {
            let groups = bracket_groups(b).map_err(|e| err(&e))?;
            if groups.len() != 1 || groups[0] != board {
                return Err(err("summary board disagrees with streets"));
            }
            continue;
        }
        if summary && text.starts_with("Seat ") {
            // Summary is redundant with action/collection records, but monetary
            // winner amounts and shown cards are still checked/captured below.
            let rest = text
                .split_once(": ")
                .ok_or_else(|| err("malformed summary seat"))?
                .1;
            let name = names
                .keys()
                .filter(|n| rest.starts_with(n.as_str()))
                .max_by_key(|n| n.len())
                .ok_or_else(|| err("unknown summary player"))?;
            let idx = names[name];
            if rest.contains("showed [") {
                let groups = bracket_groups(rest).map_err(|e| err(&e))?;
                if let Some(c) = groups.first() {
                    push_action(
                        &mut hh,
                        &mut seq,
                        idx,
                        Action::ShowsCards,
                        0,
                        false,
                        Some(c.clone()),
                    );
                }
            }
            if rest.contains("won (") || rest.contains("collected (") {
                let v = rest
                    .rsplit_once('(')
                    .ok_or_else(|| err("malformed summary winnings"))?
                    .1
                    .trim_end_matches(')');
                let observed = money(v).map_err(|e| err(&e))?;
                let recorded = collected
                    .values()
                    .flatten()
                    .filter(|w| w.player_id == hh.players[idx].id)
                    .try_fold(0_i64, |sum, w| sum.checked_add(w.win_amount))
                    .ok_or_else(|| err("winnings overflow"))?;
                if observed != recorded {
                    return Err(err(
                        "summary winner amount disagrees with collection records",
                    ));
                }
            } else if !rest.contains("folded")
                && !rest.contains("mucked")
                && !rest.contains("showed")
            {
                return Err(err("unsupported summary seat state"));
            }
            continue;
        }
        if text == "Hand was run once" {
            continue;
        }
        if text.starts_with("Hand was run ") {
            return Err(err("multiple runouts unsupported"));
        }
        if let Some(end) = text.strip_prefix("Game ended: ") {
            if room != Room::CoinPoker || end.len() < 19 {
                return Err(err("unsupported game-end metadata"));
            }
            continue;
        }
        if let Some(c) = COLLECT.captures(text) {
            let idx = *names
                .get(&c[1])
                .ok_or_else(|| err("unknown winning player"))?;
            let amount = money(&c[2]).map_err(|e| err(&e))?;
            let number = match &c[3] {
                "pot" | "main pot" => 0,
                "side pot" => 1,
                s => s
                    .strip_prefix("side pot-")
                    .ok_or_else(|| err("unknown pot"))?
                    .parse()
                    .map_err(|_| err("invalid side-pot number"))?,
            };
            collected.entry(number).or_default().push(PlayerWinsObj {
                player_id: hh.players[idx].id,
                win_amount: amount,
                cashout_amount: None,
                cashout_fee: None,
                bonus_amount: None,
                contributed_rake: None,
            });
            continue;
        }
        if let Some(c) = RETURN.captures(text) {
            let idx = *names
                .get(&c[2])
                .ok_or_else(|| err("unknown refund player"))?;
            let value = money(&c[1]).map_err(|e| err(&e))?;
            refund(idx, value, &mut stacks, &mut contributions, &mut round_bets)
                .map_err(|e| err(&e))?;
            continue;
        }
        if let Some(rest) = text.strip_prefix("Dealt to ") {
            let name = rest.split(" [").next().unwrap().trim();
            let idx = *names.get(name).ok_or_else(|| err("unknown dealt player"))?;
            let groups = bracket_groups(rest).map_err(|e| err(&e))?;
            let c = groups.first().cloned();
            if let Some(c) = &c {
                if c.len() != 2 {
                    return Err(err("NLHE requires two hole cards"));
                }
                hh.hero_player_id = Some(hh.players[idx].id);
            }
            push_action(&mut hh, &mut seq, idx, Action::DealtCards, 0, false, c);
            continue;
        }
        if let Some((name, verb)) = text.split_once(": ") {
            let idx = *names
                .get(name)
                .ok_or_else(|| err("unknown action player"))?;
            let allin = verb.ends_with(" and is all-in");
            let verb = verb.trim_end_matches(" and is all-in");
            if let Some(v) = verb.strip_prefix("RETURN ") {
                refund(
                    idx,
                    money(v).map_err(|e| err(&e))?,
                    &mut stacks,
                    &mut contributions,
                    &mut round_bets,
                )
                .map_err(|e| err(&e))?;
                continue;
            }
            let (action, value, cs) = if verb == "folds" {
                (Action::Fold, 0, None)
            } else if verb == "checks" {
                (Action::Check, 0, None)
            } else if let Some(v) = verb.strip_prefix("posts small blind ") {
                (Action::PostSmallBlind, money(v).map_err(|e| err(&e))?, None)
            } else if let Some(v) = verb.strip_prefix("posts big blind ") {
                (Action::PostBigBlind, money(v).map_err(|e| err(&e))?, None)
            } else if let Some(v) = verb
                .strip_prefix("posts the ante ")
                .or_else(|| verb.strip_prefix("posts ante "))
            {
                (Action::PostAnte, money(v).map_err(|e| err(&e))?, None)
            } else if let Some(v) = verb.strip_prefix("calls ") {
                (Action::Call, money(v).map_err(|e| err(&e))?, None)
            } else if let Some(v) = verb.strip_prefix("bets ") {
                (Action::Bet, money(v).map_err(|e| err(&e))?, None)
            } else if let Some(c) = RAISE.captures(verb) {
                let increment = money(&c[1]).map_err(|e| err(&e))?;
                let target = money(&c[2]).map_err(|e| err(&e))?;
                let mut current = round_bets.iter().copied().max().unwrap_or(0);
                if hh.rounds.last().is_some_and(|r| r.street == "Preflop") {
                    current = current.max(hh.big_blind_amount);
                }
                if target <= current || target - current != increment {
                    return Err(err("raise increment/target mismatch"));
                }
                let delta = target
                    .checked_sub(round_bets[idx])
                    .filter(|&v| v >= 0)
                    .ok_or_else(|| err("raise below own bet"))?;
                (Action::Raise, delta, None)
            } else if verb.starts_with("shows [") {
                (
                    Action::ShowsCards,
                    0,
                    Some(
                        bracket_groups(verb)
                            .map_err(|e| err(&e))?
                            .into_iter()
                            .next()
                            .ok_or_else(|| err("missing shown cards"))?,
                    ),
                )
            } else if verb == "mucks hand" || verb == "doesn't show hand" {
                (Action::MucksCards, 0, None)
            } else {
                return Err(err("unsupported or malformed action/monetary line"));
            };
            if action == Action::Fold {
                contenders[idx] = false;
            }
            if value > stacks[idx] {
                return Err(err("action exceeds remaining stack"));
            }
            stacks[idx] -= value;
            contributions[idx] = contributions[idx]
                .checked_add(value)
                .ok_or_else(|| err("contribution overflow"))?;
            if action != Action::PostAnte {
                round_bets[idx] = round_bets[idx]
                    .checked_add(value)
                    .ok_or_else(|| err("round bet overflow"))?;
            }
            if allin && stacks[idx] != 0 {
                return Err(err("all-in flag disagrees with exact stack"));
            }
            let is_allin = allin || (value > 0 && stacks[idx] == 0);
            push_action(&mut hh, &mut seq, idx, action, value, is_allin, cs);
            continue;
        }
        // Do not infer effects of timeout/disconnection/sitout/straddle/cashout,
        // splash drops, multiple boards, or otherwise unknown state lines.
        return Err(err(
            "unsupported line; no state-affecting input is discarded",
        ));
    }
    let last = lines.last().unwrap();
    let err = |kind: &str| fail(last.0, &last.1, kind);
    if hh.table_size == 0
        || hh.players.len() < 2
        || !hh.players.iter().any(|p| p.seat == hh.dealer_seat)
    {
        return Err(err("missing/invalid table, seats, or button"));
    }
    if stakes.len() == 2 {
        let posts = hh.rounds[0]
            .actions
            .iter()
            .filter(|a| a.action == Action::PostAnte)
            .collect::<Vec<_>>();
        if !posts.is_empty() {
            let amount=posts.iter().find(|a|!a.is_allin).map(|a|a.amount)
                .ok_or_else(||err("ante level unavailable: every ante is a short all-in and header omits the ante"))?;
            if amount == 0
                || posts
                    .iter()
                    .any(|a| (!a.is_allin && a.amount != amount) || a.amount > amount)
            {
                return Err(err("inconsistent ante postings"));
            }
            hh.ante_amount = amount;
        }
    }
    let total = total.ok_or_else(|| err("missing total pot"))?;
    let rake = rake.ok_or_else(|| err("missing rake"))?;
    let contribution_sum = contributions
        .iter()
        .try_fold(0_i64, |a, &v| a.checked_add(v))
        .ok_or_else(|| err("pot overflow"))?;
    let paid = collected
        .values()
        .flatten()
        .try_fold(0_i64, |a, w| a.checked_add(w.win_amount))
        .ok_or_else(|| err("winnings overflow"))?;
    if total != contribution_sum
        || total != paid.checked_add(rake).ok_or_else(|| err("pot overflow"))?
    {
        return Err(err(
            "exact contributions/pot/winnings/rake do not conserve money",
        ));
    }
    let mut levels = contributions
        .iter()
        .copied()
        .filter(|&v| v > 0)
        .collect::<Vec<_>>();
    levels.sort_unstable();
    levels.dedup();
    let mut previous = 0;
    let mut pots: Vec<(Chips, Vec<usize>)> = Vec::new();
    for level in levels {
        let contributors = contributions.iter().filter(|&&v| v >= level).count();
        let eligible = contributions
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| (v >= level && contenders[i]).then_some(i))
            .collect::<Vec<_>>();
        if contributors == 1 || eligible.is_empty() {
            return Err(err("unreturned excess or pot without eligible player"));
        }
        let gross = (level - previous) * contributors as Chips;
        if pots.last().is_some_and(|p| p.1 == eligible) {
            pots.last_mut().unwrap().0 += gross;
        } else {
            pots.push((gross, eligible));
        }
        previous = level;
    }
    if pots.len() != collected.len() {
        return Err(err("collection records do not identify each main/side pot"));
    }
    let mut inferred_rake = 0;
    for (number, (gross, eligible)) in pots.into_iter().enumerate() {
        let wins = collected
            .remove(&(number as u64))
            .ok_or_else(|| err("missing main/side-pot collection records"))?;
        if wins
            .iter()
            .any(|w| !eligible.iter().any(|&i| hh.players[i].id == w.player_id))
        {
            return Err(err("winner is ineligible for pot"));
        }
        let net: Chips = wins.iter().map(|w| w.win_amount).sum();
        if net > gross {
            return Err(err("pot winnings exceed gross contribution slice"));
        }
        let pot_rake = gross - net;
        inferred_rake += pot_rake;
        hh.pots.push(PotObj {
            number: number as u64,
            amount: gross,
            rake: Some(pot_rake),
            jackpot: None,
            player_wins: wins,
        });
    }
    if inferred_rake != rake {
        return Err(err(
            "per-pot gross minus winnings disagrees with recorded aggregate rake",
        ));
    }
    for round in &mut hh.rounds {
        for (number, action) in round.actions.iter_mut().enumerate() {
            action.action_number = number as u64 + 1;
        }
    }
    Ok(hh)
}
fn push_action(
    hh: &mut HandHistory,
    seq: &mut u64,
    idx: usize,
    action: Action,
    value: Chips,
    is_allin: bool,
    cards: Option<Vec<Card>>,
) {
    *seq += 1;
    hh.rounds.last_mut().unwrap().actions.push(ActionObj {
        action_number: *seq,
        player_id: hh.players[idx].id,
        action,
        amount: value,
        is_allin,
        cards,
    });
}
fn refund(
    idx: usize,
    value: Chips,
    stacks: &mut [Chips],
    contributions: &mut [Chips],
    bets: &mut [Chips],
) -> Result<(), String> {
    let matched = bets
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != idx)
        .map(|(_, v)| *v)
        .max()
        .ok_or("refund without opponent")?;
    if value <= 0 || bets[idx] - matched != value || contributions[idx] < value {
        return Err("refund does not equal unmatched street bet".into());
    }
    stacks[idx] += value;
    contributions[idx] -= value;
    bets[idx] -= value;
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn researched_room_fixtures_and_exact_currency() {
        let ps = PokerStarsParser::default()
            .parse(include_str!("../../tests/fixtures/rooms/pokerstars.txt"))
            .unwrap();
        assert_eq!(ps[0].game_number, "171562910425");
        assert_eq!(ps[0].pots[0].amount, 6);
        assert!(ps[0].rounds.iter().all(|round| {
            round
                .actions
                .iter()
                .enumerate()
                .all(|(i, action)| action.action_number == i as u64 + 1)
        }));
        let cp = CoinPokerParser::default()
            .parse(include_str!("../../tests/fixtures/rooms/coinpoker.txt"))
            .unwrap();
        assert_eq!(cp[0].currency, "USDT");
        assert_eq!(cp[0].pots[0].amount, 24);
        assert_eq!(cp[0].pots[0].rake, Some(1));

        let dollar_coin = include_str!("../../tests/fixtures/rooms/coinpoker.txt").replace('₮', "$");
        let dollar_coin = CoinPokerParser::default().parse(&dollar_coin).unwrap();
        assert_eq!(dollar_coin[0].currency, "USDT");
        let gg = GGPokerParser {
            options: ParserOptions {
                utc_offset: FixedOffset::east_opt(0),
            },
        }
        .parse(include_str!("../../tests/fixtures/rooms/ggpoker.txt"))
        .unwrap();
        assert_eq!(gg[0].pots[0].amount, 12);
        assert_eq!(gg[0].players[2].starting_stack, 387);
    }
    #[test]
    fn streaming_multihand_and_unknown_money_are_strict() {
        let fixture = include_str!("../../tests/fixtures/rooms/coinpoker.txt");
        let input = format!("{fixture}\n{fixture}");
        assert_eq!(CoinPokerParser::default().parse(&input).unwrap().len(), 2);
        let bad = fixture.replace(
            "*** SUMMARY ***",
            "Hero: pays mystery fee ₮0.01\n*** SUMMARY ***",
        );
        let error = CoinPokerParser::default().parse(&bad).unwrap_err();
        assert!(error.line > 0);
        assert_eq!(error.hand_id.as_deref(), Some("93840400001"));
        assert!(
            GGPokerParser::default()
                .parse(include_str!("../../tests/fixtures/rooms/ggpoker.txt"))
                .is_err()
        );
        assert!(
            CoinPokerParser::default()
                .parse(&fixture.replace("Rake ₮0.01", "Rake ₮0.001"))
                .is_err()
        );
    }

    #[test]
    fn overflowing_summary_winnings_return_a_structured_error() {
        let fixture = include_str!("../../tests/fixtures/rooms/coinpoker.txt");
        let bad = fixture.replace(
            "Hero collected ₮0.23 from pot",
            "Hero collected ₮92233720368547758.07 from pot\n\
             Hero collected ₮92233720368547758.07 from pot",
        );
        let error = CoinPokerParser::default().parse(&bad).unwrap_err();
        assert_eq!(error.kind, "winnings overflow");
        assert_eq!(error.hand_id.as_deref(), Some("93840400001"));
        assert!(error.line > 0);
    }
}
