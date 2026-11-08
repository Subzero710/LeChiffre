use std::collections::{HashMap, HashSet};
use std::fmt;

use rs_poker::arena::RakeConfig;
use rs_poker::core::Card;
use rs_poker::open_hand_history::{
    Action, ActionObj, BetLimitObj, BetType, GameType, HandHistory, PlayerObj, PlayerWinsObj,
    PotObj, RoundObj,
};
use serde_json::Value;

use super::hand_history::{HandHistoryAdapter, HandHistoryAdapterError};
use super::{AdaptedHand, PlayerId};
use crate::behavioural::{HandId, LegalActionSpace};

type AdaptedHandWithLegality = (AdaptedHand, HashMap<(HandId, u32), LegalActionSpace>);

#[derive(Debug, Clone, PartialEq)]
pub struct PokerNowImport {
    pub hands: Vec<AdaptedHand>,
    pub rejected: Vec<PokerNowRejectedHand>,
    /// Exact rules-engine legal spaces for each source decision, keyed by
    /// canonical hand ID and sequence index. Never a model input/target.
    pub exact_legal_actions: HashMap<(HandId, u32), LegalActionSpace>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PokerNowRejectedHand {
    pub source_hand_id: String,
    pub reason: String,
}

pub struct PokerNowAdapter;

impl PokerNowAdapter {
    pub fn adapt_csv<F>(
        session_id: &str,
        identity_namespace: &str,
        text: &str,
        mut normalize_name: F,
    ) -> Result<PokerNowImport, PokerNowAdapterError>
    where
        F: FnMut(&str) -> Option<String>,
    {
        let mut rows = parse_csv(text)?;
        rows.sort_unstable_by_key(|row| row.order);

        let mut hands = Vec::new();
        let mut rejected = Vec::new();
        let mut exact_legal_actions = HashMap::new();
        let mut current = Vec::new();

        for row in rows {
            if row.entry.starts_with("-- starting hand #") && !current.is_empty() {
                import_csv_hand(
                    session_id,
                    identity_namespace,
                    &current,
                    &mut normalize_name,
                    &mut hands,
                    &mut rejected,
                    &mut exact_legal_actions,
                );
                current.clear();
            }
            if !current.is_empty() || row.entry.starts_with("-- starting hand #") {
                current.push(row);
            }
            if current
                .last()
                .is_some_and(|row| row.entry.starts_with("-- ending hand #"))
            {
                import_csv_hand(
                    session_id,
                    identity_namespace,
                    &current,
                    &mut normalize_name,
                    &mut hands,
                    &mut rejected,
                    &mut exact_legal_actions,
                );
                current.clear();
            }
        }
        if !current.is_empty() {
            import_csv_hand(
                session_id,
                identity_namespace,
                &current,
                &mut normalize_name,
                &mut hands,
                &mut rejected,
                &mut exact_legal_actions,
            );
        }

        Ok(PokerNowImport {
            hands,
            rejected,
            exact_legal_actions,
        })
    }

    pub fn adapt_pokerlater_json<F>(
        session_id: &str,
        identity_namespace: &str,
        text: &str,
        mut normalize_name: F,
    ) -> Result<PokerNowImport, PokerNowAdapterError>
    where
        F: FnMut(&str) -> Option<String>,
    {
        let root: Value = serde_json::from_str(text)
            .map_err(|error| PokerNowAdapterError::InvalidPokerLaterJson(error.to_string()))?;
        if root["format"].as_str() != Some("pokerlater-session-log")
            || root["version"].as_u64() != Some(1)
        {
            return Err(PokerNowAdapterError::UnsupportedPokerLaterFormat);
        }

        let small_blind = json_chips(&root["stakes"]["smallBlind"])?;
        let big_blind = json_chips(&root["stakes"]["bigBlind"])?;
        let raw_hands = root["hands"]
            .as_array()
            .ok_or(PokerNowAdapterError::MissingPokerLaterHands)?;

        let mut hands = Vec::new();
        let mut rejected = Vec::new();
        let mut exact_legal_actions = HashMap::new();
        for raw in raw_hands {
            if raw["final"]["phase"].as_str() != Some("complete") {
                continue;
            }
            let source_hand_id = raw["handId"]
                .as_str()
                .filter(|value| !value.is_empty())
                .map(str::to_owned)
                .unwrap_or_else(|| raw["handNo"].to_string());
            match build_pokerlater_hand(
                session_id,
                identity_namespace,
                raw,
                small_blind,
                big_blind,
                &mut normalize_name,
            )
            .and_then(adapt_parsed_hand)
            {
                Ok((hand, legal)) => {
                    exact_legal_actions.extend(legal);
                    hands.push(hand);
                }
                Err(error) => rejected.push(PokerNowRejectedHand {
                    source_hand_id,
                    reason: error.to_string(),
                }),
            }
        }

        Ok(PokerNowImport {
            hands,
            rejected,
            exact_legal_actions,
        })
    }
}

fn import_csv_hand<F>(
    session_id: &str,
    identity_namespace: &str,
    rows: &[CsvRow],
    normalize_name: &mut F,
    hands: &mut Vec<AdaptedHand>,
    rejected: &mut Vec<PokerNowRejectedHand>,
    exact_legal_actions: &mut HashMap<(HandId, u32), LegalActionSpace>,
) where
    F: FnMut(&str) -> Option<String>,
{
    let source_hand_id = rows
        .first()
        .and_then(|row| parse_start(&row.entry).ok())
        .map(|start| start.source_hand_id)
        .unwrap_or_else(|| "unknown".to_string());

    match build_csv_hand(session_id, identity_namespace, rows, normalize_name)
        .and_then(adapt_parsed_hand)
    {
        Ok((hand, legal)) => {
            exact_legal_actions.extend(legal);
            hands.push(hand);
        }
        Err(error) => rejected.push(PokerNowRejectedHand {
            source_hand_id,
            reason: error.to_string(),
        }),
    }
}

fn adapt_parsed_hand(parsed: ParsedHand) -> Result<AdaptedHandWithLegality, PokerNowHandError> {
    let legal = HandHistoryAdapter::exact_legality_index(&parsed.hand, RakeConfig::none())?;
    let hand = HandHistoryAdapter::adapt_with_observations(
        &parsed.hand,
        RakeConfig::none(),
        &parsed.decision_times_ms,
        &parsed.player_ids,
    )?;
    Ok((hand, legal))
}

#[derive(Debug, Clone)]
struct CsvRow {
    entry: String,
    order: i64,
}

fn parse_csv(text: &str) -> Result<Vec<CsvRow>, PokerNowAdapterError> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(false)
        .from_reader(text.as_bytes());
    let headers = reader
        .headers()
        .map_err(|error| PokerNowAdapterError::InvalidCsv(error.to_string()))?;
    if headers.iter().collect::<Vec<_>>() != ["entry", "at", "order"] {
        return Err(PokerNowAdapterError::InvalidCsv(
            "expected entry,at,order columns".to_string(),
        ));
    }
    let mut rows = Vec::new();
    for result in reader.records() {
        let record = result.map_err(|error| PokerNowAdapterError::InvalidCsv(error.to_string()))?;
        let entry = record
            .get(0)
            .ok_or_else(|| PokerNowAdapterError::InvalidCsv("missing entry".into()))?;
        let order = record
            .get(2)
            .ok_or_else(|| PokerNowAdapterError::InvalidCsv("missing order".into()))?
            .parse::<i64>()
            .map_err(|error| PokerNowAdapterError::InvalidCsv(error.to_string()))?;
        rows.push(CsvRow {
            entry: entry.to_owned(),
            order,
        });
    }
    Ok(rows)
}

#[derive(Debug)]
struct ParsedHand {
    hand: HandHistory,
    decision_times_ms: Vec<Option<u32>>,
    player_ids: HashMap<u64, PlayerId>,
}

#[derive(Debug)]
struct StartLine {
    source_hand_id: String,
    dealer_name: String,
}

fn parse_start(entry: &str) -> Result<StartLine, PokerNowHandError> {
    let hand_no = entry
        .strip_prefix("-- starting hand #")
        .and_then(|rest| rest.split_whitespace().next())
        .ok_or(PokerNowHandError::MissingHandNumber)?;
    let source_hand_id = entry
        .split("(id: ")
        .nth(1)
        .and_then(|rest| rest.split(')').next())
        .filter(|value| !value.is_empty())
        .unwrap_or(hand_no)
        .to_string();
    let dealer_name = entry
        .split("(dealer: \"")
        .nth(1)
        .and_then(|rest| rest.split("\")").next())
        .ok_or(PokerNowHandError::MissingDealer)?
        .to_string();
    Ok(StartLine {
        source_hand_id,
        dealer_name,
    })
}

fn build_csv_hand<F>(
    session_id: &str,
    identity_namespace: &str,
    rows: &[CsvRow],
    normalize_name: &mut F,
) -> Result<ParsedHand, PokerNowHandError>
where
    F: FnMut(&str) -> Option<String>,
{
    let start = rows
        .iter()
        .find(|row| row.entry.starts_with("-- starting hand #"))
        .ok_or(PokerNowHandError::MissingStart)
        .and_then(|row| parse_start(&row.entry))?;
    let stacks = rows
        .iter()
        .find(|row| row.entry.starts_with("Player stacks:"))
        .ok_or(PokerNowHandError::MissingStacks)?;
    let parsed_players = parse_stack_line(&stacks.entry)?;
    if !(2..=6).contains(&parsed_players.len()) {
        return Err(PokerNowHandError::UnsupportedPlayerCount(
            parsed_players.len(),
        ));
    }

    let mut players = Vec::with_capacity(parsed_players.len());
    let mut by_name = HashMap::new();
    let mut player_ids = HashMap::new();
    let mut starting_stacks = HashMap::new();
    let mut canonical_seen = HashSet::new();

    for (index, parsed) in parsed_players.iter().enumerate() {
        let source_id = index as u64 + 1;
        let display_name = display_name(&parsed.name);
        let canonical = normalize_name(display_name)
            .ok_or_else(|| PokerNowHandError::UnknownPlayer(display_name.to_string()))?;
        let player_id = PlayerId::from_source(identity_namespace, &canonical);
        if !canonical_seen.insert(player_id.clone()) {
            return Err(PokerNowHandError::DuplicatePlayerIdentity(
                player_id.to_string(),
            ));
        }
        by_name.insert(parsed.name.clone(), source_id);
        player_ids.insert(source_id, player_id);
        starting_stacks.insert(source_id, parsed.stack);
        players.push(PlayerObj {
            id: source_id,
            seat: parsed.seat,
            name: parsed.name.clone(),
            display: Some(display_name.to_string()),
            starting_stack: parsed.stack,
            player_bounty: None,
            is_sitting_out: None,
        });
    }

    let dealer_id = *by_name
        .get(&start.dealer_name)
        .ok_or_else(|| PokerNowHandError::UnknownActor(start.dealer_name.clone()))?;
    let dealer_seat = players
        .iter()
        .find(|player| player.id == dealer_id)
        .map(|player| player.seat)
        .ok_or_else(|| PokerNowHandError::UnknownActor(start.dealer_name.clone()))?;

    let mut rounds = vec![empty_round(0, "Preflop")];
    let mut round_index = 0usize;
    let mut street_commitment: HashMap<u64, i64> = players.iter().map(|p| (p.id, 0)).collect();
    let mut total_commitment: HashMap<u64, i64> = players.iter().map(|p| (p.id, 0)).collect();
    let mut ready_ms = None;
    let mut decision_times_ms = Vec::new();
    let mut small_blind = None;
    let mut big_blind = None;
    let mut ante = 0;
    let mut awards: Vec<(u64, i64)> = Vec::new();
    let mut folded = HashSet::new();

    for row in rows {
        let entry = normalize_mojibake(&row.entry);
        let at_ms = row.order / 100;

        if entry == "Dead Small Blind" {
            return Err(PokerNowHandError::UnsupportedDeadBlind);
        }
        if let Some(cards) = parse_board(&entry)? {
            let (street, new_cards) = cards;
            rounds.push(RoundObj {
                id: rounds.len() as u64,
                street: street.to_string(),
                cards: Some(new_cards),
                actions: Vec::new(),
            });
            round_index = rounds.len() - 1;
            street_commitment.values_mut().for_each(|value| *value = 0);
            ready_ms = Some(at_ms);
            continue;
        }

        let Some((actor_name, rest)) = parse_actor_line(&entry) else {
            if let Some((name, amount)) = parse_uncalled_return(&entry) {
                let player = by_name
                    .get(name)
                    .copied()
                    .ok_or_else(|| PokerNowHandError::UnknownActor(name.to_string()))?;
                subtract_commitment(
                    player,
                    amount,
                    &mut street_commitment,
                    &mut total_commitment,
                )?;
            }
            continue;
        };
        let actor = *by_name
            .get(actor_name)
            .ok_or_else(|| PokerNowHandError::UnknownActor(actor_name.to_string()))?;

        if let Some(amount) = rest
            .strip_prefix("posts a small blind of ")
            .and_then(parse_i64_prefix)
        {
            small_blind.get_or_insert(amount);
            add_forced_action(
                &mut rounds[0],
                actor,
                Action::PostSmallBlind,
                amount,
                &starting_stacks,
                &mut street_commitment,
                &mut total_commitment,
            )?;
            ready_ms = Some(at_ms);
            continue;
        }
        if let Some(amount) = rest
            .strip_prefix("posts a big blind of ")
            .and_then(parse_i64_prefix)
        {
            big_blind.get_or_insert(amount);
            add_forced_action(
                &mut rounds[0],
                actor,
                Action::PostBigBlind,
                amount,
                &starting_stacks,
                &mut street_commitment,
                &mut total_commitment,
            )?;
            ready_ms = Some(at_ms);
            continue;
        }
        if let Some(amount) = rest
            .strip_prefix("posts an ante of ")
            .and_then(parse_i64_prefix)
        {
            if ante == 0 {
                ante = amount;
            } else if ante != amount {
                return Err(PokerNowHandError::InconsistentAnte);
            }
            add_forced_action(
                &mut rounds[0],
                actor,
                Action::PostAnte,
                amount,
                &starting_stacks,
                &mut street_commitment,
                &mut total_commitment,
            )?;
            ready_ms = Some(at_ms);
            continue;
        }
        if rest.starts_with("posts a straddle of ") {
            return Err(PokerNowHandError::UnsupportedStraddle);
        }
        if let Some(amount) = rest.strip_prefix("collected ").and_then(parse_i64_prefix) {
            awards.push((actor, amount));
            continue;
        }

        let action = if rest == "folds" {
            Some((Action::Fold, 0))
        } else if rest == "checks" {
            Some((Action::Check, 0))
        } else if let Some(target) = rest.strip_prefix("calls ").and_then(parse_i64_prefix) {
            let own = *street_commitment.get(&actor).unwrap_or(&0);
            if target <= own {
                return Err(PokerNowHandError::InvalidCallTarget);
            }
            Some((Action::Call, target - own))
        } else if let Some(amount) = rest.strip_prefix("bets ").and_then(parse_i64_prefix) {
            Some((Action::Bet, amount))
        } else if let Some(target) = rest.strip_prefix("raises to ").and_then(parse_i64_prefix) {
            let own = *street_commitment.get(&actor).unwrap_or(&0);
            if target <= own {
                return Err(PokerNowHandError::InvalidRaiseTarget);
            }
            Some((Action::Raise, target - own))
        } else {
            None
        };

        if let Some((kind, amount)) = action {
            if kind == Action::Fold {
                folded.insert(actor);
            }
            let decision_time = observed_latency(ready_ms, at_ms)?;
            let all_in = add_voluntary_action(
                &mut rounds[round_index],
                actor,
                kind,
                amount,
                &starting_stacks,
                &mut street_commitment,
                &mut total_commitment,
            )?;
            if rest.contains("and go all in") && !all_in {
                return Err(PokerNowHandError::InconsistentAllIn);
            }
            decision_times_ms.push(Some(decision_time));
            ready_ms = Some(at_ms);
        }
    }

    let small_blind = small_blind.ok_or(PokerNowHandError::MissingSmallBlind)?;
    let big_blind = big_blind.ok_or(PokerNowHandError::MissingBigBlind)?;
    let pots = build_pots(&total_commitment, &folded, &awards)?;
    let table_size = players.iter().map(|player| player.seat).max().unwrap_or(0);

    Ok(ParsedHand {
        hand: HandHistory {
            spec_version: "1.4.7".to_string(),
            site_name: "PokerNow".to_string(),
            network_name: "PokerNow".to_string(),
            internal_version: "LeChiffre-pokernow-1".to_string(),
            tournament: false,
            tournament_info: None,
            game_number: format!("{session_id}:{}", start.source_hand_id),
            start_date_utc: None,
            table_name: session_id.to_string(),
            table_handle: None,
            table_skin: None,
            game_type: GameType::Holdem,
            bet_limit: Some(BetLimitObj {
                bet_type: BetType::NoLimit,
                bet_cap: 0,
            }),
            table_size,
            currency: "CHIPS".to_string(),
            dealer_seat,
            small_blind_amount: small_blind,
            big_blind_amount: big_blind,
            ante_amount: ante,
            hero_player_id: None,
            players,
            rounds,
            pots,
            tournament_bounties: None,
        },
        decision_times_ms,
        player_ids,
    })
}

fn build_pokerlater_hand<F>(
    session_id: &str,
    identity_namespace: &str,
    raw: &Value,
    small_blind: i64,
    big_blind: i64,
    normalize_name: &mut F,
) -> Result<ParsedHand, PokerNowHandError>
where
    F: FnMut(&str) -> Option<String>,
{
    let seats = raw["seats"]
        .as_array()
        .ok_or(PokerNowHandError::MissingStacks)?;
    if !(2..=6).contains(&seats.len()) {
        return Err(PokerNowHandError::UnsupportedPlayerCount(seats.len()));
    }

    let mut players = Vec::with_capacity(seats.len());
    let mut source_by_seat = HashMap::new();
    let mut player_ids = HashMap::new();
    let mut starting_stacks = HashMap::new();
    let mut canonical_seen = HashSet::new();
    for (index, seat) in seats.iter().enumerate() {
        let source_id = index as u64 + 1;
        let source_seat = seat["seat"]
            .as_u64()
            .ok_or(PokerNowHandError::InvalidPokerLaterValue("seat"))?;
        let name = seat["name"]
            .as_str()
            .filter(|name| !name.is_empty())
            .ok_or(PokerNowHandError::InvalidPokerLaterValue("name"))?;
        let stack = json_chips_hand(&seat["startingStack"], "startingStack")?;
        let canonical = normalize_name(name)
            .ok_or_else(|| PokerNowHandError::UnknownPlayer(name.to_string()))?;
        let player_id = PlayerId::from_source(identity_namespace, &canonical);
        if !canonical_seen.insert(player_id.clone()) {
            return Err(PokerNowHandError::DuplicatePlayerIdentity(
                player_id.to_string(),
            ));
        }
        let ohh_seat = source_seat + 1;
        source_by_seat.insert(source_seat, source_id);
        player_ids.insert(source_id, player_id);
        starting_stacks.insert(source_id, stack);
        players.push(PlayerObj {
            id: source_id,
            seat: ohh_seat,
            name: name.to_string(),
            display: Some(name.to_string()),
            starting_stack: stack,
            player_bounty: None,
            is_sitting_out: None,
        });
    }

    let button = raw["final"]["button"]
        .as_u64()
        .ok_or(PokerNowHandError::MissingDealer)?;
    let dealer_seat = button + 1;
    if !players.iter().any(|player| player.seat == dealer_seat) {
        return Err(PokerNowHandError::MissingDealer);
    }

    let mut rounds = vec![empty_round(0, "Preflop")];
    let mut round_index = 0usize;
    let mut street_commitment: HashMap<u64, i64> = players.iter().map(|p| (p.id, 0)).collect();
    let mut total_commitment: HashMap<u64, i64> = players.iter().map(|p| (p.id, 0)).collect();
    let mut ready_ms = raw["startedAt"].as_i64();
    let mut decision_times_ms = Vec::new();
    let mut observed_ante = 0;

    let entries = raw["final"]["entries"]
        .as_array()
        .ok_or(PokerNowHandError::InvalidPokerLaterValue("final.entries"))?;
    let mut entries = entries.iter().collect::<Vec<_>>();
    entries.sort_unstable_by_key(|entry| entry["seq"].as_i64().unwrap_or(i64::MAX));

    for entry in entries {
        let kind = entry["kind"].as_str().unwrap_or_default();
        let at_ms = entry["at"]
            .as_i64()
            .ok_or(PokerNowHandError::InvalidPokerLaterValue("entry.at"))?;
        let source_seat = entry["seat"].as_u64();
        let actor = source_seat.and_then(|seat| source_by_seat.get(&seat).copied());

        match kind {
            "forcedBet" => {
                let actor = actor.ok_or(PokerNowHandError::InvalidPokerLaterValue("entry.seat"))?;
                let amount = json_chips_hand(&entry["amount"], "entry.amount")?;
                let action = match entry["forced"].as_str().unwrap_or_default() {
                    "smallBlind" => Action::PostSmallBlind,
                    "bigBlind" => Action::PostBigBlind,
                    "ante" => {
                        if observed_ante == 0 {
                            observed_ante = amount;
                        } else if observed_ante != amount {
                            return Err(PokerNowHandError::InconsistentAnte);
                        }
                        Action::PostAnte
                    }
                    "straddle" => return Err(PokerNowHandError::UnsupportedStraddle),
                    _ => return Err(PokerNowHandError::UnsupportedForcedBet),
                };
                add_forced_action(
                    &mut rounds[0],
                    actor,
                    action,
                    amount,
                    &starting_stacks,
                    &mut street_commitment,
                    &mut total_commitment,
                )?;
                ready_ms = Some(at_ms);
            }
            "dealBoard" if entry["boardIndex"].as_i64().unwrap_or(0) == 0 => {
                let street = match entry["street"].as_str().unwrap_or_default() {
                    "flop" => "Flop",
                    "turn" => "Turn",
                    "river" => "River",
                    _ => return Err(PokerNowHandError::InvalidPokerLaterValue("entry.street")),
                };
                let cards = parse_pokerlater_cards(&entry["cards"])?;
                rounds.push(RoundObj {
                    id: rounds.len() as u64,
                    street: street.to_string(),
                    cards: Some(cards),
                    actions: Vec::new(),
                });
                round_index = rounds.len() - 1;
                street_commitment.values_mut().for_each(|value| *value = 0);
                ready_ms = Some(at_ms);
            }
            "fold" | "check" | "call" | "bet" | "raise" => {
                let actor = actor.ok_or(PokerNowHandError::InvalidPokerLaterValue("entry.seat"))?;
                let action = match kind {
                    "fold" => Action::Fold,
                    "check" => Action::Check,
                    "call" => Action::Call,
                    "bet" => Action::Bet,
                    "raise" => Action::Raise,
                    _ => unreachable!(),
                };
                let amount = if matches!(action, Action::Fold | Action::Check) {
                    0
                } else {
                    let target = json_chips_hand(&entry["to"], "entry.to")?;
                    let own = *street_commitment.get(&actor).unwrap_or(&0);
                    if target < own
                        || (matches!(action, Action::Bet | Action::Raise) && target == own)
                    {
                        return Err(PokerNowHandError::InvalidRaiseTarget);
                    }
                    target - own
                };
                let decision_time = observed_latency(ready_ms, at_ms)?;
                let computed_all_in = add_voluntary_action(
                    &mut rounds[round_index],
                    actor,
                    action,
                    amount,
                    &starting_stacks,
                    &mut street_commitment,
                    &mut total_commitment,
                )?;
                if entry["allIn"].as_bool().unwrap_or(false) != computed_all_in {
                    return Err(PokerNowHandError::InconsistentAllIn);
                }
                decision_times_ms.push(Some(decision_time));
                ready_ms = Some(at_ms);
            }
            "uncalledReturn" => {
                let actor = actor.ok_or(PokerNowHandError::InvalidPokerLaterValue("entry.seat"))?;
                let amount = json_chips_hand(&entry["amount"], "entry.amount")?;
                subtract_observed_return(
                    actor,
                    amount,
                    &mut street_commitment,
                    &mut total_commitment,
                )?;
            }
            // PokerLater exposes exact pot indexes in final.awards; consume those
            // after the action ledger is complete instead of re-inferring them.
            "potAward" => {}
            "timeout" | "hostForceFold" | "showdownReveal" | "handComplete" => {}
            _ => {}
        }
    }

    let source_hand_id = raw["handId"]
        .as_str()
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .unwrap_or_else(|| raw["handNo"].to_string());
    let table_size = players.iter().map(|player| player.seat).max().unwrap_or(0);

    Ok(ParsedHand {
        hand: HandHistory {
            spec_version: "1.4.7".to_string(),
            site_name: "PokerLater".to_string(),
            network_name: "PokerLater".to_string(),
            internal_version: "LeChiffre-pokerlater-1".to_string(),
            tournament: false,
            tournament_info: None,
            game_number: format!("{session_id}:{source_hand_id}"),
            start_date_utc: None,
            table_name: session_id.to_string(),
            table_handle: None,
            table_skin: None,
            game_type: GameType::Holdem,
            bet_limit: Some(BetLimitObj {
                bet_type: BetType::NoLimit,
                bet_cap: 0,
            }),
            table_size,
            currency: "CHIPS".to_string(),
            dealer_seat,
            small_blind_amount: small_blind,
            big_blind_amount: big_blind,
            ante_amount: observed_ante,
            hero_player_id: None,
            players,
            rounds,
            pots: build_pokerlater_pots(raw, &source_by_seat, &total_commitment)?,
            tournament_bounties: None,
        },
        decision_times_ms,
        player_ids,
    })
}

#[derive(Debug)]
struct ParsedPlayer {
    seat: u64,
    name: String,
    stack: i64,
}

fn parse_stack_line(entry: &str) -> Result<Vec<ParsedPlayer>, PokerNowHandError> {
    let body = entry
        .strip_prefix("Player stacks: ")
        .ok_or(PokerNowHandError::MissingStacks)?;
    let mut players = Vec::new();
    for part in body.split(" | ") {
        let after_hash = part
            .strip_prefix('#')
            .ok_or(PokerNowHandError::InvalidStacks)?;
        let seat_end = after_hash
            .find(' ')
            .ok_or(PokerNowHandError::InvalidStacks)?;
        let seat = after_hash[..seat_end]
            .parse::<u64>()
            .map_err(|_| PokerNowHandError::InvalidStacks)?;
        let first_quote = part.find('"').ok_or(PokerNowHandError::InvalidStacks)?;
        let second_quote = part[first_quote + 1..]
            .find('"')
            .map(|index| first_quote + 1 + index)
            .ok_or(PokerNowHandError::InvalidStacks)?;
        let name = part[first_quote + 1..second_quote].to_string();
        let open = part[second_quote + 1..]
            .find('(')
            .map(|index| second_quote + 1 + index)
            .ok_or(PokerNowHandError::InvalidStacks)?;
        let close = part[open + 1..]
            .find(')')
            .map(|index| open + 1 + index)
            .ok_or(PokerNowHandError::InvalidStacks)?;
        let stack = part[open + 1..close]
            .parse::<i64>()
            .map_err(|_| PokerNowHandError::InvalidStacks)?;
        players.push(ParsedPlayer { seat, name, stack });
    }
    Ok(players)
}

fn parse_actor_line(entry: &str) -> Option<(&str, &str)> {
    let rest = entry.strip_prefix('"')?;
    let end = rest.find('"')?;
    let actor = &rest[..end];
    let action = rest[end + 1..].trim_start();
    Some((actor, action))
}

fn display_name(source_name: &str) -> &str {
    source_name
        .rsplit_once(" @ ")
        .map(|(name, _)| name)
        .unwrap_or(source_name)
}

fn parse_i64_prefix(value: &str) -> Option<i64> {
    value.split_whitespace().next()?.parse().ok()
}

fn parse_uncalled_return(entry: &str) -> Option<(&str, i64)> {
    let rest = entry.strip_prefix("Uncalled bet of ")?;
    let (amount, name) = rest.split_once(" returned to \"")?;
    let name = name.strip_suffix('"')?;
    Some((name, amount.parse().ok()?))
}

fn empty_round(id: u64, street: &str) -> RoundObj {
    RoundObj {
        id,
        street: street.to_string(),
        cards: None,
        actions: Vec::new(),
    }
}

fn push_action(round: &mut RoundObj, player_id: u64, action: Action, amount: i64, is_allin: bool) {
    round.actions.push(ActionObj {
        action_number: round.actions.len() as u64 + 1,
        player_id,
        action,
        amount,
        is_allin,
        cards: None,
    });
}

fn add_forced_action(
    round: &mut RoundObj,
    player: u64,
    action: Action,
    amount: i64,
    starting_stacks: &HashMap<u64, i64>,
    street_commitment: &mut HashMap<u64, i64>,
    total_commitment: &mut HashMap<u64, i64>,
) -> Result<(), PokerNowHandError> {
    if action == Action::PostAnte {
        add_total_commitment(player, amount, starting_stacks, total_commitment)?;
    } else {
        add_commitment(
            player,
            amount,
            starting_stacks,
            street_commitment,
            total_commitment,
        )?;
    }
    let is_allin = total_commitment[&player] == starting_stacks[&player];
    push_action(round, player, action, amount, is_allin);
    Ok(())
}

fn add_total_commitment(
    player: u64,
    amount: i64,
    starting_stacks: &HashMap<u64, i64>,
    total_commitment: &mut HashMap<u64, i64>,
) -> Result<(), PokerNowHandError> {
    if amount < 0 {
        return Err(PokerNowHandError::NegativeAmount);
    }
    let total = total_commitment
        .get_mut(&player)
        .ok_or(PokerNowHandError::UnknownPlayerId(player))?;
    *total = total
        .checked_add(amount)
        .ok_or(PokerNowHandError::AmountOverflow)?;
    if *total > starting_stacks[&player] {
        return Err(PokerNowHandError::ActionExceedsStack);
    }
    Ok(())
}

fn add_voluntary_action(
    round: &mut RoundObj,
    player: u64,
    action: Action,
    amount: i64,
    starting_stacks: &HashMap<u64, i64>,
    street_commitment: &mut HashMap<u64, i64>,
    total_commitment: &mut HashMap<u64, i64>,
) -> Result<bool, PokerNowHandError> {
    if amount > 0 {
        add_commitment(
            player,
            amount,
            starting_stacks,
            street_commitment,
            total_commitment,
        )?;
    }
    let is_allin = total_commitment[&player] == starting_stacks[&player];
    push_action(round, player, action, amount, is_allin);
    Ok(is_allin)
}

fn add_commitment(
    player: u64,
    amount: i64,
    starting_stacks: &HashMap<u64, i64>,
    street_commitment: &mut HashMap<u64, i64>,
    total_commitment: &mut HashMap<u64, i64>,
) -> Result<(), PokerNowHandError> {
    if amount < 0 {
        return Err(PokerNowHandError::NegativeAmount);
    }
    let total = total_commitment
        .get_mut(&player)
        .ok_or(PokerNowHandError::UnknownPlayerId(player))?;
    *total = total
        .checked_add(amount)
        .ok_or(PokerNowHandError::AmountOverflow)?;
    if *total > starting_stacks[&player] {
        return Err(PokerNowHandError::ActionExceedsStack);
    }
    *street_commitment
        .get_mut(&player)
        .ok_or(PokerNowHandError::UnknownPlayerId(player))? += amount;
    Ok(())
}

fn subtract_commitment(
    player: u64,
    amount: i64,
    street_commitment: &mut HashMap<u64, i64>,
    total_commitment: &mut HashMap<u64, i64>,
) -> Result<(), PokerNowHandError> {
    let street = street_commitment
        .get_mut(&player)
        .ok_or(PokerNowHandError::UnknownPlayerId(player))?;
    let total = total_commitment
        .get_mut(&player)
        .ok_or(PokerNowHandError::UnknownPlayerId(player))?;
    if amount < 0 || amount > *street || amount > *total {
        return Err(PokerNowHandError::InvalidUncalledReturn);
    }
    *street -= amount;
    *total -= amount;
    Ok(())
}

fn subtract_observed_return(
    player: u64,
    amount: i64,
    street_commitment: &mut HashMap<u64, i64>,
    total_commitment: &mut HashMap<u64, i64>,
) -> Result<(), PokerNowHandError> {
    let street = street_commitment
        .get_mut(&player)
        .ok_or(PokerNowHandError::UnknownPlayerId(player))?;
    let total = total_commitment
        .get_mut(&player)
        .ok_or(PokerNowHandError::UnknownPlayerId(player))?;
    if amount < 0 || amount > *total {
        return Err(PokerNowHandError::InvalidUncalledReturn);
    }
    if amount <= *street {
        *street -= amount;
    } else if *street != 0 {
        return Err(PokerNowHandError::InvalidUncalledReturn);
    }
    *total -= amount;
    Ok(())
}

fn observed_latency(ready_ms: Option<i64>, action_ms: i64) -> Result<u32, PokerNowHandError> {
    let ready_ms = ready_ms.ok_or(PokerNowHandError::MissingDecisionStart)?;
    let delta = action_ms
        .checked_sub(ready_ms)
        .ok_or(PokerNowHandError::TimingOverflow)?;
    if delta < 0 {
        return Err(PokerNowHandError::NegativeTiming);
    }
    u32::try_from(delta).map_err(|_| PokerNowHandError::TimingOverflow)
}

/// Reconstruct exact main/side pots from final contributions. For a split pot,
/// observed awards are assigned only when a unique eligible allocation exists.
/// Ambiguous payouts are rejected instead of introducing invented chip movements.
fn build_pots(
    commitments: &HashMap<u64, i64>,
    folded: &HashSet<u64>,
    awards: &[(u64, i64)],
) -> Result<Vec<PotObj>, PokerNowHandError> {
    if awards.is_empty() || awards.iter().any(|(_, value)| *value <= 0) {
        return Err(PokerNowHandError::MissingPayout);
    }
    let mut levels: Vec<i64> = commitments.values().copied().filter(|v| *v > 0).collect();
    levels.sort_unstable();
    levels.dedup();
    let mut pots: Vec<(i64, HashSet<u64>)> = Vec::new();
    let mut prev = 0_i64;
    for level in levels {
        let participants: Vec<u64> = commitments
            .iter()
            .filter(|(_, total)| **total >= level)
            .map(|(id, _)| *id)
            .collect();
        let amount = (level - prev)
            .checked_mul(participants.len() as i64)
            .ok_or(PokerNowHandError::AmountOverflow)?;
        let contributor_count = participants.len();
        let eligible: HashSet<u64> = participants
            .into_iter()
            .filter(|id| !folded.contains(id))
            .collect();
        if contributor_count > 1
            && pots
                .last()
                .is_some_and(|(_, previous_eligible)| previous_eligible == &eligible)
        {
            let previous = pots.last_mut().unwrap();
            previous.0 = previous
                .0
                .checked_add(amount)
                .ok_or(PokerNowHandError::AmountOverflow)?;
        } else {
            pots.push((amount, eligible));
        }
        prev = level;
    }
    let contributed: i64 = commitments.values().copied().sum();
    let paid: i64 = awards.iter().map(|(_, amount)| *amount).sum();
    if contributed != paid {
        return Err(PokerNowHandError::PayoutMismatch);
    }
    if awards.len() > 64 {
        return Err(PokerNowHandError::AmbiguousPotAwards);
    }

    // For each payout, enumerate eligible pots. Recurse only until the second
    // complete solution: zero = inconsistent; more than one = ambiguous.
    let mut remaining: Vec<i64> = pots.iter().map(|p| p.0).collect();
    let mut current = vec![0usize; awards.len()];
    let mut answer: Option<Vec<usize>> = None;
    let mut solutions = 0u8;
    fn assign(
        at: usize,
        awards: &[(u64, i64)],
        pots: &[(i64, HashSet<u64>)],
        remaining: &mut [i64],
        current: &mut [usize],
        answer: &mut Option<Vec<usize>>,
        solutions: &mut u8,
    ) {
        if *solutions > 1 {
            return;
        }
        if at == awards.len() {
            if remaining.iter().all(|n| *n == 0) {
                *solutions += 1;
                if answer.is_none() {
                    *answer = Some(current.to_vec());
                }
            }
            return;
        }
        let (player, amount) = awards[at];
        for i in 0..pots.len() {
            if remaining[i] < amount || !pots[i].1.contains(&player) {
                continue;
            }
            remaining[i] -= amount;
            current[at] = i;
            assign(at + 1, awards, pots, remaining, current, answer, solutions);
            remaining[i] += amount;
            if *solutions > 1 {
                return;
            }
        }
    }
    assign(
        0,
        awards,
        &pots,
        &mut remaining,
        &mut current,
        &mut answer,
        &mut solutions,
    );
    if solutions == 0 {
        return Err(PokerNowHandError::PayoutMismatch);
    }
    if solutions != 1 {
        return Err(PokerNowHandError::AmbiguousPotAwards);
    }
    let allocation = answer.ok_or(PokerNowHandError::PayoutMismatch)?;
    let mut output = Vec::new();
    for (number, (amount, _)) in pots.iter().enumerate() {
        let mut by_player = HashMap::<u64, i64>::new();
        for (index, &(player, share)) in awards.iter().enumerate() {
            if allocation[index] == number {
                *by_player.entry(player).or_insert(0) += share;
            }
        }
        let mut player_wins: Vec<_> = by_player
            .into_iter()
            .map(|(player_id, win_amount)| PlayerWinsObj {
                player_id,
                win_amount,
                cashout_amount: None,
                cashout_fee: None,
                bonus_amount: None,
                contributed_rake: None,
            })
            .collect();
        player_wins.sort_unstable_by_key(|win| win.player_id);
        output.push(PotObj {
            number: number as u64,
            amount: *amount,
            rake: Some(0),
            jackpot: None,
            player_wins,
        });
    }
    Ok(output)
}

fn build_pokerlater_pots(
    raw: &Value,
    source_by_seat: &HashMap<u64, u64>,
    commitments: &HashMap<u64, i64>,
) -> Result<Vec<PotObj>, PokerNowHandError> {
    let raw_pots = raw["final"]["pots"]
        .as_array()
        .ok_or(PokerNowHandError::InvalidPokerLaterValue("final.pots"))?;
    let raw_awards = raw["final"]["awards"]
        .as_array()
        .ok_or(PokerNowHandError::InvalidPokerLaterValue("final.awards"))?;
    if raw_pots.is_empty() || raw_awards.is_empty() {
        return Err(PokerNowHandError::MissingPayout);
    }

    let mut wins_by_pot = (0..raw_pots.len())
        .map(|_| HashMap::<u64, i64>::new())
        .collect::<Vec<_>>();
    for award in raw_awards {
        let pot_index = award["potIndex"]
            .as_u64()
            .and_then(|value| usize::try_from(value).ok())
            .filter(|index| *index < raw_pots.len())
            .ok_or(PokerNowHandError::InvalidPokerLaterValue(
                "final.awards.potIndex",
            ))?;
        if award["boardIndex"].as_u64().unwrap_or(0) != 0 {
            return Err(PokerNowHandError::InvalidPokerLaterValue(
                "final.awards.boardIndex",
            ));
        }
        let source_seat =
            award["seat"]
                .as_u64()
                .ok_or(PokerNowHandError::InvalidPokerLaterValue(
                    "final.awards.seat",
                ))?;
        let player_id =
            *source_by_seat
                .get(&source_seat)
                .ok_or(PokerNowHandError::InvalidPokerLaterValue(
                    "final.awards.seat",
                ))?;
        let amount = json_chips_hand(&award["amount"], "final.awards.amount")?;
        if amount <= 0 {
            return Err(PokerNowHandError::MissingPayout);
        }
        let current = wins_by_pot[pot_index].entry(player_id).or_insert(0);
        *current = current
            .checked_add(amount)
            .ok_or(PokerNowHandError::AmountOverflow)?;
    }

    let contributed = commitments.values().try_fold(0_i64, |sum, value| {
        sum.checked_add(*value)
            .ok_or(PokerNowHandError::AmountOverflow)
    })?;
    let mut reported = 0_i64;
    let mut output = Vec::with_capacity(raw_pots.len());
    for (number, raw_pot) in raw_pots.iter().enumerate() {
        let amount = json_chips_hand(&raw_pot["amount"], "final.pots.amount")?;
        if amount <= 0 {
            return Err(PokerNowHandError::MissingPayout);
        }
        reported = reported
            .checked_add(amount)
            .ok_or(PokerNowHandError::AmountOverflow)?;

        let mut player_wins = wins_by_pot[number]
            .iter()
            .map(|(&player_id, &win_amount)| PlayerWinsObj {
                player_id,
                win_amount,
                cashout_amount: None,
                cashout_fee: None,
                bonus_amount: None,
                contributed_rake: None,
            })
            .collect::<Vec<_>>();
        player_wins.sort_unstable_by_key(|win| win.player_id);
        let paid = player_wins.iter().try_fold(0_i64, |sum, win| {
            sum.checked_add(win.win_amount)
                .ok_or(PokerNowHandError::AmountOverflow)
        })?;
        if paid != amount {
            return Err(PokerNowHandError::PayoutMismatch);
        }
        output.push(PotObj {
            number: number as u64,
            amount,
            rake: Some(0),
            jackpot: None,
            player_wins,
        });
    }
    if reported != contributed {
        return Err(PokerNowHandError::PayoutMismatch);
    }
    Ok(output)
}

fn parse_board(entry: &str) -> Result<Option<(&'static str, Vec<Card>)>, PokerNowHandError> {
    let street = if entry.starts_with("Flop:") {
        "Flop"
    } else if entry.starts_with("Turn:") {
        "Turn"
    } else if entry.starts_with("River:") {
        "River"
    } else {
        return Ok(None);
    };
    let open = entry.rfind('[').ok_or(PokerNowHandError::InvalidBoard)?;
    let close = entry[open + 1..]
        .find(']')
        .map(|index| open + 1 + index)
        .ok_or(PokerNowHandError::InvalidBoard)?;
    let cards = entry[open + 1..close]
        .split(',')
        .map(|card| parse_card(card.trim()))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Some((street, cards)))
}

fn parse_card(value: &str) -> Result<Card, PokerNowHandError> {
    let value = normalize_mojibake(value);
    let (rank, suit) = if let Some(rank) = value.strip_suffix('♣') {
        (rank, 'c')
    } else if let Some(rank) = value.strip_suffix('♦') {
        (rank, 'd')
    } else if let Some(rank) = value.strip_suffix('♥') {
        (rank, 'h')
    } else if let Some(rank) = value.strip_suffix('♠') {
        (rank, 's')
    } else {
        return Err(PokerNowHandError::InvalidCard(value));
    };
    let rank = match rank {
        "10" => "T",
        "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" | "J" | "Q" | "K" | "A" => rank,
        _ => return Err(PokerNowHandError::InvalidCard(value)),
    };
    let token = format!("{rank}{suit}");
    Card::try_from(token.as_str()).map_err(|_| PokerNowHandError::InvalidCard(value))
}

fn normalize_mojibake(value: &str) -> String {
    value
        .replace("â™£", "♣")
        .replace("â™¦", "♦")
        .replace("â™¥", "♥")
        .replace("â™ ", "♠")
}

fn parse_pokerlater_cards(value: &Value) -> Result<Vec<Card>, PokerNowHandError> {
    let cards = value
        .as_array()
        .ok_or(PokerNowHandError::InvalidPokerLaterValue("entry.cards"))?;
    cards
        .iter()
        .map(|card| {
            let rank = card["rank"]
                .as_u64()
                .ok_or(PokerNowHandError::InvalidPokerLaterValue("card.rank"))?;
            let suit = card["suit"]
                .as_u64()
                .ok_or(PokerNowHandError::InvalidPokerLaterValue("card.suit"))?;
            let rank = match rank {
                2..=9 => rank.to_string(),
                10 => "T".to_string(),
                11 => "J".to_string(),
                12 => "Q".to_string(),
                13 => "K".to_string(),
                14 => "A".to_string(),
                _ => return Err(PokerNowHandError::InvalidPokerLaterValue("card.rank")),
            };
            let suit = match suit {
                0 => 'c',
                1 => 'd',
                2 => 'h',
                3 => 's',
                _ => return Err(PokerNowHandError::InvalidPokerLaterValue("card.suit")),
            };
            let token = format!("{rank}{suit}");
            Card::try_from(token.as_str())
                .map_err(|_| PokerNowHandError::InvalidPokerLaterValue("entry.cards"))
        })
        .collect()
}

fn json_chips(value: &Value) -> Result<i64, PokerNowAdapterError> {
    value
        .as_i64()
        .ok_or(PokerNowAdapterError::InvalidPokerLaterChipValue)
}

fn json_chips_hand(value: &Value, field: &'static str) -> Result<i64, PokerNowHandError> {
    value
        .as_i64()
        .ok_or(PokerNowHandError::InvalidPokerLaterValue(field))
}

#[derive(Debug)]
pub enum PokerNowAdapterError {
    InvalidCsv(String),
    InvalidPokerLaterJson(String),
    UnsupportedPokerLaterFormat,
    MissingPokerLaterHands,
    InvalidPokerLaterChipValue,
}

impl fmt::Display for PokerNowAdapterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidCsv(error) => write!(f, "invalid PokerNow CSV: {error}"),
            Self::InvalidPokerLaterJson(error) => write!(f, "invalid PokerLater JSON: {error}"),
            Self::UnsupportedPokerLaterFormat => write!(f, "unsupported PokerLater log format"),
            Self::MissingPokerLaterHands => write!(f, "PokerLater log has no hands array"),
            Self::InvalidPokerLaterChipValue => write!(f, "invalid PokerLater blind amount"),
        }
    }
}

impl std::error::Error for PokerNowAdapterError {}

#[derive(Debug)]
enum PokerNowHandError {
    MissingStart,
    MissingHandNumber,
    MissingDealer,
    MissingStacks,
    InvalidStacks,
    UnsupportedPlayerCount(usize),
    UnknownPlayer(String),
    UnknownActor(String),
    UnknownPlayerId(u64),
    DuplicatePlayerIdentity(String),
    MissingSmallBlind,
    MissingBigBlind,
    InconsistentAnte,
    UnsupportedStraddle,
    UnsupportedDeadBlind,
    UnsupportedForcedBet,
    InvalidRaiseTarget,
    InvalidCallTarget,
    NegativeAmount,
    AmountOverflow,
    ActionExceedsStack,
    InvalidUncalledReturn,
    MissingDecisionStart,
    NegativeTiming,
    TimingOverflow,
    InconsistentAllIn,
    InvalidBoard,
    InvalidCard(String),
    MissingPayout,
    PayoutMismatch,
    AmbiguousPotAwards,
    InvalidPokerLaterValue(&'static str),
    HandHistory(HandHistoryAdapterError),
}

impl fmt::Display for PokerNowHandError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingStart => write!(f, "missing PokerNow hand start"),
            Self::MissingHandNumber => write!(f, "missing PokerNow hand number"),
            Self::MissingDealer => write!(f, "missing or invalid dealer"),
            Self::MissingStacks => write!(f, "missing player stacks"),
            Self::InvalidStacks => write!(f, "invalid player stacks line"),
            Self::UnsupportedPlayerCount(players) => {
                write!(f, "unsupported player count {players}")
            }
            Self::UnknownPlayer(name) => write!(f, "no canonical identity for player {name}"),
            Self::UnknownActor(name) => write!(f, "action references unknown player {name}"),
            Self::UnknownPlayerId(id) => write!(f, "unknown internal player id {id}"),
            Self::DuplicatePlayerIdentity(id) => {
                write!(f, "duplicate canonical player identity {id}")
            }
            Self::MissingSmallBlind => write!(f, "missing small blind posting"),
            Self::MissingBigBlind => write!(f, "missing big blind posting"),
            Self::InconsistentAnte => write!(f, "inconsistent ante amount"),
            Self::UnsupportedStraddle => {
                write!(f, "straddled hand is outside the current adapter scope")
            }
            Self::UnsupportedDeadBlind => {
                write!(f, "dead-blind hand is outside the current adapter scope")
            }
            Self::UnsupportedForcedBet => write!(f, "unsupported forced bet"),
            Self::InvalidRaiseTarget => write!(f, "invalid bet/raise target"),
            Self::InvalidCallTarget => write!(f, "invalid call target"),
            Self::NegativeAmount => write!(f, "negative chip amount"),
            Self::AmountOverflow => write!(f, "chip amount overflow"),
            Self::ActionExceedsStack => write!(f, "observed action exceeds starting stack"),
            Self::InvalidUncalledReturn => write!(f, "invalid uncalled-bet return"),
            Self::MissingDecisionStart => write!(f, "decision has no observed start timestamp"),
            Self::NegativeTiming => write!(f, "action timestamp precedes decision start"),
            Self::TimingOverflow => write!(f, "decision timing does not fit u32 milliseconds"),
            Self::InconsistentAllIn => {
                write!(f, "all-in marker disagrees with observed stack ledger")
            }
            Self::InvalidBoard => write!(f, "invalid board line"),
            Self::InvalidCard(card) => write!(f, "invalid card {card}"),
            Self::MissingPayout => write!(f, "hand has no valid observed pot award"),
            Self::PayoutMismatch => write!(
                f,
                "observed awards cannot reconcile against contributions and eligible pots"
            ),
            Self::AmbiguousPotAwards => write!(
                f,
                "multiple valid pot allocations; cannot identify actual awards"
            ),
            Self::InvalidPokerLaterValue(field) => write!(f, "invalid PokerLater field {field}"),
            Self::HandHistory(error) => write!(f, "rs-poker replay rejected hand: {error}"),
        }
    }
}

impl From<HandHistoryAdapterError> for PokerNowHandError {
    fn from(error: HandHistoryAdapterError) -> Self {
        Self::HandHistory(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_real_heads_up_hand_with_observed_timing() {
        let csv = r#"entry,at,order
"-- starting hand #128 (id: qgojco5mayy0)  No Limit Texas Hold'em (dealer: ""arash @ RYnRljE1SO"") --",2026-10-08T02:29:50.786Z,179142659078600
"Player stacks: #3 ""goergoe @ _sqcLETZ0C"" (2640) | #8 ""arash @ RYnRljE1SO"" (3390)",2026-10-08T02:29:50.786Z,179142659078601
"""arash @ RYnRljE1SO"" posts a small blind of 10",2026-10-08T02:29:50.786Z,179142659078604
"""goergoe @ _sqcLETZ0C"" posts a big blind of 20",2026-10-08T02:29:50.786Z,179142659078605
"""arash @ RYnRljE1SO"" calls 20",2026-10-08T02:29:52.078Z,179142659207800
"""goergoe @ _sqcLETZ0C"" checks",2026-10-08T02:29:52.968Z,179142659296800
"Flop:  [Q♦, 3♥, A♣]",2026-10-08T02:29:53.790Z,179142659379000
"""goergoe @ _sqcLETZ0C"" bets 20",2026-10-08T02:29:55.034Z,179142659503400
"""arash @ RYnRljE1SO"" calls 20",2026-10-08T02:29:57.125Z,179142659712500
"Turn: Q♦, 3♥, A♣ [8♦]",2026-10-08T02:29:57.949Z,179142659794900
"""goergoe @ _sqcLETZ0C"" bets 50",2026-10-08T02:29:59.568Z,179142659956800
"""arash @ RYnRljE1SO"" folds",2026-10-08T02:30:01.306Z,179142660130600
"Uncalled bet of 50 returned to ""goergoe @ _sqcLETZ0C""",2026-10-08T02:30:02.180Z,179142660218000
"""goergoe @ _sqcLETZ0C"" collected 80 from pot",2026-10-08T02:30:02.180Z,179142660218001
"-- ending hand #128 --",2026-10-08T02:30:02.180Z,179142660218002
"#;
        let imported = PokerNowAdapter::adapt_csv("20261007", "test", csv, |name| match name {
            "arash" => Some("Arash".to_string()),
            "goergoe" => Some("George".to_string()),
            _ => None,
        })
        .unwrap();
        assert_eq!(imported.rejected.len(), 0);
        assert_eq!(imported.hands.len(), 1);
        let events = imported.hands[0].sequence().events();
        assert_eq!(imported.exact_legal_actions.len(), events.len());
        for event in events {
            assert!(
                imported
                    .exact_legal_actions
                    .contains_key(&(event.hand_id.clone(), event.sequence_index))
            );
        }
        assert_eq!(events[0].action.decision_time_ms, Some(1292));
        assert_eq!(events[1].action.decision_time_ms, Some(890));
        assert!(
            events
                .iter()
                .all(|event| event.action.decision_time_ms.is_some())
        );

        // One genuinely source-observed timed hand exercises the complete A/G
        // inference contract without fabricating additional training hands.
        use crate::behavioural::{
            BehaviourFeaturizer, BehaviourLatentConfig, NeuralBehaviourAnalyzer,
            NeuralBehaviourGenerator,
        };
        let subject_id = PlayerId::from_source("test", "Arash");
        let subject_hand = imported.hands[0].subject_hand(&subject_id).unwrap();
        let features = BehaviourFeaturizer::new()
            .featurize_hand(&subject_hand)
            .unwrap()
            .unwrap();
        let latent = BehaviourLatentConfig::new(3).unwrap();
        let analyzer = NeuralBehaviourAnalyzer::initialize(latent, 8, 91).unwrap();
        let posterior = analyzer.infer(&[features]).unwrap();
        let z = posterior.mean_profile().unwrap();
        let generator = NeuralBehaviourGenerator::initialize(latent, 8, 92).unwrap();
        let observed = events
            .iter()
            .find(|event| event.actor_position() == subject_hand.subject_position())
            .unwrap();
        let legal = imported
            .exact_legal_actions
            .get(&(observed.hand_id.clone(), observed.sequence_index))
            .unwrap();
        let policy = generator.predict(&z, &observed.context, legal).unwrap();
        let probability_sum: f64 = policy.action_probabilities().iter().sum();
        assert!((probability_sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn csv_parser_preserves_quoted_multiline_records() {
        let csv = r#"entry,at,order
"Game Config Changes:
* Allow Straddle: off
* Decision Limit Time: 20",2026-01-01T00:00:00Z,100
"-- starting hand #1 (id: a) (dealer: ""Alice"") --",2026-01-01T00:00:00Z,200
"#;
        let rows = parse_csv(csv).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(
            rows[0].entry,
            "Game Config Changes:\n* Allow Straddle: off\n* Decision Limit Time: 20"
        );
    }

    #[test]
    fn main_and_side_pots_with_different_winners() {
        let contributions = HashMap::from([(1, 100), (2, 200), (3, 200)]);
        let pots = build_pots(&contributions, &HashSet::new(), &[(1, 300), (2, 200)]).unwrap();
        assert_eq!(pots.len(), 2);
        assert_eq!(pots[0].amount, 300);
        assert_eq!(pots[0].player_wins[0].player_id, 1);
        assert_eq!(pots[1].amount, 200);
        assert_eq!(pots[1].player_wins[0].player_id, 2);
    }

    #[test]
    fn rejects_unreconciled_pot_awards() {
        let contributions = HashMap::from([(1, 100), (2, 200), (3, 200)]);
        assert!(build_pots(&contributions, &HashSet::new(), &[(1, 500)]).is_err());
    }

    #[test]
    fn folded_contribution_slices_with_same_eligibility_are_merged() {
        let contributions = HashMap::from([(1, 20), (2, 10), (3, 20)]);
        let folded = HashSet::from([2]);
        let pots = build_pots(&contributions, &folded, &[(1, 50)]).unwrap();
        assert_eq!(pots.len(), 1);
        assert_eq!(pots[0].amount, 50);
    }

    #[test]
    fn late_uncalled_return_can_follow_street_reset() {
        let mut street = HashMap::from([(1, 0)]);
        let mut total = HashMap::from([(1, 100)]);
        subtract_observed_return(1, 20, &mut street, &mut total).unwrap();
        assert_eq!(street[&1], 0);
        assert_eq!(total[&1], 80);
    }

    #[test]
    fn pokerlater_uses_explicit_pot_indexes() {
        let raw = serde_json::json!({
            "final": {
                "pots": [
                    {"amount": 300, "eligible": [0, 1, 2]},
                    {"amount": 200, "eligible": [1, 2]}
                ],
                "awards": [
                    {"potIndex": 0, "boardIndex": 0, "seat": 0, "amount": 300},
                    {"potIndex": 1, "boardIndex": 0, "seat": 1, "amount": 200}
                ]
            }
        });
        let seats = HashMap::from([(0, 1), (1, 2), (2, 3)]);
        let commitments = HashMap::from([(1, 100), (2, 200), (3, 200)]);
        let pots = build_pokerlater_pots(&raw, &seats, &commitments).unwrap();
        assert_eq!(pots.len(), 2);
        assert_eq!(pots[0].amount, 300);
        assert_eq!(pots[0].player_wins[0].player_id, 1);
        assert_eq!(pots[1].amount, 200);
        assert_eq!(pots[1].player_wins[0].player_id, 2);
    }
}
