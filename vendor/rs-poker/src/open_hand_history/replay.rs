//! Replay normalized NLHE cash OHH through the exact arena ledger.
//! Rake policy is an explicit input, never inferred from observed winnings.
use super::{Action, ActionObj, BetType, GameType, HandHistory};
use crate::{
    Chips,
    arena::{
        GameState, GameStateBuilder, RakeConfig,
        action::AgentAction,
        game_state::{Round, RoundData},
        pot::{plan_pots, planned_awards},
    },
    core::{Card, Hand, PlayerBitSet},
};
use std::collections::{HashMap, HashSet};
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("hand {hand_id}, round {round:?}, action {action_number:?}: {message}")]
pub struct ReplayError {
    pub hand_id: String,
    pub round: Option<String>,
    pub action_number: Option<u64>,
    pub message: String,
}
#[derive(Debug, Clone)]
pub struct ReplayStep {
    pub action_number: u64,
    pub player_id: u64,
    pub state_before: GameState,
    pub action: AgentAction,
    pub state_after: GameState,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReplayVerification {
    /// False only when a contested showdown omits an eligible player's cards.
    /// Eligibility, reported payouts, rake and conservation are still checked.
    pub winners_by_cards: bool,
}
#[derive(Debug, Clone)]
pub struct ReplayedHand {
    pub initial_state: GameState,
    pub steps: Vec<ReplayStep>,
    pub final_state: GameState,
    /// Index order corresponds to these original OHH player IDs/seats.
    pub player_ids: Vec<u64>,
    pub verification: ReplayVerification,
}
impl ReplayedHand {
    pub fn transitions(&self) -> impl Iterator<Item = &ReplayStep> {
        self.steps.iter()
    }
}
fn fail(
    h: &HandHistory,
    round: Option<&str>,
    action: Option<u64>,
    message: impl Into<String>,
) -> ReplayError {
    ReplayError {
        hand_id: h.game_number.clone(),
        round: round.map(str::to_string),
        action_number: action,
        message: message.into(),
    }
}
/// Validate every observable ledger transition. Missing hole cards are left
/// unknown; no random or placeholder cards are generated.
pub fn replay_hand(h: &HandHistory, rake: RakeConfig) -> Result<ReplayedHand, ReplayError> {
    let error = |message: &str| fail(h, None, None, message);
    if h.tournament
        || h.game_type != GameType::Holdem
        || h.bet_limit
            .as_ref()
            .is_none_or(|b| b.bet_type != BetType::NoLimit || b.bet_cap != 0)
    {
        return Err(error("only uncapped NLHE cash hands are supported"));
    }
    if h.tournament_info.is_some() || h.tournament_bounties.is_some() {
        return Err(error("cash hand contains tournament metadata"));
    }
    let mut ids = HashSet::new();
    let mut seats = HashSet::new();
    for p in &h.players {
        if !ids.insert(p.id)
            || !seats.insert(p.seat)
            || p.seat == 0
            || p.seat > h.table_size
            || p.starting_stack < 0
        {
            return Err(error(
                "duplicate/invalid player ID, seat, or starting stack",
            ));
        }
    }
    let mut players = h
        .players
        .iter()
        .filter(|p| p.is_sitting_out != Some(true))
        .collect::<Vec<_>>();
    players.sort_unstable_by_key(|p| p.seat);
    if !(2..=16).contains(&players.len()) {
        return Err(error("NLHE requires 2..=16 dealt players"));
    }
    if players.iter().any(|p| p.starting_stack == 0) {
        return Err(error("dealt player has no starting chips"));
    }
    let dealer = players
        .iter()
        .position(|p| p.seat == h.dealer_seat)
        .ok_or_else(|| error("button is not a dealt player's seat"))?;
    let by_id: HashMap<u64, usize> = players.iter().enumerate().map(|(i, p)| (p.id, i)).collect();
    let n = players.len();
    let sb = if n == 2 { dealer } else { (dealer + 1) % n };
    let bb = (sb + 1) % n;
    let mut gs = GameStateBuilder::new()
        .stacks(players.iter().map(|p| p.starting_stack).collect::<Vec<_>>())
        .blinds(h.big_blind_amount, h.small_blind_amount)
        .ante(h.ante_amount)
        .dealer_idx(dealer)
        .rake(rake)
        .max_raises_per_round(None)
        .build()
        .map_err(|e| error(&format!("invalid initial state: {e}")))?;
    let initial_state = gs.clone();
    gs.round = Round::Ante;
    gs.round_data = RoundData::new(n, h.big_blind_amount, gs.player_active, (dealer + 1) % n);
    let mut holes: Vec<Option<Vec<Card>>> = vec![None; n];
    let mut used = HashSet::new();
    let mut steps = Vec::new();
    let mut antes = HashSet::new();
    let mut posted_sb = false;
    let mut posted_bb = false;
    let mut begun = false;
    let mut street = 0;
    let mut showdown = false;
    for round in &h.rounds {
        let next = match round.street.as_str() {
            "Preflop" => 0,
            "Flop" => 1,
            "Turn" => 2,
            "River" => 3,
            "Showdown" => 4,
            _ => return Err(fail(h, Some(&round.street), None, "unsupported OHH street")),
        };
        if next == 0 {
            if begun {
                return Err(fail(
                    h,
                    Some(&round.street),
                    None,
                    "duplicate preflop round",
                ));
            }
            begun = true;
            if round.cards.as_ref().is_some_and(|c| !c.is_empty()) {
                return Err(error("preflop contains board cards"));
            }
        } else if next == 4 {
            if showdown {
                return Err(error("duplicate showdown"));
            }
            close_street(h, &mut gs)?;
            showdown = true;
            if round.cards.as_ref().is_some_and(|c| !c.is_empty()) {
                return Err(error("showdown contains new board cards"));
            }
        } else {
            if !begun || showdown || next != street + 1 {
                return Err(fail(
                    h,
                    Some(&round.street),
                    None,
                    "out-of-order or skipped street",
                ));
            }
            if !posted_sb || !posted_bb {
                return Err(error("street advanced before both blinds"));
            }
            close_street(h, &mut gs)?;
            if (gs.player_active | gs.player_all_in).count() < 2 {
                return Err(error("board dealt after hand was won by folds"));
            }
            let new = round
                .cards
                .as_ref()
                .ok_or_else(|| error("missing street cards"))?;
            if new.len() != if next == 1 { 3 } else { 1 } {
                return Err(error("NLHE street has incorrect card count"));
            }
            for &card in new {
                if !used.insert(card) {
                    return Err(error("duplicate board/hole card"));
                }
                gs.board.push(card);
                for hand in &mut gs.hands {
                    hand.insert(card);
                }
            }
            gs.round = match next {
                1 => Round::Flop,
                2 => Round::Turn,
                _ => Round::River,
            };
            gs.round_data = RoundData::new(n, gs.big_blind, gs.player_active, dealer);
            gs.round_data.advance_action();
            street = next;
        }
        let action_number_base = round.actions.first().map_or(1, |a| a.action_number);
        if !matches!(action_number_base, 0 | 1) {
            return Err(fail(
                h,
                Some(&round.street),
                Some(action_number_base),
                "action numbers must start at one (OHH) or zero (legacy rs-poker)",
            ));
        }
        for (position, a) in round.actions.iter().enumerate() {
            let err = |message: &str| fail(h, Some(&round.street), Some(a.action_number), message);
            if a.action_number != action_number_base + position as u64 {
                return Err(err("action numbers must be sequential within each round"));
            }
            let idx = *by_id
                .get(&a.player_id)
                .ok_or_else(|| err("action refers to unknown or sitting-out player"))?;
            if a.amount < 0 {
                return Err(err("negative action amount"));
            }
            match a.action {
                Action::DealtCards | Action::ShowsCards => {
                    if a.amount != 0 || a.is_allin {
                        return Err(err("card action carries money/all-in flag"));
                    }
                    if let Some(cards) = &a.cards {
                        if cards.len() != 2 || cards[0] == cards[1] {
                            return Err(err("NLHE hole cards must be two distinct cards"));
                        }
                        if let Some(previous) = &holes[idx] {
                            if previous.iter().copied().collect::<HashSet<_>>()
                                != cards.iter().copied().collect::<HashSet<_>>()
                            {
                                return Err(err("shown cards disagree with dealt cards"));
                            }
                        } else {
                            for &c in cards {
                                if !used.insert(c) {
                                    return Err(err("duplicate hole/board card"));
                                }
                            }
                            holes[idx] = Some(cards.clone());
                            gs.hands[idx] = Hand::default();
                            for &c in cards.iter().chain(gs.board.iter()) {
                                gs.hands[idx].insert(c);
                            }
                        }
                    }
                }
                Action::MucksCards => {
                    if a.amount != 0 || a.cards.as_ref().is_some_and(|c| !c.is_empty()) {
                        return Err(err("malformed muck"));
                    }
                }
                Action::PostAnte => {
                    if street != 0 || posted_sb || posted_bb || !antes.insert(idx) {
                        return Err(err("duplicate or out-of-order ante"));
                    }
                    if h.ante_amount == 0 || a.amount != h.ante_amount.min(gs.stacks[idx]) {
                        return Err(err("ante amount disagrees with configuration/stack"));
                    }
                    gs.round_data.to_act_idx = idx;
                    gs.do_bet(a.amount, true).map_err(|e| err(&e.to_string()))?;
                    check_allin(a, &gs, idx).map_err(|m| err(m))?;
                }
                Action::PostSmallBlind | Action::PostBigBlind => {
                    if street != 0 || showdown {
                        return Err(err("blind outside preflop"));
                    }
                    if h.ante_amount > 0 && antes.len() != n {
                        return Err(err("missing ante postings"));
                    }
                    if gs.round == Round::Ante {
                        gs.round = Round::Preflop;
                        gs.round_data = RoundData::new(n, gs.big_blind, gs.player_active, sb);
                    }
                    let (expected, seat) = if a.action == Action::PostSmallBlind {
                        (gs.small_blind, sb)
                    } else {
                        (gs.big_blind, bb)
                    };
                    if idx != seat || a.amount != expected.min(gs.stacks[idx]) {
                        return Err(err("blind seat/amount disagrees with button/stacks"));
                    }
                    if a.action == Action::PostSmallBlind {
                        if posted_sb || posted_bb {
                            return Err(err("duplicate/out-of-order small blind"));
                        }
                        posted_sb = true;
                        gs.sb_posted = true;
                    } else {
                        if posted_bb || !posted_sb {
                            return Err(err("duplicate/out-of-order big blind"));
                        }
                        posted_bb = true;
                        gs.bb_posted = true;
                    }
                    gs.round_data.to_act_idx = idx;
                    gs.do_bet(a.amount, true).map_err(|e| err(&e.to_string()))?;
                    // A short all-in big blind does not reduce the full blind owed.
                    if posted_bb {
                        gs.round_data.bet = gs.round_data.bet.max(gs.big_blind);
                    }
                    check_allin(a, &gs, idx).map_err(|m| err(m))?;
                }
                Action::Fold | Action::Check | Action::Call | Action::Bet | Action::Raise => {
                    if !posted_sb || !posted_bb || showdown {
                        return Err(err("voluntary action before blinds or after showdown"));
                    }
                    if !gs.player_active.get(idx)
                        || !gs.round_data.needs_action.get(idx)
                        || gs.to_act_idx() != idx
                    {
                        return Err(err(
                            "wrong actor/order, folded player, or action after all-in",
                        ));
                    }
                    if a.cards.is_some() {
                        return Err(err("betting action contains card metadata"));
                    }
                    let own = gs.current_round_player_bet(idx);
                    let current = gs.current_round_bet();
                    let target = own
                        .checked_add(a.amount)
                        .ok_or_else(|| err("bet overflow"))?;
                    if a.amount > gs.stacks[idx] {
                        return Err(err("action exceeds stack"));
                    }
                    let observed = match a.action {
                        Action::Fold => {
                            if a.amount != 0 || a.is_allin {
                                return Err(err("fold has money/all-in flag"));
                            }
                            AgentAction::Fold
                        }
                        Action::Check => {
                            if a.amount != 0 || own < current {
                                return Err(err("illegal check"));
                            }
                            AgentAction::Call
                        }
                        Action::Call => {
                            if a.amount != (current - own).min(gs.stacks[idx]) || a.amount == 0 {
                                return Err(err("call amount is not the exact amount owed"));
                            }
                            AgentAction::Call
                        }
                        Action::Bet => {
                            if street == 0 || current != 0 || target == 0 {
                                return Err(err("Bet requires an unopened postflop street"));
                            }
                            AgentAction::Bet(target)
                        }
                        Action::Raise => {
                            if target <= current {
                                return Err(err("Raise does not exceed current bet"));
                            }
                            AgentAction::Bet(target)
                        }
                        _ => unreachable!(),
                    };
                    let before = gs.clone();
                    if observed == AgentAction::Fold {
                        gs.fold();
                    } else {
                        gs.do_bet(target, false)
                            .map_err(|e| err(&format!("illegal exact bet: {e}")))?;
                        check_allin(a, &gs, idx).map_err(|m| err(m))?;
                    }
                    steps.push(ReplayStep {
                        action_number: a.action_number,
                        player_id: a.player_id,
                        state_before: before,
                        action: observed,
                        state_after: gs.clone(),
                    });
                }
                _ => {
                    return Err(err(
                        "unsupported OHH state action (straddle/dead blind/chip adjustment/cashout)",
                    ));
                }
            }
        }
    }
    if !begun || !posted_sb || !posted_bb {
        return Err(error("missing preflop or forced blind actions"));
    }
    close_street(h, &mut gs)?;
    let contenders = gs.player_active | gs.player_all_in;
    if contenders.empty() {
        return Err(error("hand has no contender"));
    }
    if contenders.count() > 1 && gs.board.len() != 5 {
        return Err(error("contested hand ends without a complete board"));
    }
    let pots = plan_pots(&gs, gs.board.len() >= 3);
    if pots.iter().any(|p| p.refund_to.is_some()) {
        return Err(error("unreturned excess after street closure"));
    }
    if h.pots.len() != pots.len() {
        return Err(error(
            "main/side-pot count differs from contribution ledger",
        ));
    }
    let known = contenders.count() == 1 || contenders.ones().all(|i| holes[i].is_some());
    let ranked = if known {
        Some(planned_awards(&gs))
    } else {
        None
    };
    let mut numbers = HashSet::new();
    for p in &h.pots {
        if !numbers.insert(p.number) {
            return Err(error("duplicate OHH pot number"));
        }
        let number = usize::try_from(p.number).map_err(|_| error("invalid pot number"))?;
        let expected = pots
            .get(number)
            .ok_or_else(|| error("noncontiguous/unknown pot number"))?;
        if p.amount != expected.gross || p.rake.is_some_and(|v| v != expected.rake) {
            return Err(error(
                "gross pot/rake differs from configured exact settlement",
            ));
        }
        if p.jackpot.is_some_and(|v| v != 0) {
            return Err(error("nonzero jackpot is unsupported"));
        }
        let mut paid = 0_i64;
        let mut winners = HashMap::new();
        for win in &p.player_wins {
            if win.cashout_amount.is_some()
                || win.cashout_fee.is_some()
                || win.bonus_amount.is_some()
            {
                return Err(error("cashout/bonus payout unsupported"));
            }
            let idx = *by_id
                .get(&win.player_id)
                .ok_or_else(|| error("unknown winning player"))?;
            if !expected.eligible.get(idx)
                || win.win_amount < 0
                || winners.insert(idx, win.win_amount).is_some()
            {
                return Err(error("ineligible/duplicate winner or negative winnings"));
            }
            paid = paid
                .checked_add(win.win_amount)
                .ok_or_else(|| error("winnings overflow"))?;
        }
        if paid != expected.net {
            return Err(error("pot winnings plus exact rake do not equal gross pot"));
        }
        if let Some(awards) = &ranked {
            for idx in 0..n {
                let calculated = awards
                    .iter()
                    .filter(|a| a.pot_index == number && a.idx == idx && !a.refund)
                    .map(|a| a.amount)
                    .sum::<Chips>();
                let observed = winners.get(&idx).copied().unwrap_or(0);
                if calculated != observed {
                    return Err(error(
                        "winner/share disagrees with ranked hand or odd-chip rule",
                    ));
                }
            }
        }
        gs.rake_collected += expected.rake;
        for (idx, amount) in winners {
            gs.award(idx, amount);
        }
    }
    if gs.starting_stacks.iter().sum::<Chips>()
        != gs.stacks.iter().sum::<Chips>() + gs.rake_collected
    {
        return Err(error("final stack conservation failed"));
    }
    gs.complete();
    Ok(ReplayedHand {
        initial_state,
        steps,
        final_state: gs,
        player_ids: players.iter().map(|p| p.id).collect(),
        verification: ReplayVerification {
            winners_by_cards: known,
        },
    })
}
fn check_allin(a: &ActionObj, gs: &GameState, idx: usize) -> Result<(), &'static str> {
    if a.is_allin != (gs.stacks[idx] == 0) {
        Err("all-in flag disagrees with exact remaining stack")
    } else {
        Ok(())
    }
}
fn close_street(h: &HandHistory, gs: &mut GameState) -> Result<(), ReplayError> {
    let contenders = gs.player_active | gs.player_all_in;
    if contenders.count() > 1 && !gs.round_data.needs_action.empty() {
        let can_runout = gs.num_active_players() <= 1
            && gs
                .player_active
                .ones()
                .all(|i| gs.current_round_player_bet(i) >= gs.current_round_bet());
        if !can_runout {
            return Err(fail(
                h,
                None,
                None,
                "street ended with outstanding action/call",
            ));
        }
        gs.round_data.needs_action = PlayerBitSet::default();
    }
    let bets = gs.round_data.player_bet.clone();
    if let Some((idx, &largest)) = bets.iter().enumerate().max_by_key(|(_, v)| *v) {
        let second = bets
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != idx)
            .map(|(_, v)| *v)
            .max()
            .unwrap_or(0);
        if largest > second {
            let amount = largest - second;
            if !contenders.get(idx) {
                return Err(fail(h, None, None, "folded player has unmatched live bet"));
            }
            gs.return_uncalled_bet(idx, amount);
            if gs.player_all_in.get(idx) && gs.stacks[idx] > 0 {
                gs.player_all_in.disable(idx);
                gs.player_active.enable(idx);
            }
        }
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::{RakeRate, RakeRounding};
    use crate::open_hand_history::{
        CoinPokerParser, GGPokerParser, HandHistoryParser, ParserOptions, PokerStarsParser,
    };
    #[test]
    fn parsed_room_fixtures_replay_exactly() {
        let ps = PokerStarsParser::default()
            .parse(include_str!("../../tests/fixtures/rooms/pokerstars.txt"))
            .unwrap()
            .remove(0);
        let result = replay_hand(&ps, RakeConfig::none()).unwrap();
        assert_eq!(result.final_state.board.len(), 4);
        assert_eq!(result.final_state.total_pot, 6);
        assert_eq!(
            result.final_state.stacks.as_slice(),
            &[152, 111, 234, 158, 226, 103]
        );
        assert_eq!(result.steps.len(), 14);
        assert!(result.verification.winners_by_cards);
        let cp = CoinPokerParser::default()
            .parse(include_str!("../../tests/fixtures/rooms/coinpoker.txt"))
            .unwrap()
            .remove(0);
        // Explicit caller policy; these flags are NOT asserted to be official.
        let rake = RakeConfig::new(
            RakeRate::new(5, 100).unwrap(),
            None,
            true,
            RakeRounding::Floor,
        )
        .unwrap();
        let result = replay_hand(&cp, rake).unwrap();
        assert_eq!(result.final_state.stacks.as_slice(), &[511, 488]);
        assert_eq!(result.final_state.rake_collected, 1);
        assert_eq!(result.final_state.total_pot, 24);
        assert_eq!(result.steps.len(), 4);
        let gg = GGPokerParser {
            options: ParserOptions {
                utc_offset: chrono::FixedOffset::east_opt(0),
            },
        }
        .parse(include_str!("../../tests/fixtures/rooms/ggpoker.txt"))
        .unwrap()
        .remove(0);
        let result = replay_hand(&gg, RakeConfig::none()).unwrap();
        assert_eq!(
            result.final_state.stacks.as_slice(),
            &[206, 169, 393, 251, 242, 972]
        );
        assert_eq!(result.final_state.total_pot, 12);
        assert_eq!(result.transitions().count(), 9);
    }
    #[test]
    fn errors_have_hand_action_and_exact_disagreement() {
        let mut h = CoinPokerParser::default()
            .parse(include_str!("../../tests/fixtures/rooms/coinpoker.txt"))
            .unwrap()
            .remove(0);
        let action = h.rounds[0]
            .actions
            .iter_mut()
            .find(|a| a.action == Action::Call)
            .unwrap();
        action.amount += 1;
        let number = action.action_number;
        let e = replay_hand(&h, RakeConfig::none()).unwrap_err();
        assert_eq!(e.action_number, Some(number));
        assert!(e.message.contains("exact"));

        action_number_test(&h);
    }
    fn action_number_test(h: &HandHistory) {
        let mut h = h.clone();
        h.players[1].seat = h.players[0].seat;
        assert!(replay_hand(&h, RakeConfig::none()).is_err());
    }

    #[test]
    fn replay_accepts_legacy_zero_based_action_numbers() {
        let mut hand = PokerStarsParser::default()
            .parse(include_str!("../../tests/fixtures/rooms/pokerstars.txt"))
            .unwrap()
            .remove(0);
        for round in &mut hand.rounds {
            for (i, action) in round.actions.iter_mut().enumerate() {
                action.action_number = i as u64;
            }
        }
        assert!(replay_hand(&hand, RakeConfig::none()).is_ok());
    }
    #[tokio::test]
    async fn arena_ohh_roundtrip_preserves_side_pots_refunds_and_rake() {
        use crate::arena::{
            Agent, HoldemSimulationBuilder, agent::AllInAgent, historian::VecHistorian,
        };
        use crate::open_hand_history::{ConverterConfig, HandHistoryBuilder};
        use rand::{SeedableRng, rngs::StdRng};
        let rake = RakeConfig::new(
            RakeRate::new(5, 100).unwrap(),
            Some(15),
            true,
            RakeRounding::HalfToEven,
        )
        .unwrap();
        for stacks in [vec![333, 666], vec![333, 666, 999]] {
            let historian = VecHistorian::new();
            let records = historian.get_storage();
            let state = GameStateBuilder::new()
                .stacks(&stacks)
                .blinds(2, 1)
                .dealer_idx(0)
                .max_raises_per_round(None)
                .rake(rake)
                .build()
                .unwrap();
            let agents: Vec<Box<dyn Agent>> = (0..stacks.len())
                .map(|_| Box::new(AllInAgent::default()) as Box<dyn Agent>)
                .collect();
            let mut sim = HoldemSimulationBuilder::default()
                .game_state(state)
                .agents(agents)
                .historians(vec![Box::new(historian)])
                .build_with_rng(StdRng::seed_from_u64(710))
                .unwrap();
            sim.run().await;
            let mut converter = HandHistoryBuilder::new(ConverterConfig::default());
            for record in records.lock().unwrap().iter() {
                converter
                    .record_action(sim.id, &record.action, &record.after_game_state)
                    .unwrap();
            }
            let hand = converter.build().unwrap();
            let json = serde_json::to_string(&hand).unwrap();
            let decoded: HandHistory = serde_json::from_str(&json).unwrap();
            assert_eq!(hand, decoded);
            let replayed = replay_hand(&decoded, rake).unwrap();
            assert!(replayed.verification.winners_by_cards);
            assert_eq!(replayed.final_state.stacks, sim.game_state.stacks);
            assert_eq!(replayed.final_state.player_bet, sim.game_state.player_bet);
            assert_eq!(
                replayed.final_state.player_winnings,
                sim.game_state.player_winnings
            );
            assert_eq!(replayed.final_state.total_pot, sim.game_state.total_pot);
            assert_eq!(replayed.final_state.rake_collected, 15);
            assert_eq!(
                replayed.final_state.stacks.iter().sum::<Chips>() + 15,
                stacks.iter().sum::<Chips>()
            );
            assert_eq!(hand.pots.len(), stacks.len() - 1);
            assert!(
                records.lock().unwrap().iter().any(|r| matches!(
                    r.action,
                    crate::arena::action::Action::ReturnUncalledBet(_)
                ))
            );
        }
    }

    // Synthetic edge case using researched CoinPoker grammar. Its caller-
    // supplied ceil/cap policy is not represented as official room policy.
    #[test]
    fn synthetic_sidepot_ties_odd_chips_and_shared_rake_cap_replay() {
        let hand = CoinPokerParser::default()
            .parse(include_str!(
                "../../tests/fixtures/rooms/coinpoker_synthetic_sidepots.txt"
            ))
            .unwrap()
            .remove(0);
        let rake = RakeConfig::new(
            RakeRate::new(5, 100).unwrap(),
            Some(1),
            true,
            RakeRounding::Ceil,
        )
        .unwrap();
        let replayed = replay_hand(&hand, rake).unwrap();
        assert_eq!(
            hand.pots
                .iter()
                .map(|p| (p.amount, p.rake))
                .collect::<Vec<_>>(),
            vec![(9, Some(1)), (2, Some(0))]
        );
        assert_eq!(replayed.final_state.stacks.as_slice(), &[2, 4, 4]);
        assert_eq!(replayed.final_state.player_winnings.as_slice(), &[2, 4, 4]);
        assert_eq!(replayed.final_state.total_pot, 11);
        assert_eq!(replayed.final_state.rake_collected, 1);
        assert_eq!(replayed.steps.len(), 3);
        assert!(replayed.verification.winners_by_cards);
        let mut bad = hand.clone();
        bad.rounds[3].cards = Some(vec![Card::try_from("As").unwrap()]);
        assert!(
            replay_hand(&bad, rake)
                .unwrap_err()
                .message
                .contains("duplicate")
        );
        let mut bad = hand.clone();
        bad.pots[0].player_wins[0].win_amount += 1;
        assert!(replay_hand(&bad, rake).is_err());
        let mut unknown = hand;
        for r in &mut unknown.rounds {
            r.actions
                .retain(|a| !(a.player_id == 2 && a.action == Action::ShowsCards));
            for (i, a) in r.actions.iter_mut().enumerate() {
                a.action_number = i as u64 + 1;
            }
        }
        let partial = replay_hand(&unknown, rake).unwrap();
        assert!(!partial.verification.winners_by_cards);
        assert_eq!(partial.final_state.stacks.as_slice(), &[2, 4, 4]);
    }

    #[tokio::test]
    async fn ante_allin_receives_hole_cards_and_short_blind_keeps_full_amount_owed() {
        use crate::arena::{
            Agent, HoldemSimulationBuilder, agent::CallingAgent, historian::VecHistorian,
        };
        use crate::open_hand_history::{ConverterConfig, HandHistoryBuilder};
        use rand::{SeedableRng, rngs::StdRng};
        for (stacks, ante) in [(vec![1, 100], 1), (vec![100, 100, 1], 0)] {
            let hist = VecHistorian::new();
            let records = hist.get_storage();
            let agents: Vec<Box<dyn Agent>> = (0..stacks.len())
                .map(|_| Box::new(CallingAgent::default()) as Box<dyn Agent>)
                .collect();
            let gs = GameStateBuilder::new()
                .stacks(&stacks)
                .blinds(2, 1)
                .ante(ante)
                .dealer_idx(0)
                .build()
                .unwrap();
            let mut sim = HoldemSimulationBuilder::default()
                .game_state(gs)
                .agents(agents)
                .historians(vec![Box::new(hist)])
                .build_with_rng(StdRng::seed_from_u64(101))
                .unwrap();
            sim.run().await;
            let mut builder = HandHistoryBuilder::new(ConverterConfig::default());
            for r in records.lock().unwrap().iter() {
                builder
                    .record_action(sim.id, &r.action, &r.after_game_state)
                    .unwrap();
            }
            let hand = builder.build().unwrap();
            let result = replay_hand(&hand, RakeConfig::none()).unwrap();
            assert_eq!(result.final_state.stacks, sim.game_state.stacks);
            assert!(result.verification.winners_by_cards);
            assert!(
                hand.rounds[0]
                    .actions
                    .iter()
                    .filter(|a| a.action == Action::DealtCards)
                    .count()
                    == stacks.len()
            );
            if ante == 0 {
                let first = result.steps.first().unwrap();
                assert_eq!(first.state_before.current_round_bet(), 2);
                assert_eq!(first.state_before.current_round_player_bet(2), 1);
                assert_eq!(first.state_after.current_round_player_bet(0), 2);
            }
        }
    }
}
