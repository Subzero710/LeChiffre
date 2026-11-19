use rs_poker::arena::{GameStateBuilder, HoldemSimulationBuilder, agent::CallingAgent};

#[test]
fn observed_runout_is_dealt_by_rs_poker_without_hidden_card_collisions() {
    let cfg = noregrets::engine::HandConfig {
        num_players: 3,
        stack: 20,
        sb: 1,
        bb: 2,
    };
    let mut hand = noregrets::engine::Hand::new(&cfg, 0, noregrets::cards::fresh_deck());
    let hero = hand.hole(0);
    // Includes an opponent placeholder and cards from the original runout.
    let board = [2, 6, 7, 8, 9];
    hand.set_public_board(&board, 0).unwrap();
    assert!(hand.board().is_empty());
    assert_eq!(hand.hole(0), hero);
    assert!(hand.set_public_board(&[hero[0], 20, 21], 0).is_err());
    while !hand.is_terminal() {
        hand.apply(noregrets::engine::PlayerAction::CheckCall);
    }
    let mut actual = hand.board().to_vec();
    actual.sort_unstable();
    let mut expected = board.to_vec();
    expected.sort_unstable();
    assert_eq!(actual, expected);
    let mut unique = actual;
    for p in 0..3 {
        unique.extend(hand.hole(p));
    }
    unique.sort_unstable();
    unique.dedup();
    assert_eq!(unique.len(), 11);
    assert_eq!(hand.utilities().iter().sum::<i64>(), 0);
}

#[test]
fn short_all_in_does_not_reopen_a_completed_raise() {
    let cfg = noregrets::engine::HandConfig {
        num_players: 3,
        stack: 20,
        sb: 1,
        bb: 2,
    };
    let mut hand = noregrets::engine::Hand::new_with_stacks(
        &cfg,
        0,
        noregrets::cards::fresh_deck(),
        &[20, 8, 20],
    );
    use noregrets::engine::PlayerAction::{CheckCall, RaiseTo};
    assert_eq!(hand.to_act(), 0);
    hand.apply(RaiseTo(6));
    hand.apply(RaiseTo(8));
    hand.apply(CheckCall);
    assert_eq!(hand.to_act(), 0);
    assert_eq!(hand.current_bet(), 8);
    assert_eq!(hand.raise_bounds(), None);
    hand.apply(CheckCall);
    assert_eq!(hand.board().len(), 3);
    assert!(hand.raise_bounds().is_some());
}

use async_trait::async_trait;
use noregrets::{
    cards::fresh_deck,
    engine::{Hand as NrHand, HandConfig, PlayerAction, from_rs_card, to_rs_card},
};
use rand::{SeedableRng, seq::SliceRandom};
use rs_poker::{
    arena::{Agent, GameState, action::AgentAction, game_state::Round},
    core::Hand,
};
fn scripted(gs: &GameState, style: usize) -> AgentAction {
    match (gs.to_act_idx() + gs.board.len() + style) % 4 {
        0 => AgentAction::Call,
        2 => AgentAction::Fold,
        _ => {
            let max = gs.current_round_current_player_bet() + gs.current_player_stack();
            if max <= gs.current_round_bet() || gs.validate_bet_amount(max).is_err() {
                return AgentAction::Call;
            }
            let target = if style % 2 == 0 {
                max
            } else {
                (gs.current_round_bet() + gs.current_round_min_raise()).min(max)
            };
            AgentAction::Bet(target)
        }
    }
}
struct Script(usize);
#[async_trait]
impl Agent for Script {
    fn name(&self) -> &str {
        "Scripted regression"
    }
    async fn act(&mut self, _: u128, gs: &GameState) -> AgentAction {
        scripted(gs, self.0)
    }
}
#[tokio::test]
async fn actual_noregrets_facade_matches_unpaused_rs_poker_with_unequal_stacks() {
    for n in 2..=6 {
        for seed in 0..16 {
            let button = seed as usize % n;
            let mut deck = fresh_deck();
            deck.shuffle(&mut rand::rngs::SmallRng::seed_from_u64(seed));
            let stacks: Vec<u32> = (0..n).map(|p| 3 + 7 * p as u32).collect();
            let cfg = HandConfig {
                num_players: n,
                stack: 20,
                sb: 1,
                bb: 2,
            };
            let mut hand = NrHand::new_with_stacks(&cfg, button, deck, &stacks);
            let hands: Vec<_> = (0..n)
                .map(|p| {
                    Hand::new_with_cards(vec![to_rs_card(deck[2 * p]), to_rs_card(deck[2 * p + 1])])
                })
                .collect();
            let mut gs = GameStateBuilder::new()
                .stacks(stacks.iter().map(|&s| s as i64).collect::<Vec<_>>())
                .blinds(2, 1)
                .dealer_idx(button)
                .hands(hands)
                .round(Round::Preflop)
                .max_raises_per_round(None)
                .build()
                .unwrap();
            gs.sb_posted = false;
            gs.bb_posted = false;
            let mut native = HoldemSimulationBuilder::default()
                .game_state(gs)
                .agents(
                    (0..n)
                        .map(|_| Box::new(Script(seed as usize)) as Box<dyn Agent>)
                        .collect(),
                )
                .build_with_rng(rs_poker::arena::seeded_rng(Some(seed)))
                .unwrap();
            native
                .set_deal_order(deck[2 * n..2 * n + 5].iter().copied().map(to_rs_card))
                .unwrap();
            native.run().await;
            let mut steps = 0;
            while !hand.is_terminal() {
                let style = (hand.to_act() + hand.board().len() + seed as usize) % 4;
                let action = match style {
                    0 => PlayerAction::CheckCall,
                    2 => PlayerAction::Fold,
                    _ => hand
                        .raise_bounds()
                        .map(|(lo, hi)| PlayerAction::RaiseTo(if seed % 2 == 0 { hi } else { lo }))
                        .unwrap_or(PlayerAction::CheckCall),
                };
                hand.apply(action);
                steps += 1;
                assert!(steps < 200);
            }
            assert_eq!(
                hand.board(),
                native
                    .game_state
                    .board
                    .iter()
                    .copied()
                    .map(from_rs_card)
                    .collect::<Vec<_>>()
            );
            for p in 0..n {
                assert_eq!(
                    hand.utilities()[p],
                    native.game_state.player_reward(p),
                    "n={n}, seed={seed}, p={p}"
                );
            }
            assert_eq!(hand.utilities().iter().sum::<i64>(), 0);
        }
    }
}
#[test]
fn card_conversions_preserve_suit_names_and_hidden_resampling_preserves_public_cards() {
    for c in 0..52 {
        let rs = to_rs_card(c);
        assert_eq!(from_rs_card(rs), c);
        assert_eq!(rs.to_string(), noregrets::cards::card_str(c));
    }
    let cfg = HandConfig {
        num_players: 6,
        stack: 20,
        sb: 1,
        bb: 2,
    };
    let mut hand = NrHand::new(&cfg, 0, fresh_deck());
    while hand.board().is_empty() {
        hand.apply(PlayerAction::CheckCall);
    }
    let public = hand.board().to_vec();
    let hero = hand.hole(0);
    let mut rng = rand::rngs::SmallRng::seed_from_u64(1);
    for _ in 0..32 {
        hand.resample_hidden(Some(0), &mut rng);
        assert_eq!(hand.board(), public);
        assert_eq!(hand.hole(0), hero);
        let mut cards = public.clone();
        for p in 0..6 {
            cards.extend(hand.hole(p));
        }
        let before = cards.len();
        cards.sort();
        cards.dedup();
        assert_eq!(cards.len(), before);
    }
    while !hand.is_terminal() {
        hand.apply(PlayerAction::CheckCall);
    }
    assert_eq!(hand.utilities().iter().sum::<i64>(), 0);
}

#[tokio::test]
async fn controlled_steps_match_the_normal_engine_at_all_table_sizes() {
    for n in 2..=6 {
        for seed in 0..8 {
            let build = || {
                let gs = GameStateBuilder::new()
                    .stacks(vec![20; n])
                    .blinds(2, 1)
                    .dealer_idx(seed as usize % n)
                    .max_raises_per_round(None)
                    .build()
                    .unwrap();
                let agents = (0..n)
                    .map(|p| {
                        Box::new(CallingAgent::new(format!("P{p}")))
                            as Box<dyn rs_poker::arena::Agent>
                    })
                    .collect();
                HoldemSimulationBuilder::default()
                    .game_state(gs)
                    .agents(agents)
                    .build_with_rng(rs_poker::arena::seeded_rng(Some(seed)))
                    .unwrap()
            };
            let mut normal = build();
            normal.run().await;
            let mut controlled = build();
            while controlled.advance_to_decision().await {
                controlled
                    .run_agent_action(rs_poker::arena::action::AgentAction::Call)
                    .await;
            }
            assert_eq!(normal.game_state.board, controlled.game_state.board);
            for p in 0..n {
                assert_eq!(
                    normal.game_state.player_reward(p),
                    controlled.game_state.player_reward(p)
                );
            }
            assert_eq!(
                (0..n)
                    .map(|p| controlled.game_state.player_reward(p))
                    .sum::<i64>(),
                0
            );
        }
    }
}

#[tokio::test]
async fn controlled_all_ins_and_folds_terminate_without_losing_chips() {
    for n in 2..=6 {
        let gs = GameStateBuilder::new()
            .stacks((0..n).map(|i| 3 + 7 * i as i64).collect::<Vec<_>>())
            .blinds(2, 1)
            .max_raises_per_round(None)
            .build()
            .unwrap();
        let mut sim = HoldemSimulationBuilder::default()
            .game_state(gs)
            .build_with_rng(rs_poker::arena::seeded_rng(Some(7)))
            .unwrap();
        let mut actions = 0;
        while sim.advance_to_decision().await {
            let gs = &sim.game_state;
            let all_in = gs.current_round_current_player_bet() + gs.current_player_stack();
            sim.run_agent_action(rs_poker::arena::action::AgentAction::Bet(all_in))
                .await;
            actions += 1;
            assert!(actions < 100);
        }
        assert_eq!(
            (0..n).map(|p| sim.game_state.player_reward(p)).sum::<i64>(),
            0
        );
    }
}
