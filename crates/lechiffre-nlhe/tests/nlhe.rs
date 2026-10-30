use lechiffre_nlhe::{
    game::{
        ACTIONS, ALL_IN, BIG_BLIND, CALL, FOLD, GAME_ID, HALF_POT, INPUTS, MIN_RAISE, Nlhe, POT,
        STACK,
    },
    matchplay::{self, Opponent},
};
use lechiffre_sdcfr::{
    game::{Game, Node},
    rng::Rng,
    training::{Config, Trainer},
};

const DEAL: [u8; 9] = [12, 25, 11, 24, 0, 15, 30, 45, 7]; // AA versus KK, disconnected board
fn chance(mut state: Nlhe) -> Nlhe {
    let mut rng = Rng::new(0);
    while state.node() == Node::Chance {
        state = state.sample_chance(&mut rng);
    }
    state
}
fn finish_calling(mut state: Nlhe) -> Nlhe {
    for _ in 0..30 {
        state = chance(state);
        if state.node() == Node::Terminal {
            return state;
        }
        state = state.child(CALL);
    }
    panic!("hand failed to terminate");
}

#[test]
fn blinds_action_amounts_and_duplicate_sizes_are_correct() {
    let state = Nlhe::from_deal(DEAL).unwrap();
    let gs = state.engine();
    assert_eq!(state.node(), Node::Player(0));
    assert_eq!(gs.stacks.as_slice(), [19, 18]);
    assert_eq!(gs.total_pot, 3);
    assert_eq!(state.to_call(), 1);
    let menu = state.action_targets();
    assert_eq!(menu[FOLD], Some(0));
    assert_eq!(menu[CALL], Some(2));
    assert_eq!(menu[MIN_RAISE], Some(4));
    assert_eq!(menu[HALF_POT], None);
    assert_eq!(menu[POT], Some(6));
    assert_eq!(menu[ALL_IN], Some(20));
    assert_eq!(state.child(MIN_RAISE).engine().total_pot, 6);
    assert_eq!(
        state.child(MIN_RAISE).action_targets(),
        [Some(0), Some(4), None, None, None, None]
    );
    assert_eq!(state.child(CALL).action_targets()[FOLD], None); // BB may check
}

#[test]
fn folds_and_allins_have_net_zero_sum_payoffs_in_big_blinds() {
    let state = Nlhe::from_deal(DEAL).unwrap();
    assert_eq!(state.child(FOLD).payoff(0), -0.5);
    assert_eq!(state.child(MIN_RAISE).child(FOLD).payoff(0), 1.0);
    let allin = state.child(ALL_IN);
    assert_eq!(allin.node(), Node::Player(1));
    let called = allin.child(CALL);
    assert_eq!(called.node(), Node::Chance);
    let terminal = chance(called);
    assert_eq!(terminal.node(), Node::Terminal);
    assert_eq!(terminal.engine().board.len(), 5);
    assert_eq!(terminal.payoff(0), 10.0);
    assert_eq!(terminal.payoff(1), -10.0);
    assert_eq!(terminal.engine().stacks.iter().sum::<i64>(), 40);
    let tied = Nlhe::from_deal([12, 25, 11, 24, 47, 48, 49, 50, 51]).unwrap();
    let tied = chance(tied.child(ALL_IN).child(CALL));
    assert_eq!(tied.payoff(0), 0.0);
    assert_eq!(tied.engine().stacks.as_slice(), [20, 20]);
}

#[test]
fn all_streets_card_removal_and_money_are_preserved() {
    for seed in 0..64 {
        let mut rng = Rng::new(seed);
        let mut state = Nlhe::default();
        let mut visits = Vec::new();
        for _ in 0..30 {
            assert_eq!(
                state.engine().stacks.iter().sum::<i64>() + state.engine().total_pot,
                2 * STACK
            );
            match state.node() {
                Node::Terminal => break,
                Node::Chance => state = state.sample_chance(&mut rng),
                Node::Player(player) => {
                    visits.push((state.street(), player));
                    let obs = state.observation();
                    assert_eq!(obs.features.len(), INPUTS);
                    assert_eq!(obs.features[..52].iter().sum::<f32>(), 2.0);
                    assert_eq!(
                        state.engine().hands[player].count(),
                        state.engine().board.len() + 2
                    );
                    state = state.child(CALL);
                }
            }
        }
        assert_eq!(state.node(), Node::Terminal);
        assert_eq!(state.remaining_cards(), 43);
        assert_eq!(
            visits,
            [
                (0, 0),
                (0, 1),
                (1, 1),
                (1, 0),
                (2, 1),
                (2, 0),
                (3, 1),
                (3, 0)
            ]
        );
        assert_eq!(state.payoff(0) + state.payoff(1), 0.0);
        assert!(state.engine().stacks.iter().all(|&stack| stack >= 0));
    }
}

#[test]
fn hidden_opponent_and_future_cards_do_not_enter_observations() {
    let a = Nlhe::from_deal(DEAL).unwrap();
    let b = Nlhe::from_deal([12, 25, 10, 23, 0, 15, 30, 44, 6]).unwrap();
    assert_eq!(a.observation(), b.observation());
    let a = chance(a.child(CALL).child(CALL)).child(CALL);
    let b = chance(b.child(CALL).child(CALL)).child(CALL);
    assert_eq!(a.node(), Node::Player(0));
    assert_eq!(a.observation(), b.observation());
    let c = Nlhe::from_deal([12, 25, 11, 24, 1, 15, 30, 45, 7]).unwrap();
    let c = chance(c.child(CALL).child(CALL)).child(CALL);
    assert_ne!(a.observation(), c.observation());
}

#[test]
fn complete_ordered_history_is_retained() {
    let a = Nlhe::from_deal(DEAL).unwrap();
    let direct = chance(a.child(MIN_RAISE).child(CALL));
    let limp = chance(a.child(CALL).child(MIN_RAISE).child(CALL));
    assert_eq!(direct.engine().stacks, limp.engine().stacks);
    assert_eq!(direct.node(), limp.node());
    assert_ne!(direct.observation(), limp.observation());
    assert_eq!(limp.history()[0].len(), 3);
    assert!(Nlhe::from_deal([0; 9]).is_err());
    assert!(Nlhe::from_deal([52, 25, 11, 24, 0, 15, 30, 45, 7]).is_err());
}

#[test]
fn independent_openspiel_full_nlhe_accepts_every_offered_action_and_same_returns() {
    let mut state = None;
    let mut cases = 0;
    let mut decisions = 0;
    for line in include_str!("fixtures/openspiel_nlhe.tsv").lines() {
        if line.starts_with('#') || line.starts_with(' ') {
            continue;
        }
        let fields = line.split('\t').collect::<Vec<_>>();
        match fields[0] {
            "D" => {
                let cards = fields[1]
                    .split(',')
                    .map(|c| c.parse().unwrap())
                    .collect::<Vec<u8>>();
                state = Some(Nlhe::from_deal(cards.try_into().unwrap()).unwrap());
                cases += 1;
            }
            "P" => {
                let current = chance(state.take().unwrap());
                let gs = current.engine();
                let player = fields[1].parse::<usize>().unwrap();
                assert_eq!(current.node(), Node::Player(player), "case {cases}");
                for seat in 0..2 {
                    let spent = fields[2 + seat].parse::<i64>().unwrap();
                    assert_eq!(gs.player_bet[seat], spent, "case {cases}");
                    assert_eq!(gs.stacks[seat], STACK - spent, "case {cases}");
                }
                let legal = fields[4]
                    .split(',')
                    .map(|v| v.parse().unwrap())
                    .collect::<Vec<i64>>();
                let before_street = gs.player_bet[player] - gs.current_round_player_bet(player);
                for (slot, target) in current.action_targets().into_iter().enumerate() {
                    if let Some(target) = target {
                        let acpc = if slot == FOLD {
                            0
                        } else if slot == CALL {
                            1
                        } else {
                            target + before_street
                        };
                        assert!(
                            legal.contains(&acpc),
                            "case {cases}, action {slot}, target {target}, ACPC {acpc}, legal {legal:?}"
                        );
                    }
                }
                let slot = fields[5].parse().unwrap();
                state = Some(current.child(slot));
                decisions += 1;
            }
            "T" => {
                let current = chance(state.take().unwrap());
                assert_eq!(current.node(), Node::Terminal, "case {cases}");
                assert_eq!(
                    current.payoff(0),
                    fields[1].parse::<f32>().unwrap(),
                    "case {cases}"
                );
                assert_eq!(
                    current.payoff(1),
                    fields[2].parse::<f32>().unwrap(),
                    "case {cases}"
                );
            }
            _ => panic!("unknown fixture line"),
        }
    }
    assert_eq!(cases, 400);
    assert!(decisions > 1000);
}

#[tokio::test]
async fn adapter_agrees_with_full_rs_poker_simulation() {
    use rs_poker::arena::{
        Agent, GameStateBuilder, HoldemSimulationBuilder,
        agent::{AllInAgent, CallingAgent, FoldingAgent},
    };
    for mode in 0..3 {
        for seed in 0..8 {
            let first: Box<dyn Agent> = match mode {
                0 => Box::new(CallingAgent::default()),
                1 => Box::new(AllInAgent::default()),
                _ => Box::new(FoldingAgent::default()),
            };
            let gs = GameStateBuilder::new()
                .stacks([STACK; 2])
                .blinds(BIG_BLIND, 1)
                .dealer_idx(0)
                .max_raises_per_round(None)
                .build()
                .unwrap();
            let mut sim = HoldemSimulationBuilder::default()
                .game_state(gs)
                .agents(vec![first, Box::new(CallingAgent::default())])
                .build_with_rng(rs_poker::arena::seeded_rng(Some(seed)))
                .unwrap();
            sim.run().await;
            let result = &sim.game_state;
            let mut cards = Vec::new();
            for player in 0..2 {
                cards.extend(
                    result.hands[player]
                        .iter()
                        .filter(|card| !result.board.contains(card))
                        .map(u8::from),
                );
            }
            assert_eq!(cards.len(), 4);
            cards.extend(result.board.iter().copied().map(u8::from));
            for c in 0..52 {
                if cards.len() == 9 {
                    break;
                }
                if !cards.contains(&c) {
                    cards.push(c);
                }
            }
            let state = Nlhe::from_deal(cards.try_into().unwrap()).unwrap();
            let terminal = match mode {
                0 => finish_calling(state),
                1 => chance(state.child(ALL_IN).child(CALL)),
                _ => state.child(FOLD),
            };
            assert_eq!(terminal.node(), Node::Terminal);
            assert_eq!(
                terminal.engine().stacks,
                result.stacks,
                "mode {mode} seed {seed}"
            );
        }
    }
}

fn trained() -> Trainer {
    let config = Config {
        hidden: 8,
        traversals: 2,
        train_steps: 2,
        batch_size: 4,
        reservoir_capacity: 64,
        ..Config::default()
    };
    let mut trainer = Trainer::new(GAME_ID, INPUTS, ACTIONS, config).unwrap();
    trainer.step(&Nlhe::default()).unwrap();
    trainer
}

#[test]
fn checkpoint_resume_is_identical_and_game_ids_are_enforced() {
    let mut trainer = trained();
    let path = std::env::temp_dir().join(format!("lechiffre-nlhe-{}.bin", std::process::id()));
    let _ = std::fs::remove_file(&path);
    trainer.save(&path).unwrap();
    assert!(Trainer::load(&path, lechiffre_sdcfr::game::Kuhn::GAME_ID).is_err());
    let mut resumed = Trainer::load(&path, GAME_ID).unwrap();
    trainer.step(&Nlhe::default()).unwrap();
    resumed.step(&Nlhe::default()).unwrap();
    assert_eq!(trainer, resumed);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn duplicate_self_matches_cancel_and_sampling_is_reproducible() {
    let trainer = trained();
    let own = matchplay::duel(&trainer, Opponent::Neural(&trainer), 40, 12).unwrap();
    assert_eq!(own.total_bb, 0.0);
    assert_eq!(own.ci95, [0.0; 2]);
    let a = matchplay::duel(&trainer, Opponent::Random, 100, 9).unwrap();
    let b = matchplay::duel(&trainer, Opponent::Random, 100, 9).unwrap();
    assert_eq!(a, b);
    assert!((a.total_bb / 100.0 * 100.0 - a.bb_per_100).abs() < 1e-9);
    assert!(matchplay::duel(&trainer, Opponent::Passive, 3, 0).is_err());
    let wrong = Trainer::new("wrong", 7, 2, Config::default()).unwrap();
    assert!(matchplay::duel(&wrong, Opponent::Passive, 4, 0).is_err());
}
