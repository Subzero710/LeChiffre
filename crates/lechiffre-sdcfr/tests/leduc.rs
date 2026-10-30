use lechiffre_sdcfr::{
    evaluation,
    exact::Tree,
    game::{EnumeratedGame, Game, Kuhn, Node},
    leduc::{CALL, FOLD, Leduc, RAISE},
    neural::Mlp,
    rng::Rng,
    training::{Config, Trainer},
};
use std::collections::BTreeMap;

fn play(mut state: Leduc, actions: &[usize]) -> Leduc {
    for &action in actions {
        state = state.child(action);
    }
    state
}

#[test]
fn all_9450_states_match_independent_openspiel_rules() {
    let fixture = include_str!("fixtures/leduc_openspiel.tsv");
    let mut expected = fixture
        .lines()
        .filter(|line| !line.starts_with('#'))
        .map(|line| {
            let (key, value) = line.split_once('\t').unwrap();
            (key.to_string(), value.to_string())
        })
        .collect::<BTreeMap<_, _>>();
    fn visit(state: Leduc, private: [u8; 2], expected: &mut BTreeMap<String, String>) {
        let history = state
            .history()
            .iter()
            .map(|round| {
                round
                    .iter()
                    .map(|a| char::from(b'0' + a))
                    .collect::<String>()
            })
            .collect::<Vec<_>>();
        let board = state.board_card().map_or(-1, i32::from);
        let key = format!(
            "{}:{}:{}:{}:{}",
            private[0], private[1], board, history[0], history[1]
        );
        let reference = expected
            .remove(&key)
            .unwrap_or_else(|| panic!("extra state {key}"));
        let fields = reference.split('\t').collect::<Vec<_>>();
        let node = match state.node() {
            Node::Terminal => "T".into(),
            Node::Chance => "C".into(),
            Node::Player(p) => p.to_string(),
        };
        assert_eq!(node, fields[0], "{key}");
        match state.node() {
            Node::Terminal => {
                assert_eq!(state.payoff(0), fields[2].parse::<f32>().unwrap(), "{key}");
                assert_eq!(state.payoff(0) + state.payoff(1), 0.0);
            }
            Node::Chance => {
                let children = state.chance_outcomes();
                assert_eq!(children.len(), 4);
                for (child, p) in children {
                    assert_eq!(p, 0.25);
                    visit(child, private, expected);
                }
            }
            Node::Player(_) => {
                let mask = state.legal_mask();
                let text = mask
                    .iter()
                    .map(|&b| if b { '1' } else { '0' })
                    .collect::<String>();
                assert_eq!(text, fields[1], "{key}");
                assert_eq!(state.observation().features.len(), Leduc::INPUTS);
                assert!(state.history().iter().all(|h| h.len() <= 4));
                for (a, legal) in mask.into_iter().enumerate() {
                    if legal {
                        visit(state.child(a), private, expected);
                    }
                }
            }
        }
    }
    assert_eq!(expected.len(), 9450);
    for (state, p) in Leduc::default().chance_outcomes() {
        assert!((p - 1.0 / 30.0).abs() < 1e-14);
        // The enumeration order is ascending physical private cards, excluding duplicates.
        let mut found = None;
        for a in 0..6 {
            for b in 0..6 {
                if a != b && Leduc::deal([a, b]) == state {
                    found = Some([a, b]);
                }
            }
        }
        visit(state, found.unwrap(), &mut expected);
    }
    assert!(expected.is_empty());
}

#[test]
fn chance_respects_card_removal_and_physical_multiplicities() {
    let root = Leduc::default().chance_outcomes();
    assert_eq!(root.len(), 30);
    assert!((root.iter().map(|(_, p)| p).sum::<f64>() - 1.0).abs() < 1e-12);
    let board = play(Leduc::deal([0, 2]), &[CALL, CALL]);
    assert_eq!(board.node(), Node::Chance);
    let mut ranks = [0.0; 3];
    for (state, p) in board.chance_outcomes() {
        let card = state.board_card().unwrap();
        assert!(![0, 2].contains(&card));
        ranks[(card / 2) as usize] += p;
    }
    assert_eq!(ranks, [0.25, 0.25, 0.5]);
}

#[test]
fn hidden_cards_and_irrelevant_suits_do_not_enter_the_mlp() {
    let network = Mlp::initialized(&[43, 8, 3], &mut Rng::new(4));
    for history in [vec![], vec![CALL, RAISE]] {
        let a = play(Leduc::deal([0, 2]), &history);
        let b = play(Leduc::deal([0, 4]), &history);
        assert_eq!(a.node(), Node::Player(0));
        assert_eq!(a.observation(), b.observation());
        assert_eq!(
            network.predict(&a.observation().features),
            network.predict(&b.observation().features)
        );
    }
    let a = play(Leduc::deal([2, 0]), &[CALL]);
    let b = play(Leduc::deal([4, 1]), &[CALL]);
    assert_eq!(a.node(), Node::Player(1));
    assert_eq!(a.observation(), b.observation());
    let a = play(Leduc::deal([0, 2]), &[CALL, CALL]).reveal(1);
    let b = play(Leduc::deal([0, 4]), &[CALL, CALL]).reveal(1);
    assert_eq!(a.observation(), b.observation());
    let c = play(Leduc::deal([0, 2]), &[CALL, CALL]).reveal(4);
    assert_ne!(a.observation(), c.observation());
}

#[test]
fn ordered_history_is_kept_when_pots_and_current_players_match() {
    let a = play(Leduc::deal([0, 2]), &[RAISE, CALL]).reveal(4);
    let b = play(Leduc::deal([0, 2]), &[CALL, RAISE, CALL]).reveal(4);
    assert_eq!(a.contributions(), b.contributions());
    assert_eq!(a.node(), b.node());
    assert_ne!(a.observation(), b.observation());
}

#[test]
fn bet_cap_round_transition_pair_tie_and_fold_payoffs() {
    let a = Leduc::deal([0, 4]);
    assert_eq!(a.legal_mask(), vec![false, true, true]);
    let b = a.child(RAISE);
    assert_eq!(b.legal_mask(), vec![true, true, true]);
    assert_eq!(b.child(FOLD).payoff(0), 1.0);
    let c = b.child(RAISE);
    assert_eq!(c.legal_mask(), vec![true, true, false]);
    let d = c.child(CALL);
    assert_eq!(d.node(), Node::Chance);
    assert_eq!(d.contributions(), [5, 5]);
    let d = d.reveal(1);
    assert_eq!(d.node(), Node::Player(0));
    assert_eq!(d.to_call(), 0);
    let terminal = play(d, &[RAISE, RAISE, CALL]);
    assert_eq!(terminal.contributions(), [13, 13]);
    assert_eq!(terminal.payoff(0), 13.0);
    let tied = play(
        play(Leduc::deal([0, 1]), &[CALL, CALL]).reveal(4),
        &[CALL, CALL],
    );
    assert_eq!(tied.payoff(0), 0.0);
    assert_eq!(tied.payoff(1), 0.0);
}

#[test]
fn generic_best_responses_match_exhaustive_kuhn_evaluator() {
    let tree = Tree::compile(&Kuhn::default(), Kuhn::GAME_ID);
    let mut rng = Rng::new(9);
    for _ in 0..12 {
        let mut kuhn = evaluation::uniform_policy();
        for row in &mut kuhn {
            let p = rng.unit() as f32;
            *row = [p, 1.0 - p];
        }
        let policy = tree
            .infosets()
            .iter()
            .map(|info| {
                let features = &info.observation.features;
                let card = features[..3].iter().position(|&x| x == 1.0).unwrap();
                let history = features[3..].iter().position(|&x| x == 1.0).unwrap();
                let p = kuhn[card * 4 + history];
                let sum = p[0] as f64 + p[1] as f64;
                p.into_iter().map(|x| x as f64 / sum).collect::<Vec<_>>()
            })
            .collect();
        let a = tree.evaluate(&policy);
        let b = evaluation::evaluate(&kuhn);
        assert!((a.value_player0 - b.value_player0).abs() < 1e-10);
        for player in 0..2 {
            assert!(
                (a.best_response_values[player] - b.best_response_values[player]).abs() < 1e-10
            );
        }
    }
}

#[test]
fn leduc_uniform_exploitability_matches_openspiel_and_oracle_improves() {
    let tree = Tree::compile(&Leduc::default(), Leduc::GAME_ID);
    assert_eq!(tree.infosets().len(), 288);
    assert_eq!(tree.node_count(), 9451);
    let uniform = tree.evaluate(&tree.uniform_policy());
    assert!((uniform.exploitability - 2.373611111111111).abs() < 1e-10);
    assert!((uniform.value_player0 + 0.078125).abs() < 1e-10);
    let oracle = tree.evaluate(&tree.train_tabular(128));
    assert!(oracle.exploitability < 0.25, "{oracle:?}");
    assert!(oracle.best_response_values[0] >= oracle.value_player0 - 1e-10);
    assert!(oracle.best_response_values[1] >= -oracle.value_player0 - 1e-10);
}

#[test]
fn leduc_checkpoint_resumes_exactly_and_rejects_kuhn_identity() {
    let config = Config {
        hidden: 8,
        traversals: 16,
        train_steps: 4,
        batch_size: 8,
        reservoir_capacity: 64,
        ..Config::default()
    };
    let mut trainer = Trainer::new(Leduc::GAME_ID, 43, 3, config).unwrap();
    for _ in 0..2 {
        trainer.step(&Leduc::default()).unwrap();
    }
    let path =
        std::env::temp_dir().join(format!("lechiffre-leduc-test-{}.bin", std::process::id()));
    let _ = std::fs::remove_file(&path);
    trainer.save(&path).unwrap();
    assert!(Trainer::load(&path, Kuhn::GAME_ID).is_err());
    let mut resumed = Trainer::load(&path, Leduc::GAME_ID).unwrap();
    for _ in 0..2 {
        trainer.step(&Leduc::default()).unwrap();
        resumed.step(&Leduc::default()).unwrap();
    }
    assert_eq!(trainer, resumed);
    let tree = Tree::compile(&Leduc::default(), Leduc::GAME_ID);
    let analysis = tree.analyze(&resumed);
    assert!(
        (analysis.external_regret.iter().sum::<f64>() - analysis.evaluation.nash_conv).abs()
            < 1e-10
    );
    std::fs::remove_file(path).unwrap();
}
