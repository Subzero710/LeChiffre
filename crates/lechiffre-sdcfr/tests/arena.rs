use lechiffre_sdcfr::{
    arena::{self, Baseline},
    exact::Tree,
    game::Kuhn,
    leduc::Leduc,
};

#[test]
fn same_policy_duplicate_pairs_cancel_exactly_in_both_games() {
    for tree in [
        Tree::compile(&Kuhn::default(), Kuhn::GAME_ID),
        Tree::compile(&Leduc::default(), Leduc::GAME_ID),
    ] {
        for policy in [
            tree.uniform_policy(),
            Baseline::Aggressive.policy(&tree).unwrap(),
        ] {
            let result = arena::duel(&tree, &policy, &policy, 1000, 123).unwrap();
            assert_eq!(result.total_chips, 0.0);
            assert_eq!(result.chips_per_100, 0.0);
            assert_eq!(result.ci95_per_100, [0.0, 0.0]);
            assert_eq!(result.exact_per_100, 0.0);
        }
    }
}

#[test]
fn swap_agents_reverses_gain_without_changing_interval_width() {
    let tree = Tree::compile(&Leduc::default(), Leduc::GAME_ID);
    let hero = Baseline::Aggressive.policy(&tree).unwrap();
    let other = Baseline::Passive.policy(&tree).unwrap();
    let a = arena::duel(&tree, &hero, &other, 1000, 7).unwrap();
    let b = arena::duel(&tree, &other, &hero, 1000, 7).unwrap();
    assert_eq!(a.total_chips, -b.total_chips);
    assert!((a.chips_per_100 + b.chips_per_100).abs() < 1e-10);
    assert!((a.exact_per_100 + b.exact_per_100).abs() < 1e-10);
    assert!((a.ci95_per_100[0] + b.ci95_per_100[1]).abs() < 1e-10);
    assert_eq!(a.mean_by_seat[0], -b.mean_by_seat[1]);
}

#[test]
fn simulation_agrees_with_exact_head_to_head_expectation() {
    for tree in [
        Tree::compile(&Kuhn::default(), Kuhn::GAME_ID),
        Tree::compile(&Leduc::default(), Leduc::GAME_ID),
    ] {
        let hero = Baseline::Aggressive.policy(&tree).unwrap();
        let other = Baseline::Random.policy(&tree).unwrap();
        let r = arena::duel(&tree, &hero, &other, 100_000, 91).unwrap();
        let half = (r.ci95_per_100[1] - r.ci95_per_100[0]) / 2.0;
        assert!(
            (r.chips_per_100 - r.exact_per_100).abs() < 3.0 * half,
            "{r:?}"
        );
        assert!((r.total_chips / r.hands as f64 - r.chips_per_hand).abs() < 1e-12);
        assert!(((r.mean_by_seat[0] + r.mean_by_seat[1]) / 2.0 - r.chips_per_hand).abs() < 1e-12);
    }
}

#[test]
fn seeded_matches_are_reproducible_and_bad_hand_counts_are_rejected() {
    let tree = Tree::compile(&Kuhn::default(), Kuhn::GAME_ID);
    let a = Baseline::Passive.policy(&tree).unwrap();
    let b = Baseline::Random.policy(&tree).unwrap();
    assert_eq!(
        arena::duel(&tree, &a, &b, 1000, 5).unwrap(),
        arena::duel(&tree, &a, &b, 1000, 5).unwrap()
    );
    for hands in [0, 1, 2, 3, 5, 101] {
        assert!(arena::duel(&tree, &a, &b, hands, 0).is_err());
    }
}

#[test]
fn baseline_policies_choose_only_legal_actions() {
    for tree in [
        Tree::compile(&Kuhn::default(), Kuhn::GAME_ID),
        Tree::compile(&Leduc::default(), Leduc::GAME_ID),
    ] {
        for bot in [Baseline::Random, Baseline::Passive, Baseline::Aggressive] {
            let policy = bot.policy(&tree).unwrap();
            for (row, info) in policy.iter().zip(tree.infosets()) {
                assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-12);
                for (&p, &legal) in row.iter().zip(&info.observation.legal) {
                    assert!(legal || p == 0.0);
                }
            }
        }
    }
}
