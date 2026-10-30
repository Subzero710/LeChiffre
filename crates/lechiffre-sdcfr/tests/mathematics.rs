use lechiffre_sdcfr::{
    evaluation,
    game::{Game, Kuhn, Node, Observation},
    memory::{Reservoir, Sample},
    neural::Mlp,
    rng::Rng,
    strategy::regret_matching,
    tabular,
    training::{Config, Trainer, traverse},
};

#[test]
fn opponent_cards_never_change_observation_or_policy() {
    let network = Mlp::initialized(&[7, 8, 2], &mut Rng::new(1));
    for history in 0..4 {
        let actor = if history == 0 || history == 3 { 0 } else { 1 };
        for own in 0..3 {
            let opponents = (0..3).filter(|&c| c != own).collect::<Vec<_>>();
            let mut a = [own, own];
            a[1 - actor] = opponents[0];
            let mut b = [own, own];
            b[1 - actor] = opponents[1];
            let obs_a = Kuhn::at(a, history).observation();
            let obs_b = Kuhn::at(b, history).observation();
            assert_eq!(obs_a, obs_b);
            assert_eq!(
                network.predict(&obs_a.features),
                network.predict(&obs_b.features)
            );
        }
    }
}

#[test]
fn kuhn_terminal_utilities_are_zero_sum_and_net_of_ante() {
    let root = Kuhn::deal([2, 0]);
    let cases = [
        (vec![0, 0], 1.0),
        (vec![1, 0], 1.0),
        (vec![1, 1], 2.0),
        (vec![0, 1, 0], -1.0),
        (vec![0, 1, 1], 2.0),
    ];
    for (actions, expected) in cases {
        let mut state = root;
        for action in actions {
            state = state.child(action);
        }
        assert_eq!(state.node(), Node::Terminal);
        assert_eq!(state.payoff(0), expected);
        assert_eq!(state.payoff(0) + state.payoff(1), 0.0);
    }
}

#[test]
fn regret_matching_masks_actions_and_handles_negative_regrets() {
    assert_eq!(
        regret_matching(&[-2.0, 100.0, -1.0], &[true, false, true]),
        vec![0.5, 0.0, 0.5]
    );
    assert_eq!(
        regret_matching(&[1.0, 3.0, 100.0], &[true, true, false]),
        vec![0.25, 0.75, 0.0]
    );
    assert_eq!(
        regret_matching(&[f32::MAX, f32::MAX], &[true, true]),
        vec![0.5, 0.5]
    );
}

#[test]
fn exact_best_response_recovers_analytic_equilibrium_and_uniform_exploitability() {
    let nash = evaluation::evaluate(&evaluation::nash_policy());
    assert!((nash.value_player0 + 1.0 / 18.0).abs() < 1e-7);
    assert!(nash.exploitability < 1e-7, "{nash:?}");
    let uniform = evaluation::evaluate(&evaluation::uniform_policy());
    assert!((uniform.value_player0 - 0.125).abs() < 1e-8);
    assert!((uniform.exploitability - 11.0 / 24.0).abs() < 1e-8);
}

#[test]
fn tabular_oracle_approaches_nash() {
    let e = evaluation::evaluate(&tabular::train(3000));
    assert!(e.exploitability < 0.015, "{e:?}");
    assert!((e.value_player0 + 1.0 / 18.0).abs() < 0.01);
}

#[test]
fn external_sampling_targets_have_expected_uniform_policy_mean() {
    let models = [Mlp::zeros(&[7, 2]), Mlp::zeros(&[7, 2])];
    let mut memory = Reservoir::new(40_000);
    let mut rng = Rng::new(16);
    for _ in 0..20_000 {
        traverse(&Kuhn::deal([0, 1]), 0, &models, 1, &mut memory, &mut rng);
    }
    let roots = memory
        .samples()
        .iter()
        .filter(|s| s.observation.features[3] == 1.0)
        .collect::<Vec<_>>();
    assert_eq!(roots.len(), 20_000);
    let mean = roots.iter().map(|s| s.targets[0] as f64).sum::<f64>() / roots.len() as f64;
    // E[v(check)] = -1.25, E[v(bet)] = -.5, so E[adv(check)] = -.375.
    assert!((mean + 0.375).abs() < 0.025, "mean={mean}");
    for sample in roots {
        assert!((sample.targets[0] + sample.targets[1]).abs() < 1e-6);
    }
}

#[test]
fn reservoir_keeps_early_and_late_samples_with_uniform_marginals() {
    let mut frequencies = [0; 20];
    for seed in 0..2000 {
        let mut memory = Reservoir::new(5);
        let mut rng = Rng::new(seed);
        for i in 0..20 {
            memory.push(
                Sample {
                    observation: Observation {
                        features: vec![i as f32],
                        legal: vec![true],
                    },
                    targets: vec![0.0],
                    iteration: 1,
                },
                &mut rng,
            );
        }
        assert_eq!(memory.seen(), 20);
        for sample in memory.samples() {
            frequencies[sample.observation.features[0] as usize] += 1;
        }
    }
    assert!(
        frequencies.iter().all(|n| (400..600).contains(n)),
        "{frequencies:?}"
    );
}

#[test]
fn checkpoint_resume_is_identical_to_uninterrupted_training() {
    let config = Config {
        traversals: 32,
        train_steps: 8,
        batch_size: 8,
        reservoir_capacity: 32,
        hidden: 8,
        ..Config::default()
    };
    let mut uninterrupted = Trainer::new(Kuhn::GAME_ID, 7, 2, config).unwrap();
    for _ in 0..2 {
        uninterrupted.step(&Kuhn::default()).unwrap();
    }
    let path =
        std::env::temp_dir().join(format!("lechiffre-sdcfr-test-{}.bin", std::process::id()));
    let _ = std::fs::remove_file(&path);
    uninterrupted.save(&path).unwrap();
    assert!(uninterrupted.save(&path).is_err());
    assert!(Trainer::load(&path, "wrong-encoder").is_err());
    let mut resumed = Trainer::load(&path, Kuhn::GAME_ID).unwrap();
    assert_eq!(uninterrupted, resumed);
    for _ in 0..3 {
        uninterrupted.step(&Kuhn::default()).unwrap();
        resumed.step(&Kuhn::default()).unwrap();
    }
    assert_eq!(uninterrupted, resumed);
    assert_eq!(
        evaluation::average_table(&uninterrupted),
        evaluation::average_table(&resumed)
    );
    let mut bytes = std::fs::read(&path).unwrap();
    bytes.truncate(bytes.len() / 2);
    std::fs::write(&path, bytes).unwrap();
    assert!(Trainer::load(&path, Kuhn::GAME_ID).is_err());
    std::fs::remove_file(path).unwrap();
}

#[test]
fn invalid_training_configuration_is_rejected() {
    for c in [
        Config {
            batch_size: 0,
            ..Config::default()
        },
        Config {
            learning_rate: f32::NAN,
            ..Config::default()
        },
        Config {
            train_steps: 0,
            ..Config::default()
        },
    ] {
        assert!(Trainer::new(Kuhn::GAME_ID, 7, 2, c).is_err());
    }
}

#[test]
fn measured_sequence_regret_matches_uniform_policy_and_nash_conv() {
    let config = Config {
        traversals: 8,
        train_steps: 1,
        batch_size: 8,
        hidden: 4,
        ..Config::default()
    };
    let mut trainer = Trainer::new(Kuhn::GAME_ID, 7, 2, config).unwrap();
    trainer.step(&Kuhn::default()).unwrap();
    let e = evaluation::evaluate(&evaluation::average_table(&trainer));
    let regrets = evaluation::sequence_regret(&trainer, &e);
    assert!((regrets[0] - 0.375).abs() < 1e-8);
    assert!((regrets[1] - 13.0 / 24.0).abs() < 1e-8);
    for _ in 0..3 {
        trainer.step(&Kuhn::default()).unwrap();
    }
    let e = evaluation::evaluate(&evaluation::average_table(&trainer));
    let regrets = evaluation::sequence_regret(&trainer, &e);
    assert!((regrets[0] + regrets[1] - e.nash_conv).abs() < 1e-12);
}
