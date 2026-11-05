use lechiffre_nlhe::game::{ACTIONS, GAME_ID, INPUTS, Nlhe};
use lechiffre_noregrets::{
    bridge::{Bridge, card_to_noregrets, duel},
    reference::{Reference, Settings, train},
};
use lechiffre_sdcfr::{
    game::{Game, Node},
    rng::Rng,
    training::{Config, Trainer},
};
use noregrets::{
    abstraction::{AbsConfig, Abstraction, MenuShape},
    bot::Policy,
};
use rand::{SeedableRng, rngs::SmallRng};
use std::{
    fs,
    sync::{Arc, OnceLock},
};

fn abs() -> Abstraction {
    Abstraction::new(AbsConfig {
        menu: MenuShape::LeChiffreReduced,
        ..AbsConfig::default()
    })
}

fn reference() -> Reference {
    static TRAINED: OnceLock<noregrets::cfr::Blueprint> = OnceLock::new();
    static META: OnceLock<lechiffre_noregrets::reference::Metadata> = OnceLock::new();
    let bp = TRAINED
        .get_or_init(|| {
            let r = train(&Settings {
                iterations: 256,
                seed: 7,
                threads: 1,
                buckets: 4,
                equity_rollouts: 8,
                centroid_samples: 0,
            })
            .unwrap();
            META.set(r.metadata).unwrap();
            (*r.policy.blueprint).clone()
        })
        .clone();
    let a = Arc::new(Abstraction::with_centroids(
        bp.abs_cfg.clone(),
        bp.centroids.clone(),
    ));
    Reference {
        policy: Policy::new(bp, a),
        metadata: META.get().unwrap().clone(),
    }
}

#[test]
fn both_engines_match_all_independent_openspiel_deals_and_decisions() {
    let abs = abs();
    let mut game = None;
    let (mut deals, mut decisions) = (0, 0);
    for line in include_str!("../../lechiffre-nlhe/tests/fixtures/openspiel_nlhe.tsv").lines() {
        let fields: Vec<_> = line.split('\t').collect();
        match fields[0] {
            "D" => {
                let cards: Vec<u8> = fields[1].split(',').map(|x| x.parse().unwrap()).collect();
                game = Some(Bridge::from_deal(cards.try_into().unwrap()).unwrap());
                deals += 1;
            }
            "P" => {
                let g = game.as_mut().unwrap();
                g.finish_chance();
                let p: usize = fields[1].parse().unwrap();
                assert_eq!(g.reduced.node(), Node::Player(p));
                assert_eq!(g.native.to_act(), p);
                for player in 0..2 {
                    assert_eq!(
                        g.native.hand_commit(player),
                        fields[player + 2].parse::<u32>().unwrap()
                    );
                }
                let pairs = g.actions(&abs).unwrap();
                assert_eq!(
                    pairs.len(),
                    g.reduced.observation().legal.iter().filter(|b| **b).count()
                );
                g.apply(fields[5].parse().unwrap(), &abs).unwrap();
                decisions += 1;
            }
            "T" => {
                let g = game.as_ref().unwrap();
                assert!(g.native.is_terminal());
                assert_eq!(g.reduced.node(), Node::Terminal);
                for p in 0..2 {
                    assert_eq!(
                        g.native.utilities()[p] as f64 / 2.0,
                        fields[p + 1].parse::<f64>().unwrap()
                    );
                }
            }
            _ => {}
        }
    }
    assert_eq!((deals, decisions), (400, 1147));
}

#[test]
fn random_replays_keep_engines_and_all_menus_in_sync() {
    let abs = abs();
    let mut rng = Rng::new(1234);
    for _ in 0..200 {
        let mut g = Bridge::from_deal(Nlhe::shuffled_deal(&mut rng)).unwrap();
        g.finish_chance();
        while g.reduced.node() != Node::Terminal {
            let actions = g.actions(&abs).unwrap();
            let slot = actions[rng.below(actions.len() as u64) as usize].0;
            g.apply(slot, &abs).unwrap();
        }
    }
}

#[test]
fn cards_preserve_rank_suit_and_are_bijective() {
    let mut seen = [false; 52];
    for card in 0..52 {
        let n = card_to_noregrets(card);
        assert_eq!(n / 4, card % 13);
        assert_eq!(n % 4, card / 13);
        assert!(!seen[n as usize]);
        seen[n as usize] = true;
    }
}

#[test]
fn reference_ignores_opponent_cards_and_unrevealed_board() {
    let r = reference();
    let a = Bridge::from_deal([0, 1, 2, 3, 4, 5, 6, 7, 8]).unwrap();
    let b = Bridge::from_deal([0, 1, 30, 31, 40, 41, 42, 43, 44]).unwrap();
    let (pa, ka) = a
        .reference_policy(&r, &mut SmallRng::seed_from_u64(2))
        .unwrap();
    let (pb, kb) = b
        .reference_policy(&r, &mut SmallRng::seed_from_u64(2))
        .unwrap();
    assert_eq!((pa, ka), (pb, kb));
    // Same public flop and hero's cards; change only villain and turn/river.
    let mut a = Bridge::from_deal([0, 1, 2, 3, 4, 5, 6, 7, 8]).unwrap();
    let mut b = Bridge::from_deal([0, 1, 30, 31, 4, 5, 6, 43, 44]).unwrap();
    for _ in 0..3 {
        a.apply(1, &r.policy.abs).unwrap();
        b.apply(1, &r.policy.abs).unwrap();
    }
    assert_eq!(a.native.to_act(), 0);
    let pa = a
        .reference_policy(&r, &mut SmallRng::seed_from_u64(2))
        .unwrap();
    let pb = b
        .reference_policy(&r, &mut SmallRng::seed_from_u64(2))
        .unwrap();
    assert_eq!(pa, pb);
}

#[test]
fn real_upstream_training_checkpoint_integrity_and_rule_guards() {
    let mut r = reference();
    assert!(!r.policy.blueprint.strategies.is_empty());
    let dir = std::env::temp_dir().join(format!("lechiffre-noregrets-{}", std::process::id()));
    if dir.exists() {
        fs::remove_dir_all(&dir).unwrap();
    }
    r.save(&dir).unwrap();
    assert!(r.save(&dir).is_err());
    let loaded = Reference::load(&dir).unwrap();
    assert_eq!(
        loaded.policy.blueprint.strategies,
        r.policy.blueprint.strategies
    );
    r.metadata.game_id = "wrong-game".into();
    assert!(r.validate().is_err());
    let path = dir.join("blueprint.bin");
    let mut bytes = fs::read(&path).unwrap();
    bytes[0] ^= 1;
    fs::write(path, bytes).unwrap();
    assert!(Reference::load(&dir).err().unwrap().contains("SHA-256"));
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn paired_duel_is_reproducible_and_rejects_incompatible_inputs() {
    let r = reference();
    let mut hero = Trainer::new(
        GAME_ID,
        INPUTS,
        ACTIONS,
        Config {
            traversals: 1,
            train_steps: 1,
            hidden: 4,
            reservoir_capacity: 64,
            batch_size: 4,
            ..Config::default()
        },
    )
    .unwrap();
    hero.step(&Nlhe::default()).unwrap();
    let a = duel(&hero, &r, 40, 123).unwrap();
    let b = duel(&hero, &r, 40, 123).unwrap();
    assert_eq!(a, b);
    assert_eq!(a.0.pairs, 20);
    assert!((a.0.total_bb / 40.0 * 100.0 - a.0.bb_per_100).abs() < 1e-9);
    assert!(duel(&hero, &r, 41, 123).is_err());
    let wrong = Trainer::new("wrong", INPUTS, ACTIONS, Config::default()).unwrap();
    assert!(duel(&wrong, &r, 40, 123).is_err());
}

#[test]
fn settings_reject_empty_work_and_invalid_centroid_budget() {
    let mut s = Settings {
        iterations: 1,
        seed: 1,
        threads: 1,
        buckets: 4,
        equity_rollouts: 8,
        centroid_samples: 0,
    };
    assert!(s.validate().is_ok());
    s.centroid_samples = 2;
    assert!(s.validate().is_err());
    s.centroid_samples = 0;
    s.iterations = 0;
    assert!(s.validate().is_err());
}
