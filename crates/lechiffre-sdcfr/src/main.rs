use lechiffre_sdcfr::{
    arena::{self, Baseline},
    exact::{Analysis, Tree},
    game::{EnumeratedGame, Kuhn},
    leduc::Leduc,
    training::{Config, Trainer},
};
use std::{
    collections::BTreeMap,
    error::Error,
    fs::{self, OpenOptions},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    time::{Instant, SystemTime, UNIX_EPOCH},
};
type Result<T> = std::result::Result<T, Box<dyn Error>>;

fn main() {
    if let Err(e) = run() {
        eprintln!("Erreur : {e}");
        std::process::exit(1);
    }
}
fn run() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let command = args.next().unwrap_or_else(|| "help".into());
    if ["help", "--help", "-h"].contains(&command.as_str()) {
        println!(
            "LeChiffre : MLP Single Deep CFR sur Kuhn et Leduc\n\
train [--game kuhn|leduc] [--iterations N] [--seed 42] [--out dossier]\n\
      [--traversals 512] [--steps N] [--batch-size 64] [--capacity 50000]\n\
      [--hidden N] [--learning-rate N] [--eval-every 16]\n\
resume --game leduc --checkpoint fichier.bin [--iterations 32] [--out dossier]\n\
eval --game leduc --checkpoint fichier.bin\n\
oracle [--game kuhn|leduc] [--iterations N]\n\n\
match --game leduc --checkpoint fichier.bin [--opponent all] [--hands 100000]\n\
      [--seed 123] [--oracle-iterations 1000] [--out dossier]\n\
      Profils : all, random, passive, aggressive, oracle, self, checkpoint\n\
      Pour checkpoint : ajouter --opponent-checkpoint autre.bin\n\n\
Sans --game, le jeu reste Kuhn pour conserver les anciennes commandes.\n\
Leduc : 6 cartes, ante 1, mises 2/4, une ouverture et une relance par tour.\n\
Exploitabilite = NashConv/2, jetons d'ante par main. Reprise : iterations ajoutees.\n\
Les sauvegardes Kuhn et Leduc sont separees ; le raccordement NLHE reste a faire."
        );
        return Ok(());
    }
    let mut options = BTreeMap::new();
    while let Some(key) = args.next() {
        if !key.starts_with("--") {
            return Err(format!("option invalide : {key}").into());
        }
        let value = args
            .next()
            .ok_or_else(|| format!("valeur manquante : {key}"))?;
        if value.starts_with("--") || options.insert(key.clone(), value).is_some() {
            return Err(format!("option repetee ou valeur manquante : {key}").into());
        }
    }
    let allowed = match command.as_str() {
        "train" => vec![
            "--game",
            "--iterations",
            "--seed",
            "--out",
            "--traversals",
            "--steps",
            "--batch-size",
            "--capacity",
            "--hidden",
            "--learning-rate",
            "--eval-every",
        ],
        "resume" => vec![
            "--game",
            "--checkpoint",
            "--iterations",
            "--out",
            "--eval-every",
        ],
        "eval" => vec!["--game", "--checkpoint"],
        "oracle" => vec!["--game", "--iterations"],
        "match" => vec![
            "--game",
            "--checkpoint",
            "--opponent",
            "--hands",
            "--seed",
            "--oracle-iterations",
            "--opponent-checkpoint",
            "--out",
        ],
        _ => return Err(format!("commande inconnue : {command}").into()),
    };
    for key in options.keys() {
        if !allowed.contains(&key.as_str()) {
            return Err(format!("option inconnue : {key}").into());
        }
    }
    match options.get("--game").map_or("kuhn", String::as_str) {
        "kuhn" => run_game(
            &Kuhn::default(),
            "kuhn",
            Kuhn::GAME_ID,
            7,
            2,
            &command,
            &options,
        ),
        "leduc" => run_game(
            &Leduc::default(),
            "leduc",
            Leduc::GAME_ID,
            Leduc::INPUTS,
            3,
            &command,
            &options,
        ),
        name => Err(format!("jeu inconnu : {name}").into()),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_game<G: EnumeratedGame>(
    root: &G,
    name: &str,
    game_id: &str,
    inputs: usize,
    actions: usize,
    command: &str,
    options: &BTreeMap<String, String>,
) -> Result<()> {
    let number = |key: &str, default: u64| -> Result<u64> {
        Ok(options
            .get(key)
            .map(|s| s.parse())
            .transpose()?
            .unwrap_or(default))
    };
    let tree = Tree::compile(root, game_id);
    println!(
        "Jeu : {name} | {} situations d'information | {} noeuds",
        tree.infosets().len(),
        tree.node_count()
    );
    if command == "oracle" {
        let iterations = number("--iterations", if name == "leduc" { 1000 } else { 10_000 })?;
        if iterations == 0 {
            return Err("iterations doit etre positif".into());
        }
        let e = tree.evaluate(&tree.train_tabular(iterations));
        println!(
            "Oracle tabulaire, {iterations} iterations | valeur J0 {:+.9} | exploitabilite {:.9}",
            e.value_player0, e.exploitability
        );
        return Ok(());
    }
    let checkpoint = || -> Result<&str> {
        options
            .get("--checkpoint")
            .map(String::as_str)
            .ok_or_else(|| "--checkpoint requis".into())
    };
    let mut trainer = if ["resume", "eval", "match"].contains(&command) {
        Trainer::load(checkpoint()?, game_id)?
    } else {
        let defaults = if name == "leduc" {
            Config {
                hidden: 64,
                train_steps: 128,
                learning_rate: 0.003,
                ..Config::default()
            }
        } else {
            Config::default()
        };
        let config = Config {
            traversals: usize::try_from(number("--traversals", defaults.traversals as u64)?)?,
            train_steps: usize::try_from(number("--steps", defaults.train_steps as u64)?)?,
            batch_size: usize::try_from(number("--batch-size", defaults.batch_size as u64)?)?,
            reservoir_capacity: usize::try_from(number(
                "--capacity",
                defaults.reservoir_capacity as u64,
            )?)?,
            hidden: usize::try_from(number("--hidden", defaults.hidden as u64)?)?,
            learning_rate: options
                .get("--learning-rate")
                .map(|s| s.parse())
                .transpose()?
                .unwrap_or(defaults.learning_rate),
            seed: number("--seed", 42)?,
        };
        Trainer::new(game_id, inputs, actions, config)?
    };
    let shape = trainer.network_shape();
    if shape[0] != inputs || shape[shape.len() - 1] != actions {
        return Err("sauvegarde : dimensions incompatibles avec le jeu".into());
    }
    if command == "match" {
        return run_matches(&tree, &trainer, game_id, inputs, actions, options);
    }
    if command == "eval" {
        if trainer.completed() == 0 {
            return Err("aucune iteration sauvegardee".into());
        }
        print_analysis(trainer.completed(), &tree.analyze(&trainer));
        return Ok(());
    }
    let iterations = number(
        "--iterations",
        if command == "resume" {
            32
        } else if name == "leduc" {
            128
        } else {
            64
        },
    )?;
    let eval_every = number("--eval-every", if name == "leduc" { 16 } else { 8 })?;
    if iterations == 0 || eval_every == 0 {
        return Err("iterations et eval-every doivent etre positifs".into());
    }
    let default_out = format!(
        "runs/{name}-{}-{}",
        trainer.config().seed,
        SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis()
    );
    let out = PathBuf::from(options.get("--out").unwrap_or(&default_out));
    fs::create_dir_all(&out)?;
    let mut metrics = BufWriter::new(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(out.join("metrics.csv"))?,
    );
    writeln!(
        metrics,
        "iteration,train_mse0,train_mse1,value0,nash_conv,exploitability,external_regret0,external_regret1,seen0,seen1,stored0,stored1,seconds"
    )?;
    println!(
        "MLP {inputs} -> {} -> {} -> {actions}, CPU Rust, seed {}",
        trainer.config().hidden,
        trainer.config().hidden,
        trainer.config().seed
    );
    println!("Dossier : {}", out.display());
    let start = Instant::now();
    let mut final_analysis = None;
    for i in 0..iterations {
        let report = trainer.step(root)?;
        if report.iteration % eval_every == 0 || i == 0 || i + 1 == iterations {
            let analysis = tree.analyze(&trainer);
            print_analysis(report.iteration, &analysis);
            println!(
                "  MSE {:.4}/{:.4} | echantillons memorises {}/{}",
                report.loss[0], report.loss[1], report.samples_stored[0], report.samples_stored[1]
            );
            let e = analysis.evaluation;
            let r = analysis.external_regret;
            writeln!(
                metrics,
                "{},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{},{},{},{},{:.3}",
                report.iteration,
                report.loss[0],
                report.loss[1],
                e.value_player0,
                e.nash_conv,
                e.exploitability,
                r[0],
                r[1],
                report.samples_seen[0],
                report.samples_seen[1],
                report.samples_stored[0],
                report.samples_stored[1],
                start.elapsed().as_secs_f64()
            )?;
            metrics.flush()?;
            if i + 1 == iterations {
                final_analysis = Some(analysis);
            }
        }
        if report.iteration % 16 == 0 || i + 1 == iterations {
            trainer.save(out.join(format!("checkpoint-{:06}.bin", report.iteration)))?;
        }
    }
    write_policy(&out, &tree, &final_analysis.unwrap())?;
    println!(
        "Sauvegarde : {}",
        out.join(format!("checkpoint-{:06}.bin", trainer.completed()))
            .display()
    );
    println!("Termine en {:.2} s.", start.elapsed().as_secs_f64());
    Ok(())
}

fn run_matches(
    tree: &Tree,
    trainer: &Trainer,
    game_id: &str,
    inputs: usize,
    actions: usize,
    options: &BTreeMap<String, String>,
) -> Result<()> {
    let number = |key: &str, default: u64| -> Result<u64> {
        Ok(options
            .get(key)
            .map(|s| s.parse())
            .transpose()?
            .unwrap_or(default))
    };
    if trainer.completed() == 0 {
        return Err("aucune strategie sauvegardee".into());
    }
    let hands = number("--hands", 100_000)?;
    if hands < 4 || (hands & 1) != 0 {
        return Err("--hands doit etre pair et au moins egal a 4".into());
    }
    let seed = number("--seed", 123)?;
    let oracle_iterations = number("--oracle-iterations", 1000)?;
    if oracle_iterations == 0 {
        return Err("--oracle-iterations doit etre positif".into());
    }
    let requested = options.get("--opponent").map_or("all", String::as_str);
    let opponents = match requested {
        "all" => vec!["random", "passive", "aggressive", "oracle"],
        "random" | "passive" | "aggressive" | "oracle" | "self" | "checkpoint" => vec![requested],
        _ => return Err(format!("adversaire inconnu : {requested}").into()),
    };
    if requested != "checkpoint" && options.contains_key("--opponent-checkpoint") {
        return Err("--opponent-checkpoint exige --opponent checkpoint".into());
    }
    if requested == "checkpoint" && !options.contains_key("--opponent-checkpoint") {
        return Err("--opponent-checkpoint requis".into());
    }
    let hero = tree.analyze(trainer).policy;
    let mut csv = if let Some(folder) = options.get("--out") {
        fs::create_dir_all(folder)?;
        let mut file = BufWriter::new(
            OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(Path::new(folder).join("matches.csv"))?,
        );
        writeln!(
            file,
            "opponent,hands,pairs,seed,total_chips,chips_per_100,ci95_low,ci95_high,exact_per_100,mean_seat0,mean_seat1"
        )?;
        Some(file)
    } else {
        None
    };
    println!(
        "Strategie moyenne de ton IA : {} iterations",
        trainer.completed()
    );
    println!(
        "{hands} mains par adversaire ; {} paires avec echange des places ; seed {seed}",
        hands / 2
    );
    println!("Gains de TON IA en jetons/100 mains ; IC95 approximatif calcule sur les paires.");
    for opponent in opponents {
        let policy = match opponent {
            "random" => Baseline::Random.policy(tree)?,
            "passive" => Baseline::Passive.policy(tree)?,
            "aggressive" => Baseline::Aggressive.policy(tree)?,
            "oracle" => tree.train_tabular(oracle_iterations),
            "self" => hero.clone(),
            "checkpoint" => {
                let other = Trainer::load(&options["--opponent-checkpoint"], game_id)?;
                if other.completed() == 0 {
                    return Err("strategie adverse vide".into());
                }
                let shape = other.network_shape();
                if shape[0] != inputs || shape[shape.len() - 1] != actions {
                    return Err("dimensions de la strategie adverse incompatibles".into());
                }
                tree.analyze(&other).policy
            }
            _ => unreachable!(),
        };
        let result = arena::duel(tree, &hero, &policy, hands, seed)?;
        let label = if opponent == "oracle" {
            format!("oracle({oracle_iterations})")
        } else {
            opponent.into()
        };
        println!(
            "{label:16} | gain {:+.3} | IC95 [{:+.3}, {:+.3}] | exact {:+.3} | total {:+.1}",
            result.chips_per_100,
            result.ci95_per_100[0],
            result.ci95_per_100[1],
            result.exact_per_100,
            result.total_chips
        );
        if let Some(file) = &mut csv {
            writeln!(
                file,
                "{label},{},{},{seed},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9}",
                result.hands,
                result.pairs,
                result.total_chips,
                result.chips_per_100,
                result.ci95_per_100[0],
                result.ci95_per_100[1],
                result.exact_per_100,
                result.mean_by_seat[0],
                result.mean_by_seat[1]
            )?;
            file.flush()?;
        }
    }
    if let Some(folder) = options.get("--out") {
        println!(
            "Resultats : {}",
            Path::new(folder).join("matches.csv").display()
        );
    }
    Ok(())
}
fn print_analysis(iteration: u64, a: &Analysis) {
    println!(
        "iteration {iteration:4} | exploitabilite {:.6} | valeur J0 {:+.6} | regret {:.6}/{:.6}",
        a.evaluation.exploitability,
        a.evaluation.value_player0,
        a.external_regret[0],
        a.external_regret[1]
    );
}
fn write_policy(out: &Path, tree: &Tree, analysis: &Analysis) -> Result<()> {
    let mut file = BufWriter::new(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(out.join("average-policy.csv"))?,
    );
    write!(file, "infoset,player,features")?;
    for action in 0..analysis.policy[0].len() {
        write!(file, ",action{action}")?;
    }
    writeln!(file)?;
    for (id, (info, row)) in tree.infosets().iter().zip(&analysis.policy).enumerate() {
        write!(file, "{id},{},\"{}\"", info.player, info.label)?;
        for p in row {
            write!(file, ",{p:.9}")?;
        }
        writeln!(file)?;
    }
    Ok(())
}
