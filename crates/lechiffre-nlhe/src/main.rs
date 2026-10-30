use lechiffre_nlhe::{
    game::{ACTIONS, GAME_ID, INPUTS, Nlhe},
    matchplay::{self, Opponent},
};
use lechiffre_sdcfr::training::{Config, Trainer};
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
    if let Err(error) = run() {
        eprintln!("Erreur : {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let mut arguments = std::env::args().skip(1);
    let command = arguments.next().unwrap_or_else(|| "help".into());
    if ["help", "--help", "-h"].contains(&command.as_str()) {
        println!(
            "LeChiffre : NLHE heads-up REDUIT, rs_poker + MLP Single Deep CFR\n\
Regles : 10 BB, blindes 1/2, 52 cartes, un bet/raise volontaire maximum par street.\n\
train [--iterations 16] [--traversals 8] [--steps 64] [--batch-size 64]\n\
      [--hidden 64] [--capacity 5000] [--learning-rate 0.001] [--seed 42] [--out dossier]\n\
resume --checkpoint fichier.bin [--iterations 8] [--out dossier]\n\
match --checkpoint fichier.bin [--opponent all] [--hands 2000] [--seed 123] [--out dossier]\n\
      Profils : random, passive, aggressive, all, self, checkpoint\n\
      Pour checkpoint : --opponent-checkpoint autre.bin\n\
eval --checkpoint fichier.bin [--hands 2000] [--seed 123] [--out dossier]\n\n\
Les gains des matchs sont en bb/100. Exploitabilite exacte : non calculee.\n\
Kuhn/Leduc restent dans le binaire crates/lechiffre-sdcfr."
        );
        return Ok(());
    }
    let mut options = BTreeMap::new();
    while let Some(key) = arguments.next() {
        if !key.starts_with("--") {
            return Err(format!("option invalide : {key}").into());
        }
        let value = arguments
            .next()
            .ok_or_else(|| format!("valeur manquante : {key}"))?;
        if value.starts_with("--") || options.insert(key.clone(), value).is_some() {
            return Err(format!("option repetee ou valeur manquante : {key}").into());
        }
    }
    let allowed = match command.as_str() {
        "train" => vec![
            "--iterations",
            "--traversals",
            "--steps",
            "--batch-size",
            "--hidden",
            "--capacity",
            "--learning-rate",
            "--seed",
            "--out",
        ],
        "resume" => vec!["--checkpoint", "--iterations", "--out"],
        "match" => vec![
            "--checkpoint",
            "--opponent",
            "--opponent-checkpoint",
            "--hands",
            "--seed",
            "--out",
        ],
        "eval" => vec!["--checkpoint", "--hands", "--seed", "--out"],
        "oracle" => return Err("oracle et exploitabilite exacte indisponibles en NLHE".into()),
        _ => return Err(format!("commande inconnue : {command}").into()),
    };
    for key in options.keys() {
        if !allowed.contains(&key.as_str()) {
            return Err(format!("option inconnue : {key}").into());
        }
    }
    let integer = |key: &str, default: u64| -> Result<u64> {
        Ok(options
            .get(key)
            .map(|s| s.parse())
            .transpose()?
            .unwrap_or(default))
    };
    let checkpoint = || -> Result<&str> {
        options
            .get("--checkpoint")
            .map(String::as_str)
            .ok_or_else(|| "--checkpoint requis".into())
    };
    let mut trainer = if command == "train" {
        let config = Config {
            traversals: usize::try_from(integer("--traversals", 8)?)?,
            train_steps: usize::try_from(integer("--steps", 64)?)?,
            batch_size: usize::try_from(integer("--batch-size", 64)?)?,
            reservoir_capacity: usize::try_from(integer("--capacity", 5000)?)?,
            hidden: usize::try_from(integer("--hidden", 64)?)?,
            learning_rate: options
                .get("--learning-rate")
                .map(|s| s.parse())
                .transpose()?
                .unwrap_or(0.001),
            seed: integer("--seed", 42)?,
        };
        Trainer::new(GAME_ID, INPUTS, ACTIONS, config)?
    } else {
        Trainer::load(checkpoint()?, GAME_ID)?
    };
    matchplay::validate(&trainer)?;
    if command == "match" || command == "eval" {
        return matches(&trainer, &options);
    }
    let iterations = integer("--iterations", if command == "resume" { 8 } else { 16 })?;
    if iterations == 0 {
        return Err("iterations doit etre positif".into());
    }
    let default_out = format!(
        "runs/nlhe-{}-{}",
        trainer.config().seed,
        SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis()
    );
    let out = PathBuf::from(options.get("--out").unwrap_or(&default_out));
    fs::create_dir_all(&out)?;
    let mut csv = BufWriter::new(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(out.join("metrics.csv"))?,
    );
    writeln!(
        csv,
        "iteration,train_mse0,train_mse1,seen0,seen1,stored0,stored1,seconds"
    )?;
    println!("NLHE reduit : 10 BB, 52 cartes, 4 streets, cap 1 bet/raise volontaire par street");
    println!(
        "MLP {INPUTS} -> {} -> {} -> {ACTIONS} ; exploitabilite exacte non calculee",
        trainer.config().hidden,
        trainer.config().hidden
    );
    println!("Dossier : {}", out.display());
    let start = Instant::now();
    for index in 0..iterations {
        let report = trainer.step(&Nlhe::default())?;
        println!(
            "iteration {:4} | MSE {:.6}/{:.6} | echantillons {}/{} | {:.2} s",
            report.iteration,
            report.loss[0],
            report.loss[1],
            report.samples_stored[0],
            report.samples_stored[1],
            start.elapsed().as_secs_f64()
        );
        writeln!(
            csv,
            "{},{:.9},{:.9},{},{},{},{},{:.3}",
            report.iteration,
            report.loss[0],
            report.loss[1],
            report.samples_seen[0],
            report.samples_seen[1],
            report.samples_stored[0],
            report.samples_stored[1],
            start.elapsed().as_secs_f64()
        )?;
        csv.flush()?;
        if report.iteration % 8 == 0 || index + 1 == iterations {
            trainer.save(out.join(format!("checkpoint-{:06}.bin", trainer.completed())))?;
        }
    }
    let final_path = out.join(format!("checkpoint-{:06}.bin", trainer.completed()));
    println!("Sauvegarde : {}", final_path.display());
    Ok(())
}

fn matches(trainer: &Trainer, options: &BTreeMap<String, String>) -> Result<()> {
    let hands = options
        .get("--hands")
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(2000_u64);
    let seed = options
        .get("--seed")
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(123_u64);
    if hands < 4 || (hands & 1) != 0 {
        return Err("--hands doit etre pair et au moins egal a 4".into());
    }
    let requested = options.get("--opponent").map_or("all", String::as_str);
    let names = match requested {
        "all" => vec!["random", "passive", "aggressive"],
        "random" | "passive" | "aggressive" | "self" | "checkpoint" => vec![requested],
        "oracle" => return Err("oracle tabulaire indisponible en NLHE".into()),
        _ => return Err(format!("adversaire inconnu : {requested}").into()),
    };
    let other = if requested == "checkpoint" {
        let path = options
            .get("--opponent-checkpoint")
            .ok_or("--opponent-checkpoint requis")?;
        Some(Trainer::load(path, GAME_ID)?)
    } else {
        if options.contains_key("--opponent-checkpoint") {
            return Err("option checkpoint incoherente".into());
        }
        None
    };
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
            "opponent,hands,pairs,seed,total_bb,bb_per_100,ci95_low,ci95_high,mean_bb_seat0,mean_bb_seat1"
        )?;
        Some(file)
    } else {
        None
    };
    println!(
        "NLHE reduit : strategie moyenne SD-CFR, reseau tire UNE fois par joueur et par main."
    );
    println!(
        "{hands} mains par adversaire, places alternees ; gains en bb/100 ; IC95 approximatif."
    );
    println!("Exploitabilite exacte et esperance exacte : non calculees.");
    for name in names {
        let opponent = match name {
            "random" => Opponent::Random,
            "passive" => Opponent::Passive,
            "aggressive" => Opponent::Aggressive,
            "self" => Opponent::Neural(trainer),
            "checkpoint" => Opponent::Neural(other.as_ref().unwrap()),
            _ => unreachable!(),
        };
        let report = matchplay::duel(trainer, opponent, hands, seed)?;
        println!(
            "{name:12} | gain {:+.3} bb/100 | IC95 [{:+.3}, {:+.3}] | total {:+.2} BB",
            report.bb_per_100, report.ci95[0], report.ci95[1], report.total_bb
        );
        if let Some(file) = &mut csv {
            writeln!(
                file,
                "{name},{},{},{seed},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9}",
                report.hands,
                report.pairs,
                report.total_bb,
                report.bb_per_100,
                report.ci95[0],
                report.ci95[1],
                report.by_seat[0],
                report.by_seat[1]
            )?;
            file.flush()?;
        }
    }
    Ok(())
}
