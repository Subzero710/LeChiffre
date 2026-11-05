# LeChiffre : NoRegrets comme référence réelle

Pour une **table à six joueurs et le choix de chaque siège**, voir
`CHOISIR_LES_JOUEURS.md` et `configs/table-6.json`. Le modèle six joueurs
fourni est distinct du modèle HU ; le réseau SD-CFR reste à deux joueurs.

Pour **NoRegrets contre les agents déjà présents dans rs_poker**, utilise
maintenant `match-agents` et le guide `TESTER_NOREGRETS_CONTRE_AGENTS.md`.
Le nouvel adaptateur implémente la véritable interface `Agent`. Les commandes
`train` / `match` détaillées ci-dessous conservent le duel réduit contre SD-CFR.

Ton IA précédente est une implémentation indépendante de **Single Deep CFR**.
Elle partage avec NoRegrets la minimisation du regret, mais ne dérive pas de
son code. Cette livraison ajoute son vrai code public à côté du SD-CFR, avec
un raccordement pour comparer les deux approches sur le même jeu réduit.
Elle ne transforme pas tes anciennes sauvegardes en modèles NoRegrets.

Source : https://github.com/conorarmstrong/noregrets

Version intégrée : `757f7692738069522195d2b486eec60a8b010c0c`, sous licence MIT.
Les stratégies préentraînées de l'auteur ne sont pas publiques : **il faut
entraîner la référence**. Une petite référence entraînée ici depuis zéro est
incluse pour tester immédiatement ; son niveau dépend du budget choisi.

## Ce que font les fichiers de NoRegrets

Tous ces chemins sont dans `vendor/noregrets/`.

| Fichier | Fonction |
| --- | --- |
| `Cargo.toml`, `Cargo.lock` | Dépendances, compilation et versions Rust fixées |
| `src/main.rs` | Commandes pour entraîner, jouer, évaluer et lancer les expériences |
| `src/cards.rs` | Cartes encodées par rang et couleur |
| `src/eval.rs` | Classe les mains de poker au showdown |
| `src/engine.rs` | Règles NLHE, tours de parole, mises, tapis et pots secondaires |
| `src/abstraction.rs` | Réduit les choix de mises et regroupe les mains en buckets |
| `src/cfr.rs` | Self-play external-sampling Linear MCCFR, regrets tabulaires, stratégie moyenne, sauvegardes et options de pruning |
| `src/bot.rs` | Joue le blueprint et dirige les recherches supplémentaires pendant une main |
| `src/search.rs` | Suit les distributions de mains adverses à partir de leurs actions |
| `src/river.rs` | Résout le sous-jeu river sur les distributions de mains |
| `src/turn.rs` | Résout le sous-jeu turn et les rivers possibles |
| `src/flop.rs` | Recherche au flop, avec estimation des valeurs à la limite de profondeur |
| `src/net.rs` | Petit MLP ReLU, perte MSE pondérée, Adam et sauvegardes |
| `src/valuenet.rs` | Entraîne et utilise le réseau de valeurs des états de croyance ; ce réseau ne prédit pas les regrets de Deep CFR |
| `src/table.rs` | Table de jeu, historique abstrait, raccordement des actions et matchs |
| `src/aivat.rs` | Estimation des gains avec réduction de variance |
| `src/lbr.rs` | Adversaire de réponse locale pour chercher des faiblesses exploitables |
| `src/br.rs` | Sonde plus poussée, avec meilleures réponses exactes dans les sous-jeux turn/river ; pas dans tout le NLHE |
| `src/benchmark.rs` | Compare les décisions avec des historiques publiés de Pluribus ; accord des décisions et gain en match sont des mesures différentes |
| `src/play.rs` | Permet à un humain de jouer dans le terminal |
| `src/slumbot.rs` | Adaptateur réseau pour jouer contre Slumbot ; aucune connexion n'est lancée par notre comparaison locale |
| `src/clone.rs` | Construit un modèle tabulaire du comportement de Slumbot depuis des historiques |
| `src/distill.rs` | Réinjecte les décisions améliorées par la recherche dans une stratégie de base |
| `src/portfolio.rs` | Choisit entre plusieurs stratégies à l'aide d'un bandit |
| `src/ablate.rs` | Expériences comparant différentes méthodes de résolution sous des croyances perturbées |
| `README.md` | Utilisation, architecture, limites et références de l'auteur |
| `ROADMAP.md` | Évolutions prévues par l'auteur |
| `BASELINES.md` | Mesures historiques de l'auteur, liées à des entraînements précis |
| `LICENSE` | Autorisations et conditions de la licence MIT |
| `legacy/README_python.md`, `legacy/requirements.txt` | Documentation et dépendance de l'ancienne version Python |
| `legacy/poker_bot_pluribus.py`, `poker_bot_utils.py`, `poker_game_engine.py` | Ancien bot, outils de cartes et ancien moteur Python, retirés au profit de Rust |
| `legacy/train.py`, `play.py` | Anciennes commandes d'entraînement et de jeu |
| `legacy/minimal_test.py`, `quick_test.py`, `test_installation.py`, `test_play.py`, `log.txt` | Anciens tests et journal de cette version Python |

`src/lib.rs`, `LECHIFFRE_UPSTREAM.md` et `UPSTREAM_SHA256.json` ont été ajoutés
pour l'intégration ; leurs changements sont explicitement documentés.

## Différence avec ton réseau

| LeChiffre SD-CFR | NoRegrets public |
| --- | --- |
| Un MLP approxime les regrets des actions | Le blueprint conserve les regrets dans des tables abstraites |
| Stratégie moyenne représentée par une archive de réseaux | Stratégie moyenne enregistrée dans le blueprint |
| Pas de recherche supplémentaire actuellement | Recherche en ligne et solveurs de sous-jeux disponibles dans le programme d'origine |
| Pas de réseau de valeurs séparé actuellement | Réseau de valeurs optionnel pour certaines recherches |

## Installation

Décompresse l'archive à la racine de LeChiffre en conservant les dossiers.
Les ajouts utiles sont `vendor/noregrets/`, `crates/lechiffre-noregrets/` et
ce guide. Le cœur SD-CFR et le module NLHE livrés précédemment sont conservés.
Le moteur déjà installé dans `vendor/rs-poker/` doit rester présent.
Le `Cargo.toml` racine et le `src/main.rs` racine ne sont pas remplacés.

Les commandes suivantes sont exécutées depuis la racine du projet. Aucun
Python, Docker, compte Slumbot ou modèle commercial n'est nécessaire.

## 1. Tester immédiatement avec la référence incluse

```sh
cargo run --release --locked --manifest-path crates/lechiffre-noregrets/Cargo.toml -- match --checkpoint runs/nlhe-suite-64/checkpoint-000064.bin --reference references/noregrets-hu10bb-10000-seed42 --hands 10000 --seed 123 --out runs/comparaison-noregrets-64
```

Cette référence a été entraînée ici sur 10 000 parcours, graine 42, un thread.
Ce n'est pas l'un des modèles commerciaux de l'auteur. Son fichier de
provenance et son SHA-256 sont inclus dans le même dossier.

## 2. Entraîner ta propre référence NoRegrets

```sh
cargo run --release --locked --manifest-path crates/lechiffre-noregrets/Cargo.toml -- train --iterations 10000 --seed 42 --threads 1 --out runs/noregrets-10bb
```

Le dossier de sortie doit être nouveau. Il contient `blueprint.bin` et
`reference.json` : conserve les deux ensemble. Les règles, le commit source,
le budget et un SHA-256 empêchant les mélanges de fichiers y sont enregistrés.

Cette commande entraîne le vrai MCCFR de NoRegrets avec 12 buckets postflop
et 64 simulations d'équité pour les buckets flop/turn sans centroids. La river
utilise le calcul d'équité exact du code amont. C'est un budget de démarrage,
pas une certification de force. Les infosets encore inconnus déclenchent le
repli check/call d'origine, qui est compté pendant les matchs.

Pour utiliser aussi les regroupements EMD k-medians de NoRegrets, ajoute par
exemple `--centroid-samples 1000`. Leur coût de construction augmente le temps
total ; le choix doit être conservé dans les comparaisons. Le défaut sans
centroids est un mode réellement supporté par le code amont.

Pour une deuxième référence, relance `train` avec un autre budget, une autre
graine et un nouveau dossier. Ce wrapper ne propose pas encore de commande
`resume` ; il n'accepte pas les checkpoints SD-CFR comme blueprints.

`--threads 1` rend les mises à jour séquentielles. Plusieurs threads augmentent
le débit mais l'ordre de leurs mises à jour peut varier ; ne pas exiger des
résultats identiques bit pour bit dans ce cas.

## 3. Comparer ta nouvelle référence avec ton checkpoint existant

```sh
cargo run --release --locked --manifest-path crates/lechiffre-noregrets/Cargo.toml -- match --checkpoint runs/nlhe-suite-64/checkpoint-000064.bin --reference runs/noregrets-10bb --hands 10000 --seed 123 --out runs/comparaison-noregrets-64
```

Les gains affichés sont ceux de **ton SD-CFR** : positif signifie qu'il gagne
contre cette référence NoRegrets, négatif qu'il perd. Les deux agents sont
figés. Même distribution rejouée avec échange des places, réseau SD-CFR tiré
une seule fois par joueur et par main, intervalles calculés sur les paires.

La sortie contient `matches.csv` et `comparison.json`, ainsi que le nombre de
replis de NoRegrets par street. Si beaucoup de décisions utilisent ce repli,
le match évalue surtout une référence encore peu couverte par l'entraînement.
Augmenter les mains de test resserre l'incertitude ; cela n'entraîne aucun bot.

## Ce que le match compare exactement

- Deux joueurs, 20 jetons chacun, blindes 1/2 : **10 BB**.
- Même menu des six sorties et même cap d'une mise/relance par street.
- Vrai blueprint NoRegrets entraîné sur ces règles, sans pruning ni recherche
  en ligne. Les seuils de pruning du grand jeu ne sont pas transposés au hasard.
- Même moteur rs_poker que précédemment pour le SD-CFR ; NoRegrets rejoue les
  cartes et les actions en parallèle avec contrôle des contributions et gains.
- Conversion exacte entre les codages de cartes ; aucune projection de mises
  vers la « taille la plus proche » et aucun changement des anciens checkpoints.

Ce n'est **pas** un match contre la configuration complète NoRegrets 6-max,
son réseau de valeurs, sa recherche en ligne, ses modèles commerciaux, ou
le Pluribus original. Aucun de leurs résultats publiés ne doit être repris
comme résultat de nos entraînements.

Un gain positif face à un blueprint peu entraîné ne prouve pas un niveau GTO.
Comparer plusieurs budgets et plusieurs graines, suivre la couverture, puis
utiliser des graines de test séparées. Les nombres d'itérations des deux
algorithmes ne représentent pas la même quantité de calcul : relever aussi
le temps et les données générées. L'exploitabilité exacte NLHE n'est pas calculée.

## Vérifier l'intégration

```sh
cargo test --locked --manifest-path crates/lechiffre-noregrets/Cargo.toml
cargo test --locked --manifest-path crates/lechiffre-nlhe/Cargo.toml
cargo test --locked --manifest-path crates/lechiffre-sdcfr/Cargo.toml
```

La validation détaillée figure dans
`crates/lechiffre-noregrets/validation/VALIDATION.md`.
