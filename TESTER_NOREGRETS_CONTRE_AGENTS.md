# NoRegrets contre les agents existants de rs_poker

Pour choisir **chaque siège d'une table de 2 à 6 joueurs**, utilise maintenant
`table` et le guide `CHOISIR_LES_JOUEURS.md`. La commande `match-agents`
ci-dessous reste une série de duels à deux joueurs.

La première intégration ne proposait qu'un duel SD-CFR contre NoRegrets.
Cette version ajoute **`NoRegretsAgent: rs_poker::arena::Agent`** et utilise
le vrai `HoldemSimulationBuilder`. Les adversaires sont construits par le
`ConfigAgentBuilder` existant ; leurs politiques ne sont pas réécrites.

## Test immédiat

Depuis la racine de LeChiffre, après extraction du ZIP :

```sh
cargo run --release --locked --manifest-path crates/lechiffre-noregrets/Cargo.toml -- match-agents --reference references/noregrets-hu10bb-agent-10000-seed42 --opponent all --hands 1000 --out runs/noregrets-vs-agents
```

Le dossier de sortie doit être nouveau. **Les gains affichés sont ceux de
NoRegrets** en bb/100, avec un IC95 approximatif. Les places sont échangées
dans chaque paire et le flux de cartes est rejoué. Les tapis sont remis à
leur valeur initiale à chaque main ; ce n'est pas un tournoi à élimination.

Pour un seul adversaire, remplace `all` par :

| Option | Vraie implémentation utilisée |
| --- | --- |
| `random` | `RandomAgent` |
| `random_pot_control` | `RandomPotControlAgent`, estimation Monte-Carlo |
| `calling` | `CallingAgent` |
| `all_in` | `AllInAgent` |
| `folding` | `FoldingAgent` |
| `cfr_basic` | `CFRAgent<BasicCFRActionGenerator>` |
| `cfr_simple` | `CFRAgent<SimpleActionGenerator>` |

Les variantes CFR utilisent l'estimateur existant **`uniform`**, pour qu'elles
n'aient pas accès aux vraies cartes adverses. Leur budget par défaut dans
cette commande est léger : fast-forward, deadline 50 ms. Le budget effectif
est enregistré dans `matches.json`. Les agents CFR utilisent un RNG interne :
la graine de cette commande ne suffit pas à garantir leur reproductibilité
bit pour bit. Elle fixe le flux de cartes et les agents aléatoires qui
acceptent une graine.

Les mesures sont écrites dans `matches.csv` et `matches.json`, avec les
configurations, les décisions sur infosets inconnus et les décisions
effectuées après une mise adverse hors du menu abstrait.

## Configuration déjà présente dans le projet

On peut fournir directement un fichier au format du `ConfigAgentBuilder` :

```sh
cargo run --release --locked --manifest-path crates/lechiffre-noregrets/Cargo.toml -- match-agents --reference references/noregrets-hu10bb-agent-10000-seed42 --opponent-config chemin/agent.json --hands 1000 --out runs/noregrets-vs-config
```

Le budget CFR explicite du fichier est conservé. Son estimateur est réglé sur
`uniform` et la configuration effective est enregistrée. Ce chemin permet
notamment d'utiliser les autres variantes CFR déjà supportées par le moteur.

## Utiliser NoRegrets dans une simulation normale

`crates/lechiffre-noregrets/examples/no_regrets_vs_random.rs` contient un
exemple exécutable, avec ces deux agents dans le même vecteur :

```rust
let agents: Vec<Box<dyn Agent>> = vec![
    Box::new(NoRegretsAgent::new(reference, 42)),
    Box::new(RandomAgent::new_with_seed(
        "Random", vec![0.25, 0.30, 0.50], vec![0.50, 0.60, 0.45], 43,
    )),
];
```

Lance l'exemple depuis la racine :

```sh
cargo run --release --locked --manifest-path crates/lechiffre-noregrets/Cargo.toml --example no_regrets_vs_random
```

L'adaptateur fournit automatiquement son `Historian` au simulateur. Celui-ci
ne conserve que les actions publiques ; le bot lit uniquement ses propres
cartes et le tableau. Il reconstruit le vrai état public dans le moteur de
NoRegrets et suit l'historique abstrait avec le `Table` d'origine.

Les mises arbitraires des autres agents restent jouées à leur **vrai montant**
dans rs_poker. L'historique abstrait de NoRegrets utilise la projection amont
sur son menu de mises. C'est le mécanisme hors menu de NoRegrets, dont la
qualité n'est pas une garantie de GTO ; aucune action de l'adversaire n'est
remplacée par une politique inventée.

## Quelle référence utiliser ?

La nouvelle référence `references/noregrets-hu10bb-agent-10000-seed42/` est
entraînée ici depuis zéro : **HU, 10 BB, blindes 1/2, sans rake, relances
libres**, menu Pluribus de NoRegrets, blueprint seul. Elle ne réutilise pas
le modèle à cap d'une relance du test SD-CFR. Les formats sont distincts pour
éviter un mélange involontaire de règles.

La commande `match-agents` ne charge aucun checkpoint SD-CFR. L'ancienne
commande `match --checkpoint ...` reste disponible pour le duel réduit.

Pour entraîner un autre NoRegrets compatible avec la simulation normale :

```sh
cargo run --release --locked --manifest-path crates/lechiffre-noregrets/Cargo.toml -- train-agent --iterations 100000 --stack 20 --seed 43 --out runs/noregrets-agent-plus
```

`--stack` est exprimé en jetons ; avec la grosse blinde 2, `--stack 100`
signifie 50 BB et permet de retrouver les tapis du programme racine actuel.
Le simulateur doit employer deux joueurs, les mêmes tapis et blindes, aucun
rake, et `.max_raises_per_round(None)`. Une configuration incompatible ou un
historique manquant produit une erreur explicite, sans jouer silencieusement
un bot passif. `try_action` est disponible pour gérer ces erreurs soi-même.

L'entraînement peut améliorer la couverture ; la référence incluse sert de
point de départ, sans prétendre au niveau des modèles privés de l'auteur.
La recherche en ligne de NoRegrets n'est pas activée dans cet adaptateur.
