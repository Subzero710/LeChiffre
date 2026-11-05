# Référence NoRegrets pour six joueurs

Entraînée ici depuis zéro avec le NoRegrets public MIT fixé au commit
`757f7692738069522195d2b486eec60a8b010c0c`.

- 6 joueurs, tapis20 jetons chacun, blindes1/2 : 10 BB.
- Relances libres, sans rake, menu abstrait Pluribus.
- 10 000 parcours, graine42, un thread, sans pruning ni recherche en ligne.
- 12 buckets postflop, 64 simulations d'équité, sans centroids.

Il s'agit d'une référence de démarrage peu entraînée ; aucun niveau de force
ou équilibre de Nash n'est certifié. Le modèle n'est pas fourni par l'auteur.
Garde `blueprint.bin` et `agent-reference.json` ensemble.

Génération :

```sh
cargo run --release --locked --manifest-path crates/lechiffre-noregrets/Cargo.toml -- train-agent --players 6 --iterations 10000 --stack 20 --seed 42 --out nouveau-dossier
```
