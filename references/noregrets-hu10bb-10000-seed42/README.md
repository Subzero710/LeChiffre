# Référence NoRegrets entraînée pour LeChiffre

Cette référence a été entraînée ici, depuis zéro, avec le code NoRegrets
public adapté au jeu réduit. Ce n'est pas un modèle préentraîné de l'auteur.

- 10 000 parcours du MCCFR tabulaire NoRegrets, graine 42, un thread.
- Deux joueurs, 20 jetons, blindes 1/2 (10 BB), cap d'une mise/relance.
- Même menu de mises que le SD-CFR LeChiffre ; aucun pruning ni recherche.
- 12 buckets postflop, 64 simulations d'équité, sans centroids.
- 12 818 infosets stockés. Ce nombre n'est pas une mesure d'exploitabilité.

Le fichier `reference.json` contient la provenance, l'identifiant des règles
et le SHA-256 de `blueprint.bin`. Conserve les deux fichiers ensemble.

Commande de génération, depuis la racine de LeChiffre :

```sh
cargo run --release --locked --manifest-path crates/lechiffre-noregrets/Cargo.toml -- train --iterations 10000 --seed 42 --threads 1 --out runs/noregrets-10bb
```

La stratégie obtenue avec un thread et les mêmes versions est reproductible
dans cet environnement ; l'ordre de sérialisation des tables peut différer.
Ne pas exiger que les fichiers `.bin` de deux entraînements aient le même hash.
Le hash fourni contrôle l'intégrité de l'artefact distribué.
