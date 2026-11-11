# Nouveau : profils JSON, versions NoRegrets et interface graphique

**Positions relatives équilibrées :** utiliser désormais `lechiffre-arena compare`,
expliqué pas à pas dans `MODE_D_EMPLOI_NOREGRET.md`, avec les commandes pour
entraîner un modèle et le sélectionner. Ce mode appelle le comparateur natif
rs_poker : six profils à six places avec `-n 10` donnent 720 × 10 = 7 200 mains.
Il permute les joueurs entre les sièges. Les anciens modes `table` ci-dessous
gardent les sièges fixes et font tourner le bouton.

**Correction du 11 novembre 2026 :** le module NoRegrets est autonome. Il ne
requiert plus `crates/lechiffre-nlhe` ni `crates/lechiffre-sdcfr`. Si Cargo
réclame encore l'un de leurs `Cargo.toml`, extraire la version corrigée de
`NoRegret.zip` à la racine du projet, avec les nouveaux fichiers du module et
son `Cargo.lock`. Les commandes de ce guide fonctionnent avec cette version.
Les anciens outils du duel SD-CFR réduit (`train` et `match`) ont été retirés ;
pour entraîner un agent de table, utiliser `train-agent`.

L'archive s'appelle désormais `NoRegret.zip`. Pour sélectionner NoRegrets comme
les autres agents et comparer plusieurs blueprints dans la TUI rs_poker, suivre
`MODE_D_EMPLOI_NOREGRET.md`. Le mode `table` décrit ci-dessous reste disponible.

# Choisir chaque joueur d'une table de 2 à 6

Les résultats précédents marqués **HU** concernent deux joueurs. HU signifie
heads-up. Le NoRegrets public supporte jusqu'à six joueurs, mais l'ancien
adaptateur et la référence fournie étaient limités à deux.

La nouvelle commande `table` utilise le simulateur rs_poker avec 2 à 6 sièges.
Elle conserve la place de chaque agent et fait tourner le bouton pour
équilibrer les positions. Les tapis sont remis à leur valeur initiale après
chaque main. Une référence NoRegrets entraînée pour six joueurs est incluse.

## Six joueurs : choisir les six sièges

Depuis la racine de LeChiffre :

```sh
cargo run --release --locked --manifest-path crates/lechiffre-noregrets/Cargo.toml -- table --table-config configs/table-6.json --hands 6000 --out runs/table-6
```

Modifie `configs/table-6.json` pour choisir chaque joueur :

```json
{
  "stack": 20,
  "players": [
    {"type": "noregrets", "reference": "references/noregrets-6max-10bb-agent-10000-seed42"},
    {"type": "random"},
    {"type": "calling"},
    {"type": "all_in"},
    {"type": "cfr_basic", "hand_estimator": "uniform"},
    {"type": "random_pot_control", "percent_call": [0.5, 0.3]}
  ]
}
```

L'ordre de la liste correspond aux sièges 1 à 6. Tu peux répéter un type,
mettre plusieurs NoRegrets, changer leur référence individuellement, ou
supprimer des lignes pour jouer à moins de joueurs.

Les agents existants restent configurés par leur format rs_poker habituel.
Pour NoRegrets, le chemin du modèle est dans `reference`. Les références des
différents sièges doivent avoir été entraînées pour le nombre de joueurs de
la table, avec les mêmes tapis et les mêmes blindes.

## Variante courte en ligne de commande

```sh
cargo run --release --locked --manifest-path crates/lechiffre-noregrets/Cargo.toml -- table --players "noregrets,random,calling,all_in,cfr_basic,random_pot_control" --reference references/noregrets-6max-10bb-agent-10000-seed42 --hands 6000 --out runs/table-6-autre
```

Chaque mot définit un siège. Types disponibles directement : `noregrets`,
`random`, `calling`, `all_in`, `folding`, `random_pot_control`, `cfr_basic`,
`cfr_simple`. Les autres variantes et paramètres déjà présents dans
`ConfigAgentBuilder` restent accessibles par le fichier JSON.

Sans NoRegrets, `--reference` est facultatif. Avec NoRegrets, le nombre de
sièges doit correspondre au modèle chargé. Pour jouer à deux, utilise par
exemple la référence `references/noregrets-hu10bb-agent-10000-seed42`.

Le nombre de mains doit être un multiple du nombre de joueurs, avec au moins
deux tours complets de bouton : à six joueurs, 600, 6 000 ou 60 000 mains
conviennent. Ces mains sont le total de la table, pas un duel séparé contre
chaque adversaire. Tous les agents participent au même jeu.

## Lire les résultats

Chaque siège possède son propre gain en bb/100 et son IC95 approximatif.
Les intervalles sont calculés sur les cycles complets du bouton. Les flux de
cartes sont indépendants entre les mains ; ce mode n'effectue pas le rejeu
apparié des duels HU. Les gains cumulés des sièges somment à zéro sans rake.
Le programme écrit `table.csv` et `table.json`, y compris les configurations
effectives et les compteurs NoRegrets de chaque siège.

Les vrais agents CFR utilisent l'estimateur `uniform` et un budget léger par
défaut. Leur budget explicite dans le JSON est conservé. Leurs tirages internes
ne sont pas contrôlés par la seule graine du simulateur.

## Pourquoi NoRegrets ne gagne pas toujours

Le nom du projet ne garantit pas qu'il est le meilleur dans cette configuration.
Les références fournies ici ont seulement 10 000 parcours d'entraînement et
jouent le **blueprint seul**. La recherche en ligne et les modèles privés de
l'auteur ne sont pas activés ou fournis. De plus, le hasard du poker peut
faire perdre un joueur fort sur un échantillon limité de mains.

Les gains contre un adversaire précis ne prouvent pas une proximité de GTO.
À trois joueurs et plus, les garanties de convergence à un équilibre des
jeux à deux joueurs ne se transfèrent pas automatiquement. Cette intégration
mesure la performance empirique ; elle ne certifie pas un équilibre à six.

## Entraîner une référence pour la bonne table

```sh
cargo run --release --locked --manifest-path crates/lechiffre-noregrets/Cargo.toml -- train-agent --players 6 --iterations 100000 --stack 20 --seed 43 --out runs/noregrets-6-plus
```

`--players` accepte 2 à 6. La grosse blinde vaut 2 jetons ; `--stack 20` vaut
10 BB. Dans le JSON, change les chemins `reference` pour charger cette
nouvelle stratégie. Modifier seulement le nombre de joueurs dans les
métadonnées d'une ancienne sauvegarde ne convertit pas son apprentissage.
Les anciennes références HU restent compatibles avec les tables à deux.
