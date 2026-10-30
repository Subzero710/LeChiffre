# Tester ton réseau contre d'autres IA

Installe cette mise à jour en décompressant le ZIP à la racine de LeChiffre. Accepte le remplacement des fichiers du module `crates/lechiffre-sdcfr/` et des documents fournis.

## 1. Avoir une sauvegarde de ton réseau

La commande `train` entraîne le MLP et crée les fichiers `checkpoint-....bin`. La commande `oracle` affiche une référence tabulaire ; pour les matchs de ton réseau, utilise une sauvegarde issue de `train`.

Si tu as déjà une sauvegarde Leduc, utilise directement son chemin à l'étape 2. Sinon, lance depuis la racine de LeChiffre :

```sh
cargo run --release --manifest-path crates/lechiffre-sdcfr/Cargo.toml -- train --game leduc --iterations 128 --out runs/leduc-test
```

Le dossier `runs/leduc-test` doit être nouveau. À la fin, tu obtiens `runs/leduc-test/checkpoint-000128.bin`.

## 2. Jouer contre plusieurs adversaires

```sh
cargo run --release --manifest-path crates/lechiffre-sdcfr/Cargo.toml -- match --game leduc --checkpoint runs/leduc-test/checkpoint-000128.bin --opponent all --hands 100000 --seed 123
```

Remplace le chemin du checkpoint par celui de ta sauvegarde si besoin. Cette commande joue 100 000 mains **par adversaire**, soit 400 000 mains au total avec `all`.

| Option `--opponent` | Comportement |
| --- | --- |
| `random` | Choisit uniformément parmi les actions légales |
| `passive` | Checke ou suit systématiquement |
| `aggressive` | Mise ou relance dès que possible, puis suit quand la limite est atteinte |
| `oracle` | Stratégie moyenne issue du CFR tabulaire, 1 000 itérations par défaut |
| `all` | Teste les quatre adversaires ci-dessus |
| `self` | Joue contre la même stratégie moyenne |
| `checkpoint` | Joue contre une autre sauvegarde neuronale du même jeu |

Les trois premiers profils sont des robots à règles simples. L'oracle est une référence entraînée. La commande utilise **la stratégie moyenne** de ta sauvegarde et laisse ses réseaux inchangés.

Pour choisir un seul adversaire, remplace `all`, par exemple par `random`. Pour un oracle plus entraîné, ajoute `--oracle-iterations 10000`.

## 3. Comprendre une ligne de résultat

Exemple mesuré avec notre sauvegarde Leduc de test à 128 itérations :

```text
random | gain +85.727 | IC95 [+83.640, +87.814] | exact +86.368 | total +85727.0
```

- **gain +85.727** : ton IA a gagné environ 85,7 jetons pour 100 mains simulées.
- **total +85727** : elle a gagné 85 727 jetons sur les 100 000 mains de ce match.
- **IC95** : intervalle d'incertitude approximatif autour du gain moyen, en jetons pour 100 mains. Avec beaucoup de mains, un intervalle entièrement positif indique que le résultat est peu compatible avec un gain moyen nul. S'il traverse zéro, la simulation seule est moins concluante.
- **exact +86.368** : gain moyen théorique contre cet adversaire, calculé en énumérant toutes les possibilités de Kuhn ou Leduc. Il est utile pour séparer la qualité de la stratégie et la chance des mains simulées.

Un gain négatif signifie que ton IA perd contre cet adversaire. L'exploitabilité et le gain contre un profil précis mesurent des choses différentes : un résultat positif contre un robot simple ne suffit pas à établir une stratégie proche de GTO.

Les places sont équilibrées : chaque distribution est rejouée avec les deux agents échangés. Les mêmes tirages de cartes et les mêmes flux aléatoires par siège sont utilisés dans les deux mains d'une paire. L'intervalle est calculé sur les **moyennes des paires**, pour tenir compte de cette dépendance. `--hands` doit être pair et au moins égal à 4 ; pour lire l'intervalle approximatif, préfère un grand nombre de mains, comme 100 000.

Chaque main repart des règles et antes initiales du jeu ; le gain est net des mises. Les réseaux et les robots restent figés pendant les matchs. L'incertitude affichée concerne les mains simulées, tandis que la variabilité entre plusieurs entraînements se mesure séparément.

## 4. Comparer deux réseaux

Par exemple, une sauvegarde de 128 itérations contre celle de 64 itérations :

```sh
cargo run --release --manifest-path crates/lechiffre-sdcfr/Cargo.toml -- match --game leduc --checkpoint runs/leduc-test/checkpoint-000128.bin --opponent checkpoint --opponent-checkpoint runs/leduc-test/checkpoint-000064.bin --hands 100000 --seed 123
```

Le gain affiché est celui du réseau indiqué par `--checkpoint`, ici celui de 128 itérations. Les deux sauvegardes doivent appartenir au même jeu.

Pour enregistrer un tableau de résultats, ajoute un dossier de sortie :

```sh
cargo run --release --manifest-path crates/lechiffre-sdcfr/Cargo.toml -- match --game leduc --checkpoint runs/leduc-test/checkpoint-000128.bin --opponent all --hands 100000 --seed 123 --out runs/matchs-leduc
```

Le fichier créé est `runs/matchs-leduc/matches.csv`. Utilise un nouveau dossier pour chaque test enregistré.

Les mêmes commandes fonctionnent avec `--game kuhn` et une sauvegarde Kuhn. Pour le nouveau Hold'em réduit, utilise le binaire distinct `crates/lechiffre-nlhe/` et les commandes de [LIRE-MOI_NLHE.md](LIRE-MOI_NLHE.md).
