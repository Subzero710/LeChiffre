# LeChiffre — Single Deep CFR sur Kuhn et Leduc

Cette mise à jour ajoute **Leduc** au réseau neuronal déjà validé sur Kuhn. Le programme reste entièrement en Rust et fonctionne sur CPU, sans Python ni dépendance externe. Il entraîne un MLP avec des données de self-play et MCCFR.

La commande `match` permet désormais de tester une sauvegarde contre des robots, l'oracle tabulaire ou un autre réseau. Consulte [TESTER_CONTRE_IA.md](TESTER_CONTRE_IA.md) pour les commandes et la lecture des gains.

## Installer la mise à jour

Décompresse le ZIP à la racine de `LeChiffre/`, là où se trouve ton `Cargo.toml` principal. Accepte le remplacement des fichiers du module `crates/lechiffre-sdcfr/` et des deux documents de cette archive. Tes fichiers du moteur et le `src/main.rs` principal ne sont pas remplacés.

Dans un terminal ouvert à cette racine, lance Leduc :

```sh
cargo run --release --manifest-path crates/lechiffre-sdcfr/Cargo.toml -- train --game leduc
```

Cela démarre un nouvel entraînement de 128 itérations sur Leduc. Tu peux commencer par 64 :

```sh
cargo run --release --manifest-path crates/lechiffre-sdcfr/Cargo.toml -- train --game leduc --iterations 64 --out runs/leduc-42
```

Le dossier de sortie doit être nouveau. Le programme y écrit les mesures dans `metrics.csv`, la stratégie moyenne dans `average-policy.csv` et une sauvegarde toutes les 16 itérations ainsi qu'à la fin.

Pour évaluer puis prolonger cet exemple :

```sh
cargo run --release --manifest-path crates/lechiffre-sdcfr/Cargo.toml -- eval --game leduc --checkpoint runs/leduc-42/checkpoint-000064.bin
cargo run --release --manifest-path crates/lechiffre-sdcfr/Cargo.toml -- resume --game leduc --checkpoint runs/leduc-42/checkpoint-000064.bin --iterations 64 --out runs/leduc-42-suite
```

La reprise ajoute 64 itérations aux 64 déjà réalisées. Les sauvegardes Kuhn restent utilisables avec `--game kuhn`. Leduc possède ses propres sauvegardes : ses entrées et ses actions sont différentes de celles de Kuhn.

Les anciennes commandes sans `--game` continuent de sélectionner Kuhn. Pour les tests :

```sh
cargo test --locked --manifest-path crates/lechiffre-sdcfr/Cargo.toml
```

Pour une référence tabulaire indépendante :

```sh
cargo run --release --manifest-path crates/lechiffre-sdcfr/Cargo.toml -- oracle --game leduc --iterations 1000
```

Cette référence n'entre pas dans l'entraînement neuronal. Le réseau apprend à partir de ses propres parcours de self-play.

## Ce que change Leduc

Il y a six cartes physiques : deux J, deux Q et deux K. Chaque joueur reçoit une carte privée. Après le premier tour de mises, une carte publique est révélée, puis un second tour commence. Une paire avec la carte publique bat une carte haute ; deux mains identiques partagent le pot.

La variante implémentée est à deux joueurs, avec une ante de 1, une ouverture et au maximum une relance par tour. L'incrément de mise est de 2 au premier tour et de 4 au second. Le joueur 0 parle en premier aux deux tours. Les couleurs équivalentes sont regroupées dans l'observation ; les distributions et le retrait des cartes utilisent bien les six cartes physiques.

Le MLP Leduc a **43 entrées → 64 → 64 → 3 sorties**, soit 7 171 paramètres par joueur par défaut.

| Entrées | Nombre |
| --- | ---: |
| Rang de la carte privée | 3 |
| Rang de la carte publique ou absence de carte | 4 |
| Joueur et tour de mises | 4 |
| Historique public complet, dans l'ordre | 24 |
| Masque des actions légales | 3 |
| Pot, contributions, montant à payer et compteur de mises | 5 |

Les trois sorties estiment les avantages des actions : **0 = fold, 1 = check/call, 2 = bet/raise**. Le Regret Matching les transforme en probabilités. Les cartes privées adverses ne figurent jamais dans l'entrée du réseau.

Les deux réseaux restent figés pendant la collecte des parcours d'une itération, puis sont entraînés sur les réservoirs historiques. La perte est une MSE masquée et pondérée par le numéro d'itération. Adam met à jour les poids après réinitialisation du réseau, avec limitation de la norme du gradient à 1.

Jusqu'à 512 observations distinctes, le programme regroupe les exemples strictement identiques pour calculer le gradient complet de la même MSE plus efficacement, à l'arrondi près. Ce regroupement ne fusionne aucune observation différente. Les cibles brutes sont conservées dans les réservoirs. Au-delà, le programme utilise des mini-batches uniformes. Les probabilités d'actions sont aussi mises en cache pendant que les réseaux sont figés ; cette cache est vidée à chaque itération.

La stratégie moyenne utilise tous les réseaux archivés, pondérés par l'itération et par la probabilité d'atteindre la situation par les propres décisions du joueur. Les poids des réseaux ne sont pas moyennés.

## Comprendre les résultats

**Regarde surtout l'exploitabilité : plus elle se rapproche de zéro, mieux c'est.** La stratégie uniforme commence à environ **2,373611 sur Leduc**. La valeur initiale diffère de celle de Kuhn ; il faut suivre la nouvelle courbe Leduc.

| Mesure | Signification |
| --- | --- |
| `iteration` | Nombre de cycles d'apprentissage, chacun contenant plusieurs parcours |
| `exploitabilite` | Faiblesses exploitables par une meilleure réponse respectant les cartes cachées |
| `valeur J0` | Gain moyen du premier joueur contre la stratégie apprise de l'autre joueur |
| `regret` | Regret externe moyen pondéré, mesuré pour chacun des joueurs sur la séquence de stratégies |
| `MSE` | Erreur du réseau sur des cibles du réservoir d'entraînement, pour les deux joueurs |

La MSE contient le bruit des cibles MCCFR et peut rester élevée. Elle n'est pas une mesure de performance au poker ni une erreur sur un jeu de validation indépendant. Des remontées temporaires de l'exploitabilité sont possibles. La somme des deux regrets externes vaut NashConv, et **exploitabilité = NashConv / 2**, en jetons d'ante par main.

La valeur de référence −1/18 concernait Kuhn. Ne l'utilise pas comme objectif pour Leduc. Consulte `crates/lechiffre-sdcfr/validation/LEDUC_VALIDATION.md` pour les résultats réellement obtenus avec cette livraison.

## Organisation du code

| Fichier dans `crates/lechiffre-sdcfr/src/` | Rôle |
| --- | --- |
| `leduc.rs` | Règles, distribution, observation et gains Leduc |
| `game.rs` | Interfaces des jeux et règles de Kuhn |
| `exact.rs` | Arbre exact, stratégie moyenne, meilleures réponses, exploitabilité et oracle tabulaire |
| `training.rs` | Self-play MCCFR et entraînement SD-CFR commun aux deux jeux |
| `neural.rs` | MLP, rétropropagation et Adam |
| `memory.rs` | Réservoirs et préparation des données |
| `strategy.rs` | Regret Matching et choix d'un ancien réseau pour une main |
| `checkpoint.rs` | Sauvegarde et reprise |
| `evaluation.rs`, `tabular.rs` | Références indépendantes de Kuhn conservées pour vérifier le nouvel évaluateur |
| `main.rs` | Commandes `train`, `resume`, `eval` et `oracle` |
| `arena.rs` | Matchs appariés, profils adverses et intervalles d'incertitude ; commande `match` |

L'évaluateur exact regroupe les états cachés dans les mêmes ensembles d'information et choisit une seule action de meilleure réponse par ensemble. Il vérifie la conservation du souvenir des décisions propres du joueur. Il est destiné aux petits jeux : l'énumération complète ne convient pas au NLHE.

## Validation et suite

Les tests comparent tous les 9 450 états situés après la distribution privée à une référence produite par OpenSpiel 2.0.2 : acteurs, cartes restantes, actions légales et gains. Cette référence statique est incluse ; les tests et l'entraînement n'ont besoin ni d'OpenSpiel ni de Python. L'évaluateur générique est également comparé à l'évaluation exhaustive indépendante de Kuhn.

Les sauvegardes contiennent les réseaux, toute leur archive, les réservoirs et l'état aléatoire. La reprise est vérifiée comme identique à une exécution continue dans le même environnement. La limite de cette implémentation de référence reste de 64 Mio par sauvegarde ; l'archive grandit avec les itérations. Aucun ancien réseau n'est supprimé silencieusement.

La prochaine étape est l'adaptateur `rs_poker`, puis un premier NLHE heads-up réduit. Le GPU, la généralisation au NLHE, le multiway et le Cerveau Psychologue restent à développer. Le programme principal existant de LeChiffre n'utilise pas encore ce réseau. Le backend neuronal reste une référence CPU Rust ; l'intégration de Burn viendra avec l'étude des performances sur des jeux plus grands.

## Références

- [Single Deep CFR](https://arxiv.org/abs/1901.07621)
- [Deep CFR](https://proceedings.mlr.press/v97/brown19b.html)
- [Règles de Leduc dans OpenSpiel](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/games/leduc_poker/leduc_poker.h)
- [Exploitabilité dans OpenSpiel](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/algorithms/exploitability.py)
