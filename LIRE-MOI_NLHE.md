# LeChiffre — NLHE heads-up réduit

Cette étape ajoute un adaptateur Hold'em autour des `GameState` de **rs_poker**, avec le même MLP Single Deep CFR que le module précédent. Le code est entièrement en Rust. Les dépendances et le classement des mains viennent de la bibliothèque déjà présente dans `vendor/rs-poker/`.

## Installer

Décompresse le ZIP à la racine du projet `LeChiffre/`, dans ta branche `feature/ia_gto`. Accepte le remplacement des fichiers fournis. L'archive ajoute `crates/lechiffre-nlhe/` et met à jour le cœur neuronal ainsi que les documents. Elle ne contient aucun remplacement du moteur `vendor/rs-poker/`, du `Cargo.toml` racine ni du `src/main.rs` racine.

La version a été vérifiée avec le moteur de ta branche au commit `55e83d88f886de97f8240148fc78e852c5d68bf9`. Le dossier `vendor/rs-poker/` doit donc déjà exister dans ton projet. La première compilation peut télécharger les dépendances Rust du moteur. Les vérifications ont utilisé Rust 1.99.1, édition 2024, sous Linux.

## 1. Entraîner le réseau Hold'em

Depuis la racine de LeChiffre :

```sh
cargo run --release --locked --manifest-path crates/lechiffre-nlhe/Cargo.toml -- train --iterations 16 --out runs/nlhe-test
```

Le dossier `runs/nlhe-test` doit être nouveau. Cette commande utilise deux MLP **312 → 64 → 64 → 6**, 8 parcours par joueur et par itération, 64 étapes Adam et un réservoir de 5 000 observations par joueur. Les utilités et les cibles sont exprimées en grosses blindes.

Le programme crée `metrics.csv` et une sauvegarde toutes les 8 itérations, ainsi qu'à la fin. Après cette commande, le fichier final est :

```text
runs/nlhe-test/checkpoint-000016.bin
```

Pour prolonger l'entraînement de 16 itérations :

```sh
cargo run --release --locked --manifest-path crates/lechiffre-nlhe/Cargo.toml -- resume --checkpoint runs/nlhe-test/checkpoint-000016.bin --iterations 16 --out runs/nlhe-suite
```

La configuration et l'état aléatoire sont conservés dans la sauvegarde. La reprise est testée comme identique à un entraînement continu dans le même environnement. Les sauvegardes de Kuhn et Leduc ne sont pas converties en sauvegardes Hold'em : les règles et les dimensions diffèrent.

Pour un nouvel entraînement avec davantage de données par itération, par exemple :

```sh
cargo run --release --locked --manifest-path crates/lechiffre-nlhe/Cargo.toml -- train --iterations 32 --traversals 32 --steps 128 --out runs/nlhe-long
```

Ces paramètres augmentent le travail de calcul ; leur qualité doit être mesurée avec les matchs. Les 16 premières itérations servent surtout à vérifier et démarrer le pipeline.

## 2. Tester contre d'autres IA

```sh
cargo run --release --locked --manifest-path crates/lechiffre-nlhe/Cargo.toml -- match --checkpoint runs/nlhe-test/checkpoint-000016.bin --hands 10000 --out runs/nlhe-matchs
```

Cette commande joue 10 000 mains par adversaire, face à trois profils :

- `random` : actions légales équiprobables ;
- `passive` : checke ou suit ;
- `aggressive` : part à tapis quand cette action est disponible, sinon suit.

Tu peux ajouter `--opponent random`, `--opponent passive`, `--opponent aggressive` ou `--opponent self` pour un seul test. `eval --checkpoint ...` lance aussi l'évaluation contre ces trois profils. Le nombre de mains doit être pair.

Pour comparer deux réseaux Hold'em du même jeu réduit :

```sh
cargo run --release --locked --manifest-path crates/lechiffre-nlhe/Cargo.toml -- match --checkpoint runs/nlhe-suite/checkpoint-000032.bin --opponent checkpoint --opponent-checkpoint runs/nlhe-test/checkpoint-000016.bin --hands 10000
```

Les résultats sont les gains de la première sauvegarde. **`+20 bb/100` signifie un gain moyen de 20 grosses blindes pour 100 mains.** Les cartes sont rejouées avec échange des deux agents ; les flux aléatoires par siège sont aussi réutilisés. L'intervalle à 95 % est approximatif et calculé sur les moyennes des paires de mains. Les agents restent figés pendant l'évaluation.

L'oracle tabulaire et l'exploitabilité exacte sont indisponibles pour ce jeu. Le programme ne lance pas l'énumération utilisée pour Kuhn/Leduc et n'affiche pas de faux score d'exploitabilité nul. Le fichier `matches.csv` contient les gains, leurs intervalles, les nombres de mains et la graine de simulation.

## Règles exactes de cette première réduction

| Élément | Choix |
| --- | --- |
| Joueurs | 2, bouton fixe au siège 0 |
| Tapis initiaux | 20 jetons chacun, soit 10 BB |
| Blindes | petite blinde 1, grosse blinde 2 |
| Cartes | paquet complet de 52 cartes, deux cartes privées par joueur |
| Tours | préflop, flop, turn, river |
| Réduction des mises | une seule mise ou relance volontaire au maximum par street ; les blindes ne comptent pas |
| Rake | aucun |

Les six sorties du MLP correspondent à :

| Sortie | Action |
| --- | --- |
| 0 | Fold, seulement s'il reste un montant à payer |
| 1 | Check ou call, y compris un call à tapis |
| 2 | Mise ou relance minimale |
| 3 | Relance d'un demi-pot après avoir compté le call, arrondie au jeton supérieur et au minimum légal |
| 4 | Relance d'un pot après avoir compté le call |
| 5 | Tapis |

Les actions illégales et les doublons de montants sont masqués. Un tapis qui sert uniquement à payer reste représenté par la sortie check/call. Une fois la limite de mises atteinte, le joueur peut encore payer ou se coucher. Après un tapis payé, toutes les cartes manquantes sont distribuées jusqu'au showdown, avec retrait des cartes déjà connues.

`rs_poker` reçoit des montants de **contribution totale sur la street**, conformément à son API `Bet`. Il valide les mises, tient les comptes et classe les mains. L'adaptateur gère le parcours clonable, les nœuds de hasard, l'historique public et le règlement terminal. Les tapis non payés sont rendus au showdown si nécessaire ; les égalités partagent le pot. Aucun rollout arbitraire remplaçant les décisions futures n'est utilisé.

## Entrées du réseau et stratégie moyenne

Les 312 entrées comprennent les deux cartes privées du joueur (52 valeurs), le flop/turn/river (3 × 52), la street et la position (6), huit informations monétaires publiques, le masque de six actions et douze emplacements d'historique public (action et montant).

L'historique est complet dans cette réduction : il existe au maximum trois décisions par street. Une violation de cette borne provoque une erreur ; aucune décision n'est tronquée silencieusement. Les cartes adverses, le paquet restant et les futures cartes du tableau n'entrent pas dans le MLP.

Le self-play utilise le parcours external-sampling MCCFR du cœur neuronal : hasard et adversaire échantillonnés, toutes les actions légales du joueur entraîné explorées jusqu'aux gains terminaux, puis régression sur les avantages signés. Les deux politiques restent figées pendant la collecte d'une itération. Le Regret Matching convertit les sorties en probabilités.

Pour jouer la stratégie moyenne SD-CFR, un ancien réseau est tiré avec un poids proportionnel à son numéro d'itération **une fois par joueur et par main**, puis conservé jusqu'à la fin. Un nouveau réseau n'est pas tiré à chaque décision. Les archives ne sont pas remplacées par une moyenne de leurs poids.

## Lire les mesures et les limites

La MSE mesure l'ajustement aux cibles dynamiques et bruitées de MCCFR ; elle peut augmenter pendant que les adversaires changent. Les gains contre des profils précis ne prouvent pas la proximité d'un équilibre. Consulte `crates/lechiffre-nlhe/validation/VALIDATION.md` pour les résultats réels, y compris les défaites restantes.

Ce module entraîne et évalue uniquement le jeu réduit ci-dessus. Les piles différentes, les relances supplémentaires et les mises hors menu nécessiteront une extension de l'adaptateur, de l'encodeur et de l'identifiant de jeu. La performance au Hold'em sans ces restrictions n'est pas établie. Le binaire historique à la racine du projet n'est pas remplacé : utilise les commandes `crates/lechiffre-nlhe/` pour cette étape.

L'archive de réseaux est conservée intégralement. Le format de checkpoint du cœur de référence reste limité à 64 Mio ; les longs entraînements et les réseaux plus grands demanderont d'adapter le stockage. Le cache d'inférence est maintenant borné à 16 384 observations par itération. Le GPU, le multiway et le Cerveau Psychologue restent des étapes suivantes.

## Vérifier les tests

```sh
cargo test --locked --manifest-path crates/lechiffre-nlhe/Cargo.toml
cargo test --locked --manifest-path crates/lechiffre-sdcfr/Cargo.toml
```

Les tests Hold'em comparent les règles à des fixtures issues d'OpenSpiel et au simulateur complet de `rs_poker`. Les fixtures sont statiques : aucun Python ni OpenSpiel n'est requis pour utiliser le projet. Les tests précédents de Kuhn/Leduc sont conservés.
