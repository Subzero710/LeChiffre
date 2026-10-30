# Validation NLHE réduit — 29 octobre 2026

Environnement : Linux x86_64, Rust 1.99.1. Moteur `vendor/rs-poker/` lu sur la branche `feature/ia_gto`, commit `55e83d88f886de97f8240148fc78e852c5d68bf9`. Aucun fichier du moteur n'est modifié par cette livraison.

## Règles et information

Le jeu possède deux joueurs, 52 cartes physiques, des tapis initiaux égaux de 20 jetons, des blindes de 1/2, aucun rake et quatre streets. Le bouton est au siège 0, qui parle en premier préflop ; le siège 1 parle en premier après le flop. Il existe au maximum une mise ou relance volontaire par street, avec six sorties d'action masquées et dédupliquées. Cette restriction fait partie de l'identifiant de jeu ; ce n'est pas une certification de stratégie pour le NLHE libre.

Neuf tests automatisés supplémentaires passent :

- blindes, montant total des mises sur la street, minimum de relance et déduplication des tailles ;
- folds, tapis payés, runout complet, égalités et gains nets à somme nulle ;
- ordre des joueurs sur les quatre streets, retrait des cartes et conservation des 40 jetons ;
- invariance des observations aux cartes adverses et aux futures cartes du tableau ;
- conservation de l'historique ordonné quand les pots et les joueurs actifs sont identiques ;
- comparaison indépendante avec OpenSpiel ;
- comparaison avec le simulateur complet de `rs_poker` ;
- reprise de checkpoint identique à un entraînement continu et rejet des identifiants de jeu incompatibles ;
- matchs appariés reproductibles, annulation exacte contre soi-même et validation des entrées.

Les 30 tests du cœur Kuhn/Leduc passent également, soit **39 tests au total** sur les deux crates. `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings` et la construction release sont vérifiés.

## Références indépendantes

`tests/fixtures/openspiel_nlhe.tsv` contient 400 distributions et 1 147 décisions issues d'OpenSpiel 2.0.2, avec le jeu `universal_poker`, abstraction `fullgame`, deux joueurs, quatre rounds, `blind="1 2"`, `stack="20 20"`, `firstPlayer="1 2 2 2"`, quatre couleurs, treize valeurs, deux cartes privées et `numBoardCards="0 3 1 1"`.

Les cartes ont été traduites entre les deux codages physiques. Pour chaque décision, les tests comparent le joueur actif, les contributions et les tapis, puis vérifient que chaque mise proposée par l'adaptateur est légale dans le NLHE complet d'OpenSpiel. Les décisions supplémentaires interdites par notre réduction ne sont pas requises dans le masque réduit. Les gains terminaux sont comparés en BB, y compris après les tapis et les runouts.

Un second test utilise `HoldemSimulationBuilder` et les agents fournis par `rs_poker`, sur 24 mains : calling/calling, all-in/calling et folding/calling. L'adaptateur rejoue les cartes observées et retrouve exactement les mêmes tapis terminaux. Les dépendances de validation OpenSpiel/Python ont uniquement servi à produire la fixture ; elles ne sont pas nécessaires pour compiler ou exécuter les tests livrés.

## Entraînements réellement exécutés

Deux entraînements de 16 itérations ont été réalisés avec les graines 42 et 43, chacun avec 8 parcours par joueur et par itération, deux MLP 312 → 64 → 64 → 6, 64 étapes Adam par mise à jour, un batch de 64, un taux de 0,001 et un réservoir de 5 000 observations par joueur. Le réseau est réinitialisé à chaque nouvel ajustement, comme dans le cœur de référence. Les fichiers `train-seed-42.csv` et `train-seed-43.csv` conservent les mesures.

La MSE finale sur les cibles bruitées vaut environ 97,33/74,21 pour la graine 42 et 95,99/65,37 pour la graine 43. Elle n'est pas interprétée comme une mesure de proximité de Nash et n'est pas comparable directement aux valeurs de Kuhn/Leduc. Les adversaires changent pendant l'entraînement ; cette erreur n'est pas une perte sur un ensemble de validation indépendant.

## Matchs de contrôle

Chaque ligne repose sur 10 000 mains, 5 000 paires avec échange des agents et réutilisation des cartes, graine de simulation 123. Les réseaux sont tirés selon la représentation SD-CFR de la stratégie moyenne, une fois par agent et par main, puis conservés pour tout le parcours. Les IC95 sont approximatifs et calculés sur les moyennes des paires.

| Réseau | Adversaire | Gain bb/100 | IC95 approximatif |
| --- | --- | ---: | --- |
| Graine 42, 16 itérations | Aléatoire | +76,280 | [+66,438 ; +86,122] |
| Graine 42, 16 itérations | Passif | −6,455 | [−12,920 ; +0,010] |
| Graine 42, 16 itérations | Agressif | −12,145 | [−18,320 ; −5,970] |
| Graine 43, 16 itérations | Aléatoire | +73,290 | [+63,573 ; +83,007] |
| Graine 43, 16 itérations | Passif | −2,405 | [−7,059 ; +2,249] |
| Graine 43, 16 itérations | Agressif | −12,555 | [−18,259 ; −6,851] |

Une stratégie initiale uniforme a également été mesurée : +3,270 bb/100 contre l'aléatoire, −6,155 contre le passif et −54,405 contre l'agressif. Les données correspondantes sont dans `match-uniform.csv`. Les deux réseaux entraînés progressent sur certains profils, mais restent exploitables et perdent contre le profil agressif. Ce premier budget ne permet pas d'affirmer un niveau GTO.

Les résultats détaillés sont dans `match-seed-42.csv` et `match-seed-43.csv`. Le test `self` sur 1 000 mains donne exactement 0 BB avec IC95 [0 ; 0], conformément au protocole apparié. La reprise de la sauvegarde 16 jusqu'à l'itération 25 a aussi été exécutée, avec création des sauvegardes 24 et 25.

**Exploitabilité exacte, espérance exacte et regret externe global exact : non calculés pour ce jeu.** Les CSV NLHE ne remplacent pas ces mesures par des zéros. Les garanties des petits jeux ne sont pas transférées automatiquement à ces réseaux ni au Hold'em sans les restrictions de cette réduction.
