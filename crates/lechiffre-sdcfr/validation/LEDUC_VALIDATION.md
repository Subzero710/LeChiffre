# Validation Leduc — 22 octobre 2026

Environnement : Linux x86_64, Rust 1.99.1. L'entraînement et les tests livrés fonctionnent entièrement en Rust, sans dépendances externes. OpenSpiel 2.0.2 a été utilisé séparément pour produire une référence indépendante ; il n'est pas nécessaire pour lancer le projet.

## Règles et information

La variante est heads-up, ante 1, mises 2/4, deux mises au maximum par tour (ouverture et une relance), joueur 0 en premier aux deux tours. Le paquet comporte six cartes physiques ; les observations regroupent les couleurs équivalentes. Les actions sont fold, check/call et bet/raise, avec masque de légalité.

Le fichier `tests/fixtures/leduc_openspiel.tsv` contient une référence produite à partir de `pyspiel.load_game("leduc_poker", {"players": 2, "suit_isomorphism": False})` sous OpenSpiel 2.0.2. Tous les 9 450 états après la distribution privée sont comparés : joueur actif, nœud de hasard ou terminal, actions légales et gains nets. Les quatre cartes publiques restantes sont équiprobables. L'arbre Rust contient 9 451 nœuds, en ajoutant un nœud initial de distribution privée conjointe.

Les observations ont 288 ensembles d'information, contre 936 en distinguant les couleurs physiques. Le retrait des cartes reste calculé sur le paquet physique complet. Les tests vérifient aussi l'invariance aux cartes adverses, l'historique ordonné, les paires, les égalités, les folds, la conservation des gains et la limite des relances.

## Apprentissage réellement exécuté

Paramètres : deux MLP 43 → 64 → 64 → 3, 512 parcours par joueur et par itération, 128 étapes Adam, taux 0,003, réservoir uniforme de 50 000 échantillons par joueur. Les observations identiques sont regroupées pour calculer le gradient de la MSE pondérée ; les cibles brutes restent mémorisées. Les deux réseaux sont réinitialisés avant chaque nouvel ajustement.

Chaque entraînement a été exécuté en 64 itérations, puis repris depuis sa sauvegarde pour 64 itérations supplémentaires, avec la même configuration.

| Graine | Exploitabilité initiale | Après 64 itérations | Après 128 itérations | Valeur J0 après 128 |
| --- | ---: | ---: | ---: | ---: |
| 42 | 2,373611 | 0,345139 | 0,331626 | −0,150257 |
| 43 | 2,373611 | 0,340530 | 0,281954 | −0,118543 |

Convention : exploitabilité = NashConv / 2, en jetons d'ante par main. Il reste une exploitabilité mesurable ; ces résultats montrent une amélioration et ne constituent pas une résolution exacte de Leduc. Les courbes peuvent osciller. Les fichiers `leduc-seed-42.csv` et `leduc-seed-43.csv` contiennent les mesures réelles des deux phases ; le temps de la reprise est additionné au temps de la première phase, sans inclure l'attente entre les commandes.

L'oracle tabulaire complet indépendant atteint une exploitabilité de 0,039859534 et une valeur J0 de −0,087657692 après 1 000 itérations. Les budgets diffèrent : cette référence ne constitue pas une comparaison à coût égal. Aucune donnée de cet oracle n'est utilisée pour entraîner les réseaux.

## Vérification indépendante des politiques apprises

Les stratégies moyennes exportées ont été chargées dans une politique tabulaire OpenSpiel sur les 936 ensembles d'information physiques. OpenSpiel a recalculé leur valeur et leur exploitabilité avec ses propres règles et son propre évaluateur.

| Graine | Exploitabilité Rust | Exploitabilité OpenSpiel | Valeur J0 OpenSpiel |
| --- | ---: | ---: | ---: |
| 42 | 0,331626236 | 0,331626235762294 | −0,150256956002654 |
| 43 | 0,281953540 | 0,281953540412680 | −0,118542692519351 |

Les écarts avec les mesures Rust sont inférieurs à 10⁻⁸, y compris après l'arrondi du CSV. La stratégie uniforme a également la même exploitabilité : 2,373611111111111. Le nouvel évaluateur générique est comparé, sur plusieurs politiques de Kuhn, à l'ancien calcul qui énumère toutes les stratégies pures.

## Contrôles

- 24 tests automatisés, dont gradients numériques sur deux couches cachées, perte pondérée, Regret Matching, moyennes avec reach propre, cibles MCCFR et réservoir.
- Comparaison complète des règles Leduc avec la référence statique OpenSpiel.
- Reprise de Leduc identique à un entraînement continu dans le même environnement ; rejet d'une sauvegarde identifiée comme Kuhn.
- Chargement d'une ancienne sauvegarde Kuhn de 128 itérations : exploitabilité 0,023384 et valeur J0 −0,056133, comme avant.
- `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings` et construction release.

Les anciennes mesures de Kuhn sont conservées dans `VALIDATION.md` et `seed-*.csv`. Le branchement à `rs_poker`, l'apprentissage NLHE, la généralisation à ce jeu et le multiway restent à développer et ne sont pas validés par ces résultats.

Sources : [OpenSpiel Leduc](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/games/leduc_poker/leduc_poker.h), [évaluation de l'exploitabilité](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/algorithms/exploitability.py).
