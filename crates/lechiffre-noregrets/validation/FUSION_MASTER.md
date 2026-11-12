# Historique de la correction après fusion — version 11

Ce compte rendu décrit la version précédente, qui avait réintroduit un agent
retiré de master. Cette compatibilité a été supprimée ; utiliser désormais
`ALIGNEMENT_EQUITY.md` et le guide courant `CORRIGER_APRES_FUSION.md`.

Référence du projet lue par le connecteur GitHub en lecture seule :
`feature/ia_gto`, commit `3ccd95ce30120971f4fafafeb7bc654145a63bd8`, après fusion
de `master` (`73957f00e05863d6de92da529c7e1efe2709c698`). Aucun push effectué.

## Correctifs

- `vendor/rs-poker/src/arena/rotation/builder.rs` : extraction du nom de la
  variante `Generated`, comme dans le comparateur. Supprime l'erreur E0004.
- `vendor/rs-poker/src/arena/comparison/runner.rs` : application effective de
  `max_raises_override` après création de l'état, en conservant son rake.
  Sans cela, le plafond natif de trois relances faisait refuser la partie par
  les modèles NoRegrets entraînés avec relances libres.
- Ancien `RandomPotControlAgent` conservé dans un module séparé, avec son
  implémentation issue du fichier `random.rs` avant la fusion, au commit
  `0651671df4bee3d8989deab9aece079b70c0c808`. Variante JSON, validation,
  constructeur et lecture du nom rétablis dans les deux constructeurs natifs
  et dans la consultation des charts. EquityAgent est conservé séparément.
- `vendor/rs-poker/src/arena/cfr/agent/mod.rs`, uniquement dans deux tests
  d'énumération et de décision river : choix explicite de KnownHandsEstimator pour fixer leurs mains
  et tester la stabilité exacte de l'énumération du board. Le constructeur
  CFR bas niveau utilise maintenant UniformRandomEstimator par défaut, ce
  qui rendait l'ancienne hypothèse de mains fixes incorrecte. Les assertions
  d'égalité exacte et de décision sont conservées et aucun estimateur de jeu n'est modifié.
- Profil `configs/agents/equity.json` avec les paramètres fournis.

## Régressions couvertes

Le nouveau test `merged_agents.rs` vérifie les noms des profils générés dans
le constructeur de rotation, une vraie table six joueurs avec NoRegrets,
Equity et l'ancien pot-control, et une comparaison native NoRegrets contre
Equity avec permutation des sièges. Les 14 tests préexistants de l'adaptateur
sont conservés.

Résultats vérifiés dans cet environnement Linux avec Rust 1.99.1 :

- 17 tests de l'adaptateur réussis avec `--locked --features gui`.
- Les 1 071 tests de la bibliothèque rs_poker réussissent après correction des
  deux fixtures à mains fixes, avec quatre threads et une pile de 50 Mo pour
  les tests Linux. Aucun test n'est ignoré et les assertions sont conservées.
- Clippy de l'adaptateur réussi avec `--all-targets --features gui -- -D warnings`.
- Compilation release des deux binaires.
- Compilation du binaire natif `rsp` avec `rsp,arbitrary` et `--locked`.
- Commande de table six joueurs terminée sur 6 000 mains ; somme des gains nulle.
- Comparaison native de six profils, dont Equity à 1 000 simulations et NoRegrets,
  terminée sur 720 mains ; 120 mains par siège pour chaque agent, gains à somme nulle.

Les rapports `merge-table-6000.*` et `merge-equity-720-*` sont joints comme
preuves d'exécution. Ces tests ne mesurent pas une exploitabilité exacte ni
la force GTO des modèles. Les cartes des simulations Equity et CFR ne sont
pas toutes contrôlées par la graine du comparateur.

Le message `unused_mut` dans `vendor/noregrets/src/abstraction.rs` est un
avertissement amont préexistant. Il n'empêche ni la compilation ni les tests.
Les sources NoRegrets et le format des modèles restent ceux de l'archive
précédente ; le correctif concerne leur intégration après fusion.
