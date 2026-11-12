# NoRegrets autonome — correction du 11 novembre 2026

Compte rendu historique de la correction des dépendances, avant l'ajout du
comparateur natif. Pour l'état actuel et le dernier suivi de `master`, voir
`COMPARATEUR_NATIF.md` et `MODE_D_EMPLOI_NOREGRET.md` à la racine du projet.

La branche `feature/ia_gto` au commit
`0651671df4bee3d8989deab9aece079b70c0c808` a supprimé les dossiers des anciens
modules SD-CFR et NLHE, mais le manifeste et des imports NoRegrets les
référençaient encore. Cela empêchait toute compilation, y compris `table`.

La correction retire ces dépendances du manifeste et de son verrouillage,
retire les anciens outils de comparaison SD-CFR réduite du binaire et conserve
le petit flux de graines SplitMix64 localement, avec exactement la même
séquence. L'adaptateur de cartes reste dans `bridge.rs`, et la provenance dans
`reference.rs`. Les tests de l'agent construisent désormais leur état de jeu
directement avec rs_poker.

Les commandes de table, le lanceur graphique, les règles, le vrai MCCFR
NoRegrets et le format des sauvegardes Agent sont conservés. Le JSON de démo
est corrigé pour employer `reference`, avec la référence six joueurs fournie
à 10 BB ; les cinq autres choix de sièges du dépôt sont conservés.

La validation est exécutée dans une arborescence où les deux anciens dossiers
n'existent pas. Le manifeste et le Cargo.lock ne contiennent plus leurs
dépendances. Les vérifications comprennent la compilation des deux binaires,
les tests du module, un entraînement natif suivi d'un chargement et une table
exécutée avec la commande du guide. Les résultats de ces petits contrôles ne
constituent pas une mesure de force des stratégies.

La branche principale `master` est toujours au commit
`dbc6067615a042c28484cac8dd66a4a878556e61`. Cette correction n'introduit aucun
changement supplémentaire dans rs_poker ou dans le code NoRegrets amont.

Résultats vérifiés : les 11 tests du module passent, ainsi que la compilation
release des deux binaires et Clippy `--all-targets` avec `-D warnings`. La
commande exacte du guide a terminé 6 000 mains à six joueurs avec les deux
anciens dossiers absents. Un entraînement natif de 64 parcours à six joueurs
a été sauvegardé et rechargé par `inspect`. Les rapports de table sont conservés
dans `autonomous-table-6000.csv/json`. L'avertissement amont `unused_mut`
préexistant n'est pas modifié.
