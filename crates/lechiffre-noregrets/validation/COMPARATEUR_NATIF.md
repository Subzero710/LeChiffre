# Comparateur natif et modèles entraînés — 11 novembre 2026

## Changement

`lechiffre-arena compare` charge les JSON NoRegrets et rs_poker puis appelle
`rs_poker::arena::comparison::ArenaComparison`. Les permutations, simulations,
historiens et calculs statistiques sont ceux du comparateur natif. Il n'y a pas
une seconde boucle de comparaison. Le même tableau de bord natif est réutilisé.
Le chargement/contrôle des profils est partagé avec l'ancien mode `table`.

NoRegrets reste dans son adaptateur. rs_poker ne dépend pas du projet ou de
NoRegrets. Un générateur générique d'agents permet d'injecter un blueprint
chargé une seule fois, de créer un nouvel agent à chaque main/siège, et de lui
transmettre la graine calculée par le comparateur. Les autres générateurs
conservent leur comportement par défaut.

Le plafond natif de trois relances est explicitement désactivé pour cette
commande afin de conserver les règles des modèles NoRegrets. Le comportement
des anciens appels natifs reste inchangé tant que cette option n'est pas donnée.
Les tapis sont fixes et identiques, déduits des modèles. Blindes 1/2, ante 0,
rake 0. Les configurations effectives et les hashes sont sauvegardés séparément
des fabriques vivantes dans `comparison.json`.

La CLI `rsp` seule ne comprend pas un profil JSON NoRegrets. Il faut utiliser
le lanceur de l'adaptateur, qui appelle bien son comparateur natif. Les anciens
modes `table` gardent les sièges fixes et font tourner le bouton.

## Suivi de la branche principale

Sources consultées par le connecteur GitHub en lecture seule :

- `feature/ia_gto` : `0651671df4bee3d8989deab9aece079b70c0c808`.
- Branche principale **`master`** : `73957f00e05863d6de92da529c7e1efe2709c698`.
- NoRegrets amont : `757f7692738069522195d2b486eec60a8b010c0c`.

Cette archive cible la branche feature ci-dessus et conserve les corrections
précédentes retirant les dépendances aux modules SD-CFR/NLHE supprimés.
Les sources du solveur NoRegrets et du tableau de bord ne changent pas.
Six fichiers communs rs_poker reçoivent seulement les points d'extension :

| Fichier sous `vendor/rs-poker/src/` | Ajout |
|---|---|
| `arena/agent/generator.rs` | Méthode de construction avec graine, avec implémentation par défaut |
| `arena/agent/config.rs` | Variante programmatique `Generated` et délégation au générateur |
| `arena/agent/mod.rs` | Export du trait de générateur accepté par la configuration |
| `arena/comparison/builder.rs` | Lecture du nom d'un agent généré |
| `arena/comparison/runner.rs` | Option explicite pour le plafond de relances |
| `bin/rsp/arena/charts.rs` | Traitement de la variante supplémentaire dans les correspondances exhaustives |

`master` a déjà évolué, notamment pour EquityAgent et le rake du comparateur.
Ses versions complètes ne sont pas recopiées sur la feature. En cas de fusion
ultérieure, conserver ces nouveautés et ajouter les points d'extension décrits
ci-dessus ; ne pas remplacer ses fichiers par les anciennes versions complètes
de l'archive. Une vérification d'application des différences a identifié des
ajustements de contexte dans `config.rs`, `mod.rs` et `runner.rs` : une absence
de conflit de fusion n'est pas promise. Aucune fusion ni écriture distante effectuée.

## Vérifications

- Les **14 tests** du module passent, dont trois nouveaux tests du comparateur.
- Deux petits modèles distincts entraînés pour trois joueurs occupent les six
  ordres de sièges, deux fois chacun ; noms et hashes restent distincts.
- Le rapport natif agrège correctement toutes les positions ; le total des gains
  est nul dans cette partie sans rake.
- Sept profils à deux places donnent 42 mains par série ; la répétition avec la
  même graine reproduit les statistiques des agents aléatoires natifs.
- Les modèles absents, incompatibles et les paramètres invalides échouent avant
  la simulation.
- Clippy `--all-targets --features gui -- -D warnings` passe pour l'adaptateur.
- Compilation release des deux binaires, avec Cargo.lock et `--locked`.
- Le binaire rs_poker `rsp` et la bibliothèque compilent avec les fonctionnalités
  `rsp,arbitrary`, y compris la variante programmatique exclue du fuzzing dérivé.
- Les **55 tests natifs** du module `arena::comparison` passent également.
- Un entraînement natif de 64 parcours à six joueurs a été sauvegardé et rechargé
  par `inspect`, puis comparé au modèle fourni dans la TUI sur 720 mains.
- La TUI a été ouverte dans un vrai terminal ; une main a été ouverte dans
  l'historique et Ctrl+C a quitté proprement avec le code 0.

Test complet de la commande de comparaison avec le modèle fourni : six profils,
`-p 6 -n 10 --seed 123 --no-tui`, soit **7 200 mains**, et **1 200 mains par siège
pour chaque profil**. Les 7 200 mains OHH sont enregistrées et les gains restent
à somme nulle. Les trois rapports `native-7200-*` sont joints comme preuve de
fonctionnement, sans constituer une mesure suffisante de force stratégique.

Les permutations équilibrent les positions relatives. Les cartes futures
utilisent des sous-flux aléatoires distincts par permutation. L'exploitabilité
exacte et un intervalle de confiance apparié ne sont pas calculés. Le petit
modèle de validation à 64 parcours n'est pas un modèle de jeu recommandé.
L'avertissement amont `unused_mut` préexistant n'est pas modifié.
