# LeChiffre — Behavioral Engine hiérarchique Z₀ / Z₁ / Z₂

LeChiffre dispose d'un **seul Analyzer A**, d'un **seul Generator G** et d'un **même espace latent** pour les trois niveaux. Les entraînements A/G historiques, la CLI qui les pilotait et les anciens checkpoints ont été supprimés. **Aucune migration silencieuse ni repli sur l'ancien trainer n'est autorisé.**

## Signification des niveaux

- **Z₀ (blueprint)** : vecteur latent global entraîné uniquement à partir des mains humaines de la partition Train. Le vecteur est appris conjointement avec G. Son écart-type de référence reste un hyperparamètre explicite ; ce n'est pas une incertitude populationnelle calibrée.
- **Z₁ (baseline individuelle)** : vecteur initialisé **exactement à Z₀**. Il est adapté après chaque main observée du joueur, avec G fixé, en utilisant la vraisemblance jointe action, sizing et timing. Ses précisions diagonales accumulent une approximation empirique de l'information. La distance de Z₁ à Z₀ n'est pas forcée à croître.
- **Z₂ (état instantané)** : distribution inférée par A pour une décision donnée. Le **prior** ne lit que les caractéristiques avant action et Z₁ ; le **posterior** peut lire `ObservedAction` après l'action. Ces deux chemins partagent les mêmes paramètres de A. La reconstruction posterior n'est jamais présentée comme une prédiction prospective.

Aucune carte privée, carte publique, nom de joueur ou identifiant n'entre dans A ou G. L'identifiant pseudonyme sert seulement au regroupement des mains et à la persistance des profils.

## Entraînement

Le seul trainer est `HierarchicalTrainer` dans `src/datasets/behaviour_parquet_loader/hierarchical_training.rs`.

1. **Blueprint** : `train_blueprint_epoch()` met à jour Z₀ et G sur toutes les décisions Train, y compris celles des sujets non sélectionnés pour Personalization. Les mini-batches déterministes suivent l'ordre d'ingestion des mains, puis l'ordre des sujets dans la main. `blueprint_batch_decisions` est obligatoire ; le dernier mini-batch incomplet est utilisé. Le nombre de décisions doit égaler la couverture Train et `optimizer_steps` indique le nombre réel de mises à jour.
2. **Personnalisation** : `train_personalization_epoch()` parcourt toutes les décisions des sujets sélectionnés dans Train, par `(hand_index, subject_id, sequence_index)`. Chaque sujet possède un état Z₁ et précision initialisé depuis Z₀ au début de chaque trajectoire disjointe de `support_hands` mains exploitables (au moins une décision du sujet). Le dernier bloc incomplet est conservé. Les mains sans décision du sujet ne consomment pas le budget. G apprend uniquement la NLL de G(C, Z₁ avant la main). Après la dernière décision du sujet dans une main complète, l'adaptateur met à jour son état une seule fois, sans modifier G. Les états traversent les mises à jour Adam ; si une main traverse deux batches, son adaptation utilise le G du batch contenant sa dernière décision. Il n'y a ni fenêtre Support glissante, ni replay des anciennes mains, ni gradient à travers l'adaptation. L'adaptateur réévalue uniquement les décisions de la main qui vient de se terminer pour son gradient latent. Les batches mixtes conservent `blueprint_batch_decisions` décisions, dernier batch incomplet inclus. Z₀ et A restent figés. Le vecteur, la précision et les compteurs sont réinitialisés ensemble à chaque frontière de trajectoire et au début de chaque nouvelle époque. Les profils utilisés pour prédire les mains d’une trajectoire complète ont de 0 à `support_hands - 1` mains d’historique. Le protocole Momentary reste inchangé.
3. **Momentané** : `train_momentary_epoch()` entraîne A et G à partir du prior, du posterior et d'une KL conditionnelle, toujours à paramètres fixes pendant la construction d'un batch. Le prior ne lit pas l'action actuelle. G utilise uniquement la moyenne des NLL des deux branches Z₂. Z₁ reste adapté directement via G sur Support, sans dépendance aux Z₂ inférés.

Chaque phase a son objectif propre, sans termes auxiliaires aux niveaux précédents. `mean_nll` rapporte la NLL de la phase (Z₀, Z₁, ou moyenne des tirages prior/posterior), hors KL. Les mesures Z₀ et Z₁ restent des diagnostics, pas des objectifs ajoutés. Les modifications de G peuvent changer les prédictions à Z₀ même si le vecteur Z₀ reste figé.

La séparation globale est **Train / Validation / Test** sur des mains entières, par manifeste validé. Pour Momentary, Support et Query sont des rôles locaux **strictement à l'intérieur de Train** : le sampler de `training_data.rs` crée des Query successives non dupliquées pour chaque joueur, en prenant pour chaque Query un Support constitué des `support_hands` mains Train immédiatement précédentes. Le dernier bloc Query peut contenir moins de `query_hands` mains. Les mains qui n'ont pas de décision cible peuvent servir de Support. Une main rejetée à l'import ou structurellement invalide ne fournit aucun événement. Toute décision exploitée requiert un `LegalActionSpace` issu du replay exact `rs-poker`, jamais inféré à partir de l'action observée.

### Découpage des mains retenues, sans fuite entre décisions

La commande `--split-behaviour-dataset-hands <parquet-dir> <new-manifest.json>`
crée, en Rust, un manifeste **70 % train / 15 % validation / 15 % test**, en
comptant les **mains retenues complètes**, jamais les décisions. Les deux
premières tailles sont arrondies à l'entier inférieur et le reliquat est
attribué au test. Le manifeste conserve pour chaque main son `hand_id`, son
`hand_index`, son nombre de décisions et sa taille de table. Il inclut
l'empreinte déterministe non cryptographique utilisée par les checkpoints.
Il indique aussi les effectifs disponibles pour chaque sujet pseudonymisé
dans chaque partition.

Exécution depuis le Makefile existant avec le répertoire de données monté en
`/host` dans Docker :

```bash
DATA="$HOME/lechiffre-data-v02"
RUN_OPTIONS="--user $(id -u):$(id -g) --volume ${DATA}:/host"
sudo make behavioural RUN_OPTIONS="$RUN_OPTIONS" ARGS="--split-behaviour-dataset-hands /host/parquet-v02-prepared /host/hand-split-v02.json"
```

Pour 17 611 mains, le découpage comprend 12 327 mains train, 2 641 validation
et 2 643 test. Tous les événements d'une même main et toutes ses associations
aux joueurs restent dans une partition unique. Le dataset Parquet d'origine est
conservé intact ; le chemin du manifeste doit être nouveau.

**Le `hand_index` représente l'ordre d'ingestion, pas un horodatage réel.** Le manifeste ne prouve donc pas la chronologie des sessions. En revanche, les identités de toutes les mains, leurs décisions, les effectifs des partitions, les effectifs par sujet et l'empreinte du dataset sont vérifiés contre le Parquet à chaque commande.

Le format canonique est **`lechiffre.ingestion-hand-split.v2`** (algorithme 2). Le champ incorrect `hierarchical_training_integration` a été supprimé. Le lecteur rejette un manifeste v1 : regénérer uniquement le manifeste à partir du même Parquet par `--split-behaviour-dataset-hands`, sans réimporter ni réauditer le dataset. Aucun manifeste incomplet, altéré ou lié à un autre dataset n'est accepté.

### Dataset autonome : légalité exacte stockée une fois

La source de vérité de la légalité reste le replay `rs-poker` : à l'import CSV/JSON,
le moteur reconstruit les masques d'actions autorisées et les bornes intégrales de mise.
À partir de cette version, `--export-poker-with-the-boys-parquet` scelle également
`exact_legality.v1.json` dans le répertoire du dataset. Le fichier ne contient
aucun identifiant de joueur ; il conserve pour chaque décision retenue le masque
`fold/check/call/bet/raise` et les bornes exactes (`min_added_chips`,
`max_added_chips`, `chip_unit`, `big_blind_chips`). Un champ ne peut pas être reconstruit
ou déduit d'une action effectivement observée.

Pour **le Parquet déjà exporté**, exécuter une seule fois
`--attach-behaviour-legality <parquet-dir> <fichier-source> <session-id> [...]`
avec **les mêmes sessions sources utilisées lors de l'export initial**. La commande
valide toutes les décisions retenues et refuse les entrées manquantes, les indices
supplémentaires de mains retenues et toute incompatibilité avec le contexte ou
la mise réellement observée. L'attachement est atomique (publication par lien),
non réinscriptible. Les fichiers `events.parquet`, `subjects.parquet`, le split v2
et leur empreinte **restent inchangés** ; aucun nouvel import ou split n'est requis.

**Ensuite toutes les commandes Train / Resume / Evaluate / Select / Create-Baseline /
Advance-Baseline fonctionnent sans CSV/JSON ni session-id.** La lecture vérifie
la version, les identités, le nombre et l'ordre de toutes les décisions, les
limites exactes, l'empreinte du dataset et celle de la légalité. Si la légalité
manque ou ne correspond pas, aucun entraînement ni évaluation ne démarre.
Les checkpoints globaux sont strictement versionnés v9, avec empreinte séparée
de la légalité pour refuser une reprise sur d'autres bornes de jeu.

Le blueprint Z₀/G apprend uniquement sur Train. En Personalization, Z₁ est cumulatif à l’intérieur de chaque trajectoire Train ; en Momentary, Z₁ est adapté sur chaque Support de Train et G/A sont entraînés sur toutes les Query distinctes de Train. En Validation et Test, tous les poids sont figés et Z₁ n'utilise que les dernières `support_hands` mains Train exploitables de chaque joueur (au moins une décision du sujet), un budget explicite lié au checkpoint. Un historique insuffisant est une erreur. **Test ne consomme jamais Validation comme historique.** La métrique Z₂ posterior demeure une reconstruction après observation.

### Configuration explicite

`configs/hierarchical.example.json` contient **toutes les clés obligatoires**. Les anciennes clés `first_query_hand_index`, `first_evaluation_hand_index` et `evaluation_hands` sont refusées ; `support_hands` et `query_hands` décrivent un épisode Momentary interne à Train ; `support_hands` fixe aussi la longueur des trajectoires Personalization (en mains exploitables) et le budget de validation. Il s'agit d'un exemple de syntaxe, non d'hyperparamètres optimisés pour des données réelles : les nombres de mains par épisode et les pseudonymes doivent être remplacés par les valeurs de ton dataset. Toute clé inconnue, clé manquante ou champ de longueur incorrecte entraîne une erreur.

```bash
cargo run --release -- --train-behavioural configs/hierarchical.example.json data/parquet hand-split-v2.json model-v9.json
```

Chaque époque Blueprint et Personalization imprime une validation de Z₀ sur la cohorte de sujets sélectionnés ; chaque époque Personalization imprime aussi une validation de Z₁ avec `support_hands` mains exploitables. Les scores ne consultent jamais Test. Une campagne peut s'arrêter à Blueprint (`personalization_epochs: 0`, `momentary_epochs: 0`) ou à Personalization (`momentary_epochs: 0`). Le checkpoint est alors enregistré et peut être repris avec une configuration de reprise explicite ; aucun retour à une phase antérieure n'est accepté.

```bash
cargo run --release -- --evaluate-blueprint blueprint-v9.json data/parquet hand-split-v2.json blueprint-validation.json
```

L'ordre d'ingestion ne constitue pas une preuve de chronologie temporelle. L'application refuse de remplacer un chemin existant.

### Contrôle de la personnalisation avec G figé

Dès la fin de Blueprint, `--benchmark-personalization` mesure l'apport de Z₁ indépendamment de toute phase ultérieure. La commande accepte également un checkpoint après Personalization ou Momentary. Elle ne modifie ni Z₀, ni G, ni A, ni leurs états d'optimisation et ne sauvegarde aucun checkpoint. Pour chaque budget explicite (zéro, puis valeurs strictement croissantes), elle reconstruit Z₁ depuis Z₀ avec les dernières mains exploitables de Train et mesure sa NLL et son accuracy sur les mêmes décisions Validation. Le score Z₀ est calculé une seule fois ; à zéro main les rapports numériques doivent être exactement identiques. Toute couverture différente, tout historique insuffisant ou tout fichier de sortie existant est une erreur.

```bash
cargo run --release -- --benchmark-personalization blueprint-v9.json data/parquet hand-split-v2.json 0,1,10,50,100 personalization-benchmark.json
```

Le rapport `lechiffre.behavioural.personalization-benchmark.v1` contient la référence Z₀ (`reference`), les scores Z₁ par budget (`reports`), les compteurs de phases, les identifiants et empreintes de la campagne. Aucun score de Z₂ n'est produit. Le benchmark complet `--benchmark-behavioural` exige toujours un entraînement Momentary terminé. Le contrôle ne constitue ni un replay en ligne ni une preuve de chronologie réelle.

### Validation, Test et score probabiliste Z₂

```bash
# Rapport JSON créé sans remplacement du fichier existant :
cargo run --release -- --evaluate-behavioural model-v9.json data/parquet hand-split-v2.json validation validation-report-v4.json

# Test final, une seule fois le checkpoint sélectionné :
cargo run --release -- --evaluate-behavioural selected-v9.json data/parquet hand-split-v2.json test test-report-v3.json

# Sélection du meilleur checkpoint selon la NLL *marginale* du prior Z₂ :
cargo run --release -- --select-best-behavioural data/parquet hand-split-v2.json selected-v9.json model-a.json,model-b.json
```

L'évaluation calcule `z0_predictive_nll`, `z1_predictive_nll` et `z2_prior_marginal_nll` sur les mêmes décisions. Pour ce dernier score, 64 tirages indépendants de la loi prior de Z₂ passent par G ; l'intégration Monte Carlo est réalisée par log-mean-exp stable. Les tirages viennent d'un flux déterministe propre à l'évaluation et ne modifient jamais la reprise d'entraînement. Le score `z2_posterior_reconstruction_nll` reste **une reconstruction après observation**, jamais une prédiction prospective ni un critère de sélection. Les métriques action, sizing conditionnel aux mises/relances et timing sont séparées avec leurs dénominateurs exacts dans le JSON `lechiffre.behavioural.validation.v4`. Le timing utilise la densité normale en `log1p(ms)` (pas une masse discrète en millisecondes arrondies). La décomposition marginale utilise la règle de chaîne : NLL(action), NLL(sizing | action) et NLL(timing | action, sizing). La moyenne jointe égale la moyenne action + la fréquence de mises/relances × la moyenne sizing conditionnelle + la moyenne timing.

Le benchmark d'historique compare exactement les mêmes décisions Validation pour tous les budgets explicites (zéro, puis valeurs strictement croissantes). Chaque Z₁ est construit depuis Z₀ avec les dernières mains exploitables Train, puis figé. Aucun sujet n'est écarté pour satisfaire un budget trop grand. Les trois accuracies d'action utilisent l'argmax, pour Z₂ après moyenne des cinq probabilités d'action sur les 64 tirages du prior ; les mêmes tirages sont réutilisés entre budgets. NLL et accuracy mesurent les prédictions et non une vérité terrain de l'espace latent. Ce benchmark mesure l'effet de la quantité d'historique, pas un replay en ligne ni une généralisation à des joueurs inconnus.

```bash
cargo run --release -- --benchmark-behavioural model-v9.json data/parquet hand-split-v2.json 0,1,10,50,100 history-benchmark.json
```

Le format `lechiffre.behavioural.history-benchmark.v1` contient les rapports, identifiants et empreintes de la campagne. La sélection existante par NLL Z₂ seule ne qualifie pas le modèle aux trois niveaux : vérifier séparément la préservation des références avant Test.

La commande `--evaluate-unseen-behavioural` a été supprimée : elle ne garantissait pas la séparation des joueurs reliés par des mains communes. Le protocole courant mesure des nouvelles mains de joueurs présents dans Train ; un benchmark **réellement** indépendant par joueur exige une définition distincte de partitions par groupes disjoints. Aucun résultat de ce type n'est revendiqué par le test actuel.

### Sauvegarde et reprise

Le format unique est `lechiffre.hierarchical.checkpoint.v9`. Les versions V1 à V8 sont refusées, sans conversion : la tête sizing et ses moments Adam ont changé. Les anciennes configurations `personalization_history_hands` sont refusées ; `personalization_epochs` et `personalization_extra_epochs` restent des entiers obligatoires.

V9 conserve la NLL jointe, les trajectoires de Personalization, l'adaptation Z1, les têtes action/timing et OpenBLAS. Pour Bet/Raise, G produit 25 sorties : un logit all-in, huit logits de mélange, huit centres libres et huit log-échelles. L'axe est `log1p(montant ajouté / unité)`, unité = BB préflop, pot avant action postflop. Chaque logistique est tronquée et normalisée sur les cellules des montants non-all-in légaux (demi-incrément de part et d'autre, bord physique à zéro). L'all-in est un atome distinct. Les poids sont softmax, les échelles exponentielles, sans plancher de largeur. La NLL reste celle du montant légal exact. L'échantillonnage inverse la CDF discrète par recherche binaire. Les métriques CRPS/pot, MAP exact et tolérances sont calculées dans le diagnostic Validation, sans objectif auxiliaire.

```bash
cargo run --release -- --resume-behavioural model-v9.json data/parquet hand-split-v2.json resumed-v9.json configs/hierarchical-resume.example.json
```

La configuration de reprise est stricte. Les phases ne peuvent pas être rejouées dans le mauvais ordre. Les changements de dataset, de sujets, d'épisode Support/Query ou de manifeste sont rejetés. Le checkpoint `lechiffre.ag.checkpoint` est **incompatible et explicitement refusé**.

## Persistance des baselines Z₁

Chaque `PlayerBaselineState` possède le vecteur, l'incertitude, la précision diagonale, les compteurs de mains/décisions, la dernière main exploitée et une frontière globale **exclusive** `processed_before_hand_index`. Ces informations sont stockées à part du modèle global dans le format `lechiffre.player_baseline.v1`.

```bash
# Première construction (depuis Z₀, avant la main d'index 800)
cargo run --release -- --create-player-baseline model-v9.json data/parquet 42 800 player42-v1.json

# Mise à jour uniquement sur les mains nouvelles d'index < 1000
cargo run --release -- --advance-player-baseline model-v9.json data/parquet 42 1000 player42-v1.json player42-v2.json
```

Les profils sont liés au **model_id exact** et au `subject_id` pseudonyme. Toute discordance est bloquante. Le nouvel état est écrit atomiquement dans un fichier inédit : ni l'entrée, ni le modèle, ni les poids A/G ne sont modifiés. Une erreur de légalité ou de décodage annule l'intégralité de la mise à jour. Le dernier index déjà couvert interdit de rejouer les mêmes mains. Un autre modèle global nécessite une personnalisation distincte ; les coordonnées Z₁ ne sont pas interchangeables entre versions.

Z₂ n'est pas un profil permanent. Il se calcule lors de l'observation d'une décision ou lorsqu'un prior est demandé avant celle-ci. Son archivage n'est pas nécessaire à la persistance de Z₁.

## Code maintenu, code supprimé

- Conservés : `GeneratorGradientBatch`, les têtes action/sizing/timing, `PlayerBaselineAdapter`, `ExactLegalityProvider`, la validation du replay exact, les adaptateurs sources et le Parquet strict.
- Une seule implémentation d'épisodes : `training_data.rs` et ses blocs Query successifs dans Train. `sampling.rs` et `training_episode.rs` ont été supprimés avec leurs réexports : ces deux chemins n'étaient pas consommés par `HierarchicalTrainer` et ne respectaient pas directement le manifeste global.
- Le split par composantes de sujets (`split_subjects`, `SubjectSplitConfig`, `SubjectDatasetSplit`) a été supprimé : la CLI utilise exclusivement le manifeste par mains ; `cooccurrence_components` demeure pour les statistiques exactes.
- Les abstractions `BehaviourGenerator` et `validate_latent_dimensions` ont été supprimées : le seul trainer vérifie déjà les dimensions concrètes A/G/Z₀ dans son constructeur. `BehaviourLatentModel`, réellement utilisé par A et G, demeure.
- `evaluate_unseen` a été supprimé : il ne prouvait qu'une différence de listes d'identifiants, pas l'isolation des groupes de cooccurrence. La CLI, les docs et les tests liés ont été nettoyés.
- Supprimés aussi : le champ obsolète du manifeste, l'ancien score Z₂ au latent moyen et les lecteurs des formats précédents. Aucune compatibilité silencieuse n'est proposée.

Le même `G(C, Z)` génère les distributions d'actions légales, tailles de mise et temps de décision. Les montants autorisés restent définis en unités de jetons exactes par le moteur NLHE.

## Contrôles d'intégration

```bash
cargo fmt --all -- --check
cargo clippy --locked --all-targets --all-features -- -D warnings
cargo test --locked --all-targets --all-features
cargo test --locked --doc --all-features
```

Un résultat CI positif prouve seulement que les contrôles automatisés du dépôt réussissent. Une validation sur des mains réelles retenues pour test doit encore démontrer la valeur prédictive de Z₁/Z₂, les diagnostics de collapse et la calibration du modèle.

### Coût de calcul de l'adaptation

L'adaptation de Z₁ utilise la même NLL et la même rétropropagation d'entrée que l'entraînement de G, sans allouer ni calculer les produits extérieurs des gradients de paramètres. Les batches Blueprint/Personalization utilisent leur snapshot G immuable directement, puis vérifient encore l'identité du modèle avant Adam. Le buffer de gradients par décision est réutilisé pour la somme candidate ; une erreur n'altère pas l'accumulateur précédent. G utilise OpenBLAS LP64 de Debian (dépendance de compilation et d'exécution obligatoire) : DGEMV pour les couches denses et la propagation des dérivées, DGER pour les gradients individuels, et DGEMM pour les sommes de produits extérieurs des batches Blueprint/Personalization. Les lignes de X et D sont accumulées par décision, puis la matrice DᵀX est calculée avant Adam. Les poids persistés restent f32, leurs conversions f64 sont mises en cache et rafraîchies après Adam. Les calculs restent f64, mais l'ordre des sommes et les arrondis peuvent différer. Les images fixent OPENBLAS_NUM_THREADS=1. Le réseau A conserve son implémentation actuelle.
