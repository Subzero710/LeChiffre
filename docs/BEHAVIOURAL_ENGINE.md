# Behavioral Engine — invariants et Generator

## Architecture Z0/Z1/Z2 — entraînement hiérarchique en bibliothèque (étape 6)

The behavioural model has one shared Analyzer A and one shared Generator G.

### Étape 6 — `HierarchicalTrainer` (in-memory, distinct du CLI historique)

`src/datasets/behaviour_parquet_loader/hierarchical_training.rs` définit
`HierarchicalTrainer`, avec **trois phases explicites** et des snapshots fixes
pendant chaque lot de gradients :

1. `train_blueprint_epoch(dataset, subjects, split, legality)` ajuste le vecteur
   explicite **Z0** et G sur les décisions humaines dont le `hand_index` précède
   strictement `split.first_query_hand_index`. Le prior standard deviation est
   *fourni explicitement* ; il n'est pas appris ni calibré à cette étape.
2. `train_personalization_epoch(..., history_hands, legality)` initialise
   strictement Z1 à Z0, l'adapte uniquement avec `history_hands` mains passées
   et entraîne G sur les mains `query` postérieures. Le nombre de mains est un
   paramètre strict, sans duplication ni substitution si l'historique manque.
   On peut exécuter cette phase plusieurs fois avec différents nombres de
   mains. Les paramètres A/G ne sont jamais personnalisés par identifiant.
3. `train_momentary_epoch(..., history_hands, legality)` met à jour A et G en
   utilisant **deux branches du même Analyzer** :
   `p(Z2 | C,Z1)` avant action et `q(Z2 | C,O,Z1)` après action. La loss est
   `(NLL_prior + NLL_posterior)/2 + beta_kl * KL(q||p)` ; G prédit l'action,
   le sizing légal exact et le timing. La rétropropagation analytique de la KL
   et de G passe par les deux branches de A. Les mêmes paramètres A sont utilisés
   dans les deux calculs ; les updates de A et G sont atomiques par sujet.

`NeuralBehaviourAnalyzer::prior_encoded` n'accède **jamais** à `ObservedAction`.
`infer_encoded` utilise l'observation actuelle uniquement pour son posterior.
Le moteur de règles fournit tous les espaces légaux exacts via
`ExactLegalityProvider`; leur absence est bloquante.

`HierarchicalTrainer::evaluate(...)` rapporte **séparément** la NLL future
à Z0, à Z1, au prior Z2 (avant action), et la NLL reconstructive du posterior
Z2 (après action). La NLL au posterior ne doit jamais être présentée comme
une prédiction. Tous les profils de validation sont calculés avec les mains
antérieures à la frontière globale d'évaluation ; aucune action future ne
contribue aux gradients ou à Z1.

**Validation scientifique non acquise :** un modèle ne devient pas fiable parce
que sa loss baisse. La séparation Z0/Z1/Z2 reste partiellement non identifiable ;
le KL évite certains raccourcis mais ne garantit pas l'absence d'effondrement
latent. Il faudra mesurer les ablations sur des joueurs/mains non vus et calibrer
les incertitudes. Les tests unitaires vérifient seulement les contrats et les
transitions d'entraînement, pas une supériorité prédictive réelle.

**Limites d'intégration :** l'ancien `GeneratorTrainer` et les commandes
`--train-behavioural` / `--resume-behavioural` continuent d'utiliser le chemin
historique à Z unique. Ils ne sont **pas** silencieusement redirigés vers le
nouveau modèle. Versionner les checkpoints, migrer le CLI et sauvegarder les
profils Z1 appartiennent à l'étape 7. Les poids anciens ne sont pas traités
comme des poids hiérarchiques compatibles.

- `BehaviourBlueprint` stores the population blueprint Z0 and an explicitly supplied
  prior standard-deviation vector. No population parameters are inferred by its constructor.
- `PlayerBaseline::from_blueprint` creates Z1 with **zero observed hands** and
  bit-for-bit equality `Z1 == Z0`. `PlayerBaselineAdapter` performs the
  *step-5 implementation* of online Z1 personalization; its changes are
  applied to the player embedding only, never to A/G weights.
- `NeuralBehaviourAnalyzer::infer_action(context, observed, baseline)` computes
  a decision-specific Z2 posterior. `prior(context, baseline)` shares the same
  Analyzer parameters but never accepts a current action, sizing or timing.
  `MomentaryBehaviour` exposes the prior, posterior and conditional KL.
- The Generator still outputs the legal discrete action distribution and
  conditional sizing/timing distributions, using one shared multidimensional Z.
  `GeneratorGradientBatch` computes gradients at fixed model parameters; Adam
  applies their mean once. The existing joint trainer now batches query decisions
  **per subject** rather than updating G between those decisions.
- Parser-rejected and structurally invalid hands are excluded *in full*. The
  Parquet writer records a `dropped_hands_invalid` count alongside the existing
  missing-timing count. A training decision missing exact source replay legality
  is still fatal; no legal space is inferred from the target action.

### Step 5 — individual Z1 refinement (implemented, not globally trained)

`PlayerBaselineAdapter` is a blueprint-bound, per-hand online updater. Its
`initialize_player()` returns `Z1(0) == Z0` exactly. `adapt_hand()` receives
all observed voluntary decisions of **one** subject from **one** retained hand,
with the legal spaces recovered through exact game replay. It computes the
Generator's full joint NLL and `dL/dZ` using **fixed G weights** at the **same
pre-update Z1** for all decisions of that hand. It performs one mean-gradient
update, and increments `observed_hands` only after successful completion.

The update uses a diagonal empirical-Fisher *approximation* to precision. For
each coordinate, per hand:

```text
I_hand = mean_over_decisions((dL/dZ)^2)
precision_next = precision_previous + I_hand
step = clip(-learning_rate * mean(dL/dZ) / precision_next,
            +/- max_step_prior_std * Z0_prior_std)
Z1_next = Z1_previous + step
uncertainty_std_next = 1 / sqrt(precision_next)
```

The evidence can move Z1 **towards or away from Z0**, depending on observed
behaviour. Accumulating information slows updates as evidence grows, but neither
strict logarithmic distance to Z0 nor calibrated psychological confidence is
assumed. These precisions are **not** an exact Bayesian posterior. Per-hand
state updates are atomic; invalid sizing, missing exact legality, repeated or
out-of-order global hand indexes, and dimension errors leave Z1 unchanged.

`BehaviourParquetDataset::adapt_baseline_before(subject_id,
before_hand_index, generator, adapter, provider)` is the replay-backed integration.
The boundary is **exclusive and global**: no action at or after that hand index
is read or supplied to the optimizer. Only the selected subject's actions
are scored. Retained hands without the subject's voluntary decisions contribute
no evidence and do not increment the observed-hand count. Other players' actions
remain in the pre-decision context if they occurred earlier in the hand.

The caller holds one `PlayerBaselineState` per player, keyed outside the model.
Model weights, blueprint and other players' baselines are not changed. Stage 7
will add the checkpoint format and managed storage for these states.
Stage 6 introduces the global Z0/A/G hierarchy in the library ;
**the existence of this update API is not evidence that Z1 improves predictions**.

**Migration status:** The CLI joint training objective and `infer(support)` path
still use one historical support-level Z per subject. They do **not** yet train
Z0/Z1/Z2 as a hierarchy. A separate `HierarchicalTrainer` now implements
this training in memory, but the historical CLI will be replaced in step 7.
Prior checkpoints with the old Analyzer layer shape are rejected ; checkpoint
versioning and deployment migration belong to step 7.

Datasets must be re-exported with the new per-hand-drop summary metadata; the
loader validates this metadata strictly. Do not represent a green CI on the
previous commit as validation of this unmerged refactor.


Statut : **Generator différentiable, Analyzer probabiliste et entraînement variationnel conjoint implémentés ; aucun poids entraîné sur un dataset réel fourni**. Aucun blueprint de référence n'est disponible à ce stade. L'entraînement du blueprint est un chantier indépendant, et ne doit pas être confondu avec l'inférence effectuée par le Behavioral Engine.

## Objectif

Le Behavioral Engine apprend un **espace latent comportemental continu** `Z`, **sans labels de joueurs** et sans profils discrets prédéfinis (TAG, LAG, etc.).

- **Generator G** : `G(contexte comportemental C, z) -> distribution de comportements observables`. Génère les probabilités d'actions, de tailles de mises et de temps de décision. Tirer une `ObservedAction` de cette distribution est une opération distincte.
- **Analyzer A** : `A(historique de support du sujet) -> q(Z | support)`, une Gaussienne diagonale sur le profil latent. Une observation isolée ne doit pas être traitée comme l’identification certaine d’un Z unique.
- **Blueprint** (futur composant externe) : stratégie de référence entraînée séparément, qui pourra ultérieurement être injectée dans le système. **G ne suppose pas qu'il existe actuellement.**

A et G décrivent et génèrent des comportements. La recherche d'un profil contre-stratégique et le mélange final avec la stratégie GTO sont explicitement **hors périmètre** de ce module.

## Frontière stricte : aucune carte dans le Behavioral Engine

Le `DecisionContext` contient les états individuels des joueurs (stacks
restants, engagements de street et de main), les contraintes de mise, la position,
les compteurs antérieurs, ainsi que `last_aggressor` avec son action antérieure.
`ObservedAction` contient le type d'action, le sizing relatif au pot et au stack,
le timing et les compteurs d'agressivité après action. Le Fold n'a pas de sizing ;
le Check a un sizing nul explicite. Toutes les mesures relatives proviennent de
montants exacts observés par les adaptateurs ; aucun montant absolu redondant
n'est conservé dans `ObservedAction`. Aucun rang, aucune couleur, aucune
carte privée et aucune carte publique ne fait partie du contrat.

Les adaptateurs du moteur de jeu peuvent lire les cartes pour rejouer une main
à leur niveau, mais ne les transmettent **jamais** au pipeline comportemental.
Les tenseurs et fichiers Parquet du Behavioral Engine n'ont aucun champ de carte.
Le Parquet utilise exclusivement le schéma actuel de features ; tout jeu de
fichiers dont les colonnes ne correspondent pas à ce schéma est rejeté.
Les jeux de données précédemment produits doivent être régénérés depuis les
historiques sources, sans compatibilité avec les anciens formats.

## Quatre propriétés de référence

### 1. Continuité

Des points voisins dans `Z` doivent induire des politiques comportementales proches : pour un contexte fixé et un ensemble de mouvements légaux fixé, une petite variation de `z` ne doit pas provoquer un changement arbitrairement discontinu des probabilités.

C'est un **objectif qualitatif et mesurable**, pas une garantie que donne automatiquement un réseau continu : une grande norme des poids peut rendre les variations extrêmement abruptes. Une future procédure d'entraînement devra contrôler ce phénomène et tester les distances de politiques sur des contextes échantillonnés indépendamment.

### 2. Diversité

Il doit exister des profils `z1` et `z2` tels que `G(C,z1)` et `G(C,z2)` produisent des distributions de comportements significativement différentes sur des contextes pertinents. On ne veut ni effondrement latent (`G` ignore `Z`) ni labels imposés aux coordonnées.

Les tests doivent mesurer la diversité *conditionnellement à C*, et pas uniquement la dispersion numérique des embeddings.

### 3. Cohérence cyclique

L'expression souhaitée `A(G(z)) ≈ z` doit être interprétée avec le **même contexte C** et le **même contrat d'observation**. Un Generator stochastique qui émet une seule action ne permet pas, en général, de reconstruire exactement un vecteur continu arbitraire ; plusieurs `z` peuvent expliquer une même action. La cible opérationnelle sera une reconstruction ou une inférence probabiliste calibrée de l'information effectivement identifiable.

La perte de cycle **ne suffit pas** à empêcher une collusion entre A et G, par exemple un code caché dans les détails des tailles ou du timing. Les contrôles externes et les contraintes de précision ont donc un rôle distinct du cycle.

### 4. Incertitude explicite

A doit représenter les Z compatibles avec une observation, par exemple une loi `q(z | C, a)`. Une faible quantité d'information implique une incertitude large ; l'architecture ne doit pas convertir cette incertitude en un point prétendument certain.

Cela n'oblige pas A à accepter un historique de mains : l'accumulation de plusieurs observations peut être organisée à l'extérieur de son interface élémentaire.

## Géométrie et heatmap

`Z` reste un espace continu de dimension configurable. Une carte 2D est une **projection** d'un espace éventuellement plus grand, pas une classification de joueurs. Les distances géométriques en `Z` doivent être comparées à des distances entre politiques `G(C,z)` pour des contextes définis. Une telle projection perd de l'information ; elle ne fournit pas une évaluation stratégique ou un profil « complémentaire » par elle-même.

Les cartes de densité, d'incertitude et de comportement sont distinctes des cartes d'EV, qui demanderont plus tard un évaluateur poker externe.

## Generator : architecture et entraînement

`src/behavioural/generator.rs` contient **une seule implémentation** du Generator.

1. Un MLP partagé reçoit `DecisionContext` et le profil continu `Z`, sans carte ni action cible. Les valeurs numériques non négatives sont compressées par `log1p` et un masque distinct préserve les valeurs absentes.
2. La tête **action** produit cinq logits (`Fold`, `Check`, `Call`, `Bet`, `Raise`). `LegalActionSpace`, fourni par le moteur de jeu, masque strictement les actions illégales avant le softmax. Le modèle ne reconstruit pas les règles NLHE.
3. La tête **sizing**, conditionnée à l'action Bet/ Raise, produit trois paramètres : position et échelle d'une logistique tronquée sur l'intervalle des mises non all-in, et probabilité discrète d'all-in. L'espace des mises est défini par `WagerBounds` : montant minimal, stack maximal, incrément légal en **jetons entiers** et taille de la BB en jetons. Les probabilités sont calculées pour **chaque intervalle d'un incrément légal** via la différence des CDF, pas en scorant une liste arbitraire de quelques bet sizes. La dernière unité (stack maximal) est un all-in atomique. Une relance all-in courte est représentée par une borne minimale égale à la maximale, si le moteur l'autorise.
4. La tête **timing** reçoit l'état caché, l'action et la fraction de stack engagée. Elle prédit une loi normale sur `log(1 + decision_time_ms)` ; l'échantillonnage produit un nombre de millisecondes non négatif.
5. Le `GeneratorDraws` possède des tirages reproductibles distincts pour action, sizing, masse all-in et timing. `ObservedAction::from_exact` reconstruit les ratios et les streaks depuis le contexte, donc `Fold` garde un sizing absent et `Check` un sizing nul.
6. `Adam::train_step` calcule la NLL masquée de l'action, la NLL du sizing discrétisé uniquement pour Bet/Raise, et la NLL du timing lorsqu'il est observé ; puis effectue une **rétropropagation analytique** (reverse-mode) dans les quatre couches et une mise à jour Adam. Les deux représentations des ratios du sizing ne constituent jamais deux cibles indépendantes.
7. `TrainingLoss.latent_gradient` expose le gradient par rapport à Z. `GeneratorTrainer` l’utilise maintenant pour rétropropager dans A grâce à la reparamétrisation de la distribution latente, sans conserver de vecteurs Z optimisés par identifiant de joueur.

### Contrats obligatoires

- Le moteur externe fournit le masque exact et les bornes de chaque mise ou relance à chaque décision. Le Generator rejette une observation incompatible, plutôt que d'inférer des règles manquantes.
- Les bounds utilisent des montants exacts en jetons entiers ; la conversion BB ne sert qu'à la représentation comportementale. Le montant max représente l'all-in. Pour le NLHE, il égale le stack disponible.
- La loss temporelle est définie sur les millisecondes transformées par `log1p`, pas sur les millisecondes brutes.
- G est optimisé par lot de décisions d'un sujet et A par sujet, avec deux optimiseurs Adam. Le trainer apprend depuis les mains query en inférant Z exclusivement depuis les mains de support, puis évalue sur une troisième partition future sans gradients.
- Le code d’apprentissage est en Rust standard : rétropropagation analytique et Adam, sans backend externe d’autodiff. Le type blueprint Z0 est présent, mais son préentraînement hiérarchique n'est pas encore implémenté.

## Analyzer probabiliste et entraînement A/G

`src/behavioural/analyzer.rs` contient un **unique** `NeuralBehaviourAnalyzer`.
Chaque action du sujet dans les mains de support est encodée à partir des features Parquet, en ignorant explicitement les actions des autres joueurs pour la statistique comportementale du sujet. Ces événements, uniquement issus du passé, passent par un MLP partagé ; leurs représentations sont agrégées par moyenne, puis une tête fournit `(mu, std)` pour la loi diagonale `q(Z | support)`. A n'a accès ni aux cartes, ni aux identifiants de joueur, ni aux actions query/validation. `BehaviourPosterior::sample(epsilon)` réalise la reparamétrisation déterministe en fonction d'un tirage normal fourni ; l'écart-type est strictement positif et borné par construction.

`GeneratorTrainer` optimise G et A ensemble :

1. `support` : historique passé utilisé **uniquement par A** pour inférer `q(Z | support)` ;
2. `query` : décisions futures servant de supervision à G et aux gradients vers A ;
3. `evaluation` : mains encore plus futures, **jamais utilisées pour les gradients**. Les frontières sont globales par `hand_index` pour éviter la fuite inter-joueurs.

Pour chaque sujet et chaque époque, un Z est tiré de la distribution A via une source normale déterministe. G optimise la NLL des décisions query (action, sizing, timing) sous les espaces légaux réels ; les gradients remontent jusqu'aux paramètres de A. A reçoit également une pénalité KL contre le prior `N(0,I)` pondérée par `beta_kl` explicite. Cette pénalité régularise le latent et évite que sa variance tende arbitrairement vers zéro ; elle ne garantit pas que Z sera effectivement exploité par G.

La validation calcule : NLL par décision au **posterior moyen**, NLL de référence pour `Z=0`, NLL marginale par Monte-Carlo en échantillonnant la distribution q, variance moyenne des probabilités d'action sous q, calibration ECE à 10 classes pour l'action au posterior moyen, KL moyen, écart-type moyen de q, divergence de Jensen-Shannon entre politiques de deux joueurs sous **le même contexte** et divergence locale de continuité en perturbant Z. La diversité n'est rapportée que si au moins deux sujets sont sélectionnés ; une absence de comparaison n'est jamais remplacée par un zéro fictif. `--evaluate-unseen-behavioural` applique exactement la même évaluation à une sélection de sujets absents de l'entraînement, en rejetant les sujets précédemment entraînés. Les métriques de validation n'optimisent jamais de paramètres.

Toutes les contraintes légales proviennent du **replay original** `rs-poker` : le Parquet ne permet pas d'inférer exactement les règles après un all-in court. Tout espace légal manquant ou invalide est une erreur ; l'époque échoue atomiquement sans changer le trainer. Le fingerprint déterministe du dataset, les frontières globales et la sélection des sujets sont conservés dans le checkpoint ; un dataset différent est rejeté lors de la reprise et de l'évaluation.

## Commandes de bout en bout

La configuration est un **fichier JSON strict** : chaque champ ci-dessous est obligatoire, les champs inconnus sont refusés. Les valeurs sont un exemple de syntaxe, **pas des hyperparamètres optimisés pour un corpus réel** :

```json
{
  "latent_dimension": 8,
  "generator_hidden": 64,
  "analyzer_hidden": 64,
  "generator_lr": 0.0005,
  "analyzer_lr": 0.0005,
  "beta_kl": 0.01,
  "seed": 42,
  "validation_draws": 16,
  "continuity_delta": 0.001,
  "first_query_hand_index": 400,
  "first_evaluation_hand_index": 700,
  "support_hands": 20,
  "query_hands": 10,
  "evaluation_hands": 10,
  "epochs": 20,
  "subject_ids": [1, 2]
}
```

Ces bornes et nombres de mains doivent être ajustés aux **comptages réels** du dataset ; aucune main n'est fabriquée ni dupliquée pour compléter une partition.

```bash
cargo run --release -- --train-behavioural run.json data/parquet model.json session.csv session-id
cargo run --release -- --evaluate-behavioural model.json data/parquet session.csv session-id
cargo run --release -- --resume-behavioural model.json data/parquet model-next.json 10 session.csv session-id
cargo run --release -- --evaluate-unseen-behavioural model.json data/parquet 3,4 session.csv session-id
```

Plusieurs fichiers sources sont acceptés par couples `chemin session-id`. Les sources PokerNow sont rejouées pour obtenir les bornes légales exactes. Une main source rejetée est exclue en entier et signalée. Une décision conservée sans légalité source exacte, ou des entrées incohérentes déjà persistées, restent des erreurs bloquantes. Les chemins de checkpoints existants ne sont jamais écrasés. Le fichier stocke A, G, les deux optimiseurs, leurs moments, l'état de l'aléatoire, les hyperparamètres, le découpage et la provenance déterministe du dataset.

**État de validation** : aucun entraînement réel ni résultat de performance n'est fourni avec ce changement. Les commandes ci-dessus ont besoin du Parquet et des sources réelles du projet. La CI doit vérifier la compilation, Clippy et les tests unitaires avant utilisation. Les diagnostics du latent ne remplacent pas une validation statistique sur plusieurs jeux de données et joueurs inédits.

## Contrôles requis

`cargo fmt --all --check`, `cargo clippy --locked --all-targets --all-features -- -D warnings` et `cargo test --locked --all-targets --all-features` doivent réussir avant intégration.

Les tests du Generator vérifient le masque légal, la normalisation, les incréments de sizing, la masse all-in, `Fold=None`, `Check=0`, la correspondance des gradients de sizing aux différences finies, la baisse de la loss sous Adam et la présence du gradient jusqu'à Z. Les tests ne remplacent pas la validation empirique sur des mains jamais vues.

## Phases distinctes

1. **Blueprint** : apprentissage externe de la stratégie de référence (non réalisé).
2. **A/G** : l'Analyzer probabiliste, le Generator, leur entraînement conjoint et la validation sur mains futures sont implémentés. La capacité à généraliser reste à mesurer empiriquement.
3. **Inférence** : A transforme un historique de support en distribution latente ; G transforme un contexte et un Z échantillonné en distributions d'action, sizing et timing. Les décisions stratégiques GTO ou contre-stratégiques restent hors périmètre.
