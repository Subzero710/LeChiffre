# LeChiffre — Behavioral Engine hiérarchique Z₀ / Z₁ / Z₂

LeChiffre dispose d'un **seul Analyzer A**, d'un **seul Generator G** et d'un **même espace latent** pour les trois niveaux. Les entraînements A/G historiques, la CLI qui les pilotait et les anciens checkpoints ont été supprimés. **Aucune migration silencieuse ni repli sur l'ancien trainer n'est autorisé.**

## Signification des niveaux

- **Z₀ (blueprint)** : vecteur latent global entraîné à partir des mains humaines antérieures à la frontière `first_query_hand_index`. Le vecteur est appris conjointement avec G. Son écart-type de référence reste un hyperparamètre explicite ; ce n'est pas une incertitude populationnelle calibrée.
- **Z₁ (baseline individuelle)** : vecteur initialisé **exactement à Z₀**. Il est adapté après chaque main observée du joueur, avec G fixé, en utilisant la vraisemblance jointe action, sizing et timing. Ses précisions diagonales accumulent une approximation empirique de l'information. La distance de Z₁ à Z₀ n'est pas forcée à croître.
- **Z₂ (état instantané)** : distribution inférée par A pour une décision donnée. Le **prior** ne lit que les caractéristiques avant action et Z₁ ; le **posterior** peut lire `ObservedAction` après l'action. Ces deux chemins partagent les mêmes paramètres de A. La reconstruction posterior n'est jamais présentée comme une prédiction prospective.

Aucune carte privée, carte publique, nom de joueur ou identifiant n'entre dans A ou G. L'identifiant pseudonyme sert seulement au regroupement des mains et à la persistance des profils.

## Entraînement

Le seul trainer est `HierarchicalTrainer` dans `src/datasets/behaviour_parquet_loader/hierarchical_training.rs`.

1. **Blueprint** : `train_blueprint_epoch()` met à jour Z₀ et G sur le passé de la population.
2. **Personnalisation** : `train_personalization_epoch()` construit Z₁ avec `history_hands` mains strictement antérieures à la frontière query, puis entraîne G sur les décisions query. Les longueurs sont explicitement variées d'une époque à l'autre : zéro est autorisé et signifie Z₁=Z₀.
3. **Momentané** : `train_momentary_epoch()` entraîne A et G à partir du prior, du posterior et d'une KL conditionnelle, toujours à paramètres fixes pendant la construction d'un batch. Le prior ne lit pas l'action actuelle.

La séparation est **globale par `hand_index`** (support < query < validation). Une main rejetée à l'import ou structurellement invalide ne fournit aucun événement. Toute décision exploitée requiert un `LegalActionSpace` issu du replay exact `rs-poker`, jamais inféré à partir de l'action observée.

### Configuration explicite

`configs/hierarchical.example.json` contient **toutes les clés obligatoires**. Il s'agit d'un exemple de syntaxe, non d'hyperparamètres optimisés pour des données réelles : les frontières, comptes de mains et pseudonymes doivent être remplacés par les valeurs de ton dataset. Toute clé inconnue, clé manquante ou champ de longueur incorrecte entraîne une erreur.

```bash
cargo run --release -- --train-behavioural configs/hierarchical.example.json data/parquet model-v1.json source.csv session-id
```

Le fichier de checkpoint n'est écrit qu'après l'exécution des phases et la validation prospective. L'application refuse de remplacer un chemin existant.

### Validation et sujets inédits

```bash
cargo run --release -- --evaluate-behavioural model-v1.json data/parquet source.csv session-id
cargo run --release -- --evaluate-unseen-behavioural model-v1.json data/parquet 3,4 source.csv session-id
```

La validation distingue sans ambiguïté quatre scores : `z0_predictive_nll`, `z1_predictive_nll`, `z2_prior_predictive_nll` et `z2_posterior_reconstruction_nll`. Le dernier score est une **reconstruction après observation**, pas une prédiction. Une baisse de ces scores ou une CI verte ne suffit pas à démontrer l'absence de posterior collapse ni la qualité humaine du latent. Les sujets de la commande `--evaluate-unseen-behavioural` doivent être absents du jeu de sujets entraînés.

### Sauvegarde et reprise

Le format unique est `lechiffre.hierarchical.checkpoint.v1`. Il sauvegarde Z₀, A, G, les états Adam complets, les paramètres d'adaptation, les trois compteurs d'époques, l'état du flux pseudo-aléatoire, le découpage global, la liste des sujets et l'empreinte exacte du dataset. Le modèle est relié à un identifiant déterministe (`model_id`) fondé sur ses paramètres ; l'empreinte n'est pas une signature cryptographique.

```bash
cargo run --release -- --resume-behavioural model-v1.json data/parquet model-v2.json configs/hierarchical-resume.example.json source.csv session-id
```

La configuration de reprise est stricte. Les phases ne peuvent pas être rejouées dans le mauvais ordre. Les changements de dataset, de sujets ou de frontières sont rejetés. Le checkpoint `lechiffre.ag.checkpoint` est **incompatible et explicitement refusé**.

## Persistance des baselines Z₁

Chaque `PlayerBaselineState` possède le vecteur, l'incertitude, la précision diagonale, les compteurs de mains/décisions, la dernière main exploitée et une frontière globale **exclusive** `processed_before_hand_index`. Ces informations sont stockées à part du modèle global dans le format `lechiffre.player_baseline.v1`.

```bash
# Première construction (depuis Z₀, avant la main d'index 800)
cargo run --release -- --create-player-baseline model-v1.json data/parquet 42 800 player42-v1.json source.csv session-id

# Mise à jour uniquement sur les mains nouvelles d'index < 1000
cargo run --release -- --advance-player-baseline model-v1.json data/parquet 42 1000 player42-v1.json player42-v2.json source.csv session-id
```

Les profils sont liés au **model_id exact** et au `subject_id` pseudonyme. Toute discordance est bloquante. Le nouvel état est écrit atomiquement dans un fichier inédit : ni l'entrée, ni le modèle, ni les poids A/G ne sont modifiés. Une erreur de légalité ou de décodage annule l'intégralité de la mise à jour. Le dernier index déjà couvert interdit de rejouer les mêmes mains. Un autre modèle global nécessite une personnalisation distincte ; les coordonnées Z₁ ne sont pas interchangeables entre versions.

Z₂ n'est pas un profil permanent. Il se calcule lors de l'observation d'une décision ou lorsqu'un prior est demandé avant celle-ci. Son archivage n'est pas nécessaire à la persistance de Z₁.

## Code maintenu, code supprimé

- Conservés : `GeneratorGradientBatch`, les têtes action/sizing/timing, `PlayerBaselineAdapter`, `ExactLegalityProvider`, la validation du replay exact, les adaptateurs sources et le Parquet strict.
- Nouveaux : `training_data.rs` (décodage et découpage partagés), checkpoints hiérarchiques versionnés, CLI hiérarchique, états Z₁ persistants.
- Supprimés : `generator_training.rs`, `GeneratorTrainer`, l'agrégation `Analyzer::infer(support)` pour l'apprentissage et `AnalyzerAdam::train_step(support)`, les commandes reliées à ces implémentations, la prise en charge des anciens checkpoints.

Le même `G(C, Z)` génère les distributions d'actions légales, tailles de mise et temps de décision. Les montants autorisés restent définis en unités de jetons exactes par le moteur NLHE.

## Contrôles d'intégration

```bash
cargo fmt --all -- --check
cargo clippy --locked --all-targets --all-features -- -D warnings
cargo test --locked --all-targets --all-features
cargo test --locked --doc --all-features
```

Un résultat CI positif prouve seulement que les contrôles automatisés du dépôt réussissent. Une validation sur des mains réelles retenues pour test doit encore démontrer la valeur prédictive de Z₁/Z₂, les diagnostics de collapse et la calibration du modèle.
