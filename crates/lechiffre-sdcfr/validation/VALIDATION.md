# Validation du 22 octobre 2026

Environnement : Linux x86_64, Rust 1.99.1. Commandes testées depuis la racine du projet.

Paramètres : 128 itérations, 512 parcours par joueur et par itération, 256 étapes Adam, batch 64, réservoir de 50 000 échantillons par joueur, MLP 7 → 32 → 32 → 2, taux 0,005.

| Graine | Exploitabilité initiale | Exploitabilité après 128 itérations | Valeur du premier joueur |
| --- | ---: | ---: | ---: |
| 42 | 0,458333 | 0.023384 | -0.056133 |
| 43 | 0,458333 | 0.024173 | -0.057213 |
| 44 | 0,458333 | 0.020743 | -0.054750 |

Unité : jeton d’ante par main ; exploitabilité = NashConv/2. Il subsiste une exploitabilité mesurable : ce réseau constitue une première validation, pas une solution exacte de Kuhn ni un agent NLHE validé.

Référence indépendante : oracle CFR tabulaire complet, 10 000 itérations ; exploitabilité 0,002641537, valeur du premier joueur −0,055540689. Les budgets et méthodes diffèrent : ces chiffres ne sont pas une comparaison de vitesse ou de qualité à coût égal.

Contrôles réussis :

- 16 tests : gradients numériques, MSE pondérée, masque des actions, apprentissage de cibles signées, regroupement exact des observations, moyenne avec poids de reach, tirage des anciens réseaux, informations cachées, gains, Regret Matching, équilibre analytique, oracle, cibles MCCFR, réservoir, reprise et mesures de regret.
- `cargo fmt --check`.
- `cargo clippy --all-targets -- -D warnings`.
- Construction en mode release et trois entraînements complets.
- Chargement et évaluation de la sauvegarde finale.

Les CSV joints proviennent des exécutions réelles. La colonne `seconds` dépend de cet environnement et inclut calculs, évaluations et écritures ; ce n’est pas un benchmark matériel reproductible.

La validation des modifications du moteur rs_poker et les tests de son programme principal restent hors du périmètre de ce module autonome.
