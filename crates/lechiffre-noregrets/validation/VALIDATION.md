# Validation de la comparaison NoRegrets — 30 octobre 2026

## Sources et périmètre

- LeChiffre : branche `feature/ia_gto`, commit
  `37c535446c3932354898f0e96129c6f48b863576`.
- NoRegrets public : commit `757f7692738069522195d2b486eec60a8b010c0c`, MIT.
- Environnement de validation : Linux x86_64, Rust/Cargo 1.99.1.
- Les anciens modules SD-CFR et NLHE sont identiques aux fichiers lus sur
  cette branche. Aucun changement de leurs modèles ou de leurs checkpoints.
- Quatre ajouts/adaptations aux sources NoRegrets sont détaillés dans
  `vendor/noregrets/LECHIFFRE_UPSTREAM.md`. Les autres sources originales,
  leur licence et leur ancien Cargo.lock sont conservés.

L'entraînement appelle `noregrets::cfr::Trainer`. L'adversaire utilise
`noregrets::bot::Policy::blueprint_dist`. Ce n'est pas une réécriture d'un
robot passif ou agressif. Le mode comparatif est limité au blueprint sur le
menu réduit commun, sans pruning ni recherche en ligne.

## Tests exécutés

Les **46 tests LeChiffre** passent : 30 du cœur SD-CFR, 9 du NLHE et 7 de
l'intégration NoRegrets. Ces derniers vérifient :

1. Rejeu des 400 distributions et 1 147 décisions OpenSpiel déjà fournies :
   même acteur, mêmes contributions, mêmes menus complets et mêmes gains
   dans rs_poker et NoRegrets. Il n'y a aucune projection approximative des
   montants de mises.
2. 200 autres mains aléatoires, avec vérification des deux moteurs à chaque
   décision et au règlement terminal.
3. Bijection du codage des 52 cartes, préservant rangs et couleurs.
4. Invariance des décisions de la référence quand seules les cartes adverses
   ou les futures cartes du tableau changent, préflop et postflop.
5. Entraînement réel amont, sauvegarde/chargement du blueprint, contrôle de
   SHA-256, rejet des règles incompatibles et refus d'écraser un dossier.
6. Matchs appariés reproductibles, cohérence des gains et rejet des mains
   impaires ainsi que des checkpoints d'un autre jeu.
7. Rejet des budgets nuls et des paramètres de regroupement incohérents.

**24 tests ciblés NoRegrets** passent aussi : 13 du moteur, 2 des cartes,
3 de l'évaluateur de mains, 1 du Regret Matching et 5 des menus/montants
d'origine. La suite complète amont de 136 tests n'a pas été exécutée.
Le binaire original NoRegrets passe `cargo check --locked --bins`.

Le nouveau module passe `cargo fmt` et `cargo clippy --all-targets -- -D warnings`.
La compilation release du nouveau programme réussit. Des avertissements
préexistants du code amont sont conservés et ne sont pas présentés comme
des avertissements introduits par l'adaptateur.

## Entraînements et match exécutés

Référence NoRegrets incluse :

- 10 000 parcours, graine 42, un thread.
- 12 buckets postflop, 64 simulations d'équité, sans centroids.
- HU 10 BB, blindes 1/2, cap 1 et menu de LeChiffre.
- 12 818 infosets, blueprint de 479 735 octets.
- SHA-256 : `7f7d20ab486f38878457b64cbb9e3d0c3b2c0ab97458ad373230e415a7754ae2`.
- Environ 1,33 s d'entraînement sur cet environnement, compilation exclue.

SD-CFR mesuré : un **nouvel entraînement local de 16 itérations**, graine42,
paramètres par défaut : 8 parcours/joueur/itération, 64 étapes Adam, largeur64,
batch64, mémoire5000. Environ 9,66 s hors compilation. Son checkpoint n'est
pas celui de 64 itérations de l'utilisateur et n'est pas inclus.

Match : 10 000 mains, 5 000 paires, graine de simulation123.

| Mesure | Résultat du SD-CFR face à cette référence |
| --- | ---: |
| Gain | −24,465 bb/100 |
| IC95 approximatif | [−33,625 ; −15,305] |
| Total | −2 446,50 BB |

| Street | Décisions de NoRegrets | Replis check/call pour infoset inconnu |
| --- | ---: | ---: |
| Préflop | 9 751 | 20 |
| Flop | 4 845 | 0 |
| Turn | 2 936 | 2 |
| River | 1 653 | 14 |

Le SD-CFR perd ce match de contrôle. Ce résultat ne mesure ni le checkpoint
64 de l'utilisateur, ni les modèles commerciaux de l'auteur, ni une version
avec recherche en ligne. Les budgets sont différents ; ce n'est pas une
comparaison des algorithmes à temps de calcul égal.

Les fichiers `matches.csv`, `comparison.json` et `sdcfr-train-16.csv` conservent
les mesures de cet essai. L'exploitabilité exacte du NLHE n'est pas calculée.
