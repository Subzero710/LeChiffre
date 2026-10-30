# Validation des matchs — 22 octobre 2026

Environnement : Linux x86_64, Rust 1.99.1. Les matchs utilisent des stratégies figées ; ils n'entraînent ni ne modifient les réseaux.

La sauvegarde principale est le réseau Leduc entraîné 128 itérations avec la graine 42 dans la validation précédente. La simulation utilise la graine 123 et 100 000 mains par adversaire, regroupées en 50 000 paires avec échange des places. Les mêmes cartes et flux aléatoires par siège sont réutilisés entre les deux mains d'une paire. L'erreur standard est calculée sur les moyennes de ces paires, avec réduction en ligne de Welford et intervalle normal approximatif à 95 %.

| Adversaire | Gain simulé, jetons/100 mains | IC95 approximatif | Espérance exacte, jetons/100 mains |
| --- | ---: | --- | ---: |
| Aléatoire | +85,727 | [+83,640 ; +87,814] | +86,368 |
| Passif | +75,036 | [+73,824 ; +76,248] | +74,890 |
| Agressif | +68,334 | [+65,125 ; +71,543] | +68,609 |
| Oracle tabulaire, 1 000 itérations | −10,133 | [−11,293 ; −8,973] | −10,631 |

Les résultats complets sont dans `matches-leduc.csv`. Le total correspond aux gains nets de l'IA principale sur les 100 000 mains, dont la moitié à chaque place. La stratégie moyenne est calculée avec les poids d'itération et de reach propre, puis utilisée comme politique comportementale pour jouer.

Le même réseau a été opposé à celui entraîné avec la graine 43, également pendant 128 itérations : −1,944 jeton/100 mains, IC95 [−2,987 ; −0,901], espérance exacte −2,158. Le gain concerne le réseau de graine 42. Les données figurent dans `matches-neural.csv`.

Les mêmes commandes ont été exécutées sur la sauvegarde Kuhn de la première livraison. Le fichier `matches-kuhn.csv` conserve les mesures. Face à l'oracle de 1 000 itérations, ce réseau obtient +0,065 jeton/100 mains simulées, IC95 [−0,055 ; +0,185], pour une espérance exacte de +0,011.

## Contrôles

- 30 tests passent : les 24 tests précédents et six nouveaux tests des matchs et des statistiques.
- Deux politiques identiques produisent un gain moyen exact nul. Avec les flux aléatoires réutilisés et les agents échangés, chaque paire identique s'annule aussi exactement dans la simulation.
- L'échange des deux agents inverse le signe des gains et de l'intervalle.
- Les moyennes simulées concordent avec les espérances calculées par énumération exacte sur Kuhn et Leduc.
- Les profils adverses respectent les actions légales ; les résultats sont reproductibles avec une même graine.
- Les nombres de mains invalides, noms d'adversaires inconnus et options de checkpoint incohérentes sont rejetés.
- `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings` et construction release passent.

L'espérance exacte est la moyenne à parts égales du gain à chaque siège contre l'adversaire indiqué. Cette mesure concerne le duel entre ces deux politiques, tandis que l'exploitabilité utilise les meilleures réponses possibles. Les intervalles concernent la simulation de mains, et ne couvrent pas la variabilité entre entraînements. Le mode exact est limité aux petits jeux énumérables de cette livraison.
