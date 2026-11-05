# LeChiffre — feuille de route neuronale

| Étape | État | Résultat |
| --- | --- | --- |
| 1. MLP Rust | Fait | Couches cachées, sorties signées, rétropropagation et Adam |
| 2. SD-CFR sur Kuhn | Fait | Self-play, MCCFR, réservoirs et stratégie moyenne |
| 3. Validation de Kuhn | Fait | Gradients, équilibre analytique, meilleures réponses et checkpoints |
| 4. Leduc | Fait | Deux tours, carte publique, actions masquées, MLP 43 → 64 → 64 → 3, évaluateur exact et oracle indépendant |
| 5. Adaptateur `rs_poker` | Fait pour la réduction | GameState clonable, cartes et mises du moteur, observations sûres, parcours jusqu'au showdown |
| 6. NLHE heads-up réduit | Prototype fonctionnel | 52 cartes, 10 BB, quatre streets, six sorties masquées et cap d'une mise/relance volontaire par street |
| 7. Performance | À faire | Mesurer l'inférence par lots, le backend Burn/GPU, la mémoire et les archives |
| 8. Évaluation NLHE | Référence NoRegrets ajoutée | Matchs appariés contre le véritable blueprint NoRegrets adapté au jeu réduit, d'autres checkpoints et les profils historiques ; bb/100, intervalles et couverture des infosets ; exploitabilité exacte indisponible |
| 9. Multiway | À faire | Validation empirique avec limites théoriques explicites |
| 10. Cerveau Psychologue | À faire | Module séparé et comparaison avec la stratégie de référence |

La validation de Leduc comprend la comparaison des 9 450 états après distribution privée avec OpenSpiel, des tests des informations cachées et de l'historique, la mesure exacte de l'exploitabilité ainsi que la reprise reproductible. La qualité apprise reste mesurée ; elle ne constitue pas une solution exacte ni une preuve de généralisation au Texas Hold'em.

La commande `match` compare maintenant les stratégies moyennes sur Kuhn et Leduc : robots aléatoire/passif/agressif, oracle tabulaire, même stratégie ou autre checkpoint. Les distributions sont rejouées avec échange des places ; le programme mesure gains, incertitude sur les paires et espérance exacte dans ces petits jeux.

Le nouveau module se trouve dans `crates/lechiffre-nlhe/` ; le cœur neuronal partagé reste dans `crates/lechiffre-sdcfr/`. L'interface `Game` conserve l'état complet pour la simulation, tandis que le MLP reçoit seulement `Observation`. Le paquet restant, les événements de hasard et l'historique public sont conservés. Les montants et les gains sont vérifiés contre OpenSpiel et le simulateur complet de `rs_poker`. `AgentAction::Bet` représente une contribution totale sur la street. Le `CFRAgent` fourni par le moteur n'est pas utilisé pour produire les cibles.

Prochaine tâche : améliorer les budgets et la précision des réseaux, mesurer leur généralisation sur des distributions séparées, puis étendre les tapis et le menu de mises. Il faut adapter le stockage des archives avant les longs entraînements. Le prototype réduit ne certifie pas une stratégie GTO sur le NLHE libre.

La référence NoRegrets publique est intégrée dans `vendor/noregrets/`, avec
sa licence MIT et son commit source. `crates/lechiffre-noregrets/` entraîne
son vrai MCCFR tabulaire et compare son blueprint au SD-CFR sur les règles
réduites communes. Voir `LIRE-MOI_NOREGRETS.md`. Le SD-CFR demeure une
implémentation indépendante ; cette intégration ne le convertit pas en
l'architecture complète NoRegrets avec recherche en ligne.

NoRegrets dispose aussi maintenant d'un véritable adaptateur `Agent`
rs_poker et de `match-agents`, pour le comparer aux agents déjà présents
dans le moteur. Une référence distincte à relances libres est incluse.
Voir `TESTER_NOREGRETS_CONTRE_AGENTS.md`.

La commande `table` permet maintenant de choisir individuellement 2 à 6
agents existants, avec bouton tournant et bilan par siège. Le NoRegrets
tabulaire dispose d'une référence six joueurs distincte. Le réseau SD-CFR
reste HU : cette extension de table ne constitue pas un SD-CFR multiway.
Voir `CHOISIR_LES_JOUEURS.md` et `configs/table-6.json`.

Le code se trouve dans `crates/lechiffre-sdcfr/` et `crates/lechiffre-nlhe/`. L'archive contient ces modules et les documents ; aucun fichier du moteur ni du programme racine n'est remplacé, et aucune publication sur GitHub n'est effectuée.
