# LeChiffre — feuille de route neuronale

| Étape | État | Résultat |
| --- | --- | --- |
| 1. MLP Rust | Fait | Couches cachées, sorties signées, rétropropagation et Adam |
| 2. SD-CFR sur Kuhn | Fait | Self-play, MCCFR, réservoirs et stratégie moyenne |
| 3. Validation de Kuhn | Fait | Gradients, équilibre analytique, meilleures réponses et checkpoints |
| 4. Leduc | Fait | Deux tours, carte publique, actions masquées, MLP 43 → 64 → 64 → 3, évaluateur exact et oracle indépendant |
| 5. Adaptateur `rs_poker` | À faire | État de recherche complet et clonable, observations sûres, historique et montants corrects |
| 6. NLHE heads-up réduit | À faire | Tailles de mises fixées, parcours jusqu'aux gains terminaux et premiers entraînements |
| 7. Performance | À faire | Mesurer l'inférence par lots, le backend Burn/GPU, la mémoire et les archives |
| 8. Évaluation NLHE | À faire | Matchs appariés, bb/100 et intervalles de confiance ; exploitabilité seulement si calculable |
| 9. Multiway | À faire | Validation empirique avec limites théoriques explicites |
| 10. Cerveau Psychologue | À faire | Module séparé et comparaison avec la stratégie de référence |

La validation de Leduc comprend la comparaison des 9 450 états après distribution privée avec OpenSpiel, des tests des informations cachées et de l'historique, la mesure exacte de l'exploitabilité ainsi que la reprise reproductible. La qualité apprise reste mesurée ; elle ne constitue pas une solution exacte ni une preuve de généralisation au Texas Hold'em.

La commande `match` compare maintenant les stratégies moyennes sur Kuhn et Leduc : robots aléatoire/passif/agressif, oracle tabulaire, même stratégie ou autre checkpoint. Les distributions sont rejouées avec échange des places ; le programme mesure gains, incertitude sur les paires et espérance exacte dans ces petits jeux.

Prochaine tâche : inspecter et adapter le parcours du moteur `rs_poker`. L'interface `Game` conserve l'état complet pour la simulation, tandis que le MLP reçoit seulement `Observation`. Il faudra conserver le paquet restant, les événements de hasard et l'historique public, puis vérifier les mises et les gains sur des fixtures. `AgentAction::Bet` représente une contribution totale sur la street. Le `CFRAgent` fourni par le moteur n'est pas utilisé comme réseau neuronal ni comme source des cibles.

Le code s'ajoute dans `crates/lechiffre-sdcfr/`. Les changements de l'archive concernent ce module et ces documents ; aucune publication sur GitHub n'est effectuée.
