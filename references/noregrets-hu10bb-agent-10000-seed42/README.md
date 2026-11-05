# NoRegrets Agent — HU 10 BB, relances libres

Référence entraînée ici depuis zéro avec le véritable NoRegrets public,
commit `757f7692738069522195d2b486eec60a8b010c0c` (MIT).

10 000 parcours, graine42, un thread ; deux joueurs, tapis20, blindes1/2,
sans rake, relances libres. Menu Pluribus, 12 buckets postflop, équité64,
sans centroids ni pruning ni recherche en ligne.

Les fichiers `agent-reference.json` et `blueprint.bin` doivent rester
ensemble. Cette référence est destinée à `match-agents` et au vrai
`NoRegretsAgent`, et non au duel SD-CFR à cap d'une relance.

Génération : `train-agent --iterations 10000 --seed 42 --stack 20 --out dossier`.
Ce modèle n'est pas fourni par l'auteur et son niveau n'est pas certifié GTO.
