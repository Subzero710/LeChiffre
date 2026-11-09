# Note du 9 novembre 2026

La correction du fast-forward CFR est incluse dans `NoRegret.zip`. Les anciens
scores de duels impliquant CFR doivent être refaits ; le réglage `uniform`
seul ne masquait pas ses cartes dans le fast-forward à la racine. Les résultats
contre les autres agents restent des mesures historiques de leurs entraînements.

# Validation NoRegrets comme Agent rs_poker — 30 octobre 2026

Le nouveau `NoRegretsAgent` implémente réellement `rs_poker::arena::Agent`.
Les adversaires sont les classes construites par `ConfigAgentBuilder` et les
mains tournent dans `HoldemSimulation`. Aucun remplacement par les anciens
robots `passive` / `aggressive` n'est effectué.

## Référence distincte

`references/noregrets-hu10bb-agent-10000-seed42/` : 10 000 parcours du NoRegrets
public fixé au commit `757f7692738069522195d2b486eec60a8b010c0c`, graine42,
un thread. HU, stack20, blindes1/2, sans rake, **relances libres** et menu
Pluribus d'origine. Blueprint seul ; pas de pruning ni recherche en ligne.

SHA-256 du blueprint :
`e54eb25a0571d2ca5b64bc4cbb290fb9c502d12fc81af03dabe688226bd96bea`.

## Vérifications

Les dix tests du module passent : sept du duel réduit, trois du raccordement
rs_poker. Ces trois derniers vérifient :

- Vrais matchs contre RandomAgent, CallingAgent, AllInAgent, FoldingAgent,
  RandomPotControlAgent et les deux variantes CFRAgent basic/simple : huit
  mains par profil avec échange des places, sans divergence de l'adaptateur.
- Utilisation dans un `Box<dyn Agent>`, historique public branché par
  `Agent::historian`, exclusion des cartes privées adverses, rejet des
  historiques manquants et des règles incompatibles ; vrais matchs aussi
  avec le bouton au siège1.
- Reproductibilité de matchs seedés contre RandomAgent, séparation des
  formats de sauvegarde réduit/natif, et estimateur `uniform` des agents CFR.

L'exemple `no_regrets_vs_random` compile et s'exécute ; le test enregistré
se termine avec des tapis `[22, 18]`. La compilation release et Clippy
`--all-targets -- -D warnings` du module réussissent (avertissement amont
préexistant conservé).

## Matchs de contrôle

Un match de **10 000 mains contre le véritable RandomAgent**, graine123,
référence ci-dessus, donne pour NoRegrets : **+73,110 bb/100**, IC95 approximatif
**[+64,462 ; +81,758]**. 15 814 décisions de NoRegrets, 814 infosets inconnus ;
4 588 décisions après une mise hors menu. Aucune erreur de l'adaptateur.

Les sept profils ont également été exécutés sur 100 mains chacun. Ce petit
budget vérifie le fonctionnement, sans conclusion sur leur classement :
voir `rs-all-100.csv/json`. Le budget CFR est le fast-forward natif avec une
deadline50ms ; l'estimateur est `uniform`. Les tirages internes CFR ne sont
pas contrôlés par la seule graine du match.

Les fichiers `rs-random-10000.csv/json` conservent le premier match.
Les gains sont ceux de **NoRegrets**, et non ceux du SD-CFR. Les deux
expériences utilisent des règles différentes du duel réduit à cap1 : leurs
scores ne sont pas directement comparables.

Les mises hors menu gardent leur vrai montant dans rs_poker et dans le vrai
état NoRegrets. Son état abstrait emploie la projection de `Table` fournie
par NoRegrets. Les propres décisions déjà choisies conservent leurs tokens
abstraits pendant le rejeu. Un historique incohérent ou une configuration
incompatible arrête explicitement l'agent ; aucun repli silencieux ne vient
fabriquer un résultat valide.
