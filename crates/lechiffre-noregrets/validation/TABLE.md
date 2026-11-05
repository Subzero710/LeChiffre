# Table configurable 2 à 6 joueurs — validation du 30 octobre 2026

Le NoRegrets public conserve son moteur et son véritable entraînement MCCFR.
L'adaptateur `Agent` généralise la rotation des sièges, les positions des
cartes dans le paquet reconstruit et les contrôles d'argent à 2..6 joueurs.
Le blueprint est associé au nombre de joueurs dans les métadonnées et à un
format v2 pour le multiway. Les anciens fichiers HU v1 sans ce champ restent
lisibles comme références à deux joueurs. Un modèle d'un autre nombre de
joueurs est refusé avant de jouer.

## Vérifications exécutées

Les **14 tests du module** passent : 7 du duel réduit, 3 de l'interface Agent
et 4 de la table configurable. Les nouveaux tests comprennent :

- 120 mains à six sièges, avec deux NoRegrets séparés, RandomAgent,
  CallingAgent, AllInAgent et FoldingAgent, en parcourant toutes les positions
  du bouton ; répétition seedée et mêmes gains/intervalles.
- Tables de 2 à 6 joueurs avec les agents existants, y compris une table où
  les deux variantes CFR basic/simple jouent ensemble.
- NoRegrets entraîné séparément pour 3, 4 et 5 joueurs et testé dans chacun
  de ces jeux avec rotation du bouton.
- Gains sans rake à somme nulle, rejet des nombres de sièges invalides, des
  modèles à mauvais nombre de joueurs, des mains ne couvrant pas des cycles
  complets et vérification de compatibilité des anciennes références HU.

Clippy `--all-targets -- -D warnings` et la compilation release du module
réussissent ; l'avertissement amont déjà documenté reste conservé.

## Référence six joueurs fournie

`references/noregrets-6max-10bb-agent-10000-seed42/` : 10 000 parcours,
graine42, un thread ; 6 joueurs, stack20, blindes1/2, sans rake, relances
libres. Blueprint seul, 12 buckets postflop, équité64, sans centroids,
pruning ou recherche en ligne. Ce budget sert de point de départ et ne
certifie pas une stratégie de qualité élevée ou un équilibre multiway.

SHA-256 : `263928f216707efd518e53ae975d62e92d4480695c4b9b32f245c8b449679b2c`.
Taille du blueprint : 5 657 370 octets.

## Test réel de la configuration fournie

Commande `table --table-config configs/table-6.json --hands 6000`, graine123.
Les agents restent à leur siège ; le bouton tourne. Les intervalles sont
calculés sur 1 000 cycles de six mains, sans rejeu apparié des cartes.

| Siège | Agent | Gain bb/100 | IC95 approximatif |
| --- | --- | ---: | --- |
| 1 | NoRegrets | +11,500 | [−24,159 ; +47,159] |
| 2 | RandomAgent | −57,500 | [−90,657 ; −24,343] |
| 3 | CallingAgent | −1,758 | [−42,757 ; +39,241] |
| 4 | AllInAgent | −29,492 | [−69,060 ; +10,076] |
| 5 | CFRAgent basic | +82,325 | [+50,234 ; +114,416] |
| 6 | RandomPotControlAgent | −5,075 | [−15,073 ; +4,923] |

Les tirages internes de CFR ne sont pas déterminés par la seule graine de
la table, donc le classement exact peut changer lors d'une répétition.
NoRegrets a rencontré 2 257 infosets inconnus sur 7 435 décisions ;
636 décisions ont suivi une mise hors menu. Ses gains restent incertains
dans ce test. Le résultat ne prouve pas sa supériorité.

Les fichiers `table-six-6000.csv/json` conservent les résultats et les
configurations effectives. L'exploitabilité exacte n'est pas calculée.
