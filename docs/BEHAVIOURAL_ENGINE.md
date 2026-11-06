# Behavioral Engine — invariants et Generator

Statut : **conception et noyau d'inférence non entraîné**. Aucun blueprint de référence n'est disponible à ce stade. L'entraînement du blueprint est un chantier indépendant, et ne doit pas être confondu avec l'inférence effectuée par le Behavioral Engine.

## Objectif

Le Behavioral Engine apprend un **espace latent comportemental continu** `Z`, **sans labels de joueurs** et sans profils discrets prédéfinis (TAG, LAG, etc.).

- **Generator G** : `G(contexte comportemental C, z) -> distribution de comportements observables`. Génère les probabilités d'actions, de tailles de mises et de temps de décision. Tirer une `ObservedAction` de cette distribution est une opération distincte.
- **Analyzer A** (à développer) : `A(C, ObservedAction) -> distribution de probabilité sur Z`. Une observation isolée ne doit pas être traitée comme l'identification certaine d'un Z unique.
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

## Generator : périmètre actuel

`src/behavioural/generator.rs` est un **noyau d'inférence avec poids explicites** :

1. Encode `DecisionContext` (street, position, valeurs en BB, masque des valeurs absentes), **sans cartes**, et `BehaviourProfile` (latent Z).
2. Évalue par un petit MLP chaque **mouvement légal candidat** fourni par le moteur de jeu.
3. Renvoie une distribution normalisée sur ces mouvements, plus les paramètres d'une distribution log-normale de timing.
4. Échantillonne un `ObservedAction` uniquement parmi les mouvements fournis, avec des valeurs uniformes explicites, reproductibles.

**Aucun poids entraîné, aucune boucle d'entraînement, aucune autodifférentiation, aucun blueprint, aucun Analyzer et aucun plugin rs-poker de production ne sont inclus.** Il faudra notamment remplacer ou prolonger le scoring sur candidats discrets pour représenter plus finement les sizings continus. Un candidat « légal » est ici déclaré par l'appelant : la validation intégrale de sa légalité appartient aux règles du jeu. Le Generator contrôle seulement les invariants numériques et certaines cohérences de montants ; il ne reconstruit pas les règles NLHE à partir de `DecisionContext`.

L'interface `BehaviourGenerator::predict(profile, context, legal_moves)` ne reçoit que le contexte courant. Aucun historique de mains n'entre dans G.

## Points d'attention avant entraînement de G

- La sortie probabiliste de G doit rester différentiable **par rapport à Z** pendant l'entraînement futur de A/G. Le noyau actuel fait uniquement l'inférence CPU en Rust standard ; il ne fournit **pas** encore de gradient.
- La quantification des tailles de mises et la granularité de timing doivent respecter le format observable. Ne jamais encoder silencieusement Z dans des décimales non observables.
- Une politique sur les seuls candidats fournis ne couvre que ces candidats. La génération exhaustive des sizings légaux et son coût restent à traiter avec le moteur.
- Des tests de validité, de continuité, de diversité, d'incertitude et de sensibilité à Z seront nécessaires **après** l'entraînement ; des tests unitaires de forme ne démontrent pas ces propriétés.
- La relation entre le blueprint futur et l'espace Z (déviation conditionnelle ou autre conditionnement) reste à définir. Il ne faut pas imposer aujourd'hui une sémantique non décidée aux coordonnées de Z.

## Phases distinctes

1. **Blueprint** : apprentissage externe de la stratégie de référence (non réalisé).
2. **A/G** : entraînement de l'espace latent comportemental et des deux modèles, en contrôlant les raccourcis/collusions (non réalisé).
3. **Inférence** : A infère des informations sur Z depuis des observations, G transforme C et Z en distribution d'actions (noyau G commencé seulement).

Ce document décrit les invariants de conception, et distingue explicitement les fonctionnalités disponibles des objectifs à venir.
