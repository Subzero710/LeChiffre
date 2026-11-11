# NoRegret — entraîner un modèle et comparer toutes les positions

Extraire `NoRegret.zip` à la racine de LeChiffre, sur `feature/ia_gto`.
Exécuter les commandes ci-dessous depuis cette racine, dans PowerShell.
L'adaptateur est conservé : le lanceur reste `crates/lechiffre-noregrets/Cargo.toml`.
Il utilise les agents, le simulateur, le comparateur et le tableau de bord de rs_poker.
Les anciens modules SD-CFR/NLHE ne sont plus nécessaires. Aucun push n'est effectué.

## 1. Entraîner ta version six joueurs à 10 BB

```powershell
cargo run --release --locked --manifest-path crates/lechiffre-noregrets/Cargo.toml -- train-agent --players 6 --stack 20 --iterations 100000 --seed 43 --out runs/noregrets-v2
```

- `--players 6` : modèle entraîné pour six joueurs.
- `--stack 20` : 20 jetons avec une BB de 2, donc **10 BB** par joueur.
- `--iterations 100000` : budget de parcours MCCFR, pas nombre de mains de test.
- `--seed 43` : graine d'entraînement.
- `--out` : nouveau dossier contenant `agent-reference.json` et `blueprint.bin`.

Attendre la fin avant d'utiliser le modèle. Cet entraînement repart de zéro ;
ce n'est pas une reprise du modèle fourni. Pour une autre version, changer le
nom et le dossier de sortie. Davantage de parcours ne garantit pas un meilleur
score contre chaque adversaire : il faut mesurer le résultat.

Vérifier le modèle sauvegardé :

```powershell
cargo run --release --locked --manifest-path crates/lechiffre-noregrets/Cargo.toml -- inspect --reference runs/noregrets-v2
```

## 2. Créer son profil JSON

Après l'entraînement, créer `configs/agents/noregrets_v2.json` :

```json
{
  "type": "noregrets",
  "name": "NoRegrets-v2-100k",
  "reference": "runs/noregrets-v2"
}
```

Le fichier `noregrets_v2.json.example` fourni peut servir de modèle : retirer
`.example` après avoir terminé l'entraînement correspondant.
`reference` désigne le dossier réel, pas le fichier `blueprint.bin`.
Les chemins sont relatifs à la racine où tu exécutes la commande ; utiliser `/`
dans les JSON sous Windows. Un fichier `.json.example` n'est jamais chargé.
Pour un modèle déjà entraîné ailleurs, indiquer son vrai dossier contenant les
deux fichiers ci-dessus. Les checkpoints neuronaux et les références du duel
réduit ont un autre format.

## 3. Choisir les six profils

Créer un **nouveau dossier de sélection** et copier les six JSON souhaités :

```powershell
New-Item -ItemType Directory -Force configs/comparaison-v2 | Out-Null
"noregrets_default","noregrets_v2","calling","folding","random_default","random_pot_control" | ForEach-Object { Copy-Item "configs/agents/$_.json" -Destination "configs/comparaison-v2/" }
```

Il contient deux versions NoRegrets et quatre agents rs_poker existants.
Pour changer un joueur, changer les fichiers du dossier de sélection. Utiliser
un nom distinct par profil. L'ordre des fichiers ne fixe plus leurs voisins :
le mode `compare` permute les sièges.

Pour tester seulement le modèle fourni et cinq agents existants, sans entraîner :

```powershell
New-Item -ItemType Directory -Force configs/comparaison-defaut | Out-Null
"noregrets_default","calling","folding","all_in","random_default","random_pot_control" | ForEach-Object { Copy-Item "configs/agents/$_.json" -Destination "configs/comparaison-defaut/" }
```

## 4. Lancer le comparateur natif avec l'affichage graphique

```powershell
cargo run --release --locked --manifest-path crates/lechiffre-noregrets/Cargo.toml --features gui --bin lechiffre-arena -- compare configs/comparaison-v2 -p 6 -n 10 --seed 123 --tui --out runs/comparaison-v2-seed123
```

Pour le test sans entraînement, remplacer `configs/comparaison-v2` par
`configs/comparaison-defaut` et choisir un autre dossier `--out`.

- `-p 6` : six places à chaque table.
- `-n 10` : dix séries d'états initiaux ; chacune parcourt toutes les permutations.
- Six profils à six places donnent **720 permutations × 10 = 7 200 mains**.
- Chaque profil joue 7 200 mains, dont 1 200 à chaque siège.
- `--seed` fixe les tirages du comparateur et des agents qui acceptent sa graine.
- `--no-tui` remplace `--tui` pour un lancement texte avec les mêmes résultats.

Pour environ 72 000 mains, utiliser `-n 100` et un nouveau `--out`.
On peut fournir plus de profils que de places : sept profils à six places
donnent **5 040 permutations par série**. Le total est affiché avant la partie.
Les modèles doivent être entraînés pour **six joueurs**, même avec sept profils.

Le comparateur réutilise un état initial par série et change les affectations
aux sièges ; les distributions de cartes utilisent des sous-flux aléatoires
distincts entre permutations. Ce n'est pas un test où toutes les mains futures
sont identiques pour chaque permutation. Plusieurs graines restent utiles.

Tab change de panneau ; les flèches sélectionnent une ligne ; Entrée dans
l'historique ouvre une main ; Échap revient ; Ctrl+C quitte.
Utiliser un nouveau dossier `--out` à chaque lancement. Après interruption,
les mains déjà jouées restent consultables ; aucun rapport complet n'est écrit.

## 5. Lire les résultats

Le gain final affiché dans le terminal est en **bb/100**. Les statistiques
proviennent de `rs_poker::arena::comparison::ArenaComparison`, regroupées par
profil sur toutes ses positions.

| Fichier dans le dossier de sortie | Contenu |
|---|---|
| `results.md` | Rapport natif, classement, gains et statistiques par position |
| `results.json` | Statistiques natives par nom d'agent ; montants en jetons |
| `comparison.json` | Profils effectifs, graine, protocole, métadonnées et SHA-256 des modèles |
| `hands.jsonl` | Historique des mains pour la TUI et la consultation détaillée |

Dans `results.json`, `profit_per_100_hands` est en jetons/100 : diviser par
**2** pour obtenir les bb/100 de cette configuration. Le mode natif ne calcule
pas d'IC95 apparié ni l'exploitabilité exacte. Un petit test ne prouve pas
qu'une stratégie est GTO ou meilleure contre tous les adversaires.
Les profils CFR utilisent `uniform` pour ignorer les cartes privées adverses ;
leurs budgets explicites sont conservés. Sans budget explicite, le défaut est
50 ms avec fast-forward. Tous leurs tirages internes ne sont pas contrôlés
par la graine du comparateur.

## Compatibilité et anciens modes

Le modèle fourni par `noregrets_default.json` est choisi selon **`-p`** :

| Places à table | Référence par défaut | Tapis |
|---|---|---|
| 2 | `references/noregrets-hu10bb-agent-10000-seed42` | 10 BB |
| 6 | `references/noregrets-6max-10bb-agent-10000-seed42` | 10 BB |
| 3, 4 ou 5 | Fournir un modèle entraîné pour ce nombre de joueurs | Selon le modèle |

Tous les NoRegrets d'une comparaison doivent avoir le même nombre de joueurs,
le même tapis et les mêmes blindes. Blindes fixées à 1/2, sans ante ni rake,
relances libres. Le tapis est déduit des modèles ; `--stack` le précise en jetons.
Les modèles et leurs hashes sont vérifiés avant le lancement. Un modèle à
100 BB ne peut pas jouer avec celui fourni à 10 BB.
`--reference dossier` fournit le défaut aux profils sans `reference` ; un chemin
explicite dans un JSON est prioritaire. Les références fournies ont été
entraînées ici depuis le code public, pas publiées préentraînées par l'auteur.
NoRegrets joue son blueprint, sans recherche stratégique en ligne.

Les anciens modes `table` restent disponibles : sièges fixes et bouton tournant,
rapports `table.csv/json`, IC95 approximatifs et compteurs NoRegrets.
Utiliser **`compare` pour équilibrer aussi les positions relatives**.
`train-agent`, `match-agents` et `inspect` restent disponibles sans `gui`.
La CLI native `rsp` seule ne lit pas les JSON NoRegrets : le lanceur de
l'adaptateur charge les modèles et les injecte dans le comparateur natif.
Sa boucle de permutations, ses statistiques et sa TUI sont réutilisées.
Voir `crates/lechiffre-noregrets/validation/COMPARATEUR_NATIF.md` pour les
changements communs à rs_poker et le suivi de `master`.
