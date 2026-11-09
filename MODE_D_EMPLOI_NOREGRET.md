# NoRegret.zip — tables graphiques et comparaison des blueprints

Décompresser à la racine de LeChiffre. Exécuter les commandes ci-dessous
depuis cette racine, dans PowerShell. Le nouveau lanceur utilise les vrais
agents et le tableau de bord fournis par rs_poker. NoRegrets joue son blueprint,
sans recherche stratégique en ligne. Aucun changement du dépôt Git distant
n'est effectué par cette archive.

## Une configuration JSON par agent

Les configurations disponibles sont dans `configs/agents/`. Copier dans un
dossier de sélection exactement **2 à 6 fichiers `.json`**, un par siège.
L'ordre alphabétique des fichiers détermine les sièges. Les noms restent
distincts dans les courbes, les historiques et les rapports ; les doublons
reçoivent un suffixe de siège. Les fichiers `.json.example` sont ignorés.

Exemple : six joueurs, NoRegrets par défaut et cinq agents existants.

```powershell
New-Item -ItemType Directory -Force configs/demo-noregrets | Out-Null
```

```powershell
"noregrets_default","calling","folding","all_in","random_default","random_pot_control" | ForEach-Object { Copy-Item "configs/agents/$_.json" -Destination "configs/demo-noregrets/" }
```

```powershell
cargo run --release --locked --manifest-path crates/lechiffre-noregrets/Cargo.toml --features gui --bin lechiffre-arena -- table configs/demo-noregrets --hands 6000 --tui --out runs/demo-noregrets
```

Il y a ici **6 000 mains au total**, avec les six mêmes sièges, le bouton tournant
et des tapis réinitialisés à chaque main. Ce lanceur n'effectue pas les 720
permutations de `rsp arena compare`. `--hands` doit être un multiple du nombre
de joueurs et couvrir au moins deux rotations du bouton.

Utiliser un nouveau dossier `--out` à chaque expérience. Les résultats sont
enregistrés dans `table.csv`, `table.json` et `hands.jsonl`. Les rapports complets
ne sont écrits qu'après les mains demandées ; une interruption conserve les
mains déjà enregistrées, sans présenter un rapport partiel comme terminé.

Tab change de panneau ; les flèches sélectionnent une ligne ; Entrée dans
l'historique ouvre une main avec les cartes et les actions. Échap revient au
tableau de bord ; Ctrl+C quitte. Agrandir le terminal pour lire les noms longs.
`--no-tui` active l'affichage texte et conserve les mêmes fichiers de sortie.

## Ajouter un NoRegrets déjà entraîné

Créer `configs/agents/noregrets_v2.json`, par exemple :

```json
{
  "type": "noregrets",
  "name": "NoRegrets-100k",
  "reference": "runs/noregrets-100k"
}
```

Remplacer `reference` par le véritable dossier de ton entraînement. Il doit
contenir **`agent-reference.json` et `blueprint.bin`**, produits par `train-agent`.
Les chemins relatifs sont interprétés depuis le répertoire où la commande est
lancée. Préférer `/` dans les chemins JSON, y compris sous Windows.
Un checkpoint neural SD-CFR ou la référence du duel réduit n'est pas un modèle
Agent NoRegrets compatible : leurs règles et leur format diffèrent.

Pour comparer NoRegrets par défaut et ta version avec quatre agents existants,
utiliser un **nouveau dossier** `configs/duel-versions` :

```powershell
New-Item -ItemType Directory -Force configs/duel-versions | Out-Null
```

```powershell
"noregrets_default","noregrets_v2","calling","folding","random_default","random_pot_control" | ForEach-Object { Copy-Item "configs/agents/$_.json" -Destination "configs/duel-versions/" }
```

```powershell
cargo run --release --locked --manifest-path crates/lechiffre-noregrets/Cargo.toml --features gui --bin lechiffre-arena -- table configs/duel-versions --hands 6000 --tui --out runs/duel-versions
```

Créer autant de configurations nommées que nécessaire : v1, v2, v3… Une table
peut contenir uniquement des versions NoRegrets, uniquement des agents rs_poker,
ou un mélange. Pour modifier la sélection, retirer les anciens JSON du dossier
de sélection ou créer un nouveau dossier. Le catalogue `configs/agents/` contient
plus de six profils et n'est donc pas directement une table.

## Compatibilité des modèles et modèle par défaut

Les blueprints d'une même table doivent correspondre au **même nombre de
joueurs, aux mêmes tapis et aux mêmes blindes**. Le lanceur vérifie les
métadonnées et les SHA-256 avant le jeu et déduit le tapis des modèles lorsque
`--stack` est absent. Les blindes sont fixées à 1/2 jetons.

`noregrets_default.json` choisit automatiquement la référence fournie adaptée :

| Joueurs sélectionnés | Référence par défaut | Tapis |
|---|---|---|
| 2 | `references/noregrets-hu10bb-agent-10000-seed42` | 10 BB |
| 6 | `references/noregrets-6max-10bb-agent-10000-seed42` | 10 BB |
| 3, 4 ou 5 | Pas de référence fournie : indiquer ton propre dossier | Selon ton modèle |

`--reference chemin` remplace le défaut pour les fichiers NoRegrets sans
`reference`. Une référence explicite dans un JSON est prioritaire. Chaque modèle
fourni a été entraîné ici à partir du code public ; il n'est pas un modèle
préentraîné commercial de l'auteur. Le budget fourni reste limité à 10 000
parcours et ne garantit pas une stratégie très forte.

Un modèle personnel à **100 BB** ne peut pas être comparé au défaut à **10 BB**
dans la même table. Utiliser deux modèles à 100 BB, ou entraîner ta version à
10 BB pour la comparer au défaut. Pour entraîner un second modèle six joueurs
à 10 BB, compatible avec celui fourni :

```powershell
cargo run --release --locked --manifest-path crates/lechiffre-noregrets/Cargo.toml -- train-agent --players 6 --stack 20 --iterations 100000 --seed 43 --out runs/noregrets-100k
```

## Lire les comparaisons

Le tableau de bord fournit les statistiques natives de rs_poker et les courbes
de profit. `table.csv` donne les gains en bb/100 et leurs IC95 approximatifs,
calculés sur les moyennes des rotations complètes du bouton. `table.json`
conserve aussi les configurations, métadonnées et hashes des modèles, et les
compteurs d'inconnus et de décisions après mise hors menu.

Les places fixes et le bouton tournant permettent un premier test comparable,
mais plusieurs graines et davantage de mains sont nécessaires pour un
classement fiable. À six joueurs, les gains dépendent de tous les adversaires.
L'exploitabilité exacte n'est pas calculée. Les variantes CFR utilisent un
estimateur `uniform` ; leurs tirages internes ne sont pas tous contrôlés par la
graine de la table.

## Intégration et branche principale

La CLI native `rsp` n'a pas de type NoRegrets. Le binaire supplémentaire
`lechiffre-arena` utilise `NoRegretsAgent` et réutilise directement les sources
de sa TUI. Il se compile avec la fonctionnalité optionnelle `gui`. Les anciennes
commandes `lechiffre-noregrets train-agent`, `table`, `match-agents`, etc. restent
disponibles, et les sources du tableau de bord natif ne sont pas modifiées.

Le 9 novembre 2026, la branche principale accessible est `master`
(`dbc6067615a042c28484cac8dd66a4a878556e61`), et la branche de travail est
`feature/ia_gto` (`e278cc529d714ddbae6c60a132eb0ff9fe35dd98`). Les deux fichiers
rs_poker corrigés sont identiques dans ces deux versions avant la modification :

- `vendor/rs-poker/src/arena/cfr/agent/engine.rs` : le fast-forward à la racine
  utilise désormais les mains tirées par l'estimateur, et non les vraies mains.
- `vendor/rs-poker/src/arena/hand_estimator/estimators.rs` : l'estimateur uniforme
  retire aussi l'information privée des mains couchées lors du tirage.

Ces corrections changent les résultats des comparaisons contenant CFR. Les
anciens classements impliquant ces agents doivent être refaits. Les tests
vérifient que changer les cartes adverses réelles ne modifie pas le calcul à
modèle adverse fixé, y compris lorsque des joueurs sont couchés.
