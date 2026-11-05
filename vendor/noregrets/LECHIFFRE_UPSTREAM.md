# NoRegrets intégré à LeChiffre

Source : https://github.com/conorarmstrong/noregrets

Commit public fixé : `757f7692738069522195d2b486eec60a8b010c0c`.
Licence MIT, copyright Conor Armstrong, conservée dans `LICENSE`.

Les 43 fichiers d'origine ont été récupérés à ce commit et leurs Git blob
SHA-1 ont été vérifiés avant modification. `UPSTREAM_SHA256.json` conserve
également leurs SHA-256 et tailles **avant adaptation**. Ce fichier ne décrit
pas les octets modifiés ci-dessous.

## Modifications locales limitées

- Ajout de `src/lib.rs` : expose les modules pour un appel direct depuis Rust.
- `src/abstraction.rs` : ajout de `MenuShape::LeChiffreReduced` et de l'action
  `MinRaise`, token 11. Menu identique à LeChiffre : fold, check/call, minimum,
  demi-pot arrondi vers le haut, pot, tapis ; doublons éliminés ; une seule
  mise/relance volontaire par street. Les variantes préexistantes de menu et
  leurs tokens restent inchangés. Les variantes sérialisées sont ajoutées en
  fin d'enum, sans déplacer les variantes d'origine.
- `src/cfr.rs` : reconnaît aussi `MinRaise` comme relance pour les continuations
  biaisées ; le parcours MCCFR et les mises à jour des regrets sont conservés.
- `src/table.rs` : gère cette action avec les autres relances lorsque le moteur
  interdit toute relance.

Les autres fichiers d'origine, notamment `Cargo.toml`, `Cargo.lock`,
`engine.rs`, `bot.rs`, les solveurs, les réseaux et `LICENSE`, sont inchangés.
Le binaire d'origine se lance séparément :

```sh
cargo run --release --locked --manifest-path vendor/noregrets/Cargo.toml -- --help
```

Les indications de performances du README et de BASELINES viennent de
l'auteur ; elles ne constituent pas une mesure de cette intégration ou de
nos petits entraînements. Les stratégies et réseaux préentraînés de l'auteur
ne sont pas inclus dans le dépôt public et ne sont pas fournis ici.

## Mode de comparaison

`crates/lechiffre-noregrets/` appelle le véritable `cfr::Trainer`, puis la
véritable `bot::Policy::blueprint_dist`. Il entraîne une nouvelle référence
heads-up 10 BB, sans pruning, sur le menu réduit commun. Ce mode n'active
pas la recherche en ligne ni le réseau de valeurs de NoRegrets : ces chemins
utilisent d'autres menus et nécessiteraient leur propre adaptation/validation.
Il ne faut donc pas présenter ce match comme un duel contre la configuration
complète de NoRegrets ou contre le Pluribus original.

Le moteur rs_poker du SD-CFR reste actif. Le moteur NoRegrets suit exactement
la même main en parallèle ; acteurs, cartes publiques, contributions, mises
légales et gains sont contrôlés. Une divergence arrête le match. Aucune mise
n'est remplacée silencieusement par une taille approchante.
