# Alignement de NoRegrets sur Equity dans master

Sources consultées par le connecteur GitHub en lecture seule :

- feature/ia_gto : `3ccd95ce30120971f4fafafeb7bc654145a63bd8`.
- master : `7d84f49185f2e78743f6acd27043271c5a1e3b6f`.

Les fichiers de configuration des agents et le constructeur de rotation de
master ont été revérifiés. Equity existe, l'ancien agent de contrôle du pot
a été retiré. Le réintroduire dans l'archive précédente n'était pas le choix
souhaité : cette réintroduction est annulée.

## Modifications

- Suppression de l'import, de la variante et du constructeur RandomPotControl
  dans la configuration native ; suppression de son module et de son export.
- Suppression des branches correspondantes dans les constructeurs de comparaison,
  rotation et charts. Equity reste l'agent fourni par rs_poker, avec ses réglages
  et son implémentation inchangés.
- Le raccourci `match-agents --opponent equity` et la liste `--opponent all`
  utilisent désormais Equity. Les tests d'agents, de versions et de tables
  utilisent également Equity. Les anciens profils ne sont pas convertis
  silencieusement : sélectionner un profil de type `equity`.
- Les correctifs Generated, relances libres et fixtures CFR restent présents.
  KnownHandsEstimator n'est forcé que dans les deux fixtures qui exigent des
  mains fixes. Les comparaisons de l'adaptateur gardent l'estimateur uniforme.
- La sélection de deux callers est corrigée dans les instructions : deux
  fichiers avec des noms de fichiers distincts, puis noms d'agents distingués
  dans les rapports. Un nouveau test couvre le refus de cinq fichiers pour
  six sièges et l'acceptation des six profils après ajout de `calling_2.json`.

## Validation

Avec Rust 1.99.1 dans cet environnement Linux :

- 18 tests de l'adaptateur passent avec `--locked --features gui`.
- Les 1 071 tests de la bibliothèque rs_poker passent ; aucun n'est ignoré.
- Compilation release des deux binaires et compilation du binaire natif rsp
  avec les fonctionnalités `rsp,arbitrary` et `--locked`.
- Clippy de l'adaptateur, tous les targets et GUI, passe avec `-D warnings`.
- Un modèle six joueurs de validation à 64 parcours a été entraîné, chargé,
  puis utilisé dans une vraie comparaison native avec Equity à 1 000 simulations,
  deux Calling, Folding et Random : 720 mains, 120 par siège pour chaque profil,
  deux noms distincts pour les callers et gains à somme nulle.

Les rapports `equity-six-720-*` sont joints comme preuve d'exécution. Ce petit
modèle n'est pas un modèle de jeu recommandé et ce test n'établit aucune force
GTO. Equity et CFR possèdent des tirages internes non entièrement contrôlés
par la graine du comparateur.

Cette version n'inclut plus le fichier de l'ancien agent. Si une ancienne
archive l'avait ajouté au projet local, il peut être supprimé avec l'ancien
profil de catalogue ; les commandes sont données dans le guide courant.
Les sélections personnelles et chemins des modèles ne sont pas écrasés.
Aucun push ni modification du dépôt distant n'a été effectué.
