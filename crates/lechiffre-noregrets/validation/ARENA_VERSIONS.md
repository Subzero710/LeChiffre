# Validation : tables graphiques et versions NoRegrets

9 novembre 2026. Voir `MODE_D_EMPLOI_NOREGRET.md` à la racine.

- Les 16 tests du module couvrent le chargement des profils rs_poker et de deux
  blueprints effectivement distincts (64 et 128 parcours, graines différentes),
  leurs noms, hashes, métadonnées, gains à somme nulle et historiques OHH.
- Référence par défaut HU chargée et jouée ; annulation après une main sans
  publier un rapport complet ; noms identiques distingués par siège ; références
  absentes et formats/nombres de joueurs/tapis incompatibles refusés.
- 13 tests d'estimation des mains et 10 tests fast-forward de rs_poker passent,
  dont les deux nouveaux tests de non-utilisation des cartes cachées réelles.
- Compilation release du nouveau binaire et Clippy avec `-D warnings` réussis ;
  avertissement amont NoRegrets `unused_mut` préexistant conservé.
- Exécution réelle en texte et en TUI de 60 mains à six joueurs, avec NoRegrets
  par défaut et les agents AllIn, Calling, Folding, RandomDefault et
  RandomPotControl. Courbes et classement affichés, une main ouverte avec ses
  cartes et son journal, puis sortie propre par Ctrl+C. Les 60 historiques et le
  rapport complet sont présents. Ce contrôle n'est pas une mesure de force.

Le nouveau lanceur joue un nombre total de mains et fait tourner le bouton ;
il ne reproduit pas le protocole factoriel `rsp arena compare`. Les noms et
les résultats des modèles restent distincts. L'exploitabilité exacte n'est pas
calculée. Aucun code n'a été poussé sur GitHub.
