# Pricer d’options par arbres recombinants

Pricer d’options européennes et américaines par **arbres binomial et trinomial**, écrit en Python et piloté depuis **Excel**. Il gère les dividendes discrets, le pruning, la méthode BBS et les grecques, et mesure le temps, la mémoire et la précision.

Projet du cours d’informatique quantitative, M2 EIF, Université Paris-Dauphine (2026-2027).
**Auteurs :** Doryan Lochet et Cloé Legrand.

---

## Ce que fait le pricer

- **Options :** call et put, européens et américains.
- **Arbres :** binomial et trinomial recombinants, avec le tronc qui suit le forward.
- **Dividendes discrets :** le trinomial absorbe le saut du sous-jacent en rebranchant chaque nœud sur la grille existante.
- **Deux constructions :**
  - **forward** : tout l’arbre est construit, puis pricé de la maturité vers la racine ;
  - **backward** : l’arbre est construit colonne par colonne en partant de la maturité, et chaque colonne est libérée dès qu’elle ne sert plus.
- **Pruning :** un nœud de probabilité inférieure à 10⁻⁷, ou à plus de 6 écarts-types du tronc, n’a plus qu’une branche.
- **Méthode BBS** (Binomial Black-Scholes, Broadie et Detemple, 1996) : le dernier pas de l’arbre est remplacé par la formule de Black-Scholes.
- **Précision visée :** l’utilisateur peut donner une précision ε en euros au lieu d’un nombre de pas ; `Model.from_precision` en déduit N.
- **Grecques :**
  - delta et gamma, lus directement sur l’arbre ;
  - vega, vomma, vanna, thêta et rho, par différences finies, avec la classe `OneDimDerivative` du cours, à qui l’on passe une fonction (prix en fonction de la vol, du taux ou de la date).
- **Visualisation :** dessin de l’arbre (valeur de l’option en couleur, probabilité d’atteinte en taille), en PNG dans Excel et en SVG à côté.

L’arbre se parcourt uniquement par les liens entre nœuds, sans liste.

## Utilisation

1. Installer les dépendances :
   ```bash
   pip install -r requirements.txt
   ```
2. Remplir les cases jaunes de l’onglet **Parametres** de `parametres_pricer.xlsx` : marché, option, dates, nombre de pas ou précision visée, type d’arbre, BBS, pruning, dividendes.
3. Enregistrer et **fermer** le fichier Excel (openpyxl écrit directement dedans).
4. Lancer la cellule `run_from_excel` de `main.ipynb`.
5. Lire les résultats dans l’onglet **Resultats** (prix, écart avec Black-Scholes, grecques, informations sur l’arbre) et le dessin dans l’onglet **Visualisation**.

Python 3.13 est recommandé : environ 0,5 s par option, contre environ 2 s en Python 3.10.

## Structure du dépôt

| Fichier | Rôle |
|---|---|
| `tree.py` | Arbre, nœuds, pricing, pruning, BBS, Black-Scholes, grecques (`OneDimDerivative`) |
| `excel_interface.py` | Lecture des paramètres et écriture des résultats dans Excel |
| `visualisation.py` | Dessin de l’arbre (PNG et SVG) |
| `etudes.py` | Les études du rapport : `generate_all()` produit les graphiques de `graphiques/` |
| `tests_tree.py` | 5 tests unitaires sur des propriétés financières |
| `main.ipynb` | Point d’entrée : pricing depuis Excel, études, tests |
| `parametres_pricer.xlsx` | Interface utilisateur |
| `rapport.pdf` | Rapport du projet |
| `graphiques/` | Figures du rapport |

## Validation

Les tests de `tests_tree.py` vérifient que l’arbre respecte cinq propriétés financières :

- un call de strike nul vaut le sous-jacent ;
- la parité call-put est respectée ;
- à 200 pas, le prix est à moins de 2 centimes de Black-Scholes ;
- un put américain vaut plus que le put européen ;
- un dividende fait baisser le call et monter le put.

Le rapport ajoute deux contrôles : forward et backward donnent exactement le même prix, et la parité call-put tient à 4·10⁻¹² près.

## Principaux résultats

Sauf mention contraire : call européen à la monnaie, S₀ = K = 100, r = 3 %, σ = 25 %, T = 1 an, sans dividende.

**Temps de calcul**
- La construction forward est 1,5 à 2 fois plus rapide que la backward, qui construit chaque nœud deux fois (un aller pour les probabilités, puis le retour).
- Le pruning divise le temps par 4 à 5 à N = 500, pour un écart de prix d’au plus 1,6·10⁻⁸ €.

**Mémoire**
- La construction backward est environ 70 fois plus économe : 1 Mo à N = 500 (croissance en N), contre 80 Mo en forward (croissance en N²).
- Ce gain suppose de libérer les nœuds à la main, car ils se pointent mutuellement : sans `cut_column`, le backward garde tout l’arbre (12 Mo à N = 200) ; avec, il reste sous 0,5 Mo.

**Précision et BBS**
- Sans BBS, l’erreur oscille avec le strike (jusqu’à environ ±0,13 € à 20 pas). Elle fait un creux chaque fois que le strike tombe sur un nœud à maturité : l’arbre arrondit les scénarios au nœud, qui paie zéro, et perd les petits gains juste au-dessus du strike.
- BBS lisse cette erreur. Le trinomial BBS reste sous 0,001 €, alors que le binomial BBS garde un biais de +0,03 à +0,05 €, soit environ 600 fois moins précis.
- Explication : les deux arbres matchent l’espérance et la variance, mais avec l’écart σ√(3Δt), les probabilités du trinomial tendent vers 1/6, 2/3, 1/6, et il matche aussi le moment d’ordre 4 de la loi normale (kurtosis de 3, contre 1 pour le binomial).
- La formule de précision visée, N = c · S₀e^(rT) · σ²T / (√(2π) · ε · √(e^(σ²T) − 1)), avec c = 1/2 en binomial et 3/8 en trinomial, est prudente : les quatre arbres atteignent toujours la précision demandée, et le trinomial BBS fait 100 à 1 000 fois mieux.

**Options américaines**
- BBS n’apporte presque rien. Le seuil d’exercice anticipé existe à toutes les dates, alors que BBS ne corrige que le dernier pas, et l’arbre ne peut exercer qu’à ses N dates : il sous-estime le prix.

**Grecques**
- La taille du choc h est un compromis : trop petit, on mesure le bruit de l’arbre ; trop grand, la différence finie est biaisée. Les chocs retenus (1 jour, 1 %, 1 point de vol) sont dans la bonne zone.
- Pour la vega et la vomma, les arbres à σ ± h gardent la grille de l’arbre de base (`grid_vol`) : seules les probabilités changent. Le binomial, dont la grille bouge avec la vol, a une vomma fausse d’un facteur 10 : il faut utiliser le trinomial.

## Conclusion

Il n’y a pas de réglage universel :
- **forward** → le plus rapide ;
- **backward** → le plus économe en mémoire ;
- **pruning** → presque gratuit en précision ;
- **trinomial BBS** → de loin le plus précis, sauf pour une option américaine ;
- **grecques** → la taille du choc est un compromis entre bruit et biais.

Le détail des études, des figures et des difficultés rencontrées est dans `rapport.pdf`.
