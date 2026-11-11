# Graphiques pour le rapport, enregistrés dans le dossier "graphiques" (numérotés dans l'ordre de generate_all).
# Graphique 1 : temps de calcul, construction forward contre backward.
# Graphique 2 : temps de calcul, sans pruning / pruning par probabilité / pruning par écarts-types.
# Graphique 3 : pic de mémoire en fonction de N, forward contre backward.
# Graphique 4 : mémoire pendant un pricing backward, avec et sans gestion de la mémoire.
# Graphique 5 : erreur (arbre - Black-Scholes) en fonction du strike, avec et sans BBS.
# Graphique 6 : détail (zoom) du graphique 5 sur le trinomial BBS, comme dans le cours.
# Graphique 7 : prix en fonction du strike avec un zoom autour du forward (coude aux nœuds sans BBS).
# Graphique 8 : précision visée (N calculé par la formule du cours) contre erreur obtenue.
# Graphique 9 : convergence avec et sans la méthode BBS (Black-Scholes sur le dernier pas).
# Graphique 10 : (arbre - Black-Scholes) x N avec et sans BBS, avec un zoom sur le trinomial BBS.
# Graphique 11 : BBS sur un put européen et américain.
# Graphique 12 : erreur des grecques (thêta, rho, vomma, vanna) en fonction de la taille du choc, N fixé.


import gc
import math
import os
import time
from datetime import date

import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, Rectangle

from tree import (Market, Model, OneDimDerivative, Option, Tree, black_scholes, black_scholes_greeks,
                  delta_vs_vol, memory_curve, peak_memory, price_tree, price_vs_days, price_vs_rate, price_vs_vol)

# ---------- paramètres : call européen à la monnaie, sans dividende (Black-Scholes donne le prix exact)
S0, K, R, VOL = 100, 100, 0.03, 0.25
PRICING_DATE, MATURITY = date(2026, 9, 1), date(2027, 9, 1)
T = (MATURITY - PRICING_DATE).days / 365
MARKET = Market(S0, R, VOL)
OPTION = Option("call", K, "european")
BS_PRICE = black_scholes(S0, K, R, VOL, T, "call")

BLUE, ORANGE = "#2a78d6", "#eb6834"                 # binomial, trinomial
LIGHT_BLUE, LIGHT_ORANGE = "#8fbcef", "#f5a98a"     # mêmes couleurs en clair (construction backward)
FOLDER = "graphiques"
TIME_STEPS = [10, 20, 50, 100, 200, 500]      # valeurs de N pour les graphiques de temps et de mémoire
TREES = (("binomial", LIGHT_BLUE, BLUE), ("trinomial", LIGHT_ORANGE, ORANGE))   # type d'arbre, couleur claire, foncée
STRIKE_TEXT = (f"call européen, S0 = {S0}, r = {R:.0%}, σ = {VOL:.0%}, T = 1 an, sans pruning, "
               "un arbre pour tous les K")


# =====================================================================
# Outils communs à tous les graphiques
# =====================================================================

def _tree_price(n: int, tree_type: str, bbs: bool = False) -> float:
    """Prix de l’arbre à n pas, sans pruning (construction forward), avec ou sans la méthode BBS."""
    model = Model(PRICING_DATE, MATURITY, n, tree_type, prune_mode="none", bbs=bbs)
    return price_tree(MARKET, model, OPTION)


def _pricing_time(n: int, tree_type: str, method: str = "forward",
                  prune_mode: str = "none") -> tuple[float, float]:
    """Temps de pricing (construction + pricing) d’un arbre à n pas, en une seule mesure.
    Renvoie (prix, temps en secondes)."""
    model = Model(PRICING_DATE, MATURITY, n, tree_type, prune_mode=prune_mode, prune_proba=1e-7, prune_std=6)
    gc.collect()                                         # on part d'une mémoire propre
    tree = Tree(MARKET, model)
    start = time.perf_counter()
    price = tree.build_and_price(OPTION, method)
    elapsed = time.perf_counter() - start
    tree.release()
    return price, elapsed


def _style(ax: plt.Axes, title: str, xlabel: str, ylabel: str, log_x: bool = False, log_y: bool = False,
           legend_loc: str = "best", legend_cols: int = 1):
    """Mise en forme commune : échelles, titre, noms des axes, grille légère et légende sans cadre."""
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(color="#e1e0d9", linewidth=0.8)
    ax.legend(frameon=False, loc=legend_loc, ncol=legend_cols)


def _save(fig: plt.Figure, name: str) -> str:
    """Enregistre la figure dans le dossier "graphiques" et renvoie le chemin."""
    os.makedirs(FOLDER, exist_ok=True)
    path = os.path.join(FOLDER, name)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _maturity_spots(tree: Tree, low: float, high: float) -> list[float]:
    """Spots des nœuds de l’arbre à maturité compris entre low et high."""
    spots = []
    node = tree.last_trunk.top()
    while node is not None:
        if low <= node.spot <= high:
            spots.append(node.spot)
        node = node.down
    return spots


def _mark_nodes(ax: plt.Axes, spots: list[float], color: str, width: float = 2) -> None:
    """Petits traits verticaux en bas du graphique, aux nœuds à maturité."""
    for spot in spots:
        ax.axvline(spot, ymax=0.04, color=color, linewidth=width)


# =====================================================================
# Erreur en fonction du strike (graphiques 5 à 7)
# =====================================================================

def _gaps_and_nodes(n: int, tree_type: str, bbs: bool, strikes: list[float]) -> tuple[list[float], list[float]]:
    """Erreur (arbre - Black-Scholes) pour chaque strike, calculée sur un seul arbre (le strike ne change pas
    l’arbre), et nœuds à maturité entre 80 et 120."""
    tree = Tree(MARKET, Model(PRICING_DATE, MATURITY, n, tree_type, prune_mode="none", bbs=bbs)).build()
    gaps = [tree.price(Option("call", k)) - black_scholes(S0, k, R, VOL, T, "call") for k in strikes]
    spots = _maturity_spots(tree, 80, 120)
    tree.release()
    return gaps, spots


def plot_gap_vs_strike(n: int = 20) -> str:
    """Graphique 5 : erreur (arbre - Black-Scholes) d’un call européen en fonction du strike, à N fixé,
    sans et avec la méthode BBS (Black-Scholes sur le dernier pas).
    Le strike ne change pas l’arbre (cours) : on construit l’arbre une fois et on price tous les strikes dessus
    (BBS ne change que le pricing du dernier pas, pas les nœuds).
    Les traits orange en bas marquent les nœuds du trinomial à maturité."""
    strikes = [80 + 0.1 * i for i in range(401)]                    # strike de 80 à 120
    fig, ax = plt.subplots(figsize=(9, 5))
    for tree_type, light, dark in TREES:
        for bbs, color, width, name in ((False, light, 1.3, "classique"), (True, dark, 1.8, "BBS")):
            gaps, spots = _gaps_and_nodes(n, tree_type, bbs, strikes)
            ax.plot(strikes, gaps, color=color, linewidth=width, label=f"{tree_type} {name}", zorder=3)
            if tree_type == "trinomial" and bbs:                    # nœuds à maturité entre 80 et 120
                _mark_nodes(ax, spots, ORANGE)
        print(f"{tree_type} : fait")
    ax.axhline(0, color="black", linewidth=0.8, zorder=2)          # sous les courbes (le trinomial BBS est ~0)
    ax.set_ylim(-0.15, 0.19)                                       # place pour la légende en haut
    _style(ax, f"Erreur (arbre − Black-Scholes) en fonction du strike, N = {n} pas, avec et sans BBS\n{STRIKE_TEXT}",
           "strike K (€)  —  traits orange en bas : nœuds du trinomial à maturité", "arbre − Black-Scholes (€)",
           legend_loc="upper center", legend_cols=4)
    return _save(fig, "05_erreur_vs_strike.png")


def plot_gap_vs_strike_zoom(n: int = 20) -> str:
    """Graphique 6 : détail du graphique 5 (comme dans le cours, "Evaluate the error (7)") : seulement le
    trinomial BBS, invisible à l’échelle du graphique 5. L’échelle des y est donnée par la courbe elle-même ;
    le facteur de zoom est le rapport entre l’erreur max du trinomial classique et celle du trinomial BBS."""
    strikes = [80 + 0.1 * i for i in range(401)]                    # strike de 80 à 120
    classic_gaps, _ = _gaps_and_nodes(n, "trinomial", False, strikes)
    gaps, spots = _gaps_and_nodes(n, "trinomial", True, strikes)
    zoom = max(abs(g) for g in classic_gaps) / max(abs(g) for g in gaps)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(strikes, gaps, color=ORANGE, linewidth=1.8, label="trinomial BBS", zorder=3)
    _mark_nodes(ax, spots, ORANGE)                                  # nœuds à maturité entre 80 et 120
    ax.axhline(0, color="black", linewidth=0.8, zorder=2)
    # cadre "Zoom x ..." comme sur la slide du cours
    ax.text(0.36, 0.95, f"Zoom × {zoom:.0f}\npar rapport au graphique 5", transform=ax.transAxes, fontsize=10,
            va="top", bbox=dict(boxstyle="square,pad=0.4", facecolor="white", edgecolor="black"))
    _style(ax, f"Détail du graphique 5 : erreur du trinomial BBS en fonction du strike, N = {n} pas\n{STRIKE_TEXT}",
           "strike K (€)  —  traits orange en bas : nœuds du trinomial à maturité", "arbre − Black-Scholes (€)",
           legend_loc="upper right")
    print("trinomial BBS (zoom) : fait")
    return _save(fig, "06_erreur_vs_strike_zoom.png")


def _kink_curves(axes: tuple[plt.Axes, plt.Axes], strike_sets: tuple[list[float], list[float]],
                 n: int) -> dict[float, str]:
    """Trace Black-Scholes et les 4 arbres (binomial / trinomial, classique / BBS) sur la vue complète et sur
    le zoom ; renvoie les nœuds à maturité situés dans la fenêtre du zoom, avec leur couleur."""
    for ax, strikes in zip(axes, strike_sets):
        ax.plot(strikes, [black_scholes(S0, k, R, VOL, T, "call") for k in strikes], color="black",
                linewidth=1.2, label="Black-Scholes", zorder=4)
    zoom = strike_sets[1]
    nodes = {}                                                      # nœuds à maturité dans la fenêtre du zoom
    for tree_type, light, dark in TREES:
        for bbs, color, name in ((False, dark, "classique"), (True, light, "BBS")):
            tree = Tree(MARKET, Model(PRICING_DATE, MATURITY, n, tree_type, prune_mode="none", bbs=bbs)).build()
            label = f"{tree_type} {name}" + (" (confondu avec Black-Scholes)" if bbs and tree_type == "trinomial"
                                             else "")
            for ax, strikes in zip(axes, strike_sets):
                ax.plot(strikes, [tree.price(Option("call", k)) for k in strikes], color=color, linewidth=1.5,
                        linestyle="-" if not bbs else "--", label=label, zorder=3)
            nodes.update({spot: dark for spot in _maturity_spots(tree, zoom[0], zoom[-1])})
            tree.release()
        print(f"{tree_type} : fait")
    return nodes


def _zoom_details(fig: plt.Figure, ax_full: plt.Axes, ax_zoom: plt.Axes, zoom: list[float],
                  nodes: dict[float, str]) -> None:
    """Sur le zoom : prix forward et nœuds à maturité (là où les courbes classiques font un coude).
    Sur la vue complète : cadre de la fenêtre du zoom, relié au graphique de droite."""
    forward = S0 * math.exp(R * T)
    ax_zoom.axvline(forward, color="grey", linewidth=1, linestyle=":")
    ax_zoom.annotate(f"prix forward\nS0 e^(rT) = {forward:.2f}", (forward, ax_zoom.get_ylim()[1]), xytext=(8, -6),
                     textcoords="offset points", va="top", fontsize=9, color="grey")
    for spot, color in nodes.items():
        _mark_nodes(ax_zoom, [spot], color, width=2.5)
    y0, y1 = ax_zoom.get_ylim()
    ax_full.add_patch(Rectangle((zoom[0], y0), zoom[-1] - zoom[0], y1 - y0, fill=False, edgecolor="grey",
                                linewidth=1, zorder=5))
    for y in (y0, y1):
        fig.add_artist(ConnectionPatch((zoom[-1], y), (zoom[0], y), "data", "data", axesA=ax_full, axesB=ax_zoom,
                                       color="grey", linewidth=0.8, linestyle="--"))


def plot_price_kink(n: int = 10, half_width: float = 1.0) -> str:
    """Graphique 7 : prix du call en fonction du strike avec seulement n pas (cours, "Evaluate the error (6)").
    À gauche, toutes les courbes sont confondues ; à droite, zoom autour du prix forward S0 e^(rT).
    Sans BBS, le prix de l’arbre est linéaire par morceaux en K : entre deux nœuds à maturité le payoff est
    une droite, et la courbe fait un coude à chaque nœud. Avec BBS, la courbe est lisse comme Black-Scholes."""
    forward = S0 * math.exp(R * T)
    full = [80 + 0.25 * i for i in range(241)]                      # strike de 80 à 140
    zoom = [forward - half_width + 0.005 * i for i in range(int(2 * half_width / 0.005) + 1)]
    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1, 1.25]})
    nodes = _kink_curves((ax_full, ax_zoom), (full, zoom), n)
    _zoom_details(fig, ax_full, ax_zoom, zoom, nodes)
    _style(ax_full, "Vue complète : toutes les courbes sont confondues", "strike K (€)", "prix du call (€)",
           legend_loc="upper right")
    _style(ax_zoom, f"Zoom autour du prix forward (± {half_width:g} €)",
           "strike K (€)  —  traits en bas : nœuds à maturité (bleu binomial, orange trinomial)", "prix du call (€)",
           legend_loc="lower left")
    ax_full.get_legend().remove()                                   # une seule légende, sur le zoom
    fig.suptitle(f"Prix en fonction du strike avec seulement N = {n} pas : sans BBS le prix fait un coude à chaque "
                 f"nœud, avec BBS il est lisse\n{STRIKE_TEXT}")
    return _save(fig, "07_prix_coude.png")


# =====================================================================
# Temps de calcul et mémoire (graphiques 1 à 4)
# =====================================================================

def plot_runtime_methods() -> str:
    """Graphique 1 : temps de pricing en fonction de N, forward contre backward, binomial et trinomial,
    sans pruning."""
    fig, ax = plt.subplots(figsize=(9, 5))
    # couleur foncée = forward, couleur claire = backward
    for tree_type, method, color in (("binomial", "forward", BLUE), ("binomial", "backward", LIGHT_BLUE),
                                     ("trinomial", "forward", ORANGE), ("trinomial", "backward", LIGHT_ORANGE)):
        times = [_pricing_time(n, tree_type, method)[1] for n in TIME_STEPS]
        ax.plot(TIME_STEPS, times, color=color, marker="o", markersize=4, label=f"{tree_type} {method}")
        print(f"{tree_type} {method} : fait")
    _style(ax, "Temps de pricing (construction + pricing) : forward contre backward\n"
           "call européen, sans pruning, une mesure par N",
           "nombre de pas N (échelle log)", "temps en secondes (échelle log)", log_x=True, log_y=True,
           legend_cols=2)
    return _save(fig, "01_temps_forward_backward.png")


def plot_runtime_pruning() -> str:
    """Graphique 2 : temps de pricing en fonction de N, sans pruning et avec les deux méthodes de pruning
    (arbre trinomial, construction forward). La légende donne l’écart de prix maximal avec l’arbre sans pruning."""
    fig, ax = plt.subplots(figsize=(9, 5))
    no_pruning = None
    for prune_mode, label, color in (("none", "sans pruning", "black"),
                                     ("proba", "pruning par probabilité (< 1E-07)", ORANGE),
                                     ("std", "pruning par écarts-types (> 6)", BLUE)):
        results = [_pricing_time(n, "trinomial", "forward", prune_mode) for n in TIME_STEPS]
        prices, times = [p for p, _ in results], [t for _, t in results]
        if no_pruning is None:
            no_pruning = prices
        else:                                                       # écart de prix avec l'arbre sans pruning
            gap = max(abs(p - q) for p, q in zip(prices, no_pruning))
            label += f", écart de prix max {gap:.1e} €"
        ax.plot(TIME_STEPS, times, color=color, marker="o", markersize=4, label=label)
        print(f"{prune_mode} : fait")
    _style(ax, "Effet du pruning sur le temps de pricing\n"
           "call européen, arbre trinomial, construction forward, une mesure par N",
           "nombre de pas N (échelle log)", "temps en secondes (échelle log)", log_x=True, log_y=True)
    return _save(fig, "02_temps_pruning.png")


def plot_peak_memory() -> str:
    """Graphique 3 : pic de mémoire d’un pricing en fonction de N, forward contre backward
    (trinomial, sans pruning, gestion de la mémoire activée). À gauche en Mo, à droite en nombre de nœuds."""
    Tree.memory_management = True
    fig, (ax_mo, ax_nodes) = plt.subplots(1, 2, figsize=(11, 4.8))
    for method, color in (("forward", ORANGE), ("backward", LIGHT_ORANGE)):
        results = [peak_memory(MARKET, Model(PRICING_DATE, MATURITY, n, "trinomial", prune_mode="none"),
                               OPTION, method) for n in TIME_STEPS]
        ax_mo.plot(TIME_STEPS, [mo for mo, _ in results], color=color, marker="o", markersize=4,
                   label=f"trinomial {method}")
        ax_nodes.plot(TIME_STEPS, [nodes for _, nodes in results], color=color, marker="o", markersize=4,
                      label=f"trinomial {method}")
        print(f"{method} : fait")
    # les deux panneaux : même axe des N, mémoire en Mo à gauche et en nombre de nœuds à droite
    for ax, ylabel in ((ax_mo, "pic de mémoire (Mo, échelle log)"),
                       (ax_nodes, "nœuds en mémoire au maximum (échelle log)")):
        _style(ax, "", "nombre de pas N (échelle log)", ylabel, log_x=True, log_y=True)
    fig.suptitle("Mémoire maximale d'un pricing : forward contre backward\n"
                 "call européen, arbre trinomial, sans pruning, gestion de la mémoire activée")
    return _save(fig, "03_memoire_forward_backward.png")


def plot_memory_management(n: int = 200) -> str:
    """Graphique 4 : mémoire mesurée après chaque colonne pendant un pricing backward de n pas,
    avec et sans gestion de la mémoire (coupure des colonnes devenues inutiles)."""
    model = Model(PRICING_DATE, MATURITY, n, "trinomial", prune_mode="none")
    fig, ax = plt.subplots(figsize=(9, 5))
    for management, label, color in ((False, "sans gestion de la mémoire", "black"),
                                     (True, "avec gestion de la mémoire (cut_column)", ORANGE)):
        tracker = memory_curve(MARKET, model, OPTION, method="backward", memory_management=management)
        ax.plot(range(1, len(tracker.mo) + 1), tracker.mo, color=color, linewidth=1.5, label=label)
        print(f"gestion de la mémoire = {management} : fait")
    # trait vertical : fin du passage aller (build_trunk), début du retour colonne par colonne
    ax.axvline(n, color="grey", linewidth=1)
    ax.text(n, ax.get_ylim()[1] * 0.75, "  fin du passage aller (build_trunk),\n  puis retour colonne par colonne",
            color="grey", va="top", fontsize=9)
    _style(ax, f"Mémoire pendant un pricing backward de {n} pas\ncall européen, arbre trinomial, sans pruning",
           "avancement du calcul (une mesure après chaque colonne construite ou pricée)",
           "mémoire utilisée (Mo)", legend_loc="center right")
    return _save(fig, "04_gestion_memoire.png")


# =====================================================================
# Précision visée (graphique 8)
# =====================================================================

def _precision_errors(tree_type: str, bbs: bool, targets: list[float]) -> tuple[list[float], list[int]]:
    """Pour chaque précision visée : N donné par la formule du cours et erreur |arbre - Black-Scholes| obtenue."""
    errors, n_used = [], []
    for target in targets:
        model = Model.from_precision(MARKET, PRICING_DATE, MATURITY, target, tree_type=tree_type, bbs=bbs)
        errors.append(abs(price_tree(MARKET, model, OPTION) - BS_PRICE))
        n_used.append(model.nb_steps)
    return errors, n_used


def _precision_axes(ax: plt.Axes, targets: list[float], n_used: dict[str, list[int]]) -> None:
    """Échelles log, précision la plus large à gauche, et sous chaque précision le N de chaque arbre."""
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-7, 1)
    ax.invert_xaxis()                                                   # de la précision la plus large à la plus fine
    ax.set_xticks(targets)
    ax.set_xticklabels([f"{t:g} €\nbin. N={nb}\ntri. N={nt}"
                        for t, nb, nt in zip(targets, n_used["binomial"], n_used["trinomial"])], fontsize=8)
    ax.minorticks_off()


def plot_precision() -> str:
    """Graphique 8 : pour chaque précision visée, N est calculé par Model.from_precision (formule d’erreur
    du cours, différente pour le binomial et le trinomial), puis on mesure l’erreur réellement obtenue
    |arbre - Black-Scholes|. La formule ne vaut que pour l’arbre classique : avec BBS, on garde le même N
    et on regarde l’erreur obtenue. Objectif : être sous la précision visée (zone grisée)."""
    targets = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002]          # précisions visées en €
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.fill_between(targets, 1e-7, targets, color="grey", alpha=0.12, linewidth=0,
                    label="zone où l'erreur est sous la précision visée")
    n_used = {}                                                         # N utilisé, par type d'arbre
    for tree_type, light, dark in TREES:
        for bbs, color, name in ((False, light, "classique"), (True, dark, "BBS")):
            errors, n_used[tree_type] = _precision_errors(tree_type, bbs, targets)
            ax.plot(targets, errors, color=color, marker="o", markersize=5, label=f"{tree_type} {name}")
            print(f"{tree_type} {name} : fait")
    _precision_axes(ax, targets, n_used)
    _style(ax, "Nombre de pas N donné par la formule du cours (binomial et trinomial) : erreur réellement obtenue\n"
           f"call européen, S0 = K = {S0}, r = {R:.0%}, σ = {VOL:.0%}, T = 1 an, pruning 1E-07 ; "
           "pas de formule pour BBS : même N que l'arbre classique",
           "précision visée (€, échelle log) et N utilisé", "erreur obtenue |arbre − Black-Scholes| (€, échelle log)",
           legend_loc="lower left", legend_cols=2)
    return _save(fig, "08_precision_visee.png")


# =====================================================================
# Méthode BBS (graphiques 9 à 11)
# =====================================================================

def plot_bbs(n_max: int = 250) -> str:
    """Graphique 9 : prix de l’arbre pour N = 10 à n_max, sans et avec BBS. Sans BBS, le coude du
    payoff en K fait zigzaguer le prix ; avec BBS, le dernier pas est lissé par Black-Scholes."""
    steps = list(range(10, n_max + 1))
    fig, ax = plt.subplots(figsize=(9, 5))

    for tree_type, light, dark in TREES:
        ax.plot(steps, [_tree_price(n, tree_type) for n in steps], color=light, linewidth=1.3,
                label=f"{tree_type} classique")
        ax.plot(steps, [_tree_price(n, tree_type, bbs=True) for n in steps], color=dark, linewidth=1.8,
                label=f"{tree_type} BBS")
        print(f"{tree_type} : fait")

    ax.axhline(BS_PRICE, color="black", linewidth=1, label=f"Black-Scholes = {BS_PRICE:.4f}")
    ax.set_ylim(BS_PRICE - 0.06, BS_PRICE + 0.12)
    _style(ax, "Méthode BBS (Black-Scholes sur le dernier pas) : le prix ne zigzague plus\n"
           f"call européen, S0 = K = {S0}, r = {R:.0%}, σ = {VOL:.0%}, T = 1 an, forward, sans pruning",
           "nombre de pas N", "prix du call (€)", legend_cols=3)
    return _save(fig, "09_bbs.png")


def plot_bbs_error_x_n(n_max: int = 500) -> str:
    """Graphique 10 : (arbre - Black-Scholes) x N, sans et avec BBS, comme sur la slide du cours.
    À gauche les 4 arbres ; à droite un zoom sur le trinomial BBS, invisible à l'échelle de gauche.
    N : tous les N de 1 à 100, puis de 5 en 5 jusqu'à n_max (les grands arbres coûtent cher).
    Pruning à 6 écarts-types et construction backward : plus rapide, sans changer l'erreur."""
    steps = list(range(1, 101)) + list(range(105, n_max + 1, 5))
    fig, (ax, ax_zoom) = plt.subplots(1, 2, figsize=(13, 5))
    for tree_type, light, dark in TREES:
        for bbs, color, name in ((False, light, "classique"), (True, dark, "BBS")):
            gaps = [(price_tree(MARKET, Model(PRICING_DATE, MATURITY, n, tree_type, prune_mode="std",
                                              prune_std=6, bbs=bbs), OPTION, method="backward") - BS_PRICE) * n
                    for n in steps]
            ax.plot(steps, gaps, color=color, linestyle="none", marker="o", markersize=2, label=f"{tree_type} {name}")
            if tree_type == "trinomial" and bbs:
                ax_zoom.plot(steps, gaps, color=color, linestyle="none", marker="o", markersize=2,
                             label="trinomial BBS")
            print(f"{tree_type} {name} : fait")
    for a in (ax, ax_zoom):
        a.axhline(0, color="black", linewidth=1)
    ax_zoom.set_ylim(-0.01, 0.01)                                      # zoom : l'échelle de gauche va de -2 à 2
    _style(ax, "Erreur × N avec et sans BBS", "nombre de pas N (échelle log)", "(arbre − Black-Scholes) × N",
           log_x=True, legend_loc="lower left", legend_cols=2)
    _style(ax_zoom, "Zoom sur le trinomial BBS", "nombre de pas N (échelle log)", "(arbre − Black-Scholes) × N",
           log_x=True, legend_loc="lower right")
    fig.suptitle(f"call européen, S0 = K = {S0}, r = {R:.0%}, σ = {VOL:.0%}, T = 1 an, pruning à 6 écarts-types")
    return _save(fig, "10_bbs_erreur_x_N.png")


# =====================================================================
# BBS sur un put américain (graphique 11)
# =====================================================================

def plot_bbs_american(n_max: int = 800, n_step: int = 20) -> str:
    """Graphique 11 : put européen (colonne de gauche) et américain (colonne de droite), arbre trinomial
    classique et BBS. En haut : prix en fonction de N. En bas : erreur x N = (prix - référence) x N.
    Référence : Black-Scholes pour l'européen ; pour l'américain, sans formule exacte, un arbre trinomial
    classique à très grand N (4000 pas), calculé une seule fois.
    Paramètres choisis pour que l'exercice anticipé compte beaucoup : put dans la monnaie (K = 110),
    taux élevé (8 %), volatilité 20 %. BBS supprime l'erreur en européen, mais pas en américain :
    la frontière d'exercice anticipé, présente à toutes les dates, reste vue en escalier par l'arbre."""
    s0, k, r, vol = 100, 110, 0.08, 0.20
    market = Market(s0, r, vol)
    steps = list(range(n_step, n_max + 1, n_step))
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    for col, exercise in enumerate(("european", "american")):
        ax_price, ax_error = axes[0, col], axes[1, col]
        option = Option("put", k, exercise)
        if exercise == "european":
            ref, ref_label = black_scholes(s0, k, r, vol, T, "put"), "Black-Scholes (prix exact)"
        else:
            ref = price_tree(market, Model(PRICING_DATE, MATURITY, 4000, "trinomial", prune_mode="std",
                                           prune_std=6), option, method="backward")
            ref_label = "référence : trinomial classique N = 4000"
        ax_price.axhline(ref, color="black", linewidth=1, linestyle="--", label=f"{ref_label} = {ref:.4f}")
        ax_error.axhline(0, color="black", linewidth=1, linestyle="--")
        for bbs, color, name in ((False, LIGHT_ORANGE, "trinomial classique"), (True, ORANGE, "trinomial BBS")):
            prices = [price_tree(market, Model(PRICING_DATE, MATURITY, n, "trinomial", prune_mode="std",
                                               prune_std=6, bbs=bbs), option, method="backward") for n in steps]
            ax_price.plot(steps, prices, color=color, marker="o", markersize=3, label=name)
            ax_error.plot(steps, [(p - ref) * n for p, n in zip(prices, steps)], color=color, marker="o",
                          markersize=3, label=name)
            print(f"put {exercise} {name} : fait")
        title = "Put européen" if exercise == "european" else "Put américain"
        ax_price.set_ylim(ref - 0.05, ref + 0.05)
        _style(ax_price, f"{title} : prix", "", "prix du put (€)", legend_loc="lower right")
        _style(ax_error, f"{title} : erreur × N", "nombre de pas N", "(prix − référence) × N",
               legend_loc="lower right")
    fig.suptitle(f"BBS sur un put européen et américain : S0 = {s0}, K = {k}, r = {r:.0%}, σ = {vol:.0%}, "
                 "T = 1 an, pruning à 6 écarts-types")
    return _save(fig, "11_bbs_put_americain.png")


# =====================================================================
# Grecques : erreur en fonction de la taille du choc (graphique 12)
# =====================================================================

# grecque : (chocs h testés, choc du pricer, échelle d'affichage, unité, nom de l'abscisse)
GREEK_SHIFTS = {
    "theta": ([1, 2, 5, 10, 20, 50, 100], 1, 1, "par jour", "choc h (jours)"),
    "rho": ([1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 0.01, 0.02, 0.05], 0.01, 1 / 100, "pour 1 % de taux",
            "choc h sur le taux"),
    "vomma": ([1e-6, 3e-6, 1e-5, 1e-4, 1e-3, 2e-3, 5e-3, 0.01, 0.02, 0.05, 0.1], 0.01, 1 / 10000,
              "pour 1 point de vol au carré", "choc h sur la vol"),
    "vanna": ([1e-6, 3e-6, 1e-5, 1e-4, 1e-3, 2e-3, 5e-3, 0.01, 0.02, 0.05, 0.1], 0.01, 1 / 100,
              "delta pour 1 point de vol", "choc h sur la vol"),
}


def _tree_greek(greek: str, h: float, model: Model, price: float) -> float:
    """Grecque de l’arbre avec un choc h : mêmes fonctions que le pricer (tree.py), seul le shift change."""
    params = (MARKET, model, OPTION, "forward")
    if greek == "theta":
        return OneDimDerivative(price_vs_days, params, shift=h).first(0)
    if greek == "rho":
        return OneDimDerivative(price_vs_rate, params, shift=h).first(R)
    if greek == "vanna":
        return OneDimDerivative(delta_vs_vol, params, shift=h).first(VOL)
    return OneDimDerivative(price_vs_vol, params, shift=h).second(VOL, price)


def _bs_greek(greek: str, h: float) -> float:
    """Même différence finie, même choc h, mais sur la formule de Black-Scholes (sans arbre)."""
    def bs(r: float = R, vol: float = VOL, t: float = T) -> float:
        return black_scholes(S0, K, r, vol, t, "call")

    def delta(vol: float) -> float:
        return black_scholes_greeks(S0, K, R, vol, T, "call")["delta"]

    if greek == "theta":                    # date de pricing + h jours = h jours de moins avant la maturité
        return (bs(t=T - h / 365) - bs(t=T + h / 365)) / (2 * h)
    if greek == "rho":
        return (bs(r=R + h) - bs(r=R - h)) / (2 * h)
    if greek == "vanna":
        return (delta(VOL + h) - delta(VOL - h)) / (2 * h)
    return (bs(vol=VOL + h) - 2 * bs() + bs(vol=VOL - h)) / h ** 2


def _greek_panel(ax_value: plt.Axes, ax_error: plt.Axes, greek: str, n: int, exact: float) -> None:
    """Une colonne du graphique 12 : en haut la valeur de la grecque selon le choc h, en bas l’erreur
    |grecque - Black-Scholes exact|, pour le binomial, le trinomial et Black-Scholes avec le même choc."""
    shifts, pricer_shift, scale, unit, xlabel = GREEK_SHIFTS[greek]
    curves = []                                               # (valeurs, couleur, style, nom)
    for tree_type, color in (("binomial", BLUE), ("trinomial", ORANGE)):
        model = Model(PRICING_DATE, MATURITY, n, tree_type)
        price = price_tree(MARKET, model, OPTION)
        curves.append(([_tree_greek(greek, h, model, price) for h in shifts], color, "-", tree_type))
        print(f"{greek} {tree_type} : fait")
    curves.append(([_bs_greek(greek, h) for h in shifts], "grey", "--", "Black-Scholes, même choc"))
    for values, color, style, name in curves:
        ax_value.plot(shifts, [v * scale for v in values], color=color, linestyle=style,
                      marker="o" if style == "-" else None, markersize=4, label=name)
        ax_error.plot(shifts, [abs(v - exact) * scale for v in values], color=color, linestyle=style,
                      marker="o" if style == "-" else None, markersize=4, label=name)
    ax_value.axhline(exact * scale, color="black", linewidth=1, label="Black-Scholes exact")
    for ax in (ax_value, ax_error):
        ax.axvline(pricer_shift, color="grey", linewidth=6, alpha=0.25, zorder=0)
    _style(ax_value, f"{greek} ({unit})", xlabel, greek, log_x=True)
    _style(ax_error, f"|erreur| sur le {greek}", xlabel, "|grecque − Black-Scholes exact|", log_x=True, log_y=True)


def plot_greeks_shift(n: int = 1000) -> str:
    """Graphique 12 : grecques en fonction de la taille du choc h des différences finies, N = n fixé,
    binomial et trinomial. En haut la valeur de la grecque, en bas l’erreur |grecque - Black-Scholes exact|.
    - h trop petit -> bruit : le prix de l’arbre fait des "dents de scie" selon la position du strike
      entre les nœuds ; avec un petit choc on mesure leur pente, l’erreur reste élevée et erratique
      (et pour h minuscule, l’arrondi de la machine divisé par h² fait exploser le vomma)
    - h trop grand -> biais de la différence finie : courbe grise, même calcul sur Black-Scholes
    La bande grise verticale marque le choc utilisé par le pricer (_shifted_greeks)."""
    exact = black_scholes_greeks(S0, K, R, VOL, T, "call")
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.5))
    for j, greek in enumerate(GREEK_SHIFTS):
        _greek_panel(axes[0, j], axes[1, j], greek, n, exact[greek])
    fig.suptitle(f"Grecques selon la taille du choc : choc trop petit = bruit, trop grand = biais\n"
                 f"call européen, S0 = K = {S0}, r = {R:.0%}, σ = {VOL:.0%}, T = 1 an, N = {n} ; "
                 "bande grise : choc du pricer")
    return _save(fig, "12_erreur_grecques_choc.png")


def generate_all() -> list:
    """Crée tous les graphiques du rapport et renvoie leurs chemins."""
    return [plot_runtime_methods(), plot_runtime_pruning(), plot_peak_memory(), plot_memory_management(),
            plot_gap_vs_strike(), plot_gap_vs_strike_zoom(), plot_price_kink(), plot_precision(),
            plot_bbs(), plot_bbs_error_x_n(), plot_bbs_american(), plot_greeks_shift()]
