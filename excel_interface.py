# Interface Excel : lit les parametres dans parametres_pricer.xlsx (onglet "Parametres"), lance le calcul,
# ecrit les resultats et la comparaison avec Black-Scholes (onglet "Resultats") et l'image de l'arbre
# (onglet "Visualisation").

import math
import os
import unicodedata
from datetime import date, datetime

from openpyxl import Workbook, load_workbook
from openpyxl.worksheet.worksheet import Worksheet

from tree import Market, Model, Option, Tree, black_scholes_greeks, price_all, print_results
from visualisation import TreeImage

# ---------------------------------------------------------------------
# Emplacement des cases (onglet "Parametres", colonne C)
# ---------------------------------------------------------------------
EXCEL_CELLS = {
    "s0": "C9", "r": "C10", "vol": "C11",
    "option_type": "C14", "strike": "C15", "exercise": "C16",
    "pricing_date": "C19", "maturity": "C20", "n_mode": "C21", "nb_steps": "C22", "precision": "C23",
    "tree_type": "C26", "construction": "C27", "bbs": "C28",
    "prune_mode": "C30", "prune_proba": "C31", "prune_std": "C32",
    "memory_management": "C35", "measure_memory": "C36",
    "greeks": "C39", "draw_tree": "C40", "graph_steps": "C41",
}
EXCEL_DIV_ROWS = range(45, 55)   # dividendes : date en colonne B, montant en colonne C

# Textes des listes deroulantes -> valeurs du code (sans accents ni majuscules, voir _norm)
CHOICES = {
    "option_type": {"call": "call", "put": "put"},
    "exercise": {"europeenne": "european", "americaine": "american"},
    "n_mode": {"nombre de pas fixe": "steps", "precision visee": "precision"},
    "tree_type": {"trinomial": "trinomial", "binomial": "binomial"},
    "construction": {"forward": "forward", "backward": "backward"},
    "prune_mode": {"aucun": "none", "probabilite": "proba", "ecarts-types": "std", "les deux": "both"},
    "yes_no": {"oui": True, "non": False},
}

# ---------------------------------------------------------------------
# Cases ou Python ecrit les resultats (onglet "Resultats")
# colonne C = arbre, colonne D = Black-Scholes ; les ecarts (colonnes E et F) sont des formules Excel
# ---------------------------------------------------------------------
RES_DATE, RES_SUMMARY, RES_NOTE = "C4", "B5", "B10"
RES_PRICE = {"prix": 9, "prix_europeen": 11, "prime_exercice_anticipe": 12}
RES_GREEKS = {"delta": 16, "gamma": 17, "vega": 18, "theta": 19, "rho": 20, "vomma": 21, "vanna": 22,
              "theta_funding": 23}
RES_INFO = {"tree_type": 26, "construction": 27, "nb_steps": 28, "dt": 29, "alpha": 30, "nb_nodes": 31,
            "pruning": 32, "gestion_memoire": 33, "temps_pricing": 34, "memoire_mo": 35, "noeuds_max": 36,
            "temps": 37}
# unités affichées dans Excel : pour 1 point de vol / 1 % de taux (au carré pour le vomma)
EXCEL_SCALE = {"vega": 1 / 100, "vanna": 1 / 100, "rho": 1 / 100, "vomma": 1 / 10000}


def _norm(x: object) -> str:
    """Texte sans accents, en minuscules : "Européenne" -> "europeenne"."""
    text = unicodedata.normalize("NFD", str(x).strip().lower())
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")


def _choice(x: object, name: str, kind: str) -> object:
    """Texte d’une liste déroulante -> valeur du code."""
    mapping = CHOICES[kind]
    key = _norm(x) if x is not None else ""
    if key not in mapping:
        raise ValueError(f"{name} : valeur '{x}' non reconnue, choisir dans la liste déroulante")
    return mapping[key]


def _to_date(x: object, name: str) -> date:
    """Date lue dans Excel (datetime, ou texte jj/mm/aaaa)."""
    if isinstance(x, datetime):
        return x.date()
    if isinstance(x, date):
        return x
    if isinstance(x, str):
        return datetime.strptime(x.strip(), "%d/%m/%Y").date()
    raise ValueError(f"{name} : date manquante ou illisible ({x!r})")


def _number(x: object, name: str, cell: str) -> float:
    """Nombre lu dans une case obligatoire (erreur claire si la case est vide)."""
    if x is None:
        raise ValueError(f"{name} : case {cell} vide dans l'onglet Parametres")
    return float(x)


# ---------------------------------------------------------------------
# Lecture
# ---------------------------------------------------------------------
def read_excel_inputs(path: str) -> tuple[Market, Model, Option, dict]:
    """Lit l’onglet Parametres. Renvoie (market, model, option, settings) ; settings contient
    construction, greeks, memory_management, measure_memory et graph_steps (None = pas de dessin)."""
    ws = load_workbook(path, data_only=True)["Parametres"]
    v = {k: ws[c].value for k, c in EXCEL_CELLS.items()}
    market = _read_market(ws, v)
    option = Option(_choice(v["option_type"], "type d'option", "option_type"),
                    _number(v["strike"], "strike", "C15"),
                    _choice(v["exercise"], "exercice", "exercise"))
    return market, _read_model(v, market), option, _read_settings(v)


def _read_market(ws: Worksheet, v: dict) -> Market:
    """Marché : spot, taux, volatilité et dividendes (une ligne par dividende, lignes vides ignorées)."""
    dividends = {}
    for row in EXCEL_DIV_ROWS:
        d, amount = ws[f"B{row}"].value, ws[f"C{row}"].value
        if d is None and amount is None:
            continue                                  # ligne vide
        ex_date = _to_date(d, f"date du dividende ligne {row}")
        dividends[ex_date] = _number(amount, "montant du dividende", f"C{row}")
    return Market(_number(v["s0"], "spot", "C9"), _number(v["r"], "taux", "C10"),
                  _number(v["vol"], "volatilité", "C11"), dividends=dividends)


def _read_model(v: dict, market: Market) -> Model:
    """Modèle : dates, type d’arbre, pruning et nombre de pas (fixé, ou déduit de la précision visée)."""
    pricing_date = _to_date(v["pricing_date"], "date de pricing")
    maturity = _to_date(v["maturity"], "maturité")
    if maturity <= pricing_date:
        raise ValueError("la maturité doit être après la date de pricing")
    params = dict(
        tree_type=_choice(v["tree_type"], "type d'arbre", "tree_type"),
        prune_mode=_choice(v["prune_mode"], "pruning", "prune_mode"),
        prune_proba=None if v["prune_proba"] is None else float(v["prune_proba"]),
        prune_std=None if v["prune_std"] is None else float(v["prune_std"]),
        bbs=_choice(v["bbs"] or "Non", "méthode BBS", "yes_no"))   # case vide = Non
    if _choice(v["n_mode"], "choix du nombre de pas", "n_mode") == "precision":
        return Model.from_precision(market, pricing_date, maturity,
                                    _number(v["precision"], "précision visée", "C23"), **params)
    return Model(pricing_date, maturity, int(_number(v["nb_steps"], "nombre de pas", "C22")), **params)


def _read_settings(v: dict) -> dict:
    """Options de calcul : construction, grecques, mémoire et nombre de pas du dessin de l’arbre."""
    draw = _choice(v["draw_tree"], "dessiner l'arbre", "yes_no")
    return {
        "construction": _choice(v["construction"], "construction", "construction"),
        "greeks": _choice(v["greeks"], "calculer les grecques", "yes_no"),
        "memory_management": _choice(v["memory_management"], "gestion de la mémoire", "yes_no"),
        "measure_memory": _choice(v["measure_memory"], "mesurer la mémoire", "yes_no"),
        "graph_steps": int(v["graph_steps"]) if draw and v["graph_steps"] else None,
    }


# ---------------------------------------------------------------------
# Black-Scholes de référence
# ---------------------------------------------------------------------
def black_scholes_reference(market: Market, model: Model, option: Option) -> tuple[dict, str]:
    """Prix et grecques Black-Scholes de l’option EUROPÉENNE de mêmes caractéristiques.
    Dividendes discrets : on retire du spot leur valeur actualisée (approximation classique).
    Renvoie (dictionnaire des valeurs, phrase qui dit si la comparaison est exacte ou indicative)."""
    T = model.year_fraction(model.maturity)
    pv_div = 0.0
    for ex_date, amount in market.dividends.items():
        t = model.year_fraction(ex_date)
        if 0 < t <= T:
            pv_div += amount * math.exp(-market.r * t)
    bs = black_scholes_greeks(market.s0 - pv_div, option.strike, market.r, market.vol, T, option.option_type)

    if option.is_american():
        note = ("Comparaison indicative : Black-Scholes ne price que l'option européenne. L'écart sur le prix "
                "correspond surtout à la prime d'exercice anticipé : comparer plutôt la ligne "
                "« Prix européen équivalent ».")
    elif pv_div > 0:
        note = (f"Comparaison approchée : Black-Scholes est calculé avec le spot diminué des dividendes "
                f"actualisés ({pv_div:.4f} €). L'arbre traite le dividende exactement, à sa date.")
    else:
        note = ("Comparaison exacte : même option européenne sans dividende. L'écart vient uniquement du nombre "
                "de pas de l'arbre et diminue quand N augmente.")
    return bs, note


# ---------------------------------------------------------------------
# Dessin de l'arbre
# ---------------------------------------------------------------------
def make_tree_graph(market: Market, model: Model, option: Option, nb_steps: int,
                    folder: str = ".") -> tuple[str, str]:
    """Arbre de nb_steps pas (même marché, mêmes dates, même type d’arbre et pruning) dessiné en
    SVG + PNG. Renvoie (chemin du svg, chemin du png)."""
    tree = Tree(market, model.with_changes(nb_steps=nb_steps)).build()
    tree.price(option)
    image = TreeImage(tree, option)
    tree.release()                                    # l'image est faite : l'arbre ne sert plus
    svg_path, png_path = os.path.join(folder, "arbre.svg"), os.path.join(folder, "arbre.png")
    image.save_svg(svg_path)
    image.save_png(png_path)
    return svg_path, png_path


# ---------------------------------------------------------------------
# Écriture
# ---------------------------------------------------------------------
def _clean(x: object) -> object:
    """None pour les valeurs absentes ou non numériques (NaN), pour laisser la case vide."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return x


def _pruning_text(model: Model) -> str:
    """Description du pruning utilisé, pour l’onglet Resultats."""
    mode = model.prune_mode
    parts = []
    if mode in ("proba", "both") and model.prune_proba is not None:
        parts.append(f"probabilité < {model.prune_proba:.0e}")
    if mode in ("std", "both") and model.prune_std is not None:
        parts.append(f"plus de {model.prune_std:g} écarts-types")
    return "aucun" if not parts else " ou ".join(parts)


def _summary(market: Market, model: Model, option: Option) -> str:
    """Résumé d’une ligne de l’option et du marché, en haut de l’onglet Resultats."""
    kind = {"call": "Call", "put": "Put"}[option.option_type]
    ex = "européen" if option.exercise == "european" else "américain"
    T = model.year_fraction(model.maturity)
    div = f" · {len(market.dividends)} dividende(s)" if market.dividends else " · sans dividende"
    return (f"{kind} {ex}  ·  K = {option.strike:g}  ·  S0 = {market.s0:g}  ·  r = {market.r:.2%}  ·  "
            f"σ = {market.vol:.2%}  ·  T = {T:.4f} an{div}")


def write_excel_graph(wb: Workbook, png_path: str = None, svg_path: str = None):
    """Met l’image de l’arbre dans l’onglet "Visualisation"."""
    from openpyxl.drawing.image import Image as XLImage   # a besoin de Pillow
    ws = wb["Visualisation"]
    ws._images = []                                       # on enlève l'ancienne image
    if png_path is None:
        ws["B3"].value = "Pas de dessin : « Dessiner l'arbre » = Non dans l'onglet Parametres."
        return
    ws["B3"].value = (f"Image vectorielle (zoomable dans un navigateur) : {svg_path}   ·   "
                      f"mise à jour le {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    try:
        img = XLImage(png_path)
    except ImportError:
        ws["B3"].value = f"Image non insérée : installer Pillow puis redémarrer le noyau. Image : {png_path}"
        print("image non inseree dans Excel : installer Pillow puis redemarrer le noyau Python")
        return
    img.width, img.height = 1100, int(1100 * img.height / img.width)
    ws.add_image(img, "B5")


def write_excel_results(path: str, market: Market, model: Model, option: Option, res: dict,
                        bs: dict, note: str, png_path: str = None, svg_path: str = None):
    """Écrit les résultats dans l’onglet "Resultats" (le fichier doit être fermé dans Excel)."""
    wb = load_workbook(path)
    ws = wb["Resultats"]
    ws[RES_DATE].value = datetime.now().strftime("%d/%m/%Y %H:%M")
    ws[RES_SUMMARY].value = _summary(market, model, option)
    ws[RES_NOTE].value = note
    _write_prices(ws, option, res, bs)
    _write_greeks_and_info(ws, model, res, bs)
    write_excel_graph(wb, png_path, svg_path)
    wb.active = wb.sheetnames.index("Resultats")
    wb.calculation.fullCalcOnLoad = True                  # Excel recalcule les écarts à l'ouverture
    wb.save(path)


def _write_prices(ws: Worksheet, option: Option, res: dict, bs: dict):
    """Prix de l’arbre (colonne C) et de Black-Scholes (colonne D) ; lignes de l’européenne équivalente
    et de la prime d’exercice anticipé remplies seulement pour une américaine."""
    american = option.is_american()
    ws[f"C{RES_PRICE['prix']}"].value = res["prix"]
    ws[f"D{RES_PRICE['prix']}"].value = bs["prix"]
    ws[f"C{RES_PRICE['prix_europeen']}"].value = res.get("prix_europeen") if american else None
    ws[f"D{RES_PRICE['prix_europeen']}"].value = bs["prix"] if american else None
    ws[f"C{RES_PRICE['prime_exercice_anticipe']}"].value = res.get("prime_exercice_anticipe") if american else None


def _write_greeks_and_info(ws: Worksheet, model: Model, res: dict, bs: dict):
    """Grecques de l’arbre et de Black-Scholes (dans les unités Excel), puis informations sur l’arbre."""
    for key, row in RES_GREEKS.items():
        scale = EXCEL_SCALE.get(key, 1)
        tree_value = _clean(res.get(key))
        ws[f"C{row}"].value = None if tree_value is None else tree_value * scale
        ws[f"D{row}"].value = bs[key] * scale if "delta" in res else None
    info = dict(res)
    info["pruning"] = _pruning_text(model)
    if model.bbs:
        info["construction"] += " + BBS (Black-Scholes sur le dernier pas)"
    for key, row in RES_INFO.items():
        ws[f"C{row}"].value = _clean(info.get(key))


# ---------------------------------------------------------------------
# Fonction principale
# ---------------------------------------------------------------------
def run_from_excel(path: str = "parametres_pricer.xlsx", write: bool = True) -> dict:
    """LA fonction à appeler : lit les paramètres, calcule tout, affiche et écrit les résultats."""
    market, model, option, s = read_excel_inputs(path)

    previous = Tree.memory_management                     # réglage "Gestion de la mémoire"
    Tree.memory_management = s["memory_management"]
    try:
        res = _price(market, model, option, s)
        bs, note = black_scholes_reference(market, model, option)
        print(f"  Black-Scholes     : {bs['prix']:.4f}   (ecart arbre - BS : {res['prix'] - bs['prix']:+.4f})")
        print(f"  {note}")
        svg_path, png_path = _draw(market, model, option, s["graph_steps"], path)
    finally:
        Tree.memory_management = previous

    if write:
        try:
            write_excel_results(path, market, model, option, res, bs, note, png_path, svg_path)
            print(f"resultats ecrits dans l'onglet Resultats de {path}")
        except PermissionError:
            print(f"impossible d'ecrire dans {path} : fermer le fichier dans Excel puis relancer")
    res["black_scholes"] = bs
    return res


def _price(market: Market, model: Model, option: Option, s: dict) -> dict:
    """Prix et grecques de l’arbre (price_all), avec un message clair si des probabilités sortent de [0, 1]."""
    try:
        res = price_all(market, model, option, greeks=s["greeks"], method=s["construction"],
                        memory=s["measure_memory"])
    except AssertionError as e:
        # cours "Effect of discrete dividends" : un dividende fixe en euros peut rendre les probabilités
        # négatives sur les nœuds très bas, où il pèse lourd par rapport au spot
        raise ValueError(f"{e}\nProbabilités hors de [0, 1] : le dividende est trop gros pour certains nœuds "
                         "très bas de l'arbre. Activer le pruning par probabilité (onglet Parametres, "
                         "section 5).") from None
    res["gestion_memoire"] = "oui" if s["memory_management"] else "non"
    print_results(option, res)
    return res


def _draw(market: Market, model: Model, option: Option, graph_steps: int, path: str) -> tuple[str, str]:
    """Dessine le petit arbre à côté du fichier Excel si demandé. Renvoie (svg, png), ou (None, None)."""
    if not graph_steps:
        return None, None
    try:
        svg_path, png_path = make_tree_graph(market, model, option, graph_steps,
                                             os.path.dirname(os.path.abspath(path)))
        print(f"graphe de l'arbre ({graph_steps} pas) : {svg_path}")
        return svg_path, png_path
    except ImportError as e:
        print(f"graphe non genere ({e}) : installer matplotlib avec  pip install matplotlib")
        return None, None
