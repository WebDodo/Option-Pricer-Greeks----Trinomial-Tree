# Visualisation de l'arbre : image vectorielle SVG (zoomable dans un navigateur) + PNG (pour Excel)
# Organisation reprise du memo du prof "Python #06 : how to make SVG graphs" :
#  - VectorImage : interface generique (formes + export), independante de l'arbre
#  - TreeImage(VectorImage) : dessin specifique a notre arbre (trinomial ou binomial)
# Les listes ici ne servent qu'a stocker les formes a dessiner, pas a acceder aux noeuds.

import math
from typing import TYPE_CHECKING

from tree import Node, Option, Tree

if TYPE_CHECKING:                       # seulement pour les annotations de type (matplotlib importé à la demande)
    from matplotlib.axes import Axes


def to_hexa_two_char(number: float) -> str:
    # 15 -> '0f', 0 -> '00' (comme dans le memo du prof)
    return f"{max(0, min(255, int(number))):02x}"


def _fmt(x: float, decimals: int = 2) -> str:
    # nombre au format francais : 8.1043 -> '8,10'
    return f"{x:.{decimals}f}".replace(".", ",")


def _escape(s: object) -> str:
    # caractères réservés du SVG (XML) dans les textes
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class VectorImage:
    # image vectorielle : une liste de formes, exportable en SVG ou en PNG
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.shapes = []          # dans l'ordre d'affichage : le premier est dessine en dessous

    # ---- ajout de formes
    def add_end(self, shape: dict):
        # par-dessus tout le reste
        self.shapes.append(shape)

    # ---- formes de base
    @staticmethod
    def segment(x1: float, y1: float, x2: float, y2: float, color: str = "#999999", width: float = 1.0,
                opacity: float = 1.0, dash: str = None) -> dict:
        return {"type": "line", "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "color": color, "width": width, "opacity": opacity, "dash": dash}

    @staticmethod
    def circle(cx: float, cy: float, r: float, fill: str, stroke: str = None, stroke_width: float = 0.0,
               opacity: float = 1.0) -> dict:
        return {"type": "circle", "cx": cx, "cy": cy, "r": r, "fill": fill,
                "stroke": stroke, "stroke_width": stroke_width, "opacity": opacity}

    @staticmethod
    def rect(x: float, y: float, w: float, h: float, fill: str, stroke: str = None,
             stroke_width: float = 0.0) -> dict:
        return {"type": "rect", "x": x, "y": y, "w": w, "h": h, "fill": fill,
                "stroke": stroke, "stroke_width": stroke_width}

    @staticmethod
    def text(x: float, y: float, s: str, size: float = 12, color: str = "#0b0b0b", anchor: str = "start",
             weight: str = "normal") -> dict:
        # anchor : "start" (aligne a gauche), "middle", "end" (aligne a droite)
        return {"type": "text", "x": x, "y": y, "s": s, "size": size, "color": color,
                "anchor": anchor, "weight": weight}

    # ---- export SVG (texte brut, aucune dependance)
    def to_svg(self) -> str:
        out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" '
               f'viewBox="0 0 {self.width} {self.height}" font-family="Arial, Helvetica, sans-serif">',
               f'<rect x="0" y="0" width="{self.width}" height="{self.height}" fill="#ffffff"/>']
        out += [self._shape_to_svg(s) for s in self.shapes]
        out.append("</svg>")
        return "\n".join(line for line in out if line)

    @staticmethod
    def _shape_to_svg(s: dict) -> str:
        # une forme -> une balise SVG (chaîne vide pour un type inconnu)
        if s["type"] == "line":
            dash = f' stroke-dasharray="{s["dash"]}"' if s["dash"] else ""
            return (f'<line x1="{s["x1"]:.1f}" y1="{s["y1"]:.1f}" x2="{s["x2"]:.1f}" y2="{s["y2"]:.1f}" '
                    f'stroke="{s["color"]}" stroke-width="{s["width"]}" stroke-opacity="{s["opacity"]}"{dash}/>')
        if s["type"] == "circle":
            stroke = f' stroke="{s["stroke"]}" stroke-width="{s["stroke_width"]}"' if s["stroke"] else ""
            return (f'<circle cx="{s["cx"]:.1f}" cy="{s["cy"]:.1f}" r="{s["r"]:.2f}" fill="{s["fill"]}" '
                    f'fill-opacity="{s["opacity"]}"{stroke}/>')
        if s["type"] == "rect":
            stroke = f' stroke="{s["stroke"]}" stroke-width="{s["stroke_width"]}"' if s["stroke"] else ""
            return (f'<rect x="{s["x"]:.1f}" y="{s["y"]:.1f}" width="{s["w"]:.1f}" height="{s["h"]:.1f}" '
                    f'fill="{s["fill"]}"{stroke}/>')
        if s["type"] == "text":
            return (f'<text x="{s["x"]:.1f}" y="{s["y"]:.1f}" font-size="{s["size"]}" fill="{s["color"]}" '
                    f'text-anchor="{s["anchor"]}" font-weight="{s["weight"]}">{_escape(s["s"])}</text>')
        return ""

    def save_svg(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_svg())

    # ---- export PNG (avec matplotlib, pour pouvoir inserer l'image dans Excel)
    def save_png(self, path: str, scale: float = 1.0):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(self.width / 100, self.height / 100), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, self.width)
        ax.set_ylim(self.height, 0)          # y vers le bas, comme en SVG
        ax.axis("off")
        self._draw_shapes(ax)
        fig.savefig(path, dpi=100 * scale, facecolor="white")
        plt.close(fig)

    def _draw_shapes(self, ax: "Axes"):
        # on regroupe les formes consecutives de meme type (lignes, cercles) pour aller vite ;
        # z = ordre d'affichage, comme dans le SVG
        pt = 72 / 100                         # 1 pixel = 0.72 point a 100 dpi
        z = 0
        i = 0
        while i < len(self.shapes):
            kind = self.shapes[i]["type"]
            j = i
            while j < len(self.shapes) and self.shapes[j]["type"] == kind and kind in ("line", "circle"):
                j += 1
            if kind == "line":
                self._add_lines(ax, self.shapes[i:j], z, pt)
            elif kind == "circle":
                self._add_circles(ax, self.shapes[i:j], z, pt)
            else:
                self._add_single(ax, self.shapes[i], z, pt)
            i = max(j, i + 1)
            z += 1

    @staticmethod
    def _add_lines(ax: "Axes", group: list, z: int, pt: float):
        # segments pleins puis pointilles, chacun en une seule collection
        from matplotlib.collections import LineCollection
        from matplotlib.colors import to_rgba
        for dashed in (False, True):
            g = [s for s in group if bool(s["dash"]) == dashed]
            if g:
                lc = LineCollection([[(s["x1"], s["y1"]), (s["x2"], s["y2"])] for s in g],
                                    colors=[to_rgba(s["color"], s["opacity"]) for s in g],
                                    linewidths=[s["width"] * pt for s in g],
                                    linestyles="--" if dashed else "-", zorder=z)
                ax.add_collection(lc)

    @staticmethod
    def _add_circles(ax: "Axes", group: list, z: int, pt: float):
        # cercles consecutifs en une seule collection
        from matplotlib.collections import PatchCollection
        from matplotlib.colors import to_rgba
        from matplotlib.patches import Circle
        pc = PatchCollection([Circle((s["cx"], s["cy"]), s["r"]) for s in group],
                             facecolors=[to_rgba(s["fill"], s["opacity"]) for s in group],
                             edgecolors=[s["stroke"] or "none" for s in group],
                             linewidths=[s["stroke_width"] * pt for s in group], zorder=z)
        ax.add_collection(pc)

    @staticmethod
    def _add_single(ax: "Axes", s: dict, z: int, pt: float):
        # rectangle ou texte, dessine seul
        from matplotlib.patches import Rectangle
        if s["type"] == "rect":
            ax.add_patch(Rectangle((s["x"], s["y"]), s["w"], s["h"], facecolor=s["fill"],
                                   edgecolor=s["stroke"] or "none",
                                   linewidth=s["stroke_width"] * pt, zorder=z))
        elif s["type"] == "text":
            ha = {"start": "left", "middle": "center", "end": "right"}[s["anchor"]]
            ax.text(s["x"], s["y"], s["s"], fontsize=s["size"] * pt, color=s["color"], ha=ha,
                    va="baseline", fontweight=s["weight"], family="DejaVu Sans", zorder=z)


class TreeImage(VectorImage):
    # dessin de l'arbre (trinomial ou binomial) :
    #  - un rond par noeud, place en (date, spot) avec le spot en echelle log
    #  - couleur = valeur de l'option (bleu clair -> bleu fonce), orange = exercice anticipe
    #  - taille = racine de la probabilite d'atteindre le noeud (comme dans le memo du prof)
    #  - branches grises vers les enfants (3 en trinomial, 2 en binomial)
    #  - pointilles aux dates de dividende et au strike
    INK, INK2, GRID, EDGE = "#0b0b0b", "#52514e", "#e6e4df", "#c9c8c3"
    EXERCISE = "#eb6834"
    ZERO = "#d6d5d0"                       # option sans valeur
    RAMP = [(0.0, (205, 226, 251)), (0.25, (134, 182, 239)), (0.5, (57, 135, 229)),
            (0.75, (37, 106, 191)), (1.0, (13, 54, 107))]

    def __init__(self, tree: Tree, option: Option, width: int = None, height: int = None):
        # taille de l'image : 1280 x 800, agrandie pour les gros arbres (memo du prof)
        scaler = max(round(tree.model.nb_steps / 100), 1) * 16
        super().__init__(width or scaler * 80, height or scaler * 50)
        self.tree = tree
        self.option = option
        self.left, self.right, self.top, self.bottom = 80, 250, 78, 70
        self._build_image()

    # ---------- positions
    def _x(self, i: int) -> float:
        w = self.width - self.left - self.right
        return self.left + w * i / self.tree.model.nb_steps

    def _y(self, spot: float) -> float:
        h = self.height - self.top - self.bottom
        return self.top + h * (self.ln_max - math.log(spot)) / (self.ln_max - self.ln_min)

    # ---------- couleurs
    def _value_color(self, v: float) -> str:
        # echelle en racine carree : etale les petites valeurs (les plus nombreuses)
        if v <= 1e-12:
            return self.ZERO
        t = 0.0 if self.v_max <= 0 else math.sqrt(max(0.0, min(1.0, v / self.v_max)))
        for (t0, c0), (t1, c1) in zip(self.RAMP, self.RAMP[1:]):
            if t <= t1:
                u = (t - t0) / (t1 - t0)
                return "#" + "".join(to_hexa_two_char(a + (b - a) * u) for a, b in zip(c0, c1))
        return "#0d366b"

    def _is_exercised(self, node: Node) -> bool:
        # exercice anticipe : americaine, hors maturite, valeur egale a la valeur intrinseque (> 0)
        if not self.option.is_american() or node.is_leaf():
            return False
        vi = node.intrinsic(self.option)
        return vi > 0 and abs(node.option_price - vi) <= 1e-10 * max(1.0, vi)

    # ---------- dessin
    def _build_image(self):
        # 1er passage : bornes des spots et des valeurs (pour les echelles)
        s_min, s_max = self._scan_bounds()
        self.ln_min, self.ln_max = math.log(s_min) - 0.03, math.log(s_max) + 0.03
        self._draw_axes(s_min, s_max)
        # 2e passage : branches (en dessous) et noeuds (au-dessus)
        self._draw_nodes()
        self._draw_legend()

    def _scan_bounds(self) -> tuple[float, float]:
        # spot minimal et maximal de l'arbre, et valeur maximale de l'option (self.v_max)
        s_min, s_max, self.v_max = math.inf, 0.0, 0.0
        trunk = self.tree.root
        while trunk is not None:
            node = trunk.top()
            while node is not None:
                s_min, s_max = min(s_min, node.spot), max(s_max, node.spot)
                self.v_max = max(self.v_max, node.option_price)
                node = node.down
            trunk = trunk.next_trunk
        return s_min, s_max

    def _draw_nodes(self):
        # branches vers les enfants, puis les noeuds par-dessus (dans l'ordre des colonnes)
        dx = (self.width - self.left - self.right) / self.tree.model.nb_steps
        r_max = max(2.0, min(0.42 * dx, 12.0))
        nodes_layer = []
        trunk, i = self.tree.root, 0
        while trunk is not None:
            node = trunk.top()
            while node is not None:
                x, y = self._x(i), self._y(node.spot)
                for child in (node.next_up, node.next_mid, node.next_down):
                    if child is not None:
                        self.add_end(self.segment(x, y, self._x(i + 1), self._y(child.spot),
                                                  self.EDGE, 0.7, 0.8))
                nodes_layer.append(self._node_circle(node, trunk, x, y, r_max))
                node = node.down
            trunk, i = trunk.next_trunk, i + 1
        for c in nodes_layer:
            self.add_end(c)

    def _node_circle(self, node: Node, trunk: Node, x: float, y: float, r_max: float) -> dict:
        # rond d'un noeud : taille selon la probabilite d'atteinte, couleur selon la valeur de l'option
        radius = max(r_max * math.sqrt(max(node.proba, 0.0)), 2.2)
        fill = self.EXERCISE if self._is_exercised(node) else self._value_color(node.option_price)
        stroke = self.INK if node is trunk else "#ffffff"
        return self.circle(x, y, radius, fill, stroke, 1.0 if node is trunk else 0.6)

    def _draw_axes(self, s_min: float, s_max: float):
        # titre, grille du spot (et strike), axe du temps (et dividendes)
        self._draw_title()
        self._draw_spot_grid(s_min, s_max)
        self._draw_time_axis()

    def _draw_title(self):
        tree, opt, m = self.tree, self.option, self.tree.model
        exo = "américain" if opt.is_american() else "européen"
        title = f"Arbre {tree.description} : {opt.option_type} {exo}, K = {_fmt(opt.strike)}  (N = {m.nb_steps} pas)"
        self.add_end(self.text(self.left, 32, title, 20, self.INK, weight="bold"))
        self.add_end(self.text(self.left, 56, "couleur = valeur de l'option, taille = probabilité d'atteindre le nœud",
                               13, self.INK2))

    def _draw_spot_grid(self, s_min: float, s_max: float):
        # grille horizontale + graduations du spot, puis ligne pointillee du strike
        x0, x1 = self._x(0), self._x(self.tree.model.nb_steps)
        step = self._nice_step(s_min, s_max)
        level = math.ceil(s_min / step) * step
        while level <= s_max:
            y = self._y(level)
            self.add_end(self.segment(x0, y, x1, y, self.GRID, 1.0))
            self.add_end(self.text(x0 - 10, y + 4, _fmt(level, 0), 12, self.INK2, anchor="end"))
            level += step
        self.add_end(self.text(x0 - 10, self.top - 12, "spot", 12, self.INK2, anchor="end"))
        strike = self.option.strike
        if s_min < strike < s_max:
            y = self._y(strike)
            self.add_end(self.segment(x0, y, x1, y, self.INK2, 1.0, 0.9, dash="6 4"))
            self.add_end(self.text(x1 + 16, y + 4, f"K = {_fmt(strike, 0)}", 12, self.INK2))

    def _draw_time_axis(self):
        # axe du temps : date de pricing, dividendes (pointilles oranges), maturite
        m = self.tree.model
        x0, x1 = self._x(0), self._x(m.nb_steps)
        y_bottom = self.height - self.bottom
        self.add_end(self.segment(x0, y_bottom, x1, y_bottom, self.INK2, 1.0))
        T = (m.maturity - m.pricing_date).days / 365
        ticks = [(0.0, m.pricing_date.strftime("%d/%m/%Y")), (T, m.maturity.strftime("%d/%m/%Y"))]
        for d, amount in self.tree.market.dividends.items():
            t = m.year_fraction(d)
            if 0 < t <= T:
                x = x0 + (x1 - x0) * t / T
                self.add_end(self.segment(x, self.top, x, y_bottom, self.EXERCISE, 1.2, 0.8, dash="4 4"))
                self.add_end(self.text(x, self.top - 8, f"div. {_fmt(amount)} €", 12, self.INK, anchor="middle"))
                ticks.append((t, d.strftime("%d/%m/%Y")))
        for t, label in ticks:
            x = x0 + (x1 - x0) * t / T
            self.add_end(self.segment(x, y_bottom, x, y_bottom + 6, self.INK2, 1.0))
            self.add_end(self.text(x, y_bottom + 22, label, 12, self.INK2, anchor="middle"))

    @staticmethod
    def _nice_step(lo: float, hi: float) -> float:
        # pas "rond" des graduations du spot : 8 graduations au plus
        span = hi - lo
        for step in (1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000):
            if span / step <= 8:
                return step
        return 10 ** math.ceil(math.log10(span / 8))

    def _draw_legend(self):
        # echelle de couleur des valeurs, puis pastilles explicatives et prix
        x = self.width - self.right + 70
        y = self.top + 10
        self.add_end(self.text(x, y, "Valeur de l'option", 13, self.INK, weight="bold"))
        bar_h, n = 180, 60
        for k in range(n):                       # degrade : haut = valeur max (echelle en racine)
            t = 1 - k / (n - 1)
            self.add_end(self.rect(x, y + 14 + k * bar_h / n, 22, bar_h / n + 0.5,
                                   self._value_color(self.v_max * t * t)))
        self.add_end(self.rect(x, y + 14, 22, bar_h, "none", self.EDGE, 1.0))
        self.add_end(self.text(x + 31, y + 14 + bar_h + 4, "0", 12, self.INK2))
        for t in (1.0, 0.5, 0.25):              # graduations : v_max * t^2
            yy = y + 14 + (1 - t) * bar_h
            self.add_end(self.segment(x + 22, yy, x + 27, yy, self.INK2, 1.0))
            self.add_end(self.text(x + 31, yy + 4, _fmt(self.v_max * t * t), 12, self.INK2))
        self._draw_legend_keys(x, y + bar_h + 40)

    def _draw_legend_keys(self, x: float, y: float):
        # pastilles : valeur nulle, exercice anticipe (americaine), tronc ; puis le prix
        self.add_end(self.circle(x + 11, y - 4, 7, self.ZERO))
        self.add_end(self.text(x + 26, y, "valeur nulle", 12, self.INK))
        y += 30
        if self.option.is_american():
            self.add_end(self.circle(x + 11, y - 4, 7, self.EXERCISE))
            self.add_end(self.text(x + 26, y, "exercice anticipé", 12, self.INK))
            self.add_end(self.text(x + 26, y + 16, "(valeur = intrinsèque)", 12, self.INK2))
            y += 44
        self.add_end(self.circle(x + 11, y - 4, 7, "#ffffff", self.INK, 1.0))
        self.add_end(self.text(x + 26, y, "tronc", 12, self.INK))
        y += 34
        self.add_end(self.text(x, y, f"Prix : {_fmt(self.tree.root.option_price, 4)}", 14, self.INK, weight="bold"))
