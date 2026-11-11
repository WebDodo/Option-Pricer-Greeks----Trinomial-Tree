# Pricer d'options europeennes / americaines par arbre, avec dividendes discrets (montant + date ex-div)
# et pruning :
#  - arbre trinomial, ou binomial qui bascule en trinomial au pas du premier dividende (cours)
#  - construction forward (arbre entier en memoire) ou 100% backward (2 colonnes en memoire)
#  - option BBS : prix Black-Scholes sur le dernier pas, pour lisser la convergence

import gc  # Libérer de l'espace de mémoire
import math
import time
import tracemalloc  # mesure la mémoire utilisée par le programme
from datetime import date, timedelta
from typing import Callable


# =====================================================================
# Donnees d'entree : marche, modele, option
# =====================================================================

class Market:
    """Données de marché : spot, taux, volatilité et dividendes discrets."""
    def __init__(self, s0: float, r: float, vol: float, dividends: dict = None):

        self.s0 = s0
        self.r = r
        self.vol = vol
        # Dividendes discrets {date ex-div : montant en euros},
        # ex : dividends={date(2027, 3, 1): 2.0, date(2027, 9, 1): 2.5}
        self.dividends = dict(dividends) if dividends else {}


class Model:
    """Paramètres du modèle : dates, nombre de pas, type d’arbre et pruning."""
    def __init__(self, pricing_date: date, maturity: date, nb_steps: int, tree_type: str = "trinomial",
                 prune_mode: str = "proba", prune_proba: float = 1e-7, prune_std: float = None,
                 bbs: bool = False):

        self.pricing_date = pricing_date
        self.maturity = maturity
        self.nb_steps = nb_steps
        # Type d'arbre : "trinomial" (trinomial partout) ou "binomial" (binomial, puis trinomial
        # à partir du pas qui contient le 1er dividende ; binomial pur s'il n'y a pas de dividende)
        self.tree_type = tree_type.strip().lower()
        if self.tree_type not in ("trinomial", "binomial"):
            raise ValueError(f"Type d'arbre inconnu : '{tree_type}'. Choisir 'trinomial' ou 'binomial'.")

        # Pruning : "proba", "std", "both" ou "none" (None = "proba", la valeur par défaut)
        self.prune_mode = (prune_mode or "proba").strip().lower()
        if self.prune_mode not in ("proba", "std", "both", "none"):
            raise ValueError(f"Mode de pruning inconnu : '{prune_mode}'. "
                             "Choisir 'proba', 'std', 'both' ou 'none'.")
        # Méthode 1 : seuil de probabilité d'atteinte sous lequel on élague
        self.prune_proba = prune_proba
        # Méthode 2 : seuil en nombre d'écarts-types par rapport au tronc
        self.prune_std = prune_std
        # Méthode BBS (Binomial Black-Scholes) : si True, sur le dernier pas on remplace
        # l'espérance des enfants par le prix Black-Scholes de maturité dt
        self.bbs = bool(bbs)

    def with_changes(self, **changes: object) -> "Model":
        """Copie du modèle avec certains paramètres changés (ex : model.with_changes(nb_steps=30))."""
        params = dict(pricing_date=self.pricing_date, maturity=self.maturity, nb_steps=self.nb_steps,
                      tree_type=self.tree_type, prune_mode=self.prune_mode, prune_proba=self.prune_proba,
                      prune_std=self.prune_std, bbs=self.bbs)
        params.update(changes)
        return Model(**params)

    @classmethod
    def from_precision(cls, market: Market, pricing_date: date, maturity: date,
                       precision: float, **kwargs: object) -> "Model":
        """Modèle dont le nombre de pas N est déduit d’une précision visée sur le prix (en euros).

        Formule du cours ("Evaluate the error (4)"), avec dt = T / N :
          trinomial : gap(N) ~ 3 S0 e^(rT) / (8 sqrt(2 pi)) * sigma^2 dt / sqrt(e^(sigma^2 T) - 1)
          binomial  : gap(N) ~   S0 e^(rT) / (2 sqrt(2 pi)) * sigma^2 dt / sqrt(e^(sigma^2 T) - 1)
        (l’écart entre 2 nœuds vaut ~ sigma sqrt(3 dt) en trinomial et ~ 2 sigma sqrt(dt) en binomial)"""
        T = (maturity - pricing_date).days / 365
        s0, r, vol = market.s0, market.r, market.vol
        tree_type = str(kwargs.get("tree_type", "trinomial")).strip().lower()
        coef = 1 / 2 if tree_type == "binomial" else 3 / 8
        n = coef * s0 * math.exp(r * T) * vol ** 2 * T \
            / (math.sqrt(2 * math.pi) * precision * math.sqrt(math.exp(vol ** 2 * T) - 1))
        return cls(pricing_date, maturity, max(1, math.ceil(n)), **kwargs)

    @property
    def dt(self) -> float:
        """Durée d’un pas de temps (en fraction d’année)."""
        return (self.maturity - self.pricing_date).days / 365 / self.nb_steps

    def year_fraction(self, d: date) -> float:
        """Convertit une date en fraction d’année depuis la date de pricing."""
        return (d - self.pricing_date).days / 365


class Option:
    """Caractéristiques de l’option : type ("call" / "put"), strike, exercice ("european" / "american")."""
    def __init__(self, option_type: str, strike: float, exercise: str = "european"):
        if option_type not in ("call", "put"):
            raise ValueError(f"Type d'option inconnu : '{option_type}'. Choisir 'call' ou 'put'.")
        if exercise not in ("european", "american"):
            raise ValueError(f"Exercice inconnu : '{exercise}'. Choisir 'european' ou 'american'.")
        self.option_type = option_type
        self.strike = strike
        self.exercise = exercise

    def payoff(self, spot: float) -> float:
        """Gain de l’option exercée pour un prix spot donné."""
        if self.option_type == "call":
            return max(spot - self.strike, 0)
        return max(self.strike - spot, 0)

    def is_american(self) -> bool:
        return self.exercise == "american"


# =====================================================================
# Formule de Black-Scholes (option europeenne, sans dividende)
# =====================================================================

def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _norm_pdf(x: float) -> float:
    return math.exp(-x * x / 2) / math.sqrt(2 * math.pi)


def _bs_d1_d2(s: float, k: float, r: float, vol: float, t: float) -> tuple[float, float]:
    sd = vol * math.sqrt(t)
    d1 = (math.log(s / k) + (r + vol ** 2 / 2) * t) / sd
    return d1, d1 - sd


def black_scholes(s: float, k: float, r: float, vol: float, t: float, option_type: str = "call") -> float:
    """Prix Black-Scholes d’une option européenne : spot s, strike k, taux r (continu), volatilité vol,
    maturité t en années. Sert de référence pour étudier la convergence de l’arbre et vérifier les grecques."""
    df = math.exp(-r * t)
    if t <= 0 or vol <= 0 or k <= 0:
        # cas limites : la valeur est la valeur actualisée du payoff du forward
        call = max(s - k * df, 0.0)
    else:
        d1, d2 = _bs_d1_d2(s, k, r, vol, t)
        call = s * _norm_cdf(d1) - k * df * _norm_cdf(d2)
    if option_type == "call":
        return call
    return call - s + k * df  # parité call-put


def black_scholes_greeks(s: float, k: float, r: float, vol: float, t: float, option_type: str = "call") -> dict:
    """Prix et grecques Black-Scholes (formules fermées), dans les mêmes unités que price_all :
    vega, vomma, vanna et rho pour une variation de 1 (= 100 %), thêta et thêta funding par jour."""
    d1, d2 = _bs_d1_d2(s, k, r, vol, t)
    df, sq = math.exp(-r * t), math.sqrt(t)
    n1 = _norm_pdf(d1)
    call = option_type == "call"
    price = black_scholes(s, k, r, vol, t, option_type)
    vega = s * n1 * sq
    theta_year = -s * n1 * vol / (2 * sq) + (-r * k * df * _norm_cdf(d2) if call else r * k * df * _norm_cdf(-d2))
    return {
        "prix": price,
        "delta": _norm_cdf(d1) if call else _norm_cdf(d1) - 1,
        "gamma": n1 / (s * vol * sq),
        "vega": vega,
        "vomma": vega * d1 * d2 / vol,
        "vanna": -n1 * d2 / vol,
        "theta": theta_year / 365,
        "theta_funding": theta_year / 365 - r * price / 365,
        "rho": k * t * df * _norm_cdf(d2) if call else -k * t * df * _norm_cdf(-d2),
    }


# =====================================================================
# Noeuds de l'arbre
# =====================================================================

class Node:
    """Nœud de l’arbre (binomial ou trinomial) avec ses liens vers ses voisins et ses enfants."""

    # Compteurs de nœuds en mémoire, membres statiques (cours "Count allocated objects").
    # Ils ne sont actifs qu'entre start_counting() et stop_counting() : surcharger __del__ double
    # le temps de calcul (cours, et mesuré), on ne le fait donc que pour mesurer la mémoire.
    counting = False   # comptage actif ?
    nb_alive = 0       # nœuds actuellement en mémoire
    max_alive = 0      # maximum atteint depuis start_counting()

    def _counting_del(self):
        Node.nb_alive -= 1

    @staticmethod
    def start_counting():
        """Active le comptage des nœuds créés / détruits (compteurs remis à 0)."""
        Node.nb_alive = Node.max_alive = 0
        Node.counting = True
        Node.__del__ = Node._counting_del

    @staticmethod
    def stop_counting():
        """Désactive le comptage (retour à la vitesse normale)."""
        Node.counting = False
        del Node.__del__

    def __init__(self, spot: float, tree: "Tree"):
        # Prix du sous-jacent en ce nœud
        self.spot = spot
        # Référence vers l'arbre (paramètres communs à tous les nœuds)
        self.tree = tree

        # Enfants (colonne suivante, t + dt) ; next_mid absent sur un nœud binomial
        self.next_up = None
        self.next_mid = None
        self.next_down = None
        # Voisins verticaux (même colonne)
        self.up = None
        self.down = None

        # Probabilités de transition vers les enfants
        # (différentes d'un nœud à l'autre en trinomial quand il y a un dividende)
        self.p_up = None
        self.p_mid = None
        self.p_down = None

        # Probabilité d'atteindre ce nœud depuis la racine (sert au pruning)
        self.proba = 0.0
        # Valeur de l'option en ce nœud
        self.option_price = None
        # Compteurs : nœuds créés par cet arbre, et nœuds en mémoire (si le comptage est actif)
        tree.nb_nodes += 1
        if Node.counting:
            Node.nb_alive += 1
            if Node.nb_alive > Node.max_alive:
                Node.max_alive = Node.nb_alive

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.spot:.4f})"

    # ---------- caractéristiques du nœud
    def intrinsic(self, option: Option) -> float:
        """Valeur d’exercice immédiat de l’option en ce nœud."""
        return option.payoff(self.spot)

    def forward(self, div: float = 0.0) -> float:
        """Espérance du spot au pas suivant : S e^(r dt), moins le dividende s’il tombe sur ce pas."""
        fwd = self.spot * math.exp(self.tree.market.r * self.tree.model.dt) - div
        if fwd <= 0:
            raise ValueError(f"forward negatif ({fwd:.4f}) au noeud {self}, dividende trop gros")
        return fwd

    def variance(self) -> float:  # elle convertit la vol (en %, annuelle) en
        # « de combien d’euros l’action peut bouger au prochain pas depuis ce nœud »,
        # pour que les probabilités de l’arbre reproduisent bien la volatilité.
        """Variance du spot au pas suivant (cours "Common constraints (3)")."""
        r, vol, dt = self.tree.market.r, self.tree.market.vol, self.tree.model.dt
        return self.spot ** 2 * math.exp(2 * r * dt) * (math.exp(vol ** 2 * dt) - 1)

    def is_leaf(self) -> bool:
        """Vrai si le nœud est sur la colonne de maturité (aucun enfant)."""
        return self.next_up is None and self.next_mid is None and self.next_down is None

    def is_last_step(self) -> bool:
        """Vrai si le nœud est sur l’avant-dernière colonne (ses enfants sont à maturité)."""
        for child in (self.next_up, self.next_mid, self.next_down):
            if child is not None:
                return child.is_leaf()
        return False

    # ---------- déplacements dans la colonne
    def create_up(self) -> "Node":
        """Voisin du dessus dans la colonne (créé s’il n’existe pas encore)."""
        if self.up is None:
            self.up = Node(self.spot * self.tree.alpha, self.tree)
            self.up.down = self
        return self.up

    def create_down(self) -> "Node":
        """Voisin du dessous dans la colonne (créé s’il n’existe pas encore)."""
        if self.down is None:
            self.down = Node(self.spot / self.tree.alpha, self.tree)
            self.down.up = self
        return self.down

    def top(self) -> "Node":
        """Nœud le plus haut de la colonne."""
        node = self
        while node.up is not None:
            node = node.up
        return node

    def is_close(self, target: float) -> bool:
        """Vrai si target est plus proche de ce nœud que de ses voisins (milieux entre S/a, S et S*a)."""
        a = self.tree.alpha
        return self.spot * (1 + 1 / a) / 2 <= target <= self.spot * (1 + a) / 2

    @staticmethod
    def _walk_to(guess: "Node", target: float) -> "Node":
        """Part de guess et monte / descend dans sa colonne (en créant les nœuds manquants)
        jusqu’au nœud le plus proche de target."""
        node = guess
        while not node.is_close(target):
            node = node.create_up() if target > node.spot else node.create_down()
        return node

    # ---------- connexion à la colonne suivante
    def connect_binomial(self, guess: "Node", pruned: bool = False, above_trunk: bool = True) -> "Node":
        """Connexion binomiale (2 branches, pas de next_mid) d’un pas SANS dividende.

        Cours ("Binomial lattice (2)-(4)") : le forward est au milieu géométrique des 2 enfants,
        S_up = fwd * k et S_down = fwd / k, avec p_up = 1 / (k + 1) (constante, calculée dans Tree).
        Renvoie l’enfant qui sert de point de départ au nœud voisin."""
        fwd = self.forward()
        k = self.tree.bin_factor

        if pruned:
            # Pruning (cours) : une seule branche, redirigée vers l'intérieur de l'arbre (vers le tronc)
            if above_trunk:
                child = self._walk_to(guess, fwd / k)
                self.next_down = child
                self.p_up, self.p_mid, self.p_down = 0.0, 0.0, 1.0
            else:
                child = self._walk_to(guess, fwd * k)
                self.next_up = child
                self.p_up, self.p_mid, self.p_down = 1.0, 0.0, 0.0
            child.proba += self.proba
            return child

        # Enfant du haut = nœud le plus proche de fwd * k, enfant du bas = son voisin du dessous
        up = self._walk_to(guess, fwd * k)
        self.next_up = up
        self.next_down = up.create_down()
        self.p_up, self.p_mid, self.p_down = self.tree.bin_p_up, 0.0, self.tree.bin_p_down

        # Propagation des probabilités d'atteinte vers les enfants
        self.next_up.proba += self.proba * self.p_up
        self.next_down.proba += self.proba * self.p_down
        return up

    def connect_trinomial(self, guess: "Node", div: float = 0.0, pruned: bool = False) -> "Node":
        """Connexion trinomiale (3 branches) : next_mid = nœud le plus proche du forward.
        Avec un dividende le forward baisse, donc le mid descend tout seul. Renvoie le next_mid."""
        fwd = self.forward(div)
        mid = self._walk_to(guess, fwd)
        self.next_mid = mid

        if pruned:
            # Pruning (cours) : branchement "monomial", une seule branche vers le mid avec proba 1
            self.p_up, self.p_mid, self.p_down = 0.0, 1.0, 0.0
            mid.proba += self.proba
            return mid

        self.next_up = mid.create_up()
        self.next_down = mid.create_down()
        self._compute_trinomial_probabilities(fwd)

        # Propagation des probabilités d'atteinte vers les enfants
        self.next_up.proba += self.proba * self.p_up
        self.next_mid.proba += self.proba * self.p_mid
        self.next_down.proba += self.proba * self.p_down
        return mid

    def _compute_trinomial_probabilities(self, fwd: float):
        """Probabilités trinomiales : solution du système 3x3 (somme = 1, espérance, variance),
        cours "Trinomial tree (1)-(3)"."""
        a = self.tree.alpha
        s_mid = self.next_mid.spot
        E, V = fwd, self.variance()

        p_down = ((V + E ** 2) / s_mid ** 2 - 1 - (a + 1) * (E / s_mid - 1)) / ((1 - a) * (a ** -2 - 1))
        p_up = ((E / s_mid - 1) - (1 / a - 1) * p_down) / (a - 1)
        p_mid = 1 - p_up - p_down

        assert min(p_up, p_mid, p_down) >= 0 and max(p_up, p_mid, p_down) <= 1, \
            f"proba hors de [0,1] au noeud {self} : up={p_up:.2%} mid={p_mid:.2%} down={p_down:.2%}"
        self.p_up, self.p_mid, self.p_down = p_up, p_mid, p_down

    # ---------- pricing et grecques
    def price(self, option: Option) -> float:
        """Valeur de l’option en ce nœud : espérance actualisée des enfants, puis max avec
        l’exercice immédiat pour une américaine. Un seul code pour tous les nœuds : trinomial
        (3 enfants), binomial (pas de next_mid), pruné (1 seul enfant, proba 1)."""
        if self.is_leaf():  # colonne de maturité
            self.option_price = self.intrinsic(option)
            return self.option_price

        if self.tree.model.bbs and self.is_last_step():
            v = self.bbs_price(option)  # méthode BBS : Black-Scholes sur le dernier pas
        else:
            v = self.expectation()
        if option.is_american():
            v = max(v, self.intrinsic(option))
        self.option_price = v
        return v

    def expectation(self) -> float:
        """Espérance actualisée des valeurs des enfants (1, 2 ou 3 enfants selon le nœud)."""
        v = 0.0
        if self.next_up is not None:
            v += self.p_up * self.next_up.option_price
        if self.next_mid is not None:
            v += self.p_mid * self.next_mid.option_price
        if self.next_down is not None:
            v += self.p_down * self.next_down.option_price
        return v * self.tree.df

    def bbs_price(self, option: Option) -> float:
        """Méthode BBS : valeur européenne du dernier pas donnée par Black-Scholes, de maturité dt.
        Le payoff a un coude en K : l’arbre le voit en escalier, ce qui fait osciller le prix
        selon N (pair / impair). Black-Scholes lisse ce dernier pas et supprime ces oscillations.
        S’il y a un dividende sur ce dernier pas, on prend le spot moins le dividende actualisé
        (= forward actualisé)."""
        tree = self.tree
        div = tree.dividend_on_step(tree.model.nb_steps - 1)
        spot = self.forward(div) * tree.df
        return black_scholes(spot, option.strike, tree.market.r, tree.market.vol, tree.model.dt,
                             option.option_type)

    def delta(self) -> float:
        """Delta lu sur les enfants haut et bas (à appeler sur la racine, après le pricing)."""
        u, d = self.next_up, self.next_down
        return (u.option_price - d.option_price) / (u.spot - d.spot)

    def gamma(self) -> float:
        """Gamma lu sur 3 nœuds descendants (à appeler sur la racine, après le pricing) :
        trinomial : les 3 enfants (date dt) ; binomial : haut-haut, haut-bas et bas-bas (date 2 dt)."""
        if self.next_mid is None:  # nœud binomial
            if self.next_up is None or self.next_down is None \
                    or self.next_up.next_up is None or self.next_down.next_down is None:
                return math.nan  # il faut au moins 2 pas
            u, m, d = self.next_up.next_up, self.next_up.next_down, self.next_down.next_down
        else:
            if self.next_up is None or self.next_down is None:
                return math.nan
            u, m, d = self.next_up, self.next_mid, self.next_down
        # deltas entre haut et milieu, puis entre milieu et bas
        delta_up = (u.option_price - m.option_price) / (u.spot - m.spot)
        delta_down = (m.option_price - d.option_price) / (m.spot - d.spot)
        # /2 car ces deltas sont des pentes mesurées au milieu entre 2 nœuds
        return (delta_up - delta_down) / ((u.spot - d.spot) / 2)


class TrunkNode(Node):
    """Nœud du tronc : relié aux troncs précédent et suivant, il donne accès à toute sa colonne."""
    def __init__(self, spot: float, tree: "Tree", prev_trunk: "TrunkNode" = None):
        super().__init__(spot, tree)
        # Liens vers le tronc précédent (pricing backward) et suivant (affichage)
        self.prev_trunk = prev_trunk
        self.next_trunk = None
        # Description de la colonne, utile à la construction 100% backward :
        # nombre de nœuds au-dessus / en dessous du tronc
        self.n_up = 0
        self.n_down = 0
        # nœuds prunés codés en bits dans un entier (pas de liste) :
        # bit j-1 de pruned_up = 1 si le j-ème nœud au-dessus du tronc est pruné, idem en dessous
        self.pruned_up = 0
        self.pruned_down = 0


# Règle de pruning : (nœud, tronc de sa colonne, numéro du pas i, indice j du nœud) -> nœud pruné ?
PruneRule = Callable[[Node, TrunkNode, int, int], bool]


# =====================================================================
# Suivi dynamique de la memoire
# =====================================================================

class MemoryTracker:
    """Suivi dynamique de la mémoire (cours : "suivre dynamiquement la consommation mémoire d’une
    classe avec des membres statiques"). Dans un bloc  `with MemoryTracker() as suivi:`  on relève,
    après chaque colonne construite ou pricée et après chaque libération d’arbre :
    - suivi.nodes : le nombre de nœuds en mémoire (compteurs statiques Node.nb_alive)
    - suivi.mo    : la mémoire occupée en Mo (tracemalloc)
    Ces mesures ralentissent beaucoup le calcul : à n’utiliser que pour étudier la mémoire.
    (les listes ne servent qu’à stocker les mesures, pas à accéder aux nœuds)"""
    active = None      # suivi en cours (un seul à la fois)

    def __init__(self):
        self.nodes = []
        self.mo = []

    def __enter__(self) -> "MemoryTracker":
        gc.collect()                              # on part d'une mémoire propre
        self._already = tracemalloc.is_tracing()
        if not self._already:
            tracemalloc.start()
        self._base = tracemalloc.get_traced_memory()[0]
        Node.start_counting()
        MemoryTracker.active = self
        return self

    def __exit__(self, *exc: object) -> bool:
        MemoryTracker.active = None
        Node.stop_counting()
        if not self._already:
            tracemalloc.stop()
        return False

    @staticmethod
    def tick():
        """Appelé par l’arbre au fil du calcul : enregistre une mesure si un suivi est actif."""
        tracker = MemoryTracker.active
        if tracker is not None:
            tracker.nodes.append(Node.nb_alive)
            tracker.mo.append((tracemalloc.get_traced_memory()[0] - tracker._base) / 1e6)


# =====================================================================
# Arbre : construction et pricing
# =====================================================================

class Tree:
    """Construction de l’arbre et pricing.

    - tree_type "trinomial" : trinomial du début à la fin
    - tree_type "binomial"  : binomial tant qu’il n’y a pas de dividende, puis trinomial à partir du
      pas qui contient le 1er dividende (cours "Discrete dividends (2)" : "we switch to trinomial
      when there is a dividend") ; sans dividende, binomial pur."""

    # Gestion de la mémoire (cours "Memory management") : déconnexion des colonnes / arbres devenus
    # inutiles. Membre statique, à passer à False uniquement pour comparer "sans" / "avec" (c'est
    # l'équivalent de la ligne commentée dans l'exemple du cours) : les résultats sont identiques.
    memory_management = True

    def __init__(self, market: Market, model: Model, grid_vol: float = None):
        """grid_vol : volatilité qui fixe l’écart entre les nœuds trinomiaux (par défaut market.vol).
        Pour les grecques en volatilité, les arbres à vol ± h gardent la grille de l’arbre de base
        (seules les probabilités changent : cours "Trinomial tree (4)", tout écart entre sqrt(2) et
        2 écarts-types convient) ; sinon le décalage de la grille crée un bruit qui fausse le vomma."""
        self.market = market
        self.model = model
        self.root = None
        self.last_trunk = None
        self.nb_nodes = 0
        dt = model.dt

        # Paramètres calculés une seule fois pour tout l'arbre
        # Trinomial : écart entre 2 nœuds alpha = e^(sigma sqrt(3 dt)) (cours "Trinomial tree (4)")
        self.alpha_tri = math.exp((market.vol if grid_vol is None else grid_vol) * math.sqrt(3 * dt))
        # Binomial : facteur k entre le forward et chaque enfant, seule racine > 1 de
        # k + 1/k = 1 + e^(sigma^2 dt) (cours "Binomial lattice (4)") -> espérance et variance exactes
        x = (1 + math.exp(market.vol ** 2 * dt)) / 2
        self.bin_factor = x + math.sqrt(x * x - 1)
        # écart entre 2 nœuds d'une colonne binomiale (fwd*k et fwd/k sont espacés de k^2)
        self.alpha_bin = self.bin_factor ** 2
        # probabilités binomiales, constantes dans toute la partie binomiale (toujours dans ]0, 1[)
        self.bin_p_up = 1 / (self.bin_factor + 1)
        self.bin_p_down = self.bin_factor / (self.bin_factor + 1)

        # Premier pas construit en trinomial (0 = tout trinomial, nb_steps = tout binomial)
        self.switch_step = self._compute_switch_step()
        # Écart entre nœuds de la colonne en cours de construction (change à la bascule
        # binomial -> trinomial) ; en dehors de la construction : celui de la colonne de maturité
        self.alpha = self.column_alpha(model.nb_steps)
        # Facteur d'actualisation sur 1 pas
        self.df = math.exp(-market.r * dt)

    # ---------- paramètres de l'arbre
    def dividend_on_step(self, i: int) -> float:
        """Montant des dividendes payés sur le pas i (t_i < t_ex <= t_i+1), additionnés si plusieurs.
        Petite tolérance sur les dates (cours "Case 1: error on dates")."""
        dt = self.model.dt
        tol = dt / 1000
        total = 0.0
        for ex_date, amount in self.market.dividends.items():
            t_ex = self.model.year_fraction(ex_date)
            if i * dt + tol < t_ex <= (i + 1) * dt + tol:
                total += amount
        return total

    def _compute_switch_step(self) -> int:
        """Numéro du premier pas construit en trinomial."""
        if self.model.tree_type == "trinomial":
            return 0
        for i in range(self.model.nb_steps):
            if self.dividend_on_step(i) != 0:
                return i
        return self.model.nb_steps

    def column_alpha(self, c: int) -> float:
        """Écart entre 2 nœuds voisins de la colonne c (construite par le pas c - 1)."""
        return self.alpha_bin if c <= self.switch_step else self.alpha_tri

    @property
    def description(self) -> str:
        """Type d’arbre effectivement construit (pour les affichages)."""
        if self.switch_step == 0:
            return "trinomial"
        if self.switch_step >= self.model.nb_steps:
            return "binomial"
        return f"binomial puis trinomial (dès le pas {self.switch_step})"

    # ---------- pruning
    def _is_pruned(self, node: Node, trunk: TrunkNode, i: int, j: int) -> bool:
        """Pruning de la construction forward : probabilité d’atteinte trop faible (méthode 1)
        et / ou nœud trop loin du tronc en nombre d’écarts-types (méthode 2)."""
        m = self.model
        if m.prune_mode in ("proba", "both") and m.prune_proba is not None \
                and node.proba < m.prune_proba:
            return True
        if m.prune_mode in ("std", "both") and m.prune_std is not None and i > 0:
            nb_std = abs(math.log(node.spot / trunk.spot)) / (self.market.vol * math.sqrt(i * m.dt))
            if nb_std > m.prune_std:
                return True
        return False

    @staticmethod
    def _was_pruned(node: Node, trunk: TrunkNode, i: int, j: int) -> bool:
        """Pruning de la construction backward : on relit les nœuds prunés mémorisés par
        build_trunk() (les probabilités d’atteinte n’existent qu’en avançant dans le temps)."""
        if j > 0:
            return (trunk.pruned_up >> (j - 1)) & 1 == 1
        return (trunk.pruned_down >> (-j - 1)) & 1 == 1

    # ---------- un pas de construction : relier la colonne i à la colonne i + 1
    def _build_step(self, t: TrunkNode, i: int, prune_rule: PruneRule, next_trunk: TrunkNode = None) -> TrunkNode:
        """Relie tous les nœuds de la colonne i (tronc t) à la colonne i + 1 et renvoie le tronc i + 1.

        - pas binomial (i < switch_step) : le tronc alterne haut / bas, 2 branches par nœud
        - pas trinomial : le tronc suivant est le forward du tronc (dividende compris), 3 branches
        prune_rule(node, tronc, i, j) dit si le nœud d’indice j (0 = tronc) est pruné.
        next_trunk : tronc i + 1 déjà existant (construction backward), sinon il est créé."""
        self.alpha = self.column_alpha(i + 1)
        binomial = i < self.switch_step
        div = 0.0 if binomial else self.dividend_on_step(i)

        # 1) le tronc d'abord (il n'est jamais pruné)
        next_trunk, first = self._link_trunk(t, i, binomial, div, next_trunk)
        # 2) les nœuds au-dessus du tronc (direction +1) puis en dessous (direction -1) ;
        # les nœuds prunés sont mémorisés sur le tronc pour la construction backward
        t.pruned_up = self._link_side(t, i, 1, first, prune_rule, binomial, div)
        t.pruned_down = self._link_side(t, i, -1, first, prune_rule, binomial, div)

        t.next_trunk = next_trunk
        MemoryTracker.tick()
        return next_trunk

    def _link_trunk(self, t: TrunkNode, i: int, binomial: bool, div: float,
                    next_trunk: TrunkNode) -> tuple[TrunkNode, Node]:
        """Relie le tronc t à la colonne suivante, en créant le tronc suivant s’il n’existe pas encore.
        Renvoie (tronc suivant, premier enfant trouvé = point de départ pour les nœuds voisins)."""
        if binomial:
            if next_trunk is None:
                fwd, k = t.forward(), self.bin_factor
                # branche du haut aux pas pairs, du bas aux pas impairs (cours "Binomial lattice building")
                next_trunk = TrunkNode(fwd * k if i % 2 == 0 else fwd / k, self, prev_trunk=t)
            return next_trunk, t.connect_binomial(next_trunk)
        if next_trunk is None:
            next_trunk = TrunkNode(t.forward(div), self, prev_trunk=t)
        return next_trunk, t.connect_trinomial(next_trunk, div)

    @staticmethod
    def _link_side(t: TrunkNode, i: int, direction: int, first: Node, prune_rule: PruneRule,
                   binomial: bool, div: float) -> int:
        """Relie à la colonne suivante les nœuds d’un côté du tronc (direction +1 : au-dessus,
        -1 : en dessous), du plus proche au plus éloigné. Renvoie les nœuds prunés codés en bits :
        bit j-1 = 1 si le j-ème nœud depuis le tronc est pruné (pas de liste)."""
        node = t.up if direction > 0 else t.down
        guess, j, mask = first, direction, 0
        while node is not None:
            pruned = prune_rule(node, t, i, j)
            if pruned:
                mask |= 1 << (abs(j) - 1)
            if binomial:
                guess = node.connect_binomial(guess, pruned, above_trunk=direction > 0)
            else:
                start = guess.create_up() if direction > 0 else guess.create_down()
                guess = node.connect_trinomial(start, div, pruned)
            node = node.up if direction > 0 else node.down
            j += direction
        return mask

    def _price_column(self, trunk: TrunkNode, option: Option):
        """Valeur de l’option sur tous les nœuds de la colonne du tronc."""
        node = trunk.top()
        while node is not None:
            node.price(option)
            node = node.down
        MemoryTracker.tick()

    # ---------- construction forward + pricing backward
    def build(self) -> "Tree":
        """Construit l’arbre complet en avançant de t=0 vers la maturité (forward)."""
        self.root = TrunkNode(self.market.s0, self)
        self.root.proba = 1.0
        t = self.root
        for i in range(self.model.nb_steps):
            t = self._build_step(t, i, self._is_pruned)
        self.last_trunk = t
        return self

    def price(self, option: Option) -> float:
        """Price l’option sur l’arbre construit par build() : de la maturité vers la racine."""
        trunk = self.last_trunk
        while trunk is not None:
            self._price_column(trunk, option)
            trunk = trunk.prev_trunk
        return self.root.option_price

    # ---------- construction 100% backward (cours "Memory management", price_tree_backward)
    @staticmethod
    def cut_column(trunk: TrunkNode):
        """Déconnecte tous les nœuds de la colonne (liens remis à None) pour que le garbage
        collector libère la mémoire (cours "Memory management"). Le tronc est conservé."""
        for direction in (1, -1):
            node = trunk.up if direction > 0 else trunk.down
            while node is not None:
                nxt = node.up if direction > 0 else node.down
                node.up = node.down = node.next_up = node.next_mid = node.next_down = None
                node = nxt
        trunk.up = trunk.down = trunk.next_up = trunk.next_mid = trunk.next_down = None

    @staticmethod
    def _column_height(trunk: TrunkNode):
        """Mémorise sur le tronc le nombre de nœuds au-dessus et en dessous de lui."""
        n, node = 0, trunk.up
        while node is not None:
            n, node = n + 1, node.up
        trunk.n_up = n
        n, node = 0, trunk.down
        while node is not None:
            n, node = n + 1, node.down
        trunk.n_down = n

    def build_trunk(self) -> TrunkNode:
        """Étape 1 de la construction backward : même parcours forward que build() (lui seul donne
        les probabilités d’atteinte du pruning et la position des nœuds après un dividende), mais
        chaque colonne est détruite dès que la suivante existe. Il ne reste que le tronc, avec sur
        chaque nœud du tronc la description de sa colonne. Renvoie le tronc de la maturité."""
        self.root = TrunkNode(self.market.s0, self)
        self.root.proba = 1.0
        t = self.root
        for i in range(self.model.nb_steps):
            new_trunk = self._build_step(t, i, self._is_pruned)
            self._column_height(t)
            if self.memory_management:
                self.cut_column(t)
            t = new_trunk
        self._column_height(t)
        if self.memory_management:
            self.cut_column(t)
        self.last_trunk = t
        return t

    def _build_column(self, trunk: TrunkNode, c: int):
        """Recrée les nœuds de la colonne c autour de son tronc, avec exactement les mêmes spots
        qu’en forward (multiplications successives par alpha)."""
        self.alpha = self.column_alpha(c)
        node = trunk
        for _ in range(trunk.n_up):
            node = node.create_up()
        node = trunk
        for _ in range(trunk.n_down):
            node = node.create_down()

    def price_backward(self, option: Option) -> float:
        """Construction 100% backward + pricing :
        1) build_trunk() : le tronc seul (et la description de chaque colonne)
        2) de la maturité vers la racine : on reconstruit la colonne c, on la relie à la colonne c + 1
           (déjà pricée), on price la colonne c, puis on coupe la colonne c + 1 devenue inutile.
        Jamais plus de 2 colonnes complètes en mémoire (+ les colonnes 1 et 2, gardées pour le delta
        et le gamma). Résultat identique à build() + price()."""
        t = self.build_trunk()
        c = self.model.nb_steps
        self._build_column(t, c)
        self._price_column(t, option)
        while t.prev_trunk is not None:
            prev, c = t.prev_trunk, c - 1
            self._build_column(prev, c)
            nb_before = self.nb_nodes
            self._build_step(prev, c, self._was_pruned, next_trunk=t)
            if self.nb_nodes != nb_before:
                # tous les enfants doivent déjà exister (et être pricés) dans la colonne c + 1
                raise RuntimeError(f"construction backward incohérente au pas {c}")
            self._price_column(prev, option)
            if c + 1 >= 3 and self.memory_management:
                self.cut_column(t)
            t = prev
        self.alpha = self.column_alpha(self.model.nb_steps)
        return self.root.option_price

    def release(self):
        """Déconnecte tout l’arbre (cours "Memory management") : les nœuds se pointent les uns les
        autres (up <-> down, prev_trunk <-> next_trunk), ce qui forme des cycles que Python ne libère
        qu’au passage, occasionnel, du garbage collector. En remettant les liens à None, chaque nœud
        n’est plus référencé par personne et sa mémoire est libérée immédiatement.
        L’arbre n’est plus utilisable ensuite (les prix déjà calculés restent valables)."""
        if not self.memory_management:
            return
        trunk = self.root
        while trunk is not None:
            nxt = trunk.next_trunk
            self.cut_column(trunk)
            trunk.prev_trunk = trunk.next_trunk = None
            trunk = nxt
        self.root = self.last_trunk = None
        MemoryTracker.tick()

    def build_and_price(self, option: Option, method: str = "forward") -> float:
        """Construit l’arbre et price l’option avec la méthode choisie :
        "forward" = build() + price() (arbre entier en mémoire),
        "backward" = price_backward() (construction 100% backward, 2 colonnes en mémoire)."""
        if method == "forward":
            return self.build().price(option)
        if method == "backward":
            return self.price_backward(option)
        raise ValueError(f"methode de construction inconnue : '{method}'. Choisir 'forward' ou 'backward'.")


# =====================================================================
# Grecques par differences finies (cours, Assignment (3) : "pass a function as a parameter")
# =====================================================================

class OneDimDerivative:
    """Dérivées numériques d’une fonction f(paramètres, x) par différences centrées.
    Les valeurs f(x + h) et f(x - h) sont gardées : first() puis second() au même x ne
    reconstruisent pas d’arbre supplémentaire (vega puis vomma par exemple)."""
    def __init__(self, function: Callable[[tuple, float], float], other_parameters: tuple, shift: float = 1.0):
        self.f = function                 # fonction à dériver
        self.param = other_parameters     # paramètres fixes
        self.shift = shift                # pas de variation h
        self._x = None                    # dernier x calculé, avec f(x + h) et f(x - h)
        self._up = self._down = None

    def _around(self, x: float) -> tuple[float, float]:
        if self._x != x:
            h = self.shift
            self._x, self._up, self._down = x, self.f(self.param, x + h), self.f(self.param, x - h)
        return self._up, self._down

    def first(self, x: float) -> float:
        """Dérivée première en x : (f(x + h) - f(x - h)) / 2h."""
        up, down = self._around(x)
        return (up - down) / (2 * self.shift)

    def second(self, x: float, f_x: float) -> float:
        """Dérivée seconde en x : (f(x + h) - 2 f(x) + f(x - h)) / h², avec f_x = f(x) déjà connu."""
        up, down = self._around(x)
        return (up - 2 * f_x + down) / self.shift ** 2


def price_tree(market: Market, model: Model, option: Option, method: str = "forward",
               grid_vol: float = None) -> float:
    """Construit un arbre, price l’option puis libère l’arbre (release) : pour les arbres
    "jetables" dont on ne garde que le prix (grecques, prix européen, mesures)."""
    tree = Tree(market, model, grid_vol)
    price = tree.build_and_price(option, method)
    tree.release()
    return price


# Fonctions "prix en fonction d'un paramètre" à donner à OneDimDerivative.
# params = (market, model, option, method) : à chaque appel on reconstruit un arbre complet où
# seul le paramètre étudié change (même N, mêmes dates, mêmes dividendes...).

def price_vs_vol(params: tuple, vol: float) -> float:
    """Prix en fonction de la volatilité (vega, vomma), sur la grille de nœuds de l’arbre de base."""
    market, model, option, method = params
    m = Market(market.s0, market.r, vol, dividends=market.dividends)
    return price_tree(m, model, option, method, grid_vol=market.vol)


def price_vs_rate(params: tuple, r: float) -> float:
    """Prix en fonction du taux (rho)."""
    market, model, option, method = params
    m = Market(market.s0, r, market.vol, dividends=market.dividends)
    return price_tree(m, model, option, method)


def delta_vs_vol(params: tuple, vol: float) -> float:
    """Delta (lu sur l’arbre) en fonction de la volatilité (vanna = dérivée du delta par rapport à vol).
    Il n’y a pas d’astuce sur l’arbre pour la vol : il faut reconstruire l’arbre (cours)."""
    market, model, option, method = params
    tree = Tree(Market(market.s0, market.r, vol, dividends=market.dividends), model, grid_vol=market.vol)
    tree.build_and_price(option, method)
    delta = tree.root.delta()
    tree.release()
    return delta


def price_vs_days(params: tuple, days: float) -> float:
    """Prix en fonction d’un décalage de la date de pricing en jours, maturité fixe (thêta)."""
    market, model, option, method = params
    m = model.with_changes(pricing_date=model.pricing_date + timedelta(days=days))
    return price_tree(market, m, option, method)


# =====================================================================
# Fonctions principales : tout calculer d'un coup
# =====================================================================

def peak_memory(market: Market, model: Model, option: Option, method: str = "forward") -> tuple[float, int]:
    """Mémoire maximale utilisée par la construction + le pricing d’UN arbre. Renvoie :
    - le pic de mémoire en Mo, mesuré avec tracemalloc (module standard qui suit chaque allocation)
    - le nombre maximal de nœuds en mémoire en même temps (compteurs statiques de Node, cours)
    Ces deux mesures ralentissent beaucoup le calcul : on les fait dans un calcul à part, pour ne
    pas fausser le temps de calcul affiché."""
    gc.collect()                              # on part d'une mémoire propre (anciens arbres libérés)
    already = tracemalloc.is_tracing()        # si la mémoire est déjà suivie, on ne l'arrête pas
    if not already:
        tracemalloc.start()
    tracemalloc.reset_peak()
    base = tracemalloc.get_traced_memory()[0]
    Node.start_counting()
    try:
        price_tree(market, model, option, method)
    finally:
        Node.stop_counting()
        peak = tracemalloc.get_traced_memory()[1] - base
        if not already:
            tracemalloc.stop()
    return peak / 1e6, Node.max_alive


def memory_curve(market: Market, model: Model, option: Option, method: str = "forward",
                 greeks: bool = False, memory_management: bool = True) -> MemoryTracker:
    """Suit la mémoire tout au long d’un calcul complet (price_all), avec ou sans la gestion de la
    mémoire du cours, pour comparer "avant / après". Renvoie le MemoryTracker (listes nodes et mo)."""
    previous = Tree.memory_management
    Tree.memory_management = memory_management
    try:
        with MemoryTracker() as tracker:
            price_all(market, model, option, greeks=greeks, method=method)
    finally:
        Tree.memory_management = previous
    return tracker


def price_all(market: Market, model: Model, option: Option, greeks: bool = True,
              method: str = "forward", memory: bool = False) -> dict:
    """Prix, grecques, prime d’exercice anticipé et infos sur l’arbre, dans un dictionnaire
    (c’est juste la sortie, pas l’accès aux nœuds).
    method : construction de l’arbre, "forward" ou "backward" (100% backward).
    memory : si True, mesure aussi le pic de mémoire d’un pricing (calcul supplémentaire)."""
    t0 = time.perf_counter()
    tree = Tree(market, model)
    res = {"prix": tree.build_and_price(option, method)}
    res["temps_pricing"] = time.perf_counter() - t0   # construction + pricing de l'arbre, sans les grecques

    # 1) tout ce qui se lit sur l'arbre principal
    _read_main_tree(tree, option, greeks, method, res)
    # 2) l'arbre principal ne sert plus : on le libère avant de construire ceux des grecques
    tree.release()
    # 3) les autres grecques : on reconstruit un arbre avec le paramètre décalé (libéré aussitôt)
    if greeks:
        res.update(_shifted_greeks(market, model, option, method, res["prix"]))

    res["construction"] = method
    res["bbs"] = model.bbs
    res["nb_steps"] = model.nb_steps
    res["dt"] = model.dt
    res["temps"] = time.perf_counter() - t0
    if memory:
        res["memoire_mo"], res["noeuds_max"] = peak_memory(market, model, option, method)
    return res


def _read_main_tree(tree: Tree, option: Option, greeks: bool, method: str, res: dict):
    """Complète res avec ce qui se lit sur l’arbre principal déjà pricé : delta, gamma, prix européen
    (pour une américaine) et description de l’arbre."""
    if greeks:
        res["delta"] = tree.root.delta()
        res["gamma"] = tree.root.gamma()
    if option.is_american():
        # même option en européen : l'écart = valeur de l'exercice anticipé
        european = Option(option.option_type, option.strike, "european")
        if method == "forward":
            res["prix_europeen"] = tree.price(european)  # même arbre
        else:
            # en backward l'arbre n'est plus en mémoire : on le reconstruit
            res["prix_europeen"] = price_tree(tree.market, tree.model, european, method)
        res["prime_exercice_anticipe"] = res["prix"] - res["prix_europeen"]
    res["tree_type"] = tree.description
    res["alpha"] = tree.column_alpha(tree.model.nb_steps)   # écart entre nœuds à maturité
    res["nb_nodes"] = tree.nb_nodes


def _shifted_greeks(market: Market, model: Model, option: Option, method: str, price: float) -> dict:
    """Grecques obtenues en reconstruisant l’arbre avec un paramètre décalé (différences finies) :
    vega, vomma, vanna, thêta, thêta funding et rho."""
    params = (market, model, option, method)
    # pas de 1 point de vol (convention du fichier de référence) : vega et vomma calculés
    # sur les mêmes arbres à vol ± 1 %
    vol_derivative = OneDimDerivative(price_vs_vol, params, shift=0.01)
    res = {"vega": vol_derivative.first(market.vol),
           "vomma": vol_derivative.second(market.vol, price)}       # mêmes arbres que la vega
    res["vanna"] = OneDimDerivative(delta_vs_vol, params, shift=0.01).first(market.vol)
    res["theta"] = OneDimDerivative(price_vs_days, params, shift=1).first(0)
    # thêta funding : thêta sans l'effet d'actualisation de la prime (r * V par jour)
    res["theta_funding"] = res["theta"] - market.r * price / 365
    res["rho"] = OneDimDerivative(price_vs_rate, params, shift=0.01).first(market.r)
    return res


def print_results(option: Option, res: dict):
    """Affichage lisible des résultats de price_all."""
    print(f"{option.option_type} {option.exercise}, K = {option.strike}")
    print(f"  prix              : {res['prix']:.4f}")
    if "prix_europeen" in res:
        print(f"  prix europeen     : {res['prix_europeen']:.4f}   "
              f"(prime d'exercice anticipe : {res['prime_exercice_anticipe']:.4f})")
    if "delta" in res:                        # grecques, si elles ont été calculées
        print(f"  delta             : {res['delta']:+.4f}")
        print(f"  gamma             : {res['gamma']:+.4f}")
        print(f"  vega              : {res['vega'] / 100:+.4f}   pour 1 point de vol")
        print(f"  vomma             : {res['vomma'] / 10000:+.6f} pour 1 point de vol (au carré)")
        if res["tree_type"] != "trinomial":
            print("                      (vomma peu fiable avec des pas binomiaux : decaler la vol decale "
                  "toute la grille ; preferer l'arbre trinomial)")
        print(f"  vanna             : {res['vanna'] / 100:+.6f} variation du delta pour 1 point de vol")
        print(f"  theta             : {res['theta']:+.4f}   par jour")
        print(f"  theta funding     : {res['theta_funding']:+.4f}   par jour (sans l'actualisation)")
        print(f"  rho               : {res['rho'] / 100:+.4f}   pour 1 % de taux")
    # description de l'arbre, temps de calcul et mémoire
    bbs = ", methode BBS (Black-Scholes sur le dernier pas)" if res["bbs"] else ""
    print(f"  arbre {res['tree_type']}, construction {res['construction']}{bbs}")
    print(f"  N = {res['nb_steps']}, dt = {res['dt']:.6f}, alpha = {res['alpha']:.6f}, "
          f"{res['nb_nodes']} noeuds")
    print(f"  temps de pricing  : {res['temps_pricing']:.3f} s (construction + pricing), "
          f"temps total avec grecques : {res['temps']:.2f} s")
    if "memoire_mo" in res:
        print(f"  pic de memoire d'un pricing : {res['memoire_mo']:.2f} Mo, "
              f"{res['noeuds_max']} noeuds en memoire au maximum")
