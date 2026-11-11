# Tests unitaires du modèle (tree.py) : 5 tests, chacun affiche l'intuition financière qu'il vérifie.
# Lancement : python tests_tree.py   ou   run_tests() depuis le main.

import math
import unittest
from datetime import date

from tree import Market, Model, Option, black_scholes, price_tree

PRICING_DATE, MATURITY = date(2026, 9, 1), date(2027, 9, 1)
S0, K, R, VOL = 100, 100, 0.03, 0.25
T = (MATURITY - PRICING_DATE).days / 365
MARKET = Market(S0, R, VOL)
CALL, PUT = Option("call", K), Option("put", K)
DIVIDENDS = {date(2027, 3, 1): 2.0}


def tree_price(option: Option, n: int, market: Market = MARKET) -> float:
    """Prix d'une option sur un arbre trinomial à n pas, sans pruning."""
    model = Model(PRICING_DATE, MATURITY, n, tree_type="trinomial", prune_mode="none")
    return price_tree(market, model, option, "forward")


def explain(title: str, intuition: str, rows: list[tuple[str, float]]) -> None:
    """Affiche le titre du test, l'intuition financière derrière, puis les valeurs comparées."""
    print(f"\n- {title}")
    print(f"    Intuition : {intuition}")
    for label, value in rows:
        print(f"    {label:<28} {value:10.4f}")


class TestIntuition(unittest.TestCase):
    """Cinq propriétés financières que l'arbre doit respecter."""

    def test_call_strike_nul(self):
        """Call européen de strike 0 : équivaut à détenir l'action, donc vaut S0."""
        price = tree_price(Option("call", 0), 20)
        explain("S = 100, K = 0, call européen",
                "avoir le droit d'acheter pour 0 revient à posséder l'action : prix = S0 (forward actualisé)",
                [("Prix de l'arbre", price), ("S0", S0)])
        self.assertAlmostEqual(price, S0, places=8)

    def test_parite_call_put(self):
        """Parité : C - P = S - K e^(-rT)."""
        call, put = tree_price(CALL, 50), tree_price(PUT, 50)
        expected = S0 - K * math.exp(-R * T)
        explain("Parité call-put (européen, sans dividende)",
                "acheter un call et vendre un put = acheter l'action à terme au prix K : C - P = S - K e^(-rT)",
                [("Call", call), ("Put", put), ("C - P (arbre)", call - put), ("S - K e^(-rT)", expected)])
        self.assertAlmostEqual(call - put, expected, places=8)

    def test_convergence_black_scholes(self):
        """À 200 pas, le prix de l'arbre est à moins de 2 centimes de Black-Scholes."""
        tree = tree_price(CALL, 1000)
        exact = black_scholes(S0, K, R, VOL, T, "call")
        explain("Convergence vers Black-Scholes (10 00 pas)",
                "l'arbre discrétise le même modèle que Black-Scholes : plus de pas, même prix",
                [("Arbre", tree), ("Black-Scholes", exact), ("Écart", abs(tree - exact))])
        self.assertAlmostEqual(tree, exact, delta=0.02)

    def test_put_americain_superieur_a_l_europeen(self):
        """Un put américain vaut plus que le put européen : le droit d'exercer tôt a une valeur."""
        american, european = tree_price(Option("put", 110, "american"), 50), tree_price(Option("put", 110), 50)
        explain("Put américain vs européen (K = 110)",
                "un droit supplémentaire (exercer avant maturité) ne peut pas valoir moins : américain >= européen",
                [("Put américain", american), ("Put européen", european),
                 ("Prime d'exercice anticipé", american - european)])
        self.assertGreater(american, european)

    def test_dividende_baisse_le_call_et_monte_le_put(self):
        """Un dividende fait baisser le spot futur : le call perd de la valeur, le put en gagne."""
        market = Market(S0, R, VOL, DIVIDENDS)
        call_no, call_div = tree_price(CALL, 50), tree_price(CALL, 50, market)
        put_no, put_div = tree_price(PUT, 50), tree_price(PUT, 50, market)
        explain("Dividende de 2 versé en cours de vie",
                "le sous-jacent chute du montant du dividende : le call baisse, le put monte",
                [("Call sans dividende", call_no), ("Call avec dividende", call_div),
                 ("Put sans dividende", put_no), ("Put avec dividende", put_div)])
        self.assertLess(call_div, call_no)
        self.assertGreater(put_div, put_no)


def run_tests() -> unittest.TestResult:
    """Lance les tests, affiche l'intuition de chacun et un résumé, et renvoie le résultat."""
    print("TESTS UNITAIRES : propriétés financières de l'arbre")
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestIntuition)
    result = unittest.TextTestRunner(verbosity=0).run(suite)
    status = "TOUS LES TESTS PASSENT" if result.wasSuccessful() else "DES TESTS ONT ÉCHOUÉ"
    print(f"\n{status} : {result.testsRun} tests, {len(result.failures)} échec(s), {len(result.errors)} erreur(s)")
    return result


if __name__ == "__main__":
    run_tests()
