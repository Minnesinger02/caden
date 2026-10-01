import math
import unittest

from decision_lab.common import metrics, softmax, validate_probs
from dynamic_candidates.analyze import fit_temperature, topk_renorm
from dynamic_candidates.experiment import perturb


class CoreTests(unittest.TestCase):
    def test_metrics_known_distribution_and_failure(self):
        m = metrics([{"label": "a", "probabilities": {"a": .8, "b": .2}}, {"label": "b", "error": "timeout"}])
        self.assertEqual(m["accuracy_all_failures_wrong"], .5)
        self.assertAlmostEqual(m["brier_sum"], .08)
        self.assertAlmostEqual(m["nll_clipped_1e-15"], -math.log(.8))
        self.assertAlmostEqual(m["ece_10_equal_width"], .2)

    def test_ties_cannot_be_split_by_threshold(self):
        m = metrics([{"label": "a", "probabilities": {"a": .5, "b": .5}}, {"label": "b", "probabilities": {"a": .5, "b": .5}}])
        self.assertEqual(len(m["risk_coverage_diagnostic_not_threshold_selection"]), 1)

    def test_probability_validation(self):
        for probs in ({"a": float("nan"), "b": 0}, {"a": .7, "b": .7}, {"a": 1}):
            with self.assertRaises(ValueError):
                validate_probs(probs, ["a", "b"])

    def test_topk_keeps_dropped_gold_in_evaluation(self):
        p, mass = topk_renorm({"a": .4, "b": .3, "c": .2, "d": .1}, 3)
        self.assertAlmostEqual(mass, .9)
        self.assertEqual(p["d"], 0)
        self.assertAlmostEqual(p["a"], 4 / 9)
        self.assertEqual(metrics([{"label": "d", "probabilities": p}])["accuracy_valid"], 0)

    def test_mutations_preserve_semantics(self):
        row = {"id": "one", "criteria": {"a": "one", "b": "two", "c": "three"}, "label": "b"}
        shuffled = perturb(row, "shuffle", 42)
        self.assertEqual(shuffled["label"], "b")
        self.assertEqual(shuffled["criteria"], row["criteria"])
        missing = perturb(row, "missing", 42)
        self.assertEqual(missing["label"], "__none__")
        self.assertNotIn("b", missing["criteria"])
        self.assertIn("b", row["criteria"])

    def test_temperature_never_fits_test(self):
        with self.assertRaises(ValueError):
            fit_temperature([{"split": "test"}])
        self.assertAlmostEqual(sum(softmax([1000, 1001])), 1)


if __name__ == "__main__":
    unittest.main()
