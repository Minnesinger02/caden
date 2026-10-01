"""Offline checks for financial stop gates; never contact the paid API."""
import concurrent.futures
import unittest

from scripts.run_jev_budgeted_comparison import CAP, PRICE, RESERVE, Meter, normalize_wire


class JevBudgetSafetyTests(unittest.TestCase):
    def test_concurrent_reservations_cannot_exceed_cap(self):
        meter = Meter(CAP - 3 * RESERVE - 1e-9, requests=0)

        def attempt(_):
            try:
                meter.start()
                return True
            except RuntimeError:
                return False

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            accepted = list(pool.map(attempt, range(20)))
        self.assertEqual(sum(accepted), 3)
        self.assertEqual(meter.requests, 3)
        self.assertLessEqual(meter.spent + meter.reserved, CAP)

    def test_known_usage_releases_only_its_reservation(self):
        meter = Meter(.1, requests=0)
        meter.start()
        meter.start()
        meter.finish(431)
        self.assertAlmostEqual(meter.spent, .1 + 431 * PRICE)
        self.assertAlmostEqual(meter.reserved, RESERVE)

    def test_unknown_usage_stops_and_keeps_worst_case_charge(self):
        meter = Meter(.1, requests=0)
        meter.start()
        with self.assertRaisesRegex(RuntimeError, 'Unknown usage'):
            meter.finish(None)
        self.assertEqual(meter.unknown, 1)
        self.assertAlmostEqual(meter.spent, .1 + RESERVE)
        self.assertAlmostEqual(meter.reserved, 0)

    def test_rounded_sum_preserves_relative_probabilities(self):
        raw = {'a': .33, 'b': .33, 'c': .33}
        normalized, digits = normalize_wire(raw, raw)
        self.assertEqual(digits, 2)
        self.assertAlmostEqual(sum(normalized.values()), 1)
        self.assertEqual(raw['a'], .33)
        self.assertAlmostEqual(normalized['a'], 1 / 3)

    def test_rounding_tolerance_does_not_accept_invalid_distribution(self):
        with self.assertRaises(ValueError):
            normalize_wire({'a': .7, 'b': .7}, {'a': 'A', 'b': 'B'})


if __name__ == '__main__':
    unittest.main()
