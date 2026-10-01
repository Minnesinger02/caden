import unittest
from decision_lab.common import normalize_rounded_probs


class WireRoundingTests(unittest.TestCase):
    def test_known_rounding_error_preserves_support_winner_and_rejects_bad_payloads(self):
        rounded = {'a': .3334, 'b': .3333, 'c': .3334}
        normalized = normalize_rounded_probs(rounded, rounded)
        self.assertAlmostEqual(sum(normalized.values()), 1)
        self.assertEqual(max(normalized, key=normalized.get), max(rounded, key=rounded.get))
        self.assertEqual(set(normalized), set(rounded))
        for bad in [{'a': .33, 'b': .33, 'c': .33}, {'a': -0.01, 'b': .5, 'c': .51},
                    {'a': float('nan'), 'b': .5, 'c': .5}]:
            with self.assertRaises(ValueError):
                normalize_rounded_probs(bad, rounded)
        with self.assertRaises(ValueError):
            normalize_rounded_probs(rounded, {'a', 'b'})


if __name__ == '__main__':
    unittest.main()
