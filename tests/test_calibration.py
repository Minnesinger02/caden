import unittest
from scripts.calibrate_probabilities import distribution, fit_temperature, row_logits

class CalibrationTests(unittest.TestCase):
    def test_legacy_decoder_score_lists_use_declared_order(self):
        row = {'probabilities': {'a': .8, 'b': .2}, 'logits': [-2, 0], 'candidate_order': ['b', 'a']}
        self.assertEqual(row_logits(row), {'b': -2, 'a': 0})
        with self.assertRaisesRegex(ValueError, 'order'):
            row_logits({**row, 'candidate_order': ['a', 'a']})
        with self.assertRaisesRegex(ValueError, 'order'):
            row_logits({k: v for k, v in row.items() if k != 'candidate_order'})

    def test_raw_scores_recover_information_lost_to_probability_underflow(self):
        p = {'a': 1.0, 'b': 0.0, 'c': 0.0}
        calibrated = distribution(p, 50, {'c': -300.0, 'b': -200.0, 'a': 0.0})
        self.assertGreater(calibrated['b'], calibrated['c'])
        self.assertAlmostEqual(calibrated['b'] / calibrated['c'], __import__('math').exp(2))
        self.assertEqual(max(calibrated, key=calibrated.get), 'a')
        with self.assertRaisesRegex(ValueError, 'support'):
            distribution(p, 2, {'a': 0.0, 'b': -200.0})
        with self.assertRaisesRegex(ValueError, 'finite'):
            distribution(p, 2, {'a': 0.0, 'b': float('nan'), 'c': -300.0})
        with self.assertRaisesRegex(ValueError, 'consistent'):
            fit_temperature([{'split': 'calibration', 'label': 'a', 'probabilities': p, 'logits': {'a': 0, 'b': -200, 'c': -300}},
                             {'split': 'calibration', 'label': 'a', 'probabilities': p}])

    def test_calibration_improves_overconfidence_preserves_winners_and_requires_split(self):
        rows = [{'split': 'calibration', 'label': ('a' if i < 7 else 'b'),
                 'probabilities': {'a': .99, 'b': .01}} for i in range(10)]
        fit = fit_temperature(rows)
        self.assertGreater(fit['temperature'], 1)
        self.assertLess(fit['calibration_nll_after'], fit['calibration_nll_before'])
        p = distribution(rows[0]['probabilities'], fit['temperature'])
        self.assertAlmostEqual(p['a'], .7, places=5)
        self.assertEqual(max(p, key=p.get), 'a')
        self.assertAlmostEqual(sum(p.values()), 1)
        with self.assertRaisesRegex(ValueError, 'calibration'):
            fit_temperature([{**r, 'split': 'test'} for r in rows])
        with self.assertRaises(ValueError):
            distribution({'a': .8, 'b': .2}, 0)

if __name__ == '__main__':
    unittest.main()
