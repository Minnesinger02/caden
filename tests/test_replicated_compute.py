import unittest

from scripts.summarize_replicated_compute import ratio_summary


class TimingRatioTests(unittest.TestCase):
    def test_paired_common_slowdown_cancels(self):
        r = ratio_summary([20., 200., 2000.], [10., 100., 1000.])
        self.assertAlmostEqual(r['geometric_mean_speed_ratio'], 2.)
        for v in r['ratio_ci95']:
            self.assertAlmostEqual(v, 2.)

    def test_rejects_unaligned_missing_or_invalid_rates(self):
        for a, b in [([2., 2., 2.], [1.]), ([2., 2.], [1., 1.]), ([2., float('nan'), 2.], [1., 1., 1.]), ([0., 2., 2.], [1., 1., 1.])]:
            with self.assertRaises(ValueError):
                ratio_summary(a, b)


if __name__ == '__main__':
    unittest.main()
