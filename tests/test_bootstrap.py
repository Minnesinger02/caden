import unittest
import numpy as np
from scripts.paired_bootstrap import paired_interval

class BootstrapTests(unittest.TestCase):
    def test_known_differences_and_group_weighting(self):
        a = np.array([[1., 1., 1.], [0., 0., 0.]])
        b = np.zeros((1, 3))
        delta, interval, groups = paired_interval(a, b, ['same', 'same', 'other'], 1000)
        self.assertEqual(delta, .5)
        self.assertEqual(interval, [.5, .5])
        self.assertEqual(groups, 2)
        # One group is always sampled as a whole; the mean is item weighted.
        delta, interval, groups = paired_interval(np.array([[1., 1., 0.]]), b, ['g', 'g', 'g'], 1000)
        self.assertAlmostEqual(delta, 2/3)
        self.assertEqual(interval, [2/3, 2/3])
        self.assertEqual(groups, 1)
        _, zero, _ = paired_interval(b, b, ['g', 'h', 'i'], 1000)
        self.assertEqual(zero, [0., 0.])
        with self.assertRaises(ValueError):
            paired_interval(a, b, ['g'], 100)

if __name__ == '__main__':
    unittest.main()
