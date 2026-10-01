import unittest

from scripts.evaluate_saved_ensemble import combine


class SavedEnsembleTests(unittest.TestCase):
    def setUp(self):
        self.row = {'id': 'q', 'label': 'a', 'group_id': 'g', 'split': 'dev', 'criteria': {'a': 'A', 'b': 'B'}}
        self.members = [{**{k: self.row[k] for k in ['id', 'label', 'group_id']},
                         'probabilities': p, 'latency_ms': t}
                        for p, t in [({'a': .9, 'b': .1}, 3.), ({'b': .8, 'a': .2}, 4.)]]

    def test_aligns_by_candidate_key_and_reports_cost(self):
        result = combine(self.row, self.members)
        self.assertAlmostEqual(result['probabilities']['a'], .55)
        self.assertEqual(result['latency_ms'], 7.)
        self.assertTrue(result['latency_is_estimated_sequential_sum'])

    def test_failing_member_is_not_selectively_dropped(self):
        self.members[1].pop('probabilities')
        self.members[1]['error'] = 'failed'
        self.assertIn('error', combine(self.row, self.members))

    def test_rejects_changed_identity_or_support(self):
        self.members[1]['label'] = 'b'
        with self.assertRaises(ValueError):
            combine(self.row, self.members)


if __name__ == '__main__':
    unittest.main()
