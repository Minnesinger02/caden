import unittest

from scripts.prepare_support_probe import perturb, NONE_KEY


class SupportProbeTests(unittest.TestCase):
    def setUp(self):
        self.row = {'id': 'q', 'state': 'message', 'question': 'choose', 'criteria': {'a': 'A', 'b': 'B', 'c': 'C'}, 'label': 'b'}
        self.vocabulary = {**self.row['criteria'], 'd': 'D'}

    def test_drop_and_add_preserve_gold_and_candidate_descriptions(self):
        for condition, size in [('drop_nongold', 2), ('add_nongold', 4)]:
            changed = perturb(self.row, condition, self.vocabulary)
            self.assertEqual(len(changed['criteria']), size)
            self.assertEqual(changed['label'], 'b')
            for key in set(changed['criteria']) & set(self.row['criteria']):
                self.assertEqual(changed['criteria'][key], self.row['criteria'][key])

    def test_gold_absent_has_explicit_valid_target(self):
        changed = perturb(self.row, 'gold_absent', self.vocabulary)
        self.assertNotIn('b', changed['criteria'])
        self.assertEqual(changed['label'], NONE_KEY)
        self.assertFalse(changed['oracle_shortlist'])
        self.assertEqual(changed['state'], self.row['state'])


if __name__ == '__main__':
    unittest.main()
