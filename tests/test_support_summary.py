import unittest

from scripts.prepare_support_probe import perturb
from scripts.summarize_support_probe import summarize


class SupportSummaryTests(unittest.TestCase):
    def setUp(self):
        base = {'id': 'q', 'state': 'message', 'question': 'choose', 'split': 'dev', 'group_id': 'g', 'criteria': {'a': 'A', 'b': 'B', 'c': 'C'}, 'label': 'b'}
        self.data = [perturb(base, c, {'a': 'A', 'b': 'B', 'c': 'C', 'd': 'D'})
                     for c in ['original', 'drop_nongold', 'add_nongold', 'gold_absent']]
        self.predictions = [{**{k: row[k] for k in ['id', 'label', 'group_id', 'split']},
                             'probabilities': {k: 1/len(row['criteria']) for k in row['criteria']},
                             'logits': {k: 0. for k in row['criteria']}}
                            for row in self.data]

    def test_support_normalization_can_change_without_logit_drift(self):
        result = summarize(self.data, self.predictions)
        self.assertEqual(result['source_questions'], 1)
        for condition in ['drop_nongold', 'add_nongold', 'gold_absent']:
            self.assertEqual(result['conditions'][condition]['surviving_raw_logit_drift']['max_absolute_deviation'], 0.)
        self.assertNotEqual(self.predictions[0]['probabilities']['b'], self.predictions[2]['probabilities']['b'])

    def test_rejects_label_or_description_changes(self):
        self.data[1]['criteria']['b'] = 'different meaning'
        with self.assertRaises(ValueError):
            summarize(self.data, self.predictions)

    def test_failure_not_counted_as_drift_and_retains_denominator(self):
        self.predictions[1].pop('probabilities')
        self.predictions[1].pop('logits')
        self.predictions[1]['error'] = 'failed'
        result = summarize(self.data, self.predictions)
        self.assertEqual(result['conditions']['drop_nongold']['failures'], 1)
        self.assertEqual(result['conditions']['drop_nongold']['surviving_raw_logit_drift']['valid_question_comparisons'], 0)


if __name__ == '__main__':
    unittest.main()
