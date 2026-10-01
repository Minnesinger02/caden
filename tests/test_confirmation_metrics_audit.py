import unittest

from decision_lab.common import metrics
from scripts.summarize_fresh_clinc import audit_metrics


class ConfirmationAuditTests(unittest.TestCase):
    def setUp(self):
        self.data = [{'id': str(i), 'label': 'a', 'group_id': str(i), 'criteria': {'a': 'A', 'b': 'B'}} for i in range(3)]
        self.predictions = [{**{k: row[k] for k in ['id', 'label', 'group_id']}, 'probabilities': {'a': p, 'b': 1-p}}
                            for row, p in zip(self.data, [0., .5, 1.])]

    def test_matches_primary_metrics_at_probability_boundaries(self):
        result = audit_metrics(self.data, self.predictions)
        reference = metrics(self.predictions)
        for key, value in result.items():
            self.assertAlmostEqual(value, reference[key])

    def test_failed_prediction_stays_in_accuracy_denominator(self):
        self.predictions[2].pop('probabilities')
        self.predictions[2]['error'] = 'failed'
        result = audit_metrics(self.data, self.predictions)
        self.assertEqual(result['failures'], 1)
        self.assertAlmostEqual(result['accuracy_all_failures_wrong'], 1/3)

    def test_invalid_support_is_not_silently_clipped(self):
        self.predictions[0]['probabilities'] = {'wrong': 1.}
        with self.assertRaises(ValueError):
            audit_metrics(self.data, self.predictions)


if __name__ == '__main__':
    unittest.main()
