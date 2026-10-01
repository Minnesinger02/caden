import unittest
from scripts.summarize_order_probe import summarize


class OrderProbeTests(unittest.TestCase):
    def fixture(self):
        data = [{'id': f'q{i}', 'reference_id': 'q', 'condition': i, 'state': 'message',
                 'question': 'intent', 'label': 'a', 'group_id': 'group',
                 'criteria': dict([('a','one'), ('b','two')][::1 if i == 0 else -1])} for i in range(5)]
        predictions = [{'id': r['id'], 'label': 'a', 'probabilities': {'a': .9, 'b': .1}} for r in data]
        predictions[1]['probabilities'] = {'b': .9, 'a': .1}
        predictions[2] = {'id': 'q2', 'label': 'a', 'error': 'simulated runtime failure'}
        return data, predictions

    def test_repeated_conditions_are_one_question_and_failures_not_in_drift(self):
        data, predictions = self.fixture()
        report = summarize(data, predictions)
        self.assertEqual(report['source_questions'], 1)
        self.assertEqual(report['prediction_rows'], 5)
        self.assertEqual(report['failures'], 1)
        self.assertEqual(report['valid_comparisons'], 3)
        self.assertEqual(report['argmax_flips'], 1)
        self.assertEqual(report['question_count_with_flip'], 1)
        self.assertEqual(report['accuracy_by_condition']['2'], 0)
        self.assertAlmostEqual(report['max_probability_deviation'], .8)

    def test_changed_input_and_duplicate_predictions_rejected(self):
        data, predictions = self.fixture()
        with self.assertRaisesRegex(ValueError, 'ordering'):
            summarize([{**r, 'state': 'different'} if r['condition'] == 1 else r for r in data], predictions)
        with self.assertRaisesRegex(ValueError, 'IDs'):
            summarize(data, predictions + [predictions[0]])
        with self.assertRaisesRegex(ValueError, 'Duplicate condition'):
            summarize(data + [{**data[1], 'id':'extra'}], predictions + [{**predictions[1], 'id':'extra'}])


if __name__ == '__main__':
    unittest.main()
