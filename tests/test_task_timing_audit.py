import unittest

from scripts.summarize_replicated_task_timing import audit_run


class TaskTimingAuditTests(unittest.TestCase):
    def setUp(self):
        self.data = [{'id': str(i), 'label': 'a', 'group_id': str(i), 'criteria': {'a': 'A', 'b': 'B'}} for i in range(2)]
        self.predictions = [{**{k: row[k] for k in ['id', 'label', 'group_id']},
                             'probabilities': {'b': .1, 'a': .9}, 'latency_ms': value}
                            for row, value in zip(self.data, [10., 30.])]

    def test_throughput_uses_total_time_not_inverse_median(self):
        result = audit_run(self.data, self.predictions)
        self.assertEqual(result['decisions_per_second'], 50.)
        self.assertEqual(result['p50_ms'], 20.)

    def test_failed_fast_responses_do_not_inflate_decision_rate(self):
        self.predictions[0]['error'] = 'failed'
        with self.assertRaises(ValueError):
            audit_run(self.data, self.predictions)

    def test_rejects_missing_item_or_changed_probability_support(self):
        with self.assertRaises(ValueError):
            audit_run(self.data, self.predictions[:1])
        self.predictions[0]['probabilities'] = {'other': 1.}
        with self.assertRaises(ValueError):
            audit_run(self.data, self.predictions)


if __name__ == '__main__':
    unittest.main()
