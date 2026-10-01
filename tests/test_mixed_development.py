import unittest

from scripts.summarize_mixed_development import summarize, aggregate_complete_seeds, audit_recipe


class MixedDevelopmentTests(unittest.TestCase):
    def setUp(self):
        self.data = [{'id': str(i), 'group_id': str(i), 'split': 'dev', 'domain': domain,
                      'criteria': {'a': 'A', 'b': 'B'}, 'label': 'a'}
                     for i, domain in enumerate(['banking', 'clinc', 'clinc', 'clinc'])]
        self.predictions = [{'id': r['id'], 'group_id': r['group_id'], 'label': 'a',
                             'probabilities': {'a': .9 if i == 0 else .1, 'b': .1 if i == 0 else .9}}
                            for i, r in enumerate(self.data)]

    def test_macro_weights_domains_equally(self):
        result = summarize(self.data, self.predictions)
        self.assertEqual(result['pooled']['accuracy_all_failures_wrong'], .25)
        self.assertEqual(result['macro_domain_accuracy'], .5)

    def test_failures_remain_in_domain_denominator(self):
        self.predictions[0].pop('probabilities')
        self.predictions[0]['error'] = 'inference failed'
        result = summarize(self.data, self.predictions)
        self.assertEqual(result['banking']['failures'], 1)
        self.assertEqual(result['macro_domain_accuracy'], 0.)

    def test_rejects_duplicate_identity_and_candidate_support(self):
        with self.assertRaises(ValueError):
            summarize(self.data, [self.predictions[0]] * 4)
        self.predictions[0]['probabilities'] = {'wrong': .5, 'b': .5}
        with self.assertRaises(ValueError):
            summarize(self.data, self.predictions)

    def test_incomplete_seed_set_has_no_aggregate(self):
        result = summarize(self.data, self.predictions)
        runs = [{'family': 'encoder', 'seed': seed, **result} for seed in [42, 43]]
        self.assertIsNone(aggregate_complete_seeds(runs, 'encoder'))

    def test_complete_seed_mean_and_sample_sd(self):
        runs = []
        for seed, correct_count in zip([42, 43, 44], [0, 2, 4]):
            predictions = [{**row, 'probabilities': {'a': .9 if i < correct_count else .1, 'b': .1 if i < correct_count else .9}}
                           for i, row in enumerate(self.predictions)]
            runs.append({'family': 'encoder', 'seed': seed, **summarize(self.data, predictions)})
        result = aggregate_complete_seeds(runs, 'encoder')
        self.assertEqual(result['metrics']['pooled.accuracy_all_failures_wrong'], {'mean': .5, 'sample_sd': .5})

    def test_duplicate_or_unregistered_seeds_rejected(self):
        result = summarize(self.data, self.predictions)
        for seeds in [[42, 42, 44], [42, 43, 99]]:
            runs = [{'family': 'encoder', 'seed': seed, **result} for seed in seeds]
            with self.assertRaises(ValueError):
                aggregate_complete_seeds(runs, 'encoder')

    def test_all_failed_domain_keeps_accuracy_zero_and_quality_undefined(self):
        failures = [{'id': row['id'], 'group_id': row['group_id'], 'label': row['label'], 'error': 'failed'} for row in self.data]
        result = summarize(self.data, failures)
        runs = [{'family': 'encoder', 'seed': seed, **result} for seed in [42, 43, 44]]
        aggregate = aggregate_complete_seeds(runs, 'encoder')
        self.assertEqual(aggregate['metrics']['pooled.accuracy_all_failures_wrong'], {'mean': 0., 'sample_sd': 0.})
        self.assertIsNone(aggregate['metrics']['pooled.nll_clipped_1e-15'])
        self.assertEqual(aggregate['failures_total'], 12)

    def test_qwen_training_hash_field_and_wrong_seed_rejected(self):
        training = {'seed': 42, 'max_length': 512, 'data': 'data/mixed-banking-clinc8/train.jsonl',
                    'dataset_sha256': 'frozen-train-hash', 'model': 'Qwen/Qwen3-0.6B',
                    'resolved_revision': 'c1899de289a04d12100db370d81485cdf75e47ca', 'epochs': 1,
                    'lr': 1e-4, 'objective': 'candidate_ce', 'rank': 8, 'accumulation': 8, 'dtype': 'float32'}
        metadata = {'adapter_training': training, 'dtype': 'float32', 'torch_num_threads': 2}
        audit_recipe(metadata, 42, 'qwen', 'frozen-train-hash')
        with self.assertRaises(ValueError):
            audit_recipe(metadata, 43, 'qwen', 'frozen-train-hash')
        with self.assertRaises(ValueError):
            audit_recipe(metadata, 42, 'qwen', 'other-pool-hash')
        legacy = {'adapter_training': training, 'dtype': 'float32'}
        audit_recipe(legacy, 42, 'qwen', 'frozen-train-hash')
        with self.assertRaises(ValueError):
            audit_recipe({**metadata, 'torch_num_threads': 8}, 42, 'qwen', 'frozen-train-hash')


if __name__ == '__main__':
    unittest.main()
