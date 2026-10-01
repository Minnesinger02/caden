import collections
import unittest
from scripts.prepare_policy_transfer import flags, generate


class PolicyProbeTests(unittest.TestCase):
    def test_boundaries_and_multi_step_inversion(self):
        direct = dict(family='direct', threshold=40, amount=40, deadline=10, elapsed=10,
                      region='north', priority_regions=['north','east'])
        self.assertEqual(flags(direct), (True, False, True))
        self.assertEqual(flags({**direct, 'amount':39, 'elapsed':11, 'region':'west'}), (False, True, False))
        self.assertEqual(flags(dict(family='two_hop', threshold=40, amount=39,
                                    invert_priority=True, invert_hold=True)), (False, True, False))

    def test_all_routes_are_exhaustive_and_latent_targets_balanced(self):
        rows = generate()
        self.assertEqual(len(rows), 320)
        self.assertEqual(len({r['group_id'] for r in rows}), 320)
        for row in rows:
            self.assertEqual(len(row['criteria']), 8)
            self.assertEqual(tuple(row['route_flag_assignment'][int(row['label'].split('_')[1])]), flags(row['policy_spec']))
        for family in ['direct','two_hop']:
            counts = collections.Counter(flags(r['policy_spec']) for r in rows if r['family'] == family)
            self.assertEqual(set(counts.values()), {20})
            self.assertEqual(len(counts), 8)


if __name__ == '__main__':
    unittest.main()
