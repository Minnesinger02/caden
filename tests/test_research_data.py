import json
from pathlib import Path
import tempfile
import unittest

from decision_lab.common import read_data, write_run
from scripts.paired_bootstrap import read_runs

class ResearchDataTests(unittest.TestCase):
    def test_utf8_roundtrip_failed_item_and_pair_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / 'data.jsonl'
            row = {'id': '1', 'state': '欧元 € café', 'question': '哪一个？',
                   'criteria': {'a': '甲', 'b': '乙'}, 'label': 'a', 'group_id': 'g', 'split': 'dev'}
            data.write_text(json.dumps(row, ensure_ascii=False) + '\n', encoding='utf-8')
            self.assertEqual(read_data(data)[0]['state'], row['state'])
            predictions = [{**row, 'probabilities': {'a': .8, 'b': .2}}]
            write_run(root / 'good', data, predictions, {'note': '测试'})
            failed = [{**row, 'error': 'unavailable'}]
            write_run(root / 'failed', data, failed, {})
            scores, identity, groups, _ = read_runs([root / 'good', root / 'failed'])
            self.assertEqual(scores.tolist(), [[1.0], [0.0]])
            self.assertEqual(groups, ['g'])
            self.assertEqual(identity, [('1', 'a', 'g')])
            changed = json.loads((root / 'failed/metadata.json').read_text(encoding='utf-8'))
            changed['dataset_sha256'] = 'different'
            (root / 'failed/metadata.json').write_text(json.dumps(changed), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'hashes'):
                read_runs([root / 'good', root / 'failed'])

if __name__ == '__main__':
    unittest.main()
