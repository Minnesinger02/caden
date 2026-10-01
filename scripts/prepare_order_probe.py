"""Freeze four candidate permutations per development item before evaluating."""
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from decision_lab.common import read_data

source = ROOT / 'data/pilot16/dev.jsonl'
output = ROOT / 'data/order-dev8'
if output.exists():
    raise FileExistsError(output)
rows = read_data(source)
probes = []
for row in rows:
    for condition in range(5):
        options = list(row['criteria'].items())
        if condition:
            random.Random(f'order-probe-v1:{row["id"]}:{condition}').shuffle(options)
            if options == list(row['criteria'].items()):
                options = options[1:] + options[:1]
        probes.append({**row, 'id': f'{row["id"]}__order{condition}', 'reference_id': row['id'],
                       'condition': condition, 'criteria': dict(options)})
# Mix conditions rather than measuring all originals under one thermal state.
random.Random(42).shuffle(probes)
output.mkdir()
path = output / 'dev.jsonl'
path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in probes), encoding='utf-8')
manifest = {'items': len(probes), 'independent_questions': len(rows), 'conditions': 'original plus four deterministic nonidentity candidate permutations',
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'split': 'development only', 'evaluation_order_seed': 42,
            'invariants': 'same state, question, gold, group, candidate keys and descriptions; only candidate order differs',
            'statistical_unit': 'source question/group; five rows per question are not independent observations'}
(output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print(json.dumps(manifest, indent=2))
