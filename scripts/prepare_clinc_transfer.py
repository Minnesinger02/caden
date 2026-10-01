"""Freeze a CLINC150 in-scope transfer test without training on CLINC."""
import hashlib
import json
import os
from pathlib import Path
import random
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME', str(ROOT / '.cache/huggingface'))
revision = '155b9c710419136e17307b80d0a13e68cd46b4ec'
out = ROOT / 'data/clinc-transfer8'
if out.exists():
    raise FileExistsError(out)
ds = load_dataset('clinc/clinc_oos', 'plus', revision=revision, split='test')
names = ds.features['intent'].names
labels = [i for i, name in enumerate(names) if name not in {'oos', 'out_of_scope'}]
assert len(labels) == 150, names
excluded = set()
for split in ['train', 'dev', 'calibration', 'test']:
    for line in (ROOT / f'data/pilot16-5k/{split}.jsonl').read_text(encoding='utf-8').split('\n'):
        if line.strip():
            excluded.add(' '.join(json.loads(line)['state'].casefold().split()))
rows = []
seen = set()
for index, row in enumerate(ds):
    text = ' '.join(row['text'].casefold().split())
    if row['intent'] not in labels or text in excluded or text in seen:
        continue
    seen.add(text)
    digest = hashlib.sha256(text.encode()).hexdigest()
    rng = random.Random('42' + digest)
    chosen = [row['intent']] + rng.sample([i for i in labels if i != row['intent']], 7)
    rng.shuffle(chosen)
    rows.append({'id': f'clinc-test-{index}', 'group_id': digest, 'split': 'test', 'state': row['text'],
                 'question': "Which intent best describes the user's message?",
                 'criteria': {names[i]: names[i].replace('_', ' ') for i in chosen}, 'label': names[row['intent']],
                 'oracle_shortlist': True})
random.Random(42).shuffle(rows)
rows = rows[:300]
assert len(rows) == 300
out.mkdir()
path = out / 'test.jsonl'
path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')
manifest = {'dataset': 'clinc/clinc_oos', 'config': 'plus', 'revision': revision, 'source_split': 'test',
            'selection_seed': 42, 'items': 300, 'candidates': 8, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'scope': 'in-scope oracle shortlist transfer; OOS excluded; no CLINC training or development tuning',
            'deduplication': 'normalized exact text dedup, excludes every 5k BANKING77 split'}
(out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print(json.dumps(manifest, indent=2))
