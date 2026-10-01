"""Freeze a nested 5000-item training arm; preserve pilot held-out bytes."""
import hashlib
import argparse
import json
from pathlib import Path
import random
import shutil

root = Path(__file__).resolve().parents[1]
pilot = root / 'data/pilot16'
source = root / 'data/banking77-8-parquet'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--items', type=int, default=5000)
parser.add_argument('--out', default='data/pilot16-5k')
args = parser.parse_args()
output = root / args.out
if output.exists():
    raise FileExistsError(output)

def read(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').split('\n') if line.strip()]

old = {split: read(pilot / f'{split}.jsonl') for split in ['train', 'dev', 'calibration', 'test']}
excluded = {row['group_id'] for rows in old.values() for row in rows}
remaining = [row for row in read(source / 'train.jsonl') if row['group_id'] not in excluded]
random.Random(42).shuffle(remaining)
if not len(old['train']) <= args.items <= len(old['train']) + len(remaining):
    raise ValueError('Requested training size outside available disjoint pool')
train = old['train'] + remaining[:args.items-len(old['train'])]
assert len(train) == args.items and len({r['group_id'] for r in train}) == args.items
assert not {r['group_id'] for r in train} & {r['group_id'] for split in ['dev', 'calibration', 'test'] for r in old[split]}
output.mkdir()
(output / 'train.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in train), encoding='utf-8')
for split in ['dev', 'calibration', 'test']:
    shutil.copyfile(pilot / f'{split}.jsonl', output / f'{split}.jsonl')
hashes = {p.stem: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.glob('*.jsonl')}
manifest = {'parent': 'data/pilot16', 'source': json.loads((source / 'manifest.json').read_text(encoding='utf-8')),
            'nested_train': True, 'selection_seed': 42, 'counts': {'train': args.items, 'dev': 200, 'calibration': 200, 'test': 200}, 'sha256': hashes}
(output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print(json.dumps(manifest, indent=2))
