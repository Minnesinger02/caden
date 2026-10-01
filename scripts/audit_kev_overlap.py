"""Audit normalized exact-text overlap with one pinned public Kev training suite."""
import hashlib
import argparse
import json
import os
from pathlib import Path
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME', str(ROOT / '.cache/huggingface'))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data-dir', default='data/pilot16')
parser.add_argument('--splits', nargs='+', choices=['train','dev','calibration','test'], default=['train','dev','calibration','test'])
parser.add_argument('--out', default='handoff/kev-overlap-audit.json')
args = parser.parse_args()
revision = 'a88f56db5341397299137cb68775c2ea6e3f68cb'
path = Path(hf_hub_download('jaredpalmer/kev-suites', 'v7/decision-v7/train.jsonl',
                          repo_type='dataset', revision=revision, cache_dir=str(ROOT / '.cache/huggingface/hub')))
manifest = json.loads((ROOT / 'external/kev-upstream/evals/v7/decision-v7/manifest.json').read_text(encoding='utf-8'))
digest = hashlib.sha256(path.read_bytes()).hexdigest()
assert digest == manifest['files']['train.jsonl']['sha256'], 'Upstream suite hash mismatch'

def normalized(text):
    return ' '.join(text.casefold().split())

training = [json.loads(line) for line in path.read_text(encoding='utf-8').split('\n') if line.strip()]
texts = {normalized(row['state']) for row in training if isinstance(row['state'], str)}
origin_hashes = {row.get('_meta', {}).get('text_sha256') for row in training}
report = {'suite': 'v7/decision-v7', 'mirror_revision': revision, 'train_sha256': digest,
          'records': len(training), 'definition': 'casefolded exact state text with whitespace collapsed, or matching upstream origin text_sha256',
          'limitation': 'One public suite only; absence does not certify absence from all Kev training or base pretraining.',
          'splits': {}}
for split in args.splits:
    data_path = ROOT / args.data_dir / f'{split}.jsonl'
    rows = [json.loads(line) for line in data_path.read_text(encoding='utf-8').split('\n') if line.strip()]
    overlap = [row['id'] for row in rows if normalized(row['state']) in texts
               or hashlib.sha256(normalized(row['state']).encode()).hexdigest() in origin_hashes]
    report['splits'][split] = {'items': len(rows), 'overlap': len(overlap), 'ids': overlap,
                             'data_sha256': hashlib.sha256(data_path.read_bytes()).hexdigest()}
with (ROOT / args.out).open('x', encoding='utf-8') as stream:
    json.dump(report, stream, indent=2)
print(json.dumps({k: {a:b for a,b in v.items() if a != 'ids'} for k,v in report['splits'].items()}, indent=2))
