"""Freeze supervised BANKING+CLINC expansion with a fresh CLINC confirmation pool."""
import hashlib
import json
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('HF_HOME', str(ROOT / '.cache/huggingface'))
from datasets import load_dataset
from decision_lab.common import read_data


def main():
    output = ROOT / 'data/mixed-banking-clinc8'
    if output.exists():
        raise FileExistsError(output)
    revision = '155b9c710419136e17307b80d0a13e68cd46b4ec'
    ds = load_dataset('clinc/clinc_oos', 'plus', revision=revision)
    sources = {}
    for split, dataset in ds.items():
        files = [Path(f['filename']) for f in dataset.cache_files]
        if not files or any(revision not in str(p) for p in files):
            raise ValueError('Cached CLINC data do not match the declared fixed revision')
        sources[split] = [{'file': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]
    names = ds['train'].features['intent'].names
    labels = [i for i, name in enumerate(names) if name not in {'oos', 'out_of_scope'}]
    if len(labels) != 150:
        raise ValueError('Expected 150 in-scope intents')
    splits = {split: [{**r, 'domain': 'banking'} for r in read_data(ROOT / source)]
              for split, source in {'train': 'data/pilot16-8792/train.jsonl', 'dev': 'data/pilot16/dev.jsonl',
                                    'calibration': 'data/confirmation8-full/calibration.jsonl',
                                    'test': 'data/confirmation8-full/test.jsonl'}.items()}
    seen = {' '.join(r['state'].casefold().split()) for rows in splits.values() for r in rows}
    old_transfer = read_data(ROOT / 'data/clinc-transfer8/test.jsonl')
    observed = {' '.join(r['state'].casefold().split()) for r in old_transfer}
    seen |= observed
    def make(row, index, source_split, split):
        text = ' '.join(row['text'].casefold().split())
        if row['intent'] not in labels or text in seen:
            return None
        seen.add(text)
        group = hashlib.sha256(text.encode()).hexdigest()
        rng = random.Random('mixed-clinc-v1:' + group)
        candidates = [row['intent']] + rng.sample([i for i in labels if i != row['intent']], 7)
        rng.shuffle(candidates)
        return {'id': f'mixed-clinc-{source_split}-{index}', 'group_id': group, 'split': split,
                'state': row['text'], 'question': "Which intent best describes the user's message?",
                'criteria': {names[i]: names[i].replace('_', ' ') for i in candidates},
                'label': names[row['intent']], 'oracle_shortlist': True, 'domain': 'clinc'}
    for index, row in enumerate(ds['train']):
        created = make(row, index, 'train', 'train')
        if created:
            splits['train'].append(created)
    counts = {i: 0 for i in labels}
    validation_indices = list(range(len(ds['validation'])))
    random.Random(1729).shuffle(validation_indices)
    for index in validation_indices:
        row = ds['validation'][index]
        if row['intent'] not in labels:
            continue
        split = 'dev' if counts[row['intent']] < 4 else 'calibration'
        created = make(row, index, 'validation', split)
        if created:
            splits[split].append(created)
            counts[row['intent']] += 1
    for index, row in enumerate(ds['test']):
        created = make(row, index, 'test', 'test')
        if created:
            splits['test'].append(created)
    used = set()
    for split, rows in splits.items():
        groups = {r['group_id'] for r in rows}
        if len(groups) != len(rows) or used & groups:
            raise ValueError('Mixed dataset contains duplicate groups or cross-split leakage')
        used |= groups
        random.Random('mixed-shuffle:' + split).shuffle(rows)
    output.mkdir()
    files = {}
    for split, rows in splits.items():
        path = output / f'{split}.jsonl'
        path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
        files[split] = {'items': len(rows), 'banking': sum(r['domain'] == 'banking' for r in rows),
                        'clinc': sum(r['domain'] == 'clinc' for r in rows),
                        'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    manifest = {'files': files, 'clinc_revision': revision, 'clinc_source_cache': sources,
                'banking_training_items_preserved': 8792, 'clinc_intents': 150, 'oos_excluded': True,
                'old_clinc_probe_items_excluded': len(old_transfer),
                'scope': 'supervised multidomain expansion; local models will explicitly train on CLINC, so this stage is not zero-shot transfer',
                'confirmation_status': 'CLINC test items exclude all 300 previously scored probe texts and are unscored when frozen; BANKING test was already reported and subsequent scores are exploratory',
                'selection': 'train hyperparameters/checkpoints selected only from separate BANKING/CLINC development; report per-domain metrics, no test-based seed selection',
                'task': 'oracle eight candidates within each domain; not original 77-way/150-way classification'}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'files': files, 'fresh_clinc_test': files['test']['clinc']}, indent=2))


if __name__ == '__main__':
    main()
