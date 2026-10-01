"""Compare identical held-out items; output descriptive metrics, not significance claims."""
import argparse
import csv
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('runs', nargs='+')
p.add_argument('--out', required=True)
a = p.parse_args()
reference = None
reference_hash = None
records = []
for folder in map(Path, a.runs):
    meta = json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
    data_hash = meta.get('dataset_sha256')
    if not data_hash:
        raise ValueError(f'Missing dataset hash: {folder}')
    if reference_hash is None:
        reference_hash = data_hash
    elif reference_hash != data_hash:
        raise ValueError('Mismatched data SHA256: identical IDs/labels alone do not establish identical questions/candidates')
    preds = [json.loads(s) for s in (folder/'predictions.jsonl').read_text(encoding='utf-8').splitlines() if s]
    identity = {(r['id'], r['label']) for r in preds}
    if len(identity) != len(preds):
        raise ValueError(f'Duplicate identities: {folder}')
    if reference is None:
        reference = identity
    elif identity != reference:
        raise ValueError('Mismatched items/labels: compare original conditions separately from missing-gold')
    m = json.loads((folder/'metrics.json').read_text(encoding='utf-8'))
    training = meta.get('training') or meta.get('adapter_training') or {}
    records.append({'run': str(folder), 'dataset_sha256': data_hash,
                    'model': training.get('model', '') if meta.get('backend') == 'encoder' else meta.get('model', ''),
                    'revision': meta.get('resolved_revision') or meta.get('revision') or training.get('resolved_revision', ''),
                    'seed': meta.get('seed', training.get('seed', '')),
                    'dtype': meta.get('dtype', 'float32' if meta.get('backend') == 'encoder' else ''),
                    'training_seconds': training.get('training_seconds', ''),
                    'training_peak_allocated_bytes': training.get('cuda_peak_allocated_bytes', ''),
                    'evaluation_peak_allocated_bytes': meta.get('cuda_peak_allocated_bytes', ''),
                    'failure_rate': m['failures'] / m['total'],
                    'latency_scope': meta.get('latency_scope', ''),
                    **{k: v for k, v in m.items() if not isinstance(v, (dict, list))},
                    **{f'latency_{k}': v for k,v in m.get('latency_ms', {}).items()}})
keys = list(dict.fromkeys(k for r in records for k in r))
with Path(a.out).open('x', encoding='utf-8-sig', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=keys)
    writer.writeheader()
    writer.writerows(records)
print(a.out)
