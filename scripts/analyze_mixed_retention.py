"""Compare the unchanged BANKING development questions before/after domain expansion."""
import hashlib
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from decision_lab.common import read_data, validate_probs
from scripts.paired_bootstrap import paired_interval


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', choices=['encoder', 'qwen'], default='encoder')
    args = parser.parse_args()
    old_path = ROOT / 'data/pilot16-8792/dev.jsonl'
    mixed_path = ROOT / 'data/mixed-banking-clinc8/dev.jsonl'
    old_data = read_data(old_path)
    mixed_data = {r['id']: r for r in read_data(mixed_path) if r['domain'] == 'banking'}
    if {r['id'] for r in old_data} != set(mixed_data):
        raise ValueError('BANKING development question support changed')
    for r in old_data:
        for key in ['state', 'question', 'criteria', 'label', 'group_id']:
            if r[key] != mixed_data[r['id']][key]:
                raise ValueError('BANKING question semantics changed')
    report = {'scope': 'exploratory retention on the same 200 BANKING development questions; no fresh test scores; mixed CLINC development is a different pool from the old 300-question transfer probe', 'comparisons': [], 'pending_seeds': []}
    complete_before, complete_after = [], []
    for seed in [42, 43, 44]:
        if args.family == 'encoder':
            folders = [ROOT / f'results/encoder16-s{seed}-e3-8792-dev', ROOT / f'results/encoder16-s{seed}-e3-mixed23789-dev']
        else:
            folders = [ROOT / f'results/sft-eval-s{seed}-train8792-dev', ROOT / f'results/sft-eval-s{seed}-trainmixed23789-dev']
        if not all((f / 'metrics.json').exists() for f in folders):
            report['pending_seeds'].append(seed)
            continue
        scores = []
        for folder, data_path in zip(folders, [old_path, mixed_path]):
            metadata = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
            if metadata['dataset_sha256'] != hashlib.sha256(data_path.read_bytes()).hexdigest():
                raise ValueError('Trace differs from frozen data')
            training = metadata.get('training', metadata.get('adapter_training', {}))
            train_path = ROOT / ('data/pilot16-8792/train.jsonl' if data_path == old_path else 'data/mixed-banking-clinc8/train.jsonl')
            if training.get('seed') != seed or (ROOT / training['data']).resolve() != train_path.resolve() or training.get('data_sha256', training.get('dataset_sha256')) != hashlib.sha256(train_path.read_bytes()).hexdigest():
                raise ValueError('Retention seed or training pool differs')
            rows = [json.loads(line) for line in (folder / 'predictions.jsonl').read_text(encoding='utf-8').splitlines() if line]
            indexed = {r['id']: r for r in rows}
            source = read_data(data_path)
            if len(indexed) != len(rows) or set(indexed) != {row['id'] for row in source}:
                raise ValueError('Incomplete or duplicate prediction IDs')
            for row in source:
                pred = indexed[row['id']]
                if (pred['label'], pred['group_id']) != (row['label'], row['group_id']):
                    raise ValueError('Prediction identity changed')
                if 'error' in pred and 'probabilities' in pred:
                    raise ValueError('Ambiguous retention failure')
                if 'error' not in pred:
                    validate_probs(pred['probabilities'], row['criteria'])
            values = []
            for r in old_data:
                pred = indexed[r['id']]
                if (pred['label'], pred['group_id']) != (r['label'], r['group_id']):
                    raise ValueError('Prediction identity changed')
                if 'error' in pred:
                    values.append(0.)
                else:
                    validate_probs(pred['probabilities'], r['criteria'])
                    values.append(float(max(pred['probabilities'], key=pred['probabilities'].get) == r['label']))
            scores.append(np.asarray(values))
        before, after = scores
        complete_before.append(before)
        complete_after.append(after)
        delta, interval, groups = paired_interval(after[None, :], before[None, :], [r['group_id'] for r in old_data])
        report['comparisons'].append({'family': args.family, 'seed': seed, 'questions': len(old_data), 'baseline_accuracy': float(before.mean()),
                                      'mixed_accuracy': float(after.mean()), 'lost_correct_answers': int(((before == 1) & (after == 0)).sum()),
                                      'gained_correct_answers': int(((before == 0) & (after == 1)).sum()),
                                      'delta': delta, 'paired_ci95': interval, 'groups': groups, 'resamples': 5000,
                                      'interpretation': 'single observed seed development interval; no universal forgetting conclusion'})
    report['complete_seed_summary'] = None
    if not report['pending_seeds']:
        before = np.stack(complete_before)
        after = np.stack(complete_after)
        delta, interval, groups = paired_interval(after, before, [row['group_id'] for row in old_data])
        report['complete_seed_summary'] = {'family': args.family, 'seeds': [42, 43, 44], 'questions': len(old_data),
            'baseline_accuracy_mean': float(before.mean()), 'baseline_accuracy_sample_sd': float(before.mean(axis=1).std(ddof=1)),
            'mixed_accuracy_mean': float(after.mean()), 'mixed_accuracy_sample_sd': float(after.mean(axis=1).std(ddof=1)),
            'delta': delta, 'paired_ci95': interval, 'groups': groups, 'resamples': 5000,
            'interpretation': 'development paired group bootstrap conditional on the three observed seeds; source questions are the units; not a population-over-seeds interval or fresh test confirmation'}
    name = 'mixed-banking-retention.json' if args.family == 'encoder' else 'mixed-qwen-banking-retention.json'
    (ROOT / 'results' / name).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
