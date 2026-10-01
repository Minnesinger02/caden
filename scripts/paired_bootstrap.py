"""Group-paired accuracy intervals conditional on saved training seeds."""
import argparse
import json
from pathlib import Path

import numpy as np


def read_runs(folders):
    arrays, reference, groups, data_hash = [], None, None, None
    for folder in map(Path, folders):
        meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        if data_hash is not None and meta['dataset_sha256'] != data_hash:
            raise ValueError("Mismatched data hashes")
        data_hash = meta['dataset_sha256']
        rows = [json.loads(line) for line in (folder / "predictions.jsonl").read_text(encoding="utf-8").splitlines() if line]
        indexed = {row['id']: row for row in rows}
        identity = sorted((r['id'], r['label'], r.get('group_id', r['id'])) for r in rows)
        if len(indexed) != len(rows) or (reference is not None and identity != reference):
            raise ValueError("Duplicate/mismatched items, labels or groups")
        reference = identity
        groups = [r[2] for r in identity]
        scores = []
        for item, _, _ in identity:
            row = indexed[item]
            p = row.get('probabilities')
            scores.append(float('error' not in row and bool(p) and max(p, key=p.get) == row['label']))
        arrays.append(scores)
    return np.asarray(arrays), reference, groups, data_hash


def paired_interval(candidate, reference, groups, resamples=5000, seed=0):
    if candidate.ndim != 2 or reference.ndim != 2 or candidate.shape[1] != reference.shape[1] or candidate.shape[1] != len(groups):
        raise ValueError('Score arrays must share nonempty item dimension and groups')
    if not len(groups) or not candidate.shape[0] or not reference.shape[0] or resamples < 1:
        raise ValueError('Nonempty scores and positive resamples required')
    delta = candidate.mean(axis=0) - reference.mean(axis=0)
    unique = sorted(set(groups))
    sums = np.asarray([sum(delta[i] for i, g in enumerate(groups) if g == name) for name in unique])
    counts = np.asarray([groups.count(name) for name in unique])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(unique), size=(resamples, len(unique)))
    samples = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return float(delta.mean()), np.quantile(samples, [.025, .975]).tolist(), len(unique)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', nargs='+', required=True)
    parser.add_argument('--reference', nargs='+', required=True)
    parser.add_argument('--resamples', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    candidate, items, groups, data_hash = read_runs(args.candidate)
    reference, other_items, _, other_hash = read_runs(args.reference)
    if items != other_items or data_hash != other_hash:
        raise ValueError('Comparison requires the exact same items/data')
    delta, interval, groups_count = paired_interval(candidate, reference, groups, args.resamples, args.seed)
    report = {'settings': vars(args), 'dataset_sha256': data_hash, 'items': len(items), 'groups': groups_count,
              'candidate_accuracy_by_seed': candidate.mean(axis=1).tolist(),
              'reference_accuracy_by_seed': reference.mean(axis=1).tolist(),
              'accuracy_delta': delta, 'delta_ci95': interval,
              'interpretation': 'candidate minus reference; paired resampling of material groups; failures counted wrong; interval conditional on observed seeds, not a universal architecture claim'}
    with Path(args.out).open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
