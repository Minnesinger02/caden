"""Exploratory matched BANKING accuracy gaps, conditional on saved checkpoints."""
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from decision_lab.common import read_data
from scripts.paired_bootstrap import read_runs, paired_interval
from scripts.summarize_fresh_clinc import audit_metrics


def main():
    data_path = ROOT / 'data/confirmation8-full/test.jsonl'
    digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
    if digest != '444a292af999764ff91c9be649b56a51633691f2fea967cca93196c1d89437b3':
        raise ValueError('Frozen BANKING test changed')
    data = read_data(data_path)
    identity = sorted((row['id'], row['label'], row['group_id']) for row in data)

    def audit(folders):
        traces = []
        for folder in folders:
            metadata = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
            predictions = [json.loads(line) for line in (folder / 'predictions.jsonl').read_text(encoding='utf-8').splitlines() if line]
            checked = audit_metrics(data, predictions)
            saved = json.loads((folder / 'metrics.json').read_text(encoding='utf-8'))
            if metadata['dataset_sha256'] != digest:
                raise ValueError('Comparison data differs')
            for key, value in checked.items():
                if (value is None) != (saved[key] is None) or (value is not None and not math.isclose(value, saved[key], rel_tol=1e-8, abs_tol=1e-10)):
                    raise ValueError(f'Comparison metric mismatch: {key}')
            traces.append({'run': str(folder.relative_to(ROOT)),
                           'predictions_sha256': hashlib.sha256((folder / 'predictions.jsonl').read_bytes()).hexdigest(),
                           'metadata_sha256': hashlib.sha256((folder / 'metadata.json').read_bytes()).hexdigest()})
        scores, observed_identity, groups, observed_hash = read_runs(folders)
        if observed_identity != identity or observed_hash != digest:
            raise ValueError('Comparison question support changed')
        return scores, groups, traces

    candidate_folders = [ROOT / f'results/encoder8792-s{seed}-test-raw' for seed in [42, 43, 44]]
    candidate, groups, candidate_traces = audit(candidate_folders)
    report = {'dataset_sha256': digest, 'items': len(data), 'candidate': 'joint encoder, 8792 BANKING training, seeds42/43/44',
              'candidate_accuracy_by_seed': candidate.mean(axis=1).tolist(), 'candidate_traces': candidate_traces,
              'comparisons': [], 'pending': [],
              'scope': 'Exploratory analysis of an already scored oracle-eight test; question-group paired bootstrap conditional on three local seeds and one release per baseline. Intervals are individual 95% intervals without multiplicity correction; not architecture causality, training-exposure equivalence, or a noninferiority test.'}
    for name in ['laya', 'nano', 'openjev', 'von', 'decider08', 'kev', 'decider2b']:
        folder = ROOT / f'results/community-{name}-test-raw'
        if not (folder / 'metrics.json').exists():
            report['pending'].append(name)
            continue
        reference, reference_groups, traces = audit([folder])
        if reference_groups != groups:
            raise ValueError('Comparison groups changed')
        delta, interval, count = paired_interval(candidate, reference, groups, resamples=5000, seed=0)
        report['comparisons'].append({'reference': name, 'reference_released_checkpoints': 1,
            'reference_accuracy': float(reference.mean()), 'encoder_mean_minus_reference': delta,
            'individual_paired_ci95': interval, 'groups': count, 'resamples': 5000, 'reference_traces': traces})
    (ROOT / 'results/encoder8792-community-paired.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    lines = ['# Encoder versus released community models on matched BANKING questions', '', report['scope'], '',
             '| Release | Encoder minus release (pp) | Individual paired 95% interval (pp) |', '|---|---:|---:|']
    for row in report['comparisons']:
        lo, hi = row['individual_paired_ci95']
        lines.append(f"| {row['reference']} | {100*row['encoder_mean_minus_reference']:+.3f} | [{100*lo:+.3f}, {100*hi:+.3f}] |")
    lines += ['', 'Pending releases: ' + ', '.join(report['pending'])]
    (ROOT / 'results/encoder8792-community-paired.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
