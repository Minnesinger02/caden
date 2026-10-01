"""Secondary paired fresh CLINC comparisons for both fixed mixed local families."""
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
    source = ROOT / 'results/fresh-clinc-summary.json'
    summary = json.loads(source.read_text(encoding='utf-8'))
    path = ROOT / 'data/mixed-clinc-confirmation8/test.jsonl'
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    data = read_data(path)
    if digest != summary['dataset_sha256'] or len(data) != 4197 or summary['pending']:
        raise ValueError('Require complete audited frozen confirmation')
    identity = sorted((r['id'], r['label'], r['group_id']) for r in data)

    def audit(folders):
        traces = []
        for folder in folders:
            metadata = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
            predictions = [json.loads(line) for line in (folder / 'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
            metrics = audit_metrics(data, predictions)
            saved = json.loads((folder / 'metrics.json').read_text(encoding='utf-8'))
            if metadata['dataset_sha256'] != digest:
                raise ValueError('Frozen data differs')
            for key, value in metrics.items():
                if (value is None) != (saved[key] is None) or (value is not None and not math.isclose(value, saved[key], rel_tol=1e-8, abs_tol=1e-10)):
                    raise ValueError('Metric re-audit mismatch')
            traces.append({'run': str(folder.relative_to(ROOT)),
                'predictions_sha256': hashlib.sha256((folder / 'predictions.jsonl').read_bytes()).hexdigest(),
                'metadata_sha256': hashlib.sha256((folder / 'metadata.json').read_bytes()).hexdigest()})
        scores, observed, groups, observed_hash = read_runs(folders)
        if observed != identity or observed_hash != digest:
            raise ValueError('Question identity differs')
        return scores, groups, traces

    report = {'dataset_sha256': digest, 'items': len(data), 'audited_summary_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'seeds': [42, 43, 44], 'comparisons': [],
        'scope': 'Secondary matched accuracy comparisons after the registered fresh CLINC scores. All seven releases included; individual 95% paired source-question intervals conditional on three mixed local seeds and one checkpoint per release, no simultaneous multiplicity correction. Community training exposure unknown. Not architecture causality, noninferiority or universal ranking.'}
    releases = {name: audit([ROOT / f'results/freshclinc-community-{name}-test-raw'])
        for name in ['laya', 'nano', 'openjev', 'von', 'decider08', 'decider2b', 'kev']}
    for family in ['encoder', 'qwen']:
        candidate, groups, candidate_traces = audit([ROOT / f'results/freshclinc-mixed-{family}-s{seed}-test-raw' for seed in [42, 43, 44]])
        for name, (reference, reference_groups, reference_traces) in releases.items():
            if groups != reference_groups:
                raise ValueError('Source-question grouping differs')
            delta, interval, count = paired_interval(candidate, reference, groups, resamples=5000, seed=0)
            report['comparisons'].append({'candidate': 'mixed-' + family, 'reference': name,
                'candidate_accuracy_by_seed': candidate.mean(axis=1).tolist(), 'reference_accuracy': float(reference.mean()),
                'candidate_mean_minus_reference': delta, 'individual_paired_ci95': interval, 'groups': count,
                'resamples': 5000, 'candidate_traces': candidate_traces, 'reference_traces': reference_traces})
    (ROOT / 'results/fresh-clinc-community-paired.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    lines = ['# Secondary matched fresh CLINC comparisons', '', report['scope'], '',
        '| Mixed local family | Release | Local minus release (pp) | Individual paired 95% interval (pp) |',
        '|---|---|---:|---:|']
    for row in report['comparisons']:
        lo, hi = row['individual_paired_ci95']
        lines.append(f"| {row['candidate']} | {row['reference']} | {100*row['candidate_mean_minus_reference']:+.3f} | [{100*lo:+.3f}, {100*hi:+.3f}] |")
    (ROOT / 'results/fresh-clinc-community-paired.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
