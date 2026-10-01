"""Derive a full-test table only from complete three-seed raw/calibrated arms."""
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.paired_bootstrap import read_runs

manifest = json.loads((ROOT / 'data/confirmation8-full/manifest.json').read_text(encoding='utf-8'))
expected_hash = manifest['files']['test']['sha256']
summary = {'test_sha256': expected_hash, 'test_items': manifest['files']['test']['items'],
           'calibration_items': manifest['files']['calibration']['items'],
           'task': manifest['task'], 'rows': [], 'community_rows': [], 'pending': []}
for name in ['encoder', 'qwen']:
    for kind in ['raw', 'calibrated']:
        folders = [ROOT / 'results' / f'{name}8792-s{seed}-test-{kind}' for seed in [42,43,44]]
        if not all((p / 'metrics.json').exists() for p in folders):
            summary['pending'].append(f'{name}-{kind}: incomplete three-seed arm')
            continue
        scores, identities, groups, data_hash = read_runs(folders)
        if data_hash != expected_hash or len(identities) != summary['test_items']:
            raise ValueError('Run differs from frozen full test')
        metrics = [json.loads((p / 'metrics.json').read_text(encoding='utf-8')) for p in folders]
        accuracies = scores.mean(axis=1).tolist()
        if any(m['total'] != summary['test_items'] for m in metrics):
            raise ValueError('Saved metric denominator differs')
        summary['rows'].append({'model': name, 'variant': kind, 'seeds': [42,43,44],
                                'runs': [str(p.relative_to(ROOT)) for p in folders],
                                'accuracy_by_seed': accuracies, 'accuracy_mean': statistics.mean(accuracies),
                                'accuracy_sample_sd': statistics.stdev(accuracies),
                                'nll_mean': statistics.mean(m['nll_clipped_1e-15'] for m in metrics),
                                'brier_mean': statistics.mean(m['brier_sum'] for m in metrics),
                                'ece_mean': statistics.mean(m['ece_10_equal_width'] for m in metrics),
                                'failures_total': sum(m['failures'] for m in metrics)})
    arms = [r for r in summary['rows'] if r['model'] == name]
    if len(arms) == 2 and arms[0]['accuracy_by_seed'] != arms[1]['accuracy_by_seed']:
        raise ValueError('Temperature scaling changed test accuracy')
for name in ['laya', 'nano', 'openjev', 'von', 'decider08', 'kev']:
    for kind in ['raw', 'calibrated']:
        folder = ROOT / 'results' / f'community-{name}-test-{kind}'
        if not (folder / 'metrics.json').exists():
            summary['pending'].append(f'community-{name}-{kind}')
            continue
        scores, identities, groups, data_hash = read_runs([folder])
        if data_hash != expected_hash or len(identities) != summary['test_items']:
            raise ValueError('Community run differs from frozen full test')
        m = json.loads((folder / 'metrics.json').read_text(encoding='utf-8'))
        summary['community_rows'].append({'model': name, 'variant': kind, 'released_checkpoints': 1,
                                          'run': str(folder.relative_to(ROOT)), 'accuracy': float(scores.mean()),
                                          'nll': m['nll_clipped_1e-15'], 'ece': m['ece_10_equal_width'],
                                          'failures': m['failures'], 'latency_p50_ms': m['latency_p50_ms']})
(ROOT / 'results/confirmation-summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
lines = ['| Model | Variant | Seeds | Accuracy mean ± SD | NLL | ECE | Failures |',
         '|---|---|---:|---:|---:|---:|---:|']
for row in summary['rows']:
    lines.append(f"| {row['model']} | {row['variant']} | 3 | {100*row['accuracy_mean']:.3f} ± {100*row['accuracy_sample_sd']:.3f}% | {row['nll_mean']:.4f} | {row['ece_mean']:.4f} | {row['failures_total']} |")
if summary['community_rows']:
    lines += ['', 'One fixed released checkpoint per community arm; public-data provenance and different tuning/runtime apply.', '',
              '| Community model | Variant | Accuracy | NLL | ECE | Failures |', '|---|---|---:|---:|---:|---:|']
    for row in summary['community_rows']:
        lines.append(f"| {row['model']} | {row['variant']} | {100*row['accuracy']:.3f}% | {row['nll']:.4f} | {row['ece']:.4f} | {row['failures']} |")
(ROOT / 'results/confirmation-summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print('\n'.join(lines))
print('Pending: ' + '; '.join(summary['pending']))
