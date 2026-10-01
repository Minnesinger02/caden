"""Audit frozen transfer traces and derive per-task / policy-family results."""
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.paired_bootstrap import read_runs


def main():
    report = {'scope': 'BANKING-only local training; BANKING calibration temperatures applied without target refitting; community pretraining exposure is not certified', 'tasks': {}}
    lines = ['# Frozen transfer diagnostics', '', report['scope'], '',
             'CLINC uses an oracle eight-candidate shortlist. Policy uses all eight exhaustive flag routes; uniform chance is 12.5%. Policy is a synthetic diagnostic, not a representative reasoning benchmark.', '']
    for task, source in [('clinc', 'data/clinc-transfer8/test.jsonl'), ('policy', 'data/policy-transfer8/test.jsonl')]:
        data = ROOT / source
        rows = [json.loads(line) for line in data.read_text(encoding='utf-8').splitlines() if line]
        expected = sorted((r['id'], r['label'], r['group_id']) for r in rows)
        expected_hash = hashlib.sha256(data.read_bytes()).hexdigest()
        by_id = {r['id']: r for r in rows}
        result = {'items': len(rows), 'dataset_sha256': expected_hash, 'rows': []}
        lines += [f'## {task} ({len(rows)} questions)', '', '| Model | Variant | Accuracy ± seed SD | NLL | ECE | Failures | Direct / two-hop accuracy |', '|---|---|---:|---:|---:|---:|---:|']
        for model in ['encoder', 'qwen', 'laya', 'nano', 'openjev', 'von', 'decider08', 'kev']:
            names = [f'{model}-s{s}' for s in [42, 43, 44]] if model in ['encoder', 'qwen'] else [model]
            previous = None
            for variant in ['raw', 'bankcal']:
                folders = [ROOT / f'results/transfer-{task}-{name}-{variant}' for name in names]
                scores, identities, _, digest = read_runs(folders)
                if digest != expected_hash or identities != expected:
                    raise ValueError(f'Frozen identity mismatch: {task}/{model}/{variant}')
                metrics = [json.loads((p / 'metrics.json').read_text(encoding='utf-8')) for p in folders]
                acc = scores.mean(axis=1).tolist()
                if previous is not None and acc != previous:
                    raise ValueError('Temperature scaling changed choices')
                previous = acc
                if any(m['total'] != len(rows) or abs(m['accuracy_all_failures_wrong'] - a) > 1e-12 for m, a in zip(metrics, acc)):
                    raise ValueError('Saved aggregate differs from audited trace')
                family = {}
                if task == 'policy':
                    for name in ['direct', 'two_hop']:
                        indices = [i for i, identity in enumerate(identities) if by_id[identity[0]]['family'] == name]
                        family[name] = {'items': len(indices), 'accuracy_mean': float(scores[:, indices].mean())}
                entry = {'model': model, 'variant': variant, 'replicates': len(names), 'runs': [str(p.relative_to(ROOT)) for p in folders], 'accuracy_by_seed_or_release': acc,
                         'accuracy_mean': statistics.mean(acc), 'accuracy_seed_sd': statistics.stdev(acc) if len(acc) > 1 else None,
                         'nll_mean': statistics.mean(m['nll_clipped_1e-15'] for m in metrics), 'ece_mean': statistics.mean(m['ece_10_equal_width'] for m in metrics),
                         'failures': sum(m['failures'] for m in metrics), 'families': family}
                result['rows'].append(entry)
                sd = f" ± {100*entry['accuracy_seed_sd']:.2f}" if entry['accuracy_seed_sd'] is not None else ''
                parts = ' / '.join(f"{100*family[n]['accuracy_mean']:.2f}%" for n in ['direct', 'two_hop']) if family else '—'
                lines.append(f"| {model} | {variant} | {100*entry['accuracy_mean']:.2f}{sd}% | {entry['nll_mean']:.4f} | {entry['ece_mean']:.4f} | {entry['failures']} | {parts} |")
        report['tasks'][task] = result
        lines.append('')
    (ROOT / 'results/transfer-summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (ROOT / 'results/transfer-summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
