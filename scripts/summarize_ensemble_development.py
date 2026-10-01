"""Descriptive development ensemble gains and explicitly estimated serial cost."""
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def main():
    rows = []
    for name in ['encoder-joint5k', 'encoder-independent5k', 'encoder-joint8792', 'qwen8792']:
        path = ROOT / f'results/ensemble-{name}-dev'
        meta = json.loads((path / 'metadata.json').read_text(encoding='utf-8'))
        scores = json.loads((path / 'metrics.json').read_text(encoding='utf-8'))
        members = [json.loads((ROOT / member / 'metrics.json').read_text(encoding='utf-8')) for member in meta['members']]
        if len(members) != 3 or any(m['total'] != scores['total'] for m in members):
            raise ValueError('Expected aligned three-member ensemble')
        rows.append({'arm': name, 'items': scores['total'], 'members': meta['members'],
                     'member_mean_accuracy': statistics.mean(m['accuracy_all_failures_wrong'] for m in members),
                     'best_member_accuracy': max(m['accuracy_all_failures_wrong'] for m in members),
                     'member_mean_nll': statistics.mean(m['nll_clipped_1e-15'] for m in members),
                     'ensemble_accuracy': scores['accuracy_all_failures_wrong'], 'ensemble_nll': scores['nll_clipped_1e-15'],
                     'estimated_sequential_p50_ms': scores['latency_p50_ms'], 'failures': scores['failures'],
                     'scope': meta['scope'], 'cost_scope': meta['latency_scope']})
    report = {'rows': rows, 'interpretation': 'Equal-weight seed ensemble, development only; improvement over seed mean is not necessarily improvement over best seed; three model forwards and hypothetical summed serial cost, not a measured single-forward speedup.'}
    (ROOT / 'results/ensemble-development-summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    lines = ['# Development seed ensembles', '', report['interpretation'], '', '| Arm | Mean member acc. | Best member acc. | Ensemble acc. | Mean member NLL | Ensemble NLL | Estimated serial p50 ms |', '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['arm']} | {100*r['member_mean_accuracy']:.2f}% | {100*r['best_member_accuracy']:.2f}% | {100*r['ensemble_accuracy']:.2f}% | {r['member_mean_nll']:.4f} | {r['ensemble_nll']:.4f} | {r['estimated_sequential_p50_ms']:.2f} |")
    (ROOT / 'results/ensemble-development-summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
