"""Re-audit every registered order/support trace before aggregating complete arms."""
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from decision_lab.common import read_data
from scripts.summarize_order_probe import summarize as order_summary
from scripts.summarize_support_probe import summarize as support_summary


def main():
    plan_path = ROOT / 'results/candidate-perturbations/plan.json'
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    data = {probe: read_data(ROOT / f'data/{probe}-dev8/dev.jsonl') for probe in ['order', 'support']}
    audited = {}
    traces = []
    for stage in plan['stages']:
        saved_path = ROOT / stage['summary']
        if not saved_path.is_file():
            raise FileNotFoundError(f'All registered stages required; pending {saved_path}')
        folder = ROOT / stage['output']
        metadata = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
        probe = stage['probe']
        digest = hashlib.sha256((ROOT / f'data/{probe}-dev8/dev.jsonl').read_bytes()).hexdigest()
        if metadata['dataset_sha256'] != digest or digest != stage['dataset_sha256']:
            raise ValueError('Frozen perturbation input changed')
        predictions = [json.loads(line) for line in (folder / 'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
        checked = (order_summary if probe == 'order' else support_summary)(data[probe], predictions)
        saved = json.loads(saved_path.read_text(encoding='utf-8'))
        if saved != {**checked, 'run': stage['output'], 'dataset_sha256': digest}:
            raise ValueError('Saved perturbation summary differs from full re-audit')
        if (probe, stage['model']) in audited:
            raise ValueError('Duplicate registered stage')
        audited[(probe, stage['model'])] = checked
        traces.append({'model': stage['model'], 'probe': probe,
            'summary_sha256': hashlib.sha256(saved_path.read_bytes()).hexdigest(),
            'predictions_sha256': hashlib.sha256((folder / 'predictions.jsonl').read_bytes()).hexdigest(),
            'metadata_sha256': hashlib.sha256((folder / 'metadata.json').read_bytes()).hexdigest()})
    groups = {name: [f'{name}-s{s}' for s in [42, 43, 44]] for name in
        ['encoder-e3-5k', 'encoder-e3-independent-5k', 'encoder-e3-8792', 'qwen-train8792']}
    groups.update({name: [name] for name in ['laya', 'nano', 'openjev', 'von', 'decider08', 'decider2b', 'kev']})
    if len(audited) != 38 or set(audited) != {(probe, model) for probe in ['order', 'support'] for models in groups.values() for model in models}:
        raise ValueError('Registered 19-model support changed')
    report = {'plan_sha256': hashlib.sha256(plan_path.read_bytes()).hexdigest(), 'rows': [], 'traces': traces,
        'source_questions': 200,
        'scope': 'Exploratory development diagnostics, 200 source questions; order has original plus four permutations, support has original/drop/add/gold-absent-with-explicit-none. Three observed seeds for local arms, one released checkpoint for community arms. SD is descriptive, not confidence. Gold-absent is a separate task; no decision threshold selected. Mixed-domain checkpoints are not part of this frozen robustness matrix.'}
    for name, models in groups.items():
        orders = [audited[('order', model)] for model in models]
        supports = [audited[('support', model)] for model in models]
        row = {'arm': name, 'models': models, 'observed_seeds': [42, 43, 44] if len(models) == 3 else [],
            'order_accuracy_mean_by_condition': {str(i): statistics.mean(r['accuracy_by_condition'][str(i)] for r in orders) for i in range(5)},
            'order_argmax_flips_by_seed_or_release': [r['argmax_flips'] for r in orders],
            'order_valid_comparisons_by_seed_or_release': [r['valid_comparisons'] for r in orders],
            'order_flip_rate_mean': statistics.mean(r['argmax_flips']/r['valid_comparisons'] for r in orders) if all(r['valid_comparisons'] for r in orders) else None,
            'order_questions_with_flip_by_seed_or_release': [r['question_count_with_flip'] for r in orders],
            'order_max_probability_deviation_by_seed_or_release': [r['max_probability_deviation'] for r in orders],
            'order_failures_total': sum(r['failures'] for r in orders), 'support': {}}
        for condition in ['original', 'drop_nongold', 'add_nongold', 'gold_absent']:
            values = [r['conditions'][condition] for r in supports]
            accuracy = [r['accuracy_all_failures_wrong'] for r in values]
            entry = {'accuracy_by_seed_or_release': accuracy, 'accuracy_mean': statistics.mean(accuracy),
                'accuracy_seed_sd': statistics.stdev(accuracy) if len(accuracy) == 3 else None,
                'failures_total': sum(r['failures'] for r in values)}
            if condition != 'original':
                entry['surviving_raw_logit_drift_by_seed_or_release'] = [r.get('surviving_raw_logit_drift') for r in values]
            row['support'][condition] = entry
        report['rows'].append(row)
    (ROOT / 'results/candidate-perturbation-summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    lines = ['# Audited candidate perturbations', '', report['scope'], '',
        '| Arm | Mean order flip rate | Original accuracy | Drop nongold | Add nongold | Gold absent + none | Failures |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in report['rows']:
        values = [row['support'][c] for c in ['original', 'drop_nongold', 'add_nongold', 'gold_absent']]
        flip = f"{100*row['order_flip_rate_mean']:.2f}%" if row['order_flip_rate_mean'] is not None else 'undefined'
        cells = ' | '.join(f"{100*r['accuracy_mean']:.2f}%" for r in values)
        failures = row['order_failures_total'] + sum(r['failures_total'] for r in values)
        lines.append(f"| {row['arm']} | {flip} | {cells} | {failures} |")
    (ROOT / 'results/candidate-perturbation-summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
