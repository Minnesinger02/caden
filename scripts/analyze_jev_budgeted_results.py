"""Reaudit complete pinned API traces and compare identical frozen source groups."""
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from decision_lab.common import metrics, read_data, validate_probs
from scripts.paired_bootstrap import read_runs, paired_interval


def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))


def main():
    base = ROOT / 'results/jev113-budgeted-v2'
    completion = read('results/jev113-budgeted-v2/completed.json')
    plan = read('results/jev113-budgeted-v2/plan.json')
    if completion['completed_datasets'] != [row['name'] for row in plan['datasets']]:
        raise ValueError('All API tasks must complete before final analysis')
    if completion['unknown_usage_requests'] or completion['estimated_spent_usd'] > .50:
        raise ValueError('Budget/usage anomaly; inspect before publishing')
    tasks = {}
    for task in plan['datasets']:
        name = task['name']
        folder = base / name
        source = read_data(ROOT / task['path'])
        meta = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
        if hashlib.sha256((ROOT / task['path']).read_bytes()).hexdigest() != task['sha256'] or meta['dataset_sha256'] != task['sha256']:
            raise ValueError('API frozen dataset hash mismatch')
        predictions = [json.loads(line) for line in (folder / 'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
        if len(predictions) != len(source) or len(source) != task['items']:
            raise ValueError('API task size mismatch')
        grid_counts, nonmax, sum_errors, zeros, official_correct, tie_differences = {}, 0, 0, 0, 0, 0
        tokens = 0
        for actual, row in zip(predictions, source):
            if (actual['id'], actual['label'], actual['group_id']) != (row['id'], row['label'], row.get('group_id', row['id'])):
                raise ValueError('Prediction/source identities differ')
            if actual.get('served_model') != 'jev-1.13.0' or 'error' in actual:
                raise ValueError('Unpinned/failed API response')
            validate_probs(actual['probabilities'], row['criteria'])
            raw = actual['response_probabilities_raw']
            digits = actual['observed_wire_decimal_grid']
            tolerance = len(raw) * .5 * 10 ** (-digits) + 1e-8 if digits else 1e-5
            validate_probs(raw, row['criteria'], tolerance)
            total = sum(raw.values())
            if any(not math.isclose(actual['probabilities'][key], raw[key] / total, abs_tol=1e-12) for key in raw):
                raise ValueError('Wire normalization changed')
            official_max = raw[actual['official_choice']] == max(raw.values())
            if official_max != actual['official_choice_is_probability_maximum']:
                raise ValueError('Wire-choice audit flag mismatch')
            nonmax += not official_max
            official_correct += actual['official_choice'] == row['label']
            tie_differences += official_max and actual['official_choice'] != max(raw, key=raw.get)
            sum_errors += not math.isclose(total, 1, abs_tol=1e-8)
            zeros += raw[row['label']] == 0
            grid_counts[str(digits)] = grid_counts.get(str(digits), 0) + 1
            tokens += actual['usage']['input_tokens']
        recomputed = metrics(predictions)
        saved = json.loads((folder / 'metrics.json').read_text(encoding='utf-8'))
        if recomputed != saved:
            raise ValueError('Saved API metrics do not match recomputation')
        tasks[name] = {'items': len(source), 'dataset_sha256': task['sha256'],
            'predictions_sha256': hashlib.sha256((folder / 'predictions.jsonl').read_bytes()).hexdigest(),
            'metadata_sha256': hashlib.sha256((folder / 'metadata.json').read_bytes()).hexdigest(),
            'metrics': {key: value for key, value in recomputed.items() if not key.startswith('risk_')},
            'observed_wire_decimal_grid_counts': grid_counts, 'wire_sum_adjusted_items': sum_errors,
            'official_choice_not_probability_maximum': nonmax, 'zero_gold_probability_items': zeros,
            'official_choice_accuracy': official_correct / len(source),
            'official_choice_argmax_tie_differences': tie_differences,
            'input_tokens': tokens, 'successful_response_estimated_usd': tokens * .042 / 1_000_000,
            'paired_comparisons': []}
    comparisons = {
        'banking': {**{family + '-bank': [f'results/{family}8792-s{seed}-test-raw' for seed in [42, 43, 44]] for family in ['encoder', 'qwen']},
                    **{family + '-mixed': [f'results/mixedbank-{family}-s{seed}-raw' for seed in [42, 43, 44]] for family in ['encoder', 'qwen']},
                    **{model: [f'results/community-{model}-test-raw'] for model in ['laya', 'nano', 'openjev', 'von', 'decider08', 'decider2b', 'kev']}},
        'fresh-clinc': {row['arm']: row['runs'] for row in read('results/fresh-clinc-summary.json')['rows'] if not row['arm'].endswith('calibrated')},
        'policy': {**{row['model']: row['runs'] for row in read('results/transfer-summary.json')['tasks']['policy']['rows'] if row['variant'] == 'raw'},
                   'decider2b': ['results/transfer-policy-decider2b-raw']},
        'banking-dev': {row['model']: [f'results/replicated-task-fp32-8792/block0-{row["model"]}'] for row in read('results/replicated-task-fp32-8792/summary-mixed-reference.json')['rows']}}
    for name, arms in comparisons.items():
        api, identities, groups, data_hash = read_runs([base / name])
        for model, folders in arms.items():
            candidate, other_ids, _, other_hash = read_runs([ROOT / folder for folder in folders])
            if identities != other_ids or data_hash != other_hash:
                raise ValueError('Paired comparison is not on identical frozen items')
            delta, ci, count = paired_interval(candidate, api, groups, resamples=5000, seed=20261001)
            tasks[name]['paired_comparisons'].append({'system': model, 'runs': folders,
                'source_prediction_sha256': {str(folder): hashlib.sha256((ROOT / folder / 'predictions.jsonl').read_bytes()).hexdigest() for folder in folders},
                'accuracy_by_seed_or_release': candidate.mean(axis=1).tolist(),
                'system_minus_jev_accuracy': delta, 'ci95': ci, 'groups': count})
    previous_cost = read('results/jev113-budgeted/stopped.json')['estimated_spent_usd']
    previous_rows = [json.loads(line) for line in (ROOT / 'results/jev113-budgeted/banking-responses.jsonl').read_text(encoding='utf-8').splitlines()]
    previous_unsaved_cost = previous_cost - sum(row['usage']['input_tokens'] for row in previous_rows) * .042 / 1_000_000
    successful_cost = sum(row['successful_response_estimated_usd'] for row in tasks.values())
    if not math.isclose(successful_cost + previous_unsaved_cost, completion['estimated_spent_usd'], abs_tol=1e-10):
        raise ValueError('Usage ledger differs from successful plus previous unsaved billed responses')
    if completion['attempted_paid_requests_including_probe'] != sum(task['items'] for task in tasks.values()) + 4:
        raise ValueError('Unexpected duplicate or missing paid request count')
    report = {'model': 'jev-1.13.0', 'tasks': tasks, 'budget': completion,
        'previous_unpersisted_response_estimated_usd': previous_unsaved_cost,
        'successful_response_cost_usd': sum(row['successful_response_estimated_usd'] for row in tasks.values()),
        'scope': 'Late fixed-version API system comparison on previously scored frozen tasks. Paired source-group bootstrap5000, conditional on observed local seeds/one API pass; individual intervals without multiplicity correction. Wire-rounded NLL with zero clipping does not recover internal Jev logits. API concurrency4 latency includes transport/server and is separate from single-GPU local FP32 timing. No upstream training-exposure certification, temperature refit or paid teacher collection.'}
    (ROOT / 'results/jev113-comparison-summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    lines = ['# 固定Jev API与本地系统对照', '', report['scope'], '',
             f'按文档费率估算全部已尝试请求费用${completion["estimated_spent_usd"]:.6f}，含首轮停机未保存回答；不是账户账单。', '',
             '| 任务 | 题数 | Jev准确率/% | wire NLL | ECE | API p50/ms（并发4） |', '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name, task in tasks.items():
        m = task['metrics']
        lines.append(f'| {name} | {task["items"]} | {100*m["accuracy_all_failures_wrong"]:.3f} | {m["nll_clipped_1e-15"]:.4f} | {m["ece_10_equal_width"]:.4f} | {m["latency_p50_ms"]:.2f} |')
    for name, task in tasks.items():
        lines += ['', '## ' + name, '', '| 系统 | 系统减Jev/百分点 | 条件95%区间 |', '| --- | ---: | --- |']
        for row in task['paired_comparisons']:
            lines.append(f'| {row["system"]} | {100*row["system_minus_jev_accuracy"]:.3f} | [{100*row["ci95"][0]:.3f}, {100*row["ci95"][1]:.3f}] |')
    (ROOT / 'results/jev113-comparison-summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'model': report['model'], 'estimated_spent_usd': completion['estimated_spent_usd'],
        'tasks': {name: {'items': row['items'], 'accuracy': row['metrics']['accuracy_all_failures_wrong']} for name, row in tasks.items()}}, indent=2))


if __name__ == '__main__':
    main()
