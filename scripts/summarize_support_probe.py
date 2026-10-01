"""Audit support-change metrics and surviving raw-logit drift by source question."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from decision_lab.common import metrics, read_data, validate_probs
from scripts.calibrate_probabilities import row_logits
from scripts.prepare_support_probe import NONE_KEY

CONDITIONS = ['original', 'drop_nongold', 'add_nongold', 'gold_absent']


def summarize(data, predictions):
    indexed = {r['id']: r for r in predictions}
    if len(indexed) != len(predictions) or set(indexed) != {r['id'] for r in data}:
        raise ValueError('Support predictions must exactly cover frozen IDs')
    grouped = {}
    for row in data:
        conditions = grouped.setdefault(row['reference_id'], {})
        if row['condition'] in conditions:
            raise ValueError('Duplicate source condition')
        conditions[row['condition']] = row
    outputs = {condition: [] for condition in CONDITIONS}
    drifts = {condition: [] for condition in CONDITIONS if condition != 'original'}
    for variants in grouped.values():
        if set(variants) != set(CONDITIONS):
            raise ValueError('Incomplete support conditions')
        base = variants['original']
        original_scores = row_logits(indexed[base['id']])
        for condition, row in variants.items():
            if (row['state'], row['question'], row['group_id']) != (base['state'], base['question'], base['group_id']):
                raise ValueError('Question identity changed')
            old_keys, keys = set(base['criteria']), set(row['criteria'])
            shared = old_keys & keys
            if any(row['criteria'][k] != base['criteria'][k] for k in shared):
                raise ValueError('Surviving candidate description changed')
            if condition == 'drop_nongold' and not (keys < old_keys and len(old_keys - keys) == 1 and row['label'] == base['label'] and row['label'] in keys):
                raise ValueError('Invalid nongold removal')
            if condition == 'add_nongold' and not (old_keys < keys and len(keys - old_keys) == 1 and row['label'] == base['label']):
                raise ValueError('Invalid nongold addition')
            if condition == 'gold_absent' and not (keys == (old_keys - {base['label']}) | {NONE_KEY} and row['label'] == NONE_KEY):
                raise ValueError('Invalid gold-absence condition')
            prediction = indexed[row['id']]
            if (prediction['label'], prediction['group_id']) != (row['label'], row['group_id']):
                raise ValueError('Prediction label/group mismatch')
            if 'error' in prediction and 'probabilities' in prediction:
                raise ValueError('Ambiguous failure response')
            if 'error' not in prediction:
                validate_probs(prediction['probabilities'], row['criteria'])
            outputs[condition].append(prediction)
            if condition != 'original' and 'error' not in prediction and 'error' not in indexed[base['id']]:
                current_scores = row_logits(prediction)
                if original_scores is not None and current_scores is not None:
                    if set(original_scores) != old_keys or set(current_scores) != keys:
                        raise ValueError('Raw score candidate support changed')
                    deviations = [abs(current_scores[k] - original_scores[k]) for k in shared]
                    if any(not math.isfinite(value) for value in deviations):
                        raise ValueError('Nonfinite raw score drift')
                    drifts[condition].append(max(deviations))
    result = {'source_questions': len(grouped), 'prediction_rows': len(data), 'conditions': {},
              'scope': 'development diagnostic; group unit is source question; absence is separate task; changing support may change normalized probabilities even with invariant raw logits'}
    for condition, rows in outputs.items():
        result['conditions'][condition] = {key: value for key, value in metrics(rows).items() if key != 'risk_coverage_diagnostic_not_threshold_selection'}
        if condition in drifts:
            values = drifts[condition]
            result['conditions'][condition]['surviving_raw_logit_drift'] = {'valid_question_comparisons': len(values),
                  'max_absolute_deviation': max(values) if values else None,
                  'questions_above_1e_4': sum(v > 1e-4 for v in values),
                  'interpretation': '1e-4 is a reported FP32 diagnostic tolerance, not an enforced generic model property; joint/causal models need not be invariant'}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--data', default='data/support-dev8/dev.jsonl')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    path = Path(args.data)
    folder = Path(args.run)
    metadata = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != metadata['dataset_sha256']:
        raise ValueError('Run differs from frozen support probe')
    predictions = [json.loads(line) for line in (folder / 'predictions.jsonl').read_text(encoding='utf-8').splitlines() if line]
    report = summarize(read_data(path), predictions)
    report.update(run=args.run, dataset_sha256=digest)
    with Path(args.out).open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
