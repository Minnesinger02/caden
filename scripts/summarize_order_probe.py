"""Compare predictions across candidate orders, treating source questions as units."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from decision_lab.common import read_data, validate_probs


def summarize(data, predictions):
    indexed = {row['id']: row for row in predictions}
    if len(indexed) != len(predictions) or set(indexed) != {r['id'] for r in data}:
        raise ValueError('Probe prediction IDs must exactly cover data without duplicates')
    questions = {}
    for row in data:
        variants = questions.setdefault(row['reference_id'], {})
        if row['condition'] in variants:
            raise ValueError('Duplicate condition for source question')
        variants[row['condition']] = row
    flips, deviations, per_question, conditions = [], [], [], {i: [] for i in range(5)}
    failures = 0
    for reference, variants in questions.items():
        if set(variants) != set(range(5)):
            raise ValueError('Each source question needs all five conditions')
        base = indexed[variants[0]['id']]
        base_probs = base.get('probabilities')
        for condition, row in variants.items():
            if (row['state'], row['question'], row['label'], row['group_id'], dict(row['criteria'])) != (
                    variants[0]['state'], variants[0]['question'], variants[0]['label'], variants[0]['group_id'], dict(variants[0]['criteria'])):
                raise ValueError('Order probe changed more than ordering')
            pred = indexed[row['id']]
            probabilities = pred.get('probabilities')
            valid = 'error' not in pred and probabilities is not None
            if valid:
                validate_probs(probabilities, row['criteria'])
                if pred.get('label') != row['label']:
                    raise ValueError('Saved gold label differs')
            else:
                failures += 1
            conditions[condition].append(int(valid and max(probabilities, key=probabilities.get) == row['label']))
        source_flips = []
        if base_probs is not None and 'error' not in base:
            base_winner = max(base_probs, key=base_probs.get)
            for condition in range(1, 5):
                current = indexed[variants[condition]['id']]
                if 'error' in current or 'probabilities' not in current:
                    continue
                p = current['probabilities']
                flipped = max(p, key=p.get) != base_winner
                source_flips.append(flipped)
                flips.append(flipped)
                deviations.append(max(abs(p[key] - base_probs[key]) for key in base_probs))
        per_question.append({'reference_id': reference, 'compared_permutations': len(source_flips),
                             'argmax_flips': sum(source_flips)})
    return {'source_questions': len(questions), 'prediction_rows': len(data), 'failures': failures,
            'accuracy_by_condition': {str(i): statistics.mean(v) for i, v in conditions.items()},
            'valid_comparisons': len(flips), 'argmax_flips': sum(flips),
            'question_count_with_flip': sum(r['argmax_flips'] > 0 for r in per_question),
            'max_probability_deviation': max(deviations, default=None),
            'mean_probability_max_deviation': statistics.mean(deviations) if deviations else None,
            'per_question': per_question,
            'interpretation': 'development diagnostic; original plus four orders per source question; failure comparisons are excluded from drift and retained in accuracy denominator; not five times as many independent questions'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--data', default='data/order-dev8/dev.jsonl')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    folder = Path(args.run)
    metadata = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
    digest = hashlib.sha256(Path(args.data).read_bytes()).hexdigest()
    if metadata['dataset_sha256'] != digest:
        raise ValueError('Run does not use this frozen order probe')
    predictions = [json.loads(s) for s in (folder / 'predictions.jsonl').read_text(encoding='utf-8').splitlines() if s]
    report = summarize(read_data(args.data), predictions)
    report.update(run=args.run, dataset_sha256=digest)
    with Path(args.out).open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != 'per_question'}, indent=2))


if __name__ == '__main__':
    main()
