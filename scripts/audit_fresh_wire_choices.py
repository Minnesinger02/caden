"""Audit stored official choices against normalized probabilities, without GPU calls."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    data = ROOT / 'data/mixed-clinc-confirmation8/test.jsonl'
    expected = {r['id'] for r in map(json.loads, data.read_text(encoding='utf-8').splitlines())}
    rows = []
    for name in ['laya', 'nano', 'openjev', 'von', 'decider08', 'decider2b', 'kev']:
        source = ROOT / f'results/freshclinc-community-{name}-test-raw/predictions.jsonl'
        predictions = [json.loads(line) for line in source.read_text(encoding='utf-8').splitlines()]
        if len(predictions) != len(expected) or {r['id'] for r in predictions} != expected:
            raise ValueError('Incomplete or duplicate question coverage')
        checked, unavailable, differences = 0, 0, []
        for row in predictions:
            if row.get('error'):
                raise ValueError('Do not hide failed rows in a wire audit')
            choice = row.get('official_choice')
            if choice is None:
                unavailable += 1
                continue
            checked += 1
            observed = max(row['probabilities'], key=row['probabilities'].get)
            if observed != choice:
                differences.append({'id': row['id'], 'official_choice': choice, 'probability_argmax': observed})
        rows.append({'model': name, 'items': len(predictions), 'official_choice_checked': checked,
            'official_choice_unavailable': unavailable, 'disagreements': differences,
            'predictions_sha256': hashlib.sha256(source.read_bytes()).hexdigest()})
    report = {'dataset_sha256': hashlib.sha256(data.read_bytes()).hexdigest(), 'rows': rows,
        'scope': 'Recorded official-choice versus normalized-probability argmax; unavailable choices are explicitly unverified. No inference or calibration change.'}
    (ROOT / 'results/fresh-clinc-wire-choice-audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
