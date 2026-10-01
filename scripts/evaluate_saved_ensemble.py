"""Development-only equal-weight probability ensemble from audited saved traces."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from decision_lab.common import read_data, validate_probs, write_run


def combine(row, members):
    for member in members:
        if (member['id'], member['label'], member['group_id']) != (row['id'], row['label'], row['group_id']):
            raise ValueError('Ensemble member identity mismatch')
        if 'error' in member and 'probabilities' in member:
            raise ValueError('Ambiguous failure in ensemble member')
        if 'error' not in member:
            validate_probs(member['probabilities'], row['criteria'])
    result = {key: row[key] for key in ['id', 'label', 'group_id', 'split']}
    if any('error' in member for member in members):
        return {**result, 'error': 'At least one ensemble member failed; no selective member dropping'}
    probabilities = {key: sum(member['probabilities'][key] for member in members) / len(members) for key in row['criteria']}
    validate_probs(probabilities, row['criteria'])
    return {**result, 'probabilities': probabilities,
            'latency_ms': sum(member['latency_ms'] for member in members),
            'latency_is_estimated_sequential_sum': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--data', default='data/pilot16/dev.jsonl')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    if len(args.runs) < 2 or len(set(args.runs)) != len(args.runs):
        parser.error('At least two distinct member runs required')
    data = read_data(args.data)
    if any(r['split'] != 'dev' for r in data):
        raise ValueError('This exploratory ensemble driver only permits development data')
    digest = hashlib.sha256(Path(args.data).read_bytes()).hexdigest()
    identities = sorted((r['id'], r['label'], r['group_id']) for r in data)
    members, metadata, recipes = [], [], []
    for run in args.runs:
        folder = Path(run)
        meta = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
        if meta['dataset_sha256'] != digest:
            raise ValueError('Ensemble member uses different frozen data')
        predictions = [json.loads(line) for line in (folder / 'predictions.jsonl').read_text(encoding='utf-8').splitlines() if line]
        indexed = {r['id']: r for r in predictions}
        if len(indexed) != len(predictions) or sorted((r['id'], r['label'], r['group_id']) for r in predictions) != identities:
            raise ValueError('Member trace coverage differs from frozen data')
        members.append(indexed)
        metadata.append(meta)
        recipe = meta.get('training', meta.get('adapter_training', {}))
        if not recipe:
            raise ValueError('Seed-ensemble experiment requires recorded training metadata')
        if not recipe.get('data_sha256'):
            training_path = Path(recipe['data'])
            recipe = {**recipe, 'data_sha256': hashlib.sha256(training_path.read_bytes()).hexdigest()}
        recipes.append({k: recipe.get(k) for k in ['model', 'revision', 'data_sha256', 'epochs', 'readout', 'input_mode', 'lr', 'objective', 'rank']})
    if any(recipe != recipes[0] for recipe in recipes):
        raise ValueError('This seed-ensemble experiment requires identical training recipes')
    predictions = [combine(row, [member[row['id']] for member in members]) for row in data]
    write_run(args.out, args.data, predictions, {'backend': 'saved_probability_ensemble', 'members': args.runs,
              'weights': [1 / len(members)] * len(members), 'recipe': recipes[0],
              'latency_scope': 'sum of separately observed member latencies; descriptive sequential estimate, not a measured ensemble service; excludes aggregation/loading/cache effects',
              'scope': 'exploratory development-only optimization; consumes all member forwards; not a single-forward model or blind confirmation',
              'parameters_sum_if_all_members_resident': sum(meta.get('parameters', 0) for meta in metadata)})
    print(Path(args.out) / 'metrics.json')


if __name__ == '__main__':
    main()
