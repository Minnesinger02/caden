"""Fit temperature only on calibration predictions, apply without changing winners."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from decision_lab.common import validate_probs, write_run

def row_logits(row):
    if 'logits' not in row:
        return None
    raw = row['logits']
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, list):
        order = row.get('candidate_order')
        if not isinstance(order, list) or len(order) != len(raw) or len(set(order)) != len(order) or set(order) != set(row['probabilities']):
            raise ValueError('List scores require exact declared candidate order')
        return dict(zip(order, raw))
    raise ValueError('Raw scores must be a map or list with candidate order')

def distribution(probabilities, temperature, raw_logits=None):
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('Positive finite temperature required')
    values = np.asarray(list(probabilities.values()), dtype=np.float64)
    if raw_logits is not None:
        if set(raw_logits) != set(probabilities):
            raise ValueError('Raw score support differs from probabilities')
        scores = np.asarray([raw_logits[k] for k in probabilities], dtype=np.float64)
        if not np.isfinite(scores).all():
            raise ValueError('Raw scores must be finite')
        logits = scores / temperature
    else:
        logits = np.log(np.maximum(values, 1e-15)) / temperature
    weights = np.exp(logits - logits.max())
    return dict(zip(probabilities, (weights / weights.sum()).tolist()))

def fit_temperature(rows):
    if not rows or any(r.get('split') != 'calibration' for r in rows):
        raise ValueError('Only nonempty calibration split can fit temperature')
    valid = [r for r in rows if 'probabilities' in r and 'error' not in r]
    if not valid:
        raise ValueError('No valid calibration probabilities')
    if any('logits' in r for r in valid) and not all('logits' in r for r in valid):
        raise ValueError('Calibration score source must be consistent across items')
    for row in valid:
        validate_probs(row['probabilities'], row['probabilities'])
        if row['label'] not in row['probabilities']:
            raise ValueError('Gold must belong to support')
    def loss(log_temperature):
        temperature = np.exp(log_temperature)
        return float(np.mean([-np.log(max(distribution(r['probabilities'], temperature, row_logits(r))[r['label']], 1e-300)) for r in valid]))
    grid = np.linspace(-4, 4, 257)
    values = [loss(t) for t in grid]
    best = int(np.argmin(values))
    lo, hi = grid[max(0, best - 1)], grid[min(len(grid) - 1, best + 1)]
    for _ in range(60):
        left, right = lo + (hi-lo)/3, hi - (hi-lo)/3
        if loss(left) < loss(right):
            hi = right
        else:
            lo = left
    log_temperature = float((lo+hi)/2)
    return {'temperature': float(np.exp(log_temperature)), 'valid_items': len(valid), 'failures_excluded': len(rows)-len(valid),
            'score_source': 'raw_logits' if all('logits' in r for r in valid) else 'clipped_log_probabilities' if all('logits' not in r for r in valid) else 'mixed',
            'calibration_nll_before': loss(0), 'calibration_nll_after': loss(log_temperature),
            'at_search_boundary': best in {0, len(grid)-1}, 'search_log_temperature_bounds': [-4, 4]}

def identity(meta):
    training = meta.get('training') or meta.get('adapter_training') or {}
    return {key: meta.get(key) for key in ['backend','model','revision','resolved_revision','checkpoint','adapter','requested','dtype','attention_impl']} | {'training': training}

def read(folder):
    folder = Path(folder)
    return json.loads((folder/'metadata.json').read_text(encoding='utf-8')), [json.loads(s) for s in (folder/'predictions.jsonl').read_text(encoding='utf-8').split('\n') if s.strip()]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    fit = sub.add_parser('fit')
    fit.add_argument('--run', required=True)
    fit.add_argument('--out', required=True)
    apply = sub.add_parser('apply')
    apply.add_argument('--run', required=True)
    apply.add_argument('--temperature', required=True)
    apply.add_argument('--out', required=True)
    args = parser.parse_args()
    meta, rows = read(args.run)
    if args.command == 'fit':
        result = fit_temperature(rows)
        result.update(model_identity=identity(meta), calibration_dataset_sha256=meta['dataset_sha256'], calibration_run=args.run)
        with Path(args.out).open('x', encoding='utf-8') as stream:
            json.dump(result, stream, indent=2)
        print(json.dumps(result, indent=2))
        return
    fitted = json.loads(Path(args.temperature).read_text(encoding='utf-8'))
    if identity(meta) != fitted['model_identity']:
        raise ValueError('Calibration and target must use identical model/checkpoint')
    valid = [r for r in rows if 'probabilities' in r and 'error' not in r]
    raw_source = fitted.get('score_source') == 'raw_logits'
    if any(('logits' in r) != raw_source for r in valid):
        raise ValueError('Calibration and target must use identical score source')
    data = Path(meta['data'])
    if hashlib.sha256(data.read_bytes()).hexdigest() != meta['dataset_sha256']:
        raise ValueError('Data changed after evaluation')
    for row in rows:
        if 'probabilities' in row and 'error' not in row:
            before = row['probabilities']
            row['probabilities'] = distribution(before, fitted['temperature'], row_logits(row))
            assert max(before, key=before.get) == max(row['probabilities'], key=row['probabilities'].get)
    meta.update(temperature_calibration=fitted, raw_run=args.run,
                postprocessing_latency='saved latency belongs to raw model; CPU temperature postprocessing excluded')
    write_run(args.out, data, rows, meta)

if __name__ == '__main__':
    main()
