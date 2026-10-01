"""Paired-block speed ratios; intervals describe timing variability only."""
import argparse
import json
import math
from pathlib import Path
import statistics

import numpy as np


def ratio_summary(encoder, qwen, resamples=5000, seed=0):
    if len(encoder) != len(qwen) or len(encoder) < 3 or any(not math.isfinite(v) or v <= 0 for v in [*encoder, *qwen]):
        raise ValueError('At least three aligned positive timing blocks required')
    logs = np.log(np.asarray(encoder) / np.asarray(qwen))
    draws = np.random.default_rng(seed).integers(0, len(logs), size=(resamples, len(logs)))
    interval = np.exp(np.quantile(logs[draws].mean(axis=1), [.025, .975])).tolist()
    return {'encoder_decisions_per_second': list(encoder), 'qwen_decisions_per_second': list(qwen),
            'paired_ratios': np.exp(logs).tolist(), 'geometric_mean_speed_ratio': float(np.exp(logs.mean())),
            'ratio_ci95': interval, 'encoder_rate_median': statistics.median(encoder),
            'qwen_rate_median': statistics.median(qwen), 'blocks': len(logs)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    folder = Path(args.run)
    completed = json.loads((folder / 'completed.json').read_text(encoding='utf-8'))
    measurements = completed['measurements']
    expected = {(p['block'], backend) for p in completed['plan'] for backend in p['order']}
    if len(measurements) != len(expected) or {(m['block'], m['backend']) for m in measurements} != expected:
        raise ValueError('Incomplete or duplicate timing blocks')
    results = {}
    identities = {}
    for m in measurements:
        report = json.loads((folder / m['file']).read_text(encoding='utf-8'))
        if report['settings']['backend'] != m['backend'] or report['dtype'] != completed['settings']['dtype'] or report['attention'] != completed['settings']['attention_impl']:
            raise ValueError('Runtime settings differ from plan')
        if report['settings'].get('include_generation'):
            raise ValueError('This plan excludes generation')
        identity = report['model_metadata']
        if m['backend'] in identities and identities[m['backend']] != identity:
            raise ValueError('Model identity changed between blocks')
        identities[m['backend']] = identity
        shapes = {}
        for row in report['rows']:
            key = (row['batch'], row['input_tokens_per_item'])
            if key in shapes or row['task'] != '8-candidate probability readout' or row['iterations'] != completed['settings']['repeats']:
                raise ValueError('Unexpected task, iterations or duplicate shape')
            shapes[key] = row['decisions_per_second']
        results[(m['block'], m['backend'])] = shapes
    first = next(iter(results.values()))
    if any(set(r) != set(first) for r in results.values()):
        raise ValueError('Different shapes across timing blocks')
    output = {'settings': completed['settings'], 'model_identities': identities, 'rows': [],
              'scope': completed['scope'], 'interpretation': 'paired timing-block bootstrap conditional on this machine/runtime and observed blocks; not accuracy confidence or architecture/FLOP advantage; no generated-token comparison'}
    blocks = sorted(p['block'] for p in completed['plan'])
    for batch, length in sorted(first):
        rates = [[results[(block, backend)][(batch, length)] for block in blocks] for backend in ['encoder', 'qwen']]
        output['rows'].append({'batch': batch, 'input_tokens_per_item': length, **ratio_summary(*rates)})
    (folder / 'summary.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps(output['rows'], indent=2))


if __name__ == '__main__':
    main()
