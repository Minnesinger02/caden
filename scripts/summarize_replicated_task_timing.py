"""Audit complete real-task blocks and derive paired decision-speed ratios."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from decision_lab.common import read_data, validate_probs
from scripts.calibrate_probabilities import identity
from scripts.summarize_replicated_compute import ratio_summary


def audit_run(data, predictions):
    indexed = {r['id']: r for r in predictions}
    if len(indexed) != len(predictions) or set(indexed) != {r['id'] for r in data}:
        raise ValueError('Timing trace does not exactly cover fixed questions')
    latencies, correct = [], []
    for row in data:
        pred = indexed[row['id']]
        if (pred['label'], pred['group_id']) != (row['label'], row['group_id']):
            raise ValueError('Timing trace labels or groups changed')
        if 'error' in pred:
            raise ValueError('Failed predictions cannot be used as a successful decision-speed comparison')
        validate_probs(pred['probabilities'], row['criteria'])
        latency = pred['latency_ms']
        if not isinstance(latency, (int, float)) or not math.isfinite(latency) or latency <= 0:
            raise ValueError('Positive finite item latency required')
        latencies.append(latency)
        correct.append(max(pred['probabilities'], key=pred['probabilities'].get) == row['label'])
    return {'items': len(data), 'accuracy': statistics.mean(correct),
            'decisions_per_second': len(data) / (sum(latencies) / 1000),
            'p50_ms': statistics.median(latencies), 'total_timed_seconds': sum(latencies) / 1000}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--reference', default='encoder-eager')
    args = parser.parse_args()
    folder = Path(args.run)
    completed = json.loads((folder / 'completed.json').read_text(encoding='utf-8'))
    source = ROOT / completed['data']
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != completed['dataset_sha256']:
        raise ValueError('Timing input changed since plan')
    data = read_data(source)
    models = completed['settings']['models']
    blocks = sorted(item['block'] for item in completed['plan'])
    expected = {(block, name) for block in blocks for name in models}
    observations = completed['measurements']
    if len(observations) != len(expected) or {(m['block'], m['model']) for m in observations} != expected:
        raise ValueError('Incomplete or duplicate repeated task blocks')
    if args.reference not in models or len(blocks) < 3:
        raise ValueError('Reference and at least three paired blocks required')
    stats, identities, runtimes = {}, {}, {}
    for observation in observations:
        name = observation['model']
        path = folder / observation['run']
        meta = json.loads((path / 'metadata.json').read_text(encoding='utf-8'))
        if meta['dataset_sha256'] != digest or meta['dtype'] != 'float32':
            raise ValueError('Wrong data or precision for fixed task blocks')
        threads = meta.get('torch_num_threads', meta.get('threads'))
        if threads != 2:
            raise ValueError('Two recorded CPU threads required')
        model_identity = {**identity(meta), 'source_commit': meta.get('source_commit'),
                          'base_revision': meta.get('base_revision'), 'head_sha256': meta.get('head_sha256')}
        runtime = {k: meta.get(k) for k in ['latency_scope', 'torch', 'transformers', 'attention', 'attention_impl', 'fused', 'merge', 'cuda_graphs', 'dtype']}
        if name in identities and (identities[name] != model_identity or runtimes[name] != runtime):
            raise ValueError('Model or runtime changed between repeated blocks')
        identities[name], runtimes[name] = model_identity, runtime
        predictions = [json.loads(line) for line in (path / 'predictions.jsonl').read_text(encoding='utf-8').splitlines() if line]
        stats[(observation['block'], name)] = {**audit_run(data, predictions),
                     'peak_allocated_bytes': meta.get('cuda_peak_allocated_bytes')}
    report = {'dataset_sha256': digest, 'items': len(data), 'blocks': blocks, 'reference': args.reference,
              'model_identities': identities, 'runtimes': runtimes, 'rows': [], 'paired_ratios': [],
              'scope': completed['scope'],
              'interpretation': 'reference decision-rate divided by compared rate; paired-block geometric mean and conditional bootstrap interval; same GPU and semantic questions, different prompts/parameter counts/kernels; not equal FLOPs or optimized deployment ceiling; descriptive intervals without simultaneous multiple-comparison correction'}
    reference_rates = [stats[(b, args.reference)]['decisions_per_second'] for b in blocks]
    for name in models:
        values = [stats[(b, name)] for b in blocks]
        rates = [v['decisions_per_second'] for v in values]
        report['rows'].append({'model': name, 'decision_rates_by_block': rates,
                              'median_decisions_per_second': statistics.median(rates),
                              'p50_ms_by_block': [v['p50_ms'] for v in values],
                              'accuracy_by_block': [v['accuracy'] for v in values],
                              'peak_allocated_bytes_by_block': [v['peak_allocated_bytes'] for v in values]})
        if name != args.reference:
            ratio = ratio_summary(reference_rates, rates)
            ratio['reference'] = args.reference
            ratio['compared'] = name
            # The mathematical helper's legacy field names denote its two inputs;
            # replace them for comparisons against community systems.
            ratio['reference_rates'] = ratio.pop('encoder_decisions_per_second')
            ratio['compared_rates'] = ratio.pop('qwen_decisions_per_second')
            ratio['reference_rate_median'] = ratio.pop('encoder_rate_median')
            ratio['compared_rate_median'] = ratio.pop('qwen_rate_median')
            report['paired_ratios'].append(ratio)
    (folder / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report['paired_ratios'], indent=2))


if __name__ == '__main__':
    main()
