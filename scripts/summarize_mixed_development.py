"""Audit mixed development traces and report equal-domain and pooled metrics."""
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from decision_lab.common import metrics, read_data, validate_probs


def audit_recipe(metadata, seed, family, train_digest):
    training = metadata.get('training', metadata.get('adapter_training', {}))
    if training.get('seed') != seed or training.get('max_length') != 512:
        raise ValueError('Mixed development training seed or length differs')
    if training.get('data_sha256', training.get('dataset_sha256')) != train_digest:
        raise ValueError('Mixed development training hash differs')
    if (ROOT / training['data']).resolve() != (ROOT / 'data/mixed-banking-clinc8/train.jsonl').resolve():
        raise ValueError('Mixed development training pool differs')
    # Earlier accuracy traces predate actual-thread telemetry. Preserve unknown;
    # these traces cannot establish controlled timing, but labels remain valid.
    if metadata.get('dtype') != 'float32' or metadata.get('torch_num_threads') not in [None, 2]:
        raise ValueError('Mixed development runtime differs')
    expected = ({'model': 'distilbert/distilbert-base-uncased', 'resolved_revision': '12040accade4e8a0f71eabdb258fecc2e7e948be',
                 'epochs': 3, 'batch_size': 1, 'lr': 2e-5, 'readout': 'scalar', 'input_mode': 'joint', 'freeze_backbone': False}
                if family == 'encoder' else
                {'model': 'Qwen/Qwen3-0.6B', 'resolved_revision': 'c1899de289a04d12100db370d81485cdf75e47ca',
                 'epochs': 1, 'lr': 1e-4, 'objective': 'candidate_ce', 'rank': 8, 'accumulation': 8, 'dtype': 'float32'})
    for key, value in expected.items():
        if training.get(key) != value:
            raise ValueError(f'Mixed development registered recipe differs: {key}')


def aggregate_complete_seeds(runs, family):
    selected = [row for row in runs if row['family'] == family]
    seeds = [row['seed'] for row in selected]
    if len(seeds) != len(set(seeds)) or not set(seeds) <= {42, 43, 44}:
        raise ValueError('Unexpected or duplicate mixed development seed')
    if set(seeds) != {42, 43, 44}:
        return None
    result = {'family': family, 'seeds': [42, 43, 44], 'scope': 'complete observed seeds; mean and sample SD, not confidence intervals', 'metrics': {}}
    values = {'macro_domain_accuracy': [row['macro_domain_accuracy'] for row in selected]}
    for domain in ['banking', 'clinc', 'pooled']:
        for key in ['accuracy_all_failures_wrong', 'nll_clipped_1e-15', 'brier_sum', 'ece_10_equal_width']:
            values[f'{domain}.{key}'] = [row[domain].get(key, 0. if key == 'accuracy_all_failures_wrong' else None) for row in selected]
    for key, observations in values.items():
        result['metrics'][key] = {'mean': statistics.mean(observations), 'sample_sd': statistics.stdev(observations)} if all(value is not None for value in observations) else None
    result['failures_total'] = sum(row['pooled']['failures'] for row in selected)
    return result


def summarize(data, predictions):
    by_id = {r['id']: r for r in predictions}
    if len(by_id) != len(predictions) or set(by_id) != {r['id'] for r in data}:
        raise ValueError('Predictions must cover each frozen development ID exactly once')
    domains = {'banking': [], 'clinc': []}
    for row in data:
        pred = by_id[row['id']]
        if row['domain'] not in domains or row['split'] != 'dev':
            raise ValueError('Expected declared development domains')
        if (pred['label'], pred['group_id']) != (row['label'], row['group_id']):
            raise ValueError('Prediction identity differs from frozen data')
        if 'error' in pred and 'probabilities' in pred:
            raise ValueError('Ambiguous failure response')
        if 'error' not in pred:
            validate_probs(pred['probabilities'], row['criteria'])
        domains[row['domain']].append(pred)
    output = {}
    for name, rows in [*domains.items(), ('pooled', predictions)]:
        if not rows:
            raise ValueError('Both development domains must be present')
        output[name] = {k: v for k, v in metrics(rows).items() if k != 'risk_coverage_diagnostic_not_threshold_selection'}
    output['macro_domain_accuracy'] = sum(output[name].get('accuracy_all_failures_wrong', 0.) for name in domains) / len(domains)
    return output


def main():
    data_path = ROOT / 'data/mixed-banking-clinc8/dev.jsonl'
    data = read_data(data_path)
    digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
    manifest = json.loads((data_path.parent / 'manifest.json').read_text(encoding='utf-8'))
    train_digest = hashlib.sha256((data_path.parent / 'train.jsonl').read_bytes()).hexdigest()
    if digest != manifest['files']['dev']['sha256'] or train_digest != manifest['files']['train']['sha256']:
        raise ValueError('Frozen mixed training/development files changed')
    report = {'development_sha256': digest, 'scope': 'development only; no test selection or confirmation claims; equal-domain macro is distinct from pooled accuracy', 'runs': [], 'pending': []}
    for seed in [42, 43, 44]:
        for family, prefix, tag in [('encoder', 'encoder16', 'e3-mixed23789'), ('qwen', 'sft-eval', 'trainmixed23789')]:
            folder = ROOT / f'results/{prefix}-s{seed}-{tag}-dev'
            if not (folder / 'metrics.json').exists():
                report['pending'].append(str(folder.relative_to(ROOT)))
                continue
            metadata = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
            if metadata['dataset_sha256'] != digest:
                raise ValueError('Run differs from frozen mixed development data')
            audit_recipe(metadata, seed, family, train_digest)
            predictions = [json.loads(line) for line in (folder / 'predictions.jsonl').read_text(encoding='utf-8').splitlines() if line]
            report['runs'].append({'family': family, 'seed': seed, 'run': str(folder.relative_to(ROOT)),
                                   'torch_num_threads': metadata.get('torch_num_threads'), **summarize(data, predictions)})
    report['complete_seed_aggregates'] = [result for family in ['encoder', 'qwen'] if (result := aggregate_complete_seeds(report['runs'], family)) is not None]
    (ROOT / 'results/mixed-development-summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    lines = ['# Supervised mixed-domain development', '', '| Model | Seed | BANKING acc. | CLINC acc. | Equal-domain acc. | Pooled acc. |', '|---|---:|---:|---:|---:|---:|']
    for r in report['runs']:
        values = [r[d].get('accuracy_all_failures_wrong', 0.) for d in ['banking', 'clinc']]
        values += [r['macro_domain_accuracy'], r['pooled'].get('accuracy_all_failures_wrong', 0.)]
        lines.append(f"| {r['family']} | {r['seed']} | " + ' | '.join(f'{100*v:.2f}%' for v in values) + ' |')
    lines += ['', f"Pending complete runs: {len(report['pending'])}. Fresh CLINC test remains separate."]
    if report['complete_seed_aggregates']:
        lines += ['', 'Complete three-seed development aggregates (mean ± sample SD, not CI):', '']
        for result in report['complete_seed_aggregates']:
            pooled = result['metrics']['pooled.accuracy_all_failures_wrong']
            macro = result['metrics']['macro_domain_accuracy']
            lines.append(f"{result['family']}: pooled {100*pooled['mean']:.3f} ± {100*pooled['sample_sd']:.3f}%; equal-domain {100*macro['mean']:.3f} ± {100*macro['sample_sd']:.3f}%.")
    (ROOT / 'results/mixed-development-summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
