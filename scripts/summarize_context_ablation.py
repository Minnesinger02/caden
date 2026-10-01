"""Matched-data context ablation from all three completed development seeds."""
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from transformers import AutoTokenizer
from decision_lab.common import read_data
from scripts.encoder_workload import question_workload
from scripts.paired_bootstrap import read_runs, paired_interval


def main():
    dev = ROOT / 'data/pilot16-5k/dev.jsonl'
    data = read_data(dev)
    dev_hash = hashlib.sha256(dev.read_bytes()).hexdigest()
    train_hash = hashlib.sha256((ROOT / 'data/pilot16-5k/train.jsonl').read_bytes()).hexdigest()
    expected = sorted((r['id'], r['label'], r['group_id']) for r in data)
    report = {'dataset_sha256': dev_hash, 'train_sha256': train_hash, 'seeds': [42, 43, 44], 'rows': [],
              'scope': 'exploratory 200-question development ablation; historical serial latency, not randomized replicated timing; attention cells are a shape proxy, not measured FLOPs'}
    arrays = {}
    recipe_reference = None
    for mode, tag in [('joint', 'e3-5k'), ('independent', 'e3-independent-5k')]:
        folders = [ROOT / f'results/encoder16-s{s}-{tag}-dev' for s in report['seeds']]
        scores, identities, groups, digest = read_runs(folders)
        if digest != dev_hash or identities != expected:
            raise ValueError('Different development identity')
        arrays[mode] = scores
        metadata = [json.loads((p / 'metadata.json').read_text(encoding='utf-8')) for p in folders]
        settings = [m['training'] for m in metadata]
        if any(s['data_sha256'] != train_hash or s['epochs'] != 3 or s.get('input_mode', 'joint') != mode or s['readout'] != 'scalar' for s in settings):
            raise ValueError('Training recipe mismatch')
        for seed, setting in zip(report['seeds'], settings):
            if setting['seed'] != seed:
                raise ValueError('Training seed differs from declared arm')
            recipe = {key: setting.get(key) for key in ['model', 'revision', 'resolved_revision', 'max_length', 'lr', 'head_lr', 'batch_size', 'freeze_backbone']}
            if recipe_reference is not None and recipe != recipe_reference:
                raise ValueError('Context arms differ in another training setting')
            recipe_reference = recipe
        tokenizer = AutoTokenizer.from_pretrained(ROOT / settings[0]['out'] / 'backbone', local_files_only=True)
        workloads = [question_workload(r, tokenizer, mode) for r in data]
        metrics = [json.loads((p / 'metrics.json').read_text(encoding='utf-8')) for p in folders]
        rate = []
        for folder in folders:
            predictions = [json.loads(line) for line in (folder / 'predictions.jsonl').read_text(encoding='utf-8').splitlines() if line]
            rate.append(len(predictions) / (sum(r['latency_ms'] for r in predictions) / 1000))
        acc = scores.mean(axis=1).tolist()
        row = {'mode': mode, 'runs': [str(p.relative_to(ROOT)) for p in folders], 'accuracy_by_seed': acc,
               'accuracy_mean': statistics.mean(acc), 'accuracy_seed_sd': statistics.stdev(acc),
               'nll_mean': statistics.mean(m['nll_clipped_1e-15'] for m in metrics),
               'failures_total': sum(m['failures'] for m in metrics),
               'latency_p50_mean_ms': statistics.mean(m['latency_p50_ms'] for m in metrics),
               'saved_serial_decisions_per_second_by_seed': rate,
               'training_seconds_by_seed': [s['training_seconds'] for s in settings],
               'training_peak_allocated_bytes_by_seed': [s['cuda_peak_allocated_bytes'] for s in settings],
               'workload_means': {key: statistics.mean(w[key] for w in workloads) for key in workloads[0]}}
        report['rows'].append(row)
    delta, interval, count = paired_interval(arrays['independent'], arrays['joint'], groups)
    report['paired_accuracy'] = {'independent_minus_joint': delta, 'ci95': interval, 'groups': count,
                                'resamples': 5000, 'seed': 0, 'interpretation': 'conditional on observed seeds; development exploratory, not test confirmation'}
    (ROOT / 'results/context-ablation-summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    lines = ['# Candidate context ablation', '', report['scope'], '', '| Mode | Accuracy ± seed SD | NLL | p50 mean (ms) | Encoder rows/question | Valid / padded tokens/question |', '|---|---:|---:|---:|---:|---:|']
    for r in report['rows']:
        w = r['workload_means']
        lines.append(f"| {r['mode']} | {100*r['accuracy_mean']:.2f} ± {100*r['accuracy_seed_sd']:.2f}% | {r['nll_mean']:.4f} | {r['latency_p50_mean_ms']:.3f} | {w['encoder_rows']:.0f} | {w['valid_tokens']:.1f} / {w['padded_tokens']:.1f} |")
    lines += ['', f'Independent minus joint: {100*delta:.2f} pp, exploratory paired 95% interval [{100*interval[0]:.2f}, {100*interval[1]:.2f}] pp. Candidate-order GPU checks remain pending.']
    (ROOT / 'results/context-ablation-summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
