"""Exploratory original-domain retention using fixed mixed models and pooled T."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from decision_lab.common import read_data
from scripts.paired_bootstrap import read_runs, paired_interval
from scripts.summarize_fresh_clinc import audit_metrics

DATA = 'data/confirmation8-full/test.jsonl'
DIGEST = '444a292af999764ff91c9be649b56a51633691f2fea967cca93196c1d89437b3'
CAL_DIGEST = '59d21d289f39f623dee7402d88d970621e9c8b70afb3211dcd694fbcffbab230'


def build_plan():
    python = str(ROOT / '.venv-win/Scripts/python.exe')
    stages = []
    for family in ['encoder', 'qwen']:
        for seed in [42, 43, 44]:
            prefix = f'mixedbank-{family}-s{seed}'
            temperature = f'results/freshclinc-mixed-{family}-s{seed}-temperature.json'
            if family == 'encoder':
                checkpoint = f'checkpoints/encoder16-s{seed}-e3-mixed23789'
                loader = ['-m', 'decision_lab', 'evaluate', '--backend', 'encoder', '--model', 'distilbert/distilbert-base-uncased',
                          '--checkpoint', checkpoint, '--device', 'cuda', '--dtype', 'float32', '--save-logits']
            else:
                checkpoint = f'checkpoints/sft16-s{seed}-trainmixed23789'
                loader = ['-m', 'prefill_renorm_sft', 'evaluate', '--adapter', checkpoint, '--device', 'cuda',
                          '--dtype', 'float32', '--max-length', '512', '--seed', str(seed)]
            raw, calibrated = f'results/{prefix}-raw', f'results/{prefix}-calibrated'
            stages.append({'family': family, 'seed': seed, 'variant': 'raw', 'output': raw, 'checkpoint': checkpoint,
                           'temperature': temperature, 'log': f'handoff/{prefix}-raw.log',
                           'command': [python, '-u', *loader, '--data', DATA, '--out', raw]})
            stages.append({'family': family, 'seed': seed, 'variant': 'calibrated', 'output': calibrated, 'checkpoint': checkpoint,
                           'temperature': temperature, 'log': f'handoff/{prefix}-apply.log',
                           'command': [python, '-u', 'scripts/calibrate_probabilities.py', 'apply', '--run', raw,
                                       '--temperature', temperature, '--out', calibrated]})
    return stages


def summarize():
    data = read_data(ROOT / DATA)
    identity = sorted((row['id'], row['label'], row['group_id']) for row in data)
    report = {'dataset_sha256': DIGEST, 'items': len(data), 'rows': [],
              'scope': 'Exploratory retention on already scored BANKING test; mixed pooled3396 calibration versus original BANKING1000 calibration. Accuracy is unaffected by T; different calibration pools confound probability-quality comparisons. All observed seeds retained; no recipe or threshold selection from these scores.'}
    for family in ['encoder', 'qwen']:
        previous = None
        for variant in ['raw', 'calibrated']:
            folders = [ROOT / f'results/mixedbank-{family}-s{seed}-{variant}' for seed in [42, 43, 44]]
            before_folders = [ROOT / f'results/{family}8792-s{seed}-test-{variant}' for seed in [42, 43, 44]]
            candidate, observed, groups, digest = read_runs(folders)
            baseline, other, _, old_digest = read_runs(before_folders)
            if observed != identity or other != identity or digest != DIGEST or old_digest != DIGEST:
                raise ValueError('Retention test identities differ')
            checked = []
            for seed, folder in zip([42, 43, 44], folders):
                metadata = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
                training = metadata.get('training', metadata.get('adapter_training', {}))
                train_path = ROOT / 'data/mixed-banking-clinc8/train.jsonl'
                if training['seed'] != seed or training.get('data_sha256', training.get('dataset_sha256')) != hashlib.sha256(train_path.read_bytes()).hexdigest() or metadata['dtype'] != 'float32':
                    raise ValueError('Retention mixed model recipe differs')
                if variant == 'calibrated' and metadata['temperature_calibration']['calibration_dataset_sha256'] != CAL_DIGEST:
                    raise ValueError('Retention must reuse registered pooled mixed calibration')
                predictions = [json.loads(line) for line in (folder / 'predictions.jsonl').read_text(encoding='utf-8').splitlines() if line]
                metrics = audit_metrics(data, predictions)
                saved = json.loads((folder / 'metrics.json').read_text(encoding='utf-8'))
                for key, value in metrics.items():
                    if (value is None) != (saved[key] is None) or (value is not None and not math.isclose(value, saved[key], rel_tol=1e-8, abs_tol=1e-10)):
                        raise ValueError('Retention saved metrics mismatch')
                checked.append(metrics)
            if previous is not None and not (candidate == previous).all():
                raise ValueError('Retention T changed per-question accuracy outcomes')
            previous = candidate
            delta, interval, groups_count = paired_interval(candidate, baseline, groups)
            report['rows'].append({'family': family, 'variant': variant, 'seeds': [42, 43, 44],
                'runs': [str(folder.relative_to(ROOT)) for folder in folders], 'metrics_by_seed': checked,
                'baseline_accuracy_by_seed': baseline.mean(axis=1).tolist(), 'mixed_accuracy_by_seed': candidate.mean(axis=1).tolist(),
                'mixed_minus_baseline': delta, 'paired_ci95': interval, 'groups': groups_count, 'resamples': 5000,
                'interval_scope': 'paired source-question group bootstrap conditional on observed seeds, not a population-over-seeds interval'})
    with (ROOT / 'results/mixed-banking-test-retention.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    if hashlib.sha256(Path(DATA).read_bytes()).hexdigest() != DIGEST:
        raise ValueError('Frozen BANKING test changed')
    stages = build_plan()
    missing = sorted({stage['temperature'] for stage in stages if not Path(stage['temperature']).exists()})
    for path in [stage[key] for stage in stages for key in ['output', 'log']] + ['results/mixed-banking-test-retention.json']:
        if Path(path).exists():
            raise FileExistsError(f'Preserve existing {path}; resume explicitly')
    for stage in stages:
        if not Path(stage['checkpoint']).is_dir():
            raise FileNotFoundError(stage['checkpoint'])
    if args.plan_only:
        print(json.dumps({'stages': stages, 'missing_temperatures': missing, 'ready': not missing,
                          'launch_gate': 'Observe prior GPU queue terminal before execution; all six pooled T must exist'}, indent=2))
        return
    if missing:
        raise FileNotFoundError(f'Complete mixed calibration first: {missing}')
    environment = os.environ.copy()
    environment.update(HF_HOME=str(ROOT / '.cache/huggingface'), HF_HUB_OFFLINE='1', PYTHONUTF8='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    for stage in stages:
        print(f"Start {stage['family']} / seed{stage['seed']} / {stage['variant']}", flush=True)
        with Path(stage['log']).open('x', encoding='utf-8') as stream:
            result = subprocess.run(stage['command'], env=environment, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"Retention failed: {stage['log']}; preserve partial evidence")
    summarize()


if __name__ == '__main__':
    main()
