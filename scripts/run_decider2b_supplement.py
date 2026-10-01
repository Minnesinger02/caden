"""Pinned larger community baseline on existing, already examined diagnostic pools."""
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
from scripts.summarize_fresh_clinc import audit_metrics

DATA = {
    'calibration': ('data/confirmation8-full/calibration.jsonl', 'aa20052659ac2553cf1f6978a6fc2291cab81452b2ed46223d37f91d3e84c86e'),
    'test': ('data/confirmation8-full/test.jsonl', '444a292af999764ff91c9be649b56a51633691f2fea967cca93196c1d89437b3'),
    'clinc': ('data/clinc-transfer8/test.jsonl', '8db9007592a14e5056b2e1127d51d3d62dd075e834c42fa37c14ec8247e7a589'),
    'policy': ('data/policy-transfer8/test.jsonl', 'edab1aae77ce23ad7695eb34c8c171d0ca0699c345a3e2f1f72c6d55f365c71f'),
}
TEMPERATURE = 'results/community-decider2b-temperature.json'


def build_plan():
    gpu_python = str(ROOT / '.venv-kev/Scripts/python.exe')
    cpu_python = str(ROOT / '.venv-win/Scripts/python.exe')
    plan = []
    for task, (data, _) in DATA.items():
        prefix = f'community-decider2b-{task}' if task in ['calibration', 'test'] else f'transfer-{task}-decider2b'
        raw = f'results/{prefix}-raw'
        plan.append({'command': [gpu_python, '-u', 'scripts/evaluate_decider_local.py', '--model', 'Mapika/decider-2b',
                                 '--dtype', 'float32', '--threads', '2', '--data', data, '--out', raw],
                     'output': raw, 'log': f'handoff/{prefix}.log', 'task': task, 'variant': 'raw'})
        if task == 'calibration':
            command = [cpu_python, '-u', 'scripts/calibrate_probabilities.py', 'fit', '--run', raw, '--out', TEMPERATURE]
            output = TEMPERATURE
        else:
            output = f'results/{prefix}-' + ('calibrated' if task == 'test' else 'bankcal')
            command = [cpu_python, '-u', 'scripts/calibrate_probabilities.py', 'apply', '--run', raw,
                       '--temperature', TEMPERATURE, '--out', output]
        plan.append({'command': command, 'output': output, 'log': f'handoff/{prefix}-postprocess.log', 'task': task,
                     'variant': 'temperature' if task == 'calibration' else 'calibrated'})
    return plan


def summarize(plan):
    rows = []
    choices = {}
    for stage in plan:
        if stage['variant'] == 'temperature':
            continue
        folder = ROOT / stage['output']
        metadata = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
        data_path, expected_hash = DATA[stage['task']]
        if metadata['dataset_sha256'] != expected_hash or metadata['model'] != 'Mapika/decider-2b' or metadata['revision'] != '533964dae8be954c5b5e19fa4948e48408094c1e' or metadata['source_commit'] != '50d0be0d7cb43d2066965ce5fa7f3fe4e489a60f':
            raise ValueError('Supplement model or data identity differs')
        if metadata['dtype'] != 'float32' or metadata['torch_num_threads'] != 2:
            raise ValueError('Supplement precision or actual thread count differs')
        predictions = [json.loads(line) for line in (folder / 'predictions.jsonl').read_text(encoding='utf-8').splitlines() if line]
        checked = audit_metrics(read_data(ROOT / data_path), predictions)
        saved = json.loads((folder / 'metrics.json').read_text(encoding='utf-8'))
        for key, value in checked.items():
            if (value is None) != (saved[key] is None) or (value is not None and not math.isclose(value, saved[key], rel_tol=1e-8, abs_tol=1e-10)):
                raise ValueError(f'Supplement saved metric mismatch: {key}')
        winners = {row['id']: max(row['probabilities'], key=row['probabilities'].get) if 'error' not in row else None for row in predictions}
        if stage['variant'] == 'raw':
            choices[stage['task']] = winners
        else:
            if choices[stage['task']] != winners:
                raise ValueError('Supplement temperature changed choices')
            calibration = metadata['temperature_calibration']
            if calibration['calibration_dataset_sha256'] != DATA['calibration'][1]:
                raise ValueError('Supplement temperature fitted on incorrect pool')
        rows.append({'task': stage['task'], 'variant': stage['variant'], 'run': stage['output'], **checked})
    report = {'model': 'Mapika/decider-2b', 'revision': '533964dae8be954c5b5e19fa4948e48408094c1e', 'rows': rows,
              'scope': 'One released checkpoint; supplementary larger baseline on already examined BANKING/old CLINC/policy pools, not a new untouched confirmation; no policy/CLINC temperature refit.'}
    with (ROOT / 'results/decider2b-supplement-summary.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    for data, expected in DATA.values():
        if hashlib.sha256(Path(data).read_bytes()).hexdigest() != expected:
            raise ValueError('Frozen supplementary input changed')
    plan = build_plan()
    for python in {stage['command'][0] for stage in plan}:
        if not Path(python).exists():
            raise FileNotFoundError(python)
    checkpoint = Path('.cache/community/Mapika--decider-2b/533964dae8be954c5b5e19fa4948e48408094c1e')
    if not checkpoint.is_dir() or not (checkpoint / 'config.json').exists() or not (checkpoint / 'model.safetensors').exists():
        raise FileNotFoundError('Complete pinned Decider2B checkpoint required')
    for path in [stage[key] for stage in plan for key in ['output', 'log']] + ['results/decider2b-supplement-summary.json']:
        if Path(path).exists():
            raise FileExistsError(f'Preserve existing {path}; resume explicitly')
    if args.plan_only:
        print(json.dumps({'stages': plan, 'launch_gate': 'Observe current training session terminal before GPU execution; this option is CPU-only'}, indent=2))
        return
    environment = os.environ.copy()
    environment.update(HF_HOME=str(ROOT / '.cache/huggingface'), HF_HUB_OFFLINE='1', PYTHONUTF8='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    for stage in plan:
        print(f"Start {stage['task']} / {stage['variant']}", flush=True)
        with Path(stage['log']).open('x', encoding='utf-8') as stream:
            result = subprocess.run(stage['command'], env=environment, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"Supplement failed: {stage['log']}; preserve partial evidence")
    summarize(plan)


if __name__ == '__main__':
    main()
