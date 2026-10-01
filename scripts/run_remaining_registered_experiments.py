"""Run remaining frozen diagnostics and timing sequentially on the single GPU."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]


def main():
    os.chdir(ROOT)
    python = str(ROOT / '.venv-win/Scripts/python.exe')
    stages = [
        ('candidate-perturbations', ['scripts/run_candidate_perturbations.py']),
        ('fixed-compute-blocks', ['scripts/run_replicated_fixed_compute.py', '--out', 'results/replicated-fp32-eager-8792']),
        ('fixed-compute-audit', ['scripts/summarize_replicated_compute.py', '--run', 'results/replicated-fp32-eager-8792']),
        ('task-timing-blocks', ['scripts/run_replicated_task_timing.py', '--out', 'results/replicated-task-fp32-8792']),
        ('task-timing-original-reference', ['scripts/summarize_replicated_task_timing.py', '--run', 'results/replicated-task-fp32-8792']),
        ('task-timing-mixed-reference', ['scripts/summarize_replicated_task_timing.py', '--run', 'results/replicated-task-fp32-8792', '--reference', 'encoder-mixed-eager']),
    ]
    for required in ['results/decider2b-supplement-summary.json', 'results/mixed-banking-test-retention.json']:
        if not Path(required).is_file():
            raise FileNotFoundError(required)
    for path in ['results/candidate-perturbations', 'results/replicated-fp32-eager-8792',
                 'results/replicated-task-fp32-8792', 'handoff/remaining-registered-experiments']:
        if Path(path).exists():
            raise FileExistsError(f'Preserve existing {path}; resume explicitly')
    folder = Path('handoff/remaining-registered-experiments')
    folder.mkdir()
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
        PYTHONUTF8='1', HF_HOME=str(ROOT / '.cache/huggingface'), HF_HUB_OFFLINE='1')
    report = {'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'launch_gate': 'Caller directly observed prior GPU queue terminal before this launch',
        'stages': [{'name': name, 'command': [python, '-u', *command]} for name, command in stages],
        'scope': 'Existing frozen model/input/seed plans; single GPU serial subprocesses; no retraining, recipe selection or paid calls',
        'completed': []}
    (folder / 'plan.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    for name, command in stages:
        start = datetime.now().astimezone().isoformat()
        print(f'{start} Start {name}', flush=True)
        with (folder / f'{name}.log').open('x', encoding='utf-8') as stream:
            result = subprocess.run([python, '-u', *command], env=environment, stdout=stream, stderr=subprocess.STDOUT)
        end = datetime.now().astimezone().isoformat()
        report['completed'].append({'name': name, 'start': start, 'end': end, 'exit_code': result.returncode})
        (folder / 'progress.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        with Path('handoff/PROGRESS.md').open('a', encoding='utf-8') as stream:
            stream.write(f'\n\n### {end} — registered remaining experiment: {name}\n\n'
                f'Exit code {result.returncode}. Started {start}; evidence in handoff/remaining-registered-experiments/{name}.log. '
                'One GPU subprocess at a time; final tables require the stage-specific audit, not a file-existence assumption.\n')
        if result.returncode:
            raise RuntimeError(f'Failed {name}; preserve partial evidence, no automatic restart')
        if name.startswith('task-timing-') and name.endswith('reference'):
            label = 'original' if name == 'task-timing-original-reference' else 'mixed'
            source = Path('results/replicated-task-fp32-8792/summary.json')
            target = source.with_name(f'summary-{label}-reference.json')
            if target.exists():
                raise FileExistsError(target)
            shutil.copyfile(source, target)
        print(f'{end} Completed {name}', flush=True)
    (folder / 'completed.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
