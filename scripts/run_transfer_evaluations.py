"""Frozen CLINC and fresh rule probes, using BANKING calibration without refitting."""
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
environment = os.environ.copy()
environment.update(HF_HOME=str(ROOT / '.cache/huggingface'), HF_HUB_OFFLINE='1',
                   PYTHONUTF8='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
local_python = str(ROOT / '.venv-win/Scripts/python.exe')
community_python = str(ROOT / '.venv-kev/Scripts/python.exe')


def run(command, log):
    with Path(log).open('x', encoding='utf-8') as stream:
        result = subprocess.run(command, env=environment, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'Failed with code {result.returncode}; inspect {log}')


def evaluate(name, python, loader, data, temperature, task):
    raw = f'results/transfer-{task}-{name}-raw'
    calibrated = f'results/transfer-{task}-{name}-bankcal'
    if Path(raw).exists() or Path(calibrated).exists():
        raise FileExistsError(raw)
    print(f'Start {task} {name}', flush=True)
    run([python, '-u', *loader, '--data', data, '--out', raw], f'handoff/transfer-{task}-{name}.log')
    run([local_python, 'scripts/calibrate_probabilities.py', 'apply', '--run', raw,
         '--temperature', temperature, '--out', calibrated], f'handoff/transfer-{task}-{name}-apply.log')
    print(f'Completed {task} {name}', flush=True)


for task, data in [('clinc', 'data/clinc-transfer8/test.jsonl'), ('policy', 'data/policy-transfer8/test.jsonl')]:
    for seed in [42, 43, 44]:
        evaluate(f'encoder-s{seed}', local_python,
                 ['-m', 'decision_lab', 'evaluate', '--backend', 'encoder', '--model', 'distilbert/distilbert-base-uncased',
                  '--checkpoint', f'checkpoints/encoder16-s{seed}-e3-8792', '--device', 'cuda', '--save-logits'],
                 data, f'results/encoder8792-s{seed}-temperature.json', task)
        evaluate(f'qwen-s{seed}', local_python,
                 ['-m', 'prefill_renorm_sft', 'evaluate', '--adapter', f'checkpoints/sft16-s{seed}-train8792',
                  '--device', 'cuda', '--max-length', '512', '--seed', str(seed)],
                 data, f'results/qwen8792-s{seed}-temperature.json', task)
    for name, loader in [('laya', ['scripts/evaluate_encoder_community.py', '--model', 'laya']),
                         ('nano', ['scripts/evaluate_opendecider_local.py']),
                         ('openjev', ['scripts/evaluate_openjev_local.py']),
                         ('von', ['scripts/evaluate_encoder_community.py', '--model', 'von']),
                         ('decider08', ['scripts/evaluate_decider_local.py', '--model', 'Mapika/decider-0.8b']),
                         ('kev', ['scripts/evaluate_kev_local.py'])]:
        evaluate(name, community_python, loader, data, f'results/community-{name}-temperature.json', task)
