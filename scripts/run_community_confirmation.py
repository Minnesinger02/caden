"""Serial official community calibration/test evaluation; no paid Jev requests."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
environment = os.environ.copy()
environment.update(HF_HOME=str(ROOT / '.cache/huggingface'), HF_HUB_OFFLINE='1',
                   PYTHONUTF8='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
python = str(ROOT / '.venv-kev/Scripts/python.exe')
post_python = str(ROOT / '.venv-win/Scripts/python.exe')
models = [('laya', ['scripts/evaluate_encoder_community.py', '--model', 'laya']),
          ('nano', ['scripts/evaluate_opendecider_local.py']),
          ('openjev', ['scripts/evaluate_openjev_local.py']),
          ('von', ['scripts/evaluate_encoder_community.py', '--model', 'von']),
          ('decider08', ['scripts/evaluate_decider_local.py', '--model', 'Mapika/decider-0.8b']),
          ('kev', ['scripts/evaluate_kev_local.py'])]


def run(command, log):
    with Path(log).open('x', encoding='utf-8') as stream:
        result = subprocess.run(command, env=environment, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'Failed with code {result.returncode}; inspect {log}; no automatic overwrite/retry')


for name, loader in models:
    for split in ['calibration', 'test']:
        output = f'results/community-{name}-{split}-raw'
        if Path(output).exists():
            raise FileExistsError(output)
        print(f'Start {name} {split}', flush=True)
        run([python, '-u', *loader, '--data', f'data/confirmation8-full/{split}.jsonl', '--out', output],
            f'handoff/community-{name}-{split}.log')
        print(f'Completed {name} {split}', flush=True)
        if split == 'calibration':
            run([post_python, 'scripts/calibrate_probabilities.py', 'fit', '--run', output,
                 '--out', f'results/community-{name}-temperature.json'], f'handoff/community-{name}-temperature.log')
        else:
            run([post_python, 'scripts/calibrate_probabilities.py', 'apply', '--run', output,
                 '--temperature', f'results/community-{name}-temperature.json',
                 '--out', f'results/community-{name}-test-calibrated'], f'handoff/community-{name}-apply.log')
    record = json.loads(Path(f'results/community-{name}-test-raw/metrics.json').read_text(encoding='utf-8'))
    print(json.dumps({'model': name, 'test_accuracy': record.get('accuracy_all_failures_wrong'),
                      'items': record['total'], 'failures': record['failures']}), flush=True)
