"""Serial pinned-release comparison on the identical fresh CLINC questions."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    os.chdir(ROOT)
    data = Path('data/mixed-clinc-confirmation8/test.jsonl')
    manifest = json.loads((data.parent / 'manifest.json').read_text(encoding='utf-8'))
    if hashlib.sha256(data.read_bytes()).hexdigest() != manifest['sha256']:
        raise ValueError('Fresh confirmation input changed')
    models = [('laya', ['scripts/evaluate_encoder_community.py', '--model', 'laya']),
              ('nano', ['scripts/evaluate_opendecider_local.py']),
              ('openjev', ['scripts/evaluate_openjev_local.py']),
              ('von', ['scripts/evaluate_encoder_community.py', '--model', 'von']),
              ('decider08', ['scripts/evaluate_decider_local.py', '--model', 'Mapika/decider-0.8b']),
              ('decider2b', ['scripts/evaluate_decider_local.py', '--model', 'Mapika/decider-2b']),
              ('kev', ['scripts/evaluate_kev_local.py'])]
    for name, _ in models:
        if Path(f'results/freshclinc-community-{name}-test-raw').exists() or Path(f'handoff/freshclinc-community-{name}.log').exists():
            raise FileExistsError(f'Existing fresh community arm {name}; do not overwrite')
    environment = os.environ.copy()
    environment.update(HF_HOME=str(ROOT / '.cache/huggingface'), HF_HUB_OFFLINE='1', PYTHONUTF8='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    python = str(ROOT / '.venv-kev/Scripts/python.exe')
    for name, loader in models:
        output = f'results/freshclinc-community-{name}-test-raw'
        print(f'Start {name} on {manifest["items"]} fresh CLINC questions', flush=True)
        with Path(f'handoff/freshclinc-community-{name}.log').open('x', encoding='utf-8') as stream:
            result = subprocess.run([python, '-u', *loader, '--data', str(data), '--out', output], env=environment, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'Failed {name}; inspect log, preserve partial output')
        metrics = json.loads((Path(output) / 'metrics.json').read_text(encoding='utf-8'))
        print(json.dumps({'model': name, 'items': metrics['total'], 'accuracy': metrics.get('accuracy_all_failures_wrong'), 'failures': metrics['failures']}), flush=True)
    print('All seven pinned fresh CLINC community arms complete; no paid Jev calls.', flush=True)


if __name__ == '__main__':
    main()
