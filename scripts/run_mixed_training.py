"""Execute registered supervised expansion serially, with persistent stage telemetry."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    os.chdir(ROOT)
    manifest = json.loads((ROOT / 'data/mixed-banking-clinc8/manifest.json').read_text(encoding='utf-8'))
    preflight = json.loads((ROOT / 'handoff/mixed-data-preflight.json').read_text(encoding='utf-8'))
    for split, spec in manifest['files'].items():
        digest = hashlib.sha256((ROOT / f'data/mixed-banking-clinc8/{split}.jsonl').read_bytes()).hexdigest()
        if digest != spec['sha256'] or digest != preflight['splits'][split]['sha256']:
            raise ValueError('Mixed split differs from registered length preflight')
    environment = os.environ.copy()
    environment.update(HF_HOME=str(ROOT / '.cache/huggingface'), HF_HUB_OFFLINE='1', PYTHONUTF8='1',
                       OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    # Freshness must be verified for the whole plan before launching any stage.
    plan = []
    for seed in [42, 43, 44]:
        for family, tag, epochs in [('encoder', 'e3-mixed23789', 3), ('sft', 'trainmixed23789', 1)]:
            if (ROOT / f'checkpoints/{family}16-s{seed}-{tag}').exists() or (ROOT / f'results/{"encoder16" if family == "encoder" else "sft-eval"}-s{seed}-{tag}-dev').exists():
                raise FileExistsError(f'Existing mixed arm {family}/{seed}; resume explicitly instead of overwriting')
            options = ['--seed', str(seed), '--tag', tag, '--data-dir', 'data/mixed-banking-clinc8']
            plan.extend([(f'{family}-train', [*options, '--epochs', str(epochs)]), (f'{family}-eval', options)])
    for stage, options in plan:
        print(f'Start {stage}: {" ".join(options)}', flush=True)
        result = subprocess.run([sys.executable, '-u', 'scripts/local_stage.py', stage, *options], env=environment)
        if result.returncode:
            raise RuntimeError(f'Stage {stage} failed with code {result.returncode}; inspect handoff/runs before continuing')
    print('All registered mixed training and development arms complete; fresh CLINC test remains unscored.', flush=True)


if __name__ == '__main__':
    main()
