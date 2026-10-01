"""Fixed all-seed calibration and matched fresh CLINC comparison, after training exits."""
import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def confirmation_paths():
    """All prerequisites and fresh artifacts, checked before any model loads."""
    required, fresh = [], []
    for seed in [42, 43, 44]:
        for family in ['encoder', 'qwen']:
            required.append(Path(f'results/{family}8792-s{seed}-temperature.json'))
            for stage in ['bankingonly', 'mixed']:
                tag = ('e3-8792' if stage == 'bankingonly' else 'e3-mixed23789') if family == 'encoder' else ('train8792' if stage == 'bankingonly' else 'trainmixed23789')
                checkpoint_family = 'encoder' if family == 'encoder' else 'sft'
                required.append(Path(f'checkpoints/{checkpoint_family}16-s{seed}-{tag}'))
                prefix = f'freshclinc-{stage}-{family}-s{seed}'
                fresh.extend(Path(f'results/{prefix}-{suffix}') for suffix in ['test-raw', 'test-calibrated'])
                log_stages = ['test', 'apply']
                if stage == 'mixed':
                    fresh.extend([Path(f'results/{prefix}-calibration-raw'), Path(f'results/{prefix}-temperature.json')])
                    log_stages += ['calibration', 'fit']
                fresh.extend(Path(f'handoff/{prefix}-{step}.log') for step in log_stages)
    return required, fresh


def validate_artifact_paths(root):
    required, fresh = confirmation_paths()
    missing = [str(path) for path in required if not (root / path).exists()]
    existing = [str(path) for path in fresh if (root / path).exists()]
    if missing or existing:
        raise RuntimeError(f'Confirmation preflight failed before GPU launch: missing={missing}; existing={existing}. Preserve artifacts; resume explicitly.')


def main():
    os.chdir(ROOT)
    mixed_manifest = json.loads(Path('data/mixed-banking-clinc8/manifest.json').read_text(encoding='utf-8'))
    fresh_manifest = json.loads(Path('data/mixed-clinc-confirmation8/manifest.json').read_text(encoding='utf-8'))
    for path, expected in [('data/mixed-banking-clinc8/calibration.jsonl', mixed_manifest['files']['calibration']['sha256']),
                           ('data/mixed-banking-clinc8/dev.jsonl', mixed_manifest['files']['dev']['sha256']),
                           ('data/mixed-clinc-confirmation8/test.jsonl', fresh_manifest['sha256'])]:
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError('Frozen mixed-domain confirmation input changed')
    # Necessary readiness check, not a substitute for the caller observing the live
    # training session's terminal state before launching this GPU queue.
    for seed in [42, 43, 44]:
        for prefix, tag in [('encoder16', 'e3-mixed23789'), ('sft-eval', 'trainmixed23789')]:
            if not Path(f'results/{prefix}-s{seed}-{tag}-dev/metrics.json').exists():
                raise RuntimeError('All registered mixed training/development seeds must complete first')
    validate_artifact_paths(ROOT)
    environment = os.environ.copy()
    environment.update(HF_HOME=str(ROOT / '.cache/huggingface'), HF_HUB_OFFLINE='1', PYTHONUTF8='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')

    def run(arguments, name):
        log = Path(f'handoff/{name}.log')
        with log.open('x', encoding='utf-8') as stream:
            result = subprocess.run([sys.executable, '-u', *arguments], env=environment, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'Failed {name}, code {result.returncode}; inspect log, no overwrite/retry')
        print(f'Completed {name}', flush=True)

    for seed in [42, 43, 44]:
        for family in ['encoder', 'qwen']:
            for stage in ['bankingonly', 'mixed']:
                if family == 'encoder':
                    tag = 'e3-8792' if stage == 'bankingonly' else 'e3-mixed23789'
                    loader = ['-m', 'decision_lab', 'evaluate', '--backend', 'encoder', '--model', 'distilbert/distilbert-base-uncased',
                              '--checkpoint', f'checkpoints/encoder16-s{seed}-{tag}', '--device', 'cuda', '--save-logits']
                else:
                    tag = 'train8792' if stage == 'bankingonly' else 'trainmixed23789'
                    loader = ['-m', 'prefill_renorm_sft', 'evaluate', '--adapter', f'checkpoints/sft16-s{seed}-{tag}',
                              '--device', 'cuda', '--max-length', '512', '--seed', str(seed)]
                prefix = f'freshclinc-{stage}-{family}-s{seed}'
                if stage == 'mixed':
                    calibration = f'results/{prefix}-calibration-raw'
                    temperature = f'results/{prefix}-temperature.json'
                    run([*loader, '--data', 'data/mixed-banking-clinc8/calibration.jsonl', '--out', calibration], prefix + '-calibration')
                    run(['scripts/calibrate_probabilities.py', 'fit', '--run', calibration, '--out', temperature], prefix + '-fit')
                else:
                    temperature = f'results/{family}8792-s{seed}-temperature.json'
                raw = f'results/{prefix}-test-raw'
                run([*loader, '--data', 'data/mixed-clinc-confirmation8/test.jsonl', '--out', raw], prefix + '-test')
                run(['scripts/calibrate_probabilities.py', 'apply', '--run', raw, '--temperature', temperature,
                     '--out', f'results/{prefix}-test-calibrated'], prefix + '-apply')
    print('All fixed local matched CLINC arms complete; community arms remain a separate serial queue.', flush=True)


if __name__ == '__main__':
    main()
