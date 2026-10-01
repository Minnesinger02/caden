"""Serial fixed development perturbations; launch only after training exits."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PROBES = {
    'order': ('data/order-dev8/dev.jsonl', '8d1e3b50dd520b28e8bb517e2f323e192cec92cb5771dd050f154ef575e4c7ab'),
    'support': ('data/support-dev8/dev.jsonl', '6b6dfa0551dc3a032529737a86acd60fbb09c85cfc6bcafe332b602e32d93e1b'),
}


def build_plan():
    local_python = str(ROOT / '.venv-win/Scripts/python.exe')
    community_python = str(ROOT / '.venv-kev/Scripts/python.exe')
    models = []
    for tag in ['e3-5k', 'e3-independent-5k', 'e3-8792']:
        for seed in [42, 43, 44]:
            checkpoint = f'checkpoints/encoder16-s{seed}-{tag}'
            models.append((f'encoder-{tag}-s{seed}', local_python, checkpoint,
                           ['-m', 'decision_lab', 'evaluate', '--backend', 'encoder',
                            '--model', 'distilbert/distilbert-base-uncased', '--checkpoint', checkpoint,
                            '--device', 'cuda', '--dtype', 'float32', '--attention-impl', 'eager', '--save-logits']))
    for seed in [42, 43, 44]:
        adapter = f'checkpoints/sft16-s{seed}-train8792'
        models.append((f'qwen-train8792-s{seed}', local_python, adapter,
                       ['-m', 'prefill_renorm_sft', 'evaluate', '--adapter', adapter,
                        '--device', 'cuda', '--dtype', 'float32', '--max-length', '512', '--seed', str(seed)]))
    for name, command in [
        ('laya', ['scripts/evaluate_encoder_community.py', '--model', 'laya']),
        ('nano', ['scripts/evaluate_opendecider_local.py']),
        ('openjev', ['scripts/evaluate_openjev_local.py']),
        ('von', ['scripts/evaluate_encoder_community.py', '--model', 'von']),
        ('decider08', ['scripts/evaluate_decider_local.py', '--model', 'Mapika/decider-0.8b']),
        ('decider2b', ['scripts/evaluate_decider_local.py', '--model', 'Mapika/decider-2b']),
        ('kev', ['scripts/evaluate_kev_local.py']),
    ]:
        models.append((name, community_python, None, command))
    plan = []
    for name, python, checkpoint, command in models:
        for probe, (data, digest) in PROBES.items():
            prefix = f'perturbation-{probe}-{name}'
            output = f'results/{prefix}-raw'
            summary = f'results/{prefix}-summary.json'
            plan.append({'model': name, 'probe': probe, 'checkpoint': checkpoint, 'dataset_sha256': digest,
                         'output': output, 'summary': summary, 'log': f'handoff/{prefix}.log',
                         'command': [python, '-u', *command, '--data', data, '--out', output],
                         'summary_command': [local_python, '-u', f'scripts/summarize_{probe}_probe.py',
                                             '--data', data, '--run', output, '--out', summary]})
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-only', action='store_true', help='CPU-only validation and plan; does not load models')
    parser.add_argument('--out', default='results/candidate-perturbations')
    args = parser.parse_args()
    os.chdir(ROOT)
    for data, expected in PROBES.values():
        if hashlib.sha256(Path(data).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Frozen perturbation input changed: {data}')
    plan = build_plan()
    for stage in plan:
        for required in [stage['command'][0], stage['checkpoint']]:
            if required is not None and not Path(required).exists():
                raise FileNotFoundError(required)
        for key in ['output', 'summary', 'log']:
            if Path(stage[key]).exists():
                raise FileExistsError(f'Existing {stage[key]}; preserve evidence and resume explicitly')
    report = {'stages': plan, 'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'scope': '200 development source questions; order/support variants are paired diagnostic rows; no threshold selection',
              'launch_gate': 'Caller must directly observe training session terminal before running; plan-only is CPU safe',
              'threads': 2, 'single_gpu_serial': True}
    if args.plan_only:
        print(json.dumps(report, indent=2))
        return
    folder = Path(args.out)
    folder.mkdir(parents=True, exist_ok=False)
    (folder / 'plan.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    environment = os.environ.copy()
    environment.update(HF_HOME=str(ROOT / '.cache/huggingface'), HF_HUB_OFFLINE='1',
                       PYTHONUTF8='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    for stage in plan:
        print(f"Start {stage['probe']} / {stage['model']}", flush=True)
        with Path(stage['log']).open('x', encoding='utf-8') as stream:
            for key in ['command', 'summary_command']:
                result = subprocess.run(stage[key], env=environment, stdout=stream, stderr=subprocess.STDOUT)
                if result.returncode:
                    raise RuntimeError(f"Failed {key}: {stage['log']}; preserve partial evidence")
        print(f"Completed {stage['summary']}", flush=True)
    print('All fixed perturbation runs complete; descriptive development diagnostics only.', flush=True)


if __name__ == '__main__':
    main()
