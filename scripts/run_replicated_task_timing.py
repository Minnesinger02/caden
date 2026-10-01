"""Randomized repeated official real-task latency blocks, after training exits."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_replicated_fixed_compute import telemetry


def main():
    local = str(ROOT / '.venv-win/Scripts/python.exe')
    community = str(ROOT / '.venv-kev/Scripts/python.exe')
    encoder = ['-m', 'decision_lab', 'evaluate', '--backend', 'encoder', '--model', 'distilbert/distilbert-base-uncased', '--checkpoint', 'checkpoints/encoder16-s42-e3-8792', '--device', 'cuda', '--dtype', 'float32']
    mixed_encoder = ['-m', 'decision_lab', 'evaluate', '--backend', 'encoder', '--model', 'distilbert/distilbert-base-uncased', '--checkpoint', 'checkpoints/encoder16-s42-e3-mixed23789', '--device', 'cuda', '--dtype', 'float32']
    context_encoder = lambda checkpoint: ['-m', 'decision_lab', 'evaluate', '--backend', 'encoder', '--model', 'distilbert/distilbert-base-uncased', '--checkpoint', checkpoint, '--device', 'cuda', '--dtype', 'float32', '--attention-impl', 'eager']
    arms = {'encoder-eager': (local, [*encoder, '--attention-impl', 'eager']),
            'encoder-sdpa': (local, [*encoder, '--attention-impl', 'sdpa']),
            'qwen': (local, ['-m', 'prefill_renorm_sft', 'evaluate', '--adapter', 'checkpoints/sft16-s42-train8792', '--seed', '42', '--device', 'cuda', '--dtype', 'float32', '--max-length', '512']),
            'encoder-mixed-eager': (local, [*mixed_encoder, '--attention-impl', 'eager']),
            'qwen-mixed': (local, ['-m', 'prefill_renorm_sft', 'evaluate', '--adapter', 'checkpoints/sft16-s42-trainmixed23789', '--seed', '42', '--device', 'cuda', '--dtype', 'float32', '--max-length', '512']),
            'encoder-5k-joint': (local, context_encoder('checkpoints/encoder16-s42-e3-5k')),
            'encoder-5k-independent': (local, context_encoder('checkpoints/encoder16-s42-e3-independent-5k')),
            'laya': (community, ['scripts/evaluate_encoder_community.py', '--model', 'laya']),
            'nano': (community, ['scripts/evaluate_opendecider_local.py']),
            'openjev': (community, ['scripts/evaluate_openjev_local.py']),
            'von': (community, ['scripts/evaluate_encoder_community.py', '--model', 'von']),
            'decider08': (community, ['scripts/evaluate_decider_local.py', '--model', 'Mapika/decider-0.8b']),
            'decider2b': (community, ['scripts/evaluate_decider_local.py', '--model', 'Mapika/decider-2b']),
            'kev': (community, ['scripts/evaluate_kev_local.py'])}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--blocks', type=int, default=5)
    parser.add_argument('--seed', type=int, default=1729)
    parser.add_argument('--models', nargs='+', choices=list(arms), default=list(arms))
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    if args.blocks < 3 or len(args.models) != len(set(args.models)):
        parser.error('At least three blocks and distinct model names required')
    os.chdir(ROOT)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    data = 'data/pilot16/dev.jsonl'
    digest = hashlib.sha256(Path(data).read_bytes()).hexdigest()
    rng = random.Random(args.seed)
    plan = []
    for block in range(args.blocks):
        order = list(args.models)
        rng.shuffle(order)
        plan.append({'block': block, 'order': order})
    report = {'settings': vars(args), 'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'arm_extension': 'Before any GPU timing blocks: fixed mixed encoder/Qwen seed42 plus matched 5k joint/independent seed42; 14 default arms. Seed42 is fixed for timing, not selected by accuracy. Same question payloads for all arms. Earlier 10/12-arm plans remain CPU-only previews.',
              'commands': {name: [*arms[name][1], '--warmup', '3', '--data', data] for name in args.models},
              'data': data, 'dataset_sha256': digest, 'plan': plan, 'measurements': [],
              'scope': 'FP32 B1 same 200 questions and candidate payloads; default official model implementations differ; two CPU threads; warmup3; tokenization, transfer, validation and probability readback included; loading/download excluded; cold per-item state caches; no generation'}
    (out / 'plan.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    if args.plan_only:
        print(f'Plan saved; no GPU model loaded: {out}', flush=True)
        return
    environment = os.environ.copy()
    environment.update(HF_HOME=str(ROOT / '.cache/huggingface'), HF_HUB_OFFLINE='1', PYTHONUTF8='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    for block in plan:
        for name in block['order']:
            key = f"block{block['block']}-{name}"
            python, loader = arms[name]
            before = telemetry()
            with (out / f'{key}.log').open('x', encoding='utf-8') as stream:
                result = subprocess.run([python, '-u', *loader, '--warmup', '3', '--data', data, '--out', str(out / key)], env=environment, stdout=stream, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f'Failed {key}; preserve partial results and inspect log')
            report['measurements'].append({'block': block['block'], 'model': name, 'run': key, 'before': before, 'after': telemetry()})
            (out / 'progress.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
            print(f'Completed {key}', flush=True)
    (out / 'completed.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
