"""Randomize sequential model timing blocks; run only after the training queue exits."""
import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def telemetry():
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw,clocks.sm,clocks.mem', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=15)
    if result.returncode:
        raise RuntimeError('GPU telemetry unavailable')
    return {'unix_time': time.time(), 'fields': ['memory_mib', 'utilization_percent', 'temperature_c', 'power_w', 'sm_clock_mhz', 'memory_clock_mhz'], 'values': result.stdout.strip().splitlines()[0]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--blocks', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1729)
    parser.add_argument('--dtype', choices=['float32', 'bfloat16'], default='float32')
    parser.add_argument('--attention-impl', choices=['eager', 'sdpa'], default='eager')
    parser.add_argument('--checkpoint', default='checkpoints/encoder16-s42-e3-8792')
    args = parser.parse_args()
    if args.blocks < 3 or args.repeats < 20:
        parser.error('At least three blocks and twenty timed iterations per shape required')
    os.chdir(ROOT)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    environment.update(HF_HOME=str(ROOT / '.cache/huggingface'), HF_HUB_OFFLINE='1', PYTHONUTF8='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    rng = random.Random(args.seed)
    plan = []
    for block in range(args.blocks):
        order = ['encoder', 'qwen']
        rng.shuffle(order)
        plan.append({'block': block, 'order': order})
    report = {'settings': vars(args), 'plan': plan, 'measurements': [],
              'scope': 'one model process at a time; GPU-resident synthetic B x L input and eight-way softmax; FP32/eager primary; warmup five per shape; loading and tokenization excluded; different parameter counts and architecture, not equal FLOPs; probability-only with no generation'}
    (out / 'plan.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    for item in plan:
        for backend in item['order']:
            name = f"block{item['block']}-{backend}"
            before = telemetry()
            command = [sys.executable, '-u', 'scripts/benchmark_fixed_compute.py', '--backend', backend,
                       '--checkpoint', args.checkpoint, '--repeats', str(args.repeats), '--dtype', args.dtype,
                       '--attention-impl', args.attention_impl, '--out', str(out / f'{name}.json')]
            with (out / f'{name}.log').open('x', encoding='utf-8') as stream:
                result = subprocess.run(command, env=environment, stdout=stream, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f'Failed {name}; inspect log and do not overwrite partial blocks')
            report['measurements'].append({'block': item['block'], 'backend': backend,
                                            'file': f'{name}.json', 'before': before, 'after': telemetry()})
            (out / 'progress.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
            print(f'Completed {name}', flush=True)
    (out / 'completed.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
