"""Evaluate all three frozen Caden mixed seeds on registered full-label tasks."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/benchmark-expansion-v1/caden'
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'

def main():
    if OUT.exists():raise FileExistsError('Preserve existing expansion run')
    OUT.mkdir(parents=True)
    stages=[]
    for task in ['snips','agnews']:
        for seed in [42,43,44]:
            out=OUT/f'{task}-s{seed}'
            command=[sys.executable,'-m','decision_lab','evaluate','--data',f'data/expansion-v1/{task}/test.jsonl',
                '--backend','encoder','--checkpoint',f'checkpoints/encoder16-s{seed}-e3-mixed23789','--device','cuda',
                '--dtype','float32','--attention-impl','eager','--out',str(out)]
            with (OUT/f'{task}-s{seed}.log').open('w',encoding='utf-8') as stream:
                result=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            if result.returncode:raise RuntimeError(f'{task} seed {seed} failed; inspect log, do not reuse output')
            stages.append({'task':task,'seed':seed,'out':out.relative_to(ROOT).as_posix(),'exit_code':result.returncode})
            (OUT/'progress.json').write_text(json.dumps(stages,indent=2),encoding='utf-8')
            print(json.dumps(stages[-1]),flush=True)
    (OUT/'completed.json').write_text(json.dumps({'stages':stages,'scope':'Frozen mixed checkpoints transferred to full-label tasks; no target-domain Caden supervision'},indent=2),encoding='utf-8')

if __name__=='__main__':main()
