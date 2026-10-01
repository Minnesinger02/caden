"""Independently check new predictions, dataset identity, split separation and timing."""
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from decision_lab.common import read_data,validate_probs
OUT=ROOT/'results/caden-multidomain-v2'

def main():
    report=json.loads((OUT/'completed.json').read_text());protocol=report['protocol']
    train=read_data(ROOT/'data/caden-multidomain-v2/train.jsonl');cal=read_data(ROOT/'data/caden-multidomain-v2/calibration.jsonl')
    training_groups={r['group_id'] for r in train};held={r['group_id'] for r in cal};verified=[]
    assert len(train)==12000 and len(cal)==3996 and not training_groups&held
    for task in ['source','snips','agnews']:held.update(r['group_id'] for r in read_data(ROOT/f'data/caden-multidomain-v2/{task}-dev.jsonl'))
    for task,spec in protocol['test_files'].items():
        data=read_data(ROOT/spec['path']);assert hashlib.sha256((ROOT/spec['path']).read_bytes()).hexdigest()==spec['sha256'];held.update(r['group_id'] for r in data)
        for seed in [42,43,44]:
            folder=OUT/f's{seed}/{task}-test-raw';path=folder/'predictions.jsonl';preds=[json.loads(s) for s in path.read_text().splitlines()]
            assert len(data)==len(preds)
            correct=0;nll=[]
            for row,pred in zip(data,preds):
                assert row['id']==pred['id'] and row['group_id']==pred['group_id'] and row['label']==pred['label']
                validate_probs(pred['probabilities'],row['criteria']);assert set(pred['logits'])==set(row['criteria'])
                assert all(math.isfinite(v) for v in pred['logits'].values())
                correct+=max(pred['probabilities'],key=pred['probabilities'].get)==row['label']
                z=pred['logits'];maximum=max(z.values());nll.append(math.log(sum(math.exp(v-maximum) for v in z.values()))+maximum-z[row['label']])
            summary=next(r for r in report['runs'] if r['task']==task and r['seed']==seed)
            assert abs(summary['accuracy_percent']-100*correct/len(data))<1e-10 and abs(summary['nll_raw_scores']-statistics.mean(nll))<1e-9
            assert summary['prediction_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
            verified.append({'task':task,'seed':seed,'items':len(data),'accuracy_percent':summary['accuracy_percent']})
    assert not training_groups&held
    timing=json.loads((OUT/'timing.json').read_text());assert len(timing['blocks'])==5
    for block in timing['blocks']:
        assert len(block['latency_ms'])==200 and min(block['latency_ms'])>0
        assert abs(block['latency_rate_per_second']-200/(sum(block['latency_ms'])/1000))<1e-8
    integrity={'all_prediction_files_verified':True,'runs_verified':len(verified),'questions_verified':sum(r['items'] for r in verified),
        'train_heldout_groups_disjoint':True,'train_items':len(train),'calibration_items':len(cal),'timing_blocks_verified':5,
        'scope':'File identity, exact per-item support/labels/group IDs, accuracy/NLL recomputation, no local exact-text training overlap; no full upstream leakage or deployment certification.',
        'completed_sha256':hashlib.sha256((OUT/'completed.json').read_bytes()).hexdigest()}
    (OUT/'integrity-audit.json').write_text(json.dumps(integrity,indent=2),encoding='utf-8');print(json.dumps(integrity,indent=2))

if __name__=='__main__':main()
