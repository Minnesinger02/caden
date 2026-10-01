"""One registered, development-only warm-start pilot with source-domain replay."""
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from decision_lab.common import read_data
from decision_lab.encoder import CandidateEncoder
OUT=ROOT/'results/benchmark-expansion-v1/warmstart-pilot'
CHECKPOINT=ROOT/'checkpoints/caden-s42-expansion-pilot4000-e1'

def accuracy(model,rows):
    correct=0
    for r in rows:
        with torch.inference_mode():z=model([r])[0]
        correct+=list(r['criteria'])[z.argmax().item()]==r['label']
    return {'items':len(rows),'accuracy_percent':100*correct/len(rows)}

def main():
    if OUT.exists() or CHECKPOINT.exists():raise FileExistsError('Preserve pilot output')
    OUT.mkdir(parents=True)
    torch.set_num_threads(2);random.seed(42);torch.manual_seed(42)
    source=read_data(ROOT/'data/mixed-banking-clinc8/train.jsonl')
    devs={'source':read_data(ROOT/'data/mixed-banking-clinc8/dev.jsonl')}
    target={}
    for task in ['snips','agnews']:
        target[task]=read_data(ROOT/f'data/expansion-v1/{task}/train.jsonl')
        devs[task]=read_data(ROOT/f'data/expansion-v1/{task}/dev.jsonl')
    forbidden={r['group_id'] for rows in devs.values() for r in rows}
    for name in ['data/confirmation8-full/test.jsonl','data/mixed-clinc-confirmation8/test.jsonl',
        'data/expansion-v1/snips/test.jsonl','data/expansion-v1/agnews/test.jsonl']:
        forbidden.update(r['group_id'] for r in read_data(ROOT/name))
    rng=random.Random(20261001);rows=[];counts={}
    for task,pool in [('banking',[r for r in source if r['domain']=='banking']),('clinc',[r for r in source if r['domain']=='clinc']),*target.items()]:
        pool=[r for r in pool if r['group_id'] not in forbidden];rng.shuffle(pool)
        chosen=pool[:1000];assert len(chosen)==1000
        rows.extend({**r,'id':'pilot-'+task+'-'+r['id']} for r in chosen);counts[task]=len(chosen)
    assert len(rows)==4000 and not {r['group_id'] for r in rows}&forbidden
    data=OUT/'train.jsonl';data.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
    initial=ROOT/'checkpoints/encoder16-s42-e3-mixed23789'
    protocol={'initial_checkpoint':str(initial.relative_to(ROOT)),'epochs':1,'lr':1e-5,'seed':42,'data_seed':20261001,
        'train_items':4000,'domain_counts':counts,'data_sha256':hashlib.sha256(data.read_bytes()).hexdigest(),
        'checkpoint':str(CHECKPOINT.relative_to(ROOT)),'microbatch':1,'gradient_clip':1,'candidate_order_augmentation':True,
        'validation_only':True,'test_scoring':False,'scope':'Exploratory continuation after earlier baseline test scores; not an unseen-test confirmation or equal-target-supervision architecture comparison.'}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    model=CandidateEncoder.load(initial).to('cuda')
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5)
    rng.shuffle(rows);losses=[];start=time.perf_counter();model.train()
    for i,r in enumerate(rows):
        items=list(r['criteria'].items());rng.shuffle(items);r={**r,'criteria':dict(items)}
        z=model([r])[0];loss=torch.nn.functional.cross_entropy(z[None],torch.tensor([list(r['criteria']).index(r['label'])],device='cuda'))
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();losses.append(loss.item())
        if (i+1)%500==0:print(json.dumps({'step':i+1,'total':4000,'loss_recent':sum(losses[-500:])/500,'elapsed_s':time.perf_counter()-start}),flush=True)
    torch.cuda.synchronize();elapsed=time.perf_counter()-start
    metadata={**protocol,'model':'distilbert/distilbert-base-uncased','resolved_revision':'12040accade4e8a0f71eabdb258fecc2e7e948be',
        'torch':torch.__version__,'training_seconds':elapsed,'loss_history':[sum(losses)/len(losses)],'device':'cuda','max_length':512,
        'initial_weight_sha256':hashlib.sha256((initial/'backbone/model.safetensors').read_bytes()).hexdigest()}
    model.save(CHECKPOINT,metadata);model.eval()
    before={task:json.loads((ROOT/f'results/benchmark-expansion-v1/transfer-diagnosis/summary.json').read_text())['runs'] for task in []}
    report={'protocol':protocol,'training_seconds':elapsed,'dev':{task:accuracy(model,values) for task,values in devs.items()},
        'test_scored':False,'published':False,'one_seed_pilot':True}
    (OUT/'completed.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
