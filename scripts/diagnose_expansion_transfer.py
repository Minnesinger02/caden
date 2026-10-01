"""Check source fit and target development transfer; never train on test items."""
import collections
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from decision_lab.common import read_data
from decision_lab.encoder import CandidateEncoder
OUT=ROOT/'results/benchmark-expansion-v1/transfer-diagnosis'

def evaluate(model,rows,path):
    preds=[]
    for row in rows:
        with torch.inference_mode():
            logits=model([row])[0];probs=torch.softmax(logits.float(),-1).cpu().tolist()
        pred=dict(zip(row['criteria'],probs));choice=max(pred,key=pred.get)
        preds.append({'id':row['id'],'group_id':row['group_id'],'label':row['label'],'choice':choice,'confidence':pred[choice],
            'probabilities':pred,'correct':choice==row['label']})
    path.write_text(''.join(json.dumps(r)+'\n' for r in preds),encoding='utf-8')
    correct=[r for r in preds if r['correct']];errors=[r for r in preds if not r['correct']]
    counts=collections.Counter((r['label'],r['choice']) for r in preds)
    return {'items':len(rows),'accuracy_percent':100*len(correct)/len(rows),
        'mean_confidence':statistics.mean(r['confidence'] for r in preds),
        'wrong_mean_confidence':statistics.mean(r['confidence'] for r in errors) if errors else None,
        'wrong_over_90pct':sum(r['confidence']>.9 for r in errors),
        'per_class_accuracy_percent':{label:100*sum(r['correct'] for r in preds if r['label']==label)/sum(r['label']==label for r in preds) for label in sorted({r['label'] for r in preds})},
        'largest_confusions':[{'gold':a,'predicted':b,'n':n} for (a,b),n in counts.most_common() if a!=b][:10]}

def main():
    if OUT.exists():raise FileExistsError('Preserve diagnostic runs')
    OUT.mkdir(parents=True)
    torch.set_num_threads(2)
    train=read_data(ROOT/'data/mixed-banking-clinc8/train.jsonl')
    rng=random.Random(20261001);sample=[]
    for domain in ['banking','clinc']:
        pool=[r for r in train if r.get('domain')==domain];rng.shuffle(pool);sample.extend(pool[:200])
    assert len(sample)==400
    samplepath=OUT/'source-train-sample.jsonl';samplepath.write_text(''.join(json.dumps(r)+'\n' for r in sample),encoding='utf-8')
    variants={'source-train-sample':sample}
    for task in ['snips','agnews']:
        rows=read_data(ROOT/f'data/expansion-v1/{task}/dev.jsonl')
        manifest=json.loads((ROOT/f'data/expansion-v1/{task}/manifest.json').read_text(encoding='utf-8'))
        names={str(k):str(v) for k,v in manifest['labels'].items()}
        variants[task+'-dev-original']=rows
        renamed=[]
        for row in rows:
            mapping={key:'_'.join(names[key].lower().split()) for key in row['criteria']}
            assert len(set(mapping.values()))==len(mapping)
            renamed.append({**row,'criteria':{mapping[k]:v for k,v in row['criteria'].items()},'label':mapping[row['label']]})
        variants[task+'-dev-semantic-ids']=renamed
    protocol={'seed':20261001,'source_train_sample':400,'diagnostic_only':True,
        'variants':{name:{'items':len(rows),'payload_sha256':hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()} for name,rows in variants.items()},
        'hypotheses':['source overfit versus domain transfer','numeric versus semantic candidate-ID format shift'],
        'scope':'Controls ID text while retaining all descriptions, order and gold identity. Target dev only; no test prompt selection, no new training, no paid API.'}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    results=[]
    for seed in [42,43,44]:
        model=CandidateEncoder.load(ROOT/f'checkpoints/encoder16-s{seed}-e3-mixed23789').to('cuda').eval()
        for name,rows in variants.items():
            report={'seed':seed,'variant':name,**evaluate(model,rows,OUT/f'{name}-s{seed}.jsonl')}
            results.append(report);print(json.dumps({'seed':seed,'variant':name,'accuracy_percent':report['accuracy_percent']}),flush=True)
        del model;torch.cuda.empty_cache()
    grouped={name:{'accuracy_mean_percent':statistics.mean(r['accuracy_percent'] for r in results if r['variant']==name),
        'accuracy_sample_sd_pp':statistics.stdev(r['accuracy_percent'] for r in results if r['variant']==name)} for name in variants}
    (OUT/'summary.json').write_text(json.dumps({'protocol':protocol,'grouped':grouped,'runs':results},indent=2),encoding='utf-8')

if __name__=='__main__':main()
