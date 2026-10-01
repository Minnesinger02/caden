"""Audit canonical classic accuracy and time descriptor scoring on matched dev."""
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from decision_lab.common import read_data,softmax,validate_probs
OUT=ROOT/'results/benchmark-expansion-v1'

def main():
    classic=json.loads((OUT/'classic/summary.json').read_text(encoding='utf-8'))
    full=json.loads((OUT/'classic-banking-full/summary.json').read_text(encoding='utf-8'))
    report={'rows':[],'scope':'Same frozen BANK3079/CLINC4197 payloads; mixed train labels for linear baseline. CPU timing versus GPU rows is a complete-system comparison, not matched compute.',
        'training_sha256':hashlib.sha256((ROOT/'data/mixed-banking-clinc8/train.jsonl').read_bytes()).hexdigest(),
        'excluded_pilot':'classic/banking used200 pilot questions; excluded from this3079 main comparison.'}
    for system in ['tfidf-description','tfidf-linear']:
        item={'system':system,'tasks':{}}
        for task,source,folder in [('banking',full,'classic-banking-full'),('clinc',classic,'classic')]:
            records=[r for r in source['rows'] if r['task']==task and r['system']==system]
            vals=[];nlls=[]
            for r in records:
                sub='tfidf-description' if system=='tfidf-description' else f"tfidf-linear-s{r['seed']}"
                preds=[json.loads(s) for s in (OUT/folder/task/sub/'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
                data=read_data(ROOT/r['dataset'])
                assert len(preds)==len(data)==(3079 if task=='banking' else 4197)
                for a,b in zip(data,preds):
                    assert a['id']==b['id'] and a['group_id']==b['group_id'] and a['label']==b['label']
                    validate_probs(b['probabilities'],a['criteria'])
                accuracy=sum(max(p['probabilities'],key=p['probabilities'].get)==p['label'] for p in preds)/len(preds)
                assert abs(accuracy-r['accuracy'])<1e-12
                vals.append(accuracy*100);nlls.append(statistics.mean(-math.log(max(p['probabilities'][p['label']],1e-15)) for p in preds))
            item['tasks'][task]={'items':len(data),'seeds':[r.get('seed') for r in records],'accuracy_by_seed_percent':vals,'accuracy_mean_percent':statistics.mean(vals),
                'sample_sd_pp':statistics.stdev(vals) if len(vals)>1 else None,
                'sample_variance_pp_squared':statistics.variance(vals) if len(vals)>1 else None,'raw_nll_mean':statistics.mean(nlls),
                'dataset_sha256':records[0]['dataset_sha256']}
        report['rows'].append(item)
    train=read_data(ROOT/'data/mixed-banking-clinc8/train.jsonl');devpath=ROOT/'data/pilot16-8792/dev.jsonl';dev=read_data(devpath)
    vec=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=30000,sublinear_tf=True)
    vec.fit([r['state'] for r in train]);cache={tuple(r['criteria'].values()):vec.transform(list(r['criteria'].values())) for r in dev}
    def predict(r):
        scores=(cache[tuple(r['criteria'].values())]@vec.transform([r['state']]).T).toarray().ravel()
        return dict(zip(r['criteria'],softmax(scores.tolist())))
    blocks=[]
    for block in range(5):
        for r in dev[:3]:predict(r)
        lat=[]
        for r in dev:
            start=time.perf_counter();p=predict(r);validate_probs(p,r['criteria']);lat.append(1000*(time.perf_counter()-start))
        blocks.append({'block':block,'latency_ms':lat,'p50_ms':statistics.median(lat),'latency_rate_per_second':len(lat)/(sum(lat)/1000)})
    assert hashlib.sha256(devpath.read_bytes()).hexdigest()==json.loads((ROOT/'results/replicated-task-fp32-8792/summary-mixed-reference.json').read_text())['dataset_sha256']
    report['description_timing']={'dataset_sha256':hashlib.sha256(devpath.read_bytes()).hexdigest(),'blocks':blocks,'cpu_threads':2,'scope':'CPU B1 query vectorization+description similarity+probabilities+validation; static description vectors precomputed, no loading/training'}
    report['linear_timing']=json.loads((OUT/'classic-banking-full/banking/tfidf-linear-s42/timing-blocks.json').read_text())
    (OUT/'classic-main-comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'rows':report['rows'],'cpu_timing_completed':True},indent=2))

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
