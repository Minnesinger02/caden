"""Run registered CPU TF-IDF description and supervised linear baselines."""
import hashlib
import argparse
import json
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
from pathlib import Path
import statistics
import sys
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
import sklearn
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from decision_lab.common import read_data, softmax, validate_probs
OUT=ROOT/'results/benchmark-expansion-v1/classic'

def evaluate(rows,predict,path,data):
    path.mkdir(parents=True,exist_ok=False)
    preds=[]
    for row in rows:
        t=time.perf_counter(); p=predict(row);validate_probs(p,row['criteria'])
        preds.append({'id':row['id'],'group_id':row['group_id'],'label':row['label'],'probabilities':p,
            'latency_ms':1000*(time.perf_counter()-t)})
    (path/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in preds),encoding='utf-8')
    acc=sum(max(r['probabilities'],key=r['probabilities'].get)==r['label'] for r in preds)/len(preds)
    report={'items':len(preds),'accuracy':acc,'failures':0,'dataset':str(data.relative_to(ROOT)),
        'dataset_sha256':hashlib.sha256(data.read_bytes()).hexdigest(),'latency_p50_ms':statistics.median(r['latency_ms'] for r in preds),
        'latency_rate_per_second':len(preds)/(sum(r['latency_ms'] for r in preds)/1000)}
    (path/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8');return report

def main():
    global OUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--banking-full-only',action='store_true')
    args=parser.parse_args()
    if args.banking_full_only:OUT=ROOT/'results/benchmark-expansion-v1/classic-banking-full'
    if OUT.exists():raise FileExistsError('Preserve existing baseline runs')
    OUT.mkdir(parents=True)
    tasks=[('banking',ROOT/'data/mixed-banking-clinc8/train.jsonl',ROOT/'data/confirmation8-full/test.jsonl'),
           ('clinc',ROOT/'data/mixed-banking-clinc8/train.jsonl',ROOT/'data/mixed-clinc-confirmation8/test.jsonl')]
    tasks += [(name,ROOT/f'data/expansion-v1/{name}/train.jsonl',ROOT/f'data/expansion-v1/{name}/test.jsonl') for name in ['snips','agnews']]
    if args.banking_full_only:tasks=tasks[:1]
    summaries=[]
    for task,trainpath,testpath in tasks:
        train=read_data(trainpath);test=read_data(testpath)
        vectorizer=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=30000,sublinear_tf=True)
        X=vectorizer.fit_transform([r['state'] for r in train]);labels=[r['label'] for r in train]
        cache={}
        # Static descriptions cached explicitly; unknown dynamic descriptions encoded once.
        def description_predict(row):
            keys=list(row['criteria']);descriptions=tuple(row['criteria'].values())
            if descriptions not in cache:cache[descriptions]=vectorizer.transform(descriptions)
            scores=(cache[descriptions] @ vectorizer.transform([row['state']]).T).toarray().ravel()
            return dict(zip(keys,softmax(scores.tolist())))
        report=evaluate(test,description_predict,OUT/task/'tfidf-description',testpath)
        summaries.append({'task':task,'system':'tfidf-description','training_scope':'train-fitted vocabulary/IDF; no label supervision',**report})
        for seed in [42,43,44]:
            model=SGDClassifier(loss='log_loss',alpha=1e-4,max_iter=5,tol=None,random_state=seed)
            model.fit(X,labels);classes={str(c):i for i,c in enumerate(model.classes_)}
            def linear_predict(row):
                scores=np.asarray(model.decision_function(vectorizer.transform([row['state']]))).ravel()
                assert len(scores)==len(classes)
                return dict(zip(row['criteria'],softmax([float(scores[classes[k]]) if k in classes else -100 for k in row['criteria']])))
            report=evaluate(test,linear_predict,OUT/task/f'tfidf-linear-s{seed}',testpath)
            summaries.append({'task':task,'system':'tfidf-linear','seed':seed,'training_scope':'target train supervised' if task in ['snips','agnews'] else 'same mixed BANKING+CLINC training as Caden mixed',**report})
            if task=='banking':
                dev=read_data(ROOT/'data/pilot16-8792/dev.jsonl');blocks=[]
                for block in range(5):
                    for r in dev[:3]:linear_predict(r)
                    lat=[]
                    for r in dev:
                        start=time.perf_counter();p=linear_predict(r);validate_probs(p,r['criteria']);lat.append(1000*(time.perf_counter()-start))
                    blocks.append({'block':block,'latency_ms':lat,'p50_ms':statistics.median(lat),'latency_rate_per_second':len(lat)/(sum(lat)/1000)})
                (OUT/task/f'tfidf-linear-s{seed}'/'timing-blocks.json').write_text(json.dumps({'cpu_threads':2,'blocks':blocks,'scope':'CPU B1 vectorization+linear head+probabilities+validation, no train/load; fixed classes'},indent=2),encoding='utf-8')
        print(json.dumps({'task':task,'rows':summaries[-4:]}),flush=True)
    (OUT/'summary.json').write_text(json.dumps({'rows':summaries,'sklearn':sklearn.__version__,'cpu_threads':2,
        'recipe':{'ngram':[1,2],'min_df':2,'max_features':30000,'sublinear_tf':True,'loss':'log_loss','alpha':1e-4,'epochs':5},
        'scope':'Registered classic CPU baselines; supervision scopes differ across tasks, descriptive rather than causal comparison'},indent=2),encoding='utf-8')

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
