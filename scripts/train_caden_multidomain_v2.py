"""Frozen three-seed multi-domain continuation, validation, calibration and timing."""
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time
import numpy as np
from scipy.optimize import minimize_scalar
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from decision_lab.common import read_data,validate_probs,percentile
from decision_lab.encoder import CandidateEncoder
DATA=ROOT/'data/caden-multidomain-v2'
OUT=ROOT/'results/caden-multidomain-v2'

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save_json(path,value):path.write_text(json.dumps(value,indent=2),encoding='utf-8')

def prepare():
    if DATA.exists() or OUT.exists():raise FileExistsError('Preserve frozen data/run; use explicit resume')
    DATA.mkdir(parents=True);OUT.mkdir(parents=True)
    old=read_data(ROOT/'data/mixed-banking-clinc8/train.jsonl')
    devs={'source':read_data(ROOT/'data/mixed-banking-clinc8/dev.jsonl')}
    tests={'banking':read_data(ROOT/'data/confirmation8-full/test.jsonl'),'clinc':read_data(ROOT/'data/mixed-clinc-confirmation8/test.jsonl')}
    paths={'banking':'data/confirmation8-full/test.jsonl','clinc':'data/mixed-clinc-confirmation8/test.jsonl',
        'snips':'data/expansion-v1/snips/test.jsonl','agnews':'data/expansion-v1/agnews/test.jsonl'}
    target={}
    for task in ['snips','agnews']:
        target[task]=read_data(ROOT/f'data/expansion-v1/{task}/train.jsonl')
        devs[task]=read_data(ROOT/f'data/expansion-v1/{task}/dev.jsonl')
        tests[task]=read_data(ROOT/paths[task])
    excluded={r['group_id'] for rows in [*devs.values(),*tests.values()] for r in rows}
    original_cal=read_data(ROOT/'data/mixed-banking-clinc8/calibration.jsonl')
    original_cal_groups={r['group_id'] for r in original_cal}
    excluded|=original_cal_groups
    old_train_groups={r['group_id'] for r in old}
    rng=random.Random(20261002);calibration=list(original_cal);newcalgroups=set()
    for task,pool in target.items():
        forbidden_calibration=excluded|old_train_groups|newcalgroups
        choices=[r for r in pool if r['group_id'] not in forbidden_calibration]
        rng.shuffle(choices);picked=choices[:300];assert len(picked)==300
        calibration.extend({**r,'id':'v2-cal-'+task+'-'+r['id'],'split':'calibration'} for r in picked)
        newcalgroups|={r['group_id'] for r in picked}
    excluded|=newcalgroups
    selected=[];counts={}
    for task,pool in [('banking',[r for r in old if r['domain']=='banking']),('clinc',[r for r in old if r['domain']=='clinc']),*target.items()]:
        pool=[r for r in pool if r['group_id'] not in excluded];rng.shuffle(pool)
        chosen=pool[:3000];assert len(chosen)==3000
        selected.extend({**r,'id':'v2-train-'+task+'-'+r['id']} for r in chosen);counts[task]=3000
    assert len(selected)==12000 and not {r['group_id'] for r in selected}&excluded
    for name,rows in [('train',selected),('calibration',calibration),*[(task+'-dev',rows) for task,rows in devs.items()]]:
        (DATA/(name+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
    protocol={'version':'Caden-Multidomain-v2','seeds':[42,43,44],'epochs':1,'lr':1e-5,'microbatch':1,'gradient_clip':1,
        'train_items':12000,'domain_counts':counts,'data_seed':20261002,'candidate_order_augmentation':True,
        'checkpoint_initialization':'matching original mixed seed, not the exploratory pilot',
        'calibration_items':len(calibration),'new_target_calibration':600,
        'selection':'one pre-fixed epoch; release gate checks mean old/source dev retention (no worse than -1pp) and both target dev improvements. No test selection.',
        'data_files':{p.name:{'sha256':digest(p),'bytes':p.stat().st_size} for p in DATA.glob('*.jsonl')},
        'test_files':{name:{'path':path,'sha256':digest(ROOT/path)} for name,path in paths.items()},
        'scope':'Continuation registered after exploratory prior benchmark reads; full test reruns are exploratory, not new untouched confirmations. Source replay and target supervision; no paid API. Old artifacts remain recoverable.'}
    save_json(DATA/'protocol.json',protocol);save_json(OUT/'protocol.json',protocol)
    return protocol,selected,calibration,devs,tests

def evaluate(model,rows,folder,temperature=1.0):
    folder.mkdir(parents=True,exist_ok=False);records=[];correct=0;loss=0;lat=[]
    model.eval()
    for r in rows:
        torch.cuda.synchronize();start=time.perf_counter()
        with torch.inference_mode():z=model([r])[0].float().cpu().tolist()
        scaled=np.array(z)/temperature;weights=np.exp(scaled-scaled.max());p=weights/weights.sum()
        probs=dict(zip(r['criteria'],p.tolist()));validate_probs(probs,r['criteria']);torch.cuda.synchronize();ms=1000*(time.perf_counter()-start)
        gold=list(r['criteria']).index(r['label']);correct+=int(np.argmax(p)==gold)
        loss+=float(np.log(np.exp(scaled-scaled.max()).sum())+scaled.max()-scaled[gold]);lat.append(ms)
        records.append({'id':r['id'],'group_id':r['group_id'],'label':r['label'],'split':r.get('split'),
            'logits':dict(zip(r['criteria'],z)),'probabilities':probs,'latency_ms':ms})
    (folder/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records),encoding='utf-8')
    metrics={'items':len(rows),'failures':0,'accuracy_percent':100*correct/len(rows),'nll_raw_scores':loss/len(rows),
        'latency_p50_ms':statistics.median(lat),'latency_p95_ms':percentile(lat,.95),'latency_rate_per_second':len(rows)/(sum(lat)/1000),
        'temperature':temperature,'latency_scope':'FP32 eager GPU, two CPU threads, B1 tokenization+forward+scores readback+normalization+validation, no loading; raw scores readback included'}
    save_json(folder/'metrics.json',metrics);return records,metrics

def fit_calibration(records):
    if not records or any(r.get('split')!='calibration' for r in records):raise ValueError('Only explicit calibration records can fit temperature')
    for r in records:
        if r['label'] not in r['logits'] or set(r['logits'])!=set(r['probabilities']):raise ValueError('Calibration support mismatch')
        if not all(math.isfinite(z) for z in r['logits'].values()):raise ValueError('Nonfinite calibration score')
    width=max(len(r['logits']) for r in records);matrix=np.full((len(records),width),-np.inf);gold=np.zeros(len(records))
    for i,r in enumerate(records):
        scores=list(r['logits'].values());matrix[i,:len(scores)]=scores;gold[i]=r['logits'][r['label']]
    def objective(logt):
        t=math.exp(logt);scores=matrix/t;maximum=scores.max(axis=1)
        return float(np.mean(np.log(np.exp(scores-maximum[:,None]).sum(axis=1))+maximum-gold/t))
    result=minimize_scalar(objective,bounds=(-4,4),method='bounded',options={'xatol':1e-9})
    assert result.success
    return {'temperature':math.exp(result.x),'valid_items':len(records),'score_source':'raw_logits','calibration_nll_before':objective(0),
        'calibration_nll_after':float(result.fun),'search_log_temperature_bounds':[-4,4],'at_search_boundary':bool(abs(float(result.x))>3.99)}

def main():
    torch.set_num_threads(2)
    protocol,train,cal,devs,tests=prepare();completed=[];summary=[]
    olddev={s:100*json.loads((ROOT/f'results/encoder16-s{s}-e3-mixed23789-dev/metrics.json').read_text())['accuracy_all_failures_wrong'] for s in [42,43,44]}
    diagnosis=json.loads((ROOT/'results/benchmark-expansion-v1/transfer-diagnosis/summary.json').read_text())
    for seed in protocol['seeds']:
        random.seed(seed);torch.manual_seed(seed);rng=random.Random(seed)
        initial=ROOT/f'checkpoints/encoder16-s{seed}-e3-mixed23789';checkpoint=ROOT/f'checkpoints/caden-s{seed}-multidomain-v2-e1'
        if checkpoint.exists():raise FileExistsError('Preserve checkpoint')
        model=CandidateEncoder.load(initial).to('cuda');optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5)
        rows=list(train);rng.shuffle(rows);losses=[];start=time.perf_counter();model.train()
        with (OUT/f'training-s{seed}.jsonl').open('w') as stream:
            for i,r in enumerate(rows):
                pairs=list(r['criteria'].items());rng.shuffle(pairs);r={**r,'criteria':dict(pairs)}
                z=model([r])[0];loss=torch.nn.functional.cross_entropy(z[None],torch.tensor([list(r['criteria']).index(r['label'])],device='cuda'))
                optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();losses.append(loss.item())
                if (i+1)%1000==0:
                    report={'seed':seed,'step':i+1,'total':len(rows),'loss_recent':statistics.mean(losses[-1000:]),'elapsed_s':time.perf_counter()-start}
                    stream.write(json.dumps(report)+'\n');stream.flush();print(json.dumps(report),flush=True)
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        model.save(checkpoint,{**protocol,'seed':seed,'model':'distilbert/distilbert-base-uncased','resolved_revision':'12040accade4e8a0f71eabdb258fecc2e7e948be',
            'initial_checkpoint':initial.relative_to(ROOT).as_posix(),'initial_weight_sha256':digest(initial/'backbone/model.safetensors'),
            'training_seconds':elapsed,'loss_history':[statistics.mean(losses)],'torch':torch.__version__,'max_length':512,'device':'cuda'})
        model.eval();dev={task:evaluate(model,rows,OUT/f's{seed}/{task}-dev')[1] for task,rows in devs.items()}
        completed.append({'seed':seed,'checkpoint':checkpoint.relative_to(ROOT).as_posix(),'training_seconds':elapsed,'dev':dev})
        save_json(OUT/'progress.json',{'completed':completed})
        del optimizer,model;torch.cuda.empty_cache()
    source_change=statistics.mean(r['dev']['source']['accuracy_percent']-olddev[r['seed']] for r in completed)
    gates={'source_dev_change_pp':source_change,'source_retention_passed':source_change>=-1}
    for task in ['snips','agnews']:
        before=diagnosis['grouped'][task+'-dev-original']['accuracy_mean_percent'];after=statistics.mean(r['dev'][task]['accuracy_percent'] for r in completed)
        gates[task+'_dev_change_pp']=after-before;gates[task+'_improvement_passed']=after>before
    save_json(OUT/'release-gates.json',gates)
    if not all(gates[k] for k in ['source_retention_passed','snips_improvement_passed','agnews_improvement_passed']):
        raise RuntimeError('Pre-fixed release gates failed; preserve results without publishing')
    for row in completed:
        seed=row['seed'];checkpoint=ROOT/row['checkpoint'];model=CandidateEncoder.load(checkpoint).to('cuda').eval()
        preds,metrics=evaluate(model,cal,OUT/f's{seed}/calibration');temperature=fit_calibration(preds)
        temperature.update(calibration_dataset_sha256=digest(DATA/'calibration.jsonl'),checkpoint=row['checkpoint'],backend='encoder',dtype='float32',attention_impl='eager')
        save_json(OUT/f'temperature-s{seed}.json',temperature)
        for task,rows in tests.items():
            preds,metrics=evaluate(model,rows,OUT/f's{seed}/{task}-test-raw')
            scores=[]
            for r in preds:
                z=np.array(list(r['logits'].values()))/temperature['temperature'];g=list(r['logits']).index(r['label']);scores.append(float(np.log(np.exp(z-z.max()).sum())+z.max()-z[g]))
            metrics['calibrated_nll_raw_scores']=statistics.mean(scores)
            summary.append({'seed':seed,'task':task,**metrics,'prediction_sha256':digest(OUT/f's{seed}/{task}-test-raw/predictions.jsonl')})
            save_json(OUT/'evaluation-progress.json',summary)
        del model;torch.cuda.empty_cache()
    model=CandidateEncoder.load(ROOT/completed[0]['checkpoint']).to('cuda').eval();dev=read_data(ROOT/'data/pilot16-8792/dev.jsonl');blocks=[]
    for b in range(5):
        with torch.inference_mode():
            for r in dev[:3]:model([r])
            torch.cuda.synchronize();lat=[]
            for r in dev:
                torch.cuda.synchronize();start=time.perf_counter();z=model([r])[0];p=z.float().softmax(-1).cpu().tolist();validate_probs(dict(zip(r['criteria'],p)),r['criteria']);torch.cuda.synchronize();lat.append(1000*(time.perf_counter()-start))
        blocks.append({'block':b,'p50_ms':statistics.median(lat),'latency_rate_per_second':len(lat)/(sum(lat)/1000),'latency_ms':lat})
    save_json(OUT/'timing.json',{'seed':42,'dataset_sha256':digest(ROOT/'data/pilot16-8792/dev.jsonl'),'blocks':blocks,'dtype':'float32','cpu_threads':2,'scope':'Five serial blocks after all training/evaluation, same200 BANK development payloads; no paired concurrent original timing comparison'})
    aggregated={}
    for task in tests:
        rows=[r for r in summary if r['task']==task];values=[r['accuracy_percent'] for r in rows]
        aggregated[task]={'accuracy_mean_percent':statistics.mean(values),'sample_sd_pp':statistics.stdev(values),'sample_variance_pp_squared':statistics.variance(values),'seeds':[r['seed'] for r in rows],
            'nll_raw_mean':statistics.mean(r['nll_raw_scores'] for r in rows),'calibrated_nll_mean':statistics.mean(r['calibrated_nll_raw_scores'] for r in rows),'items':rows[0]['items'],'failures_total':sum(r['failures'] for r in rows)}
    save_json(OUT/'completed.json',{'protocol':protocol,'training':completed,'gates':gates,'tasks':aggregated,'runs':summary,
        'timing_p50_ms':statistics.median(r['p50_ms'] for r in blocks),'timing_latency_rate_per_second':statistics.median(r['latency_rate_per_second'] for r in blocks),'published':False})
    print(json.dumps({'completed':True,'gates':gates,'tasks':aggregated}),flush=True)

if __name__=='__main__':main()
