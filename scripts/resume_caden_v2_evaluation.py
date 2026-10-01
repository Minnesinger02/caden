"""Resume completed training at calibration/evaluation without retraining seeds."""
import json
from pathlib import Path
import statistics
import sys
import time
import torch
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.train_caden_multidomain_v2 import OUT,DATA,save_json,digest,evaluate,fit_calibration
from decision_lab.common import read_data,validate_probs
from decision_lab.encoder import CandidateEncoder

def main():
    torch.set_num_threads(2)
    protocol=json.loads((OUT/'protocol.json').read_text());completed=json.loads((OUT/'progress.json').read_text())['completed'];assert len(completed)==3
    gates=json.loads((OUT/'release-gates.json').read_text());assert all(gates[k] for k in ['source_retention_passed','snips_improvement_passed','agnews_improvement_passed'])
    cal=read_data(DATA/'calibration.jsonl');tests={task:read_data(ROOT/r['path']) for task,r in protocol['test_files'].items()}
    summary=[]
    for row in completed:
        seed=row['seed'];checkpoint=ROOT/row['checkpoint'];model=CandidateEncoder.load(checkpoint).to('cuda').eval()
        folder=OUT/f's{seed}/calibration'
        if folder.exists():
            preds=[json.loads(s) for s in (folder/'predictions.jsonl').read_text().splitlines()]
            assert len(preds)==len(cal) and all(p['id']==r['id'] for p,r in zip(preds,cal))
        else:preds,_=evaluate(model,cal,folder)
        temperature=fit_calibration(preds)
        temperature.update(calibration_dataset_sha256=digest(DATA/'calibration.jsonl'),checkpoint=row['checkpoint'],backend='encoder',dtype='float32',attention_impl='eager')
        save_json(OUT/f'temperature-s{seed}.json',temperature)
        for task,rows in tests.items():
            folder=OUT/f's{seed}/{task}-test-raw'
            if folder.exists():
                preds=[json.loads(s) for s in (folder/'predictions.jsonl').read_text().splitlines()];metrics=json.loads((folder/'metrics.json').read_text())
                assert len(preds)==len(rows) and all(p['id']==r['id'] for p,r in zip(preds,rows))
            else:preds,metrics=evaluate(model,rows,folder)
            scores=[]
            for r in preds:
                z=np.array(list(r['logits'].values()))/temperature['temperature'];g=list(r['logits']).index(r['label']);scores.append(float(np.log(np.exp(z-z.max()).sum())+z.max()-z[g]))
            metrics['calibrated_nll_raw_scores']=statistics.mean(scores)
            summary.append({'seed':seed,'task':task,**metrics,'prediction_sha256':digest(folder/'predictions.jsonl')})
            save_json(OUT/'evaluation-progress.json',summary);print(json.dumps({'seed':seed,'task':task,'accuracy_percent':metrics['accuracy_percent']}),flush=True)
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
        'timing_p50_ms':statistics.median(r['p50_ms'] for r in blocks),'timing_latency_rate_per_second':statistics.median(r['latency_rate_per_second'] for r in blocks),'published':False,
        'recovery':'All training succeeded. Original driver stopped while serializing a NumPy boundary boolean; scalar conversion fixed and calibration/evaluation resumed. Cached seed42 calibration reused; no retraining.'})
    print(json.dumps({'completed':True,'gates':gates,'tasks':aggregated}),flush=True)

if __name__=='__main__':main()
