"""Audit per-question new benchmark predictions and report actual seed variance."""
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from decision_lab.common import read_data,validate_probs
OUT=ROOT/'results/benchmark-expansion-v1'

def main():
    classic=json.loads((OUT/'classic/summary.json').read_text(encoding='utf-8'))
    completed=json.loads((OUT/'caden/completed.json').read_text(encoding='utf-8'))
    assert len(completed['stages'])==6
    report={'tasks':{},'scope':'Full-label target-domain transfer for Caden mixed versus target-trained classic baselines; supervision is unequal, not a causal architecture comparison.',
        'classic_pilot_note':'First classic banking run used the historical200 question pilot. The main3079-question comparison is separately recorded in classic-banking-full; never substitute200 for3079.',
        'paid_api_requests':0,'community_models_on_new_tasks':'not yet evaluated'}
    for task in ['snips','agnews']:
        path=ROOT/f'data/expansion-v1/{task}/test.jsonl';data=read_data(path);rows=[]
        for system in ['caden-mixed','tfidf-description','tfidf-linear']:
            seeds=[None] if system=='tfidf-description' else [42,43,44]
            values=[];per=[]
            for seed in seeds:
                folder=(OUT/f'caden/{task}-s{seed}' if system=='caden-mixed' else OUT/f'classic/{task}'/('tfidf-description' if seed is None else f'tfidf-linear-s{seed}'))
                preds=[json.loads(s) for s in (folder/'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
                assert len(preds)==len(data)
                correct=0;failures=0
                for datum,pred in zip(data,preds):
                    assert pred['id']==datum['id'] and pred['label']==datum['label'] and pred['group_id']==datum['group_id']
                    if 'probabilities' not in pred:failures+=1;continue
                    validate_probs(pred['probabilities'],datum['criteria'])
                    correct+=max(pred['probabilities'],key=pred['probabilities'].get)==datum['label']
                saved=json.loads((folder/'metrics.json').read_text(encoding='utf-8'))
                accuracy=correct/len(data)
                assert abs(accuracy-saved.get('accuracy',saved.get('accuracy_all_failures_wrong')))<1e-12
                values.append(accuracy*100)
                per.append({'seed':seed,'accuracy_percent':accuracy*100,'failures':failures,'prediction_sha256':hashlib.sha256((folder/'predictions.jsonl').read_bytes()).hexdigest(),
                    'run':folder.relative_to(ROOT).as_posix()})
            rows.append({'system':system,'accuracy_mean_percent':statistics.mean(values),
                'sample_sd_pp':statistics.stdev(values) if len(values)>1 else None,
                'sample_variance_pp_squared':statistics.variance(values) if len(values)>1 else None,
                'runs':per,'training_scope':'BANK+CLINC, target-domain transfer' if system=='caden-mixed' else 'target train IDF only' if system=='tfidf-description' else 'target train supervised'})
        report['tasks'][task]={'items':len(data),'groups':len({r['group_id'] for r in data}),'candidate_count':len(data[0]['criteria']),
            'dataset_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'oracle_shortlist':False,'rows':rows}
    (OUT/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    lines=['# Full-label benchmark expansion v1','',report['scope'],'', '| Task | n | System | Train exposure | Mean accuracy/% | SD/pp | Variance/pp² |','|---|---:|---|---|---:|---:|---:|']
    for name,task in report['tasks'].items():
        for r in task['rows']:
            sd='N/A' if r['sample_sd_pp'] is None else f"{r['sample_sd_pp']:.4f}"
            var='N/A' if r['sample_variance_pp_squared'] is None else f"{r['sample_variance_pp_squared']:.6f}"
            lines.append(f"| {name} | {task['items']} | {r['system']} | {r['training_scope']} | {r['accuracy_mean_percent']:.4f} | {sd} | {var} |")
    lines += ['', 'SNIPS uses the pinned DeepPavlov HF variant1400 test rows (1395 normalized text groups), not an asserted original700-question test. AG News has7600 test rows. Full candidate sets use no gold-dependent distractor sampling.',
        'All Caden runs use the three original mixed-training seeds, not three target-domain retrainings. Classic linear baselines train on the new task; test scores cannot establish an equal-exposure architecture ranking. Community/API new task results remain unmeasured. No paid API calls.']
    (OUT/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
