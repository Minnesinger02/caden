"""Join recorded accuracy, seed variance and timing without rerunning experiments."""
import hashlib
import json
from pathlib import Path
import re
import statistics

ROOT=Path(__file__).resolve().parents[1]

def read(name):return json.loads((ROOT/name).read_text(encoding='utf-8'))

def seed_summary():
    bank=read('results/confirmation-summary.json');fresh=read('results/fresh-clinc-summary.json')
    retention=read('results/mixed-banking-test-retention.json')
    rows=[]
    for family in ['encoder','qwen']:
        original=next(r for r in bank['rows'] if r['model']==family and r['variant']=='raw')
        mixed=next(r for r in retention['rows'] if r['family']==family and r['variant']=='raw')
        for recipe,values in [('BANK',original['accuracy_by_seed']),('mixed',mixed['mixed_accuracy_by_seed'])]:
            arm=f'bankingonly-{family}-raw' if recipe=='BANK' else f'mixed-{family}-raw'
            clinc=next(r for r in fresh['rows'] if r['arm']==arm)
            for task,scores in [('BANK',values),('CLINC',clinc['accuracy_by_seed_or_release'])]:
                pp=[100*x for x in scores]
                rows.append({'system':('Caden' if family=='encoder' else 'Qwen')+' '+recipe,'task':task,
                    'seeds':[42,43,44],'accuracy_percent_by_seed':pp,'mean_percent':statistics.mean(pp),
                    'sample_sd_pp':statistics.stdev(pp),'sample_variance_pp_squared':statistics.variance(pp),'ddof':1})
    report={'rows':rows,'scope':'Training-seed variability on fixed tasks; not item-level confidence intervals or timing-block variance.',
        'sources':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['results/confirmation-summary.json','results/fresh-clinc-summary.json','results/mixed-banking-test-retention.json']}}
    (ROOT/'results/seed-variance-summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    lines=['# Seed accuracy variance (ddof=1, pp²)','', '| System | Task | seed42/% | seed43/% | seed44/% | Mean/% | SD/pp | Variance/pp² |','|---|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:lines.append('| '+r['system']+' | '+r['task']+' | '+' | '.join(f'{v:.6f}' for v in r['accuracy_percent_by_seed']+[r['mean_percent'],r['sample_sd_pp'],r['sample_variance_pp_squared']])+' |')
    (ROOT/'results/seed-variance-summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return report

def enhance(source):
    if 'Latency rate/s' in source.split(r'\end{table*}',1)[0]:return source
    timing=read('results/replicated-task-fp32-8792/summary-mixed-reference.json')
    runtime={r['model']:r for r in timing['rows']}
    mapping={'Caden BANK':'encoder-eager','Qwen BANK':'qwen','Caden mixed':'encoder-mixed-eager','Qwen mixed':'qwen-mixed',
        'Laya':'laya','Nano':'nano','OpenJev':'openjev','Von':'von','Decider 0.8B':'decider08','Decider 2B':'decider2b','Kev':'kev'}
    api=next(r for r in read('results/jev-local-timing-comparison.json')['rows'] if r['system']=='jev-1.13.0')
    match=re.search(r'\\begin\{table\*\}.*?\\end\{table\*\}',source,re.S)
    table=match.group()
    table=table.replace(r'\begin{tabular}{lrrr}',r'\begin{tabular}{lrrrrr}')
    table=table.replace(r'CLINC raw NLL\\\midrule',r'CLINC raw NLL & p50/ms & Latency rate/s\\\midrule')
    lines=[]
    for line in table.splitlines():
        label=line.split(' & ')[0]
        if label in mapping:
            r=runtime[mapping[label]];line=line[:-2]+f" & {statistics.median(r['p50_ms_by_block']):.2f} & {r['median_decisions_per_second']:.1f}"+r'\\'
        elif label=='Jev 1.13 API':line=line[:-2]+f" & {api['p50_ms']:.2f} & {api['latency_rate_per_second']:.2f}"+r'\\'
        lines.append(line)
    table='\n'.join(lines)
    table=table.replace('Community raw includes upstream calibration;',
        'Speed uses the same 200 BANKING development payloads: local GPU rows summarize five FP32/B1 blocks with fixed seed42; these are not CLINC timings. Jev is one transport-inclusive HTTP pass at concurrency4, not matched GPU compute. Latency rate is questions divided by summed latency, not concurrent throughput. Community raw includes upstream calibration;')
    source=source[:match.start()]+table+source[match.end():]
    note=('We report training-seed means and sample standard deviations (ddof=1) in the main table; the sample variance is in squared percentage points (pp$^2$). '
          'Exact seed42/43/44 accuracies and variances are recorded in the replication supplement. Single released community checkpoints and the API pass have no estimated training-seed variance; missing variability is not zero. Timing-block variability and paired source-question intervals are separate quantities. ')
    source=source.replace(r'\paragraph{Calibration and statistics.}',r'\paragraph{Calibration and statistics.} '+note)
    return source

if __name__=='__main__':
    seed_summary();p=ROOT/'paper/arr/main_en.tex';p.write_text(enhance(p.read_text(encoding='utf-8')),encoding='utf-8')
