"""Add versioned continuation evidence without reassigning old benchmark rows."""
import hashlib
import json
from pathlib import Path
import re
import statistics

ROOT=Path(__file__).resolve().parents[1]

def apply_v2(source):
    completed=ROOT/'results/caden-multidomain-v2/completed.json'
    publication=ROOT/'release-staging/caden-v2/publication-model-completed.json'
    if not completed.exists() or not publication.exists() or r'\paragraph{Caden v2 continuation.}' in source:return source
    result=json.loads(completed.read_text(encoding='utf-8'));pub=json.loads(publication.read_text(encoding='utf-8'))
    tasks=result['tasks']
    def acc(task):r=tasks[task];return rf"{r['accuracy_mean_percent']:.2f} $\pm$ {r['sample_sd_pp']:.2f}"
    row=f"Caden v2 (4 domains) & {acc('banking')} & {acc('clinc')} & {tasks['clinc']['nll_raw_mean']:.4f} & {result['timing_p50_ms']:.2f} & {result['timing_latency_rate_per_second']:.1f}"+r'\\'
    first=re.search(r'\\begin\{table\*\}.*?\\end\{table\*\}',source,re.S);table=first.group()
    table=re.sub(r'(?m)^(Caden mixed & .*?)$',lambda m:m.group(0)+'\n'+row,table)
    table=table.replace('All final runs have zero failures.', 'All final runs have zero failures. Caden v2 adds supervision in SNIPS/AG News with old-domain replay; its later five serial timing blocks are descriptive and not paired with the original randomized blocks.')
    source=source[:first.start()]+table+source[first.end():]
    methods=(r'\paragraph{Caden v2 continuation.} We continue each matching original seed42/43/44 checkpoint for one fixed epoch on12,000 items:3,000 each BANKING,CLINC,SNIPS and AG News. AdamW learning rate is $10^{-5}$, microbatch1, clipping1, FP32/eager with candidate-order augmentation and source replay. Original pilot weights are not reused. All development/test/calibration text groups are excluded from continuation training. Calibration uses3,996 separate items (original3,396 plus300 per new domain). Release gates fixed before runs require mean source-development decline no worse than1pp and improvement in both new development tasks. Final test reruns are exploratory because earlier benchmark scores were viewed; the three seeds are not selected by test scores. Training exposure differs from original community and Qwen baselines.'+'\n')
    source=source.replace(r'\paragraph{Classic baselines.}',methods+r'\paragraph{Classic baselines.}')
    for name,label in [('snips','SNIPS'),('agnews','AG News')]:
        r=tasks[name];run=next(x for x in result['runs'] if x['task']==name and x['seed']==42)
        newline=f"{label} & Caden v2 & 4-domain train & {acc(name)} & {r['sample_variance_pp_squared']:.6f} & {run['latency_p50_ms']:.2f}"+r'\\'
        source=re.sub(r'(?m)^('+label+r' & Caden mixed & .*?)$',lambda m:m.group(0)+'\n'+newline,source)
    note=('The v2 rows add target-domain supervision to the frozen transfer rows: the SNIPS/AG News development improvement motivated this development-gated continuation, while test scores are reported for all three seeds. '
        'Target-training exposure remains unequal: the linear baseline uses12,167/119,439 target examples, while v2 continuation uses3,000 per new domain plus replay. The original transfer and v2 supervised rows must not be conflated.\n')
    source=source.replace('Neither community systems nor the paid Jev API have been evaluated on these new tasks.', 'Neither community systems nor the paid Jev API have been evaluated on these new tasks. '+note)
    release=(r'\paragraph{Updated release.} The current Caden-Encoder-Mixed root contains v2 weights at \url{https://huggingface.co/Leonard02/caden-encoder-mixed} (revision \texttt{'+pub['revision']+r'}). The original BANKING+CLINC-only revision above remains recoverable; each version uses its own calibration. The Qwen adapter release is unchanged. GitHub source and publication audit records identify the continuation recipe and full test/seed/timing results.'+'\n')
    github_record=ROOT/'release-staging/caden-v2/publication-github-completed.json'
    if github_record.exists():
        gh=json.loads(github_record.read_text(encoding='utf-8'))
        release += r'The updated public source is pinned to GitHub commit \texttt{'+gh['git_commit']+r'}; the original source revision remains historical.'+'\n'
    source=source.replace(r'\begin{thebibliography}{99}',release+r'\begin{thebibliography}{99}')
    return source

def main():
    paper=ROOT/'paper/arr';old=(paper/'main_en.tex').read_bytes();digest=hashlib.sha256(old).hexdigest()
    archive=paper/'archive'/('before-caden-v2-'+digest[:16]);archive.mkdir(parents=True,exist_ok=True)
    for name in ['main_en.tex','中文对应稿.md','manuscript-evidence.json','BUILD_STATUS.json','translation-audit.json','translation-status.json','translation-source-lock.json']:
        import shutil
        if not (archive/name).exists():shutil.copyfile(paper/name,archive/name)
    source=apply_v2(old.decode('utf-8'));assert source!=old.decode('utf-8'),'Completed release required'
    (paper/'main_en.tex').write_text(source,encoding='utf-8');newhash=hashlib.sha256((paper/'main_en.tex').read_bytes()).hexdigest()
    for name,key in [('BUILD_STATUS.json','english_sha256'),('manuscript-evidence.json','english_master_sha256')]:
        path=paper/name;r=json.loads(path.read_text(encoding='utf-8'));r[key]=newhash
        if name=='manuscript-evidence.json':
            for key,f in [('caden_v2','results/caden-multidomain-v2/completed.json'),('caden_v2_protocol','data/caden-multidomain-v2/protocol.json')]:
                r['source_files'][key]={'path':f,'sha256':hashlib.sha256((ROOT/f).read_bytes()).hexdigest()}
        path.write_text(json.dumps(r,indent=2),encoding='utf-8')
    print(newhash)

if __name__=='__main__':main()
