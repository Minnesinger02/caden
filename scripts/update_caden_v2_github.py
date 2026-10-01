"""Sync reviewed source changes and versioned model documentation; never force-push."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[1]
STAGE=ROOT/'release-staging/caden-v2'
CODE=ROOT/'release-staging/caden/github'

def run(*args):return subprocess.check_output(args,cwd=CODE,text=True,encoding='utf-8')

def prepare():
    record=json.loads((STAGE/'publication-model-completed.json').read_text(encoding='utf-8'))
    results=json.loads((ROOT/'results/caden-multidomain-v2/completed.json').read_text(encoding='utf-8'))
    for folder in ['decision_lab','dynamic_candidates','prefill_renorm_sft','scripts','tests','configs']:
        for source in (ROOT/folder).rglob('*'):
            if not source.is_file() or '__pycache__' in source.parts:continue
            if source.suffix not in {'.py','.md','.json','.toml','.txt','.ps1','.jinja'} and not source.name.upper().startswith(('LICENSE','NOTICE')):continue
            target=CODE/source.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    for name in ['pyproject.toml','uv.lock','LICENSE']:shutil.copyfile(ROOT/name,CODE/name)
    path=CODE/'README.md';text=path.read_text(encoding='utf-8')
    text=text.replace('The encoder release is planned at','The encoder release is published at')
    text=text.replace("folder = snapshot_download('Leonard02/caden-encoder-mixed')",f"folder = snapshot_download('Leonard02/caden-encoder-mixed', revision='{record['revision']}')")
    text=text.replace('Download a fixed Hub commit once available for reproducible use.','The example pins the verified v2 Hub commit for reproducible use.')
    text=text.replace('Each seed\'s `calibration.json` stores a temperature fitted on3396 separate calibration questions;','V2 each seed\'s `calibration.json` stores a temperature fitted on3996 separate calibration questions;')
    text=text.replace('Model repository links are release targets until publication is verified.','The model links are published and verified; source drafts and private research handoffs remain separate.')
    lines=['\n## Caden v2 multi-domain continuation\n',f"Current encoder revision: `{record['revision']}`. Original BANKING+CLINC release remains reproducible at `{record['previous_revision']}`. Qwen adapters remain the original mixed-domain baseline.",
        'Each original seed42/43/44 continues for one epoch on12000 items (3000 each BANKING, CLINC, SNIPS and AG News), AdamW1e-5, FP32, random candidate order, with old-domain replay. This extends training exposure; it is not an equal-training-budget comparison against the older baselines. Independent3996-item calibration is included.',
        '| Task | Test items | Accuracy/% ± sample SD/pp | Variance/pp² |','|---|---:|---:|---:|']
    for task,r in results['tasks'].items():lines.append(f"| {task} | {r['items']} | {r['accuracy_mean_percent']:.4f} ± {r['sample_sd_pp']:.4f} | {r['sample_variance_pp_squared']:.6f} |")
    lines += [f"\nSame200 BANKING development payloads, five serial GPU blocks: p50 {results['timing_p50_ms']:.2f}ms, latency-sum rate {results['timing_latency_rate_per_second']:.1f}/s. This is not a paired ratio confirmation versus earlier blocks or generated-token speed.",
        'BANKING/CLINC remain oracle-eight shortlist tasks; SNIPS and AG News present the full seven/four labels. SNIPS is the pinned DeepPavlov1400-row variant. These are exploratory re-evaluations after prior test reads. Earlier original-release figures above remain historical, rather than being silently reassigned to v2.',
        'Two CPU TF-IDF baselines were added; source analysis scripts require the separately frozen private research artifacts. `scripts/windows/requirements-baselines.txt` records the tested classic-analysis dependencies. The new training driver is `scripts/train_caden_multidomain_v2.py`; it freezes splits, runs three seeds, checks development gates, calibrates and evaluates before release. It expects the original research data and checkpoints, preserving old outputs.',
        'Dataset attribution: [DeepPavlov/snips](https://huggingface.co/datasets/DeepPavlov/snips), revision45f42ebd9641832fd31137317ca5fc5885c86094; [fancyzhx/ag_news](https://huggingface.co/datasets/fancyzhx/ag_news), revisioneb185aade064a813bc0b7f42de02595523103ca4. Model licensing does not relicense training data.']
    path.write_text(text+'\n\n'.join(lines)+'\n',encoding='utf-8')
    patterns=[re.compile(p) for p in [rb'apikey_[A-Za-z0-9_]{50,}',rb'hf_[A-Za-z0-9]{30,}',rb'sk-[A-Za-z0-9_-]{30,}']]
    manifest=[]
    for p in sorted(CODE.rglob('*')):
        if not p.is_file() or any(x in {'.git','__pycache__'} for x in p.relative_to(CODE).parts) or p.suffix=='.pyc':continue
        if any(pat.search(p.read_bytes()) for pat in patterns):raise ValueError('Credential pattern in source release')
        manifest.append({'path':p.relative_to(CODE).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    (STAGE/'code-manifest.json').write_text(json.dumps({'files':manifest,'credential_scan_passed':True},indent=2),encoding='utf-8')

def publish():
    idx=int(os.environ.get('GIT_CONFIG_COUNT','0'));os.environ.update({'GIT_CONFIG_COUNT':str(idx+1),f'GIT_CONFIG_KEY_{idx}':'safe.directory',f'GIT_CONFIG_VALUE_{idx}':CODE.as_posix()})
    gh=json.loads((ROOT/'.tools/github-cli/installed.json').read_text())['executable']
    assert json.loads(run(gh,'api','user'))['login'].lower()=='minnesinger02'
    settings=json.loads(run(gh,'api','repos/Minnesinger02/caden'));assert not settings['private']
    run(gh,'repo','edit','Minnesinger02/caden','--enable-issues=false')
    remote=json.loads(run(gh,'api','repos/Minnesinger02/caden/git/ref/heads/main'))['object']['sha']
    assert run('git','rev-parse','HEAD').strip()==remote,'Remote changed; inspect before committing'
    manifest=json.loads((STAGE/'code-manifest.json').read_text())
    for entry in manifest['files']:assert hashlib.sha256((CODE/entry['path']).read_bytes()).hexdigest()==entry['sha256']
    run('git','add','--all');run('git','commit','-m','Release Caden v2 continuation recipe and updated benchmark documentation')
    run('git','push','origin','main')
    head=run('git','rev-parse','HEAD').strip();remote=json.loads(run(gh,'api','repos/Minnesinger02/caden/git/ref/heads/main'))['object']['sha'];assert head==remote
    tree=json.loads(run(gh,'api',f'repos/Minnesinger02/caden/git/trees/{head}?recursive=1'));assert not tree['truncated']
    remote_files={r['path']:r['sha'] for r in tree['tree'] if r['type']=='blob'};local={}
    for line in run('git','ls-tree','-r','HEAD').splitlines():
        details,path=line.split('\t',1);local[path]=details.split()[2]
    assert remote_files==local and set(local)=={r['path'] for r in manifest['files']}
    settings=json.loads(run(gh,'api','repos/Minnesinger02/caden'));assert not settings['has_issues'] and not settings['private']
    record={'url':settings['html_url'],'git_commit':head,'files_verified':len(local),'has_issues':False,'public':True,'verified':True}
    (STAGE/'publication-github-completed.json').write_text(json.dumps(record,indent=2),encoding='utf-8');print(json.dumps(record,indent=2))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--publish',action='store_true');args=p.parse_args()
    publish() if args.publish else prepare()
