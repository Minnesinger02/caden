"""Project completed baseline and full-label benchmark evidence into the draft."""
import hashlib
import json
from pathlib import Path
import re
import statistics

ROOT=Path(__file__).resolve().parents[1]

def apply_expansion(source):
    if r'\paragraph{Classic baselines.}' in source:return source
    classic_path=ROOT/'results/benchmark-expansion-v1/classic-main-comparison.json'
    expansion_path=ROOT/'results/benchmark-expansion-v1/summary.json'
    if not classic_path.exists() or not expansion_path.exists():return source
    classic=json.loads(classic_path.read_text(encoding='utf-8'))
    expansion=json.loads(expansion_path.read_text(encoding='utf-8'))
    rows=[]
    for record in classic['rows']:
        linear=record['system']=='tfidf-linear'
        values=[]
        for task in ['banking','clinc']:
            r=record['tasks'][task];value=f"{r['accuracy_mean_percent']:.2f}"
            if r['sample_sd_pp'] is not None:value+=rf" $\pm$ {r['sample_sd_pp']:.2f}"
            values.append(value)
        blocks=classic['linear_timing' if linear else 'description_timing']['blocks']
        rows.append(('TF-IDF linear (CPU)' if linear else 'TF-IDF cosine (CPU)')+' & '+' & '.join(values)+
            f" & {record['tasks']['clinc']['raw_nll_mean']:.4f} & {statistics.median(r['p50_ms'] for r in blocks):.2f} & {statistics.median(r['latency_rate_per_second'] for r in blocks):.1f}"+r'\\')
    match=re.search(r'\\begin\{table\*\}.*?\\end\{table\*\}',source,re.S)
    table=match.group().replace(r'\bottomrule\end{tabular}','\n'.join(rows)+'\n'+r'\bottomrule\end{tabular}')
    table=table.replace('local GPU rows summarize','neural local GPU rows summarize')
    table=table.replace('Community raw includes upstream calibration;',
        'TF-IDF rows use two-thread CPU execution, with five matched development blocks and cached static descriptions for cosine; the linear timing fixes seed42. These do not match GPU compute or precision. Community raw includes upstream calibration;')
    source=source[:match.start()]+table+source[match.end():]
    methods=(r'\paragraph{Classic baselines.} We add word unigram/bigram TF-IDF description cosine scoring and TF-IDF with an SGD linear classifier (log-loss, $\alpha=10^{-4}$, five epochs, seeds42/43/44). '
        'Vocabulary/IDF fit only on training text with min document frequency2, at most30,000 features and sublinear term frequency. The supervised classifier uses fixed training-label logits restricted to requested candidate IDs, whereas Caden scores descriptions dynamically. Both main-table classic baselines fit the same BANKING+CLINC training text as Caden mixed; only the linear model uses labels. Raw scores are softmax-normalized without test-fitted temperatures. This baseline is a competitive fixed-label alternative, not equivalent arbitrary-candidate capability. Its different objective, epochs, capacity, CPU implementation and numerical precision prevent an equal-compute architectural interpretation.\n')
    source=source.replace(r'\paragraph{Calibration and statistics.}',methods+r'\paragraph{Calibration and statistics.}')
    findings=('The mixed-trained fixed-label TF-IDF linear baseline reaches 95.55/97.91\\% on BANKING/CLINC, compared with Caden mixed94.42/97.48\\%. '
        'This strong simple baseline further limits an encoder-accuracy-superiority claim. Its raw NLL and CPU decision cost expose different calibration and implementation tradeoffs; it does not score unseen candidate classes from descriptions. These baselines were added after earlier test reads and are exploratory, with recipes fixed before their own scoring.\n')
    source=source.replace(r'\section{Candidate Context and Robustness}',findings+r'\section{Candidate Context and Robustness}')
    benchmark_rows=[]
    for name,task in expansion['tasks'].items():
        for record in task['rows']:
            system=record['system'];label={'caden-mixed':'Caden mixed','tfidf-description':'TF-IDF cosine','tfidf-linear':'TF-IDF linear'}[system]
            sd=record['sample_sd_pp'];variance=record['sample_variance_pp_squared']
            acc=f"{record['accuracy_mean_percent']:.2f}"+(rf" $\pm$ {sd:.2f}" if sd is not None else '')
            var='--' if variance is None else f'{variance:.6f}'
            run=ROOT/record['runs'][0]['run']
            m=json.loads((run/'metrics.json').read_text(encoding='utf-8'))
            ms=m.get('latency_p50_ms',m.get('latency_p50_ms'))
            benchmark_rows.append(f"{'SNIPS' if name=='snips' else 'AG News'} & {label} & {'transfer' if system=='caden-mixed' else 'IDF only' if system=='tfidf-description' else 'target train'} & {acc} & {var} & {ms:.2f}"+r'\\')
    block=(r'\subsection{New Full-Label Benchmarks}'+'\n'+
        r'We additionally freeze the DeepPavlov SNIPS variant\citep{snipsvariant} (1,400 test rows, 1,395 normalized text groups) and AG News\citep{agnews} (7,600 test rows). All seven/four class descriptions are presented, without gold-dependent distractor selection. We retain official test splits, remove exact normalized train/test overlaps and reserve up to100 development items per class from deduplicated training text. Class descriptions use names only, not the separately supplied LLM-generated SNIPS descriptions. Frozen source revisions, split hashes and overlap records are retained.'+'\n'+
        r'Caden uses the three already trained mixed checkpoints with no target-domain update; the classic linear baselines train on the respective target training split (12,167 SNIPS and119,439 AG News items), while cosine fits only training vocabulary/IDF. These unequal-exposure rows diagnose domain transfer, not an equal-supervision architecture effect. There are zero failures in all new runs. Neither community systems nor the paid Jev API have been evaluated on these new tasks.'+'\n'+
        r'\begin{table*}[t]\centering\small\setlength{\tabcolsep}{4pt}'+'\n'+r'\begin{tabular}{lllrrr}\toprule'+'\n'+
        r'Task & System & Exposure & Accuracy/\% & Variance/pp$^2$ & p50/ms\\\midrule'+'\n'+'\n'.join(benchmark_rows)+'\n'+
        r'\bottomrule\end{tabular}'+'\n'+
        r'\caption{New full-label tasks, distinct from oracle-eight main-table metrics. Means $\pm$ sample SD and sample variance use three seeds (ddof1); deterministic cosine has no seed variance estimate. Speed is a single seed42 pass on each task, or the single deterministic cosine pass; Caden runs on GPU and TF-IDF on CPU. Cosine caches class-description vectors. These timings are descriptive, not the five-block main-table experiment. Transfer and target-supervised exposure differ. SNIPS here is the pinned HF variant, not an asserted original700-question benchmark.}'+'\n'+r'\end{table*}'+'\n')
    source=source.replace(r'\section{Synthetic Probability Throughput}',block+r'\section{Synthetic Probability Throughput}')
    refs=(r'\bibitem[DeepPavlov(2026)]{snipsvariant} DeepPavlov. SNIPS text-classification dataset variant. Revision45f42ebd9641832fd31137317ca5fc5885c86094. \url{https://huggingface.co/datasets/DeepPavlov/snips}'+'\n'+
          r'\bibitem[Zhang et al.(2015)]{agnews} Xiang Zhang, Junbo Zhao, and Yann LeCun. AG News classification data, distributed via fancyzhx/ag\_news. Revisioneb185aade064a813bc0b7f42de02595523103ca4. \url{https://huggingface.co/datasets/fancyzhx/ag_news}'+'\n')
    source=source.replace(r'\end{thebibliography}',refs+r'\end{thebibliography}')
    return source

if __name__=='__main__':
    p=ROOT/'paper/arr/main_en.tex';p.write_text(apply_expansion(p.read_text(encoding='utf-8')),encoding='utf-8')
