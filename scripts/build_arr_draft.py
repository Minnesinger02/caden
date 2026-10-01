"""Adapt the English master to the pinned official ACL anonymous review style."""
import json
import hashlib
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
out = ROOT/'paper/arr'
out.mkdir(exist_ok=True)
if (out / 'manuscript-evidence.json').exists():
    raise RuntimeError('Compact final mother manuscript exists. Use build_arr_submission_draft.py and build_arr_final_chinese.py; preserve this historical builder without overwriting the final source.')
if not (out/'acl.sty').exists():
    raise FileNotFoundError('Run prepare_arr_template.py first; official style required')
source = (ROOT/'paper/encoder_probability_pilot.tex').read_text(encoding='utf-8')
source = source.replace(r'\usepackage[margin=1in]{geometry}', r'\usepackage[review]{acl}' + '\n' + r'\usepackage{times,latexsym}' + '\n' + r'\usepackage[T1]{fontenc}' + '\n' + r'\usepackage{microtype}')
source = source.replace(r'\title{Probability-Only Candidate Decisions on a 16 GB Windows GPU:\\A Reproducible Encoder--Decoder Pilot}', r'\title{Compact Candidate Encoders for Probability-Only Decisions\\under a Single-GPU Budget}')
source = source.replace(r'\author{Local experimental draft}', r'\author{Anonymous ACL submission}')
source = source.replace(r'\date{September 30, 2026}', r'\date{}')
source = source.replace(r'\begin{table}[ht]', r'\begin{table*}[t]').replace(r'\end{table}', r'\end{table*}')
source = source.replace(r'\begin{center}\begin{tabular}', r'\begin{table*}[t]\centering\small\begin{tabular}')
source = source.replace(r'\end{tabular}\end{center}', r'\end{tabular}\caption{GPU-resident synthetic probability readout; identical token shapes, different model sizes.}\end{table*}')
source = source.replace('Training-scale and transfer experiments remain in progress.', 'With 5000 training items, three-seed development means rise to 91.67\\% for the encoder and 92.17\\% for the tuned decoder. Full-pool and transfer experiments remain in progress.')
related = r'''\section{Related Work and Design Differences}
Sharing a typed probability interface does not imply a shared architecture. Our selected model has a six-layer DistilBERT backbone and a scalar head at the end of each candidate description. The state and question precede the joint candidate list. Laya\citep{laya} uses ModernBERT-large with a separately trained two-layer Transformer decision head, marker scoring, and an act/escalate head; its published training uses RLCD proper-score rewards. Von\citep{von} uses option markers with a checkpoint-specific independent-options attention mask and reset positions, together with an input-conditioned calibration temperature. OpenDecider-nano\citep{opendecider} uses an Ettin encoder and an MLP at option-leading mask markers; its input orders the question and options before the state, and its training uses distributions distilled from open teachers. These systems differ in capacity, representations, cross-option context, objectives, provenance, and calibration. We compare their pinned released implementations as distinct systems.
The community source snapshots and model revisions are recorded in the replication package. We treat existing encoder decision heads as related work; the contribution studied here is the empirical accuracy, robustness, calibration, and cost tradeoff under a constrained GPU budget. Jev's unpublished internals are not inferred from independent replicas.
'''
source = source.replace(r'\section{Task and data}', related + '\n' + r'\section{Task and data}')
source = source.replace(r'\section{Interpretation and remaining work}', r'\section{Discussion}')
start = source.index(r'\section{Replication artifacts}')
end = source.index(r'\begin{thebibliography}', start)
source = source[:start] + r'''\section{Conclusion}
The compact joint encoder exhibits low historical local decision latency; replicated timing remains pending. The full BANKING comparison shows no established accuracy advantage over the tuned decoder. BANKING-only CLINC transfer is substantially worse, while supervised domain expansion raises fresh CLINC accuracy to 97.48\%, below the tuned decoder's 98.41\%. Exploratory original-domain retention intervals include zero. The pointer-head ablation does not improve the three-seed mean, and local policy performance remains near chance. These findings describe a domain-specific accuracy--cost tradeoff among complete systems. Candidate robustness probes, further larger-capacity transfer diagnostics and replicated timing remain in progress; this ARR-formatted draft is not submission ready.
\section*{Limitations}
Our task forces the correct answer into an eight-option shortlist and does not evaluate retrieval or the original 77-way BANKING or 150-way CLINC benchmark, nor CLINC OOS detection. Exploratory scores use a small development set involved in model selection; fresh CLINC confirmation excludes the previously scored probe and reports all frozen arms. Architecture, parameter count, pretraining, training exposure, and runtime differ. Mixed and original local calibration pools also differ. Community upstream exposure is not certified; a public Kev suite has confirmed development-text overlap, and zero overlap with one suite is not proof of absence from all training. Synthetic fixed-shape throughput excludes application overhead, and reference Windows kernels limit the Kev deployment conclusion. An observed training timing discontinuity is preserved and excluded from uninterrupted-training-speed claims. Probability-only output has no generated-token rate. Jev remains unmeasured. Further larger-capacity transfer diagnostics, candidate robustness and replicated timing remain incomplete.
\section*{Ethical Considerations}
This study uses public benchmark utterances and does not collect human-participant data. The experiments do not deploy decisions affecting users. Probabilities may be confidently wrong outside the evaluated setting; the paper measures calibration rather than treating bounded output as factual correctness. Dataset and checkpoint provenance, licenses, and possible overlap must remain documented in the release.
\section*{Reproducibility}
The local replication package records frozen split hashes, checkpoint and code revisions, training telemetry, per-item probabilities, and measured failures. Independent calibration and paired group-resampling scripts are retained. Anonymous supplementary packaging and rendered page-limit verification remain in progress.
''' + source[end:]
authors = {'banking':'Casanueva et al.(2020)', 'distilbert':'Sanh et al.(2019)', 'lora':'Hu et al.(2021)',
           'qwen':'Qwen Team(2025)', 'kev':'Palmer(2026)', 'jev':'TypeSafe AI(2026)', 'clinc':'Larson et al.(2019)'}
for key, author in authors.items():
    source = source.replace(r'\bibitem{'+key+'}', r'\bibitem['+author+r']{'+key+'}')
source = source.replace(r'\begin{thebibliography}{9}', r'\begin{thebibliography}{99}')
source = source.replace(r'\end{thebibliography}', r'''\bibitem[NandhaKishorM(2026)]{laya} NandhaKishorM. Laya source and checkpoint. \url{https://github.com/NandhaKishorM/laya}
\bibitem[Panisa(2026)]{von} Victor Hugo Panisa. Von: An Open-Source System One Decision Model. \url{https://huggingface.co/wfzyx/von}
\bibitem[Janardhan(2026)]{opendecider} Manjunath Janardhan. OpenDecider source and model release. \url{https://github.com/manjunathshiva/opendecider}
\end{thebibliography}''')
(out/'main_en.tex').write_text(source, encoding='utf-8')
translation_current = False
translator = ROOT / 'scripts/build_chinese_translation.py'
if translator.exists() and (out / 'translation-source-lock.json').exists():
    translation_current = subprocess.run([sys.executable, str(translator)], cwd=ROOT).returncode == 0
    if not translation_current:
        print('Chinese source snapshot requires translation review; preserving the last translated version.')
else:
    chinese = (ROOT/'paper/论文草稿.md').read_text(encoding='utf-8')
    (out/'中文对应稿.md').write_text('> 英文母本：main_en.tex。当前为对应研究草稿，最终翻译一致性、结果完整性和ARR版式仍需核验。\n\n'+chinese, encoding='utf-8')
(out/'BUILD_STATUS.json').write_text(json.dumps({'english_master':'main_en.tex','style':'official ACL review',
    'target':'ARR long paper','main_page_limit':8,'anonymous':True,'pdf_compile_verified':False,
    'limitations_present':True,'submission_ready':False,'chinese_matches_current_english':translation_current,
    'english_sha256':hashlib.sha256((out/'main_en.tex').read_bytes()).hexdigest(),
    'reason':'remaining empirical optimization, robustness, replicated timing and page verification; native compiler platform initialization previously unavailable'},indent=2),encoding='utf-8')
print(out/'main_en.tex')


