"""Build a standalone, evidence-linked LaTeX pilot draft from saved metrics."""
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
paper = ROOT / 'paper'
paper.mkdir(exist_ok=True)
def metrics(run):
    return json.loads((ROOT / 'results' / run / 'metrics.json').read_text(encoding='utf-8'))

groups = [('DistilBERT scalar, 3 epochs', [f'encoder16-s{s}-e3-dev' for s in [42,43,44]]),
          ('DistilBERT pointer, higher head LR', [f'encoder16-s{s}-pointer-e3-hlr-dev' for s in [42,43,44]]),
          ('Qwen3 LoRA candidate CE, 1 epoch', [f'sft-eval-s{s}-dev' for s in [42,43,44]]),
          ('Qwen3 frozen base', ['decoder-base-s42-dev']),
          ('Kev official release (local)', ['kev16-local-dev'])]
for label, run in [('Laya official SDK, FP32', 'laya-fp32-dev'),
                   ('OpenDecider Nano official, FP32', 'opendecider-nano-fp32-dev'),
                   ('OpenJev official, Qwen3 FP32', 'openjev-qwen-fp32-normalized-dev'),
                   ('Von official SDK, FP32', 'von-fp32-normalized-dev'),
                   ('Decider 0.8B official, FP32', 'decider08-fp32-normalized-dev')]:
    if (ROOT / 'results' / run / 'metrics.json').exists():
        groups.append((label, [run]))
for label, template in [('DistilBERT scalar, 5000 training items', 'encoder16-s{seed}-e3-5k-dev'),
                        ('Qwen3 LoRA, 5000 training items', 'sft-eval-s{seed}-train5k-dev'),
                        ('DistilBERT scalar, 8792 training items', 'encoder16-s{seed}-e3-8792-dev'),
                        ('Qwen3 LoRA, 8792 training items', 'sft-eval-s{seed}-train8792-dev')]:
    runs = [template.format(seed=seed) for seed in [42,43,44]
            if (ROOT/'results'/template.format(seed=seed)/'metrics.json').exists()]
    if runs:
        groups.append((label, runs))
table = []
evidence = {}
for label, runs in groups:
    ms = [metrics(run) for run in runs]
    acc = [m['accuracy_all_failures_wrong'] * 100 for m in ms]
    spread = f"{statistics.stdev(acc):.2f}" if len(acc)>1 else '--'
    table.append(f"{label} & {len(runs)} & {statistics.mean(acc):.2f} & {spread} & {statistics.mean(m['nll_clipped_1e-15'] for m in ms):.3f} & {statistics.mean(m['latency_p50_ms'] for m in ms):.2f} " + r'\\')
    evidence[label] = runs
source = r'''\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath,booktabs,hyperref}
\hypersetup{colorlinks=true,urlcolor=blue}
\title{Probability-Only Candidate Decisions on a 16 GB Windows GPU:\\A Reproducible Encoder--Decoder Pilot}
\author{Local experimental draft}
\date{September 30, 2026}
\begin{document}
\maketitle
\begin{abstract}
We compare a joint bidirectional encoder with a candidate-restricted causal decoder and the public Kev probability-only decision model on one laptop GPU. This draft reports development experiments, not a final test-set result. In an eight-option BANKING77 task with the correct option included by construction, three runs of the scalar DistilBERT method obtain 81.17\% mean accuracy; Qwen3-0.6B with candidate cross-entropy LoRA obtains 86.67\%. The public Kev release obtains 95.5\%, but 16 of 200 development texts overlap one public Kev training suite. The encoder is substantially faster in the measured local implementation. A query--candidate pointer head does not improve mean accuracy and worsens negative log likelihood. Jev has not been measured. Training-scale and transfer experiments remain in progress.
\end{abstract}
\section{Question and scope}
We ask whether a compact encoder can select from a variable set of described candidates while returning a normalized probability distribution, and what accuracy--latency tradeoff it offers on a single 16 GB GPU. This is a comparison of complete systems. Attention direction, model scale, pretraining, tuning data, readout, and runtime implementation are not independently controlled; the experiment cannot establish a causal advantage of one architecture.
\section{Task and data}
BANKING77\cite{banking} contains 77 banking intents. We pin the Parquet mirror revision in the replication artifacts. The initial experiment uses 1000 training items and disjoint 200-item development, calibration, and confirmation sets. Development is sampled from the official training split; confirmation uses the official test split. Normalized text groups are disjoint within our experiment.
Each item has eight candidate keys and descriptions. A deterministic shortlist includes the gold option plus seven randomly selected distractors. This is an oracle-shortlist decision task; the results are not 77-class BANKING77 benchmark scores and do not evaluate retrieval. Candidate ordering is randomized during encoder training. No input is silently truncated.
An audit of the pinned public Kev v7/decision-v7 training suite finds overlaps of 92/1000 training, 16/200 development, 27/200 calibration, and 0/200 confirmation texts. Matching uses normalized exact text or upstream origin hashes. This audit covers one suite only and cannot certify the entire release's training provenance. Development comparisons with Kev consequently have a known training-data confound.
\section{Methods}
\paragraph{Joint encoder.} DistilBERT\cite{distilbert} receives the state, question, and all candidate descriptions in one sequence. Each candidate ends in a dedicated marker. Its final hidden representation $h_i$ is scored by a shared scalar head:
\[
z_i=w^\top h_i+b,\qquad p_i=\frac{\exp z_i}{\sum_j\exp z_j}.
\]
The head is shared across candidates and is not a fixed 77-class classifier. The whole encoder is tuned by gold candidate cross entropy with AdamW, learning rate $2\times10^{-5}$, microbatch one, gradient clipping one, and three epochs. A one-epoch pilot reached 78\% on development; epoch selection used development only.
\paragraph{Pointer ablation.} A jointly contextualized CLS query and candidate marker are layer normalized and separately projected to 128 dimensions. Scores are $q^\top k_i/\sqrt{128}$. We test head learning rates $2\times10^{-5}$ and $2\times10^{-4}$ with the same backbone rate and data. The higher-rate arm receives three seeds after a single-seed improvement of one item. All observed runs, including the negative result, are retained.
\paragraph{Causal decoder.} The Qwen3-0.6B\cite{qwen} baseline scores the permitted first-response candidate tokens and renormalizes within that set. Candidate cross entropy tunes query/value LoRA\cite{lora} adapters of rank eight for one epoch, with microbatch one and accumulation eight. The run artifacts contain the exact tokenizer, prompt, optimizer settings, revisions, and adapter metadata. This method performs one decision readout rather than a generated explanation.
\paragraph{Kev and Jev.} Kev\cite{kev} uses the official pinned checkpoint loader, including its pointer head and learned temperature. The 0.8B release uses Qwen3.5-0.8B-Base and rank-16 adapters; it has different training data from the local methods. Direct local inference uses FP32, unmerged adapters, eager attention, and no fused kernels or CUDA graphs. Jev's choice API\cite{jev} is a planned external baseline; no available version, credentials, or request budget were supplied, so no Jev numbers are inferred.
\section{Experimental protocol}
The machine has an RTX 4090 Laptop GPU with 16376 MiB board memory, an i9-13980HX CPU, and approximately 31.6 GiB RAM. Local experiments use FP32 and PyTorch 2.8.0 with CUDA 12.8. The main environment uses Transformers 4.57.6; Kev uses a separate environment with Transformers 5.17.0. One GPU model runs at a time. Training seeds are 42, 43, and 44. Model and data revisions are pinned, and split files are hashed.
Accuracy counts failures as incorrect. Negative log likelihood uses valid responses with probability clipping at $10^{-15}$; Brier score and ten-bin ECE are also retained. Means and sample standard deviations describe the three observed seeds. Development selection means these are descriptive results, not unbiased confirmation estimates. Paired material-group bootstrap is reserved for locked confirmation comparisons.
Local single-item timing includes tokenization, transfers, model execution, and probability readback, after warmup and CUDA synchronization; loading and downloads are excluded. Kev includes request validation and uses cold per-item state caches. These are implementation timings, not equal-token FLOP comparisons. HTTP latency is reported separately. Peak allocator memory and sampled board memory are different quantities; board measurements include desktop usage.
\section{Development results}
\begin{table}[ht]
\centering\small
\begin{tabular}{p{6cm}rrrrr}
\toprule
System & Runs & Acc. (\%) & SD (pp) & NLL & p50 (ms)\\
\midrule
TABLE_ROWS
\bottomrule
\end{tabular}
\caption{Same 200 development items. p50 column averages each run's median when multiple seeds exist. All listed runs have zero failures. Single-release and frozen-base rows have no training-seed SD.}
\end{table}
The scalar encoder averages 81.17\% and the higher-rate pointer averages 80.83\%. The latter's worse NLL rules out a claim that the new head is an improvement. The locally trained decoder averages 86.67\%. Kev's stronger development accuracy is consistent with a stronger system but cannot isolate architectural effects, given confirmed development overlap and different training and scale.
Kev direct inference has median 112.16 ms and p95 118.22 ms. Local HTTP median is 128.19 ms. The two interfaces agree on every argmax; normalized probabilities differ by at most 0.000185. Runtime logs identify reference PyTorch fallbacks for causal convolution and gated delta rules. Kev timings therefore describe this Windows setup and are not its best achievable deployment latency.
\section{Engineering readout experiment}
With identical Qwen3 weights, an optimization of the vendored LitJev readout projects only candidate vocabulary rows at the required suffix position. Paired randomized GPU timing on synthetic 64-character states with one and two questions gives speedups of 1.033 and 1.115. Maximum probability discrepancy is about $1.6\times10^{-6}$ with full argmax agreement. This is a numerical and performance experiment on synthetic prompts, not a task accuracy comparison or a Jev API measurement. Larger workloads and BF16 remain to be tested.
\section{Interpretation and remaining work}
The compact encoder offers low measured local latency and modest memory use. Its full BANKING accuracy is close to the tuned decoder, with a paired interval that does not establish an accuracy advantage, while its BANKING-only CLINC transfer is substantially worse. The systems do not represent all bidirectional encoders or probability-only decoders; the results describe deployment tradeoffs rather than a universal ranking.
The completed nested 1000-, 5000- and 8792-item scale arms preserve development file bytes and show how accuracy changes with added gold data. Both local methods use the same examples within an arm, but three encoder epochs and one decoder epoch do not equalize training exposure. The completed CLINC150\cite{clinc} probe excludes OOS detection and cannot certify absence from community pretraining or model selection. The registered supervised mixed-domain stage is now training; its fresh CLINC test remains unscored.
Full BANKING confirmation, independent calibration and frozen transfer diagnostics are complete. Candidate-order and missing-option GPU checks, larger-capacity community confirmation, supervised mixed-domain evaluation and randomized timing replications remain unfinished. Jev requires access to a fixed version and a request budget. The paper remains a working draft pending these experiments and rendered page verification.
\section{Replication artifacts}
All paths are relative to \texttt{E:/jev}. See \texttt{handoff/RESEARCH\_PROTOCOL.md}, \texttt{handoff/PROGRESS.md}, \texttt{handoff/kev-overlap-audit.json}, \texttt{configs/kev-resolved.json}, and the per-run \texttt{metadata.json}, \texttt{metrics.json}, and \texttt{predictions.jsonl}. \texttt{paper/evidence.json} maps table rows to run folders. Original handoff manifests and archives are retained; source modifications intentionally do not match the original package checksum list.
\begin{thebibliography}{9}
\bibitem{banking} Casanueva et al. Efficient Intent Detection with Dual Sentence Encoders. 2020. \url{https://arxiv.org/abs/2003.04807}
\bibitem{distilbert} Sanh et al. DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. 2019. \url{https://arxiv.org/abs/1910.01108}
\bibitem{lora} Hu et al. LoRA: Low-Rank Adaptation of Large Language Models. 2021. \url{https://arxiv.org/abs/2106.09685}
\bibitem{qwen} Qwen Team. Qwen3 Technical Report. 2025. \url{https://arxiv.org/abs/2505.09388}
\bibitem{kev} Palmer. Kev source and model release. Pinned local snapshots, retrieved September 30, 2026. \url{https://github.com/jaredpalmer/kev} and \url{https://huggingface.co/jaredpalmer/kev-0.8b}
\bibitem{jev} TypeSafe AI. Choice primitive documentation. Retrieved September 30, 2026. \url{https://docs.typesafe.ai/primitives/choice}
\bibitem{clinc} Larson et al. An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction. EMNLP-IJCNLP 2019. \url{https://aclanthology.org/D19-1131/}
\end{thebibliography}
\end{document}
'''
fixed_encoder = ROOT / 'results/fixed-compute-encoder-fp32.json'
fixed_qwen = ROOT / 'results/fixed-compute-qwen-fp32.json'
throughput_section = ''
if fixed_encoder.exists() and fixed_qwen.exists():
    a = json.loads(fixed_encoder.read_text(encoding='utf-8'))['rows']
    b = json.loads(fixed_qwen.read_text(encoding='utf-8'))['rows']
    comparisons = []
    for row in a:
        other = next(r for r in b if r['task'] == row['task'] and r['batch'] == row['batch']
                     and r['input_tokens_per_item'] == row['input_tokens_per_item'])
        comparisons.append(f"{row['batch']} & {row['input_tokens_per_item']} & {row['decisions_per_second']:.1f} & {other['decisions_per_second']:.1f} & {row['decisions_per_second']/other['decisions_per_second']:.2f} " + r'\\')
    throughput_section = r'''\section{Fixed-shape throughput and token semantics}
Probability-only output does not generate autoregressive tokens. We report decisions per second and processed input tokens per second. GPU-resident synthetic nonpadding inputs use identical batch sizes and lengths, FP32, eager attention, eight candidate probabilities, five warmups, and twenty repeated forwards. These measurements exclude tokenization, transfers, and readback; they are separate from the real-item service timings. Encoder and decoder parameter counts remain different.
\begin{center}\begin{tabular}{rrrrr}\toprule
Batch & Input tokens & Encoder decisions/s & Qwen decisions/s & Ratio\\\midrule
ROWS
\bottomrule\end{tabular}\end{center}
The frozen Qwen base also performs genuine greedy generation with a KV cache, three repeats of exactly 64 new tokens. With batch one and 128 input tokens, its measured throughput is 44.18 generated tokens/s, including prefill and decode. At batch eight it is 329.68 generated tokens/s across the batch. Encoder generated-token throughput is undefined. A ratio of encoder decision throughput to decoder generated-token throughput would mix different tasks and is not reported.
'''.replace('ROWS', '\n'.join(comparisons))
    source = source.replace(r'\section{Engineering readout experiment}', throughput_section + '\n' + r'\section{Engineering readout experiment}')
ablation_file = ROOT/'results/encoder-speed-ablation-dev.json'
ablation = json.loads(ablation_file.read_text(encoding='utf-8')) if ablation_file.exists() else None
if ablation:
    ablation_rows = '\n'.join(f"{r['name']} & {100*r['accuracy']:.1f} & {r['nll']:.3f} & {r['latency_p50_ms']:.2f} & {r['argmax_disagreement_count']} " + r'\\' for r in ablation['rows'])
    section = r'''\section{Attention Implementation and Precision Ablation}
The same scalar encoder weights trained on 8792 items are evaluated under eager or native SDPA attention, in FP32 or BF16. Training and readout are unchanged. All four conditions use the same 200 development items with zero failures.
\begin{table}[ht]\centering\small\begin{tabular}{lrrrr}\toprule
Implementation & Acc. (\%) & NLL & p50 (ms) & Flips\\\midrule
ABLATION_ROWS
\bottomrule\end{tabular}\caption{Flips relative to FP32 eager on identical items. Single-run implementation timings; randomized timing replications remain pending.}\end{table}
FP32 SDPA keeps all argmax decisions unchanged, with maximum probability difference $8.73\times10^{-6}$. BF16 eager and SDPA each change one prediction; maximum probability differences are 0.2125 and 0.0854. The faster BF16 execution is therefore not probability-preserving. FP32 SDPA is the current candidate for acceleration without observed development prediction drift. These single-run timings require randomized replication before robust deployment speed conclusions.
'''.replace('ABLATION_ROWS',ablation_rows)
    source = source.replace(r'\section{Interpretation and remaining work}',section+'\n'+r'\section{Interpretation and remaining work}')
confirmation_file = ROOT / 'results/confirmation-summary.json'
confirmation = json.loads(confirmation_file.read_text(encoding='utf-8')) if confirmation_file.exists() else None
if confirmation and confirmation['rows']:
    confirmation_rows = '\n'.join(f"{r['model']} {r['variant']} & 3 & {100*r['accuracy_mean']:.2f} & {100*r['accuracy_sample_sd']:.2f} & {r['nll_mean']:.3f} & {r['ece_mean']:.3f} " + r'\\' for r in confirmation['rows'])
    section = r'''\section{Full Held-Out Confirmation and Calibration}
Before reading test scores, we froze 1000 independent calibration items and 3079 deduplicated official test items. The earlier 200-item pools are unchanged subsets. The 8792-item training pool, development, calibration and test groups are disjoint locally. All three training seeds are retained; test scores do not choose the model, prompt, or seed. This remains an oracle eight-option task and does not establish that community models have never seen these public data.
For each checkpoint, one positive temperature is fit only on calibration, then applied unchanged to the test. Raw encoder and decoder scores are retained to avoid probability-underflow information loss. The calibration preserves argmax decisions. The table includes only complete three-seed arms; missing arms remain pending.
\begin{table}[ht]\centering\small\begin{tabular}{lrrrrr}\toprule
Model and variant & Seeds & Acc. (\%) & SD & NLL & ECE\\\midrule
CONFIRMATION_ROWS
\bottomrule\end{tabular}\caption{Frozen 3079-item test. Sample SD is across observed training seeds, not a confidence interval.}\end{table}
Public SDKs may round probabilities: Laya and OpenJev preserve four-decimal wire values. We retain those responses and normalize only sum errors bounded by the stated rounding precision. NLL from those outputs reflects rounding. An initial strict-sum adapter rejection is retained as an integration diagnostic and excluded from the valid model comparison. Runtime precision and attention implementations are recorded separately; timings and model differences do not isolate an architectural cause.
'''.replace('CONFIRMATION_ROWS', confirmation_rows)
    source = source.replace(r'\section{Interpretation and remaining work}', section + '\n' + r'\section{Interpretation and remaining work}')
    source = source.replace('This draft reports development experiments, not a final test-set result.', 'This research draft reports development experiments and initial frozen full-test confirmation; expanded comparisons remain in progress.')
    evidence['full_confirmation'] = confirmation
    paired_file = ROOT / 'results/encoder-qwen8792-test-paired.json'
    if paired_file.exists():
        paired = json.loads(paired_file.read_text(encoding='utf-8'))
        if paired['dataset_sha256'] != confirmation['test_sha256']:
            raise ValueError('Paired interval uses a different test')
        delta = 100 * paired['accuracy_delta']
        lo, hi = [100*x for x in paired['delta_ci95']]
        statement = f"The encoder-minus-Qwen accuracy difference is {delta:.3f} percentage points; the 95\\% paired material-group bootstrap interval is [{lo:.3f}, {hi:.3f}]. It is conditional on the three observed training seeds. The interval crosses zero and does not establish an encoder accuracy advantage or a predeclared equivalence claim."
        source = source.replace(r'\section{Interpretation and remaining work}', statement + '\n' + r'\section{Interpretation and remaining work}')
        evidence['full_confirmation_paired'] = paired
        by_arm = {(r['model'], r['variant']): r for r in confirmation['rows']}
        encoder = by_arm[('encoder', 'calibrated')]
        qwen = by_arm[('qwen', 'calibrated')]
        abstract = f"We study probability-only selection from described candidates using a compact joint encoder, a candidate-restricted causal decoder, and released community decision models on a single 16 GB laptop GPU. After tuning on 8792 local training items, three encoder seeds average {100*encoder['accuracy_mean']:.2f}\\% accuracy on 3079 frozen test items; Qwen3-0.6B LoRA averages {100*qwen['accuracy_mean']:.2f}\\%. The task is an oracle eight-option BANKING77 shortlist, not the original 77-way benchmark. A paired accuracy-difference interval crosses zero, so these data do not establish encoder accuracy superiority. Temperature scaling fitted on 1000 separate calibration items preserves accuracy and gives test NLLs of {encoder['nll_mean']:.3f} and {qwen['nll_mean']:.3f}. Development data-scale and readout ablations show that more gold data improves accuracy, whereas a pointer head does not improve the three-seed mean. The encoder has lower measured local latency, but parameter counts, tuning exposure and runtime differ. We separate decisions per second from genuine autoregressive token generation and disclose public-data overlap. Six community releases and frozen CLINC and policy diagnostics have been evaluated. BANKING-only encoder training transfers worse than Qwen, motivating supervised domain expansion. Robustness and replicated timing remain in progress; Jev is unmeasured."
        start, end = source.index(r'\begin{abstract}'), source.index(r'\end{abstract}')
        source = source[:start] + r'\begin{abstract}' + '\n' + abstract + '\n' + source[end:]
    if confirmation.get('community_rows'):
        community = {(r['model'], r['variant']): r for r in confirmation['community_rows']}
        rows = []
        for name in ['laya','nano','openjev','von','decider08','kev']:
            if (name, 'raw') not in community or (name, 'calibrated') not in community:
                continue
            raw, calibrated = community[(name, 'raw')], community[(name, 'calibrated')]
            if raw['accuracy'] != calibrated['accuracy']:
                raise ValueError('Community calibration changed accuracy')
            rows.append(f"{name} & {100*raw['accuracy']:.2f} & {raw['nll']:.3f} & {calibrated['nll']:.3f} & {calibrated['ece']:.3f} " + r'\\')
        if rows:
            section = r'''\section{Released Community Models on the Full Test}
The fixed released checkpoints use the same 3079 test questions and candidate sets, through their official loaders. They are not locally tuned to our BANKING training pool. One released checkpoint per model is measured; these are not multi-seed retrainings. Original checkpoint calibration is preserved. The additional temperature uses the same independent 1000-item calibration pool for every system, but community interfaces expose probabilities, sometimes rounded, whereas our local checkpoints expose raw scores. This difference limits calibration comparisons.
\begin{table}[ht]\centering\small\begin{tabular}{lrrrr}\toprule
Model & Acc. (\%) & Raw NLL & Postcal NLL & Postcal ECE\\\midrule
COMMUNITY_ROWS
\bottomrule\end{tabular}\caption{Released models, one pinned checkpoint each. Calibration does not change argmax. Different upstream training, sizes and runtime implementations remain confounds.}\end{table}
For the audited public Kev training suite, 118 of 1000 calibration texts overlap, and zero of 3079 test texts overlap. This covers one suite, not all training or model selection. The four-decimal Laya, OpenJev, Von and Decider wire outputs have zero argmax disagreement with their official reported choices on these test items. The pinned Von calibration identifies its weights as von-1.2.0 while the newer SDK labels responses 1.3.0; we identify this experiment by the exact Hugging Face revision rather than the SDK display label.
These released-model comparisons do not show a general encoder accuracy advantage. The compact local encoder is trained for the banking domain; transfer, rule following and context ablations are needed to determine its practical boundary. Kev and Decider's reference Windows recurrent kernels also constrain conclusions about their optimized deployment speed.
'''.replace('COMMUNITY_ROWS', '\n'.join(rows))
            source = source.replace(r'\section{Interpretation and remaining work}', section + '\n' + r'\section{Interpretation and remaining work}')
context_path = ROOT / 'results/context-ablation-summary.json'
if context_path.exists():
    context = json.loads(context_path.read_text(encoding='utf-8'))
    context_rows = []
    for r in context['rows']:
        w = r['workload_means']
        context_rows.append(f"{r['mode']} & {100*r['accuracy_mean']:.2f} & {r['nll_mean']:.3f} & {r['latency_p50_mean_ms']:.2f} & {w['valid_tokens']:.1f} & {w['padded_tokens']:.1f} " + r'\\')
    section = r'''\section{Candidate-Context Ablation}
The same 5000 training items, three seeds, optimizer and three epochs compare joint encoding against independent candidate sequences. Independent mode repeats state and question for each of the eight options, batches the sequences, scores their end markers with the same scalar head and regroups the scores for question-level cross entropy. It is not Von's packed attention-mask implementation and does not share premise computation. A CPU structural test verifies that permuting or adding candidates leaves surviving raw logits unchanged; normalized probabilities need not be unchanged when support changes. GPU candidate-order probes remain pending.
\begin{table}[ht]\centering\small\begin{tabular}{lrrrrr}\toprule
Context & Acc. (\%) & NLL & p50 (ms) & Valid tokens & Padded tokens\\\midrule
CONTEXT_ROWS
\bottomrule\end{tabular}\caption{Exploratory context ablation on 200 development questions. Token counts are per question across actual encoder rows; p50 averages historical run medians, without randomized timing replication.}\end{table}
Independent accuracy averages 93.67\%, versus 91.67\% for joint encoding. The development paired difference is 2.00 percentage points (95\% interval $[0.17,4.00]$, conditional on these seeds), which is exploratory rather than test confirmation. Independent mode processes eight encoder rows rather than one, increasing total nonpadding tokens from 129.6 to 345.4 and padded tokens to 398.1; mean run p50 rises from 3.11 to 4.49 ms. Its apparent development benefit therefore has a measurable compute and latency cost. Shape-based attention-cell counts are recorded as proxies, not measured FLOPs.
'''.replace('CONTEXT_ROWS', '\n'.join(context_rows))
    source = source.replace(r'\section{Interpretation and remaining work}', section + '\n' + r'\section{Interpretation and remaining work}')
transfer_path = ROOT / 'results/transfer-summary.json'
if transfer_path.exists():
    transfer = json.loads(transfer_path.read_text(encoding='utf-8'))
    rows = []
    for name in ['encoder', 'qwen', 'laya', 'nano', 'openjev', 'von', 'decider08', 'kev']:
        c = next(r for r in transfer['tasks']['clinc']['rows'] if r['model'] == name and r['variant'] == 'raw')
        p = next(r for r in transfer['tasks']['policy']['rows'] if r['model'] == name and r['variant'] == 'raw')
        rows.append(f"{name} & {100*c['accuracy_mean']:.2f} & {100*p['accuracy_mean']:.2f} & {100*p['families']['direct']['accuracy_mean']:.2f} & {100*p['families']['two_hop']['accuracy_mean']:.2f} " + r'\\')
    section = r'''\section{Transfer and Policy Diagnostics}
A frozen 300-item in-scope CLINC150 oracle-eight shortlist is evaluated without local CLINC tuning. A separate 320-case synthetic policy diagnostic uses eight exhaustive flag routes, requiring numerical threshold comparisons or two Boolean dependencies. Gold flags and routes are mechanically verified. Uniform-choice accuracy is 12.5\%; this diagnostic is not a representative real-world reasoning benchmark. No target examples are used for training or calibration, and all evaluations have zero failures.
\begin{table}[ht]\centering\small\begin{tabular}{lrrrr}\toprule
Model & CLINC & Policy & Direct & Two-hop\\\midrule
TRANSFER_ROWS
\bottomrule\end{tabular}\caption{Transfer accuracy (\%). Local rows average three seeds; community rows use one pinned release with uncertified upstream exposure.}\end{table}
The BANKING-only encoder obtains 80.67\% on CLINC versus 93.33\% for Qwen; the paired difference is $-12.67$ percentage points (95\% group-bootstrap interval $[-16.44,-9.11]$, conditional on the three seeds). Normalized exact CLINC texts are disjoint from our full BANKING source pools, but this does not certify absence from community pretraining or model selection. Encoder and Qwen policy accuracies are 10.00\% and 12.29\%, respectively, providing no evidence of reliable transferable rule following. Kev and Decider-0.8B reach 30.31\% and 32.50\%, still far below dependable correctness.
BANKING-fitted temperatures are applied without target refitting. They preserve choices and reduce encoder CLINC NLL from 1.203 to 0.660 and Qwen NLL from 0.281 to 0.232, but do not recover task competence; community NLL sometimes worsens under this calibration shift. Complete transferred ECE and policy-family traces are retained.
These observations motivate a separately registered supervised BANKING plus CLINC expansion with 23,789 training items. Its 4,197 fresh CLINC test items exclude all 300 previously evaluated probe texts. This stage is not zero-shot transfer, and further BANKING test results are exploratory because those test items have already been scored.
'''.replace('TRANSFER_ROWS', '\n'.join(rows))
    source = source.replace(r'\section{Interpretation and remaining work}', section + '\n' + r'\section{Interpretation and remaining work}')
fresh_path = ROOT / 'results/fresh-clinc-summary.json'
if fresh_path.exists():
    fresh = json.loads(fresh_path.read_text(encoding='utf-8'))
    if fresh['pending'] or fresh['items'] != 4197:
        raise ValueError('Fresh CLINC paper section requires all complete local and community arms')
    fresh_table = []
    for row in fresh['rows']:
        spread = f"{100*row['accuracy_seed_sd']:.2f}" if row['accuracy_seed_sd'] is not None else '--'
        fresh_table.append(f"{row['arm'].replace('_', r'\_')} & {100*row['accuracy_mean']:.2f} & {spread} & {row['nll_mean']:.4f} & {row['ece_mean']:.4f} " + r'\\')
    section = r'''\section{Supervised Domain Expansion and Fresh CLINC Confirmation}
We freeze 23,789 mixed training items (8,792 BANKING and 14,997 deduplicated CLINC), 800 development items, and 3,396 calibration items before scoring the new CLINC confirmation. The 4,197 test questions exclude all 300 previously scored CLINC probe texts. The original BANKING-only and mixed checkpoints use seeds 42, 43, and 44 and identical target questions and candidate payloads. This is supervised domain expansion, not zero-shot transfer. The task remains an oracle shortlist of eight among 150 in-scope intents, excluding OOS detection.
The encoder's accuracy increases from 79.65\% to 97.48\%, and Qwen's from 93.69\% to 98.41\%. Paired source-question bootstrap gains are +17.830 percentage points (95\% interval [16.782, 18.855]) and +4.718 [4.154, 5.321], respectively. The mixed encoder minus mixed Qwen difference is -0.929 points [-1.350, -0.516]. These intervals are conditional on three observed seeds, not population-over-seeds or causal architecture estimates. Parameter count, pretraining and the three-versus-one training epochs differ.
\begin{table}[ht]\centering\small
\begin{tabular}{lrrrr}\toprule
Arm & Accuracy/\% & Seed SD/pp & NLL & ECE\\\midrule
FRESH_ROWS
\bottomrule\end{tabular}
\caption{Identical 4,197 fresh CLINC oracle-eight questions. Local rows average three seeds; community rows each use one fixed release. All runs have zero failures. Raw community probabilities retain the official checkpoint calibration; no additional CLINC temperature is fitted for these releases.}
\end{table}
Mixed local models use the same held-out pooled calibration set: encoder NLL decreases from 0.1904 to 0.1033 and Qwen from 0.0648 to 0.0577, without changing choices. BANKING-only temperatures use 1,000 BANKING calibration items; this source difference prevents attributing cross-stage calibration changes solely to training or architecture. Community upstream training and model-selection exposure is not certified. Decider-2B achieves 99.33\% on this task, Kev 88.90\%, and Von 90.40\%; these results are system comparisons on this candidate construction, not a universal capability ranking.
\begin{figure*}[t]\centering
\includegraphics[width=\textwidth]{fresh_clinc_expansion.pdf}
\caption{Audited local domain expansion. Left: three-seed mean and sample SD. Middle: paired source-question intervals conditional on these seeds. Right: mixed-model calibration using the same held-out pool.}
\end{figure*}
'''.replace('FRESH_ROWS', '\n'.join(fresh_table))
    retention_path = ROOT / 'results/mixed-banking-test-retention.json'
    if retention_path.exists():
        retention = json.loads(retention_path.read_text(encoding='utf-8'))
        retention_lines = []
        for row in retention['rows']:
            if row['variant'] != 'raw':
                continue
            mean = 100*statistics.mean(row['mixed_accuracy_by_seed'])
            delta = 100*row['mixed_minus_baseline']
            lo, hi = [100*v for v in row['paired_ci95']]
            retention_lines.append(f"{row['family']} attains {mean:.2f}\\% with a paired change of {delta:+.3f} points [{lo:+.3f}, {hi:+.3f}].")
        section += '\n' + 'On the already scored 3,079 BANKING test questions, ' + ' '.join(retention_lines) + ' These exploratory retention intervals include zero; they neither establish systematic forgetting nor prove equivalence. They are conditional on the three observed seeds. Mixed and original calibration pools differ; calibrated probability-quality changes cannot isolate a training effect.\n'
        evidence['mixed_banking_retention'] = retention
    source = source.replace(r'\section{Interpretation and remaining work}', section + '\n' + r'\section{Interpretation and remaining work}')
    source = source.replace(r'\usepackage{amsmath,booktabs,hyperref}', r'\usepackage{amsmath,booktabs,hyperref,graphicx}' + '\n' + r'\graphicspath{{figures/}{../figures/}}')
    source = source.replace('The registered supervised mixed-domain stage is now training; its fresh CLINC test remains unscored.', 'The registered supervised mixed-domain stage and fresh CLINC local/community confirmation are complete; the expanded encoder improves to 97.48\\% on the fresh supervised task.')
    source = source.replace('Candidate-order and missing-option GPU checks, larger-capacity community confirmation, supervised mixed-domain evaluation and randomized timing replications remain unfinished.', 'Candidate-order and missing-option GPU checks, further larger-capacity transfer diagnostics and randomized timing replications remain unfinished.')
    begin = source.index(r'\begin{abstract}') + len(r'\begin{abstract}')
    end = source.index(r'\end{abstract}')
    source = source[:begin] + r'''
We study probability-only candidate decisions with a compact joint encoder, a tuned causal decoder, and seven pinned community releases on one 16 GB laptop GPU. Identical tasks provide eight candidate descriptions including the gold option. On 3,079 BANKING questions, three-seed encoder and decoder accuracies are 94.60\% and 94.74\%, with no established encoder accuracy advantage. Expanding training from 8,792 BANKING items to 23,789 BANKING-plus-CLINC items raises the encoder's fresh CLINC accuracy from 79.65\% to 97.48\%; the tuned decoder reaches 98.41\%. Released systems range from 88.52\% to 99.33\% on the same 4,197 CLINC questions. Joint versus independent candidate context exposes a development accuracy--compute tradeoff, while a synthetic policy diagnostic reveals poor local rule transfer. Historical local timings motivate randomized replications that are still pending. These are comparisons of complete systems with different capacity and training exposure, not causal architecture claims. Jev remains unmeasured, and this research draft is not submission ready.
''' + source[end:]
    evidence['fresh_clinc'] = {'source': str(fresh_path.relative_to(ROOT)), 'dataset_sha256': fresh['dataset_sha256'], 'items': fresh['items'], 'arms': fresh['rows'], 'paired_comparisons': fresh['paired_comparisons']}
(paper / 'encoder_probability_pilot.tex').write_text(source.replace('TABLE_ROWS', '\n'.join(table)), encoding='utf-8')
(paper / 'evidence.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
markdown_rows = []
for label, runs in groups:
    ms = [metrics(run) for run in runs]
    accuracies = [m['accuracy_all_failures_wrong'] * 100 for m in ms]
    sd = f'{statistics.stdev(accuracies):.2f}' if len(runs) > 1 else '未适用'
    markdown_rows.append(f"| {label} | {len(runs)} | {statistics.mean(accuracies):.2f}% | {sd} | {statistics.mean(m['nll_clipped_1e-15'] for m in ms):.3f} | {statistics.mean(m['latency_p50_ms'] for m in ms):.2f} |")
chinese = '''# 单张16GB显卡上的概率决策模型：Encoder-only 与因果解码器的准确率、速度和训练成本

**研究草稿，2026-09-30。当前表格是开发集结果；确认集成绩尚未读取，Jev未实测。**

## 摘要

本研究在一台RTX 4090 Laptop 16GB Windows电脑上，比较联合双向编码的候选评分模型、Qwen3-0.6B候选概率读出及LoRA训练，以及Kev公开发布模型。小模型无需生成解释，就可输出候选概率。1000题训练条件下，DistilBERT标量头三种子平均准确率81.17%，候选CE LoRA为86.67%。Kev当前开发集准确率95.5%，但其中16/200题与一个公开Kev训练套件有文本重叠。加入CLS-query指针头未提升三种子平均准确率，且NLL更差。扩展训练数据的实验与独立迁移评测正在推进。当前结果体现系统的准确率与速度取舍，不能证明注意力方向本身的因果优势。

## 任务与实验协议

BANKING77固定Parquet版本，训练/开发/校准/确认四组规范化文本互不重叠。初始训练1000题，其余各200题。每题从77种意图中构造8个候选，强制包含正确项。这个子任务不等于原77类BANKING77跑分，也没有测候选检索。训练时随机候选顺序，超长输入报错而不截断。训练种子42/43/44，FP32，单GPU串行运行。

DistilBERT同时编码状态、问题和候选；候选末位标记用共享线性头评分，softmax只在当前候选之间归一化，因此没有固定类别数。标量头骨干学习率2e-5，3轮训练。指针头把CLS与候选表示分别投影到128维；骨干学习率固定，头学习率对照2e-5与2e-4。Qwen3采用首个回答位置的候选token投影，q/v LoRA rank8，累积8步，candidate CE训练1轮。完整超参数、revision及耗时见逐运行metadata。

Kev采用官方loader、0.8B发布版及其学习温度，FP32/unmerged/eager，不启用CUDA graphs或融合算子。主环境Transformers4.57.6，Kev环境5.17.0；PyTorch2.8.0/CUDA12.8。模型大小、预训练、微调语料、实现均不同，因此只作系统比较。Jev因暂无接口与预算保留为空，不以公开异集分数填补。

## 当前实测

| 系统 | 运行数 | 平均准确率 | 种子标准差/百分点 | 平均NLL | 平均单运行p50/ms |
|---|---:|---:|---:|---:|---:|
MD_ROWS

以上均200题、零失败。多种子p50列是各运行中位数的平均，不是合并样本的中位数。新指针头的三种子平均80.83%，低于标量头81.17%；负结果完整保留。首个5000题标量头实验的准确率为90.5%，对比同种子1000题81.0%；需要补另外两个种子及确认集才能稳健下结论。

## 速度与token口径

概率模型直接输出分布，没有自回归解码，不能把8个概率值称为8个生成token。报告决策/秒及输入token/秒；生成式基模另测固定64-token解码速度。固定算力实验统一GPU、FP32、eager注意力和相同批量、输入token数，但模型规模仍不同。合成固定形状算子测试与含tokenization的真实任务服务延迟分开。

真实任务单题串行测量含分词、传输、前向及概率回传，排除下载和加载。Kev本地p50=112.16ms、p95=118.22ms，HTTP p50=128.19ms单列。Kev日志显示因未安装causal_conv1d及flash-linear-attention而使用参考PyTorch路径，因此这是本机实现的速度，不能当作Kev最优部署速度。HTTP和本地argmax全部一致，概率最大差约0.000185，不宣称逐位等价。

## 数值读出优化

LitJev同权重的末位候选投影避免整个词表投影。64字符合成输入、1/2问题、4候选FP32 GPU测试，实测仅1.033/1.115倍加速；最大概率误差约1.6e-6，argmax一致。它验证数值和工程优化，不等于Jev实测，也不是任务准确率证据。更长输入、多问题和BF16还需完整报告。

## 数据偏差与限制

固定公开Kev v7/decision-v7 train套件的审计发现：本地train92/1000、dev16/200、calibration27/200重复，test0/200重复。匹配采用casefold与空白归一化的精确文本或上游来源哈希。审计只覆盖一个套件，不能证明整个发布模型或其基模无其他重叠。开发分数带有已确认的训练数据混杂，不能据此宣称Kev的架构优势。

三种子只描述已运行的种子变化，200题样本及开发筛选限制可靠性。NLL使用有效响应，失败准确率计错；ECE和Brier同时保留。校准只能拟合独立calibration。确认集冻结方法后读取一次；使用同材料group配对bootstrap，区间条件于实际运行的种子，不能外推到所有架构。

## 后续研究

扩大训练集采用新目录：保留原1000题，再追加4000个不与任何本地held-out group重叠的题目。held-out逐字节不变，两种训练方案使用相同5000题，但encoder3轮与decoder1轮曝光量不同，必须同时报告时间与样本数。CLINC150固定版本冻结300题in-scope/8候选迁移集，不用于训练或开发筛选，也不声称测OOS识别。后续补多种子、确认集、候选乱序/缺失、校准和Jev真实对照。论文仍是草稿。

## 可复现材料及文献

根目录E:/jev；研究协议和进度见handoff/RESEARCH_PROTOCOL.md与handoff/PROGRESS.md，表格映射paper/evidence.json。逐运行保存metrics.json、metadata.json与predictions.jsonl。训练耗时和显存分allocator峰值与包含桌面的整卡采样，不能混用。

- [BANKING77论文](https://arxiv.org/abs/2003.04807)
- [DistilBERT论文](https://arxiv.org/abs/1910.01108)
- [LoRA论文](https://arxiv.org/abs/2106.09685)
- [Qwen3报告](https://arxiv.org/abs/2505.09388)
- [Kev官方源码](https://github.com/jaredpalmer/kev)、[公开checkpoint](https://huggingface.co/jaredpalmer/kev-0.8b)
- [Jev choice接口文档](https://docs.typesafe.ai/primitives/choice)
- [CLINC150原论文](https://aclanthology.org/D19-1131/)
'''
if throughput_section:
    chinese += '\n## 固定形状GPU实测吞吐\n\n合成GPU驻留输入，FP32/eager，5次热身、20次前向，8候选；表格排除分词与拷贝。\n\n| Batch | 输入token数 | Encoder决策/秒 | Qwen决策/秒 | 前者/后者 |\n|---:|---:|---:|---:|---:|\n'
    for row in a:
        other = next(r for r in b if r['task'] == row['task'] and r['batch'] == row['batch'] and r['input_tokens_per_item'] == row['input_tokens_per_item'])
        chinese += f"| {row['batch']} | {row['input_tokens_per_item']} | {row['decisions_per_second']:.1f} | {other['decisions_per_second']:.1f} | {row['decisions_per_second']/other['decisions_per_second']:.2f}倍 |\n"
    chinese += '\nQwen固定64-token greedy/KV-cache生成另测：batch1、输入128token时44.18生成token/秒；batch8合计329.68生成token/秒，均含prefill和decode。Encoder不生成token，不能拿其决策/秒除以Qwen生成token/秒宣称生成加速。以上不同模型参数规模未控制，只表示本机实现。5000题seed42条件下两者开发集均90.5%；encoder训练344.28秒、Qwen训练477.80秒，曝光量分别3轮和1轮，不属于等训练预算。\n'
if ablation:
    chinese += '\n## 同权重注意力实现与精度消融\n\n8792题训练的同一个scalar checkpoint；200题相同开发集、全部零失败。\n\n| 实现 | 准确率 | NLL | p50/ms | 相对FP32/eager预测变化题数 |\n|---|---:|---:|---:|---:|\n'
    for r in ablation['rows']:
        chinese += f"| {r['name']} | {100*r['accuracy']:.1f}% | {r['nll']:.3f} | {r['latency_p50_ms']:.2f} | {r['argmax_disagreement_count']} |\n"
    chinese += '\nFP32/SDPA最大概率偏差8.73e-6，所有argmax保持一致；BF16/eager及BF16/SDPA分别最大偏差0.2125/0.0854，各改变一题，不能称无损。当前优先考察FP32/SDPA。单轮按实现顺序测量，尚需随机顺序重复计时以减小热状态和顺序混杂。\n'
if confirmation and confirmation['rows']:
    chinese += '\n## 完整独立测试与校准\n\n预先冻结1000题独立校准集与3079题测试集；8792题训练与dev、校准、测试group不重合。保留三个种子，不依据测试换模型、提示词或种子。仍是含gold的八候选任务，不代表社区权重未见过公开数据。温度只在校准集拟合，保持测试argmax不变。\n\n| 模型 | 变体 | 准确率均值±种子SD | NLL | ECE |\n|---|---|---:|---:|---:|\n'
    for r in confirmation['rows']:
        chinese += f"| {r['model']} | {r['variant']} | {100*r['accuracy_mean']:.2f}±{100*r['accuracy_sample_sd']:.2f}% | {r['nll_mean']:.3f} | {r['ece_mean']:.3f} |\n"
    chinese += '\n只汇总完整三种子实验；未齐的比较继续进行。Laya/OpenJev官方概率舍入四位，保留raw并仅在已知舍入范围内归一化，NLL仍受舍入影响。SDK接入失败诊断不作为模型成绩。\n'
    if confirmation.get('community_rows'):
        chinese += '\n## 社区发布权重完整测试\n\n各模型为一个固定发布checkpoint，使用同3079题及候选集合，未按本地BANKING训练集再次微调。保留上游校准，再在同1000题上拟合额外温度。社区接口概率可能舍入，而本地模型可用raw scores，不能忽略这一差异。\n\n| 模型 | 变体 | 准确率 | NLL | ECE |\n|---|---|---:|---:|---:|\n'
        for r in confirmation['community_rows']:
            chinese += f"| {r['model']} | {r['variant']} | {100*r['accuracy']:.2f}% | {r['nll']:.3f} | {r['ece']:.3f} |\n"
        chinese += '\nKev一份公开训练suite与完整校准集重合118/1000，测试集0/3079，不代表其全部训练/模型选择无重合。Laya、OpenJev、Von、Decider舍入后的argmax与官方choice在完整测试均0差异。Von固定权重metadata标1.2.0，SDK显示1.3.0，本研究用固定HF revision识别实际权重。当前结果不支持encoder普遍准确率优势；本地模型已针对银行域训练。参考Windows内核使Kev/Decider最优部署速度仍未证明。\n'
if transfer_path.exists():
    chinese += '\n## 已完成的跨领域迁移和规则诊断\n\n只用BANKING训练的encoder在CLINC300题上三种子均值80.67%，Qwen为93.33%；配对差-12.67个百分点，95%区间[-16.44,-9.11]，条件于这三个种子。CLINC题目与本地全部BANKING来源文本无规范化精确交集，不代表社区预训练或模型选择未见过CLINC。\n\n320题新生成规则诊断使用全部8种flag路线，均匀随机准确率12.5%。encoder均值10.00%，Qwen12.29%，Kev30.31%，Decider0.8B32.50%。当前银行域训练未带来可靠规则执行；该诊断不能代表全部真实推理任务。BANKING温度直接迁移、不在目标集重拟合，详见results/transfer-summary.md。\n\n已冻结23789条BANKING+CLINC监督扩展训练集，另留4197条未评分CLINC测试题并排除已评分300题。后续必须称监督领域扩展；已读取BANKING测试的后续结果作为探索性结果。\n'
if context_path.exists():
    chinese += '\n## 独立候选上下文消融\n\n相同5000条训练、三种子、3epoch，independent重复state/question并独立编码八个候选，然后重新组合分数计算问题级CE，不等于Von的打包mask实现。开发均值93.67%，joint91.67%；探索性配对差2.00个百分点，95%区间[0.17,4.00]，条件于这些种子，不是测试确认。独立编码每题八行，总有效token从129.6增至345.4，padding后398.1；各运行p50均值从3.11增至4.49ms。原始logit的结构独立性已有CPU验证，GPU乱序实验仍待执行。历史时延尚未完成随机多轮复测。\n'
(paper / '论文草稿.md').write_text(chinese.replace('MD_ROWS', '\n'.join(markdown_rows)), encoding='utf-8')
print(paper / 'encoder_probability_pilot.tex')


