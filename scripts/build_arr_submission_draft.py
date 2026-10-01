"""Build a compact ARR research manuscript only from complete audited experiments."""
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'paper/arr'
sys.path.insert(0, str(ROOT))
from scripts.summarize_replicated_compute import ratio_summary
from scripts.caden_paper_branding import apply_branding
from scripts.enhance_main_results import enhance
from scripts.add_expansion_to_paper import apply_expansion
from scripts.update_caden_v2_paper import apply_v2


def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))


def table(headers, rows, caption, alignment):
    return ('\\begin{table*}[t]\\centering\\small\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{' + alignment + '}\\toprule\n'
        + ' & '.join(headers) + r'\\\midrule' + '\n'
        + '\n'.join(' & '.join(row) + r'\\' for row in rows)
        + '\n' + r'\bottomrule\end{tabular}' + '\n\\caption{' + caption + '}\n\\end{table*}\n')


def accuracy(mean, sd=None):
    return f'{100*mean:.2f}' + (rf' $\pm$ {100*sd:.2f}' if sd is not None else '')


def main():
    sources = {
        'bank': 'results/confirmation-summary.json', 'fresh': 'results/fresh-clinc-summary.json',
        'retention': 'results/mixed-banking-test-retention.json', 'context': 'results/context-ablation-summary.json',
        'perturbation': 'results/candidate-perturbation-summary.json',
        'synthetic': 'results/replicated-fp32-eager-8792/summary.json',
        'timing': 'results/replicated-task-fp32-8792/summary-mixed-reference.json',
        'timing_original': 'results/replicated-task-fp32-8792/summary-original-reference.json',
        'supplement': 'results/decider2b-supplement-summary.json',
        'exposure': 'handoff/training-exposure-declarations.json', 'licensing': 'handoff/release-provenance.json',
        'training_budget': 'handoff/RECORDED_TRAINING_BUDGET.json',
        'jev': 'results/jev113-comparison-summary.json',
        'jev_timing': 'results/jev-local-timing-comparison.json',
    }
    records = {key: read(path) for key, path in sources.items()}
    fresh, bank, perturbation = records['fresh'], records['bank'], records['perturbation']
    timing, synthetic = records['timing'], records['synthetic']
    jev = records['jev']
    if jev['model'] != 'jev-1.13.0' or any(jev['tasks'][name]['items'] != count for name, count in
            [('banking', 3079), ('fresh-clinc', 4197), ('policy', 320), ('banking-dev', 200)]):
        raise ValueError('Complete pinned Jev comparison required')
    if any(task['metrics']['failures'] for task in jev['tasks'].values()):
        raise ValueError('Update Jev failure narrative before building')
    if fresh['pending'] or len(fresh['rows']) != 15 or fresh['items'] != 4197:
        raise ValueError('Complete fresh confirmation required')
    if len(perturbation['traces']) != 38 or len(perturbation['rows']) != 11:
        raise ValueError('Complete nineteen-model perturbation matrix required')
    if len(timing['blocks']) != 5 or len(timing['rows']) != 14 or timing['reference'] != 'encoder-mixed-eager':
        raise ValueError('Five complete fourteen-model task blocks required')
    if len(synthetic['rows']) != 6 or any(row['blocks'] != 5 for row in synthetic['rows']):
        raise ValueError('Five complete six-shape synthetic blocks required')
    if any(row['failures_total'] for row in fresh['rows']):
        raise ValueError('Update failure narrative explicitly before building')
    template = read('paper/arr/template-lock.json')
    for name in ['acl.sty', 'acl_natbib.bst']:
        if hashlib.sha256((OUT / name).read_bytes()).hexdigest() != template['files'][name]:
            raise ValueError('Official pinned template changed')
    f = {row['arm']: row for row in fresh['rows']}
    b = {(row['model'], row['variant']): row for row in bank['rows']}
    community_bank = {(row['model'], row['variant']): row for row in bank['community_rows']}
    runtime = {row['model']: row for row in timing['rows']}
    ratios = {row['compared']: row for row in timing['paired_ratios']}
    names = {'encoder-eager': 'Encoder BANK, eager', 'encoder-sdpa': 'Encoder BANK, SDPA',
        'encoder-mixed-eager': 'Encoder mixed', 'qwen': 'Qwen BANK', 'qwen-mixed': 'Qwen mixed',
        'encoder-5k-joint': 'Joint encoder, 5k', 'encoder-5k-independent': 'Independent encoder, 5k',
        'laya': 'Laya', 'nano': 'Nano', 'openjev': 'OpenJev', 'von': 'Von',
        'decider08': 'Decider 0.8B', 'decider2b': 'Decider 2B', 'kev': 'Kev'}
    joint, independent = [runtime[key]['decision_rates_by_block'] for key in ['encoder-5k-joint', 'encoder-5k-independent']]
    context_ratio = ratio_summary(joint, independent)
    sdpa_ratio = next(row for row in records['timing_original']['paired_ratios'] if row['compared'] == 'encoder-sdpa')
    qspeed, kspeed = ratios['qwen-mixed'], ratios['kev']
    primary = []
    for family, label in [('encoder', 'Encoder BANK'), ('qwen', 'Qwen BANK')]:
        primary.append([label, accuracy(b[(family, 'raw')]['accuracy_mean'], b[(family, 'raw')]['accuracy_sample_sd']),
            accuracy(f[f'bankingonly-{family}-raw']['accuracy_mean'], f[f'bankingonly-{family}-raw']['accuracy_seed_sd']),
            f"{f[f'bankingonly-{family}-raw']['nll_mean']:.4f}"])
    retention = {row['family']: row for row in records['retention']['rows'] if row['variant'] == 'raw'}
    for family, label in [('encoder', 'Encoder mixed'), ('qwen', 'Qwen mixed')]:
        row = retention[family]
        primary.append([label, accuracy(statistics.mean(row['mixed_accuracy_by_seed']), statistics.stdev(row['mixed_accuracy_by_seed'])),
            accuracy(f[f'mixed-{family}-raw']['accuracy_mean'], f[f'mixed-{family}-raw']['accuracy_seed_sd']),
            f"{f[f'mixed-{family}-raw']['nll_mean']:.4f}"])
    for name in ['laya', 'nano', 'openjev', 'von', 'decider08', 'decider2b', 'kev']:
        value = (next(row['accuracy_all_failures_wrong'] for row in records['supplement']['rows']
            if row['task'] == 'test' and row['variant'] == 'raw') if name == 'decider2b'
            else community_bank[(name, 'raw')]['accuracy'])
        primary.append([names[name], accuracy(value), accuracy(f['community-' + name]['accuracy_mean']),
            f"{f['community-' + name]['nll_mean']:.4f}"])
    primary.append(['Jev 1.13 API', accuracy(jev['tasks']['banking']['metrics']['accuracy_all_failures_wrong']),
        accuracy(jev['tasks']['fresh-clinc']['metrics']['accuracy_all_failures_wrong']),
        f"{jev['tasks']['fresh-clinc']['metrics']['nll_clipped_1e-15']:.4f}"])
    speed = []
    for name in ['encoder-mixed-eager', 'qwen-mixed', 'encoder-eager', 'encoder-sdpa', 'qwen',
                 'encoder-5k-joint', 'encoder-5k-independent', 'laya', 'nano', 'openjev', 'von', 'decider08', 'decider2b', 'kev']:
        row = runtime[name]
        if name == 'encoder-mixed-eager':
            comparison = '1.00 (reference)'
        else:
            ratio = ratios[name]
            lo, hi = ratio['ratio_ci95']
            comparison = f"{ratio['geometric_mean_speed_ratio']:.2f} [{lo:.2f}, {hi:.2f}]"
        memory = row['peak_allocated_bytes_by_block']
        peak = f'{max(memory)/2**30:.2f}' if all(value is not None for value in memory) else '--'
        speed.append([names[name], f"{statistics.median(row['p50_ms_by_block']):.2f}",
            f"{row['median_decisions_per_second']:.1f}", comparison, peak])
    api_time = next(row for row in records['jev_timing']['rows'] if row['system'] == 'jev-1.13.0')
    if records['jev_timing']['dataset_sha256'] != jev['tasks']['banking-dev']['dataset_sha256']:
        raise ValueError('API timing task mismatch')
    speed.append(['Jev 1.13 API (HTTP, C=4)', f"{api_time['p50_ms']:.2f}",
        f"{api_time['latency_rate_per_second']:.2f}", '--', '--'])
    robust = []
    local_labels = {'encoder-e3-5k': 'Joint encoder, 5k', 'encoder-e3-independent-5k': 'Independent encoder, 5k',
        'encoder-e3-8792': 'Encoder BANK', 'qwen-train8792': 'Qwen BANK'}
    for row in perturbation['rows']:
        support = row['support']
        robust.append([local_labels.get(row['arm'], names.get(row['arm'], row['arm'])),
            f"{100*row['order_flip_rate_mean']:.2f}",
            *[f"{100*support[condition]['accuracy_mean']:.2f}" for condition in ['original', 'drop_nongold', 'add_nongold', 'gold_absent']]])
    primary_table = table(['System', r'BANK accuracy/\%', r'Fresh CLINC accuracy/\%', 'CLINC raw NLL'], primary,
        'Identical oracle-eight payloads: 3,079 BANKING and 4,197 fresh CLINC questions. Local values are three-seed mean $\\pm$ sample SD (pp); public/API rows use one fixed system. Mixed BANKING retention, the later 2B supplement and the Jev API pass are exploratory on already scored questions. All final runs have zero failures. Community raw includes upstream calibration; Jev NLL uses rounded wire probabilities.', 'lrrr')
    speed_table = table(['System', 'p50/ms', 'Latency rate/s', r'Mixed encoder / system [95\% CI]', 'Peak/GiB'], speed,
        'Identical 200 BANKING development questions. Local rows: five randomized serial blocks, same GPU, FP32, two CPU threads, B=1; p50 is median of block medians and rate is median block rate. Rate means questions divided by summed request latency. Jev: one HTTP pass, concurrency4, no warmup, unknown server hardware/precision; includes transport and server. Its rate is not aggregate concurrent throughput or measured serial throughput. GPU ratios and memory are unavailable for Jev. Local ratios use paired-block geometric means; peak is largest allocator peak excluding desktop/driver. Prompts, sizes and kernels differ.', 'lrrrr')
    robust_table = table(['System', r'Order flips/\%', r'Original/\%', r'Drop/\%', r'Add/\%', r'Absent+none/\%'], robust,
        'Exploratory diagnostics on 200 source questions; original plus four orders or three support changes. Flip rate compares four permutations to each original. Drop/add modify a nongold candidate. Absent removes gold and supplies an explicit none candidate: a separate task. Local rows average three seeds; community rows use one system. All runs have zero failures. Mixed checkpoints are outside this frozen robustness matrix.', 'lrrrrr')
    qlo, qhi = qspeed['ratio_ci95']
    klo, khi = kspeed['ratio_ci95']
    source = r'''\documentclass[11pt]{article}
\usepackage[review]{acl}
\usepackage{times,latexsym}
\usepackage[T1]{fontenc}
\usepackage{microtype,amsmath,booktabs,graphicx,hyperref}
\graphicspath{{../figures/}}
\title{Compact Candidate Encoders for Probability-Only Decisions\\under a Single-GPU Budget}
\author{Anonymous ACL submission}
\begin{document}
\maketitle
\begin{abstract}
We study the accuracy, candidate robustness, and decision cost of a compact encoder on one 16 GB laptop GPU. On identical oracle-eight BANKING questions, a three-seed encoder reaches 94.60\% accuracy and a tuned decoder 94.74\%, without an established encoder accuracy advantage. Expanding training to 23,789 BANKING-plus-CLINC items raises fresh CLINC encoder accuracy from 79.65\% to 97.48\%; the tuned decoder reaches 98.41\%. Seven pinned community systems range from 88.52\% to 99.33\% on the same 4,197 CLINC questions. Five randomized timing blocks measure encoder/decoder and encoder/Kev decision-rate ratios of MIX_Q_SPEED and MIX_KEV_SPEED in this Windows runtime. Independently encoding candidates removes observed order flips but repeats premise computation; none-option recognition remains a separate weakness. These complete-system comparisons have different capacity, upstream exposure, and kernels. A pinned Jev API pass adds a transport-inclusive comparator. Our contribution is a reproducible account of domain-specific accuracy--cost and context tradeoffs, not a causal architecture ranking.
\end{abstract}
\section{Introduction}
A decision interface can return probabilities over user-described candidates without generating an explanation. That interface does not determine whether the model jointly encodes options, isolates them, or reads probabilities from a causal language model. We ask what a small, locally trainable encoder offers under a 16 GB GPU budget: domain accuracy after additional supervision, stability under candidate changes, and measured decision cost.
We compare complete systems on identical questions and candidate payloads, not on their authors' differently constructed benchmark tables. Three observations shape the analysis. Domain expansion produces a large supervised CLINC gain, while BANKING-only transfer and synthetic policy execution expose limitations. Candidate isolation improves order stability but changes computational work and does not ensure reliable abstention. Finally, repeated implementation timing is necessary to separate empirical speed from a single serial trace. We retain frozen input specifications, per-item probabilities, version pins and audit scripts as local replication artifacts; anonymous packaging and rendered manuscript validation are tracked separately.
\section{Related Systems}
DistilBERT\citep{distilbert} provides our approximately 66M-parameter backbone. The tuned causal baseline is Qwen3-0.6B\citep{qwen} with candidate cross-entropy LoRA\citep{lora}; it performs one probability readout. OpenJev\citep{openjev} is a pinned public probability-readout implementation using the same pinned Qwen base, rather than a separately trained checkpoint in our study.
The other public systems differ materially. Laya\citep{laya} combines ModernBERT-large with a Transformer decision head and published proper-score training. Nano\citep{opendecider} uses an Ettin encoder, option-leading mask markers and a small MLP, with teacher-distribution training. Von\citep{von} uses independent-options attention, reset positions and input-conditioned calibration; its packed implementation differs from our repeated-premise ablation. Kev\citep{kev} and Decider\citep{decider} use larger hybrid language-model backbones and distinct decision training/readouts. Official loaders, source snapshots, upstream temperatures and checkpoint revisions are preserved. Jev\citep{jev} is a proprietary API comparator fixed to jev-1.13.0 by requested and returned version; its parameters, precision, hardware and full training exposure are not available.
\section{Task and Experimental Protocol}
Each record contains state text, a question and a dynamic map of candidate descriptions. Our primary BANKING77\citep{banking} and CLINC150\citep{clinc} constructions include gold plus seven deterministic distractors. Both datasets contain English intent utterances; speaker demographic representativeness is not established. They measure oracle-shortlist choice, not original 77-way/150-way classification, retrieval or CLINC OOS detection. Frozen pools deduplicate normalized exact text and keep source groups disjoint locally. Full BANKING calibration/test contain 1,000/3,079 items. Initial 1k and expanded 5k/8,792 BANKING training pools preserve the development bytes.
Mixed training contains 8,792 BANKING and 14,997 CLINC items, with 800 development and 3,396 calibration items. Fresh CLINC confirmation contains 4,197 questions and excludes all 300 previously scored CLINC probe texts. Recipe and seeds 42/43/44 are frozen before its scores. Subsequent BANKING retention and the larger 2B supplement are explicitly exploratory because those pools were already scored.
\paragraph{Local readouts.} The joint encoder places state/question and all candidates in one sequence. A shared scalar head reads each candidate-end marker, $z_i=w^\top h_i+b$, then $p_i=\exp(z_i)/\sum_j\exp(z_j)$. It is not a fixed-class head. Three epochs of full tuning use AdamW, learning rate $2\times10^{-5}$, one-question microbatches, clipping and candidate-order augmentation. The independent ablation repeats the same premise for each candidate, batches those sequences, and recombines scalar scores into question-level cross entropy. It controls backbone, training data, epochs and optimizer, but does not share premise computation. Qwen uses permitted first-response token logits, rank-eight query/value LoRA, one epoch, learning rate $10^{-4}$ and accumulation eight. Different capacity, pretraining and training exposure prevent architectural causal claims.
\paragraph{Calibration and statistics.} One positive temperature per local checkpoint is fit only on held-out calibration and applied unchanged. Original BANK-only models use 1,000 BANKING items; mixed models use 3,396 pooled items. Community raw retains upstream calibration; an additional BANKING temperature is reported in the appendix. Raw local scores avoid probability-underflow information loss. Choices are unchanged by temperature. Failures count wrong in accuracy; NLL clips at $10^{-15}$ over valid responses, with Brier and ten-bin ECE retained. Local means/sample SD describe three observed seeds. Accuracy intervals use 5,000 paired source-question resamples conditional on those seeds; secondary comparisons are individual intervals without simultaneous correction.
\paragraph{Hardware and timing.} RTX 4090 Laptop GPU (16,376 MiB), i9-13980HX, approximately 31.6 GiB host RAM, PyTorch 2.8/CUDA 12.8 and FP32 are fixed. Main/community environments use Transformers 4.57.6/5.17. Each GPU model runs alone with two recorded CPU threads. Real-task timing uses five randomized model blocks, seed 1729, identical 200 BANKING development payloads, B=1 and three warmups. Seed42 is fixed for timing, not selected by accuracy. Tokenization, transfers, validation and probability readback are included; loads/downloads are excluded and per-item state caches are cold. Decision rate is questions divided by summed synchronized latency, not inverse p50. Paired-block geometric ratios and 5,000 resamples describe this machine/runtime. Synthetic FP32/eager B=1/8/16, L=128/256 blocks warm five times and time 100 forwards, excluding application overhead. Equal hardware/shapes do not equalize parameters or FLOPs. No generated-token rate is defined for the encoder.
\section{Accuracy and Domain Expansion}
PRIMARY_TABLE
\paragraph{Pinned API supplement.} After the local recipes and task bytes were fixed, Jev-1.13.0 was evaluated once on the same 3,079 BANKING, 4,197 CLINC, 320 policy and 200 BANKING development payloads. BANKING/CLINC probability-argmax accuracy is JEV_BANK/JEV_CLINC\%; policy accuracy is JEV_POLICY\%. This late comparator is exploratory, not a new held-out model selection claim. Four concurrent HTTP requests have BANKING/development p50 JEV_LAT_BANK/JEV_LAT_DEV ms, including transport and server work; Jev is excluded from the local-GPU speed-ratio table. We retain original wire probabilities and normalize only sum errors bounded by their observable decimal grid. Rounded zeros make clipped NLL distinct from unavailable internal-logit NLL. All 7,796 final responses are valid; official choices that are not probability maxima number JEV_NONMAX, with tie differences separately audited. Documented-price expenditure is estimated at \$JEV_COST for 7,800 paid requests, including four unpersisted responses from a stopped validation attempt that were remeasured once. The 596 saved answers were reused, and no automatic retries or paid teacher collection were used.
The BANK-only encoder minus tuned Qwen difference on full BANKING is -0.141 pp, with a paired 95\% interval [-0.823, 0.530]. This does not establish encoder superiority or predefined equivalence. Fresh CLINC expansion gains are +17.830 pp [16.782, 18.855] for the encoder and +4.718 [4.154, 5.321] for Qwen. Mixed encoder minus mixed Qwen is -0.929 pp [-1.350, -0.516]. These are supervised gains on a frozen local confirmation, not zero-shot performance or architecture effects.
Original-domain retention is 94.42\% for the mixed encoder, change -0.173 pp [-0.671, 0.314], and 95.00\% for mixed Qwen, +0.260 [-0.195, 0.704]. Both intervals include zero; neither stable forgetting nor equivalence is proved. On the shared mixed calibration pool, test NLL changes from 0.1904 to 0.1033 (encoder) and 0.0648 to 0.0577 (Qwen), with ECE 0.0080/0.0048 after calibration. Cross-stage calibration comparisons also change pool composition.
Community rankings depend on the task: Kev and Von perform well on BANKING but substantially below their local mixed counterparts on this CLINC construction; Decider 2B has the highest observed CLINC accuracy. Author cards explicitly name BANKING in Von/Decider training and list CLINC in Nano dataset metadata. These declarations are not per-item overlap proofs. Our separate exact-text audit finds 118/1,000 calibration and 0/3,079 BANKING test overlaps with one public Kev suite; it cannot certify all training or selection histories.
\section{Candidate Context and Robustness}
ROBUST_TABLE
On matched 5k training, independent versus joint development accuracy is 93.67\% versus 91.67\%, with an exploratory paired gain of 2.00 pp [0.17, 4.00]. Across three seeds, independent scoring has zero flips over each seed's 800 permutation comparisons. Its per-candidate score depends only on the shared premise and that candidate; support changes can still change normalized probabilities. Von also has zero observed order flips under its official independent-options implementation. This is empirical stability on 200 source questions, not proof of robustness on every input.
Independent processing expands one encoder row to eight: 129.6 valid joint tokens versus 345.4 independent valid tokens, 398.1 padded tokens. Paired repeated timing gives a joint/independent decision-rate ratio of CONTEXT_RATIO [CONTEXT_CI]; capacity and examples are matched for these two ablation checkpoints. Independence does not solve missing-gold recognition: independent5k chooses explicit none on 34.50\%, joint5k on 43.17\%, BANK-only full-pool joint on 21.50\%. This separate task is not an OOS-detection or deployment abstention guarantee. The mixed-domain checkpoints are outside this frozen robustness matrix.
\section{Measured Cost and Engineering Ablations}
SPEED_TABLE
The mixed encoder/Qwen decision-rate ratio is MIX_Q_SPEED [MIX_Q_CI], and mixed encoder/Kev is MIX_KEV_SPEED [MIX_KEV_CI]. These intervals reflect five paired blocks, not future platforms. Kev/Decider use Windows reference recurrent kernels because fused causal-convolution/gated-delta packages are unavailable; the comparison is the measured runtime, not their optimized deployment ceiling. Different tokenizer/prompt lengths and validation paths remain part of system cost. Allocator peaks and sampled board memory are distinct quantities.
For identical BANK encoder weights, FP32 SDPA previously preserved all 200 development choices with maximum probability difference $8.73\times10^{-6}$. The repeated eager/SDPA decision-rate ratio is SDPA_RATIO [SDPA_CI]; a ratio below one favors SDPA. BF16 changed one development prediction in each attention implementation and is not called lossless. A pointer readout failed to improve the initial three-seed accuracy and worsened NLL. Candidate-only vocabulary projection with identical Qwen weights produced modest synthetic engineering gains (1.033/1.115 for one/two questions), distinct from Jev API results.
\section{Discussion and Conclusion}
A small locally trained joint encoder is a useful domain-specific probability scorer: added supervision sharply improves a fresh second-domain confirmation, and its current runtime has a measurable decision-cost advantage. It is not a general replacement for larger decision systems. Candidate isolation provides one route to order stability at a repeated-premise cost, while explicit-none behavior and policy execution remain separate capabilities. These controlled context results and complete-system measurements provide reproducible deployment tradeoffs rather than a new claim that encoder directionality causes superior accuracy.
\section*{Limitations}
Oracle candidate inclusion and benchmark descriptions limit external validity. Three seeds, one public checkpoint per system and five timing blocks do not represent model or hardware populations; small development data was involved in recipe selection. Community training exposure is partly declared and not fully audited. Mixed/source calibration pools differ; rounded official probabilities limit local-versus-community calibration equivalence. The robustness matrix measures BANK-only checkpoints, not expanded models. Our independently encoded sequences do not implement Von's packed mask. Synthetic fixed shapes exclude application overhead and are not equal FLOPs. Policy320 is mechanically labeled synthetic data, not general reasoning. An hours-long training timing discontinuity is preserved and excluded from uninterrupted training-speed claims. The late Jev API pass has unknown server hardware and transport-inclusive latency, not matched local GPU compute. Final anonymous artifact checks and rendered ARR page-limit validation must be complete before submission; the local native compiler currently fails platform initialization.
\section*{Ethical Considerations}
We use public benchmark utterances and synthetic diagnostics; no new human participants or deployed user-impact decisions are involved. We have not performed a dedicated personal-information, offensive-content or demographic-fairness audit. Normalized probabilities are not evidence of factual reliability outside the measured tasks. Dataset/model cards, licenses, source revisions and overlap evidence are recorded. Dataset cards declare CC BY 4.0 (BANKING) and CC BY 3.0 (CLINC); these declarations do not establish ownership of all underlying material. An eventual distributed package must retain relevant notices and separate code/weight terms.
\section*{Acknowledgements}
An OpenAI coding assistant was used extensively for research planning, implementation and debugging, experiment orchestration, analysis scripts, manuscript drafting and revision, and the Chinese counterpart. Numerical claims are tied to executed experiment traces and saved audits, rather than generated estimates. The assistant is not an author. Human authors must review and verify the complete manuscript, code, citations and disclosures and assume responsibility before submission; that final author review is not certified by this working draft.
\section*{Reproducibility Statement}
Frozen data manifests, training recipes, checkpoint/source pins, failure-inclusive metrics, raw probabilities/scores, GPU telemetry and group/block resampling scripts accompany this study. English is the mother manuscript; Chinese is reviewed against its source hash with identical table projections. A local artifact manifest records evidence hashes and the scope of each audit. The source uses the pinned official ACL anonymous review template. Compilation, page count and anonymous release validation are tracked as separate gates rather than inferred from source existence.
\begin{thebibliography}{99}
\bibitem[Casanueva et al.(2020)]{banking} I\~nigo Casanueva, Tadas Tem\v{c}inas, Daniela Gerz, Matthew Henderson, and Ivan Vuli\'{c}. 2020. Efficient Intent Detection with Dual Sentence Encoders. In \textit{Proceedings of the 2nd Workshop on Natural Language Processing for Conversational AI}, pages 38--45. Association for Computational Linguistics. \url{https://aclanthology.org/2020.nlp4convai-1.5/}
\bibitem[Larson et al.(2019)]{clinc} Stefan Larson, Anish Mahendran, Joseph J. Peper, Christopher Clarke, Andrew Lee, Parker Hill, Jonathan K. Kummerfeld, Kevin Leach, Michael A. Laurenzano, Lingjia Tang, and Jason Mars. 2019. An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction. In \textit{Proceedings of EMNLP-IJCNLP}, pages 1311--1316. Association for Computational Linguistics. \url{https://aclanthology.org/D19-1131/}
\bibitem[Sanh et al.(2019)]{distilbert} Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf. 2019. DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. Workshop on Energy Efficient Machine Learning and Cognitive Computing at NeurIPS; arXiv:1910.01108. \url{https://arxiv.org/abs/1910.01108}
\bibitem[Hu et al.(2021)]{lora} Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2021. LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685. \url{https://arxiv.org/abs/2106.09685}
\bibitem[Yang et al.(2025)]{qwen} An Yang et al. 2025. Qwen3 Technical Report. arXiv:2505.09388. \url{https://arxiv.org/abs/2505.09388}
\bibitem[Palmer(2026)]{kev} Palmer. Kev source and released adapter. \url{https://github.com/jaredpalmer/kev}
\bibitem[NandhaKishorM(2026)]{laya} NandhaKishorM. Laya. \url{https://github.com/NandhaKishorM/laya}
\bibitem[Janardhan(2026)]{opendecider} Manjunath Janardhan. OpenDecider. \url{https://github.com/manjunathshiva/opendecider}
\bibitem[Panisa(2026)]{von} Victor Hugo Panisa. Von. \url{https://huggingface.co/wfzyx/von}
\bibitem[Mapika(2026)]{decider} Mapika. Decider. \url{https://github.com/Mapika/decider}
\bibitem[lookski(2026)]{openjev} lookski. OpenJev. \url{https://github.com/lookski/openjev}
\bibitem[TypeSafe AI(2026)]{jev} TypeSafe AI. Choice primitive documentation. \url{https://docs.typesafe.ai/primitives/choice}
\end{thebibliography}
\appendix
\section{Additional Diagnostics and Historical Experiments}
The initial 1k scalar/pointer/Qwen development accuracies average 81.17/80.83/86.67\%; scalar5k/Qwen5k rise to 91.67/92.17\%, and scalar8,792/Qwen8,792 to 94.00/93.33\%. These development results selected recipes and are not independent test estimates. Pointer higher head learning rate and all negative runs are retained. Probability averaging across three saved seed predictions did not beat the best member in the development diagnostics; summed stored latencies are hypothetical serial costs, not a measured ensemble service.
On the frozen 300-question BANKING-only CLINC probe, encoder/Qwen means are 80.67/93.33\%, paired difference -12.67 pp [-16.44, -9.11]. On 320 synthetic policy questions (balanced direct/two-hop rules, eight exhaustive routes, uniform12.5\%), encoder/Qwen reach 10.00/12.29\%, Kev30.31\%, Decider0.8B32.50\%, and the later Decider2B supplement41.25\%. This negative diagnostic does not measure all reasoning. Target domains were not used to refit BANKING temperatures; transferred calibration sometimes worsens community NLL. Decider2B reaches 99.67\% on the old CLINC probe and 96.62\% on BANKING; these are supplementary already-read pools.
\section{Synthetic Probability Throughput}
SYNTHETIC_TABLE
These paired blocks compare the selected BANK encoder checkpoint with the frozen pinned Qwen base, not its tuned-task accuracy. Inputs are GPU-resident, nonpadding and identical in shape. Both read eight probabilities without explanation generation; the decoder projects only eight vocabulary rows. Parameters and architecture differ. Reported processing input-token rates are not generated-token rates, and no decision/generation speed ratio is defined.
\section{Calibration, Provenance and Training Timing}
Full BANK local post-temperature NLL is 0.1865 (encoder) and 0.1829 (Qwen), with ECE0.0143/0.0093. The same1000-item pool additionally calibrates community BANK results; raw wire rounding and the Kev suite's118 calibration overlaps are retained caveats. Laya/OpenJev/Von/Decider fresh official choices match probability argmax where saved; Nano/Kev do not provide that stored field and are explicitly unverified in that wire audit. The pinned Von weights' calibration label1.2.0 differs from the newer SDK response alias1.3.0; exact weight/source revisions identify our run.
Mixed encoder seed44 records wall19,268.76s with a17,615.24s telemetry gap and a corresponding training-log elapsed jump. We retain original timing and completed accuracy artifacts, do not infer the cause or subtract guessed pause time, and do not use that wall total as uninterrupted training speed. Independent five-block inference timing supplies the cost evidence instead.
The 32 saved CUDA training-loop records sum to 10.54 recorded hours; the 32 successful training-stage records sum to 10.63 wall hours. These overlapping totals must not be added. They include the documented discontinuity and exclude inference, downloads, unlogged attempts and upstream pretraining, so neither is total active project GPU compute. Our Qwen adapters have 1,146,880 recorded trainable parameters. A separate header-only community tensor inventory is storage evidence, not certified model parameter counts because buffers, tied weights and decision heads may differ.
\end{document}
'''
    synthetic_rows = [[str(row['batch']), str(row['input_tokens_per_item']), f"{row['encoder_rate_median']:.1f}",
        f"{row['qwen_rate_median']:.1f}", f"{row['geometric_mean_speed_ratio']:.2f} [{row['ratio_ci95'][0]:.2f}, {row['ratio_ci95'][1]:.2f}]"]
        for row in synthetic['rows']]
    substitutes = {'PRIMARY_TABLE': primary_table, 'ROBUST_TABLE': robust_table, 'SPEED_TABLE': speed_table,
        'SYNTHETIC_TABLE': table(['B', 'Input length', 'Encoder decisions/s', 'Qwen decisions/s', r'Paired ratio [95\% CI]'],
            synthetic_rows, 'Five paired randomized FP32/eager blocks per shape, five warmups and 100 timed forwards; model loading, tokenization and readback excluded.', 'rrrrr'),
        'MIX_Q_SPEED': f"{qspeed['geometric_mean_speed_ratio']:.2f}", 'MIX_KEV_SPEED': f"{kspeed['geometric_mean_speed_ratio']:.2f}",
        'JEV_BANK': f"{100*jev['tasks']['banking']['metrics']['accuracy_all_failures_wrong']:.2f}",
        'JEV_CLINC': f"{100*jev['tasks']['fresh-clinc']['metrics']['accuracy_all_failures_wrong']:.2f}",
        'JEV_POLICY': f"{100*jev['tasks']['policy']['metrics']['accuracy_all_failures_wrong']:.2f}",
        'JEV_LAT_BANK': f"{jev['tasks']['banking']['metrics']['latency_p50_ms']:.2f}",
        'JEV_LAT_DEV': f"{jev['tasks']['banking-dev']['metrics']['latency_p50_ms']:.2f}",
        'JEV_COST': f"{jev['budget']['estimated_spent_usd']:.4f}",
        'JEV_NONMAX': str(sum(task['official_choice_not_probability_maximum'] for task in jev['tasks'].values())),
        'MIX_Q_CI': f'{qlo:.2f}, {qhi:.2f}', 'MIX_KEV_CI': f'{klo:.2f}, {khi:.2f}',
        'CONTEXT_RATIO': f"{context_ratio['geometric_mean_speed_ratio']:.2f}",
        'CONTEXT_CI': ', '.join(f'{value:.2f}' for value in context_ratio['ratio_ci95']),
        'SDPA_RATIO': f"{sdpa_ratio['geometric_mean_speed_ratio']:.2f}",
        'SDPA_CI': ', '.join(f'{value:.2f}' for value in sdpa_ratio['ratio_ci95'])}
    contrasts = []
    for task, arm in [('banking', 'encoder-bank'), ('fresh-clinc', 'mixed-encoder-raw'), ('fresh-clinc', 'mixed-qwen-raw')]:
        row = next(row for row in jev['tasks'][task]['paired_comparisons'] if row['system'] == arm)
        contrasts.append(f"{100*row['system_minus_jev_accuracy']:+.3f} pp [{100*row['ci95'][0]:.3f}, {100*row['ci95'][1]:.3f}]")
    api_contrast = ('The BANK-only encoder minus Jev BANKING difference is ' + contrasts[0] +
        '; mixed encoder and mixed Qwen minus Jev on CLINC are ' + contrasts[1] + ' and ' + contrasts[2] +
        '. These paired source-group intervals are conditional on the three observed local seeds and one API pass; individual intervals have no simultaneous multiplicity correction. Jev solves all 320 synthetic policy items, versus 10.00/12.29\\% for the BANK-only encoder/Qwen, which highlights task-specific capability limits without proving general reasoning performance.\n')
    source = source.replace('The BANK-only encoder minus tuned Qwen difference', api_contrast + 'The BANK-only encoder minus tuned Qwen difference')
    source = source.replace('Jev is excluded from the local-GPU speed-ratio table', 'Jev has a separate HTTP row in the application timing table and is excluded from local-GPU ratio estimates')
    source = source.replace('The mixed encoder/Qwen decision-rate ratio is',
        f"The Jev HTTP supplement on these same 200 development questions has p50/p95 {api_time['p50_ms']:.2f}/{api_time['p95_ms']:.2f} ms and mean {api_time['mean_ms']:.2f} ms. Its latency-sum rate is {api_time['latency_rate_per_second']:.2f}/s under four concurrent requests, not aggregate API throughput; no additional paid requests were made for this comparison. The mixed encoder/Qwen decision-rate ratio is")
    for key, value in substitutes.items():
        source = source.replace(key, value)
    if any(key in source for key in substitutes):
        raise ValueError('Unresolved manuscript placeholder')
    source = apply_v2(apply_expansion(enhance(apply_branding(source))))
    if (OUT / 'main_en.tex').exists():
        old_hash = hashlib.sha256((OUT / 'main_en.tex').read_bytes()).hexdigest()
        archive = OUT / 'archive' / ('before-compact-' + old_hash[:16])
        archive.mkdir(parents=True, exist_ok=True)
        for name in ['main_en.tex', '中文对应稿.md', 'BUILD_STATUS.json', 'translation-source-lock.json', 'translation-status.json', 'translation-audit.json']:
            original = OUT / name
            target = archive / name
            if original.exists():
                if target.exists() and target.read_bytes() != original.read_bytes():
                    raise FileExistsError(f'Conflicting archived snapshot {target}')
                if not target.exists():
                    shutil.copyfile(original, target)
    (OUT / 'main_en.tex').write_text(source, encoding='utf-8')
    evidence = {'source_files': {key: {'path': path, 'sha256': hashlib.sha256((ROOT / path).read_bytes()).hexdigest()}
        for key, path in sources.items()}, 'matched_context_speed': context_ratio,
        'english_master_sha256': hashlib.sha256((OUT / 'main_en.tex').read_bytes()).hexdigest(),
        'scope': 'Complete audited frozen experiments; compact main narrative with additional diagnostics in appendices. Rendering/page count and renewed Chinese review remain required.'}
    (OUT / 'manuscript-evidence.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    (OUT / 'BUILD_STATUS.json').write_text(json.dumps({'english_master': 'main_en.tex', 'style': 'official ACL review',
        'target': 'ARR long paper', 'main_page_limit': 8, 'anonymous': True, 'pdf_compile_verified': False,
        'limitations_present': True, 'submission_ready': False, 'chinese_matches_current_english': False,
        'english_sha256': evidence['english_master_sha256'],
        'reason': 'Renewed full Chinese review, final anonymous packaging and rendered page verification required; native compiler platform initialization fails.'}, indent=2), encoding='utf-8')
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    main()
