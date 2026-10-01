"""Prepare the reviewed Caden v2 model update without contacting remote services."""
import hashlib
import json
from pathlib import Path
import re
import shutil

ROOT=Path(__file__).resolve().parents[1]
STAGE=ROOT/'release-staging/caden-v2'
OLD_MODEL_REV='fcf0b281309fb1720d86515f45ec5c573b63216d'

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    result=json.loads((ROOT/'results/caden-multidomain-v2/completed.json').read_text(encoding='utf-8'))
    if STAGE.exists():raise FileExistsError('Preserve reviewed staging')
    if not all(result['gates'][key] for key in ['source_retention_passed','snips_improvement_passed','agnews_improvement_passed']):raise ValueError('Release gates not passed')
    if any(r['failures_total'] for r in result['tasks'].values()):raise ValueError('Evaluation contains failures')
    folder=STAGE/'huggingface-encoder';folder.mkdir(parents=True)
    shutil.copyfile(ROOT/'release-staging/caden/huggingface-encoder/LICENSE',folder/'LICENSE')
    shutil.copytree(ROOT/'release-staging/caden/huggingface-encoder/upstream-model-cards',folder/'upstream-model-cards')
    (folder/'NOTICE').write_text('Caden local fine-tuned artifacts, 2026 Minnesinger02 / Leonard02.\nDerived from DistilBERT, whose upstream Apache-2.0 terms remain applicable.\nV2 continues the three original mixed-training seeds with BANKING77/CLINC150 replay and SNIPS/AG News supervision.\nTraining data are not redistributed here and are not relicensed under the model license. Dataset source variants and fixed revisions are attributed in README and release-protocol.json.\n',encoding='utf-8')
    for record in result['training']:
        seed=record['seed'];shutil.copytree(ROOT/record['checkpoint'],folder/f'seed-{seed}')
        shutil.copyfile(ROOT/f'results/caden-multidomain-v2/temperature-s{seed}.json',folder/f'seed-{seed}/calibration.json')
    result.pop('published',None)
    result['scope']='V2 exploratory multi-domain continuation; same fixed test tasks, recipes registered after earlier test reads. No causal architecture effect or pristine unseen-test claim.'
    (folder/'evaluation-summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    shutil.copyfile(ROOT/'data/caden-multidomain-v2/protocol.json',folder/'release-protocol.json')
    task_rows=[]
    for task,r in result['tasks'].items():task_rows.append(f"| {task} | {r['items']} | {r['accuracy_mean_percent']:.4f} ± {r['sample_sd_pp']:.4f} | {r['sample_variance_pp_squared']:.6f} | {r['nll_raw_mean']:.4f} | {r['calibrated_nll_mean']:.4f} |")
    (folder/'README.md').write_text('''---
license: apache-2.0
language:
- en
base_model: distilbert/distilbert-base-uncased
library_name: pytorch
datasets:
- legacy-datasets/banking77
- clinc/clinc_oos
- DeepPavlov/snips
- fancyzhx/ag_news
tags:
- candidate-scoring
- research
---

# Caden-Encoder-Mixed v2

Multi-domain continuation of the original Caden candidate encoder: a fine-tuned DistilBERT backbone with a shared scalar candidate-end head. This is a multi-checkpoint bundle, requiring the custom loader from [Minnesinger02/caden](https://github.com/Minnesinger02/caden). Plain root-level `AutoModel` or a Transformers pipeline does not load the decision head.

## Version and loading

The current root contains v2 seeds42/43/44, three independent models rather than a measured ensemble. To reproduce the original BANKING+CLINC-only release, use revision `'''+OLD_MODEL_REV+'''`. Git history preserves its artifacts; do not mix old calibration with new weights. Qwen LoRA at [Leonard02/candidate-qwen3-06b-lora-mixed](https://huggingface.co/Leonard02/candidate-qwen3-06b-lora-mixed) remains the original separately trained baseline, not a v2 encoder or target-domain continuation.

```python
from huggingface_hub import snapshot_download
from decision_lab.encoder import CandidateEncoder
import torch
import json

folder = snapshot_download('Leonard02/caden-encoder-mixed')
model = CandidateEncoder.load(folder + '/seed-42').eval()
row = {'state': 'Please add this song to my playlist.',
       'question': 'Which intent best describes the user request?',
       'criteria': {'add': 'add a song to a playlist', 'weather': 'get a weather forecast'}}
with torch.inference_mode():
    scores = model([row])[0]
    temperature = json.load(open(folder + '/seed-42/calibration.json'))['temperature']
    probabilities = (scores / temperature).softmax(-1)
print(dict(zip(row['criteria'], probabilities.tolist())))
```

For reproducible use pin this update's Hub commit from repository history or the local publication record. Raw probabilities use temperature1.0. Each v2 temperature fits a separate3996-question calibration pool (original3396 plus300 SNIPS and300 AG News). It changes confidence rather than argmax accuracy; it is not a confidence guarantee outside these tasks.

## Training recipe

Each matching original mixed-training seed starts from its own recorded checkpoint (not the exploratory seed42 pilot). Original training was23789 BANKING+CLINC examples for3 epochs. V2 adds12000 samples,3000 each from BANKING,CLINC,SNIPS,AG News, for one epoch: AdamW1e-5, FP32, microbatch1, gradient clipping1, random candidate-order augmentation, max length512. New task supervision and old-domain replay are explicit. Source revisions and split SHA256 are in release-protocol.json; per-seed initialization weight hashes and training metadata are retained. No test/dev/calibration text group appears in continuation training.

The recipe was fixed before these continuation runs; all three seeds and their development results are retained. Gates require mean old-domain dev decline no worse than1pp and improved dev accuracy in both new tasks. Final test scores are exploratory reruns of already viewed benchmark pools, not a new untouched confirmation or a test-selected model. Different systems have unequal training exposure, objectives and implementations.

## Recorded evaluation

Accuracy is mean ± sample standard deviation in percentage points (three seeds,42/43/44). Variance uses ddof1, unit pp². BANKING3079/CLINC4197 are gold-included oracle-eight shortlists, not original77-/150-way classification or retrieval. SNIPS uses the pinned DeepPavlov variant1400 test rows (1395 normalized text groups) with all seven labels; AG News uses all four labels and7600 test rows. These full-label tasks do not use gold-dependent distractor selection. All runs have zero failures.

| Task | Items | Accuracy/% ± SD/pp | Variance/pp² | Raw NLL | Calibrated NLL |
|---|---:|---:|---:|---:|---:|
'''+ '\n'.join(task_rows)+f'''

Five serial FP32/eager B1 timing blocks on RTX4090 Laptop GPU (16GB), two CPU threads, same200 BANKING development payloads: p50 **{result['timing_p50_ms']:.2f}ms**, latency-sum decision rate **{result['timing_latency_rate_per_second']:.1f}/s**. Includes tokenization, transfer, forward, probability readback and validation; excludes loading. This is not generated-token speed, API concurrent throughput, or a paired speed-ratio confirmation versus older blocks. No new paid Jev API pass was run. Community systems have not yet been evaluated on SNIPS/AG News.

## Sources and limitations

- DistilBERT base revision12040accade4e8a0f71eabdb258fecc2e7e948be, Apache-2.0.
- BANKING77: legacy-datasets/banking77; the recorded source card declares CC BY4.0.
- CLINC150: clinc/clinc_oos; the recorded source card declares CC BY3.0.
- SNIPS variant: [DeepPavlov/snips](https://huggingface.co/datasets/DeepPavlov/snips), revision45f42ebd9641832fd31137317ca5fc5885c86094, class-name descriptions only.
- AG News: [fancyzhx/ag_news](https://huggingface.co/datasets/fancyzhx/ag_news), revisioneb185aade064a813bc0b7f42de02595523103ca4.

The model license does not relicense datasets. Source descriptions, checkpoint metadata and empirical performance do not certify complete upstream training exposure, privacy, fairness, general reasoning, missing-gold recognition or deployment safety. V2 robustness and policy behavior must not be inferred from original BANK-only diagnostics. Candidate format and task domain can materially affect results. Raw data, private research/API traces and manuscript drafts are not uploaded here. AI assistance was used extensively in implementation, experiments, analysis and documentation; no published-paper citation is claimed.
''',encoding='utf-8')
    patterns=[re.compile(p) for p in [rb'apikey_[A-Za-z0-9_]{50,}',rb'hf_[A-Za-z0-9]{30,}',rb'sk-[A-Za-z0-9_-]{30,}']]
    files=[]
    for path in sorted(folder.rglob('*')):
        if not path.is_file():continue
        if path.suffix!='.safetensors' and any(p.search(path.read_bytes()) for p in patterns):raise ValueError('Credential pattern in release')
        files.append({'path':path.relative_to(folder).as_posix(),'bytes':path.stat().st_size,'sha256':digest(path)})
    manifest={'repo':'Leonard02/caden-encoder-mixed','previous_revision':OLD_MODEL_REV,'public':True,'version':'v2','files':files,'credential_pattern_scan_passed':True,'published':False}
    (STAGE/'model-manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({'files':len(files),'bytes':sum(r['bytes'] for r in files),'release_prepared':True}))

if __name__=='__main__':main()
