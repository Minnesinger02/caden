# Caden — Candidate Decision Encoder

A compact encoder that assigns probabilities to user-provided candidate descriptions. The core model is a fine-tuned DistilBERT with a shared scalar readout at candidate end markers. It produces one normalized distribution without generating an explanation. This is a research implementation, not an official Jev model.

## Install and run

Clone this repository and run from its root. Python3.12, PyTorch2.8, Transformers4.57.6, PEFT0.21.1 and safetensors were used. Install an appropriate official PyTorch wheel for your CUDA/CPU environment, then `python -m pip install -r scripts/windows/requirements.txt`. This checkout is a source tree; the project uses `uv package=false`.

The encoder release is published at `Leonard02/caden-encoder-mixed`. Qwen candidate-CE LoRA is a separate baseline at `Leonard02/candidate-qwen3-06b-lora-mixed`. Each model repository preserves seeds42/43/44 under `seed-42`, `seed-43`, `seed-44`; these are individual models, not a measured ensemble.

```python
from huggingface_hub import snapshot_download
from decision_lab.encoder import CandidateEncoder
import torch

folder = snapshot_download('Leonard02/caden-encoder-mixed', revision='f1ab177667e47605d12e84a13f920002add0eff3')
model = CandidateEncoder.load(folder + '/seed-42').eval()
row = {'state': 'A customer wants to replace a lost card.',
       'question': 'Which intent best matches the request?',
       'criteria': {'replace': 'Replace a lost card', 'transfer': 'Make a bank transfer'}}
with torch.inference_mode():
    scores = model([row])[0]
    raw = scores.softmax(-1)
print(dict(zip(row['criteria'], raw.tolist())))
```

The example pins the verified v2 Hub commit for reproducible use. This loader includes the custom scoring head; plain `AutoModel.from_pretrained` only loads the backbone. Default output above is raw. V2 each seed's `calibration.json` stores a temperature fitted on3996 separate calibration questions; calibrated probabilities are `softmax(scores / temperature)`.

## Recorded evaluation

Mixed training uses8792 BANKING plus14997 CLINC examples, three seeds. Same gold-included oracle-eight tasks: Caden mixed BANKING94.42%, fresh CLINC97.48%; tuned Qwen mixed95.00%/98.41%; Jev-1.13.0 API94.45%/99.12%. These are custom shortlist scores, not original77-/150-class benchmark accuracy. Local values are three-seed means; the late Jev pass is exploratory. The BANK-only Caden policy diagnostic is10%; mixed policy was not evaluated in the frozen matrix. Jev solves all320 synthetic policy questions; no general reasoning equivalence is claimed.

RTX4090 Laptop FP32 B1, five randomized local timing blocks: mixed Caden p50 2.92ms, approximately326 decisions/s; mixed Qwen25.33ms, Kev103.37ms. Same200-question Jev HTTP p50 is444.31ms at concurrency4, includes network/server, and has unknown server hardware. This is not a matched GPU speed ratio or aggregate API throughput benchmark.

## Scope and assets

Original code is MIT. Vendored LitJev retains its original LICENSE/NOTICE and file hashes; the root MIT license does not replace third-party licenses. Base model weights use their upstream terms (DistilBERT/Qwen Apache2.0); released local model artifacts use Apache2.0. Training datasets are BANKING77 (card CC BY4.0) and CLINC150 (card CC BY3.0), with source references in model cards. Raw research data, private API responses, training logs, paper drafts and handoff archives are not uploaded here. Advanced analysis drivers expect separate frozen research artifacts and fixed external/community sources.

AI assistance was extensively used in planning, code, experiments, analysis and drafts; human author verification and submission preparation remain required. No published-paper citation is claimed yet. The model links are published and verified; source drafts and private research handoffs remain separate.

## Caden v2 multi-domain continuation


Current encoder revision: `f1ab177667e47605d12e84a13f920002add0eff3`. Original BANKING+CLINC release remains reproducible at `fcf0b281309fb1720d86515f45ec5c573b63216d`. Qwen adapters remain the original mixed-domain baseline.

Each original seed42/43/44 continues for one epoch on12000 items (3000 each BANKING, CLINC, SNIPS and AG News), AdamW1e-5, FP32, random candidate order, with old-domain replay. This extends training exposure; it is not an equal-training-budget comparison against the older baselines. Independent3996-item calibration is included.

| Task | Test items | Accuracy/% ± sample SD/pp | Variance/pp² |

|---|---:|---:|---:|

| banking | 3079 | 95.8212 ± 0.1984 | 0.039380 |

| clinc | 4197 | 97.4823 ± 0.3785 | 0.143251 |

| snips | 1400 | 97.0476 ± 0.5548 | 0.307823 |

| agnews | 7600 | 87.4254 ± 0.1382 | 0.019102 |


Same200 BANKING development payloads, five serial GPU blocks: p50 3.19ms, latency-sum rate 304.7/s. This is not a paired ratio confirmation versus earlier blocks or generated-token speed.

BANKING/CLINC remain oracle-eight shortlist tasks; SNIPS and AG News present the full seven/four labels. SNIPS is the pinned DeepPavlov1400-row variant. These are exploratory re-evaluations after prior test reads. Earlier original-release figures above remain historical, rather than being silently reassigned to v2.

Two CPU TF-IDF baselines were added; source analysis scripts require the separately frozen private research artifacts. `scripts/windows/requirements-baselines.txt` records the tested classic-analysis dependencies. The new training driver is `scripts/train_caden_multidomain_v2.py`; it freezes splits, runs three seeds, checks development gates, calibrates and evaluates before release. It expects the original research data and checkpoints, preserving old outputs.

Dataset attribution: [DeepPavlov/snips](https://huggingface.co/datasets/DeepPavlov/snips), revision45f42ebd9641832fd31137317ca5fc5885c86094; [fancyzhx/ag_news](https://huggingface.co/datasets/fancyzhx/ag_news), revisioneb185aade064a813bc0b7f42de02595523103ca4. Model licensing does not relicense training data.
