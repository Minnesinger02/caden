# Caden — Candidate Decision Encoder

A compact encoder that assigns probabilities to user-provided candidate descriptions. The core model is a fine-tuned DistilBERT with a shared scalar readout at candidate end markers. It produces one normalized distribution without generating an explanation. This is a research implementation, not an official Jev model.

## Install and run

Clone this repository and run from its root. Python3.12, PyTorch2.8, Transformers4.57.6, PEFT0.21.1 and safetensors were used. Install an appropriate official PyTorch wheel for your CUDA/CPU environment, then `python -m pip install -r scripts/windows/requirements.txt`. This checkout is a source tree; the project uses `uv package=false`.

The encoder release is planned at `Leonard02/caden-encoder-mixed`. Qwen candidate-CE LoRA is a separate baseline at `Leonard02/candidate-qwen3-06b-lora-mixed`. Each model repository preserves seeds42/43/44 under `seed-42`, `seed-43`, `seed-44`; these are individual models, not a measured ensemble.

```python
from huggingface_hub import snapshot_download
from decision_lab.encoder import CandidateEncoder
import torch

folder = snapshot_download('Leonard02/caden-encoder-mixed')
model = CandidateEncoder.load(folder + '/seed-42').eval()
row = {'state': 'A customer wants to replace a lost card.',
       'question': 'Which intent best matches the request?',
       'criteria': {'replace': 'Replace a lost card', 'transfer': 'Make a bank transfer'}}
with torch.inference_mode():
    scores = model([row])[0]
    raw = scores.softmax(-1)
print(dict(zip(row['criteria'], raw.tolist())))
```

Download a fixed Hub commit once available for reproducible use. This loader includes the custom scoring head; plain `AutoModel.from_pretrained` only loads the backbone. Default output above is raw. Each seed's `calibration.json` stores a temperature fitted on3396 separate calibration questions; calibrated probabilities are `softmax(scores / temperature)`.

## Recorded evaluation

Mixed training uses8792 BANKING plus14997 CLINC examples, three seeds. Same gold-included oracle-eight tasks: Caden mixed BANKING94.42%, fresh CLINC97.48%; tuned Qwen mixed95.00%/98.41%; Jev-1.13.0 API94.45%/99.12%. These are custom shortlist scores, not original77-/150-class benchmark accuracy. Local values are three-seed means; the late Jev pass is exploratory. The BANK-only Caden policy diagnostic is10%; mixed policy was not evaluated in the frozen matrix. Jev solves all320 synthetic policy questions; no general reasoning equivalence is claimed.

RTX4090 Laptop FP32 B1, five randomized local timing blocks: mixed Caden p50 2.92ms, approximately326 decisions/s; mixed Qwen25.33ms, Kev103.37ms. Same200-question Jev HTTP p50 is444.31ms at concurrency4, includes network/server, and has unknown server hardware. This is not a matched GPU speed ratio or aggregate API throughput benchmark.

## Scope and assets

Original code is MIT. Vendored LitJev retains its original LICENSE/NOTICE and file hashes; the root MIT license does not replace third-party licenses. Base model weights use their upstream terms (DistilBERT/Qwen Apache2.0); released local model artifacts use Apache2.0. Training datasets are BANKING77 (card CC BY4.0) and CLINC150 (card CC BY3.0), with source references in model cards. Raw research data, private API responses, training logs, paper drafts and handoff archives are not uploaded here. Advanced analysis drivers expect separate frozen research artifacts and fixed external/community sources.

AI assistance was extensively used in planning, code, experiments, analysis and drafts; human author verification and submission preparation remain required. No published-paper citation is claimed yet. Model repository links are release targets until publication is verified.
