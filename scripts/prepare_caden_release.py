"""Prepare allowlisted public code and models locally; never upload or authenticate."""
import hashlib
import json
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'release-staging/caden'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    if OUT.exists():
        raise FileExistsError('Preserve prepared release; review edits in place')
    code = OUT / 'github'
    code.mkdir(parents=True)
    allowed = {'.py', '.md', '.json', '.toml', '.txt', '.ps1', '.jinja'}
    for folder in ['decision_lab', 'dynamic_candidates', 'prefill_renorm_sft', 'scripts', 'tests', 'configs']:
        for source in sorted((ROOT / folder).rglob('*')):
            if not source.is_file() or any(p in {'__pycache__', '.git', 'reports'} for p in source.relative_to(ROOT / folder).parts):
                continue
            if source.suffix not in allowed and not source.name.upper().startswith(('LICENSE', 'NOTICE')):
                continue
            target = code / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    for name in ['LICENSE', 'pyproject.toml', 'uv.lock']:
        shutil.copyfile(ROOT / name, code / name)
    (code / '.gitignore').write_text('__pycache__/\n*.pyc\n.venv*/\n.env\n.env.*\n.cache/\n.tools/\ncheckpoints/\nresults/\ndata/\nexports/\nhandoff/\nrelease-staging/\n*.zip\n*.safetensors\n*.pt\n*.bin\n*.log\n', encoding='utf-8')
    (code / 'README.md').write_text('''# Caden — Candidate Decision Encoder

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
''', encoding='utf-8')
    apache = (ROOT / 'external/kev-upstream/LICENSE').read_bytes()
    fresh = json.loads((ROOT / 'results/fresh-clinc-summary.json').read_text(encoding='utf-8'))
    for family, target_name, base, checkpoint in [
        ('encoder', 'huggingface-encoder', 'distilbert/distilbert-base-uncased', 'encoder16-s{seed}-e3-mixed23789'),
        ('qwen', 'huggingface-qwen-lora', 'Qwen/Qwen3-0.6B', 'sft16-s{seed}-trainmixed23789')]:
        target = OUT / target_name
        target.mkdir()
        (target / 'LICENSE').write_bytes(apache)
        model_rows = []
        for seed in [42, 43, 44]:
            source = ROOT / 'checkpoints' / checkpoint.format(seed=seed)
            seed_target = target / f'seed-{seed}'
            shutil.copytree(source, seed_target, ignore=shutil.ignore_patterns('README.md', '__pycache__'))
            # LoRA should resolve the exact training base, not a moving revision.
            if family == 'qwen':
                config_path = seed_target / 'adapter_config.json'
                config = json.loads(config_path.read_text(encoding='utf-8'))
                config['revision'] = 'c1899de289a04d12100db370d81485cdf75e47ca'
                config_path.write_text(json.dumps(config, indent=2), encoding='utf-8')
            temp = ROOT / f'results/freshclinc-mixed-{family}-s{seed}-temperature.json'
            shutil.copyfile(temp, seed_target / 'calibration.json')
            run = ROOT / f'results/freshclinc-mixed-{family}-s{seed}-test-raw'
            metrics = json.loads((run / 'metrics.json').read_text(encoding='utf-8'))
            model_rows.append({'seed': seed, 'clinc_items': 4197, 'accuracy': metrics['accuracy_all_failures_wrong'],
                'source_predictions_sha256': digest(run / 'predictions.jsonl')})
        (target / 'evaluation-summary.json').write_text(json.dumps({'rows': model_rows, 'task': 'English oracle-eight, gold shortlisted; not original CLINC150 full classification',
            'dataset_sha256': fresh['dataset_sha256'], 'scope': 'Three observed local seeds, no seed selection from test scores'}, indent=2), encoding='utf-8')
        title = 'Caden-Encoder-Mixed' if family == 'encoder' else 'Qwen3-0.6B Candidate-CE LoRA — Caden baseline'
        accuracy = 97.4823286474 if family == 'encoder' else 98.4115638154
        readout = 'Fine-tuned DistilBERT plus a shared scalar head over candidate end markers. The custom CandidateEncoder loader is required; plain AutoModel is insufficient.' if family == 'encoder' else 'LoRA rank8 on q_proj/v_proj, one probability readout over answer-letter token candidates; not general instruction tuning. The pinned Qwen base is required; do not merge across seeds.'
        library = 'pytorch' if family == 'encoder' else 'peft'
        (target / 'README.md').write_text(f'''---
license: apache-2.0
language:
- en
base_model: {base}
library_name: {library}
datasets:
- legacy-datasets/banking77
- clinc/clinc_oos
tags:
- candidate-scoring
- research
---

# {title}

{readout}

## Structure and usage

Three individual training seeds42/43/44 are stored under `seed-42`/`seed-43`/`seed-44`. Seed42 is the fixed example, not selected as the best test seed. This repository is a multi-checkpoint bundle, not a root-level Transformers pipeline. Clone the source repository `Minnesinger02/caden` and use its loader from the project root. The code is MIT; model artifacts are Apache2.0.

Encoder: `CandidateEncoder.load(snapshot_path + '/seed-42')`. Qwen baseline: `python -m prefill_renorm_sft evaluate --adapter <snapshot>/seed-42 --model Qwen/Qwen3-0.6B --revision c1899de289a04d12100db370d81485cdf75e47ca --data <your-data.jsonl> --out <new-results-dir> --max-length 512`. The repository README provides the complete encoder inference example. Apply the per-seed `calibration.json` temperature to raw scores only when desired; fitting used3396 independent pooled calibration questions, not the test set.

## Training and evaluation

Training8792 BANKING77 +14997 CLINC150 items, total23789; development800, calibration3396; seeds42/43/44. Encoder:3 epochs, scalar joint readout, AdamW2e-5, microbatch1. Qwen:1 epoch, candidate-CE LoRA rank8, LR1e-4, accumulation8. Max length512; no silent truncation. Full recipes are saved per seed in training.json or experiment.json.

Fresh CLINC4197 oracle-eight accuracy: **{accuracy:.2f}% three-seed mean**. Exact per-seed values and source prediction hashes are in evaluation-summary.json. These are gold-included shortlist tasks, not original150-class accuracy, OOS detection, candidate retrieval or zero-shot CLINC. Expansion includes CLINC supervision. Published community training exposure is not fully audited; cross-system results are not a causal architecture comparison. Three seeds do not represent all runs or model populations.

## Limitations and responsible use

Research on English benchmark utterances. No deployment, fairness/privacy, clinical, financial or safety-critical validation. Probability normalization does not establish semantic reliability. BANK-only encoder policy accuracy10% contrasts with Jev100% on320 synthetic rules; the mixed models were not part of that frozen policy/robustness matrix. No unsupported general reasoning claim. Existing research manuscript is not yet a submitted or accepted paper.

## Provenance and licensing

DistilBERT fixed revision12040accade4e8a0f71eabdb258fecc2e7e948be; Qwen fixed revisionc1899de289a04d12100db370d81485cdf75e47ca. Upstream cards declare Apache2.0; their original terms remain applicable. This model distribution uses Apache2.0 at the owner's explicit request. It does not relicense the original datasets: BANKING77 card CC BY4.0, CLINC card CC BY3.0. Raw dataset utterances are excluded from this model repository.

BANKING77: Casanueva et al.2020, https://aclanthology.org/2020.nlp4convai-1.5/ . CLINC150: Larson et al.2019, https://aclanthology.org/D19-1131/ . Base cards: https://huggingface.co/distilbert/distilbert-base-uncased and https://huggingface.co/Qwen/Qwen3-0.6B . NOTICE and upstream cards retain source attribution. OpenAI coding assistance was used extensively; final human review remains the authors' responsibility.
''', encoding='utf-8')
        (target / 'NOTICE').write_text('Caden local fine-tuned artifacts, 2026 Minnesinger02 / Leonard02.\nDerived from DistilBERT (Hugging Face) or Qwen3-0.6B (Qwen Team), as identified in README and per-seed metadata.\nOriginal upstream licenses and notices remain applicable. BANKING77 and CLINC150 dataset attribution is retained; datasets are not relicensed as Apache2.0.\n', encoding='utf-8')
        provenance = target / 'upstream-model-cards'
        provenance.mkdir()
        for name in ['distilbert-README.md', 'qwen06-README.md', 'BANKING77-README.md', 'CLINC150-README.md']:
            shutil.copyfile(ROOT / 'handoff/licensing-source' / name, provenance / name)
    packages = {}
    token_patterns = [r'apikey_[A-Za-z0-9_]{50,}', r'hf_[A-Za-z0-9]{30,}', r'sk-[A-Za-z0-9_-]{30,}']
    for directory in [code, OUT / 'huggingface-encoder', OUT / 'huggingface-qwen-lora']:
        entries = []
        for path in sorted(p for p in directory.rglob('*') if p.is_file()):
            if path.suffix in allowed:
                contents = path.read_text(encoding='utf-8', errors='ignore')
                if any(re.search(pattern, contents) for pattern in token_patterns):
                    raise ValueError('Credential pattern in release file: ' + str(path.relative_to(directory)))
            entries.append({'path': path.relative_to(directory).as_posix(), 'bytes': path.stat().st_size, 'sha256': digest(path)})
        packages[directory.name] = {'files': entries, 'bytes': sum(e['bytes'] for e in entries), 'credential_scan_passed': True}
    manifest = {'github_repo': 'Minnesinger02/caden', 'github_has_issues': False, 'visibility': 'public',
        'encoder_repo': 'Leonard02/caden-encoder-mixed', 'qwen_repo': 'Leonard02/candidate-qwen3-06b-lora-mixed',
        'code_license': 'MIT', 'model_license': 'Apache-2.0', 'published': False, 'packages': packages}
    (OUT / 'release-manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({key: {k: v for k, v in value.items() if k != 'files'} for key, value in packages.items()}, indent=2))


if __name__ == '__main__':
    main()
