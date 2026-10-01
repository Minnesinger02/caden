# Caden

**Train and run a compact candidate decision encoder.** Caden scores user-provided candidate descriptions with a fine-tuned DistilBERT backbone and a shared scalar head, returning probabilities in one forward pass without generating an explanation. It supports choice questions; it does not implement the full Jev API.

[Model weights](https://huggingface.co/Leonard02/caden-encoder-mixed) · [Fixed v2 release](https://huggingface.co/Leonard02/caden-encoder-mixed/tree/f1ab177667e47605d12e84a13f920002add0eff3)

## Install

Python 3.11+; tested with Python 3.12, PyTorch 2.8 and Transformers 4.57.6. Install the appropriate PyTorch CPU/CUDA wheel for your machine, then:

```bash
git clone https://github.com/Minnesinger02/caden.git
cd caden
python -m pip install -e .
```

Alternatively, install `requirements.txt` and run the modules directly from the repository root. Device selection defaults to CUDA when available, otherwise CPU. Both entry points use FP32 and two CPU threads by default.

## Inference

```bash
caden-infer --input examples/requests.jsonl --output predictions.jsonl
# Equivalent source-tree command:
python -m caden.infer --input examples/requests.jsonl
```

By default this downloads only seed 42 from the fixed v2 Hub commit. Choose `--seed 43` or `--seed 44` for another independently trained checkpoint. Use `--model /path/to/checkpoint` to run local weights, or pass the bundle directory containing `seed-42/`. Checkpoints include the backbone, tokenizer, `head.safetensors` and `training.json`; plain `AutoModel` does not load the scoring head.

Each input line is a JSON object:

```json
{"state":"Please replace my lost card.","question":"Which intent best matches?","criteria":{"replace":"Replace a lost card","transfer":"Make a bank transfer"}}
```

Each output line contains `choice` and `probabilities` keyed by the same candidate IDs. An optional input `id` is copied to the output. No gold labels are needed for inference. Candidate descriptions and IDs must be strings; provide 2–255 candidates. The released model accepts up to 512 tokens including the state, question and all candidates, and rejects longer inputs without truncating them silently.

```python
from caden import Caden

model = Caden.from_pretrained(device="auto")
answer = model.predict({
    "state": "Please add this song to my playlist.",
    "question": "Which intent best matches the request?",
    "criteria": {"playlist": "Add a song to a playlist", "weather": "Get a weather forecast"},
})
print(answer)
```

Inference uses the checkpoint's fitted temperature when `calibration.json` is present. For raw probabilities, pass `--temperature 1` or `Caden.from_pretrained(temperature=1)`. The original `from decision_lab.encoder import CandidateEncoder` import remains supported for existing Hub examples.

## Training and continuation

Supply your own UTF-8 JSONL training data with the same request fields plus a unique string `id`, an explicit `"split":"train"`, and a `label` equal to one candidate ID. See [`examples/train.jsonl`](examples/train.jsonl). These two synthetic examples demonstrate the format and are not enough to train a useful model.

```bash
# Train a new scalar-head encoder from the pinned DistilBERT backbone.
caden-train --data my-train.jsonl --output checkpoints/my-caden --epochs 3 --lr 2e-5 --seed 42

# Continue an existing released seed, using your own training split.
caden-train --data my-train.jsonl --output checkpoints/my-continuation --init-checkpoint Leonard02/caden-encoder-mixed --epochs 1 --lr 1e-5 --seed 42

# Run your resulting checkpoint.
caden-infer --model checkpoints/my-continuation --input examples/requests.jsonl
```

`python -m caden.train` is equivalent to `caden-train`. Use `--device cpu` for CPU training, `--batch-size` to change the microbatch, and `--max-length` to restrict the input budget. Continuation inherits the checkpoint's input format, readout and context limit unless explicitly restricted.

Training fine-tunes the encoder and head using candidate cross-entropy, random candidate-order augmentation, AdamW and gradient clipping at 1. The checkpoint records the data SHA256, seed, recipe, loss history and initialization weight hashes for continuation. Existing checkpoint/output paths are rejected.

Keep development, calibration and test data separate from training. A newly trained checkpoint uses temperature 1: an old checkpoint's calibration is **not** copied to new weights. Fit and validate new calibration on a separate pool if needed. The small examples and default recipe do not reproduce the published release by themselves.

## Released model

`Caden-Encoder-Mixed v2` contains three seeds, 42/43/44, rather than an ensemble. Original training used 23,789 BANKING/CLINC examples for three epochs. V2 continued each matching original checkpoint for one epoch on 12,000 examples, 3,000 each from BANKING, CLINC, SNIPS and AG News, including old-domain replay. Each new temperature uses 3,996 independent calibration questions.

| Task | Accuracy (%) mean ± sample SD (pp), 3 seeds |
|---|---:|
| BANKING, oracle-eight shortlist | 95.82 ± 0.20 |
| CLINC, oracle-eight shortlist | 97.48 ± 0.38 |
| SNIPS, full seven labels | 97.05 ± 0.55 |
| AG News, full four labels | 87.43 ± 0.14 |

The first two are custom gold-included shortlist tasks, not original 77-/150-class benchmark scores. V2 test values are exploratory reruns of previously viewed pools. Five serial blocks on the same 200 BANKING development requests measured p50 about 3.19 ms and 304.7 decisions/s on an RTX 4090 Laptop GPU, FP32, batch 1, two CPU threads; includes tokenization and probability readback, excludes model loading. This is not generated-token speed or API concurrent throughput. Full evaluation details, variance and source revisions are in the [model card](https://huggingface.co/Leonard02/caden-encoder-mixed).

Original weights remain available at revision `fcf0b281309fb1720d86515f45ec5c573b63216d`. The separately published [Qwen LoRA baseline](https://huggingface.co/Leonard02/candidate-qwen3-06b-lora-mixed) is not part of this encoder package.

## Repository contents and license

`caden/` contains model loading, training and inference. `examples/` contains synthetic input formats. `tests/` checks input validation and the train/save/load/infer lifecycle. Run `python -m unittest discover -s tests -v` for the offline smoke tests.

Original code is MIT; published model artifacts are Apache-2.0 with the upstream DistilBERT terms retained. Dataset terms remain separate; see source attributions in the model card. AI assistance was used extensively in implementation and research. No published-paper citation is claimed.
