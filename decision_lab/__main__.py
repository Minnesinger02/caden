import argparse
import hashlib
import json
import os
import random
import time
import urllib.request
from pathlib import Path

from .common import read_data, sync, validate_probs, write_run


def train(args):
    import torch
    from .encoder import CandidateEncoder
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rows = read_data(args.data)
    if any(r.get("split") != "train" for r in rows):
        raise ValueError("Training requires explicit split=train on every row")
    model = CandidateEncoder.create(args.model, args.revision, args.max_length, args.readout, args.head_dim, args.input_mode).to(args.device)
    if args.freeze_backbone:
        model.backbone.requires_grad_(False)
    groups = [{"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": args.lr},
              {"params": list(model.head.parameters()), "lr": args.head_lr or args.lr}]
    optimizer = torch.optim.AdamW([g for g in groups if g["params"]])
    from .telemetry import TrainingTelemetry
    telemetry = TrainingTelemetry(args.device)
    history = []
    for epoch in range(args.epochs):
        model.train()
        if args.freeze_backbone:
            model.backbone.eval()
        random.shuffle(rows)
        total = 0.0
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start:start + args.batch_size]
            # Order augmentation preserves candidate identity and gold label.
            augmented = []
            for row in batch:
                items = list(row["criteria"].items())
                random.shuffle(items)
                augmented.append({**row, "criteria": dict(items)})
            logits = model(augmented)
            losses = [torch.nn.functional.cross_entropy(z[None], torch.tensor([list(r["criteria"]).index(r["label"])], device=args.device))
                      for z, r in zip(logits, augmented)]
            loss = torch.stack(losses).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(batch)
            telemetry.step(loss.item())
        history.append(total / len(rows))
        print(json.dumps({"epoch": epoch + 1, "train_loss": history[-1]}), flush=True)
    metadata = vars(args).copy()
    metadata.update(loss_history=history, data_sha256=hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
                    torch=torch.__version__, resolved_revision=getattr(model.backbone.config, "_commit_hash", None))
    metadata.update(telemetry.finish())
    model.save(args.out, metadata)


class Jev:
    def __init__(self, model, endpoint="https://api.typesafe.ai/v1/systemone", key_env="TYPESAFE_API_KEY", probability_decimals=None):
        self.model = model
        self.endpoint = endpoint
        self.key = os.environ[key_env] if key_env else None
        self.last = {}
        self.probability_decimals = probability_decimals

    def __call__(self, row):
        body = {"state": row["state"], "model": self.model,
                "questions": {"decision": {"type": "choice", "instructions": row["question"], "criteria": row["criteria"]}}}
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = "Bearer " + self.key
        request = urllib.request.Request(self.endpoint,
                    data=json.dumps(body).encode(),
                    headers=headers)
        with urllib.request.urlopen(request, timeout=90) as response:
            result = json.load(response)
        probs = result["answers"]["decision"]["probabilities"]
        tolerance = 1e-5 if self.probability_decimals is None else len(row["criteria"]) * .5 * 10 ** (-self.probability_decimals) + 1e-8
        validate_probs(probs, row["criteria"], sum_tolerance=tolerance)
        self.last = {"served_model": result.get("model"), "usage": result.get("usage")}
        if self.probability_decimals is not None:
            total = sum(probs.values())
            self.last.update(response_probabilities_raw=probs.copy(), response_probability_sum=total)
            probs = {key: value / total for key, value in probs.items()}
            validate_probs(probs, row["criteria"])
        return probs


def evaluate(args):
    rows = read_data(args.data)
    model = None
    if args.backend == "encoder":
        import torch
        from .encoder import CandidateEncoder
        args.dtype = args.dtype or 'float32'
        args.attention_impl = args.attention_impl or 'eager'
        dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16}[args.dtype]
        model = CandidateEncoder.load(args.checkpoint, args.attention_impl).to(device=args.device, dtype=dtype).eval()
        def predict(row):
            with torch.inference_mode():
                z = model([row])[0]
                if args.save_logits:
                    predict.last_logits = dict(zip(row["criteria"], z.float().cpu().tolist()))
                return dict(zip(row["criteria"], torch.softmax(z.float(), -1).cpu().tolist()))
        for _ in range(args.warmup):
            predict(rows[0])
        sync(args.device)
        if args.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(args.device)
    elif args.backend in {"jev", "kev"}:
        predict = Jev(args.model) if args.backend == "jev" else Jev(args.model, args.endpoint, args.api_key_env, probability_decimals=4)
    else:
        def predict(row):
            return dict.fromkeys(row["criteria"], 1 / len(row["criteria"]))
    predictions = []
    for row in rows:
        if model is not None:
            sync(args.device)
        start = time.perf_counter()
        result = {"id": row["id"], "label": row["label"], "split": row.get("split"), "group_id": row.get("group_id", row["id"])}
        try:
            result["probabilities"] = predict(row)
            validate_probs(result["probabilities"], row["criteria"])
            if args.backend == "encoder" and args.save_logits:
                result["logits"] = predict.last_logits
            if args.backend in {"jev", "kev"}:
                result.update(predict.last)
        except Exception as e:
            result.pop("probabilities", None)
            result["error"] = f"{type(e).__name__}: {e}"
        if model is not None:
            sync(args.device)
        result["latency_ms"] = (time.perf_counter() - start) * 1000
        predictions.append(result)
    meta = vars(args).copy()
    meta["latency_scope"] = "HTTP API including transport/server; no retries" if args.backend in {"jev", "kev"} else "local tokenization + transfer + model + probability readback; excludes loading"
    meta["smoke_only"] = args.backend == "uniform"
    if args.backend == "kev":
        meta['probability_processing'] = 'upstream rounds to 4 decimals; require exact candidate support, valid values and sum within K*0.00005+1e-8; retain raw response then normalize to sum one'
    if model is not None:
        import transformers
        meta.update(torch=torch.__version__, transformers=transformers.__version__,
                    torch_num_threads=torch.get_num_threads(),
                    parameters=sum(p.numel() for p in model.parameters()),
                    training=json.loads((Path(args.checkpoint) / "training.json").read_text()),
                    cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(args.device) if args.device.startswith("cuda") else None)
    write_run(args.out, args.data, predictions, meta)
    print(f"Saved {len(predictions)} records to {args.out}")


def main():
    p = argparse.ArgumentParser(description="Root project: encoder-only candidate scoring vs Jev")
    sub = p.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--model", default="answerdotai/ModernBERT-base")
    t.add_argument("--revision")
    t.add_argument("--out", required=True)
    t.add_argument("--device", default="cpu")
    t.add_argument("--max-length", type=int, default=2048)
    t.add_argument("--epochs", type=int, default=3)
    t.add_argument("--batch-size", type=int, default=4)
    t.add_argument("--lr", type=float, default=2e-5)
    t.add_argument("--head-lr", type=float, help="Separate learning rate for the freshly initialized candidate readout")
    t.add_argument("--readout", choices=["scalar", "pointer"], default="scalar")
    t.add_argument("--input-mode", choices=["joint", "independent"], default="joint")
    t.add_argument("--head-dim", type=int, default=128)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--freeze-backbone", action="store_true")
    e = sub.add_parser("evaluate")
    e.add_argument("--data", required=True)
    e.add_argument("--backend", choices=["encoder", "jev", "kev", "uniform"], required=True)
    e.add_argument("--endpoint", default="http://127.0.0.1:8009/v1/systemone", help="Kev API endpoint only")
    e.add_argument("--api-key-env", help="Optional Kev token environment-variable name; never pass the token itself")
    e.add_argument("--checkpoint")
    e.add_argument("--model", default="jev-latest")
    e.add_argument("--device", default="cpu")
    e.add_argument('--dtype', choices=['float32', 'bfloat16'], help='Encoder precision override; API precision remains unknown')
    e.add_argument('--attention-impl', choices=['eager', 'sdpa'], help='Encoder implementation override, same checkpoint weights')
    e.add_argument('--save-logits', action='store_true', help='Retain raw encoder scores for temperature calibration; adds readback to latency')
    e.add_argument("--warmup", type=int, default=3)
    e.add_argument("--out", required=True)
    args = p.parse_args()
    if Path(args.out).exists():
        p.error("Output directory already exists; choose a fresh run name")
    if args.command == "train":
        if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.head_dim < 1 or (args.head_lr is not None and args.head_lr <= 0):
            p.error("epochs, batch-size and lr must be positive")
        train(args)
    else:
        if args.backend != 'encoder' and (args.dtype or args.attention_impl or args.save_logits):
            p.error('Precision, attention and raw score options apply only to local encoder evaluation')
        if args.backend == "encoder" and not args.checkpoint:
            p.error("encoder evaluation requires --checkpoint")
        evaluate(args)


if __name__ == "__main__":
    main()
