"""Fine-tune a Caden encoder on explicitly labelled JSONL training data."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import torch

from .api import resolve_checkpoint, resolve_device
from .data import read_jsonl
from .model import CandidateEncoder

BASE_MODEL = "distilbert/distilbert-base-uncased"
BASE_REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Labelled JSONL with split=train")
    parser.add_argument("--output", required=True, help="New checkpoint directory")
    parser.add_argument("--init-checkpoint", help="Continue from a local Caden checkpoint or Hub bundle")
    parser.add_argument("--checkpoint-revision", help="Hub commit for continuation")
    parser.add_argument("--model", default=BASE_MODEL, help="Encoder backbone for new training")
    parser.add_argument("--revision", help="Backbone revision; default DistilBERT is pinned")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, help="Defaults to 512 for new training; inherited on continuation")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args(argv)
    if min(args.epochs, args.batch_size, args.cpu_threads) < 1 or not 0 < args.lr < float("inf"):
        parser.error("epochs, batch-size, cpu-threads and lr must be positive and finite")
    if args.max_length is not None and args.max_length < 1:
        parser.error("max-length must be positive")
    if Path(args.output).exists():
        parser.error("Checkpoint directory already exists; use a new output")
    rows = read_jsonl(args.data, training=True)
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    rng = random.Random(args.seed)
    init_hashes = None
    if args.init_checkpoint:
        folder = resolve_checkpoint(args.init_checkpoint, args.checkpoint_revision, args.seed)
        model = CandidateEncoder.load(folder)
        init_hashes = {p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in folder.rglob("*.safetensors")}
        if args.max_length is not None:
            model.max_length = min(args.max_length, model.backbone.config.max_position_embeddings)
    else:
        revision = args.revision or (BASE_REVISION if args.model == BASE_MODEL else None)
        model = CandidateEncoder.create(args.model, revision, args.max_length or 512)
    model = model.float().to(resolve_device(args.device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    losses = []
    start = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        rng.shuffle(rows)
        total = 0.0
        for offset in range(0, len(rows), args.batch_size):
            batch = []
            for row in rows[offset:offset + args.batch_size]:
                items = list(row["criteria"].items())
                rng.shuffle(items)
                batch.append({**row, "criteria": dict(items)})
            logits = model(batch)
            loss = torch.stack([
                torch.nn.functional.cross_entropy(scores[None], torch.tensor(
                    [list(row["criteria"]).index(row["label"])], device=scores.device))
                for scores, row in zip(logits, batch)]).mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            total += loss.item() * len(batch)
        losses.append(total / len(rows))
        print(json.dumps({"epoch": epoch + 1, "train_loss": losses[-1]}), flush=True)
    if str(next(model.parameters()).device).startswith("cuda"):
        torch.cuda.synchronize()
    metadata = {**vars(args), "loss_history": losses, "train_items": len(rows),
                "data_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
                "torch": torch.__version__, "training_seconds": time.perf_counter() - start,
                "resolved_revision": getattr(model.backbone.config, "_commit_hash", None),
                "initial_weight_sha256": init_hashes,
                "calibration": "No fitted temperature for these new weights; inference uses raw temperature=1."}
    model.save(args.output, metadata)


if __name__ == "__main__":
    main()
