"""Evaluate the pinned official OpenDecider Nano implementation on unchanged items."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external/opendecider-upstream"))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from decision_lab.common import read_data, sync, validate_probs, write_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if Path(args.out).exists():
        parser.error("Output already exists")
    rows = read_data(args.data)
    model_id = "manjunathshiva/opendecider-nano"
    lock = json.loads((ROOT / "configs/community-models-lock.json").read_text(encoding="utf-8"))[model_id]
    path = ROOT / ".cache/community" / model_id.replace("/", "--") / lock["revision"]
    source = json.loads((ROOT / "handoff/community-source-snapshots.json").read_text(encoding="utf-8"))["opendecider"]
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    from transformers import AutoConfig, AutoTokenizer
    config_object = AutoConfig.from_pretrained(path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    from opendecider.nano import NanoModel
    # Build with the official tokenizer and sequence builder, without allocating weights.
    builder = object.__new__(NanoModel)
    builder.__dict__.update(tok=tokenizer, mask_id=tokenizer.mask_token_id, max_len=2048)
    maximum = 0
    for row in rows:
        ids, truncated = builder._build(row["state"], row["question"], row["criteria"])
        if truncated or len(ids) > builder.max_len:
            raise ValueError(f"{row['id']}: truncated official input")
        if ids.count(tokenizer.mask_token_id) != len(row["criteria"]):
            raise ValueError(f"{row['id']}: extra or missing option marker")
        maximum = max(maximum, len(ids))
    preflight = {"items": len(rows), "max_input_tokens": maximum,
                 "architecture": config["architectures"], "config_class": type(config_object).__name__,
                 "dataset_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
                 "checkpoint": str(path), "revision": lock["revision"], "source_commit": source["commit"]}
    print(json.dumps(preflight), flush=True)
    if args.preflight_only:
        return
    completed = json.loads((ROOT / "handoff/community-checkpoints.json").read_text(encoding="utf-8"))[model_id]
    if completed["revision"] != lock["revision"] or Path(completed["path"]).resolve() != path.resolve():
        raise ValueError("Checkpoint verification record differs from fixed revision")
    import torch
    import transformers
    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    from opendecider import load
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; refusing silent CPU fallback")
    model = load(str(path), device="cuda", dtype=args.dtype)
    # Disable ModernBERT's optional compiler for the reference Windows PyTorch run.
    model.impl.enc.config.reference_compile = False

    def predict(row):
        probabilities, info = model.decide([(row["state"], row["question"], row["criteria"])])
        if info[0].get("truncated"):
            raise ValueError("Official loader truncated state")
        validate_probs(probabilities[0], row["criteria"])
        return {"probabilities": probabilities[0], "input_tokens": info[0]["input_tokens"]}

    for _ in range(args.warmup):
        predict(rows[0])
    sync("cuda")
    torch.cuda.reset_peak_memory_stats()
    predictions = []
    for row in rows:
        result = {key: row[key] for key in ("id", "label", "split", "group_id")}
        sync("cuda")
        started = time.perf_counter()
        try:
            result.update(predict(row))
        except Exception as error:
            result["error"] = f"{type(error).__name__}: {error}"
        sync("cuda")
        result["latency_ms"] = (time.perf_counter() - started) * 1000
        predictions.append(result)
    metadata = {**vars(args), **preflight, "backend": "opendecider-official-local", "model": model_id,
                "official_metadata": model.meta, "reference_compile": False,
                "attention_impl": model.impl.enc.config._attn_implementation,
                "torch": torch.__version__, "transformers": transformers.__version__,
                "torch_num_threads": torch.get_num_threads(),
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "latency_scope": "B1 local validation + tokenization + transfer + official decide + probability readback; excludes loading and HTTP"}
    write_run(args.out, args.data, predictions, metadata)
    print(f"Saved {len(predictions)} records to {args.out}", flush=True)


if __name__ == "__main__":
    main()
