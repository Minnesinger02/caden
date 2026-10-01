"""Same-item Kev probabilities with the official loader, measured without HTTP."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))

import torch
import transformers
from kev.api import SystemOneRequest, to_record
from kev.checkpoint import Checkpoint, LoadOptions
from decision_lab.common import read_data, sync, validate_probs, write_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if Path(args.out).exists():
        parser.error("Output already exists")
    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; refusing silent CPU fallback")
    rows = read_data(args.data)
    pinned = json.loads((ROOT / "handoff/kev-checkpoint.json").read_text(encoding="utf-8"))
    checkpoint = Checkpoint(pinned["requested"])
    tokenizer, model = checkpoint.load("cuda", LoadOptions(dtype=torch.float32, merge=False, attn="eager",
                                                           backend="torch", cuda_graphs=False, fused=False))

    def predict(row):
        request = SystemOneRequest(state=row["state"], questions={"decision": {
            "type": "choice", "instructions": row["question"], "criteria": row["criteria"]}})
        record, _ = to_record(request)
        encoded = model.encode(tokenizer, record, strict=True)
        probabilities = dict(zip(row["criteria"], model.probs(encoded)[0].tolist()))
        validate_probs(probabilities, row["criteria"])
        return {"probabilities": probabilities, "input_tokens": len(encoded["ids"])}

    for _ in range(args.warmup):
        predict(rows[0])
    sync("cuda")
    torch.cuda.reset_peak_memory_stats()
    predictions = []
    for row in rows:
        sync("cuda")
        start = time.perf_counter()
        result = {key: row[key] for key in ("id", "label", "split", "group_id")}
        try:
            result.update(predict(row))
        except Exception as error:
            result["error"] = f"{type(error).__name__}: {error}"
        sync("cuda")
        result["latency_ms"] = (time.perf_counter() - start) * 1000
        predictions.append(result)
    meta = {**vars(args), **pinned, "backend": "kev-local", "dtype": "float32", "merge": False,
            "attention": "eager", "cuda_graphs": False, "fused": False,
            "torch": torch.__version__, "transformers": transformers.__version__,
            "torch_num_threads": torch.get_num_threads(),
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "latency_scope": "single-question local validation + tokenization + transfer + official model.probs + probability readback; state cache cold per item; excludes loading and HTTP"}
    write_run(args.out, args.data, predictions, meta)
    print(f"Saved {len(predictions)} records to {args.out}", flush=True)


if __name__ == "__main__":
    main()
