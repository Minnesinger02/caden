"""Evaluate pinned OpenJev's official LocalJev on the same frozen Qwen weights."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external/openjev-upstream"))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from decision_lab.common import read_data, sync, validate_probs, write_run, normalize_rounded_probs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if Path(args.out).exists():
        parser.error("Refusing to overwrite results")
    rows = read_data(args.data)
    pin = json.loads((ROOT / "configs/resolved_revisions-parquet.json").read_text(encoding="utf-8"))["decoder"]
    source = json.loads((ROOT / "handoff/community-source-snapshots.json").read_text(encoding="utf-8"))["openjev"]
    from huggingface_hub import hf_hub_download
    from transformers import AutoConfig, AutoTokenizer
    from openjev.core import LocalJev, _build_prompt, DEFAULT_SYSTEM_PROMPT
    from openjev.types import Choice
    # Resolve the pinned cached config directly; HF Hub 1.x snapshot_download
    # also demands unused README/LICENSE files in an otherwise runnable cache.
    path = str(Path(hf_hub_download(pin["id"], "config.json", revision=pin["revision"], local_files_only=True)).parent)
    config = AutoConfig.from_pretrained(path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    maximum = 0
    for row in rows:
        q = Choice(instructions=row["question"], criteria=row["criteria"])
        q.validate()
        prompt = _build_prompt(tokenizer, row["state"], q, DEFAULT_SYSTEM_PROMPT)
        n = len(tokenizer.encode(prompt, add_special_tokens=False))
        if n > config.max_position_embeddings:
            raise ValueError(f"{row['id']}: input exceeds fixed model context")
        maximum = max(maximum, n)
    print(json.dumps({"items": len(rows), "max_input_tokens": maximum, "revision": pin["revision"],
                      "source_commit": source["commit"]}), flush=True)
    if args.preflight_only:
        return
    import torch
    import transformers
    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; refusing silent CPU fallback")
    engine = LocalJev(model_id=path, device="cuda", dtype="float32")

    def predict(row):
        tokens_before = engine.usage["input_tokens"]
        passes_before = engine.usage["forward_passes"]
        response = engine.system_one(row["state"], {"decision": Choice(instructions=row["question"], criteria=row["criteria"])})
        answer = response["answers"]["decision"]
        raw = answer["probabilities"]
        probabilities = normalize_rounded_probs(raw, row["criteria"])
        if engine.usage["forward_passes"] - passes_before != 1:
            raise ValueError("Expected exactly one forward for one choice question")
        return {"probabilities": probabilities, "response_probabilities_raw": raw,
                "response_probability_sum": sum(raw.values()), "official_choice": answer["choice"],
                "input_tokens": engine.usage["input_tokens"] - tokens_before}

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
    metadata = {**vars(args), "backend": "openjev-official-local", "model": pin["id"], "revision": pin["revision"],
                "checkpoint": path, "source_commit": source["commit"], "dtype": "float32",
                "attention_impl": engine.model.config._attn_implementation,
                "system_prompt": DEFAULT_SYSTEM_PROMPT, "torch": torch.__version__, "transformers": transformers.__version__,
                "torch_num_threads": torch.get_num_threads(),
                "probability_processing": "official four-decimal wire values retained and normalized only within K*0.00005+1e-8 rounding sum error; NLL reflects rounded values",
                "max_input_tokens": maximum, "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "latency_scope": "B1 local validation + tokenization + transfer + official LocalJev single forward + probability readback; excludes loading and HTTP"}
    write_run(args.out, args.data, predictions, metadata)
    print(f"Saved {len(predictions)} records to {args.out}", flush=True)


if __name__ == "__main__":
    main()
