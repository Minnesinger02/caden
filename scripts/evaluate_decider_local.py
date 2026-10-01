"""Pinned official Decider System One, with an explicit reference CUDA runtime."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external/decider-upstream"))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from decision_lab.common import read_data, sync, validate_probs, write_run, normalize_rounded_probs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["Mapika/decider-0.8b", "Mapika/decider-2b"], required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if Path(args.out).exists() or args.threads < 1:
        parser.error("Use fresh output and positive thread count")
    rows = read_data(args.data)
    pin = json.loads((ROOT / "configs/decider-models-lock.json").read_text(encoding="utf-8"))[args.model]
    source = json.loads((ROOT / "handoff/community-source-snapshots.json").read_text(encoding="utf-8"))["decider"]
    path = ROOT / ".cache/community" / args.model.replace("/", "--") / pin["revision"]
    cfg = json.loads((path / "decider_config.json").read_text(encoding="utf-8"))
    from decider.infer import Decider
    from decider.prompt import resolve_layout, chat_template
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    questions = lambda row: {"decision": {"type": "choice", "instructions": row["question"], "criteria": row["criteria"]}}
    probe = object.__new__(Decider)
    probe.m = SimpleNamespace(tok=tok)
    probe.layout = resolve_layout(cfg)
    probe.chat = chat_template(tok) if probe.layout == "chat" else None
    probe.schema_first = False
    probe.isolated_levels = bool(cfg.get("isolated_levels", False))
    probe.neutralize_none = bool(cfg.get("neutralize_none", True))
    maximum = 0
    for row in rows:
        if len(tok.encode(row["state"], add_special_tokens=False)) > 32768:
            raise ValueError("State exceeds official builder budget")
        _, _, items = probe._system_one_items(row["state"], questions(row), max_state_tokens=32768)
        if len(items) != 1 or items[0]["nopts"] != [len(row["criteria"])]:
            raise ValueError("Expected one intact choice question")
        maximum = max(maximum, len(items[0]["ids"]))
    print(json.dumps({"items": len(rows), "max_input_tokens": maximum, "revision": pin["revision"],
                      "source_commit": source["commit"]}), flush=True)
    if args.preflight_only:
        return
    completed = json.loads((ROOT / "handoff/decider-checkpoints.json").read_text(encoding="utf-8"))[args.model]
    if completed["revision"] != pin["revision"] or Path(completed["path"]).resolve() != path.resolve():
        raise ValueError("Verified checkpoint record differs from pin")
    import torch
    import transformers
    torch.set_num_threads(args.threads)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; refusing silent CPU fallback")
    model = Decider(str(path), device="cuda", dtype=getattr(torch, args.dtype), use_graphs=False)

    def predict(row):
        response = model.system_one(row["state"], questions(row), max_state_tokens=32768)
        answer = response["answers"]["decision"]
        raw = answer["probabilities"]
        probabilities = normalize_rounded_probs(raw, row["criteria"])
        return {"probabilities": probabilities, "response_probabilities_raw": raw,
                "response_probability_sum": sum(raw.values()), "official_choice": answer["choice"],
                "input_tokens": response["usage"]["input_tokens"]}

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
    metadata = {**vars(args), "backend": "decider-official-local", "revision": pin["revision"],
                "checkpoint": str(path), "source_commit": source["commit"], "checkpoint_config": cfg,
                "cuda_graphs": False, "attention_impl": model.m.lm.config._attn_implementation,
                "padding_multiple": 64, "max_input_tokens": maximum,
                "probability_processing": "official four-decimal wire values retained and normalized within K*0.00005+1e-8 rounding sum error; NLL reflects rounded values",
                "torch": torch.__version__, "transformers": transformers.__version__,
                "torch_num_threads": torch.get_num_threads(),
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "latency_scope": "B1 validation + tokenization + transfer + official System One + probability readback; excludes loading/HTTP; runtime uses official 64-token padding"}
    write_run(args.out, args.data, predictions, metadata)
    print(f"Saved {len(predictions)} records to {args.out}", flush=True)


if __name__ == "__main__":
    main()
