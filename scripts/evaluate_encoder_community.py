"""Same-item local Laya/Von evaluation through their pinned official SDKs."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from decision_lab.common import read_data, sync, validate_probs, write_run, normalize_rounded_probs


def load_predictor(name, path):
    import torch
    questions = lambda row: {"decision": {"type": "choice", "instructions": row["question"],
                                          "criteria": row["criteria"]}}
    if name == "laya":
        sys.path.insert(0, str(ROOT / "external/laya-upstream"))
        from laya import Agent
        agent = Agent(str(path), device="cuda", fast=False, compile=False)
        # Explicit FP32 reference arm; ship-default autocast is a separate speed arm.
        agent.amp_enabled = False
        agent.dtype = torch.float32
        agent.model.float().eval()
        if agent.device.type != "cuda":
            raise RuntimeError("Official loader unexpectedly fell back to CPU")

        def predict(row):
            response = agent.system_one(row["state"], questions(row))
            usage = response["usage"]
            if usage.get("truncated") or usage.get("options"):
                raise ValueError("State or candidate descriptions were truncated/collapsed")
            answer = response["answers"]["decision"]
            raw = answer["probabilities"]
            # The official wire formatter rounds to four decimals. Preserve
            # that response and allow only the resulting bounded sum error.
            total = sum(raw.values())
            return {"probabilities": normalize_rounded_probs(raw, row["criteria"]),
                    "response_probabilities_raw": raw, "response_probability_sum": total,
                    "official_choice": answer["choice"], "input_tokens": usage["input_tokens"]}

        metadata = {"reference_compile": False, "fast": False, "autocast": False,
                    "checkpoint_config": agent.cfg,
                    "probability_processing": "official SDK four-decimal wire probabilities retained, validated within K*0.00005+1e-8 sum error and normalized; NLL reflects rounded values",
                    "attention_impl": agent.model.encoder.config._attn_implementation}
    else:
        sys.path.insert(0, str(ROOT / "external/von-upstream/src"))
        # Disable multi-pass chain dispatch for the one-pass model comparison.
        os.environ["VON_CHAINS_DIR"] = "off"
        os.environ["VON_ON_OVERFLOW"] = "refuse"
        from von.backends.option_marker_backend import OptionMarkerBackend
        backend = OptionMarkerBackend(checkpoint_dir=str(path), device="cuda")
        model = backend._get_model()
        model.float().eval()
        model.encoder.config.reference_compile = False
        calibration = json.loads((path / "marker_calibration.json").read_text(encoding="utf-8"))
        if backend._independent_options != bool(calibration.get("independent_options", False)):
            raise ValueError("Official loader did not preserve the fixed checkpoint attention mode")
        if backend._default_temp != float(calibration.get("temperature", 1.0)):
            raise ValueError("Official loader did not preserve the fixed checkpoint temperature")

        def predict(row):
            response = backend.evaluate(row["state"], questions(row))
            if response.truncation:
                raise ValueError("Official loader truncated state")
            answer = response.answers["decision"]
            raw = answer.probabilities
            return {"probabilities": normalize_rounded_probs(raw, row["criteria"]),
                    "response_probabilities_raw": raw, "response_probability_sum": sum(raw.values()),
                    "official_choice": answer.choice,
                    "input_tokens": response.usage.input_tokens}

        metadata = {"reference_compile": False, "chains": False,
                    "probability_processing": "official four-decimal wire values retained and normalized within K*0.00005+1e-8 sum error; NLL reflects rounded values",
                    "attention_impl": model.encoder.config._attn_implementation,
                    "independent_options": backend._independent_options,
                    "calibration": calibration, "digit_split": model.digit_split}
    return predict, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["laya", "von"], required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if Path(args.out).exists():
        parser.error("Refusing to overwrite a result")
    rows = read_data(args.data)
    model_id = {"laya": "convaiinnovations/laya", "von": "wfzyx/von"}[args.model]
    lock = json.loads((ROOT / "configs/community-models-lock.json").read_text(encoding="utf-8"))[model_id]
    record = json.loads((ROOT / "handoff/community-checkpoints.json").read_text(encoding="utf-8"))[model_id]
    path = ROOT / ".cache/community" / model_id.replace("/", "--") / lock["revision"]
    if record["revision"] != lock["revision"] or Path(record["path"]).resolve() != path.resolve():
        raise ValueError("Checkpoint verification record differs from fixed revision")
    source = json.loads((ROOT / "handoff/community-source-snapshots.json").read_text(encoding="utf-8"))[args.model]
    import torch
    import transformers
    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; refusing silent CPU fallback")
    predict, model_meta = load_predictor(args.model, path)
    for _ in range(args.warmup):
        result = predict(rows[0])
        validate_probs(result["probabilities"], rows[0]["criteria"])
    sync("cuda")
    torch.cuda.reset_peak_memory_stats()
    predictions = []
    for row in rows:
        result = {key: row[key] for key in ("id", "label", "split", "group_id")}
        sync("cuda")
        started = time.perf_counter()
        try:
            proposed = predict(row)
            validate_probs(proposed["probabilities"], row["criteria"])
            result.update(proposed)
        except Exception as error:
            result["error"] = f"{type(error).__name__}: {error}"
        sync("cuda")
        result["latency_ms"] = (time.perf_counter() - started) * 1000
        predictions.append(result)
    metadata = {**vars(args), **model_meta, "backend": args.model + "-official-local", "model": model_id,
                "checkpoint": str(path), "revision": lock["revision"], "source_commit": source["commit"],
                "dtype": "float32", "torch": torch.__version__, "transformers": transformers.__version__,
                "torch_num_threads": torch.get_num_threads(),
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "latency_scope": "B1 local validation + tokenization + transfer + official prediction + probability readback; excludes loading and HTTP"}
    write_run(args.out, args.data, predictions, metadata)
    print(f"Saved {len(predictions)} records to {args.out}", flush=True)


if __name__ == "__main__":
    main()
