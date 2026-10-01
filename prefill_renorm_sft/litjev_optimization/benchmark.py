"""Paired upstream vs optimized scoring, with identical weights and actual schemas."""
import argparse
import hashlib
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

from decision_lab.common import percentile, sync
from .adapter import MANIFEST, TransformersScorer, optimized_scorer
from litjev.schema import Choice, DecisionSchema
from litjev.scoring import calibrated_distribution


class SyntheticTokenizer:
    """Boundary-preserving character tokenizer, only for random-model mechanics."""
    pad_token_id = 0
    eos_token_id = 1

    def apply_chat_template(self, messages, **kwargs):
        return "\n".join(str(m["content"]) for m in messages) + "\nAssistant:"

    def encode(self, text, **kwargs):
        return [ord(char) + 2 for char in text.replace(": ", ":")]


def distribution(raw, temperature=1.0):
    return np.array(calibrated_distribution(raw.logits, list(range(len(raw.logits))), temperature).probabilities)


def compare(baseline, optimized, state, schema, atol=1e-5):
    a, b = baseline.score(state, schema), optimized.score(state, schema)
    errors, probability_errors = [], []
    for x, y in zip(a, b, strict=True):
        if x.name != y.name:
            raise AssertionError("Question ID mismatch")
        np.testing.assert_allclose(x.logits, y.logits, rtol=atol, atol=atol)
        errors.append(float(np.max(np.abs(x.logits - y.logits))))
        # Include non-unit calibration temperatures; retain upstream renorm exactly.
        for temperature in (.7, 1.0, 2.0):
            pa, pb = distribution(x, temperature), distribution(y, temperature)
            np.testing.assert_allclose(pa, pb, rtol=atol, atol=atol)
            probability_errors.append(float(np.max(np.abs(pa - pb))))
            if int(pa.argmax()) != int(pb.argmax()):
                raise AssertionError("Top choice differs (including possible near-tie rounding)")
        if x.hidden is not None:
            np.testing.assert_allclose(x.hidden, y.hidden, rtol=atol, atol=atol)
    return {"max_logit_abs_error": max(errors), "max_probability_abs_error": max(probability_errors),
            "top_choice_agreement": 1.0}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--model", help="Omit for random tiny model; pass a local path or HF checkpoint for real profiling")
    p.add_argument("--revision")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--state-chars", type=int, nargs="+", default=[64, 256])
    p.add_argument("--questions", type=int, nargs="+", default=[1, 4])
    p.add_argument("--choices", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    if Path(a.out).exists():
        p.error("Output already exists")
    if a.repeats < 2 or a.warmup < 0 or a.threads < 1 or min(a.state_chars + a.questions) < 1 or not 2 <= a.choices <= 26:
        p.error("Invalid benchmark dimensions")
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    dtype = getattr(torch, a.dtype)
    if a.model:
        tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
        model = AutoModelForCausalLM.from_pretrained(a.model, revision=a.revision,
                        torch_dtype=dtype, attn_implementation="eager").to(a.device).eval()
    else:
        tok = SyntheticTokenizer()
        config = Qwen3Config(vocab_size=32768, hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            max_position_embeddings=4096)
        config._attn_implementation = "eager"
        model = Qwen3ForCausalLM(config).to(device=a.device, dtype=dtype).eval()
    baseline = TransformersScorer(model, tok, max_input_tokens=model.config.max_position_embeddings)
    optimized = optimized_scorer(model, tok, max_input_tokens=model.config.max_position_embeddings)
    results = []
    for count in a.state_chars:
        state = ("A customer asks for help. " * (count // 26 + 2))[:count]
        for nq in a.questions:
            schema = DecisionSchema({f"q{i}": Choice(instructions="Pick the best action." + " Explain nothing." * i,
                criteria={f"c{j}": f"Action number {j}" for j in range(a.choices)}) for i in range(nq)})
            compiled = baseline._compile(state, schema)
            parity = compare(baseline, optimized, state, schema, atol=1e-5 if a.dtype == "float32" else .03)
            # Count actual upstream LM head outputs, separately from timings.
            outputs = []
            hook = model.get_output_embeddings().register_forward_hook(lambda module, args, out: outputs.append(list(out.shape)))
            baseline.score(state, schema)
            hook.remove()
            audit = []
            optimized_scorer(model, tok, audit=audit).score(state, schema)
            for _ in range(a.warmup):
                baseline.score(state, schema)
                optimized.score(state, schema)
            times = {"baseline": [], "optimized": []}
            peaks = {"baseline": [], "optimized": []}
            order_log = []
            for _ in range(a.repeats):
                order = ["baseline", "optimized"]
                rng.shuffle(order)
                order_log.append(order)
                for name in order:
                    scorer = baseline if name == "baseline" else optimized
                    sync(a.device)
                    if a.device.startswith("cuda"):
                        torch.cuda.reset_peak_memory_stats(a.device)
                    start = time.perf_counter()
                    raw = scorer.score(state, schema)
                    # Keep upstream candidate normalization in the timed path too.
                    for item in raw:
                        distribution(item)
                    sync(a.device)
                    times[name].append((time.perf_counter() - start) * 1000)
                    if a.device.startswith("cuda"):
                        peaks[name].append(torch.cuda.max_memory_allocated(a.device))
            summary = {name: {"p50_ms": percentile(t, .5), "p95_ms": percentile(t, .95),
                        "samples_ms": t, "cuda_peak_allocated_bytes": max(peaks[name]) if peaks[name] else None}
                        for name, t in times.items()}
            result = {"state_chars": count, "questions": nq, "choices": a.choices,
                "prefix_tokens": compiled.prefix_length, "suffix_tokens": list(map(len, compiled.slot_ids)),
                "baseline_head_output_shapes": outputs, "optimized_projected_elements": sum(x["projected_elements"] for x in audit),
                "parity": parity, "timings": summary, "paired_order": order_log,
                "p50_speedup": summary["baseline"]["p50_ms"] / summary["optimized"]["p50_ms"]}
            results.append(result)
            print(json.dumps({k: result[k] for k in ("state_chars", "questions", "p50_speedup", "parity")}), flush=True)
    import transformers
    output = {"settings": vars(a), "upstream_commit": MANIFEST["commit"], "random_model_only": not bool(a.model),
        "torch": torch.__version__, "transformers": transformers.__version__, "platform": platform.platform(),
        "model_config": model.config.to_dict(), "weights_changed": False,
        "adapter_sha256": hashlib.sha256((Path(__file__).parent / "adapter.py").read_bytes()).hexdigest(),
        "timing_scope": "local scorer.score plus upstream candidate normalization, including prompt compilation, tensor creation, prefill, readout and CPU transfer; excludes model load and HTTP; cold state cache per request, hot process",
        "warning": "Tiny random model results do not estimate accuracy or full-size model/GPU speedup.", "cases": results}
    path = Path(a.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()
