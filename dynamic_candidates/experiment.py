import argparse
import json
import random
import time
from pathlib import Path

from decision_lab.common import read_data, sync, write_run


def perturb(row, mode, seed):
    row = {**row, "criteria": dict(row["criteria"])}
    if mode == "shuffle":
        items = list(row["criteria"].items())
        random.Random(str(seed) + row["id"]).shuffle(items)
        row["criteria"] = dict(items)
    elif mode == "missing":
        if "__none__" in row["criteria"]:
            raise ValueError("Reserved fallback ID already present")
        del row["criteria"][row["label"]]
        row["criteria"]["__none__"] = "None of the listed candidates is correct."
        row["label"] = "__none__"
    return row


def readout(hidden, head, candidate_ids, mode):
    import torch
    import torch.nn.functional as F
    if mode == "restricted":
        ids = torch.tensor(candidate_ids, device=hidden.device)
        bias = head.bias[ids] if getattr(head, "bias", None) is not None else None
        return F.linear(hidden, head.weight[ids], bias).float()
    if mode == "full":
        return head(hidden)[candidate_ids].float()
    raise ValueError(mode)


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--revision")
    p.add_argument("--device", default="cpu")
    p.add_argument("--projection", choices=["full", "restricted"], default="restricted")
    p.add_argument("--condition", choices=["original", "shuffle", "missing"], default="original")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--max-length", type=int, default=4096)
    args = p.parse_args()
    if Path(args.out).exists():
        p.error("Output directory already exists; choose a fresh run name")
    rows = [perturb(r, args.condition, args.seed) for r in read_data(args.data)]
    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    lm = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision, attn_implementation="eager").to(args.device).eval()
    if lm.config.model_type not in {"qwen2", "qwen3"}:
        raise ValueError("Verified code path supports Qwen2/Qwen3 text models only")
    max_length = min(args.max_length, lm.config.max_position_embeddings)

    def run(row):
        start = time.perf_counter()
        n = len(row["criteria"])
        if n > 26:
            raise ValueError("Pilot supports 2–26 verified single-token labels; no silent truncation")
        codes = [chr(65 + i) for i in range(n)]
        content = "Choose one option. Return its letter only.\nState: " + row["state"] + "\nQuestion: " + row["question"] + "\n"
        content += "\n".join(f"{c}. {key}: {desc}" for c, (key, desc) in zip(codes, row["criteria"].items()))
        prompt = tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False) + "Answer:"
        ids = tok.encode(prompt, add_special_tokens=False)
        candidate_ids = []
        for c in codes:
            extended = tok.encode(prompt + " " + c, add_special_tokens=False)
            if extended[:-1] != ids:
                raise ValueError("Label is not exactly one token at this prompt boundary")
            candidate_ids.append(extended[-1])
        if len(set(candidate_ids)) != n:
            raise ValueError("Label-token collision")
        if len(ids) > max_length:
            raise ValueError("Input exceeds max_length; no truncation")
        inputs = torch.tensor([ids], device=args.device)
        sync(args.device)
        encoded = time.perf_counter()
        with torch.inference_mode():
            # Bypass LM wrapper so no full vocabulary projection occurs in restricted mode.
            hidden = lm.model(input_ids=inputs, use_cache=False).last_hidden_state[0, -1]
            sync(args.device)
            after_backbone = time.perf_counter()
            z = readout(hidden, lm.get_output_embeddings(), candidate_ids, args.projection)
            sync(args.device)
            after_projection = time.perf_counter()
            probs = torch.softmax(z, -1).cpu().tolist()
        sync(args.device)
        end = time.perf_counter()
        return {"probabilities": dict(zip(row["criteria"], probs)), "logits": z.cpu().tolist(),
                "latency_ms": (end - start) * 1000, "tokenization_transfer_ms": (encoded - start) * 1000,
                "backbone_ms": (after_backbone - encoded) * 1000,
                "projection_ms": (after_projection - after_backbone) * 1000,
                "input_tokens": len(ids)}

    for _ in range(args.warmup):
        run(rows[0])
    predictions = []
    for row in rows:
        result = {"id": row["id"], "group_id": row.get("group_id", row["id"]), "label": row["label"],
                  "split": row.get("split"), "candidate_order": list(row["criteria"])}
        start = time.perf_counter()
        try:
            result.update(run(row))
        except Exception as e:
            result.update(error=f"{type(e).__name__}: {e}", latency_ms=(time.perf_counter() - start) * 1000)
        predictions.append(result)
    import transformers
    meta = vars(args).copy()
    meta.update(torch=torch.__version__, transformers=transformers.__version__,
                resolved_revision=getattr(lm.config, "_commit_hash", None), dtype=str(next(lm.parameters()).dtype),
                latency_scope="local single question, eager fp32, synchronized timing; excludes loading")
    write_run(args.out, args.data, predictions, meta)


if __name__ == "__main__":
    main()
