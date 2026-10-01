import argparse
import hashlib
import json
import random
import time
from pathlib import Path

from decision_lab.common import read_data, sync, validate_probs, write_run


def run(a):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .model import encode, base_model, last_hidden, objective, PROMPT_VERSION
    from dynamic_candidates.experiment import readout
    if Path(a.out).exists():
        raise ValueError("Output already exists; choose a new run directory")
    rows = read_data(a.data)
    if a.command == "train":
        if any(r.get("split") != "train" for r in rows):
            raise ValueError("Training requires split=train for every example")
        if a.objective in {"teacher_kl", "gold_kl"}:
            for r in rows:
                validate_probs(r["teacher_probabilities"], r["criteria"])
                if a.objective == "gold_kl" and r.get("label_source") != "gold":
                    raise ValueError("gold_kl requires explicit label_source=gold")
        if not 0 <= a.gold_weight <= 1:
            raise ValueError("gold_weight must be in [0, 1]")
        if a.accumulation < 1 or a.epochs < 1 or a.lr <= 0 or a.rank < 1:
            raise ValueError("Invalid training hyperparameters")
    if a.command == "evaluate" and any(r.get("label_source") == "teacher" for r in rows):
        raise ValueError("Evaluation requires independent gold labels, not teacher pseudolabels")
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    adapter_meta = None
    if a.command == "evaluate" and a.adapter:
        adapter_meta = json.loads((Path(a.adapter) / "experiment.json").read_text())
        a.model = adapter_meta["model"]
        a.revision = adapter_meta["resolved_revision"] or adapter_meta["revision"]
        if adapter_meta["prompt_version"] != PROMPT_VERSION:
            raise ValueError("Adapter prompt version mismatch")
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[a.dtype]
    lm = AutoModelForCausalLM.from_pretrained(a.model, revision=a.revision, torch_dtype=dtype,
                                             attn_implementation="eager").to(a.device)
    if lm.config.model_type not in {"qwen2", "qwen3"}:
        raise ValueError("Pilot supports Qwen2/Qwen3 text models only")
    limit = min(a.max_length, lm.config.max_position_embeddings)
    meta = {**vars(a), "resolved_revision": getattr(lm.config, "_commit_hash", None),
            "prompt_version": PROMPT_VERSION, "torch": torch.__version__, "transformers": transformers.__version__}
    if a.command == "train":
        from peft import LoraConfig, get_peft_model
        lm = get_peft_model(lm, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.0,
                            target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM", bias="none"))
        # Fail fast on overlength or invalid labels before spending time training.
        for row in rows:
            encode(row, tok, limit)
        lm.train()
        params = [p for p in lm.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=a.lr)
        from decision_lab.telemetry import TrainingTelemetry
        telemetry = TrainingTelemetry(a.device)
        history = []
        start = time.perf_counter()
        for epoch in range(a.epochs):
            rng.shuffle(rows)
            total = 0.0
            for begin in range(0, len(rows), a.accumulation):
                batch = rows[begin:begin + a.accumulation]
                optimizer.zero_grad(set_to_none=True)
                for row in batch:
                    items = list(row["criteria"].items())
                    rng.shuffle(items)
                    row = {**row, "criteria": dict(items)}
                    ids, candidates = encode(row, tok, limit)
                    loss = objective(lm, last_hidden(lm, ids), candidates, row, a.objective, a.gold_weight)
                    (loss / len(batch)).backward()
                    total += loss.item()
                    telemetry.step(loss.item())
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
            history.append(total / len(rows))
            print(json.dumps({"epoch": epoch + 1, "loss": history[-1]}), flush=True)
        sync(a.device)
        training_stats = telemetry.finish()
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=False)
        lm.save_pretrained(out)
        tok.save_pretrained(out)
        meta.update(loss_history=history, training_seconds=time.perf_counter() - start,
                    trainable_parameters=sum(p.numel() for p in params),
                    dataset_sha256=hashlib.sha256(Path(a.data).read_bytes()).hexdigest())
        meta.update(training_stats)
        (out / "experiment.json").write_text(json.dumps(meta, indent=2))
        return
    if a.adapter:
        from peft import PeftModel
        lm = PeftModel.from_pretrained(lm, a.adapter).merge_and_unload()
    lm.eval()

    def predict(row):
        ids, candidates = encode(row, tok, limit)
        with torch.inference_mode():
            z = readout(last_hidden(lm, ids), base_model(lm).get_output_embeddings(), candidates, "restricted")
            return {"probabilities": dict(zip(row["criteria"], z.softmax(-1).cpu().tolist())),
                    "logits": z.cpu().tolist(), "input_tokens": len(ids)}
    for _ in range(a.warmup):
        predict(rows[0])
    sync(a.device)
    if a.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(a.device)
    predictions = []
    for row in rows:
        sync(a.device)
        start = time.perf_counter()
        result = {k: row.get(k) for k in ("id", "label", "split")}
        result.update(group_id=row.get("group_id", row["id"]), candidate_order=list(row["criteria"]))
        try:
            result.update(predict(row))
        except Exception as e:
            result["error"] = f"{type(e).__name__}: {e}"
        sync(a.device)
        result["latency_ms"] = (time.perf_counter() - start) * 1000
        predictions.append(result)
    meta.update(adapter_training=adapter_meta, torch_num_threads=torch.get_num_threads(), latency_scope="single-question local tokenization + transfer + prefill + restricted readout; adapter merged; excludes loading")
    meta.update(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(a.device) if a.device.startswith("cuda") else None,
                cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(a.device) if a.device.startswith("cuda") else None)
    write_run(a.out, a.data, predictions, meta)


def main():
    p = argparse.ArgumentParser(description="Prefill + candidate renorm, optional answer-token LoRA SFT")
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("train", "evaluate"):
        s = sub.add_parser(name)
        s.add_argument("--data", required=True)
        s.add_argument("--out", required=True)
        s.add_argument("--model", default="Qwen/Qwen3-0.6B")
        s.add_argument("--revision")
        s.add_argument("--device", default="cpu")
        s.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
        s.add_argument("--max-length", type=int, default=2048)
        s.add_argument("--seed", type=int, default=42)
        if name == "train":
            s.add_argument("--objective", choices=["token_sft", "candidate_ce", "teacher_kl", "gold_kl"], default="token_sft")
            s.add_argument("--gold-weight", type=float, default=0.5,
                           help="gold_kl: gold CE weight; remaining weight is KL(teacher || student), T=1")
            s.add_argument("--epochs", type=int, default=1)
            s.add_argument("--rank", type=int, default=8)
            s.add_argument("--lr", type=float, default=1e-4)
            s.add_argument("--accumulation", type=int, default=8)
        else:
            s.add_argument("--adapter")
            s.add_argument("--warmup", type=int, default=3)
    run(p.parse_args())


if __name__ == "__main__":
    main()
