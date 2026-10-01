"""Keep the original vocabulary head; no new classifier and no decode loop."""
import torch
import torch.nn.functional as F
from decision_lab.common import validate_probs
from dynamic_candidates.experiment import readout

PROMPT_VERSION = "choice-answer-colon-v1"


def encode(row, tokenizer, max_length):
    if not 2 <= len(row["criteria"]) <= 26:
        raise ValueError("Pilot requires 2–26 candidates")
    codes = [chr(65 + i) for i in range(len(row["criteria"]))]
    content = "Choose one option. Return its letter only.\nState: " + row["state"] + "\nQuestion: " + row["question"] + "\n"
    content += "\n".join(f"{c}. {k}: {v}" for c, (k, v) in zip(codes, row["criteria"].items()))
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False) + "Answer:"
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(ids) > max_length:
        raise ValueError("Input exceeds max_length; no silent truncation")
    candidates = []
    for code in codes:
        extended = tokenizer.encode(prompt + " " + code, add_special_tokens=False)
        if extended[:-1] != ids:
            raise ValueError("Candidate code is not exactly one token at the answer boundary")
        candidates.append(extended[-1])
    if len(set(candidates)) != len(candidates):
        raise ValueError("Candidate-token collision")
    return ids, candidates


def base_model(model):
    return model.get_base_model() if hasattr(model, "peft_config") else model


def last_hidden(model, ids):
    base = base_model(model)
    device = next(base.parameters()).device
    # Calling the backbone retains LoRA modules attached inside its q/v projections.
    return base.model(input_ids=torch.tensor([ids], device=device), use_cache=False).last_hidden_state[0, -1]


def objective(model, hidden, candidate_ids, row, kind, gold_weight=0.5):
    head = base_model(model).get_output_embeddings()
    target = list(row["criteria"]).index(row["label"])
    if kind == "token_sft":
        # Exactly the answer-token causal LM loss: prefix positions have no loss.
        full = head(hidden).float()
        return F.cross_entropy(full[None], torch.tensor([candidate_ids[target]], device=hidden.device))
    z = readout(hidden, head, candidate_ids, "restricted")
    if kind == "candidate_ce":
        return F.cross_entropy(z[None], torch.tensor([target], device=hidden.device))
    if kind in {"teacher_kl", "gold_kl"}:
        teacher = row["teacher_probabilities"]
        validate_probs(teacher, row["criteria"])
        q = torch.tensor([teacher[k] for k in row["criteria"]], device=hidden.device, dtype=torch.float32)
        kl = F.kl_div(F.log_softmax(z.float(), -1), q, reduction="sum")
        if kind == "teacher_kl":
            return kl
        if not 0 <= gold_weight <= 1:
            raise ValueError("gold_weight must be in [0, 1]")
        if row.get("label_source") != "gold":
            raise ValueError("gold_kl requires explicit label_source=gold")
        ce = F.cross_entropy(z.float()[None], torch.tensor([target], device=hidden.device))
        return gold_weight * ce + (1 - gold_weight) * kl
    raise ValueError(kind)
