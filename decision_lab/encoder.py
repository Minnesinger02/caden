"""Joint bidirectional encoding, shared scalar head over candidate end markers."""
import json
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

MARKER = "<|decision_candidate_end|>"


class PointerReadout(nn.Module):
    """Query-conditioned scores; the same projections score every candidate."""
    def __init__(self, hidden_size, head_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.query = nn.Linear(hidden_size, head_dim, bias=False)
        self.key = nn.Linear(hidden_size, head_dim, bias=False)
        self.scale = head_dim ** -.5

    def forward(self, query, candidates):
        q = self.query(self.norm(query))
        k = self.key(self.norm(candidates))
        return (k * q).sum(-1) * self.scale


class CandidateEncoder(nn.Module):
    def __init__(self, backbone, tokenizer, max_length=2048, readout="scalar", head_dim=128, input_mode="joint"):
        super().__init__()
        if backbone.config.model_type not in {"bert", "modernbert", "roberta", "distilbert", "deberta", "deberta-v2"} or backbone.config.is_decoder or backbone.config.is_encoder_decoder:
            raise ValueError("This experiment requires an encoder-only backbone")
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.max_length = min(max_length, getattr(backbone.config, "max_position_embeddings", max_length))
        self.marker_id = tokenizer.convert_tokens_to_ids(MARKER)
        if readout not in {"scalar", "pointer"} or head_dim < 1:
            raise ValueError("Invalid readout/head_dim")
        self.readout = readout
        self.head_dim = head_dim
        if input_mode not in {"joint", "independent"}:
            raise ValueError("Invalid input_mode")
        self.input_mode = input_mode
        self.head = nn.Linear(backbone.config.hidden_size, 1) if readout == "scalar" else PointerReadout(backbone.config.hidden_size, head_dim)

    @classmethod
    def create(cls, name, revision=None, max_length=2048, readout="scalar", head_dim=128, input_mode="joint"):
        tok = AutoTokenizer.from_pretrained(name, revision=revision)
        tok.add_special_tokens({"additional_special_tokens": [MARKER]})
        base = AutoModel.from_pretrained(name, revision=revision, attn_implementation="eager")
        base.resize_token_embeddings(len(tok))
        return cls(base, tok, max_length, readout, head_dim, input_mode)

    def encode(self, rows):
        texts = []
        for r in rows:
            parts = ["State: " + r["state"], "Question: " + r["question"]]
            if any(MARKER in x for x in parts + list(r["criteria"]) + list(r["criteria"].values())):
                raise ValueError("Reserved marker in input")
            for key, description in r["criteria"].items():
                parts.append(f"Candidate {key}: {description} {MARKER}")
            texts.append("\n".join(parts))
        inputs = self.tokenizer(texts, padding=True, truncation=False, return_token_type_ids=False, return_tensors="pt")
        if inputs["input_ids"].shape[1] > self.max_length:
            raise ValueError("Input exceeds max_length; refusing silent truncation")
        positions = [(ids == self.marker_id).nonzero().flatten() for ids in inputs["input_ids"]]
        if any(len(pos) != len(r["criteria"]) for pos, r in zip(positions, rows)):
            raise ValueError("Candidate marker count mismatch")
        device = next(self.parameters()).device
        return {k: v.to(device) for k, v in inputs.items()}, positions

    def forward(self, rows):
        encoded_rows = rows
        if self.input_mode == "independent":
            # Separate sequences prevent all cross-option attention, including
            # indirect communication through the state/question representations.
            encoded_rows = [{**row, "criteria": {key: description}}
                            for row in rows for key, description in row["criteria"].items()]
        inputs, positions = self.encode(encoded_rows)
        hidden = self.backbone(**inputs).last_hidden_state
        # Each candidate has one shared scalar scorer, not a fixed class-specific output.
        if self.readout == "pointer":
            # CLS jointly attends to state, question and all options; never a fixed class ID.
            scores = [self.head(hidden[i, 0], hidden[i, pos.to(hidden.device)]) for i, pos in enumerate(positions)]
        else:
            scores = [self.head(hidden[i, pos.to(hidden.device)]).squeeze(-1) for i, pos in enumerate(positions)]
        if self.input_mode == "independent":
            grouped, offset = [], 0
            for row in rows:
                count = len(row["criteria"])
                grouped.append(torch.cat(scores[offset:offset + count]))
                offset += count
            return grouped
        return scores

    def save(self, folder, metadata):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=False)
        self.backbone.save_pretrained(folder / "backbone")
        self.tokenizer.save_pretrained(folder / "backbone")
        from safetensors.torch import save_file
        save_file(self.head.state_dict(), str(folder / "head.safetensors"))
        metadata["max_length"] = self.max_length
        metadata.update(readout=self.readout, head_dim=self.head_dim, input_mode=self.input_mode)
        (folder / "training.json").write_text(json.dumps(metadata, indent=2))

    @classmethod
    def load(cls, folder, attention_impl="eager"):
        folder = Path(folder)
        meta = json.loads((folder / "training.json").read_text())
        model = cls(AutoModel.from_pretrained(folder / "backbone", attn_implementation=attention_impl),
                    AutoTokenizer.from_pretrained(folder / "backbone"), meta["max_length"],
                    meta.get("readout", "scalar"), meta.get("head_dim", 128), meta.get("input_mode", "joint"))
        from safetensors.torch import load_file
        model.head.load_state_dict(load_file(str(folder / "head.safetensors")))
        return model
