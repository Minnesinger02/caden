"""Skip unused LM projections while retaining upstream masks, caches and renorm."""
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
MANIFEST = json.loads((ROOT / "upstream.json").read_text())
UPSTREAM = ROOT.parent / "vendor" / MANIFEST["directory"]
for filename, expected in MANIFEST["source_sha256"].items():
    if hashlib.sha256((UPSTREAM / filename).read_bytes()).hexdigest() != expected:
        raise RuntimeError(f"Pinned upstream was modified: {filename}")
sys.path.insert(0, str(UPSTREAM / "src"))

from litjev.backend import TransformersScorer  # noqa: E402


class CandidateLogits:
    """Lazy projection implementing only the indexing used by upstream _score."""
    def __init__(self, hidden, head, audit):
        self.hidden = hidden
        self.head = head
        self.audit = audit

    def __getitem__(self, index):
        row, position, candidates = index
        if not isinstance(row, int) or not isinstance(position, int):
            raise TypeError("Only a single selected position per question is supported")
        ids = torch.as_tensor(candidates, device=self.head.weight.device, dtype=torch.long)
        h = self.hidden[row, position]
        if h.device != self.head.weight.device:
            raise ValueError("This experiment requires backbone and output head on one device")
        bias = self.head.bias[ids] if self.head.bias is not None else None
        z = F.linear(h, self.head.weight[ids], bias)
        if self.audit is not None:
            self.audit.append({"position": position, "candidates": ids.numel(), "projected_elements": z.numel()})
        return z


class CandidateOnlyLM(nn.Module):
    """Qwen2/Qwen3 text-only wrapper. No weight changes or output-class changes."""
    def __init__(self, lm, audit=None):
        super().__init__()
        if lm.config.model_type not in {"qwen2", "qwen3"}:
            raise ValueError("Optimization verified only for Qwen2/Qwen3 text backbones")
        if not isinstance(lm.get_output_embeddings(), nn.Linear):
            raise ValueError("Requires an ordinary linear output head, not a quantized/custom head")
        if len({p.device for p in lm.parameters()}) != 1:
            raise ValueError("Sharded/offloaded models are not supported by this pilot")
        self.wrapped = lm
        self.config = lm.config
        self.audit = audit

    def get_input_embeddings(self):
        return self.wrapped.get_input_embeddings()

    def forward(self, **kwargs):
        kwargs.pop("logits_to_keep", None)
        output = self.wrapped.model(**kwargs, return_dict=True)
        return SimpleNamespace(past_key_values=output.past_key_values,
            hidden_states=output.hidden_states,
            logits=CandidateLogits(output.last_hidden_state, self.wrapped.get_output_embeddings(), self.audit))

    def generate(self, *args, **kwargs):
        raise NotImplementedError("Only the System One score path is optimized; thinking/generation is excluded")


def optimized_scorer(model, tokenizer, audit=None, **kwargs):
    # Upstream _score itself is unmodified. This preserves branch layout and caches.
    return TransformersScorer(CandidateOnlyLM(model, audit), tokenizer, **kwargs)
