"""Load a local checkpoint or a fixed Hub release and score candidates."""
import json
import math
from pathlib import Path

import torch

from .data import validate_row
from .model import CandidateEncoder

MODEL_ID = "Leonard02/caden-encoder-mixed"
MODEL_REVISION = "f1ab177667e47605d12e84a13f920002add0eff3"


def resolve_checkpoint(model=MODEL_ID, revision=None, seed=42):
    path = Path(model)
    if path.is_dir():
        if (path / "training.json").is_file():
            return path
        path = path / f"seed-{seed}"
    else:
        from huggingface_hub import snapshot_download
        revision = revision or (MODEL_REVISION if model == MODEL_ID else None)
        path = Path(snapshot_download(model, revision=revision,
                                     allow_patterns=[f"seed-{seed}/**"])) / f"seed-{seed}"
    if not (path / "training.json").is_file():
        raise ValueError(f"No Caden checkpoint at {path}")
    return path


def resolve_device(device):
    return ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device


class Caden:
    def __init__(self, encoder, temperature=1.0):
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Temperature must be finite and positive")
        self.encoder = encoder.eval()
        self.temperature = temperature

    @classmethod
    def from_pretrained(cls, model=MODEL_ID, revision=None, seed=42,
                        device="auto", temperature=None):
        folder = resolve_checkpoint(model, revision, seed)
        if temperature is None:
            calibration = folder / "calibration.json"
            temperature = (json.loads(calibration.read_text(encoding="utf-8"))["temperature"]
                           if calibration.is_file() else 1.0)
        encoder = CandidateEncoder.load(folder).float().to(resolve_device(device))
        return cls(encoder, float(temperature))

    def predict_many(self, rows, batch_size=1):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        for row in rows:
            validate_row(row)
        answers = []
        with torch.inference_mode():
            for start in range(0, len(rows), batch_size):
                batch = rows[start:start + batch_size]
                for row, scores in zip(batch, self.encoder(batch)):
                    if not torch.isfinite(scores).all():
                        raise ValueError("Nonfinite model scores")
                    normalized = (scores / self.temperature).softmax(-1)
                    if not torch.isfinite(normalized).all():
                        raise ValueError("Temperature produced nonfinite probabilities")
                    values = normalized.cpu().tolist()
                    probabilities = dict(zip(row["criteria"], values))
                    answer = {"choice": max(probabilities, key=probabilities.get),
                              "probabilities": probabilities}
                    if "id" in row:
                        answer["id"] = row["id"]
                    answers.append(answer)
        return answers

    def predict(self, row):
        return self.predict_many([row])[0]
