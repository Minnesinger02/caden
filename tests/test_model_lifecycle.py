"""Offline tests of the public training/inference paths, using a tiny local encoder."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch
from transformers import BertConfig, BertModel, BertTokenizerFast

from caden import Caden
from caden.data import read_jsonl, validate_row
from caden.infer import main as infer
from caden.model import CandidateEncoder
from caden.train import main as train


def request():
    return {"id": "one", "state": "replace my card", "question": "which intent",
            "criteria": {"replace": "replace a lost card", "transfer": "transfer money"}}


class ModelLifecycleTests(unittest.TestCase):
    def test_reject_nontraining_and_duplicate_training_ids(self):
        row = {**request(), "label": "replace"}
        with self.assertRaisesRegex(ValueError, "split=train"):
            validate_row(row, training=True)
        row["split"] = "train"
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "bad.jsonl"
            p.write_text((json.dumps(row) + "\n") * 2, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                read_jsonl(p, training=True)

    def test_invalid_candidate_and_temperature_rejected(self):
        with self.assertRaises(ValueError):
            validate_row({**request(), "criteria": {"only": "one"}})
        for value in [0, -1, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                Caden(None, value)

    def test_fresh_training_continuation_and_inference_cli(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "base"
            base.mkdir()
            words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "state", "question",
                     "candidate", ":", "replace", "my", "card", "which", "intent", "a", "lost",
                     "transfer", "money", "weather", "get", "forecast"]
            (base / "vocab.txt").write_text("\n".join(words) + "\n", encoding="utf-8")
            tokenizer = BertTokenizerFast(vocab_file=str(base / "vocab.txt"))
            tokenizer.save_pretrained(base)
            BertModel(BertConfig(vocab_size=len(tokenizer), hidden_size=16,
                                 num_hidden_layers=1, num_attention_heads=2,
                                 intermediate_size=32, max_position_embeddings=128)).save_pretrained(base)
            row = {**request(), "split": "train", "label": "replace"}
            data = root / "train.jsonl"
            data.write_text(json.dumps(row) + "\n", encoding="utf-8")
            first = root / "first"
            with contextlib.redirect_stdout(io.StringIO()):
                train(["--data", str(data), "--output", str(first), "--model", str(base),
                       "--device", "cpu", "--epochs", "1", "--max-length", "128"])
            original = CandidateEncoder.load(first).head.weight.detach().clone()
            (first / "calibration.json").write_text('{"temperature":2.0}', encoding="utf-8")
            self.assertEqual(Caden.from_pretrained(str(first), device="cpu").temperature, 2.0)
            second = root / "second"
            with contextlib.redirect_stdout(io.StringIO()):
                train(["--data", str(data), "--output", str(second), "--init-checkpoint", str(first),
                       "--device", "cpu", "--epochs", "1", "--lr", "0.001"])
            continued = Caden.from_pretrained(str(second), device="cpu")
            self.assertEqual(continued.temperature, 1.0)
            self.assertFalse((second / "calibration.json").exists())
            self.assertFalse(torch.equal(original, continued.encoder.head.weight.detach()))
            third = {**request(), "id": "two", "criteria": {**request()["criteria"], "weather": "get forecast"}}
            outputs = continued.predict_many([request(), third], batch_size=2)
            for row, answer in zip([request(), third], outputs):
                self.assertEqual(set(answer["probabilities"]), set(row["criteria"]))
                self.assertIn(answer["choice"], row["criteria"])
                self.assertAlmostEqual(sum(answer["probabilities"].values()), 1.0, places=5)
            requests = root / "requests.jsonl"
            requests.write_text(json.dumps(request()) + "\n" + json.dumps(third) + "\n", encoding="utf-8")
            predictions = root / "predictions.jsonl"
            infer(["--model", str(second), "--device", "cpu", "--input", str(requests),
                   "--output", str(predictions), "--batch-size", "2"])
            records = [json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(records, outputs)
            from decision_lab.encoder import CandidateEncoder as CompatibleEncoder
            self.assertIs(CompatibleEncoder, CandidateEncoder)
            with self.assertRaisesRegex(ValueError, "exceeds max_length"):
                continued.predict({**request(), "state": "card " * 200})
            with self.assertRaisesRegex(ValueError, "Reserved marker"):
                continued.predict({**request(), "state": "<|decision_candidate_end|>"})


if __name__ == "__main__":
    unittest.main()
