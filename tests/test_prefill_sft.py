import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from prefill_renorm_sft.data import convert


class DataTests(unittest.TestCase):
    def fixture(self):
        return {"id": "doc", "split": "train", "request": {"state": "hello", "questions": {
            "q": {"type": "choice", "instructions": "select", "criteria": {"a": "one", "b": "two"}}}}}

    def test_inputs_without_supervision_rejected(self):
        with self.assertRaises(ValueError):
            convert(self.fixture())

    def test_gold_and_teacher_are_distinguished(self):
        record = self.fixture()
        record["response"] = {"model": "synthetic-fixture", "answers": {"q": {"probabilities": {"a": .2, "b": .8}}}}
        self.assertEqual(convert(record)[0]["label_source"], "teacher")
        record["gold"] = {"q": "a"}
        row = convert(record)[0]
        self.assertEqual(row["label"], "a")
        self.assertEqual(row["label_source"], "gold")
        self.assertEqual(row["teacher_probabilities"]["b"], .8)


@unittest.skipUnless(all(importlib.util.find_spec(m) for m in ["torch", "transformers", "peft"]), "Install sft extra")
class SFTTests(unittest.TestCase):
    def test_mixed_loss_alignment_and_gold_requirement(self):
        import torch
        from prefill_renorm_sft.model import objective
        torch.manual_seed(19)
        head = torch.nn.Linear(4, 9)
        model = SimpleNamespace(get_output_embeddings=lambda: head)
        h = torch.randn(4, requires_grad=True)
        row = {"criteria": {"a": "one", "b": "two"}, "label": "a",
               "label_source": "gold", "teacher_probabilities": {"b": .8, "a": .2}}
        ce = objective(model, h, [2, 5], row, "candidate_ce")
        kl = objective(model, h, [2, 5], row, "teacher_kl")
        mixed = objective(model, h, [2, 5], row, "gold_kl", .3)
        torch.testing.assert_close(mixed, .3 * ce + .7 * kl)
        reversed_row = {**row, "criteria": {"b": "two", "a": "one"}}
        torch.testing.assert_close(mixed, objective(model, h, [5, 2], reversed_row, "gold_kl", .3))
        mixed.backward()
        self.assertTrue(torch.isfinite(h.grad).all())
        self.assertGreater(h.grad.abs().sum().item(), 0)
        for bad_row in [{**row, "label_source": "teacher"},
                        {**row, "teacher_probabilities": {"a": .2, "c": .8}}]:
            with self.assertRaises(ValueError):
                objective(model, h, [2, 5], bad_row, "gold_kl")
        with self.assertRaises(ValueError):
            objective(model, h, [2, 5], row, "gold_kl", float("nan"))

    def test_lora_training_save_merge_evaluate_offline(self):
        import torch
        from transformers import Qwen3Config, Qwen3ForCausalLM, PreTrainedTokenizerFast
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import WhitespaceSplit
        from prefill_renorm_sft.__main__ import run
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            words = ["[UNK]", "[PAD]", "[EOS]", "A", "B", "Answer:", "hello", "one", "two"]
            raw = Tokenizer(WordLevel({w: i for i, w in enumerate(words)}, unk_token="[UNK]"))
            raw.pre_tokenizer = WhitespaceSplit()
            tok = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]")
            tok.chat_template = "{% for message in messages %}{{message['content']}}{% endfor %}\n"
            model_path = root / "base"
            model = Qwen3ForCausalLM(Qwen3Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64,
                num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, head_dim=16, max_position_embeddings=256))
            model.save_pretrained(model_path)
            tok.save_pretrained(model_path)
            record = {"id": "train-1", "split": "train", "state": "hello", "question": "hello",
                      "criteria": {"a": "one", "b": "two"}, "label": "a", "label_source": "gold",
                      "teacher_probabilities": {"a": .7, "b": .3}}
            data = root / "train.jsonl"
            data.write_text(json.dumps(record) + "\n")
            common = dict(data=str(data), model=str(model_path), revision=None, device="cpu", dtype="float32", seed=42, max_length=256)
            # Exercise each objective, real optimizer updates and PEFT serialization.
            for objective in ["token_sft", "candidate_ce", "teacher_kl", "gold_kl"]:
                out = root / objective
                run(SimpleNamespace(**common, command="train", out=str(out), objective=objective,
                                    accumulation=2, epochs=1, lr=.001, rank=2, gold_weight=.5))
                from safetensors.torch import load_file
                weights = load_file(str(out / "adapter_model.safetensors"))
                self.assertTrue(any(v.abs().sum() > 0 for k, v in weights.items() if "lora_B" in k))
            record.update(id="eval-1", split="smoke")
            data.write_text(json.dumps(record) + "\n")
            run(SimpleNamespace(**common, command="evaluate", out=str(root / "eval"),
                                adapter=str(root / "token_sft"), warmup=1))
            metrics = json.loads((root / "eval" / "metrics.json").read_text())
            self.assertEqual(metrics["valid"], 1)
            self.assertEqual(metrics["failures"], 0)


if __name__ == "__main__":
    unittest.main()
