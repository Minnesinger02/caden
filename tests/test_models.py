"""No downloads: small randomly initialized architectures, not accuracy benchmarks."""
import importlib.util
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"), "Install model dependencies")
class ModelTests(unittest.TestCase):
    def test_projection_parity_with_bias_and_gradients(self):
        import torch
        from dynamic_candidates.experiment import readout
        torch.manual_seed(42)
        head = torch.nn.Linear(8, 101)
        h = torch.randn(8)
        a = readout(h, head, [0, 13, 100], "full")
        b = readout(h, head, [0, 13, 100], "restricted")
        torch.testing.assert_close(a, b)
        torch.testing.assert_close(a.softmax(-1), b.softmax(-1))

    def test_qwen_prefill_readout_matches_lm_wrapper(self):
        import torch
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from dynamic_candidates.experiment import readout
        config = Qwen3Config(vocab_size=101, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, head_dim=16)
        model = Qwen3ForCausalLM(config).eval()
        ids = torch.tensor([[1, 2, 3, 4]])
        with torch.inference_mode():
            expected = model(ids, use_cache=False).logits[0, -1, [4, 7, 23]]
            hidden = model.model(ids, use_cache=False).last_hidden_state[0, -1]
            actual = readout(hidden, model.get_output_embeddings(), [4, 7, 23], "restricted")
        torch.testing.assert_close(expected, actual)

    def test_encoder_train_save_reload_and_overflow(self):
        import torch
        from transformers import BertConfig, BertModel, BertTokenizerFast
        from decision_lab.encoder import CandidateEncoder, MARKER
        torch.manual_seed(42)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "vocab.txt").write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "a", "b", "hello"]))
            tok = BertTokenizerFast(vocab_file=str(root / "vocab.txt"))
            tok.add_special_tokens({"additional_special_tokens": [MARKER]})
            base = BertModel(BertConfig(vocab_size=len(tok), hidden_size=16, num_hidden_layers=1,
                                         num_attention_heads=2, intermediate_size=32, max_position_embeddings=128))
            model = CandidateEncoder(base, tok, 128)
            r = {"state": "hello", "question": "hello", "criteria": {"a": "hello", "b": "b"}, "label": "a"}
            r3 = {**r, "criteria": {"a": "hello", "b": "b", "c": "hello"}}
            logits = model([r, r3])
            self.assertEqual([len(z) for z in logits], [2, 3])
            loss = torch.nn.functional.cross_entropy(logits[0][None], torch.tensor([0]))
            loss.backward()
            self.assertTrue(torch.isfinite(model.head.weight.grad).all())
            before = model.head.weight.detach().clone()
            torch.optim.AdamW(model.parameters(), lr=.001).step()
            self.assertFalse(torch.equal(before, model.head.weight))
            model.eval()
            with torch.inference_mode():
                expected = model([r])[0]
            model.save(root / "saved", {"test_only": True})
            reloaded = CandidateEncoder.load(root / "saved").eval()
            with torch.inference_mode():
                torch.testing.assert_close(expected, reloaded([r])[0])
            model.max_length = 2
            with self.assertRaises(ValueError):
                model([r])

    def test_modernbert_forward(self):
        import torch
        from transformers import ModernBertConfig, ModernBertModel, BertTokenizerFast
        from decision_lab.encoder import CandidateEncoder, MARKER
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "vocab.txt"
            path.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello"]))
            tok = BertTokenizerFast(vocab_file=str(path))
            tok.add_special_tokens({"additional_special_tokens": [MARKER]})
            config = ModernBertConfig(vocab_size=len(tok), hidden_size=32, intermediate_size=64,
                num_hidden_layers=2, num_attention_heads=4, max_position_embeddings=128,
                local_attention=32, pad_token_id=0, cls_token_id=2, sep_token_id=3, reference_compile=False)
            config._attn_implementation = "eager"
            model = CandidateEncoder(ModernBertModel(config), tok, 128).eval()
            with torch.inference_mode():
                z = model([{"state": "hello", "question": "hello", "criteria": {"a": "hello", "b": "hello"}}])[0]
            self.assertEqual(z.shape, (2,))
            self.assertTrue(torch.isfinite(z).all())


if __name__ == "__main__":
    unittest.main()
