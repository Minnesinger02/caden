import tempfile
from pathlib import Path
import unittest

import torch
from transformers import BertConfig, BertModel, BertTokenizerFast
from decision_lab.encoder import CandidateEncoder, MARKER


class PointerTests(unittest.TestCase):
    def test_query_gradient_ragged_options_and_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'vocab.txt').write_text('\n'.join(['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', 'hello', 'one', 'two']), encoding='utf-8')
            tokenizer = BertTokenizerFast(vocab_file=str(root / 'vocab.txt'))
            tokenizer.add_special_tokens({'additional_special_tokens': [MARKER]})
            backbone = BertModel(BertConfig(vocab_size=len(tokenizer), hidden_size=16, num_hidden_layers=1,
                                           num_attention_heads=2, intermediate_size=32, max_position_embeddings=128))
            model = CandidateEncoder(backbone, tokenizer, 128, readout='pointer', head_dim=8)
            row = {'state': 'hello', 'question': 'hello', 'criteria': {'a': 'one', 'b': 'two'}}
            three = {**row, 'criteria': {'a': 'one', 'b': 'two', 'c': 'hello'}}
            logits = model([row, three])
            self.assertEqual([len(z) for z in logits], [2, 3])
            torch.nn.functional.cross_entropy(logits[0][None], torch.tensor([0])).backward()
            for layer in (model.head.query, model.head.key):
                self.assertTrue(torch.isfinite(layer.weight.grad).all())
                self.assertGreater(layer.weight.grad.abs().sum().item(), 0)
            torch.optim.AdamW(model.parameters(), lr=.001).step()
            model.eval()
            with torch.inference_mode():
                expected = model([row, three])
            model.save(root / 'saved', {'test_only': True})
            loaded = CandidateEncoder.load(root / 'saved').eval()
            self.assertEqual(loaded.readout, 'pointer')
            with torch.inference_mode():
                actual = loaded([row, three])
            for a, b in zip(expected, actual):
                torch.testing.assert_close(a, b)
