"""Structural option independence and checkpoint identity, without network/GPU."""
import tempfile
from pathlib import Path
import unittest
import torch
from transformers import BertConfig, BertModel, BertTokenizerFast
from decision_lab.encoder import CandidateEncoder, MARKER


class IndependentContextTests(unittest.TestCase):
    def test_other_candidates_cannot_change_logits_and_checkpoint_restores_mode(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'vocab.txt').write_text('\n'.join(['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', 'hello', 'one', 'two', 'three']))
            tokenizer = BertTokenizerFast(vocab_file=str(root / 'vocab.txt'))
            tokenizer.add_special_tokens({'additional_special_tokens': [MARKER]})
            backbone = BertModel(BertConfig(vocab_size=len(tokenizer), hidden_size=16, num_hidden_layers=1,
                                            num_attention_heads=2, intermediate_size=32, max_position_embeddings=128))
            model = CandidateEncoder(backbone, tokenizer, 128, input_mode='independent').eval()
            row = {'state': 'hello', 'question': 'hello', 'criteria': {'a': 'one', 'b': 'two'}}
            changed = {**row, 'criteria': {'b': 'two', 'c': 'three hello one', 'a': 'one'}}
            scores = model([row, changed])
            torch.testing.assert_close(scores[0], scores[1][torch.tensor([2, 0])], atol=1e-6, rtol=1e-5)
            torch.nn.functional.cross_entropy(scores[0][None], torch.tensor([0])).backward()
            self.assertTrue(torch.isfinite(model.head.weight.grad).all())
            self.assertGreater(model.head.weight.grad.abs().sum().item(), 0)
            model.save(root / 'saved', {'test_only': True})
            restored = CandidateEncoder.load(root / 'saved').eval()
            self.assertEqual(restored.input_mode, 'independent')
            with torch.inference_mode():
                torch.testing.assert_close(scores[0].detach(), restored([row])[0])


if __name__ == '__main__':
    unittest.main()
