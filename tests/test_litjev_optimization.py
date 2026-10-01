import importlib.util
import unittest


@unittest.skipUnless(all(importlib.util.find_spec(m) for m in ["torch", "transformers", "pydantic", "PIL"]), "Install litjev-audit extra")
class LitJevOptimizationTests(unittest.TestCase):
    def setUp(self):
        import torch
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from prefill_renorm_sft.litjev_optimization.benchmark import SyntheticTokenizer
        torch.manual_seed(8)
        config = Qwen3Config(vocab_size=512, hidden_size=32, intermediate_size=64,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, head_dim=16,
            max_position_embeddings=4096)
        config._attn_implementation = "eager"
        self.model = Qwen3ForCausalLM(config).eval()
        self.tok = SyntheticTokenizer()

    def schema(self):
        from litjev.schema import Choice, DecisionSchema, Noul, Score
        return DecisionSchema({
            "short": Choice(instructions="Pick.", criteria={"a": "First", "b": "Second"}),
            "long": Choice(instructions="Choose one carefully, without an explanation.",
                           criteria={"x": "A very long description", "y": "Other", "z": "None"}),
            "boolean": Noul(instructions="Is it urgent?"),
            "score": Score(instructions="Rate urgency.", criteria=["Low", "Medium", "High"]),
        })

    def test_ragged_typed_branches_probabilities_features_and_repeat(self):
        from prefill_renorm_sft.litjev_optimization.adapter import TransformersScorer, optimized_scorer
        from prefill_renorm_sft.litjev_optimization.benchmark import compare
        baseline = TransformersScorer(self.model, self.tok, feature_layers=(-1,))
        optimized = optimized_scorer(self.model, self.tok, feature_layers=(-1,))
        schema = self.schema()
        lengths = baseline._compile("state", schema).slot_ids
        self.assertGreater(len(set(map(len, lengths))), 1)
        for _ in range(2):
            result = compare(baseline, optimized, "state", schema)
            self.assertLess(result["max_logit_abs_error"], 1e-5)

    def test_no_full_head_projection_and_exactly_two_backbone_calls(self):
        from prefill_renorm_sft.litjev_optimization.adapter import optimized_scorer
        head_calls, body_calls, audit = [], [], []
        h1 = self.model.lm_head.register_forward_hook(lambda *args: head_calls.append(True))
        h2 = self.model.model.register_forward_hook(lambda *args: body_calls.append(True))
        try:
            optimized_scorer(self.model, self.tok, audit=audit).score("state", self.schema())
        finally:
            h1.remove()
            h2.remove()
        self.assertEqual(head_calls, [])
        self.assertEqual(len(body_calls), 2)
        self.assertEqual(sum(x["projected_elements"] for x in audit), 10)

    def test_batched_question_matches_alone(self):
        import numpy as np
        from litjev.schema import DecisionSchema
        from prefill_renorm_sft.litjev_optimization.adapter import optimized_scorer
        scorer = optimized_scorer(self.model, self.tok)
        schema = self.schema()
        together = scorer.score("state", schema)
        for r in together:
            alone = scorer.score("state", DecisionSchema({r.name: schema[r.name]}))[0]
            np.testing.assert_allclose(r.logits, alone.logits, atol=1e-6)

    def test_unsupported_backbone_rejected(self):
        from prefill_renorm_sft.litjev_optimization.adapter import CandidateOnlyLM
        self.model.config.model_type = "qwen3_5"
        with self.assertRaises(ValueError):
            CandidateOnlyLM(self.model)


if __name__ == "__main__":
    unittest.main()
