"""Unit and regression tests for Continual Learning & Catastrophic Forgetting Suite."""

from __future__ import annotations

import math
import shutil
import tempfile
import unittest
from pathlib import Path

from src.craftly.learning.continual import (
    CANONICAL_FORGETTING_SUITE,
    CANONICAL_SEQUENTIAL_DOMAINS,
    EarlyForgettingDetector,
    ElasticWeightConsolidation,
    ExperienceReplayBuffer,
    GeometryAwareLayerAnchor,
    LayerwiseDriftAnalyzer,
    ReferenceModelKLLoss,
    WeightMerger,
    compute_sequential_metrics,
    evaluate_continual_perplexity,
)
from src.craftly.model_ops.foundation import ScratchDecoderConfig
from src.craftly.model_ops.tokenizer_pipeline import TokenizerTrainConfig, load_tokenizer, train_bpe_tokenizer
from src.craftly.model_ops.torch_decoder import (
    CraftlyDecoderLM,
    require_torch,
    save_trusted_checkpoint,
)

try:
    import torch
except ImportError:
    torch = None


class TestContinualLearning(unittest.TestCase):
    def setUp(self) -> None:
        require_torch()
        self.test_dir = Path(tempfile.mkdtemp(prefix="test_continual_"))

        # Create a tiny tokenizer for fast testing
        corpus_path = self.test_dir / "corpus.txt"
        corpus_path.write_text(
            "def quicksort(arr): return arr\n"
            "def binary_search(arr, target): return 0\n"
            "def fib(n): return n\n"
            "SELECT * FROM users WHERE id=%s\n",
            encoding="utf-8",
        )
        self.tokenizer_path = self.test_dir / "tokenizer" / "tokenizer.json"
        train_bpe_tokenizer(
            [corpus_path],
            output_path=self.tokenizer_path,
            config=TokenizerTrainConfig(vocab_size=1000, min_frequency=1),
        )
        self.tokenizer = load_tokenizer(self.tokenizer_path)

        # Tiny model config
        self.tiny_config = ScratchDecoderConfig(
            name="craftly-test-tiny",
            vocab_size=1000,
            hidden_size=64,
            intermediate_size=128,
            num_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_sequence_length=128,
            tie_word_embeddings=True,
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_experience_replay_buffer(self) -> None:
        buffer = ExperienceReplayBuffer(replay_ratio=0.25, seed=42)
        self.assertGreaterEqual(len(buffer.samples), 3)

        inps, lbls = buffer.sample_batch(
            count=3,
            tokenizer=self.tokenizer,
            max_sequence_length=32,
            pad_token_id=0,
            ignore_index=-100,
        )
        self.assertEqual(len(inps), 3)
        self.assertEqual(len(lbls), 3)
        self.assertEqual(len(inps[0]), 32)
        self.assertEqual(len(lbls[0]), 32)
        # Check that targets match inputs up to pad
        non_padded = [t for t in inps[0] if t != 0]
        self.assertGreater(len(non_padded), 0)

    def test_reference_model_kl_loss(self) -> None:
        kl = ReferenceModelKLLoss(temperature=1.0)
        batch, seq_len, vocab = 2, 8, 32
        logits = torch.randn(batch, seq_len, vocab, requires_grad=True)

        # 1. Identical distributions have 0 KL
        loss_zero = kl.compute(logits, logits.detach())
        self.assertAlmostEqual(loss_zero.item(), 0.0, places=4)

        # 2. Different distributions have positive KL
        diff_logits = torch.randn(batch, seq_len, vocab)
        loss_pos = kl.compute(logits, diff_logits)
        self.assertGreater(loss_pos.item(), 0.0)

        # 3. Gradient flows back to active logits
        loss_pos.backward()
        self.assertIsNotNone(logits.grad)
        self.assertGreater(torch.norm(logits.grad).item(), 0.0)

    def test_weight_merger_slerp(self) -> None:
        # Two orthogonal unit vectors in 2D
        v0 = torch.tensor([1.0, 0.0])
        v1 = torch.tensor([0.0, 1.0])

        # At t = 0.5, angle is 45 deg, components should be cos(45) = 1/sqrt(2) ≈ 0.7071
        mid = WeightMerger.slerp(v0, v1, t=0.5)
        self.assertAlmostEqual(mid[0].item(), 1.0 / math.sqrt(2.0), places=3)
        self.assertAlmostEqual(mid[1].item(), 1.0 / math.sqrt(2.0), places=3)
        # Norm should be preserved as 1.0
        self.assertAlmostEqual(torch.norm(mid).item(), 1.0, places=3)

    def test_weight_merger_state_dicts(self) -> None:
        state_a = {"weight": torch.ones(4, 4), "bias": torch.zeros(4)}
        state_b = {"weight": torch.full((4, 4), 3.0), "bias": torch.full((4,), 2.0)}

        merged = WeightMerger.merge_state_dicts(state_a, state_b, method="linear", t=0.5)
        self.assertIn("weight", merged)
        self.assertIn("bias", merged)
        self.assertAlmostEqual(merged["weight"][0, 0].item(), 2.0, places=3)
        self.assertAlmostEqual(merged["bias"][0].item(), 1.0, places=3)

    def test_weight_merger_checkpoint_files(self) -> None:
        base_file = self.test_dir / "base.pt"
        sft_file = self.test_dir / "sft.pt"
        out_file = self.test_dir / "merged.pt"

        model_base = CraftlyDecoderLM(self.tiny_config)
        model_sft = CraftlyDecoderLM(self.tiny_config)
        # Perturb sft weights slightly
        for p in model_sft.parameters():
            p.data.add_(0.05)

        save_trusted_checkpoint({"model": model_base.state_dict(), "config": self.tiny_config.model_dump()}, base_file)
        save_trusted_checkpoint({"model": model_sft.state_dict(), "config": self.tiny_config.model_dump()}, sft_file)

        res_path = WeightMerger.merge_checkpoint_files(base_file, sft_file, out_file, method="slerp", t=0.3)
        self.assertTrue(res_path.exists())

    def test_evaluate_continual_perplexity(self) -> None:
        model = CraftlyDecoderLM(self.tiny_config)
        report = evaluate_continual_perplexity(model, self.tokenizer, CANONICAL_FORGETTING_SUITE[:3], device="cpu")
        self.assertGreater(report.evaluated_tasks_count, 0)
        self.assertGreater(report.perplexity_general, 0.0)
        self.assertGreaterEqual(report.general_retention_score, 0.0)
        self.assertLessEqual(report.general_retention_score, 100.0)

    def test_sequential_domains_and_transfer_metrics(self) -> None:
        self.assertEqual(len(CANONICAL_SEQUENTIAL_DOMAINS), 4)
        domain_names = [d.name for d in CANONICAL_SEQUENTIAL_DOMAINS]

        # Simulated transfer matrix: R[i][j]
        # Stage 0: trained only on A
        # Stage 1: trained on B -> A loss increases (forgetting)
        # Stage 2: trained on C -> A, B losses increase
        # Stage 3: trained on D -> A, B, C losses increase
        mock_matrix = [
            [1.20, 2.80, 3.10, 3.50],
            [1.50, 1.10, 2.90, 3.20],
            [1.80, 1.40, 0.95, 3.00],
            [2.10, 1.70, 1.25, 0.85],
        ]
        metrics = compute_sequential_metrics(mock_matrix, domain_names)
        self.assertEqual(len(metrics.transfer_matrix), 4)
        # BWT should be negative reflecting degradation
        self.assertLess(metrics.backward_transfer, 0.0)
        self.assertGreater(metrics.average_forgetting, 0.0)
        self.assertIn(domain_names[0], metrics.retention_rates)
        self.assertIn(domain_names[3], metrics.per_domain_losses)

    def test_layerwise_drift_analyzer(self) -> None:
        model_base = CraftlyDecoderLM(self.tiny_config)
        model_curr = CraftlyDecoderLM(self.tiny_config)
        # Perturb attention parameters specifically
        with torch.no_grad():
            for name, param in model_curr.named_parameters():
                if "attn" in name or "q_proj" in name:
                    param.add_(0.15)
                elif "mlp" in name:
                    param.add_(0.05)

        report = LayerwiseDriftAnalyzer.compute_drift(model_base.state_dict(), model_curr.state_dict())
        self.assertGreater(report.mean_total_drift, 0.0)
        self.assertGreater(report.attention_drift, report.ffn_drift)
        self.assertIn(report.max_drift_parameter, report.layer_drifts)

    def test_early_forgetting_detector(self) -> None:
        detector = EarlyForgettingDetector(drift_threshold=0.05, velocity_threshold=0.02)
        base_state = {"param": torch.ones(4, 4)}
        curr_state_low = {"param": torch.ones(4, 4) + 0.01}
        curr_state_high = {"param": torch.ones(4, 4) + 0.35}

        drift_low = LayerwiseDriftAnalyzer.compute_drift(base_state, curr_state_low)
        signal_low = detector.evaluate_step(step=50, drift_report=drift_low, reference_probe_loss=1.2)
        self.assertFalse(signal_low.warning_triggered)
        self.assertEqual(signal_low.risk_level, "LOW")

        drift_high = LayerwiseDriftAnalyzer.compute_drift(base_state, curr_state_high)
        signal_high = detector.evaluate_step(step=100, drift_report=drift_high, reference_probe_loss=2.5)
        self.assertTrue(signal_high.warning_triggered)
        self.assertIn(signal_high.risk_level, ["MODERATE", "CRITICAL"])

    def test_elastic_weight_consolidation(self) -> None:
        model = CraftlyDecoderLM(self.tiny_config)
        samples = ["def test_func(): return 42", "class Sample: pass"]
        fisher = ElasticWeightConsolidation.estimate_diagonal_fisher(model, samples, self.tokenizer, device="cpu")
        self.assertGreater(len(fisher), 0)

        ewc = ElasticWeightConsolidation(ewc_lambda=50.0)
        opt_state = {k: v.clone() for k, v in model.state_dict().items()}
        # Zero perturbation = 0 penalty
        pen_zero = ewc.compute_penalty(model, opt_state, fisher)
        self.assertAlmostEqual(pen_zero.item(), 0.0, places=4)

        # Perturb model slightly -> positive penalty
        with torch.no_grad():
            for p in model.parameters():
                p.add_(0.05)
        pen_pos = ewc.compute_penalty(model, opt_state, fisher)
        self.assertGreater(pen_pos.item(), 0.0)

    def test_geometry_aware_layer_anchor(self) -> None:
        gd = GeometryAwareLayerAnchor(base_replay_ratio=0.15, max_replay_ratio=0.40, anchor_stiffness=0.5)
        w_embed = gd.compute_layer_weight("embedding.weight")
        w_early = gd.compute_layer_weight("layers.0.mlp.down_proj.weight")
        w_late = gd.compute_layer_weight("layers.3.mlp.down_proj.weight")

        # Earlier layers must have higher stiffness than late layers
        self.assertGreater(w_embed, w_late)
        self.assertGreater(w_early, w_late)

        # Dynamic replay scale with velocity
        r_low = gd.get_dynamic_replay_ratio(drift_velocity=0.01)
        r_high = gd.get_dynamic_replay_ratio(drift_velocity=0.20)
        self.assertGreater(r_high, r_low)
        self.assertLessEqual(r_high, 0.40)


if __name__ == "__main__":
    unittest.main()
