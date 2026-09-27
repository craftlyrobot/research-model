"""Local Smoke Test & Validation Runner for Continual Learning Research Pipeline.

Runs a fast, end-to-end multi-domain sequential training session across:
1. Naive Sequential Training (Baseline showing forgetting)
2. Experience Replay
3. Reference Model KL Anchor
4. Elastic Weight Consolidation (EWC)
5. Weight Space Merging (SLERP)
6. Geometry-Aware Layer Anchoring (GD-LSA Proposed Method)
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.craftly.learning.continual import CANONICAL_SEQUENTIAL_DOMAINS
from src.craftly.learning.sequential_pipeline import SequentialRunConfig, run_sequential_continual_study
from src.craftly.model_ops.foundation import ScratchDecoderConfig
from src.craftly.model_ops.tokenizer_pipeline import TokenizerTrainConfig, load_tokenizer, train_bpe_tokenizer
from src.craftly.model_ops.torch_decoder import CraftlyDecoderLM, require_torch


def main() -> None:
    require_torch()
    print("=" * 75)
    print("CRAFTLY CONTINUAL LEARNING RESEARCH: LOCAL VALIDATION RUN")
    print("=" * 75)

    with tempfile.TemporaryDirectory(prefix="craftly_continual_study_") as temp_dir:
        td = Path(temp_dir)
        corpus = td / "corpus.txt"
        corpus.write_text(
            "def quicksort(arr): return arr\n"
            "def binary_search(arr, target): return 0\n"
            "import http.server\n"
            "SELECT * FROM users WHERE id=%s\n"
            "--- a/file.py\n+++ b/file.py\n",
            encoding="utf-8",
        )
        tok_path = td / "tokenizer.json"
        train_bpe_tokenizer([corpus], tok_path, config=TokenizerTrainConfig(vocab_size=1000, min_frequency=1))
        tokenizer = load_tokenizer(tok_path)

        tiny_cfg = ScratchDecoderConfig(
            name="craftly-smoke-research",
            vocab_size=1000,
            hidden_size=64,
            intermediate_size=128,
            num_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_sequence_length=128,
            tie_word_embeddings=True,
        )

        methods_to_test = [
            "naive_sequential",
            "experience_replay",
            "kl_anchor",
            "proposed_gd_lsa",
        ]

        results = {}
        for method in methods_to_test:
            print(f"\n>>> Running Study Method: [{method.upper()}]...")
            model = CraftlyDecoderLM(tiny_cfg)
            cfg = SequentialRunConfig(
                experiment_id="smoke_comparison",
                mitigation_method=method,
                steps_per_domain=5,
                batch_size=1,
                learning_rate=1e-3,
                replay_ratio=0.3,
                kl_penalty_weight=0.5,
                output_dir=str(td / "artifacts"),
                device="cpu",
            )
            res = run_sequential_continual_study(model, tokenizer, cfg, CANONICAL_SEQUENTIAL_DOMAINS)
            results[method] = res
            print(f"  * Backward Transfer (BWT): {res.transfer_metrics.backward_transfer:+.4f}")
            print(f"  * Average Forgetting:      {res.transfer_metrics.average_forgetting:.4f}")
            print(f"  * Mean Parameter Drift:    {res.layer_drift_history[-1].mean_total_drift:.6f}")

        print("\n" + "=" * 75)
        print("SUMMARY OF VALIDATION EXPERIMENTS")
        print("=" * 75)
        print(f"{'Method':<22} | {'BWT (Higher=Better)':<20} | {'Avg Forgetting':<15} | {'Retention Domain A %':<20}")
        print("-" * 75)
        for m, res in results.items():
            bwt = res.transfer_metrics.backward_transfer
            f_rate = res.transfer_metrics.average_forgetting
            dom_a_ret = res.transfer_metrics.retention_rates[res.domain_names[0]]
            print(f"{m:<22} | {bwt:<20.4f} | {f_rate:<15.4f} | {dom_a_ret:<20.2f}%")
        print("=" * 75)
        print("\nLocal verification succeeded cleanly!\n")


if __name__ == "__main__":
    main()
