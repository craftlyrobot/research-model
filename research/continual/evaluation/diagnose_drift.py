"""Layer-wise Parameter Displacement and Mechanistic Forgetting Diagnostic CLI.

Analyzes relative Frobenius norm drift between base model and fine-tuned checkpoints:
||theta_t - theta_0||_F / (||theta_0||_F + eps)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.craftly.learning.continual import LayerwiseDriftAnalyzer
from src.craftly.model_ops.sft_loop import resolve_base_model_assets


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Layer-wise Weight Displacement.")
    parser.add_argument("--base", required=True, help="Base model checkpoint path")
    parser.add_argument("--current", required=True, help="Fine-tuned or sequential stage checkpoint path")
    parser.add_argument("--output", default="artifacts/craftly/layer_drift_report.json", help="Output JSON path")
    args = parser.parse_args()

    _, base_state, _ = resolve_base_model_assets(args.base)
    _, curr_state, _ = resolve_base_model_assets(args.current)

    report = LayerwiseDriftAnalyzer.compute_drift(base_state, curr_state)

    print("=" * 75)
    print("CRAFTLY MECHANISTIC LAYER-WISE PARAMETER DRIFT ANALYSIS")
    print("=" * 75)
    print(f"{'Layer Component':<30} | {'Relative Frobenius Drift':<25}")
    print("-" * 75)
    print(f"{'Token Embeddings':<30} | {report.embedding_drift:<25.6f}")
    print(f"{'Self-Attention (QKVO)':<30} | {report.attention_drift:<25.6f}")
    print(f"{'Feed-Forward / MLP':<30} | {report.ffn_drift:<25.6f}")
    print(f"{'Language Model Head':<30} | {report.head_drift:<25.6f}")
    print(f"{'Global Mean Drift':<30} | {report.mean_total_drift:<25.6f}")
    print("=" * 75)
    print(f"Maximal Drift Parameter: {report.max_drift_parameter}")
    print(f"Maximal Drift Value:     {report.max_drift_value:.6f}")
    print("=" * 75)

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report.model_dump(mode="json"), indent=2), encoding="utf-8")
    print(f"[Drift report saved] -> {out_path}\n")


if __name__ == "__main__":
    main()
