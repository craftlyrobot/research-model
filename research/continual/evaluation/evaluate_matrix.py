"""Evaluation CLI for Multi-Domain Continual Learning & Catastrophic Forgetting Matrix.

Computes Backward Transfer (BWT), Forward Transfer (FWT), Per-Domain Perplexity,
and writes publication-ready summary tables.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.craftly.learning.continual import (
    CANONICAL_SEQUENTIAL_DOMAINS,
    compute_sequential_metrics,
)
from src.craftly.learning.sequential_pipeline import evaluate_domain_loss
from src.craftly.model_ops.foundation import ScratchDecoderConfig
from src.craftly.model_ops.sft_loop import resolve_base_model_assets
from src.craftly.model_ops.tokenizer_pipeline import load_tokenizer
from src.craftly.model_ops.torch_decoder import (
    CraftlyDecoderLM,
    require_torch,
    select_torch_device,
)


def load_model(checkpoint_path: Path, device: str) -> CraftlyDecoderLM:
    require_torch()
    _, state_dict, config_dict = resolve_base_model_assets(checkpoint_path)
    arch_config = ScratchDecoderConfig(**config_dict)
    model = CraftlyDecoderLM(arch_config)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Continual Learning Transfer Matrix.")
    parser.add_argument("--checkpoints", nargs="+", required=True, help="List of stage checkpoints in order (Stage 0..T-1)")
    parser.add_argument("--tokenizer", default="config/tokenizer.json", help="Path to tokenizer.json")
    parser.add_argument("--output", default="artifacts/craftly/continual_matrix_report.json", help="Output JSON path")
    parser.add_argument("--device", default="auto", help="PyTorch device ('auto', 'cpu', 'cuda')")
    args = parser.parse_args()

    dev = select_torch_device(args.device)
    tok_path = Path(args.tokenizer).resolve()
    tokenizer = load_tokenizer(tok_path)

    domains = CANONICAL_SEQUENTIAL_DOMAINS
    domain_names = [d.name for d in domains]
    num_stages = len(args.checkpoints)

    print("=" * 80)
    print("CRAFTLY SEQUENTIAL CONTINUAL LEARNING BENCHMARK")
    print(f"Device: {dev.type.upper()} | Stages: {num_stages} | Evaluated Domains: {len(domains)}")
    print("=" * 80)

    transfer_matrix: list[list[float]] = []

    for stage_idx, ckpt_str in enumerate(args.checkpoints):
        ckpt_p = Path(ckpt_str).resolve()
        print(f"\n[Stage {stage_idx + 1}/{num_stages}] Evaluating Checkpoint: {ckpt_p.name}...")
        model = load_model(ckpt_p, str(dev))

        row: list[float] = []
        for d in domains:
            loss_val = evaluate_domain_loss(model, d, tokenizer, device=str(dev))
            row.append(loss_val)
            print(f"  - {d.name:<45} Loss: {loss_val:.4f}")

        transfer_matrix.append(row)
        del model

    # Compute BWT, FWT, and Forgetting metrics
    # If the number of checkpoints matches domain count, we can calculate standard transfer matrix
    if len(transfer_matrix) == len(domains):
        metrics = compute_sequential_metrics(transfer_matrix, domain_names)
        print("\n" + "=" * 80)
        print(f"{'Domain':<45} | {'Final Loss':<10} | {'PPL':<8} | {'Retention %':<12}")
        print("-" * 80)
        for d_name in domain_names:
            l_val = metrics.per_domain_losses[d_name]
            ppl_val = metrics.per_domain_perplexities[d_name]
            ret_val = metrics.retention_rates[d_name]
            print(f"{d_name:<45} | {l_val:<10.4f} | {ppl_val:<8.2f} | {ret_val:<12.2f}%")
        print("=" * 80)
        print(f"Backward Transfer (BWT): {metrics.backward_transfer:+.4f} (Higher is better, negative = forgetting)")
        print(f"Average Forgetting:      {metrics.average_forgetting:.4f}")
        print("=" * 80)

        out_path = Path(args.output).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(metrics.model_dump(mode="json"), indent=2), encoding="utf-8")
        print(f"[Results saved] -> {out_path}\n")


if __name__ == "__main__":
    main()
