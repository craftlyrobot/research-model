"""Modal GPU Multi-Domain Sequential Continual Learning Runner.

Dispatches sequential continual training (Domain A -> B -> C -> D) to Modal Cloud GPU
with training volume persistence and automatic transfer matrix evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import modal
except ImportError:
    modal = None

from src.craftly.learning.continual import CANONICAL_SEQUENTIAL_DOMAINS
from src.craftly.learning.sequential_pipeline import (
    MitigationMethod,
    SequentialRunConfig,
    run_sequential_continual_study,
)
from src.craftly.model_ops.foundation import ScratchDecoderConfig
from src.craftly.model_ops.sft_loop import resolve_base_model_assets
from src.craftly.model_ops.tokenizer_pipeline import load_tokenizer
from src.craftly.model_ops.torch_decoder import (
    CraftlyDecoderLM,
    require_torch,
    select_torch_device,
)

APP_NAME = "craftly-sequential-continual-study"
VOLUME_NAME = "craftly-training-volume"

if modal is not None:
    app = modal.App(APP_NAME)
    training_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    training_image = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("git")
        .pip_install(
            "torch>=2.3.0",
            "tokenizers>=0.15.2",
            "pydantic>=2.7.0",
            "httpx>=0.27.0",
            "safetensors>=0.4.0",
            "tqdm>=4.66.0",
            "pyyaml>=6.0.1",
        )
        .add_local_python_source("src", "config", "research")
    )
else:
    app = None
    training_volume = None
    training_image = None


def execute_sequential_modal_study(
    method: MitigationMethod = "naive_sequential",
    steps_per_domain: int = 100,
    replay_ratio: float = 0.20,
    kl_penalty_weight: float = 0.50,
    base_checkpoint: str | None = None,
    tokenizer_path: str = "config/tokenizer.json",
    output_dir: str = "/root/research_continual_results",
    smoke: bool = False,
) -> dict[str, Any]:
    require_torch()
    dev = select_torch_device("auto")

    print("=" * 80)
    print("CRAFTLY SEQUENTIAL CONTINUAL LEARNING STUDY (MODAL GPU)")
    print(f"Method: {method.upper()} | Device: {dev.type.upper()} | Smoke: {smoke}")
    print("=" * 80)

    tok_file = Path(tokenizer_path).resolve()
    if not tok_file.exists():
        for cand in [Path("tokenizer.json"), Path("/root/tokenizer.json"), Path("/root/model_300m_extracted/tokenizer.json")]:
            if cand.exists():
                tok_file = cand.resolve()
                break

    tokenizer = load_tokenizer(tok_file)

    # Resolve base model
    if base_checkpoint and Path(base_checkpoint).exists():
        _, state_dict, config_dict = resolve_base_model_assets(Path(base_checkpoint))
        arch_config = ScratchDecoderConfig(**config_dict)
        model = CraftlyDecoderLM(arch_config)
        model.load_state_dict(state_dict)
    else:
        # Fallback to research 30M smoke model config
        arch_config = ScratchDecoderConfig(
            name="craftly-continual-smoke",
            vocab_size=128000,
            hidden_size=256,
            intermediate_size=512,
            num_layers=4,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_sequence_length=512,
        )
        model = CraftlyDecoderLM(arch_config)

    run_cfg = SequentialRunConfig(
        experiment_id="modal_continual_matrix",
        mitigation_method=method,
        steps_per_domain=10 if smoke else steps_per_domain,
        batch_size=1 if smoke else 2,
        learning_rate=1e-4 if smoke else 5e-5,
        replay_ratio=replay_ratio,
        kl_penalty_weight=kl_penalty_weight,
        output_dir=output_dir,
        device="auto",
    )

    result = run_sequential_continual_study(
        model=model,
        tokenizer=tokenizer,
        config=run_cfg,
        domains=CANONICAL_SEQUENTIAL_DOMAINS,
    )

    print("\n" + "=" * 80)
    print("STUDY EXECUTION COMPLETE")
    print(f"Backward Transfer (BWT): {result.transfer_metrics.backward_transfer:+.4f}")
    print(f"Average Forgetting:      {result.transfer_metrics.average_forgetting:.4f}")
    print("=" * 80)

    if modal is not None and training_volume is not None:
        try:
            training_volume.commit()
            print("[Modal Volume] Changes committed to durable volume.")
        except Exception:
            pass

    return result.model_dump(mode="json")


if app is not None:
    @app.function(
        image=training_image,
        gpu="A10G",
        volumes={"/root/volume": training_volume},
        timeout=3600,
    )
    def modal_run(
        method: str = "naive_sequential",
        steps_per_domain: int = 100,
        replay_ratio: float = 0.20,
        kl_penalty_weight: float = 0.50,
        smoke: bool = False,
    ) -> dict[str, Any]:
        return execute_sequential_modal_study(
            method=method,  # type: ignore[arg-type]
            steps_per_domain=steps_per_domain,
            replay_ratio=replay_ratio,
            kl_penalty_weight=kl_penalty_weight,
            output_dir="/root/volume/continual_study",
            smoke=smoke,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Sequential Continual Learning on Modal GPU.")
    parser.add_argument("--method", default="naive_sequential", choices=[
        "naive_sequential", "experience_replay", "kl_anchor", "ewc", "slerp_merge", "proposed_gd_lsa"
    ])
    parser.add_argument("--steps-per-domain", type=int, default=100)
    parser.add_argument("--replay-ratio", type=float, default=0.20)
    parser.add_argument("--kl-penalty-weight", type=float, default=0.50)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    execute_sequential_modal_study(
        method=args.method,
        steps_per_domain=args.steps_per_domain,
        replay_ratio=args.replay_ratio,
        kl_penalty_weight=args.kl_penalty_weight,
        smoke=args.smoke,
    )


if __name__ == "__main__":
    main()
