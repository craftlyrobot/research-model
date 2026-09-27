"""Early Warning Audit CLI for Catastrophic Forgetting Detection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.craftly.learning.continual import EarlyForgettingDetector, LayerwiseDriftAnalyzer
from src.craftly.model_ops.sft_loop import resolve_base_model_assets


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Checkpoint for Early Forgetting Warning.")
    parser.add_argument("--base", required=True, help="Base model checkpoint path")
    parser.add_argument("--current", required=True, help="Current stage checkpoint path")
    parser.add_argument("--reference-loss", type=float, required=True, help="Loss on reference retention probe")
    parser.add_argument("--step", type=int, default=100, help="Current training step")
    args = parser.parse_args()

    _, base_state, _ = resolve_base_model_assets(args.base)
    _, curr_state, _ = resolve_base_model_assets(args.current)

    drift_report = LayerwiseDriftAnalyzer.compute_drift(base_state, curr_state)
    detector = EarlyForgettingDetector(drift_threshold=0.08, velocity_threshold=0.03)

    signal = detector.evaluate_step(
        step=args.step,
        drift_report=drift_report,
        reference_probe_loss=args.reference_loss,
    )

    print("=" * 70)
    print("CRAFTLY EARLY FORGETTING WARNING AUDIT")
    print("=" * 70)
    print(f"Risk Level:          {signal.risk_level}")
    print(f"Warning Triggered:   {signal.warning_triggered}")
    print(f"Drift Velocity:      {signal.drift_velocity:+.6f}")
    if signal.indicators:
        print("Leading Indicators:")
        for ind in signal.indicators:
            print(f"  * {ind}")
    print(f"Recommended Action:  {signal.recommended_action}")
    print("=" * 70)


if __name__ == "__main__":
    main()
