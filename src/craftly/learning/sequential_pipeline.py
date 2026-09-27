"""Sequential Multi-Domain Continual Learning Pipeline for Craftly.

Orchestrates sequential domain progression (Domain A -> B -> C -> D), evaluates
Backward Transfer (BWT), Forward Transfer (FWT), Layerwise Parameter Displacement,
and benchmarks anti-forgetting mitigation methods.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from src.craftly.learning.continual import (
    CANONICAL_SEQUENTIAL_DOMAINS,
    EarlyForgettingDetector,
    EarlyWarningSignal,
    ElasticWeightConsolidation,
    ExperienceReplayBuffer,
    GeometryAwareLayerAnchor,
    LayerDisplacementReport,
    LayerwiseDriftAnalyzer,
    ReferenceModelKLLoss,
    SequentialDomain,
    SequentialTransferMetrics,
    WeightMerger,
    compute_sequential_metrics,
)
from src.craftly.model_ops.foundation import ScratchDecoderConfig
from src.craftly.model_ops.torch_decoder import (
    CraftlyDecoderLM,
    load_trusted_checkpoint,
    require_torch,
    save_trusted_checkpoint,
    select_torch_device,
)
from src.craftly.shared.schemas import StrictModel

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None
    F = None


MitigationMethod = Literal[
    "naive_sequential",
    "experience_replay",
    "kl_anchor",
    "ewc",
    "slerp_merge",
    "proposed_gd_lsa",
]


class SequentialRunConfig(StrictModel):
    """Configuration for a multi-domain continual learning study."""

    experiment_id: str = "exp-sequential-continual-001"
    mitigation_method: MitigationMethod = "naive_sequential"
    steps_per_domain: int = Field(default=200, ge=1)
    batch_size: int = Field(default=2, ge=1)
    learning_rate: float = Field(default=5e-5, gt=0.0)
    replay_ratio: float = Field(default=0.20, ge=0.0, le=0.9)
    kl_penalty_weight: float = Field(default=0.5, ge=0.0)
    ewc_lambda: float = Field(default=100.0, ge=0.0)
    slerp_t: float = Field(default=0.25, ge=0.0, le=1.0)
    output_dir: str = "artifacts/craftly/continual_study"
    device: str = "auto"
    seed: int = 1337


class SequentialExperimentResult(StrictModel):
    experiment_id: str
    mitigation_method: str
    domain_names: list[str]
    transfer_metrics: SequentialTransferMetrics
    layer_drift_history: list[LayerDisplacementReport]
    early_warnings: list[EarlyWarningSignal]
    checkpoint_paths: dict[str, str]
    created_at_utc: str = Field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()))


def evaluate_domain_loss(
    model: Any,
    domain: SequentialDomain,
    tokenizer: Any,
    device: str = "cpu",
    max_length: int = 128,
) -> float:
    """Calculate mean autoregressive loss across a domain's benchmark tasks."""
    require_torch()
    model.eval()
    losses: list[float] = []

    max_seq = getattr(getattr(model, "config", None), "max_sequence_length", max_length)
    effective_max = min(max_length, max_seq)

    with torch.no_grad():
        for task in domain.tasks:
            tokens = tokenizer.encode(task.prompt).ids
            if len(tokens) < 2:
                continue
            if len(tokens) > effective_max:
                tokens = tokens[:effective_max]

            inp = torch.tensor([tokens[:-1]], dtype=torch.long, device=device)
            target = torch.tensor([tokens[1:]], dtype=torch.long, device=device)

            out = model(inp, labels=target)
            losses.append(float(out.loss.item()))

    return sum(losses) / len(losses) if losses else 0.0


def run_sequential_continual_study(
    model: Any,
    tokenizer: Any,
    config: SequentialRunConfig,
    domains: list[SequentialDomain] | None = None,
) -> SequentialExperimentResult:
    """Execute complete sequential training across domains with continual evaluation."""
    require_torch()
    dev = select_torch_device(config.device)
    model.to(dev)

    study_domains = domains or CANONICAL_SEQUENTIAL_DOMAINS
    domain_names = [d.name for d in study_domains]
    num_domains = len(study_domains)

    out_root = Path(config.output_dir).resolve() / config.experiment_id / config.mitigation_method
    out_root.mkdir(parents=True, exist_ok=True)

    # 1. Capture base state for drift analysis and zero-shot baseline
    base_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    baseline_losses: list[float] = []
    for d in study_domains:
        b_loss = evaluate_domain_loss(model, d, tokenizer, device=str(dev))
        baseline_losses.append(b_loss)

    transfer_matrix: list[list[float]] = []
    drift_history: list[LayerDisplacementReport] = []
    early_warnings: list[EarlyWarningSignal] = []
    checkpoint_paths: dict[str, str] = {}

    detector = EarlyForgettingDetector(drift_threshold=0.08, velocity_threshold=0.03)

    # Prior samples replay buffer
    replay_buffer = ExperienceReplayBuffer(replay_ratio=config.replay_ratio, seed=config.seed)

    # Reference model for KL anchor if active
    ref_model = None
    kl_fn = None
    if config.mitigation_method in {"kl_anchor", "proposed_gd_lsa"}:
        kl_fn = ReferenceModelKLLoss()
        if hasattr(model, "config"):
            ref_model = CraftlyDecoderLM(model.config)
            ref_model.load_state_dict(base_state_dict)
            ref_model.to(dev)
            ref_model.eval()
            for p in ref_model.parameters():
                p.requires_grad = False

    # EWC components
    ewc_engine = ElasticWeightConsolidation(ewc_lambda=config.ewc_lambda)
    optimal_states: dict[str, dict[str, Any]] = {}
    fisher_matrices: dict[str, dict[str, Any]] = {}

    # GD-LSA proposed method
    gd_lsa = GeometryAwareLayerAnchor()

    # 2. Sequential Training Loop
    for stage_idx, current_domain in enumerate(study_domains):
        domain_id = current_domain.domain_id
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)

        domain_prompts = [t.prompt for t in current_domain.tasks]
        # Register domain samples for future replay
        for p in domain_prompts:
            replay_buffer.add_sample(p)

        steps = config.steps_per_domain
        for step in range(steps):
            optimizer.zero_grad()

            # Encode a micro-batch from current domain
            prompt_text = domain_prompts[step % len(domain_prompts)]
            toks = tokenizer.encode(prompt_text).ids
            max_seq = getattr(getattr(model, "config", None), "max_sequence_length", 128)
            if len(toks) > max_seq:
                toks = toks[:max_seq]
            if len(toks) < 2:
                toks = [0, 0]
            inp = torch.tensor([toks[:-1]], dtype=torch.long, device=dev)
            target = torch.tensor([toks[1:]], dtype=torch.long, device=dev)

            out = model(inp, labels=target)
            loss = out.loss

            if config.mitigation_method == "kl_anchor" and kl_fn is not None and ref_model is not None:
                with torch.no_grad():
                    ref_logits = ref_model(inp).logits.detach()
                kl_val = kl_fn.compute(out.logits, ref_logits)
                loss = loss + (config.kl_penalty_weight * kl_val)

            elif config.mitigation_method == "ewc" and stage_idx > 0:
                prev_domain_name = study_domains[stage_idx - 1].domain_id
                if prev_domain_name in optimal_states and prev_domain_name in fisher_matrices:
                    ewc_pen = ewc_engine.compute_penalty(
                        model,
                        optimal_states[prev_domain_name],
                        fisher_matrices[prev_domain_name],
                    )
                    loss = loss + ewc_pen

            elif config.mitigation_method == "proposed_gd_lsa":
                # Geometry-aware layer stiffness
                lsa_pen = gd_lsa.compute_loss(model, base_state_dict)
                loss = loss + (0.01 * lsa_pen)

            # Experience Replay Interleaving
            if config.mitigation_method in {"experience_replay", "proposed_gd_lsa"} and stage_idx > 0:
                cur_ratio = config.replay_ratio
                if config.mitigation_method == "proposed_gd_lsa" and drift_history:
                    cur_ratio = gd_lsa.get_dynamic_replay_ratio(drift_history[-1].mean_total_drift)

                r_inps, r_lbls = replay_buffer.sample_batch(
                    count=1,
                    tokenizer=tokenizer,
                    max_sequence_length=min(128, max_seq),
                )
                r_inp_t = torch.tensor(r_inps, dtype=torch.long, device=dev)
                r_lbl_t = torch.tensor(r_lbls, dtype=torch.long, device=dev)
                r_out = model(r_inp_t, labels=r_lbl_t)
                loss = (1.0 - cur_ratio) * loss + cur_ratio * r_out.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Post-stage processing: Optional SLERP weight merging
        if config.mitigation_method == "slerp_merge" and stage_idx > 0:
            prev_anchor_path = out_root / f"ckpt_stage_{stage_idx - 1}_{study_domains[stage_idx - 1].domain_id}.pt"
            if prev_anchor_path.exists():
                prev_payload = load_trusted_checkpoint(prev_anchor_path, map_location="cpu")
                prev_weights = prev_payload.get("model", prev_payload)
                curr_weights = {k: v.cpu() for k, v in model.state_dict().items()}
                merged = WeightMerger.merge_state_dicts(prev_weights, curr_weights, method="slerp", t=config.slerp_t)
                model.load_state_dict({k: v.to(dev) for k, v in merged.items()})

        # Save checkpoint for this domain stage
        stage_ckpt = out_root / f"ckpt_stage_{stage_idx}_{domain_id}.pt"
        save_trusted_checkpoint({"model": {k: v.cpu() for k, v in model.state_dict().items()}}, stage_ckpt)
        checkpoint_paths[domain_id] = str(stage_ckpt)

        # Update EWC optimal state & empirical Fisher for this domain
        if config.mitigation_method == "ewc":
            optimal_states[domain_id] = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            fisher_matrices[domain_id] = ElasticWeightConsolidation.estimate_diagonal_fisher(
                model=model,
                samples=domain_prompts,
                tokenizer=tokenizer,
                device=str(dev),
            )

        # 3. Evaluate on ALL domains to populate transfer matrix row R[stage_idx][j]
        matrix_row: list[float] = []
        for eval_domain in study_domains:
            eval_loss = evaluate_domain_loss(model, eval_domain, tokenizer, device=str(dev))
            matrix_row.append(eval_loss)
        transfer_matrix.append(matrix_row)

        # 4. Compute layer-wise parameter drift vs base
        curr_state_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
        drift_report = LayerwiseDriftAnalyzer.compute_drift(base_state_dict, curr_state_cpu)
        drift_history.append(drift_report)

        # 5. Check early warning indicators
        ref_probe_loss = matrix_row[0]  # Domain A is the primary retention probe
        warning_sig = detector.evaluate_step(
            step=(stage_idx + 1) * config.steps_per_domain,
            drift_report=drift_report,
            reference_probe_loss=ref_probe_loss,
        )
        early_warnings.append(warning_sig)

    # 6. Synthesize Transfer Metrics
    transfer_metrics = compute_sequential_metrics(
        transfer_matrix=transfer_matrix,
        domain_names=domain_names,
        baseline_losses=baseline_losses,
    )

    result = SequentialExperimentResult(
        experiment_id=config.experiment_id,
        mitigation_method=config.mitigation_method,
        domain_names=domain_names,
        transfer_metrics=transfer_metrics,
        layer_drift_history=drift_history,
        early_warnings=early_warnings,
        checkpoint_paths=checkpoint_paths,
    )

    result_path = out_root / "sequential_result.json"
    result_path.write_text(json.dumps(result.model_dump(mode="json"), indent=2), encoding="utf-8")
    return result
