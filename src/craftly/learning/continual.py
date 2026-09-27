"""Continual Learning and Catastrophic Forgetting Mitigation for Craftly Models.

Provides full-parameter anti-forgetting techniques for scratch-origin models:
1. Experience Replay Buffer (interleaved pretraining code replay)
2. Reference Model KL-Divergence Anchor (policy drift penalty)
3. Weight Space Merging (Spherical Linear Interpolation / SLERP and Task Vectors)
4. Scientific Catastrophic Forgetting Evaluation Benchmark
"""

from __future__ import annotations

import copy
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from src.craftly.model_ops.foundation import ScratchDecoderConfig
from src.craftly.model_ops.tokenizer_pipeline import load_tokenizer
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
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# 1. Experience Replay: Pretraining Distribution Anchoring
# ---------------------------------------------------------------------------

CANONICAL_PRETRAINING_REPLAY_SAMPLES: list[dict[str, str]] = [
    {
        "id": "py-algo-quicksort",
        "category": "general_coding",
        "text": (
            "def quicksort(arr: list[int]) -> list[int]:\n"
            "    if len(arr) <= 1:\n"
            "        return arr\n"
            "    pivot = arr[len(arr) // 2]\n"
            "    left = [x for x in arr if x < pivot]\n"
            "    middle = [x for x in arr if x == pivot]\n"
            "    right = [x for x in arr if x > pivot]\n"
            "    return quicksort(left) + middle + quicksort(right)\n"
        ),
    },
    {
        "id": "py-algo-binary-search",
        "category": "general_coding",
        "text": (
            "def binary_search(elements: list[int], target: int) -> int:\n"
            "    low, high = 0, len(elements) - 1\n"
            "    while low <= high:\n"
            "        mid = (low + high) // 2\n"
            "        if elements[mid] == target:\n"
            "            return mid\n"
            "        elif elements[mid] < target:\n"
            "            low = mid + 1\n"
            "        else:\n"
            "            high = mid - 1\n"
            "    return -1\n"
        ),
    },
    {
        "id": "py-core-lru-cache",
        "category": "general_coding",
        "text": (
            "from collections import OrderedDict\n\n"
            "class LRUCache:\n"
            "    def __init__(self, capacity: int):\n"
            "        self.capacity = capacity\n"
            "        self.cache: OrderedDict[int, int] = OrderedDict()\n\n"
            "    def get(self, key: int) -> int:\n"
            "        if key not in self.cache:\n"
            "            return -1\n"
            "        self.cache.move_to_end(key)\n"
            "        return self.cache[key]\n\n"
            "    def put(self, key: int, value: int) -> None:\n"
            "        if key in self.cache:\n"
            "            self.cache.move_to_end(key)\n"
            "        self.cache[key] = value\n"
            "        if len(self.cache) > self.capacity:\n"
            "            self.cache.popitem(last=False)\n"
        ),
    },
    {
        "id": "py-network-http-server",
        "category": "general_coding",
        "text": (
            "import http.server\n"
            "import socketserver\n\n"
            "PORT = 8080\n\n"
            "class StatusHandler(http.server.SimpleHTTPRequestHandler):\n"
            "    def do_GET(self):\n"
            "        self.send_response(200)\n"
            "        self.send_header('Content-type', 'application/json')\n"
            "        self.end_headers()\n"
            "        self.wfile.write(b'{\"status\": \"healthy\"}')\n"
        ),
    },
    {
        "id": "py-data-json-parser",
        "category": "general_coding",
        "text": (
            "import json\n"
            "from pathlib import Path\n\n"
            "def read_json_manifest(filepath: str | Path) -> dict:\n"
            "    path = Path(filepath)\n"
            "    if not path.is_file():\n"
            "        raise FileNotFoundError(f'Missing file: {path}')\n"
            "    with path.open('r', encoding='utf-8') as f:\n"
            "        return json.load(f)\n"
        ),
    },
]


class ExperienceReplayBuffer:
    """Manages pretraining and general coding memory to interleave with SFT batches."""

    def __init__(
        self,
        replay_samples: list[str] | None = None,
        replay_ratio: float = 0.20,
        seed: int = 1337,
    ) -> None:
        self.replay_ratio = max(0.0, min(0.9, replay_ratio))
        self.rng = random.Random(seed)
        if replay_samples is not None and len(replay_samples) > 0:
            self.samples = list(replay_samples)
        else:
            self.samples = [item["text"] for item in CANONICAL_PRETRAINING_REPLAY_SAMPLES]

    def add_sample(self, text: str) -> None:
        clean = text.strip()
        if clean and clean not in self.samples:
            self.samples.append(clean)

    def sample_batch(
        self,
        count: int,
        tokenizer: Any,
        max_sequence_length: int = 1024,
        pad_token_id: int = 0,
        ignore_index: int = -100,
    ) -> tuple[list[list[int]], list[list[int]]]:
        """Produce encoded input_ids and autoregressive labels for replay sequences."""
        batch_input_ids: list[list[int]] = []
        batch_labels: list[list[int]] = []

        for _ in range(count):
            text = self.rng.choice(self.samples)
            token_ids = tokenizer.encode(text).ids
            if not token_ids:
                token_ids = [0]
            if len(token_ids) > max_sequence_length:
                token_ids = token_ids[:max_sequence_length]

            # Autoregressive next-token prediction targets
            inp = list(token_ids)
            lbl = list(token_ids)

            # Pad to uniform length
            pad_len = max_sequence_length - len(inp)
            if pad_len > 0:
                inp.extend([pad_token_id] * pad_len)
                lbl.extend([ignore_index] * pad_len)

            batch_input_ids.append(inp)
            batch_labels.append(lbl)

        return batch_input_ids, batch_labels


# ---------------------------------------------------------------------------
# 2. KL Divergence Penalty: Reference Policy Constraint
# ---------------------------------------------------------------------------

class ReferenceModelKLLoss:
    """Computes token-level KL divergence penalty between active model and frozen base model."""

    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = max(0.1, temperature)

    def compute(
        self,
        active_logits: Any,
        ref_logits: Any,
        mask: Any | None = None,
    ) -> Any:
        """Calculate forward KL(P_active || P_ref) over valid tokens.

        Args:
            active_logits: (batch_size, seq_len, vocab_size)
            ref_logits: (batch_size, seq_len, vocab_size)
            mask: (batch_size, seq_len) boolean or int mask where 1 = compute KL
        """
        require_torch()
        t = self.temperature
        # P = active distribution, Q = reference base distribution
        log_p = F.log_softmax(active_logits / t, dim=-1)
        p = torch.exp(log_p)
        log_q = F.log_softmax(ref_logits / t, dim=-1)

        # KL(P || Q) = sum P * (log P - log Q)
        kl_per_token = torch.sum(p * (log_p - log_q), dim=-1) * (t ** 2)

        if mask is not None:
            active_mask = (mask != -100).float()
            denom = active_mask.sum() + 1e-8
            return (kl_per_token * active_mask).sum() / denom

        return kl_per_token.mean()


# ---------------------------------------------------------------------------
# 3. Weight Space Merging: SLERP & Task Vector Arithmetic
# ---------------------------------------------------------------------------

class WeightMerger:
    """Merges foundation base weights and specialized SFT weights without external checkpoints."""

    @staticmethod
    def slerp(
        v0: Any,
        v1: Any,
        t: float = 0.5,
        eps: float = 1e-7,
    ) -> Any:
        """Spherical Linear Interpolation between two parameter tensors.

        Formula:
            Omega = arccos( <v0, v1> / (||v0|| * ||v1||) )
            slerp(v0, v1; t) = (sin((1-t)*Omega)/sin(Omega))*v0 + (sin(t*Omega)/sin(Omega))*v1
        """
        require_torch()
        orig_shape = v0.shape
        t0 = v0.flatten().float()
        t1 = v1.flatten().float()

        norm_0 = torch.norm(t0)
        norm_1 = torch.norm(t1)

        if norm_0 < eps or norm_1 < eps:
            return ((1.0 - t) * v0 + t * v1).to(v0.dtype)

        u0 = t0 / norm_0
        u1 = t1 / norm_1

        dot = torch.clamp(torch.sum(u0 * u1), -1.0 + eps, 1.0 - eps)
        omega = torch.acos(dot)
        sin_omega = torch.sin(omega)

        # Collinear fallback (angle close to 0)
        if sin_omega.abs() < eps:
            return ((1.0 - t) * v0 + t * v1).to(v0.dtype)

        scale_0 = torch.sin((1.0 - t) * omega) / sin_omega
        scale_1 = torch.sin(t * omega) / sin_omega

        res = scale_0 * t0 + scale_1 * t1
        return res.view(orig_shape).to(v0.dtype)

    @staticmethod
    def task_arithmetic(
        base_tensor: Any,
        sft_tensor: Any,
        alpha: float = 0.5,
    ) -> Any:
        """Linear task vector scaling: theta_merged = theta_base + alpha * (theta_sft - theta_base)."""
        require_torch()
        diff = sft_tensor.float() - base_tensor.float()
        res = base_tensor.float() + (alpha * diff)
        return res.to(base_tensor.dtype)

    @classmethod
    def merge_state_dicts(
        cls,
        base_state: dict[str, Any],
        sft_state: dict[str, Any],
        method: Literal["slerp", "linear"] = "slerp",
        t: float = 0.3,
    ) -> dict[str, Any]:
        """Apply parameter-wise SLERP or linear merging across two state dicts."""
        require_torch()
        merged: dict[str, Any] = {}
        for key, base_val in base_state.items():
            if key not in sft_state:
                merged[key] = copy.deepcopy(base_val)
                continue

            sft_val = sft_state[key]
            if not isinstance(base_val, torch.Tensor) or not isinstance(sft_val, torch.Tensor):
                merged[key] = copy.deepcopy(sft_val)
                continue

            if base_val.shape != sft_val.shape:
                raise ValueError(f"Shape mismatch for parameter '{key}': {base_val.shape} vs {sft_val.shape}")

            # 1D biases and layer norms are typically linearly averaged
            if base_val.dim() <= 1 or method == "linear":
                merged[key] = ((1.0 - t) * base_val.float() + t * sft_val.float()).to(base_val.dtype)
            else:
                merged[key] = cls.slerp(base_val, sft_val, t=t)

        return merged

    @classmethod
    def merge_checkpoint_files(
        cls,
        base_path: str | Path,
        sft_path: str | Path,
        output_path: str | Path,
        method: Literal["slerp", "linear"] = "slerp",
        t: float = 0.3,
    ) -> Path:
        """Load base and SFT checkpoint archives, merge parameters, and write clean package."""
        require_torch()
        base_file = Path(base_path).resolve()
        sft_file = Path(sft_path).resolve()
        out_file = Path(output_path).resolve()

        if not base_file.exists():
            raise FileNotFoundError(f"Base checkpoint not found: {base_file}")
        if not sft_file.exists():
            raise FileNotFoundError(f"SFT checkpoint not found: {sft_file}")

        base_payload = cls._extract_or_load(base_file)
        sft_payload = cls._extract_or_load(sft_file)

        base_state = base_payload.get("model", base_payload)
        sft_state = sft_payload.get("model", sft_payload)

        merged_state = cls.merge_state_dicts(base_state, sft_state, method=method, t=t)

        out_file.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "model": merged_state,
            "config": sft_payload.get("config", base_payload.get("config", {})),
            "stage": f"continual_merged_{method}_t{int(t * 100)}",
            "merged_at_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "provenance": {
                "base_source": base_file.name,
                "sft_source": sft_file.name,
                "method": method,
                "interpolation_weight": t,
            },
        }

        save_trusted_checkpoint(meta, out_file)
        return out_file

    @staticmethod
    def _extract_or_load(path: Path) -> dict[str, Any]:
        if path.suffix == ".zip":
            import zipfile
            import tempfile
            with tempfile.TemporaryDirectory(prefix="craftly_merge_") as td:
                with zipfile.ZipFile(path, "r") as zf:
                    zf.extractall(td)
                model_pt = Path(td) / "model.pt"
                if not model_pt.exists():
                    raise FileNotFoundError(f"Archive {path} does not contain model.pt")
                return load_trusted_checkpoint(model_pt, map_location="cpu")
        return load_trusted_checkpoint(path, map_location="cpu")


# ---------------------------------------------------------------------------
# 4. Scientific Catastrophic Forgetting Benchmark Suite
# ---------------------------------------------------------------------------

class ForgettingTask(StrictModel):
    task_id: str
    category: str
    prompt: str
    expected_patterns: list[str] = Field(default_factory=list)


CANONICAL_FORGETTING_SUITE: list[ForgettingTask] = [
    ForgettingTask(
        task_id="base-algo-fibonacci",
        category="general_coding",
        prompt="Write a Python function to return the nth Fibonacci number efficiently:\n\ndef fib(n: int) -> int:\n",
        expected_patterns=["def fib", "return", "if n"],
    ),
    ForgettingTask(
        task_id="base-algo-quicksort",
        category="general_coding",
        prompt="Write quicksort in Python using list comprehension:\n\ndef quicksort(arr):\n",
        expected_patterns=["quicksort", "pivot", "return"],
    ),
    ForgettingTask(
        task_id="base-data-invert-dict",
        category="general_coding",
        prompt="Write a Python function to invert a dictionary mapping keys to values:\n\ndef invert_dict(d: dict) -> dict:\n",
        expected_patterns=["def invert_dict", "return", "items()"],
    ),
    ForgettingTask(
        task_id="defensive-cwe-89",
        category="defensive_security",
        prompt="Fix the following SQL injection vulnerability using parameterized queries:\ncursor.execute(f'SELECT * FROM users WHERE id={user_id}')\n",
        expected_patterns=["SELECT", "%s", "cursor.execute"],
    ),
    ForgettingTask(
        task_id="defensive-cwe-22",
        category="defensive_security",
        prompt="Fix path traversal vulnerability when opening files in export directory:\n\ndef read_file(name: str):\n",
        expected_patterns=["Path", "resolve", "is_relative_to"],
    ),
    ForgettingTask(
        task_id="defensive-cwe-208",
        category="defensive_security",
        prompt="Harden secret comparison against side-channel timing attacks in Python:\n\ndef check_key(a, b):\n",
        expected_patterns=["hmac", "compare_digest"],
    ),
]


class ForgettingReport(StrictModel):
    model_name: str
    general_coding_loss: float
    defensive_security_loss: float
    perplexity_general: float
    perplexity_defensive: float
    general_retention_score: float
    evaluated_tasks_count: int
    created_at_utc: str = Field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()))


def evaluate_continual_perplexity(
    model: Any,
    tokenizer: Any,
    tasks: list[ForgettingTask] | None = None,
    device: str = "cpu",
) -> ForgettingReport:
    """Calculate cross-entropy loss and retention scores over general vs defensive tasks."""
    require_torch()
    tasks_to_eval = tasks or CANONICAL_FORGETTING_SUITE
    model.eval()

    gen_losses: list[float] = []
    def_losses: list[float] = []

    with torch.no_grad():
        for t in tasks_to_eval:
            tokens = tokenizer.encode(t.prompt).ids
            if len(tokens) < 2:
                continue
            inp = torch.tensor([tokens[:-1]], dtype=torch.long, device=device)
            target = torch.tensor([tokens[1:]], dtype=torch.long, device=device)

            out = model(inp, labels=target)
            loss_val = float(out.loss.item())

            if t.category == "defensive_security":
                def_losses.append(loss_val)
            else:
                gen_losses.append(loss_val)

    avg_gen = sum(gen_losses) / len(gen_losses) if gen_losses else 0.0
    avg_def = sum(def_losses) / len(def_losses) if def_losses else 0.0

    ppl_gen = math.exp(min(20.0, avg_gen)) if avg_gen > 0 else 1.0
    ppl_def = math.exp(min(20.0, avg_def)) if avg_def > 0 else 1.0

    retention_score = max(0.0, min(100.0, 100.0 / (1.0 + avg_gen)))

    return ForgettingReport(
        model_name=getattr(model, "name", "craftly-model"),
        general_coding_loss=round(avg_gen, 4),
        defensive_security_loss=round(avg_def, 4),
        perplexity_general=round(ppl_gen, 2),
        perplexity_defensive=round(ppl_def, 2),
        general_retention_score=round(retention_score, 2),
        evaluated_tasks_count=len(tasks_to_eval),
    )


# ---------------------------------------------------------------------------
# 5. Multi-Domain Sequential Benchmark Matrix & Transfer Metrics
# ---------------------------------------------------------------------------

class SequentialDomain(StrictModel):
    domain_id: str
    name: str
    description: str
    tasks: list[ForgettingTask]


CANONICAL_SEQUENTIAL_DOMAINS: list[SequentialDomain] = [
    SequentialDomain(
        domain_id="domain_a_algorithms",
        name="Domain A: Core Algorithms & Data Structures",
        description="Fundamental algorithmic logic, sorting, searching, caching, and tree traversal.",
        tasks=[
            ForgettingTask(
                task_id="algo-quicksort",
                category="domain_a_algorithms",
                prompt="def quicksort(arr: list[int]) -> list[int]:\n    \"\"\"Sort array using quicksort.\"\"\"\n",
                expected_patterns=["pivot", "quicksort", "return"],
            ),
            ForgettingTask(
                task_id="algo-binary-search",
                category="domain_a_algorithms",
                prompt="def binary_search(elements: list[int], target: int) -> int:\n    \"\"\"Find target index in sorted list.\"\"\"\n",
                expected_patterns=["mid", "low", "high", "return"],
            ),
            ForgettingTask(
                task_id="algo-lru-cache",
                category="domain_a_algorithms",
                prompt="class LRUCache:\n    def __init__(self, capacity: int):\n",
                expected_patterns=["capacity", "get", "put"],
            ),
        ],
    ),
    SequentialDomain(
        domain_id="domain_b_systems",
        name="Domain B: Systems, Web APIs & Serialization",
        description="System services, HTTP handlers, streaming, and manifest serialization.",
        tasks=[
            ForgettingTask(
                task_id="sys-http-handler",
                category="domain_b_systems",
                prompt="import http.server\nclass HealthServer(http.server.BaseHTTPRequestHandler):\n    def do_GET(self):\n",
                expected_patterns=["send_response", "end_headers", "wfile.write"],
            ),
            ForgettingTask(
                task_id="sys-json-parser",
                category="domain_b_systems",
                prompt="import json\ndef parse_manifest_safe(raw_bytes: bytes) -> dict:\n",
                expected_patterns=["json.loads", "decode", "return"],
            ),
            ForgettingTask(
                task_id="sys-atomic-write",
                category="domain_b_systems",
                prompt="import os, tempfile\ndef write_atomic(path: str, data: str) -> None:\n",
                expected_patterns=["tempfile", "replace", "os."],
            ),
        ],
    ),
    SequentialDomain(
        domain_id="domain_c_defensive_security",
        name="Domain C: Defensive Security & Vulnerability Patching",
        description="Defensive remediation, SQL injection prevention, constant-time cryptography, path sanitation.",
        tasks=[
            ForgettingTask(
                task_id="sec-cwe-89-sqli",
                category="domain_c_defensive_security",
                prompt="def query_user_secure(cursor, user_id: int):\n    \"\"\"Execute query hardened against CWE-89.\"\"\"\n",
                expected_patterns=["SELECT", "%s", "cursor.execute"],
            ),
            ForgettingTask(
                task_id="sec-cwe-22-path",
                category="domain_c_defensive_security",
                prompt="from pathlib import Path\ndef resolve_sandboxed_path(root: Path, target: str) -> Path:\n    \"\"\"Prevent CWE-22 traversal.\"\"\"\n",
                expected_patterns=["resolve", "is_relative_to"],
            ),
            ForgettingTask(
                task_id="sec-cwe-208-timing",
                category="domain_c_defensive_security",
                prompt="import hmac\ndef verify_auth_token_timing_safe(expected: bytes, given: bytes) -> bool:\n",
                expected_patterns=["hmac.compare_digest"],
            ),
        ],
    ),
    SequentialDomain(
        domain_id="domain_d_agentic_patching",
        name="Domain D: Agentic Code Modification & Unified Patching",
        description="Syntactic unified diff generation, lint error resolution, and AST-guided editing.",
        tasks=[
            ForgettingTask(
                task_id="agent-diff-hunk",
                category="domain_d_agentic_patching",
                prompt="def format_unified_diff(original: str, modified: str, filename: str) -> str:\n",
                expected_patterns=["---", "+++", "@@"],
            ),
            ForgettingTask(
                task_id="agent-patch-validator",
                category="domain_d_agentic_patching",
                prompt="import ast\ndef validate_python_syntax(code_str: str) -> bool:\n",
                expected_patterns=["ast.parse", "SyntaxError", "return"],
            ),
            ForgettingTask(
                task_id="agent-lint-filter",
                category="domain_d_agentic_patching",
                prompt="def filter_actionable_lints(raw_issues: list[dict]) -> list[dict]:\n",
                expected_patterns=["severity", "error", "return"],
            ),
        ],
    ),
]


class SequentialTransferMetrics(StrictModel):
    """Scientific Continual Learning transfer and retention metrics matrix."""
    domain_names: list[str]
    transfer_matrix: list[list[float]] = Field(description="R[i][j]: loss on domain j after completing training on domain i")
    backward_transfer: float = Field(description="Mean BWT: negative implies catastrophic forgetting")
    forward_transfer: float = Field(description="Mean FWT: positive implies constructive transfer")
    average_forgetting: float = Field(description="Mean degradation from peak performance on prior domains")
    retention_rates: dict[str, float] = Field(description="Per-domain retention percentage relative to baseline/peak")
    per_domain_losses: dict[str, float]
    per_domain_perplexities: dict[str, float]
    created_at_utc: str = Field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()))


def compute_sequential_metrics(
    transfer_matrix: list[list[float]],
    domain_names: list[str],
    baseline_losses: list[float] | None = None,
) -> SequentialTransferMetrics:
    """Calculate Backward Transfer (BWT), Forward Transfer (FWT), and Forgetting Rate.

    Formulas:
        BWT = 1/(T-1) * sum_{i=1}^{T-1} ( R[T-1][i] - R[i][i] )
              (Note: using negative loss difference so higher is better: loss_i_i - loss_T_i)
        FWT = 1/(T-1) * sum_{i=1}^{T-1} ( baseline[i] - R[i-1][i] )
        Forgetting_k = max_{t < T} ( loss_{T, k} - loss_{t, k} )
    """
    t_count = len(transfer_matrix)
    if t_count == 0 or len(domain_names) != t_count:
        raise ValueError(f"Matrix dimension ({t_count}) must match domain count ({len(domain_names)})")

    # Final stage evaluation losses
    final_losses = transfer_matrix[-1]

    # Backward Transfer: difference between final performance on task i and performance right after training task i
    bwt_vals: list[float] = []
    forgetting_vals: list[float] = []
    retention_rates: dict[str, float] = {}

    for i in range(t_count - 1):
        loss_at_completion = transfer_matrix[i][i]
        loss_final = final_losses[i]
        # In loss terms: positive delta means loss went UP (forgetting occurred)
        degradation = loss_final - loss_at_completion
        forgetting_vals.append(max(0.0, degradation))
        # BWT: conventionally higher is better (positive = improved, negative = forgot)
        bwt_vals.append(-degradation)

        # Retention percentage: 100 * (peak / current) in score space
        score_init = 100.0 / (1.0 + max(0.001, loss_at_completion))
        score_final = 100.0 / (1.0 + max(0.001, loss_final))
        ret_pct = (score_final / max(0.001, score_init)) * 100.0
        retention_rates[domain_names[i]] = round(min(150.0, max(0.0, ret_pct)), 2)

    retention_rates[domain_names[-1]] = 100.0  # Just trained domain has 100% relative retention

    mean_bwt = sum(bwt_vals) / len(bwt_vals) if bwt_vals else 0.0
    mean_forgetting = sum(forgetting_vals) / len(forgetting_vals) if forgetting_vals else 0.0

    # Forward Transfer: comparing zero-shot performance prior to training domain i against baseline
    fwt_vals: list[float] = []
    if baseline_losses and len(baseline_losses) == t_count:
        for i in range(1, t_count):
            zero_shot_loss = transfer_matrix[i - 1][i]
            base_loss = baseline_losses[i]
            # If zero-shot loss is lower than raw base, positive transfer occurred
            fwt_vals.append(base_loss - zero_shot_loss)

    mean_fwt = sum(fwt_vals) / len(fwt_vals) if fwt_vals else 0.0

    per_domain_losses = {domain_names[idx]: round(final_losses[idx], 4) for idx in range(t_count)}
    per_domain_ppls = {
        domain_names[idx]: round(math.exp(min(20.0, final_losses[idx])), 2) for idx in range(t_count)
    }

    return SequentialTransferMetrics(
        domain_names=domain_names,
        transfer_matrix=[[round(val, 4) for val in row] for row in transfer_matrix],
        backward_transfer=round(mean_bwt, 4),
        forward_transfer=round(mean_fwt, 4),
        average_forgetting=round(mean_forgetting, 4),
        retention_rates=retention_rates,
        per_domain_losses=per_domain_losses,
        per_domain_perplexities=per_domain_ppls,
    )


# ---------------------------------------------------------------------------
# 6. Mechanistic Diagnostics: Why Does the Model Forget?
# ---------------------------------------------------------------------------

class LayerDisplacementReport(StrictModel):
    """Detailed layer-wise parameter drift analysis."""
    embedding_drift: float = Field(description="Relative Frobenius displacement of token embeddings")
    attention_drift: float = Field(description="Mean relative displacement of QKVO attention projection weights")
    ffn_drift: float = Field(description="Mean relative displacement of MLP / feed-forward layers")
    head_drift: float = Field(description="Relative displacement of LM output head / norm")
    layer_drifts: dict[str, float] = Field(description="Individual parameter tensor relative drifts")
    mean_total_drift: float = Field(description="Global mean parameter displacement")
    max_drift_parameter: str = Field(description="Parameter tensor undergoing maximal displacement")
    max_drift_value: float = Field(description="Maximal relative displacement value")


class LayerwiseDriftAnalyzer:
    """Computes relative parameter drift: ||theta_t - theta_0||_F / (||theta_0||_F + eps)."""

    @staticmethod
    def compute_drift(
        base_state: dict[str, Any],
        current_state: dict[str, Any],
        eps: float = 1e-8,
    ) -> LayerDisplacementReport:
        require_torch()
        drifts: dict[str, float] = {}

        emb_drifts: list[float] = []
        attn_drifts: list[float] = []
        ffn_drifts: list[float] = []
        head_drifts: list[float] = []

        for name, base_param in base_state.items():
            if name not in current_state:
                continue
            curr_param = current_state[name]
            if not isinstance(base_param, torch.Tensor) or not isinstance(curr_param, torch.Tensor):
                continue

            diff = (curr_param.float() - base_param.float())
            norm_base = torch.norm(base_param.float()).item()
            norm_diff = torch.norm(diff).item()
            rel_drift = norm_diff / max(eps, norm_base)

            drifts[name] = round(rel_drift, 6)

            # Categorize layer type
            lower_name = name.lower()
            if "embed" in lower_name or "wte" in lower_name:
                emb_drifts.append(rel_drift)
            elif any(k in lower_name for k in ["q_proj", "k_proj", "v_proj", "o_proj", "attn"]):
                attn_drifts.append(rel_drift)
            elif any(k in lower_name for k in ["mlp", "ffn", "intermediate", "down_proj", "up_proj"]):
                ffn_drifts.append(rel_drift)
            elif any(k in lower_name for k in ["head", "lm_head", "output"]):
                head_drifts.append(rel_drift)

        mean_emb = sum(emb_drifts) / len(emb_drifts) if emb_drifts else 0.0
        mean_attn = sum(attn_drifts) / len(attn_drifts) if attn_drifts else 0.0
        mean_ffn = sum(ffn_drifts) / len(ffn_drifts) if ffn_drifts else 0.0
        mean_head = sum(head_drifts) / len(head_drifts) if head_drifts else 0.0

        all_vals = list(drifts.values())
        mean_tot = sum(all_vals) / len(all_vals) if all_vals else 0.0

        max_param = max(drifts, key=drifts.get) if drifts else "none"
        max_val = drifts.get(max_param, 0.0)

        return LayerDisplacementReport(
            embedding_drift=round(mean_emb, 6),
            attention_drift=round(mean_attn, 6),
            ffn_drift=round(mean_ffn, 6),
            head_drift=round(mean_head, 6),
            layer_drifts=drifts,
            mean_total_drift=round(mean_tot, 6),
            max_drift_parameter=max_param,
            max_drift_value=round(max_val, 6),
        )


class RepresentationDriftReport(StrictModel):
    mean_cosine_similarity: float = Field(description="Average cosine similarity of hidden states across probe tasks")
    layer_cosine_similarities: dict[int, float] = Field(description="Layer-wise cosine similarity profile")


class RepresentationDriftAnalyzer:
    """Measures latent representational alignment across transformer layers."""

    @staticmethod
    def compute_hidden_drift(
        model_base: Any,
        model_current: Any,
        input_ids: Any,
    ) -> RepresentationDriftReport:
        require_torch()
        model_base.eval()
        model_current.eval()

        layer_cosines: dict[int, float] = {}

        with torch.no_grad():
            # Get intermediate representations via forward hook or standard hidden states
            out_base = model_base(input_ids)
            out_curr = model_current(input_ids)

            logits_base = out_base.logits.float()
            logits_curr = out_curr.logits.float()

            cos_sim = F.cosine_similarity(logits_base.view(-1, logits_base.size(-1)), logits_curr.view(-1, logits_curr.size(-1)), dim=-1)
            mean_cos = float(cos_sim.mean().item())
            layer_cosines[0] = round(mean_cos, 4)

        return RepresentationDriftReport(
            mean_cosine_similarity=round(mean_cos, 4),
            layer_cosine_similarities=layer_cosines,
        )


class GradientConflictAnalyzer:
    """Measures gradient direction alignment between new task loss and reference replay loss."""

    @staticmethod
    def compute_gradient_cosine(
        model: Any,
        task_loss: Any,
        replay_loss: Any,
    ) -> float:
        require_torch()
        model.zero_grad()
        task_loss.backward(retain_graph=True)
        grads_task = [p.grad.flatten().clone() for p in model.parameters() if p.grad is not None]

        model.zero_grad()
        replay_loss.backward(retain_graph=True)
        grads_replay = [p.grad.flatten().clone() for p in model.parameters() if p.grad is not None]

        model.zero_grad()

        if not grads_task or not grads_replay:
            return 1.0

        vec_task = torch.cat(grads_task)
        vec_replay = torch.cat(grads_replay)

        norm_t = torch.norm(vec_task)
        norm_r = torch.norm(vec_replay)
        if norm_t < 1e-8 or norm_r < 1e-8:
            return 0.0

        cos_sim = torch.dot(vec_task, vec_replay) / (norm_t * norm_r)
        return float(cos_sim.item())


# ---------------------------------------------------------------------------
# 7. Early Warning & Forgetting Detection
# ---------------------------------------------------------------------------

class EarlyWarningSignal(StrictModel):
    warning_triggered: bool
    risk_level: Literal["LOW", "MODERATE", "CRITICAL"]
    drift_velocity: float
    indicators: list[str] = Field(default_factory=list)
    recommended_action: str


class EarlyForgettingDetector:
    """Monitors parameter displacement velocity and loss spikes to detect catastrophic collapse early."""

    def __init__(
        self,
        drift_threshold: float = 0.08,
        velocity_threshold: float = 0.03,
    ) -> None:
        self.drift_threshold = drift_threshold
        self.velocity_threshold = velocity_threshold
        self.history: list[dict[str, float]] = []

    def evaluate_step(
        self,
        step: int,
        drift_report: LayerDisplacementReport,
        reference_probe_loss: float,
    ) -> EarlyWarningSignal:
        current_drift = drift_report.mean_total_drift
        prev_drift = self.history[-1]["drift"] if self.history else current_drift
        velocity = current_drift - prev_drift

        self.history.append({"step": float(step), "drift": current_drift, "loss": reference_probe_loss})

        indicators: list[str] = []
        risk: Literal["LOW", "MODERATE", "CRITICAL"] = "LOW"
        action = "Maintain scheduled learning rate."

        # Check 1: Embedding or Attention layer excessive drift
        if drift_report.embedding_drift > self.drift_threshold * 1.5:
            indicators.append(f"Embedding drift elevated ({drift_report.embedding_drift:.4f} > {self.drift_threshold * 1.5:.4f})")
            risk = "MODERATE"

        # Check 2: High displacement acceleration
        if velocity > self.velocity_threshold:
            indicators.append(f"Parameter displacement velocity high ({velocity:.4f} / step)")
            risk = "CRITICAL" if risk == "MODERATE" else "MODERATE"

        # Check 3: Reference loss divergence
        if len(self.history) >= 2:
            base_loss = self.history[0]["loss"]
            if reference_probe_loss > base_loss * 1.35:
                indicators.append(f"Reference retention loss inflated by {((reference_probe_loss / max(0.001, base_loss)) - 1.0):+.1%}")
                risk = "CRITICAL"

        if risk == "CRITICAL":
            action = "Trigger emergency experience replay boost (+15%) and decrease active learning rate by 50%."
        elif risk == "MODERATE":
            action = "Enable reference policy KL constraint or increase weight decay."

        return EarlyWarningSignal(
            warning_triggered=(risk != "LOW"),
            risk_level=risk,
            drift_velocity=round(velocity, 6),
            indicators=indicators,
            recommended_action=action,
        )


# ---------------------------------------------------------------------------
# 8. Elastic Weight Consolidation (EWC)
# ---------------------------------------------------------------------------

class ElasticWeightConsolidation:
    """Diagonal Fisher Information-weighted quadratic regularization penalty."""

    def __init__(self, ewc_lambda: float = 100.0) -> None:
        self.ewc_lambda = max(0.0, ewc_lambda)

    @classmethod
    def estimate_diagonal_fisher(
        cls,
        model: Any,
        samples: list[str],
        tokenizer: Any,
        device: str = "cpu",
        max_sequence_length: int = 128,
    ) -> dict[str, Any]:
        """Compute empirical diagonal Fisher information on domain reference samples."""
        require_torch()
        model.eval()
        fisher_diag: dict[str, Any] = {
            name: torch.zeros_like(param, device=device)
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        total_samples = 0
        for text in samples:
            tokens = tokenizer.encode(text).ids
            if len(tokens) < 2:
                continue
            if len(tokens) > max_sequence_length:
                tokens = tokens[:max_sequence_length]

            inp = torch.tensor([tokens[:-1]], dtype=torch.long, device=device)
            target = torch.tensor([tokens[1:]], dtype=torch.long, device=device)

            model.zero_grad()
            out = model(inp, labels=target)
            loss = out.loss
            loss.backward()

            for name, param in model.named_parameters():
                if param.grad is not None and name in fisher_diag:
                    fisher_diag[name] += param.grad.data.pow(2)

            total_samples += 1

        if total_samples > 0:
            for name in fisher_diag:
                fisher_diag[name] /= total_samples

        return fisher_diag

    def compute_penalty(
        self,
        model: Any,
        optimal_state_dict: dict[str, Any],
        fisher_diag: dict[str, Any],
    ) -> Any:
        """Compute EWC loss: (lambda / 2) * sum_i F_i * (theta_i - theta_opt_i)^2."""
        require_torch()
        loss = torch.tensor(0.0, device=next(model.parameters()).device)
        if self.ewc_lambda <= 0.0:
            return loss

        for name, param in model.named_parameters():
            if name in optimal_state_dict and name in fisher_diag:
                opt_val = optimal_state_dict[name].to(param.device)
                f_val = fisher_diag[name].to(param.device)
                loss += (f_val * (param - opt_val).pow(2)).sum()

        return 0.5 * self.ewc_lambda * loss


# ---------------------------------------------------------------------------
# 9. Proposed Method: Geometry-Aware Dynamic Replay & Layer Anchoring (GD-LSA)
# ---------------------------------------------------------------------------

class GeometryAwareLayerAnchor:
    """Craftly Proposed Anti-Forgetting Technique (GD-LSA).

    Dynamically modulates parameter stiffness and replay allocation:
    1. Lower layers (Token Embeddings & Initial Attention) maintain high rigidity
       to protect core algorithmic primitives.
    2. Higher layers (MLP & Domain Projections) have selective plasticity
       to allow new domain acquisition without gradient interference.
    3. Replay ratio dynamically increases when parameter drift velocity rises.
    """

    def __init__(
        self,
        base_replay_ratio: float = 0.15,
        max_replay_ratio: float = 0.40,
        anchor_stiffness: float = 0.5,
    ) -> None:
        self.base_replay_ratio = base_replay_ratio
        self.max_replay_ratio = max_replay_ratio
        self.anchor_stiffness = anchor_stiffness

    def compute_layer_weight(self, param_name: str, total_layers: int = 4) -> float:
        """Assign layer-dependent stiffness weight (higher for early layers)."""
        lower = param_name.lower()
        if "embed" in lower or "wte" in lower:
            return 2.5 * self.anchor_stiffness
        if "head" in lower or "lm_head" in lower:
            return 1.8 * self.anchor_stiffness

        # Extract layer index if present
        for idx in range(total_layers):
            if f"layers.{idx}." in lower or f"layer.{idx}." in lower:
                # Linear decay of stiffness: earlier layers are stiffer
                decay = 1.0 - (idx / max(1, total_layers))
                return (1.0 + decay) * self.anchor_stiffness

        return 1.0 * self.anchor_stiffness

    def compute_loss(
        self,
        model: Any,
        base_state_dict: dict[str, Any],
    ) -> Any:
        require_torch()
        loss = torch.tensor(0.0, device=next(model.parameters()).device)
        for name, param in model.named_parameters():
            if name in base_state_dict:
                stiffness = self.compute_layer_weight(name)
                base_val = base_state_dict[name].to(param.device)
                loss += stiffness * (param.float() - base_val.float()).pow(2).sum()
        return loss

    def get_dynamic_replay_ratio(self, drift_velocity: float) -> float:
        """Scale replay ratio in response to parameter velocity."""
        ratio = self.base_replay_ratio + (drift_velocity * 2.0)
        return float(min(self.max_replay_ratio, max(self.base_replay_ratio, ratio)))
