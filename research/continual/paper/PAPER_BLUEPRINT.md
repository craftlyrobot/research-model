# Mitigating Catastrophic Forgetting in Scratch-Origin Decoder Transformers: A Mechanistic and Geometry-Aware Sequential Study

**Authors:** Craftly Robot Research Team
**Repository:** https://github.com/Craftly-Robot/research-model
**Architecture:** Craftly Scratch-Origin Decoder (Dense & MoE/MLA Profiles)

---

## 1. Abstract

Large Language Models (LLMs) trained from scratch exhibit catastrophic forgetting when sequentially adapted to specialized downstream domains without full retraining. In this paper, we investigate the dynamics of catastrophic forgetting across sequential domain shifts:
$$\text{Domain A (Algorithms)} \longrightarrow \text{Domain B (Systems)} \longrightarrow \text{Domain C (Defensive Security)} \longrightarrow \text{Domain D (Agentic Patching)}$$
We address three fundamental questions:
1. **Quantification:** Exactly how much prior knowledge degrades across sequential stages, measured by Backward Transfer ($BWT$), Forward Transfer ($FWT$), and Forgetting Measure ($F_k$)?
2. **Mechanistic Root-Cause:** Why does the model forget? By analyzing layer-wise parameter displacement ($\|\Delta \theta_l\|_F$) and latent representational drift, we reveal which transformer layers are most susceptible to destructive overwriting.
3. **Mitigation:** We evaluate established mitigations (Experience Replay, Reference Policy KL Anchoring, Elastic Weight Consolidation, Weight Space SLERP) and propose **Geometry-Aware Dynamic Replay & Layer-Selective Anchoring (GD-LSA)**. GD-LSA stabilizes foundational algorithmic representations while maintaining high plasticity for new domain acquisition.

---

## 2. Research Questions & Hypotheses

### Research Question 1 (Quantification)
*How does sequential domain adaptation affect retention across heterogeneous coding competencies?*
- **Hypothesis 1:** Unconstrained sequential fine-tuning induces severe negative Backward Transfer ($BWT < 0$), with older domains suffering exponential degradation as the domain sequence length grows.

### Research Question 2 (Mechanistic Cause)
*Where in the transformer architecture does catastrophic forgetting physically occur?*
- **Hypothesis 2:** Lower layers (Token Embeddings and initial Self-Attention projections) undergo excessive parameter displacement under unconstrained gradient descent, destroying general syntactic understanding. Upper MLP layers account for domain specialization.

### Research Question 3 (Early Detection)
*Can catastrophic forgetting be predicted before severe task degradation manifests?*
- **Hypothesis 3:** Second-order velocity of layer displacement ($\frac{d^2 \|\theta_l\|}{dt^2}$) serves as a reliable early warning indicator preceding cross-entropy loss spikes on prior domains.

### Research Question 4 (Mitigation)
*Can layer-selective geometric anchoring outperform uniform replay and weight consolidation?*
- **Hypothesis 4:** Protecting foundational attention layers with decaying stiffness while dynamically allocating replay tokens stabilizes retention without sacrificing new task convergence.

---

## 3. Mathematical Formulation

### 3.1 Continual Transfer Matrix & Metrics
Let $T$ be the number of sequential domains. At the end of training on domain $i \in \{0, \dots, T-1\}$, the model is evaluated across all domains $j \in \{0, \dots, T-1\}$, yielding transfer matrix $R \in \mathbb{R}^{T \times T}$, where $R_{i, j}$ denotes the test loss on domain $j$.

1. **Backward Transfer (BWT):**
   $$BWT = \frac{1}{T-1} \sum_{i=0}^{T-2} (R_{i, i} - R_{T-1, i})$$
   A negative $BWT$ indicates catastrophic forgetting; $BWT \ge 0$ indicates positive retention or backward transfer.

2. **Average Forgetting ($F$):**
   $$F_k = \max_{t \in \{0, \dots, T-2\}} (R_{T-1, k} - R_{t, k})$$
   $$F = \frac{1}{T-1} \sum_{k=0}^{T-2} F_k$$

3. **Relative Retention Rate (%):**
   $$\text{Retention}_k = \frac{\text{Score}(R_{T-1, k})}{\text{Score}(R_{k, k})} \times 100\% \quad \text{where } \text{Score}(L) = \frac{100}{1 + L}$$

### 3.2 Layer-wise Parameter Displacement
For parameter tensor $\theta_l$ at layer $l$, relative Frobenius displacement is defined as:
$$\text{Drift}(l) = \frac{\|\theta_l^{(t)} - \theta_l^{(0)}\|_F}{\|\theta_l^{(0)}\|_F + \epsilon}$$

### 3.3 Proposed Method: GD-LSA
$$\mathcal{L}_{\text{total}}(\theta) = (1 - r_t) \mathcal{L}_{\text{curr}}(\theta) + r_t \mathcal{L}_{\text{replay}}(\theta) + \sum_{l} \alpha_l \|\theta_l - \theta_l^{(0)}\|_2^2$$
Where:
- $\alpha_l = \alpha_{\text{base}} \left(1 + \frac{L - l}{L}\right)$ (Earlier layers have higher stiffness).
- $r_t = \min\left(r_{\max}, r_{\text{base}} + 2 \cdot \frac{d \|\Delta \theta\|}{dt}\right)$ (Dynamic replay scaling).

---

## 4. Experimental Setup & Benchmarks

| Domain ID | Competency Domain | Evaluation Benchmark Tasks |
| :--- | :--- | :--- |
| **Domain A** | Core Algorithms & Data Structures | Quicksort, Binary Search, LRU Cache, Graph Traversal |
| **Domain B** | Systems & Web Protocols | HTTP Server, Atomic File Writes, Safe JSON Parsing |
| **Domain C** | Defensive Cybersecurity (CWEs) | SQLi (CWE-89), Path Traversal (CWE-22), Timing Attacks (CWE-208) |
| **Domain D** | Agentic Patching & Tool Use | Unified Diff Hunk Application, AST Code Validation, Lint Resolution |

---

## 5. LaTeX Results Table Template

```latex
\begin{table*}[t]
\centering
\caption{Comparative Continual Learning Performance Across Sequential Domains}
\label{tab:continual_comparison}
\begin{tabular}{lcccccc}
\toprule
\textbf{Method} & \textbf{BWT} ($\uparrow$) & \textbf{Avg Forgetting} ($\downarrow$) & \textbf{Ret. Dom A} (\%) & \textbf{Ret. Dom B} (\%) & \textbf{Ret. Dom C} (\%) & \textbf{Mean Drift} ($\downarrow$) \\
\midrule
Naive Sequential (Baseline)    & -0.4215 & 0.5830 & 62.14\% & 74.20\% & 88.10\% & 0.3842 \\
Experience Replay ($r=0.20$)   & -0.1120 & 0.1840 & 84.50\% & 87.10\% & 92.40\% & 0.2910 \\
Reference KL Anchor            & -0.1540 & 0.2105 & 82.10\% & 84.90\% & 90.15\% & 0.2850 \\
Elastic Weight Consolidation   & -0.1380 & 0.1980 & 83.40\% & 86.20\% & 91.00\% & 0.2740 \\
Weight Merging (SLERP $t=0.25$)& -0.0980 & 0.1450 & 88.20\% & 89.50\% & 93.80\% & 0.2410 \\
\textbf{Proposed GD-LSA (Ours)}& \textbf{-0.0420} & \textbf{0.0710} & \textbf{94.10\%} & \textbf{95.30\%} & \textbf{96.80\%} & \textbf{0.1980} \\
\bottomrule
\end{tabular}
\end{table*}
```
