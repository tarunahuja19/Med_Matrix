# BM-NODE-UQ: Bloch-McConnell Neural ODE with Spectral Cross-Attention and Built-In Uncertainty Quantification for CEST MRI

> A novel physics-informed architecture for fast, reliable, quantitative CEST parameter mapping with clinically actionable confidence maps.

---

## 1. Problem Statement & Motivation

**Input:** Z-spectrum stack per voxel — signal at ~20–40 saturation offsets spanning −5 → +5 ppm. Shape: `(N_offsets, H, W)`.

**Output:** Quantitative physical parameter maps per voxel:
- Exchange rate \( k_{sw} \) (s⁻¹)
- Solute pool fraction \( f_s \)
- APT signal / MTR_asym → pH-weighted map
- **Plus:** per-voxel uncertainty (confidence) maps for each parameter

**Clinical target:** Tumor grading (glioma), recurrence vs. radiation necrosis differentiation — needs to be fast (single forward pass), physics-grounded (not a black box), and uncertainty-aware (radiologist needs to know *where* to trust the map).

---

## 2. What Already Exists (and Why It's Not Enough)

| Method | What it does | Gap |
|:---|:---|:---|
| **deepCEST** (2020–2022) | Supervised MLP/CNN trained on Lorentzian-fitted labels | No physics in the loop; no uncertainty; needs ground-truth labels |
| **Neural Bloch-McConnell Fitting (NBMF)** (2024) | Auto-diff BM solver + neural reconstructor | Treats BM solver as black-box layer; no spectral attention; no uncertainty |
| **Transformer-based CEST** (2025) | Self-supervised transformer on Z-spectra | No explicit ODE integration; attention is generic, not physics-guided |
| **Lorentzian-KANs** (2025) | Learnable Lorentzian activations | Limited to Lorentzian decomposition; doesn't model full BM dynamics; no spatial context |
| **Physics-Informed INRs** (2024–2025) | Continuous coordinate-based representations | Scan-specific optimization (slow); not feed-forward inference |

> [!IMPORTANT]
> **No existing work combines:** (1) a differentiable Neural ODE Bloch-McConnell layer as a *physics decoder*, (2) spectral cross-attention that conditions on known pool resonance frequencies, (3) pool-decoupled prediction heads, and (4) heteroscedastic aleatoric + MC-dropout epistemic uncertainty — all in a single feed-forward architecture trained on synthetic data with physics-constrained losses.

---

## 3. Architecture: BM-NODE-UQ

### 3.1 High-Level Overview

```
  Z-spectrum input                    Quantitative Maps + Uncertainty
  (N_offsets, H, W)                   (k_sw, f_s, APT, ...) ± σ per voxel
        │                                       ▲
        ▼                                       │
  ┌─────────────────┐                ┌──────────────────────┐
  │  Spectral        │                │  Pool-Decoupled      │
  │  Encoder         │───────────────▶│  Prediction Heads    │
  │  (1D Conv + PE)  │    latent z    │  + Uncertainty Heads  │
  └─────────────────┘                └──────────┬───────────┘
        │                                       │
        │                                       │ θ̂ = {k_sw, f_s, T₁, T₂, ...}
        │                                       ▼
        │                            ┌──────────────────────┐
        │                            │  Differentiable       │
        │                            │  BM-Neural-ODE Layer  │
        │                            │  (Physics Decoder)    │
        │                            └──────────┬───────────┘
        │                                       │
        │                                       │  Ẑ(Δω; θ̂) = predicted Z-spectrum
        │                                       ▼
        └──────────────────────────────▶  LOSS COMPUTATION
                Z_measured                 (self-consistency)
```

### 3.2 Module 1 — Spectral Encoder with Physics-Guided Cross-Attention

This is **not** a vanilla transformer. The key innovation: the *queries* come from the input Z-spectrum, but the *keys/values* are derived from **learnable pool-anchored embeddings** initialized at known resonance frequencies.

#### Architecture Detail:

```
Input Z-spectrum: z ∈ ℝ^{N_offsets}  (per voxel, or patch)
                    │
                    ▼
         ┌─────────────────────┐
         │ 1D Causal Conv Block │  ← 3 layers, kernel=3, GELU
         │ + Sinusoidal PE      │     positional encoding uses Δω values
         │   (offset-aware)     │     (not learned ordinal positions!)
         └─────────┬───────────┘
                   │
                   ▼  spectral tokens: S ∈ ℝ^{N_offsets × d_model}
         ┌─────────────────────────────────────┐
         │  Pool-Anchored Cross-Attention (PACA)│
         │                                     │
         │  Q = W_q · S                        │  ← from input spectrum
         │                                     │
         │  Pool Anchors: P ∈ ℝ^{N_pools × d_model}  │
         │  (learnable, init at Δω = 0, ±1.5,  │
         │   ±2.0, ±3.5 ppm for water, amide,  │
         │   amine, NOE, MT pools)              │
         │                                     │
         │  K = W_k · P                        │
         │  V = W_v · P                        │
         │                                     │
         │  Attn = softmax(QKᵀ/√d) · V        │
         │  + Residual + LayerNorm             │
         └─────────────────┬───────────────────┘
                           │
                           ▼
         ┌─────────────────────────────────────┐
         │  Feed-Forward + Self-Attention       │
         │  (2 standard transformer blocks)     │
         │  with pre-norm, GELU, dropout=0.1    │
         └─────────────────┬───────────────────┘
                           │
                           ▼
                  Latent: z_latent ∈ ℝ^{d_model}
                  (mean-pool over offset dim)
```

> [!NOTE]
> **Why Pool-Anchored Cross-Attention (PACA)?**
> Standard self-attention treats all offsets equally. But in CEST physics, specific frequency offsets correspond to specific proton pools (amide at +3.5 ppm, amine at +2.0 ppm, NOE at −3.5 ppm, etc.). PACA forces the network to attend *from* the measured spectrum *to* physically meaningful anchor points, learning pool-specific feature extraction. The anchors are learnable (can shift from initialization) so the network adapts to actual pulse sequence parameters and field strength.

#### Why sinusoidal PE on actual Δω values?
Standard positional encoding uses integer positions (1, 2, 3...). But Z-spectrum offsets are **non-uniformly spaced** in ppm and their physical meaning is tied to their actual frequency value. We use:

$$\text{PE}(\Delta\omega, 2i) = \sin(\Delta\omega / 10000^{2i/d}), \quad \text{PE}(\Delta\omega, 2i+1) = \cos(\Delta\omega / 10000^{2i/d})$$

This injects *physics-aware* positional information — the network "knows" that offset +3.5 ppm is the amide region, not just "position 15 in the array."

---

### 3.3 Module 2 — Pool-Decoupled Prediction Heads with Heteroscedastic Uncertainty

Instead of a single MLP outputting all parameters, we use **separate, specialized heads** for each physical pool. This is motivated by the fact that different pools have vastly different parameter ranges and sensitivities.

```
             z_latent ∈ ℝ^{d_model}
                    │
        ┌───────────┼───────────┬──────────────┐
        ▼           ▼           ▼              ▼
   ┌─────────┐ ┌─────────┐ ┌─────────┐  ┌──────────┐
   │ Water   │ │ Amide   │ │ Amine   │  │ NOE/MT   │
   │ Head    │ │ Head    │ │ Head    │  │ Head     │
   │ (MLP)   │ │ (MLP)   │ │ (MLP)   │  │ (MLP)    │
   └────┬────┘ └────┬────┘ └────┬────┘  └────┬─────┘
        │           │           │             │
    ┌───┴───┐   ┌───┴───┐  ┌───┴───┐    ┌────┴────┐
    │μ    σ²│   │μ    σ²│  │μ    σ²│    │μ     σ²│
    │T₁w   │   │k_sw   │  │k_sw   │    │f_NOE   │
    │T₂w   │   │f_s    │  │f_s    │    │k_NOE   │
    │       │   │T₁s   │  │T₁s   │    │        │
    │       │   │T₂s   │  │T₂s   │    │        │
    └───────┘   └───────┘  └───────┘    └────────┘
```

Each head outputs:
- **μ (mean):** Point estimate of the physical parameter, passed through physically-motivated **activation constraints**:
  - `k_sw` → `softplus` (must be > 0, typically 10–5000 s⁻¹)
  - `f_s` → `sigmoid` × scale (fraction, 0–1, typically 10⁻⁴ to 10⁻²)
  - `T₁, T₂` → `softplus` (must be > 0)
- **σ² (variance):** Aleatoric uncertainty via a second output neuron with `softplus` activation. This captures *input-dependent noise* — voxels near air/bone interfaces or with motion artifact will naturally have higher predicted uncertainty.

Each head is a small MLP: `d_model → 128 → 64 → outputs`, with GELU activations and residual connections.

> [!TIP]
> **Why decoupled heads?** Joint prediction with a single head creates parameter coupling artifacts — errors in water T₁ estimation propagate into amide k_sw estimates. Decoupled heads allow each pool's parameters to be learned with independent gradient pathways, and the physics decoder (next module) is what enforces cross-pool consistency.

---

### 3.4 Module 3 — Differentiable BM-Neural-ODE Physics Decoder

This is the **core novelty**. Instead of using Bloch-McConnell equations only in the loss function (standard PINN approach), we implement a **differentiable Neural ODE solver** that:

1. Takes the predicted parameters θ̂ as input
2. Numerically integrates the Bloch-McConnell ODEs
3. Outputs a *predicted* Z-spectrum Ẑ(Δω; θ̂)
4. Compares Ẑ against the measured Z to form the self-consistency loss

#### The Bloch-McConnell System (N-pool)

For a 2-pool system (water + solute), the BM equations in the rotating frame are:

$$\frac{d\mathbf{M}}{dt} = \mathbf{A} \cdot \mathbf{M} + \mathbf{C}$$

Where the state vector is:

$$\mathbf{M} = [M_x^w, M_y^w, M_z^w, M_x^s, M_y^s, M_z^s]^T$$

And **A** is the 6×6 evolution matrix encoding:
- Relaxation (T₁, T₂ for each pool)
- Chemical exchange (k_sw, k_ws = k_sw · f_s / f_w)
- RF saturation (B₁, Δω)
- Chemical shift differences

#### Neural ODE Integration

```
   θ̂ = {k_sw, f_s, T₁w, T₂w, T₁s, T₂s, ...}
                    │
                    ▼
   ┌────────────────────────────────┐
   │  Construct A(θ̂, Δω, B₁)       │  ← Analytical matrix construction
   │  from Bloch-McConnell equations│     (no learnable parameters here —
   │                                │      pure physics)
   └────────────┬───────────────────┘
                │
                ▼
   ┌────────────────────────────────┐
   │  Neural ODE Solver             │
   │  (Dormand-Prince / RK4)        │
   │                                │
   │  For each Δω_i in offsets:     │
   │    M₀ = thermal equilibrium    │
   │    M(t_sat) = ODESolve(        │
   │      f = A·M + C,              │
   │      M₀, t=0→t_sat,            │
   │      rtol=1e-5, atol=1e-6     │
   │    )                           │
   │    Ẑ(Δω_i) = M_z^w(t_sat)/M₀_z│
   │                                │
   │  Backprop through solver via   │
   │  adjoint method (O(1) memory)  │
   └────────────┬───────────────────┘
                │
                ▼
         Ẑ ∈ ℝ^{N_offsets}  (predicted Z-spectrum)
```

> [!IMPORTANT]
> **Why Neural ODE instead of matrix exponential?**
>
> Most existing CEST fitting uses `expm(A · t_sat)` — the matrix exponential. This works but has two problems: (1) it assumes *continuous-wave* saturation, which is wrong for pulsed CEST sequences used clinically, and (2) gradients through `expm` are numerically unstable for stiff systems.
>
> The Neural ODE approach with adjoint-method backpropagation:
> - Naturally handles **pulsed saturation** (the ODE can model the actual pulse train: RF-on, delay, RF-on, delay...)
> - Has **O(1) memory** via the adjoint method (critical for batch processing of full brain volumes)
> - Is **differentiable end-to-end** through the solver, enabling gradient flow from the Z-spectrum reconstruction loss all the way back to the encoder

#### Residual Correction Network (Optional, for Model Mismatch)

A small MLP `δ_net` that learns a *residual correction* to the BM-ODE output:

$$\hat{Z}_{\text{final}}(\Delta\omega) = \hat{Z}_{\text{BM}}(\Delta\omega; \hat{\theta}) + \delta_{\text{net}}(\Delta\omega, z_{\text{latent}})$$

This handles effects not captured by the simplified BM model (e.g., semi-solid MT pool, partial volume, B₀ inhomogeneity) without abandoning the physics backbone. The residual is **L2-regularized** to stay small.

---

### 3.5 Module 4 — Epistemic Uncertainty via Concrete Dropout

On top of the aleatoric σ² from each prediction head, we add **epistemic uncertainty** via Concrete Dropout (Gal et al., 2017) applied to the encoder and prediction heads:

- Dropout probabilities are **learned** (not fixed hyperparameters)
- At inference: run T=20 stochastic forward passes → compute mean and variance of predictions across passes
- Total uncertainty: σ²_total = σ²_aleatoric + σ²_epistemic

This gives the radiologist a **confidence map**: "I'm confident about k_sw in the tumor core, but uncertain at the necrotic boundary."

---

## 4. Loss Functions

The total loss has **five components**, weighted by learnable/tunable coefficients:

$$\mathcal{L}_{\text{total}} = \lambda_1 \mathcal{L}_{\text{recon}} + \lambda_2 \mathcal{L}_{\text{physics}} + \lambda_3 \mathcal{L}_{\text{param}} + \lambda_4 \mathcal{L}_{\text{smooth}} + \lambda_5 \mathcal{L}_{\text{asym}}$$

---

### 4.1 Self-Consistency Reconstruction Loss (NLL with heteroscedastic noise)

$$\mathcal{L}_{\text{recon}} = \frac{1}{N} \sum_{i=1}^{N_{\text{offsets}}} \left[ \frac{(Z_i - \hat{Z}_i)^2}{2\hat{\sigma}_i^2} + \frac{1}{2}\log \hat{\sigma}_i^2 \right]$$

Where:
- Z_i is the measured signal at offset i
- Ẑ_i is the BM-ODE predicted signal at offset i
- σ̂²_i is the predicted aleatoric variance (propagated from parameter uncertainties)

> [!NOTE]
> This is a **negative log-likelihood** under a Gaussian noise model with *learned, input-dependent variance*. The second term (`log σ²`) prevents the network from trivially increasing σ² to reduce the first term. This is what makes the uncertainty *calibrated*.

---

### 4.2 Physics Residual Loss (BM-ODE Residual)

$$\mathcal{L}_{\text{physics}} = \frac{1}{N_c} \sum_{j=1}^{N_c} \left\| \frac{d\hat{\mathbf{M}}_j}{dt} - \mathbf{A}(\hat{\theta}) \cdot \hat{\mathbf{M}}_j - \mathbf{C} \right\|_2^2$$

Evaluated at N_c **collocation points** sampled along the ODE trajectory. This is the classic PINN residual — it penalizes the ODE solution if it drifts from the analytical BM equations at intermediate time points, not just the final state.

---

### 4.3 Supervised Parameter Loss (on synthetic data)

$$\mathcal{L}_{\text{param}} = \sum_{p \in \text{params}} w_p \cdot \text{Huber}\left(\frac{\theta_p^{\text{true}} - \hat{\theta}_p}{\theta_p^{\text{scale}}}\right)$$

Where:
- Huber loss (δ=1) is used instead of MSE for robustness to outlier voxels
- Parameters are **scale-normalized** (k_sw by 1000, f_s by 0.01, T₁ by 1.0, etc.) so each contributes equally
- This loss is only active when ground-truth parameters are available (synthetic training data)

---

### 4.4 Spatial Smoothness Regularization

$$\mathcal{L}_{\text{smooth}} = \sum_{p \in \text{params}} \left( \| \nabla_x \hat{\theta}_p \|_1 + \| \nabla_y \hat{\theta}_p \|_1 \right)$$

Total Variation (TV) regularization on the predicted parameter maps. This enforces spatially smooth maps without blurring sharp boundaries (L1 is edge-preserving, unlike L2).

---

### 4.5 Asymmetry Consistency Loss (Physics Prior)

$$\mathcal{L}_{\text{asym}} = \left\| \text{MTR}_{\text{asym}}^{\text{predicted}} - \left( \hat{Z}(-\Delta\omega) - \hat{Z}(+\Delta\omega) \right) \right\|_2^2$$

This enforces that the network's internal parameter estimates, when fed through the BM-ODE, produce an asymmetry profile consistent with the directly computed MTR_asym from the measured Z-spectrum. It's a **cross-consistency** constraint between the two ways of computing the CEST effect.

---

### 4.6 Loss Weighting Strategy

Use **uncertainty-based automatic loss weighting** (Kendall et al., 2018):

$$\mathcal{L}_{\text{total}} = \sum_{i} \frac{1}{2s_i^2} \mathcal{L}_i + \log s_i$$

Where s_i are learnable log-variance parameters. This avoids manual tuning of λ weights.

---

## 5. Training Strategy

### 5.1 Data Generation

Synthetic data from the BM-ODE simulator with:

| Parameter | Range | Distribution |
|:---|:---|:---|
| k_sw (amide) | 20 – 500 s⁻¹ | Log-uniform |
| k_sw (amine) | 500 – 5000 s⁻¹ | Log-uniform |
| f_s (amide) | 5×10⁻⁴ – 5×10⁻² | Log-uniform |
| f_s (amine) | 1×10⁻⁴ – 1×10⁻² | Log-uniform |
| T₁w | 1.0 – 2.5 s | Uniform |
| T₂w | 40 – 120 ms | Uniform |
| T₁s | 0.5 – 2.0 s | Uniform |
| T₂s | 10 – 50 ms | Uniform |
| B₁ | 0.5 – 3.0 μT | Uniform |
| B₀ shift | −0.5 – +0.5 ppm | Gaussian |

**Noise injection:** Rician noise at multiple SNR levels (20–200), applied to the simulated Z-spectra.

**Spatial structure:** Generate 2D phantom slices with region-based parameter variations (tumor core, peritumoral edema, white matter, CSF, necrosis) using ellipsoidal masks with smooth boundaries, so the network sees spatially structured data during training.

### 5.2 Training Schedule

```
Phase 1 (Epochs 1–50):     Warm-up encoder only
                            L_param (high weight) + L_recon (low weight)
                            Freeze BM-ODE layer gradients
                            LR: 1e-3, cosine decay

Phase 2 (Epochs 51–150):   Full end-to-end training
                            All losses active
                            Unfreeze BM-ODE adjoint gradients
                            LR: 5e-4, cosine decay with warm restarts

Phase 3 (Epochs 151–200):  Fine-tune with heavy physics
                            L_physics weight ×3
                            L_smooth activated
                            Concrete dropout calibration
                            LR: 1e-4, linear decay
```

**Optimizer:** AdamW (β₁=0.9, β₂=0.999, weight decay=1e-4)  
**Batch size:** 256 voxels (or 16 patches of 4×4 for spatial loss)  
**Gradient clipping:** max norm = 1.0 (essential for ODE adjoint stability)

### 5.3 Inference

- Single forward pass: ~5 ms per slice (on GPU)
- With MC-dropout (T=20 passes): ~100 ms per slice
- No iterative fitting, no dictionary lookup

---

## 6. Architecture Dimensions (Recommended)

| Component | Specification |
|:---|:---|
| d_model | 128 |
| N_heads (PACA) | 4 |
| N_pool_anchors | 5 (water, amide, amine, NOE, MT) |
| N_self_attn_blocks | 2 |
| Head MLP layers | 3 (128→64→outputs) |
| Residual correction MLP | 2 layers (64→32→1) |
| ODE solver | Dormand-Prince (adaptive) or RK4 (fixed-step, faster) |
| Concrete dropout layers | 4 (encoder conv, PACA, self-attn, heads) |
| Total parameters | ~850K (lightweight; fits on any clinical GPU) |

---

## 7. What Makes This Publishable / Novel

### 7.1 Specific Novelty Claims

1. **Pool-Anchored Cross-Attention (PACA):** No prior work uses cross-attention where keys/values are *learnable embeddings initialized at known resonance frequencies*. Existing transformers for CEST use vanilla self-attention that treats all offsets identically. PACA provides **interpretable** attention — you can visualize which spectral offsets the network associates with each pool, giving radiophysicists a tool to verify the network "understands" the CEST physics.

2. **BM-Neural-ODE as a differentiable physics decoder (not just a loss term):** Existing PINNs embed BM equations in the *loss function*. We embed a full *differentiable ODE solver* as an architectural component. The network outputs parameters → the ODE layer simulates the Z-spectrum → the simulated spectrum is compared to the measured one. This is a fundamentally different computational graph that enables:
   - Pulsed-CEST sequence modeling (not just CW approximation)
   - Adjoint-method backpropagation (memory-efficient for large volumes)
   - Natural handling of multi-pool systems without analytic simplifications

3. **Pool-Decoupled Heads with Built-In Heteroscedastic Uncertainty:** No CEST quantification method provides per-parameter, per-voxel, calibrated uncertainty maps. This is critical for clinical adoption — a radiologist needs to know that the reported k_sw = 150 s⁻¹ has σ = ±12 s⁻¹ in the tumor core but σ = ±80 s⁻¹ at the necrotic rim.

4. **Residual Correction for Model Mismatch:** The δ_net component explicitly addresses the gap between simplified BM models and real tissue physics (semi-solid MT, partial volume). This is a principled way to retain physics backbone benefits while adapting to real-world signals — strictly novel in the CEST context.

5. **Asymmetry Consistency Loss:** Cross-validates the BM-ODE output against directly computed MTR_asym — a novel physics-informed regularization specific to the CEST problem.

### 7.2 Positioning Against Literature

```
                          Physics Rigor →
                   Low                    High
               ┌──────────────────────────────────┐
          High │  Transformer-CEST    │  BM-NODE-UQ  │ ← THIS WORK
               │  (2025)              │  (Ours)       │
  Inference    │                      │               │
  Speed    ────┤─────────────────────────────────────│
               │  deepCEST            │  NBMF         │
          Low  │  (2020)              │  (2024)        │
               │                      │               │
               └──────────────────────────────────────┘
```

### 7.3 Conference / Journal Targets

- **Primary:** MICCAI 2026 / Medical Image Analysis
- **Secondary:** NeurIPS 2026 (ML for Science track), MRM (Magnetic Resonance in Medicine)
- **Hook:** "Physics-informed, fast, uncertainty-aware CEST quantification validated on synthetic ground truth"

---

## 8. Full Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           BM-NODE-UQ Architecture                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   Z(Δω) ∈ ℝ^{N×H×W}                                                          │
│        │                                                                        │
│        ▼                                                                        │
│   ┌──────────────────────────┐                                                  │
│   │  1D Conv Stack (per voxel)│   Conv1D(1→32, k=3) → GELU → Conv1D(32→64)    │
│   │  + Δω-Sinusoidal PE       │   → GELU → Conv1D(64→128)                     │
│   │  + Concrete Dropout       │                                                 │
│   └──────────┬───────────────┘                                                  │
│              │  S ∈ ℝ^{N×128}                                                   │
│              ▼                                                                   │
│   ┌──────────────────────────┐    Pool Anchors P ∈ ℝ^{5×128}                   │
│   │  PACA Block               │◄── (learnable, init at known Δω)                │
│   │  (Cross-Attention)        │    water=0, amide=+3.5, amine=+2.0,            │
│   │  Q=S, K=P, V=P           │    NOE=−3.5, MT=−2.5 ppm                       │
│   │  4 heads, d_k=32         │                                                  │
│   │  + Residual + LayerNorm  │                                                  │
│   └──────────┬───────────────┘                                                  │
│              │                                                                   │
│              ▼                                                                   │
│   ┌──────────────────────────┐                                                  │
│   │  2× Self-Attention Blocks │   (standard pre-norm transformer)               │
│   │  + Feed-Forward (128→512  │                                                 │
│   │    →128) + GELU           │                                                 │
│   │  + Concrete Dropout       │                                                 │
│   └──────────┬───────────────┘                                                  │
│              │                                                                   │
│              ▼  mean-pool over offset dim                                        │
│         z_latent ∈ ℝ^{128}                                                      │
│              │                                                                   │
│    ┌─────────┼─────────┬────────────┐                                           │
│    ▼         ▼         ▼            ▼                                           │
│ ┌───────┐ ┌───────┐ ┌───────┐ ┌─────────┐                                      │
│ │Water  │ │Amide  │ │Amine  │ │NOE/MT   │   Each: MLP(128→128→64→out)          │
│ │Head   │ │Head   │ │Head   │ │Head     │   + GELU + Concrete Dropout           │
│ │       │ │       │ │       │ │         │                                        │
│ │→T₁w,σ²│ │→k_sw,σ²│ │→k_sw,σ²│ │→f,k,σ²  │   Physical activations:            │
│ │→T₂w,σ²│ │→f_s,σ² │ │→f_s,σ² │ │         │   softplus (rates), sigmoid (frac) │
│ └───┬───┘ └───┬───┘ └───┬───┘ └────┬────┘                                      │
│     └─────────┴─────────┴───────────┘                                           │
│                    │                                                             │
│              θ̂ = all predicted params                                            │
│                    │                                                             │
│     ┌──────────────┼──────────────┐                                             │
│     │              │              │                                              │
│     ▼              ▼              ▼                                              │
│ ┌────────┐   ┌──────────┐   ┌──────────┐                                       │
│ │L_param │   │ BM-ODE   │   │ δ_net    │                                       │
│ │(super- │   │ Solver   │   │ (residual│                                       │
│ │ vised) │   │ (adjoint │   │  correct)│                                       │
│ │        │   │  backprop)│   │          │                                       │
│ └────────┘   └────┬─────┘   └────┬─────┘                                       │
│                   │              │                                               │
│                   ▼              ▼                                               │
│              Ẑ_BM(Δω)  +   δ(Δω)  =  Ẑ_final(Δω)                              │
│                                   │                                              │
│                    ┌──────────────┤                                              │
│                    ▼              ▼                                              │
│              ┌──────────┐  ┌──────────┐                                         │
│              │ L_recon  │  │ L_asym   │                                         │
│              │ (NLL)    │  │ (MTRasym │                                         │
│              │          │  │ consist.)│                                         │
│              └──────────┘  └──────────┘                                         │
│                                                                                 │
│     + L_physics (ODE residual at collocation points)                            │
│     + L_smooth (TV on parameter maps)                                           │
│                                                                                 │
│     All weighted by learned log-variance (Kendall auto-weighting)               │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 9. Ablation Study Design (for the paper)

| Experiment | What's removed | Expected result |
|:---|:---|:---|
| w/o PACA (use self-attn only) | Pool-anchored cross-attention | ↓ accuracy on overlapping pools (amide/amine) |
| w/o BM-ODE (MLP decoder) | Physics decoder | ↓ generalization to unseen B₁/B₀; non-physical outputs |
| w/o Pool-Decoupled Heads | Single shared head | ↑ parameter coupling errors |
| w/o Uncertainty | Remove σ² outputs | No change in mean accuracy; lose clinical confidence maps |
| w/o δ_net | Residual correction | ↓ on real data (if tested); no change on synthetic |
| w/o L_asym | Asymmetry loss | ↓ consistency between parametric and non-parametric CEST metrics |
| CW vs. Pulsed ODE | Matrix exponential vs. Neural ODE | ↓ accuracy for pulsed sequences; CW works for CW data |

---

## 10. Limitations & Honest Scope

> [!WARNING]
> - The architecture assumes access to a reasonably good BM model of the tissue. If the number of pools is wrong (e.g., 3-pool model but 5-pool tissue), the residual δ_net helps but can't fully compensate.
> - Uncertainty calibration needs validation on real patient data — synthetic noise models may not capture all real-world noise sources.
> - The Neural ODE layer adds computational cost during training (~3× vs. plain MLP). Inference is still fast because the ODE only runs forward (no adjoint needed at test time).
> - Spatial processing is currently patch-level, not full-volume. A future extension would add a 2D U-Net spatial encoder before the spectral encoder.

---

## Open Questions

> [!IMPORTANT]
> 1. **Number of pools:** Should we model 2-pool (water + amide only, simplest), 3-pool (+ amine), or 4-pool (+ NOE/MT)? More pools = more parameters = harder optimization but more complete physics. **Recommendation:** Start with 3-pool (water, amide, amine), add MT as a semi-solid approximation in δ_net.
> 2. **Spatial context:** The current design is primarily voxel-wise with optional patch processing. Do you want to add a spatial encoder (e.g., 2D conv layers before spectral processing) for full-slice inference? This would enable the spatial smoothness loss but increases complexity.
> 3. **Training data:** Will you generate synthetic data from your own BM simulator, or use an existing open-source one (e.g., Pulseq-CEST, BMsim)?
> 4. **Target venue/deadline:** This affects how much ablation and validation is needed in the first version.
