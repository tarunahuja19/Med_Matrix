# 🤖 Fused S4-CNN Model & C++ Inference Engine

This document outlines the detailed architecture of the Hybrid Fused Volumetric MRI Classifier (S4 + Conv2D CNN) and specifies the exact inner workings of the C++ inference engine using ONNX Runtime.

---

## 1. Model Architecture & Parameters

The model is designed as a hybrid dual-branch architecture for patient-level volumetric K-space slice classification. It processes raw multi-slice, multi-coil complex K-space data through parallel Frequency (S4 SSM) and Spatial (Conv2D CNN) branches, fusing their outputs using Slice-Level Cross-Attention.

**Key Stats**: ~281k trainable parameters, 4.3 MB ONNX file, 88.28% peak validation accuracy on 600 synthetic patients.

```
                  Raw Volumetric K-Space Input
               [B=1, S=8, Coils=16, H=128, W=128, RI=2]
                               |
            +------------------+------------------+
            |                                     |
            v                                     v
     Frequency Branch                       Spatial Branch
  (Complex Linear Encoder)            (Differentiable 2D IFFT)
            |                                     |
            v                                     v
   (2-Layer S4D Block)              (Parameter-Efficient Conv2D)
            |                                     |
            v                                     v
   Sequence of Tokens                    Sequence of Tokens
    [B, S, d_model=128]                   [B, S, d_model=128]
            |                                     |
            +------------------+------------------+
                               |
                               v
                     Cross-Attention Fusion
                  (CNN=Query, S4=Key/Value)
                               |
                               v
                      Global Average Pool
                               |
                               v
                       Linear Classifier
                       (11 Disease Classes)
```

### 1.1 Frequency Branch (S4D SSM)
*   **Purpose:** Captures raw acquisition frequency/phase-encoding sequences across slices directly in the complex domain.
*   **Projection:** Flattens each K-space slice ($C \times H \times W = 16 \times 128 \times 128 = 262144$) and projects it to a $128$-dimensional complex latent space via `input_dim=16384` (down-projected from coil-flattened representation).
*   **Recurrence:** Uses a 2-layer Diagonal Continuous State Space Model (S4D) with state dimension $N=16$ per channel, discretized using Bilinear (Tustin) transformation.
*   **Output:** The complex recurrence outputs are converted back to real values by computing their magnitude, producing sequence tokens: $Z_{\text{s4}} \in \mathbb{R}^{B \times S \times 128}$.

### 1.2 Spatial Branch (Parameter-Efficient Conv2D CNN)
*   **Purpose:** Learns spatial features from reconstructed magnitude images using lightweight 2D convolutions.
*   **Reconstruction:** Applies a differentiable 2D IFFT on the raw K-space slices, computes the root-sum-of-squares (RSS) combination across 16 coils, producing per-slice magnitude images.
*   **Convolutions:** Passes each slice through a parameter-efficient Conv2D encoder with BatchNorm and ReLU activations. This reduces model parameters from ~67M (old 3D Conv approach) to ~281k total.
*   **Pooling:** Applies adaptive average pooling to reduce spatial dimensions, then maps to `d_model=128`.
*   **Output:** The sequence feature tensor: $Z_{\text{cnn}} \in \mathbb{R}^{B \times S \times 128}$.

### 1.3 Slice-Level Cross-Attention Fusion
*   **Mechanism:** Uses the spatial features ($Z_{\text{cnn}}$) as the Query ($Q$) and frequency features ($Z_{\text{s4}}$) as the Key ($K$) and Value ($V$).
*   **Attention Matrix:** Computes scaled dot-product attention mapping correlations between the spatial profile of slice $i$ and the frequency profile of slice $j$.
*   **Pooling & Head:** Adds residual connection from Query, applies global average pooling across the slice sequence ($S=8$), dropout (0.3), and projects to classification logits for 11 pathology classes.

---

## 2. ONNX Model Export

Since ONNX does not natively support PyTorch's complex number types and certain dynamic pooling operations, the model is exported using a real-valued twin architecture:

*   **Real/Imaginary Representation:** Complex inputs and parameters are split into stacked real and imaginary channels. The raw K-space input shape is represented as `[B, S=8, Coils=16, H=128, W=128, 2]` where the last dimension holds `[Real, Imag]`.
*   **Complex Math Emulation:** Layers like `ComplexLinear` and `S4D` recurrence are mathematically emulated using real-valued operations:
    $$\text{Re}(u) = W_{\text{real}} x_{\text{real}} - W_{\text{imag}} x_{\text{imag}} + b_{\text{real}}$$
    $$\text{Im}(u) = W_{\text{real}} x_{\text{imag}} + W_{\text{imag}} x_{\text{real}} + b_{\text{imag}}$$
*   **Pooling Compatibility:** `AdaptiveAvgPool3D` was replaced with direct `torch.mean` reduction operations along spatial dimensions to ensure static computational graph export.
*   **Export File:** `fused_model_128.onnx` (4.3 MB, ONNX opset 17, IR version 8).
*   **Dynamic Batch:** The batch dimension is dynamic (`batch_size`), allowing variable batch inference.

---

## 3. C++ Inference Logic

The C++ inference engine is implemented using **ONNX Runtime (ORT) v1.24.4** in `kvision_inference.cpp`. It loads the model, sets up execution providers, runs the forward pass, and parses the outputs.

### 3.1 Tensor Dimensions & Layouts
*   **Input Name:** `kspace_real_imag`
*   **Input Shape:** `[1, 8, 16, 128, 128, 2]` (Batch, Slices, Coils, Height, Width, Real/Imaginary)
*   **Input Type:** `float32` (flat vector of size $1 \times 8 \times 16 \times 128 \times 128 \times 2 = 4,194,304$ elements)
*   **Output Name:** `logits`
*   **Output Shape:** `[1, 11]` (Batch, Classes)
*   **Output Type:** `float32`

### 3.2 Step-by-Step Execution Pipeline

1.  **Engine Initialization:**
    *   Creates an `Ort::Env` log environment.
    *   Sets session options including intra-op threads (CPU execution) and graph optimizations (`ORT_ENABLE_ALL`).
    *   Attempts to register the **CUDA Execution Provider** for hardware acceleration. If CUDA initialization fails, it gracefully falls back to the default CPU provider.
    *   Loads the ONNX model from disk and caches the input/output names and shapes.

2.  **Input Preparation:**
    *   Validates that the input size matches `KSpaceDims::total_elements` (4,194,304).
    *   Wraps the raw pointer to flat K-space float data in an `Ort::Value` tensor using `Ort::MemoryInfo::CreateCpu`. This avoids unnecessary copies.

3.  **Forward Pass (`Ort::Session::Run`):**
    *   Executes the session using CUDA/CPU providers.
    *   Measures wall-clock inference execution time in milliseconds.

4.  **Post-Processing:**
    *   **Softmax:** Computes softmax over the raw classification logits to output class probabilities:
        $$p_i = \frac{e^{z_i - \max(z)}}{\sum_j e^{z_j - \max(z)}}$$
        *(Using subtraction of the max logit for numerical stability).*
    *   **Argmax:** Finds the class index with the highest probability.
    *   **InferenceResult Packaging:** Packages the predicted class enum, confidence probability, raw logits, and execution time.

### 3.3 Build Dependencies (Arch Linux / EndeavourOS)
*   **Required packages**: `onnxruntime-cuda` (or `onnxruntime-cpu`), `abseil-cpp`, `protobuf`
*   **CMake** uses `pkg-config` to find ONNX Runtime and protobuf with all abseil transitive dependencies.
*   **Note**: The `onnxruntime-cuda` package requires matching versions of `abseil-cpp` and `protobuf`. Ensure all three are updated together to avoid linker errors.

### 3.4 Performance Benchmarks
| Backend | Avg Inference Time | Throughput |
|:---|:---:|:---:|
| **CPU** (4 threads) | 79.2 ms | ~12.6 inf/sec |
| **CUDA** (GPU) | 11.8 ms | ~84.8 inf/sec |

### 3.5 Test Suite (10 tests, all passing)
1. Model loads successfully
2. Predicted class in valid range [0, 10]
3. Confidence is a valid probability
4. All 11 logits are finite
5. Softmax probabilities sum to ~1.0
6. Same input → same predicted class (determinism)
7. Same input → same logits (determinism)
8. Different inputs → different logits (sensitivity)
9. Invalid input size → throws `std::invalid_argument`
10. Average inference time is positive (throughput benchmark)

---

## 4. Pathology Classes (11 Types)

The index maps directly to the output probabilities:

| Index | Pathology Class | Description / Target |
| :---: | :--- | :--- |
| **0** | `Normal` | Healthy MRI scan |
| **1** | `Edema` | Brain tissue swelling |
| **2** | `Tumor_Glioma` | Glioma tumor structure |
| **3** | `Tumor_Meningioma` | Meningioma tumor structure |
| **4** | `Hemorrhage` | Bleeding in the brain |
| **5** | `Ischemia` | Restricted blood supply / stroke |
| **6** | `MS_Lesions` | Multiple Sclerosis plaques |
| **7** | `Atrophy` | Shrinkage of brain volume |
| **8** | `Hydrocephalus` | Accumulation of cerebrospinal fluid |
| **9** | `AVM` | Arteriovenous Malformation |
| **10**| `Abscess` | Pus-filled brain lesion |

---

## 5. Training History

### Run 4 (Production — Current Model: `fused_model_128.onnx`)
*   **Date**: 2026-06-21
*   **Platform**: Kaggle, 2× Tesla T4 GPUs with DDP
*   **Dataset**: 600 synthetic patients, 128×128 resolution, 8 slices, 16 coils
*   **Architecture**: Parameter-efficient Conv2D encoder (~281k params, down from ~67M)
*   **Training**: 50 epochs, dropout + weight decay regularization
*   **Peak Validation Accuracy**: **88.28%** (epoch 40)
*   **Final Validation Accuracy**: **87.50%**
*   **C++ Runtime**: Verified with 10/10 tests passing on both CPU and CUDA backends
