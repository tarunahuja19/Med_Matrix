# 🧪 Attempt Log: Phase 3 Model Training & C++ Inference Integration

*   **Date**: 2026-06-20 / 2026-06-21
*   **Status**: 🟢 Solved

---

## 🎯 Goal
Train the Fused S4-CNN MRI classifier on synthetic volumetric K-space data and connect the production ONNX model to the C++ inference runtime:
1. **Synthetic Data Generation**: Generate 600 multi-coil K-space patients (8 slices, 16 coils, 128×128 resolution).
2. **Model Architecture Revision**: Replace the 67M-parameter 3D Conv encoder with a parameter-efficient 2D Conv encoder (~281k params).
3. **Multi-GPU Training**: Train on 2× Tesla T4 GPUs via Kaggle with Distributed Data Parallel (DDP) for 50 epochs.
4. **ONNX Export**: Export the trained model to `fused_model_128.onnx` with dynamic batch support.
5. **C++ Runtime Update**: Update the C++ inference engine's `KSpaceDims`, tensor shapes, and build configuration to match the new model.
6. **Verification**: Pass all 10 C++ inference tests on both CPU and CUDA backends.

---

## 🗺️ Background / Context
- Workspace: `/home/jemin/Projects/Med_Matrix`
- Remote training: Kaggle notebook (`PhantomNet_multiclass600.ipynb`)
- Model file: `fused_model_128.onnx` (4.3 MB)
- Previous models used 64 slices × 1 channel → new model uses 8 slices × 16 coils

---

## 🔬 Attempts Tracker

### Attempt 1: Initial Training (100 patients, 10 epochs)
*   **What was tried**: Generated 100 synthetic patients, trained for 10 epochs on Kaggle.
*   **Result**: Final val accuracy 42.86%. Model was too large (~67M params) and underfitting on small data.

### Attempt 2: Resume Training (100 new patients, 10 more epochs)
*   **What was tried**: Generated 100 new patients and resumed from checkpoint.
*   **Result**: Final val accuracy improved to 50.00%. Still underfitting — needed more data and epochs.
*   **Issue**: ONNX export initially failed due to missing `onnxscript` library; fixed by using legacy TorchScript exporter.

### Attempt 3: Scale Up (300 patients, 256 resolution, 50 epochs)
*   **What was tried**: Scaled to 300 patients at 256×256 resolution, trained for 50 epochs with dropout and weight decay.
*   **Result**: Peak val accuracy 57.81%, final 56.25%. Higher resolution didn't help much — model was too large for the data size, showing heavy overfitting (train loss near 0 but val loss diverging).

### Attempt 4: Parameter-Efficient Architecture (600 patients, 128 resolution, 50 epochs) ✅
*   **What was tried**: 
    - Replaced the 3D Conv encoder with a lightweight 2D Conv encoder (67M → 281k params)
    - Scaled to 600 patients at 128×128 resolution
    - 8 slices × 16 coils input format
    - 50 epochs with dropout (0.3) and weight decay
*   **Result**: **Peak val accuracy 88.28%** (epoch 40), final 87.50%. Training loss stable around 0.31 without overfitting.

### Attempt 5: C++ Runtime Integration ✅
*   **What was tried**:
    - Updated `KSpaceDims` from `[1, 64, 1, 128, 128, 2]` to `[1, 8, 16, 128, 128, 2]`
    - Updated CMakeLists.txt with protobuf/abseil transitive dependencies
    - Fixed system package mismatch (abseil-cpp 2601 → 2605, protobuf 34 → 35)
    - Ran full 10-test suite
*   **Result**: 10/10 tests passing on both CPU and CUDA. GPU throughput: ~84.8 inf/sec (11.8ms).

---

## 🛑 Blockers / Issues

### Issue 1: ONNX Export Missing `onnxscript`
- **Symptom**: `torch.onnx.export()` failed with missing `onnxscript` library on Kaggle.
- **Root Cause**: PyTorch 2.x's new dynamo-based ONNX exporter requires `onnxscript` as a dependency.
- **Fix**: Fell back to legacy TorchScript-based ONNX exporter (`torch.onnx.export` with `opset_version=17`).

### Issue 2: System Package ABI Mismatch (Arch Linux)
- **Symptom**: Linker errors with undefined references to `absl::lts_20260526::*` and `google::protobuf::*` symbols.
- **Root Cause**: `onnxruntime-cuda` 1.24.4 was built against `abseil-cpp 20260526` and `protobuf 35`, but the system had older versions (`abseil 2601`, `protobuf 34`).
- **Fix**: Updated system packages with `sudo pacman -Sy abseil-cpp protobuf`.

### Issue 3: ONNX Runtime Transitive Dependencies
- **Symptom**: `libonnxruntime.so` has `DT_NEEDED` entries for abseil/protobuf shared libraries but CMake's basic `pkg_check_modules` for onnxruntime only links `-lonnxruntime`.
- **Root Cause**: ONNX Runtime's pkg-config doesn't declare its transitive dependencies on protobuf/abseil.
- **Fix**: Added protobuf's full `STATIC_LIBRARIES` output (which includes all abseil modules) to the CMake link line.

### Issue 4: Model Overfitting with Large Architecture
- **Symptom**: Training loss approached 0 while validation loss diverged after epoch 20.
- **Root Cause**: The original 67M-parameter 3D Conv architecture was far too large for the synthetic training dataset (100–300 patients).
- **Fix**: Switched to parameter-efficient 2D Conv encoder (~281k params) which generalized dramatically better.

---

## 🏁 Resolution
*   **Root Cause**: Oversized model architecture, missing build dependencies, and system package version skew.
*   **Solution**: Redesigned the CNN branch with lightweight 2D convolutions, updated system packages, fixed CMake transitive deps, and validated with comprehensive C++ test suite.
*   **Lessons Learned**:
    1. Model capacity must match dataset size — ~281k params worked far better than 67M for 600 synthetic patients.
    2. On Arch Linux rolling-release, always ensure `onnxruntime-cuda`, `abseil-cpp`, and `protobuf` are updated together — ABI breakage is common.
    3. When linking against shared libraries with unresolved transitive deps, use the full `PROTOBUF_STATIC_LIBRARIES` from pkg-config rather than just `PROTOBUF_LIBRARIES`.
    4. 128×128 resolution is more practical than 256×256 for K-space classification — lower memory, faster training, comparable accuracy.
