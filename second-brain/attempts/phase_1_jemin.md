# 🧪 Attempt Log: Phase 1 Input Pipeline & Reconstruction (Jemin)

*   **Date**: 2026-06-16
*   **Status**: 🟢 Solved

---

## 🎯 Goal
Implement the core input pipeline and reconstruction tasks for Jemin:
1. **K-space Reader**: Support Siemens Twix `.dat`, HDF5 `.h5`, and NumPy arrays.
2. **Reconstruction**: Centered 2D IFFT, RSS coil combination, and dual phase correction (1D zero-order and 2D low-res phase maps).
3. **Motion Correction**: Retrospective rigid registration (Euler 2D/3D transforms, Mean Squares similarity, Regular Step Gradient Descent optimizer) using SimpleITK.
4. **Denoiser**: Non-Local Means (NLM) baseline and PyTorch residual DnCNN model.
5. **Artifact Detection**: ResNet-based CNN classifier to detect ghosting, wrap-around, and zipper noise.
6. **FastAPI Service**: Expose endpoints to process studies and verify using automated pytest.

---

## 🗺️ Background / Context
- Workspace: `/home/jemin/Projects/MRI/KVISION/ai-service`
- Monorepo using `pnpm` workspaces for Electron frontend and Node backend.

---

## 🔬 Attempts Tracker

### Attempt 1: Subagent Implementation and Integration
*   **What was tried**: Spawned three parallel `ml_engineer` subagents to write the respective modules (`kspace_reader.py`, `reconstruction.py`, `motion_correction.py`, `denoiser.py`, `artifact_detector.py`). After they finished, we integrated the modules into `main.py` and wrote `test_pipeline.py`.
*   **Result**: The code was written cleanly. However, running pytest revealed a few issues with package dependencies and path configuration.

---

## 🛑 Blockers / Issues

### Issue 1: Missing pywavelets Dependency
- **Symptom**: pytest failed on NLM denoising: `Denoising failed: PyWavelets is not installed. Please ensure it is installed in order to use this function.`
- **Root Cause**: `scikit-image`'s `estimate_sigma` uses wavelets to estimate noise levels, requiring the `pywavelets` library.
- **Fix**: Added `pywavelets` to `requirements.txt` and re-installed.

### Issue 2: NumPy .npy Auto-appending
- **Symptom**: `load_kspace` failed to locate `/tmp/tmpXXXX/kspace_input`.
- **Root Cause**: In our integration test mock, we used `np.save(file_path, data)`. Since `file_path` had no extension, `np.save` automatically appended `.npy`, leaving the file named `kspace_input.npy` instead of `kspace_input`.
- **Fix**: Replaced `np.save` with `shutil.copy` of the already saved temp `.npy` file.

### Issue 3: Missing Extension in Temporary Files
- **Symptom**: `load_kspace` threw: `ValueError: Unsupported file format: .`
- **Root Cause**: In `main.py`, the temporary download file was named `kspace_input` without an extension. The loader relies on `os.path.splitext` to route the file to the correct parser.
- **Fix**: Modified `main.py` to extract and preserve the extension from the MinIO `kspace_key` (e.g. `kspace_input.h5`).

---

## 🏁 Resolution
*   **Root Cause**: Incorrect path extension routing and missing sub-dependencies in `scikit-image`.
*   **Solution**: Resolved path extensions, added `pywavelets`, and ran the full 13-test suite which passed successfully in 6.8 seconds.
*   **Lessons Learned**:
    1. Always preserve file extensions when writing temporary downloads, especially when parsing code relies on extensions.
    2. Deep learning and image-processing libraries sometimes have optional soft dependencies (like `pywavelets` for `scikit-image`) that must be explicitly installed.
