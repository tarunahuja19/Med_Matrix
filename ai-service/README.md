# 🤖 Python AI Service (`ai-service`)

This directory houses the **MedMatrix Python AI Service**, a FastAPI-based microservice responsible for:
- Medical raw K-space input parsing (`Siemens Twix`, `fastMRI HDF5`, and `NumPy`).
- 2D centered IFFT reconstructions, RSS coil combination, and zero/first-order phase corrections.
- Image preprocessing (BM3D/DnCNN denoising, SimpleITK rigid motion correction).
- Volumetric artifact classification (ghosting, wrap-around, zipper noise).
- Deep learning inference using the **Hybrid Fused S4-CNN Volumetric Classifier** and **SSM Anomaly Estimator**.
- LLM Retrieval-Augmented Generation (RAG) report compilation and PDF generation.

---

## 🏛️ Deep Learning Model Architectures

The AI service hosts two primary deep learning networks deployed as ONNX graphs:

### 1. Hybrid Fused S4-CNN Volumetric Classifier
* **Input Shape:** `[Batch, Slices=8, Coils=16, Height=128, Width=128, Real/Imaginary=2]`
* **Frequency Sequence Branch (S4D SSM):** Projects K-space coordinates into a 128-dimensional complex latent space and runs a 2-layer Diagonal State Space Model recurrence to capture frequency features directly in the complex domain.
* **Spatial Branch (Conv2D CNN):** Reconstructs magnitude images via differentiable 2D IFFT and RSS coil combination, passing them through a parameter-efficient 2D convolutional encoder to capture spatial features.
* **Slice-Level Cross-Attention:** Uses spatial CNN features as the Query ($Q$) and frequency S4 SSM features as Key ($K$) and Value ($V$) to fuse sequence tokens across slices. 
* **Details:** This lightweight approach reduces parameter size from ~67M to **~281k parameters** while achieving **88.28% validation accuracy** across 11 pathology classes (Normal, Edema, Glioma, Meningioma, Hemorrhage, Ischemia, MS Lesions, Atrophy, Hydrocephalus, AVM, Abscess).
* **Model Files:** `fused_model_128.pt` (PyTorch) and `fused_model_128.onnx` (4.3 MB ONNX twin).

### 2. State Space (SSM) K-Space Anomaly Estimator
* **Input Shape:** `kspace`: `[Batch, 32, 256, 256]` (stacked real/imaginary parts of 16 coils) and `contrast`: `[Batch]` (Int64: `0` for T1, `1` for T2).
* **Architecture:** Ingests complex multi-coil K-space row-by-row using real-valued diagonal State Space Models combined with contrast embeddings to regress continuous corruption levels.
* **Output Shape:** `[Batch, 3]` predicting continuous parameters: `[noise_severity, motion_severity, phase_severity]`.
* **Model Files:** `anomaly_detector.pt` (PyTorch) and `anomaly_detector.onnx` (5.9 MB ONNX twin).

---

## 🗺️ File Map & Navigation Directory

Below is a detailed guide of all key python scripts and directories within `ai-service/`:

### 🤖 Core Application & REST APIs
* **[`main.py`](file:///home/jemin/Projects/Med_Matrix/ai-service/main.py):** FastAPI app entrypoint. Exposes `/health`, `/reconstruct`, `/predict` (AI cascade), and RAG report endpoints.
* **[`models.py`](file:///home/jemin/Projects/Med_Matrix/ai-service/models.py):** Pydantic schemas validating `/predict` input structures and API responses.
* **[`rag_agent/`](file:///home/jemin/Projects/Med_Matrix/ai-service/rag_agent):** Sub-module managing research paper parsing, embedding chunks via OpenAI/Gemini, and Redis document storage/queries.

### 🧠 Model Code & Training Scripts
* **[`fused_model.py`](file:///home/jemin/Projects/Med_Matrix/ai-service/fused_model.py):** PyTorch definition of the complex-valued hybrid S4-CNN Volumetric Classifier.
* **[`fused_model_onnx.py`](file:///home/jemin/Projects/Med_Matrix/ai-service/fused_model_onnx.py):** ONNX-compatible real-valued twin model that emulates complex operations using real matrices to allow export.
* **[`s4_model.py`](file:///home/jemin/Projects/Med_Matrix/ai-service/s4_model.py):** State Space Model (SSM) sequence layer code.
* **[`cnn_model.py`](file:///home/jemin/Projects/Med_Matrix/ai-service/cnn_model.py):** Conv2D spatial branch feature extractor.
* **[`anomaly_detector_model.py`](file:///home/jemin/Projects/Med_Matrix/ai-service/anomaly_detector_model.py):** PyTorch definition of the SSM Anomaly Estimator.
* **[`train.py`](file:///home/jemin/Projects/Med_Matrix/ai-service/train.py):** Multi-GPU Distributed Data Parallel (DDP) training script for the volumetric classifier.
* **[`train_anomaly_detector.py`](file:///home/jemin/Projects/Med_Matrix/ai-service/train_anomaly_detector.py):** Training and export pipeline for the SSM Anomaly Estimator.

### 🔄 Input Pipelines, Reconstruction, & Image Correction
* **[`kspace_reader.py`](file:///home/jemin/Projects/Med_Matrix/ai-service/kspace_reader.py):** File parser for Siemens Twix binary `.dat`, fastMRI HDF5 `.h5`, and NumPy `.npy` formats.
* **[`reconstruction.py`](file:///home/jemin/Projects/Med_Matrix/ai-service/reconstruction.py):** 2D Centered IFFT, phase correction, and Root-Sum-of-Squares (RSS) coil combination algorithms.
* **[`motion_correction.py`](file:///home/jemin/Projects/Med_Matrix/ai-service/motion_correction.py):** Retrospective rigid image registration using SimpleITK.
* **[`denoiser.py`](file:///home/jemin/Projects/Med_Matrix/ai-service/denoiser.py):** BM3D baseline and PyTorch DnCNN residual image denoiser.
* **[`artifact_detector.py`](file:///home/jemin/Projects/Med_Matrix/ai-service/artifact_detector.py):** ResNet CNN artifact classifier computing ghosting, wrap-around, and zipper noise scores.

### 🛠️ Export & Validation Utilities
* **[`export_onnx.py`](file:///home/jemin/Projects/Med_Matrix/ai-service/export_onnx.py):** Maps volumetric classifier weights from a `.pt` checkpoint to the real-valued ONNX twin model.
* **[`export_fused_onnx.py`](file:///home/jemin/Projects/Med_Matrix/ai-service/export_fused_onnx.py):** Fused model export helper utility.
* **[`evaluate_onnx.py`](file:///home/jemin/Projects/Med_Matrix/ai-service/evaluate_onnx.py):** Compares predictions between the PyTorch model and the exported ONNX model to verify correctness.
* **[`verify_model.py`](file:///home/jemin/Projects/Med_Matrix/ai-service/verify_model.py):** Quick validation script inspecting model variables.
* **[`find_max_batch.py`](file:///home/jemin/Projects/Med_Matrix/ai-service/find_max_batch.py):** GPU utility searching for the largest batch size that fits in memory without throwing CUDA Out-Of-Memory.
* **[`python_sim.py`](file:///home/jemin/Projects/Med_Matrix/ai-service/python_sim.py):** Numerical brain phantom K-space simulator.
* **[`generate_synthetic_dataset.py`](file:///home/jemin/Projects/Med_Matrix/ai-service/generate_synthetic_dataset.py):** Synthesizes multi-slice, multi-coil K-space dataset arrays.
* **[`create_demo_patient.py`](file:///home/jemin/Projects/Med_Matrix/ai-service/create_demo_patient.py):** Populates the backend database with dummy patient records for presentation.
* **[`generate_5_patients.py`](file:///home/jemin/Projects/Med_Matrix/ai-service/generate_5_patients.py):** Populates synthetic patients.
* **[`pre_reconstruct_dataset.py`](file:///home/jemin/Projects/Med_Matrix/ai-service/pre_reconstruct_dataset.py):** Pre-processes raw dataset inputs to save training time.

### 🧪 Unit & Integration Test Suite
* **[`test_reconstruction.py`](file:///home/jemin/Projects/Med_Matrix/ai-service/test_reconstruction.py):** Verifies 2D centered IFFT and RSS coil combination.
* **[`test_artifact_detector.py`](file:///home/jemin/Projects/Med_Matrix/ai-service/test_artifact_detector.py):** Verifies the ResNet artifact classifier.
* **[`test_pipeline.py`](file:///home/jemin/Projects/Med_Matrix/ai-service/test_pipeline.py):** Integration tests for the full reconstruction, gating, and prediction pipeline.

---

## ⚙️ Setup & Local Execution

### 1. Install dependencies
Make sure you are within your virtual environment (`source .venv/bin/activate`) and run:
```bash
pip install -r requirements.txt
```

### 2. Run the FastAPI Application
Start the service locally on port `8000`:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
Interactive API documentation will be available at [http://localhost:8000/docs](http://localhost:8000/docs).

### 3. Run Unit and Integration Tests
Execute the Python test suite:
```bash
pytest test_reconstruction.py
pytest test_artifact_detector.py
pytest test_pipeline.py
```
