# 🧠 MedMatrix Second Brain Vault Index

Welcome to the **MedMatrix Developer & AI Agent Second Brain Vault**. This index maps all documentation, memory notes, and core codebase source files for direct accessibility.

---

## 💾 Vault Documentation Index
Use these links to navigate through the persistent memories and attempt logs:

### 🏛️ Core Memories
*   [memories/architecture.md](memories/architecture.md) — Main system architecture, technology stack, data flow, and database schema mappings.
*   [memories/inference.md](memories/inference.md) — Fused S4-CNN model details, C++ ONNX Runtime inference engine, benchmarks, and test suite.
*   [memories/radiology_agent_rag.md](memories/radiology_agent_rag.md) — RAG-based Radiology Reporting Agent architecture, database schema, and pipelines.

### 🧪 Task & Attempt Logs
*   [attempts/template.md](attempts/template.md) — Standard markdown template for creating new attempt logs.
*   [attempts/phase_1_jemin.md](attempts/phase_1_jemin.md) — Phase 1: K-Space loader, reconstructions, motion correction, denoiser, and automated testing.
*   [attempts/phase_2_gating_logic.md](attempts/phase_2_gating_logic.md) — Phase 2: FastAPI `/predict` endpoint, Prisma schemas, Axios Node.js AI client, and BullMQ background worker.
*   [attempts/phase_3_training_and_cpp.md](attempts/phase_3_training_and_cpp.md) — Phase 3: Multi-GPU model training (4 iterations → 88.28% accuracy), ONNX export, and C++ runtime integration.

---

## 🛠️ Codebase File Index
This directory links to key files in the MedMatrix workspace, sorted by monorepo component:

### 🐳 DevOps & Environment
*   [docker-compose.yml](file:///home/jemin/Projects/Med_Matrix/docker-compose.yml) — Backing services orchestration (Postgres, Redis, MinIO, AI-Service, Backend).
*   [package.json](file:///home/jemin/Projects/Med_Matrix/package.json) — Monorepo entrypoint configuration and execution scripts.
*   [pnpm-workspace.yaml](file:///home/jemin/Projects/Med_Matrix/pnpm-workspace.yaml) — Defines `pnpm` workspaces for the monorepo.
*   [.gitignore](file:///home/jemin/Projects/Med_Matrix/.gitignore) — Git ignore patterns.
*   [.env](file:///home/jemin/Projects/Med_Matrix/.env) — Environment variables.

### 🧠 Trained Models & Training Artifacts
*   [fused_model_128.onnx](file:///home/jemin/Projects/Med_Matrix/ai-service/fused_model_128.onnx) — Production ONNX model (128-res, 8 slices × 16 coils, ~281k params, 88.28% peak accuracy).
*   [fused_model_128.pt](file:///home/jemin/Projects/Med_Matrix/ai-service/fused_model_128.pt) — PyTorch checkpoint for the production model.
*   [anomaly_detector.onnx](file:///home/jemin/Projects/Med_Matrix/ai-service/anomaly_detector.onnx) — Production ONNX anomaly detector model (256-res, 32 channels, ~546k params).
*   [anomaly_detector.pt](file:///home/jemin/Projects/Med_Matrix/ai-service/anomaly_detector.pt) — PyTorch checkpoint for the production anomaly detector model.
*   [training_progress.log](file:///home/jemin/Projects/Med_Matrix/training_progress.log) — Live training log synced from Kaggle remote GPU server.
*   [PhantomNet_multiclass600.ipynb](file:///home/jemin/Projects/Med_Matrix/PhantomNet_multiclass600.ipynb) — Kaggle notebook for remote GPU training (600 patients).

### 🤖 Python AI Service (`ai-service/`)

#### Core Application
*   [ai-service/main.py](file:///home/jemin/Projects/Med_Matrix/ai-service/main.py) — FastAPI app hosting `/health`, `/reconstruct`, and `/predict` endpoints.
*   [ai-service/models.py](file:///home/jemin/Projects/Med_Matrix/ai-service/models.py) — Pydantic schemas validating `/predict` request/response types.
*   [ai-service/Dockerfile](file:///home/jemin/Projects/Med_Matrix/ai-service/Dockerfile) — Docker container definition for the AI service.
*   [ai-service/requirements.txt](file:///home/jemin/Projects/Med_Matrix/ai-service/requirements.txt) — Python package dependencies.

#### Model Architecture
*   [ai-service/fused_model.py](file:///home/jemin/Projects/Med_Matrix/ai-service/fused_model.py) — Hybrid S4-CNN Fused Volumetric Classifier module (complex-valued, for training).
*   [ai-service/fused_model_onnx.py](file:///home/jemin/Projects/Med_Matrix/ai-service/fused_model_onnx.py) — ONNX-compatible real-valued twin model module (for export).
*   [ai-service/cnn_model.py](file:///home/jemin/Projects/Med_Matrix/ai-service/cnn_model.py) — Parameter-efficient Conv2D spatial branch module.
*   [ai-service/s4_model.py](file:///home/jemin/Projects/Med_Matrix/ai-service/s4_model.py) — Complex-valued S4 sequence branch module.
*   [ai-service/anomaly_detector_model.py](file:///home/jemin/Projects/Med_Matrix/ai-service/anomaly_detector_model.py) — State Space Model (SSM) based MRI Anomaly Estimator.

#### Training & Export
*   [ai-service/train.py](file:///home/jemin/Projects/Med_Matrix/ai-service/train.py) — Multi-GPU DDP training script.
*   [ai-service/train_anomaly_detector.py](file:///home/jemin/Projects/Med_Matrix/ai-service/train_anomaly_detector.py) — Anomaly detector training and ONNX export script.
*   [ai-service/evaluate_onnx.py](file:///home/jemin/Projects/Med_Matrix/ai-service/evaluate_onnx.py) — ONNX validation evaluation runner script.
*   [ai-service/export_onnx.py](file:///home/jemin/Projects/Med_Matrix/ai-service/export_onnx.py) — Export script mapping trained checkpoint weights to the real-valued ONNX model.
*   [ai-service/export_fused_onnx.py](file:///home/jemin/Projects/Med_Matrix/ai-service/export_fused_onnx.py) — Fused model ONNX export helper.
*   [ai-service/generate_synthetic_dataset.py](file:///home/jemin/Projects/Med_Matrix/ai-service/generate_synthetic_dataset.py) — Volumetric K-space synthetic generator script.
*   [ai-service/find_max_batch.py](file:///home/jemin/Projects/Med_Matrix/ai-service/find_max_batch.py) — GPU memory batch-size finder utility.
*   [ai-service/python_sim.py](file:///home/jemin/Projects/Med_Matrix/ai-service/python_sim.py) — Numerical brain phantom K-space simulator.
*   [ai-service/numerical_brain_cropped.mat](file:///home/jemin/Projects/Med_Matrix/ai-service/numerical_brain_cropped.mat) — MATLAB phantom data for simulation.

#### Input Pipeline & Preprocessing
*   [ai-service/kspace_reader.py](file:///home/jemin/Projects/Med_Matrix/ai-service/kspace_reader.py) — Parses Siemens Twix `.dat`, fastMRI `.h5`, and NumPy `.npy` files.
*   [ai-service/reconstruction.py](file:///home/jemin/Projects/Med_Matrix/ai-service/reconstruction.py) — 2D centered IFFT, RSS coil combination, and zero/first-order phase correction.
*   [ai-service/motion_correction.py](file:///home/jemin/Projects/Med_Matrix/ai-service/motion_correction.py) — SimpleITK rigid registration using Mean Squares similarity.
*   [ai-service/denoiser.py](file:///home/jemin/Projects/Med_Matrix/ai-service/denoiser.py) — BM3D baseline and PyTorch residual DnCNN denoiser.
*   [ai-service/artifact_detector.py](file:///home/jemin/Projects/Med_Matrix/ai-service/artifact_detector.py) — ResNet classifier computing ghosting, wrap-around, and zipper noise scores.

#### Tests
*   [ai-service/test_pipeline.py](file:///home/jemin/Projects/Med_Matrix/ai-service/test_pipeline.py) — Python integration test verifying full reconstruction & inference flow.
*   [ai-service/test_reconstruction.py](file:///home/jemin/Projects/Med_Matrix/ai-service/test_reconstruction.py) — Unit tests for the 2D IFFT and RSS coil combination.
*   [ai-service/test_artifact_detector.py](file:///home/jemin/Projects/Med_Matrix/ai-service/test_artifact_detector.py) — Unit tests for the artifact classifier.

#### Training Visualizations
*   [ai-service/training_results.png](file:///home/jemin/Projects/Med_Matrix/ai-service/training_results.png) — Training loss/accuracy plot (initial runs).
*   [ai-service/training_results_128.png](file:///home/jemin/Projects/Med_Matrix/ai-service/training_results_128.png) — Training loss/accuracy plot (production 128-res run).

### ⚙️ C++ Inference Engine (`ai-service/inference/`)
*   [ai-service/inference/CMakeLists.txt](file:///home/jemin/Projects/Med_Matrix/ai-service/inference/CMakeLists.txt) — CMake configuration (ONNX Runtime + protobuf/abseil transitive deps).
*   [ai-service/inference/kvision_inference.h](file:///home/jemin/Projects/Med_Matrix/ai-service/inference/kvision_inference.h) — C++ header: pathology enums, `KSpaceDims` (8×16×128×128×2), `InferenceEngine` class API.
*   [ai-service/inference/kvision_inference.cpp](file:///home/jemin/Projects/Med_Matrix/ai-service/inference/kvision_inference.cpp) — Implementation: ONNX Runtime C++ API, CUDA/CPU backends, softmax, argmax.
*   [ai-service/inference/test_inference.cpp](file:///home/jemin/Projects/Med_Matrix/ai-service/inference/test_inference.cpp) — 10-test CLI runner: loading, accuracy, determinism, sensitivity, error handling, throughput.
*   [ai-service/inference/anomaly_detector_inference.h](file:///home/jemin/Projects/Med_Matrix/ai-service/inference/anomaly_detector_inference.h) — C++ header: `AnomalyKSpaceDims` (32×256×256), `AnomalyDetectorEngine` class API.
*   [ai-service/inference/anomaly_detector_inference.cpp](file:///home/jemin/Projects/Med_Matrix/ai-service/inference/anomaly_detector_inference.cpp) — Implementation: ONNX Runtime C++ API, CUDA/CPU backends, dual inputs, severity predictions.
*   [ai-service/inference/test_anomaly_detector_inference.cpp](file:///home/jemin/Projects/Med_Matrix/ai-service/inference/test_anomaly_detector_inference.cpp) — 9-test CLI runner validating bounds, accuracy, determinism, error handling, throughput.

### 🦀 Rust MRI Prototype (`rust-mri/`)
*   [rust-mri/Cargo.toml](file:///home/jemin/Projects/Med_Matrix/rust-mri/Cargo.toml) — Rust project configuration.
*   [rust-mri/src/](file:///home/jemin/Projects/Med_Matrix/rust-mri/src/) — Rust MRI processing source code.

### 🌐 Express Backend (`apps/backend/`)
*   [apps/backend/prisma/schema.prisma](file:///home/jemin/Projects/Med_Matrix/apps/backend/prisma/schema.prisma) — Prisma schema: Patient, Study, Report, ModelResult, AnomalyDetection, and GatingDecision.
*   [apps/backend/src/index.ts](file:///home/jemin/Projects/Med_Matrix/apps/backend/src/index.ts) — Server listener starting Express and BullMQ worker.
*   [apps/backend/src/ai-client.ts](file:///home/jemin/Projects/Med_Matrix/apps/backend/src/ai-client.ts) — Axios client bridge calling FastAPI `/predict` and persisting inference records.
*   [apps/backend/src/worker.ts](file:///home/jemin/Projects/Med_Matrix/apps/backend/src/worker.ts) — BullMQ job processor managing study lifecycle states.
*   [apps/backend/src/queue.ts](file:///home/jemin/Projects/Med_Matrix/apps/backend/src/queue.ts) — Queue manager definition for the `study-processing` BullMQ queue.
*   [apps/backend/src/routes/studies.ts](file:///home/jemin/Projects/Med_Matrix/apps/backend/src/routes/studies.ts) — Express routes for DICOM uploads and study queries.
*   [apps/backend/src/dicom.ts](file:///home/jemin/Projects/Med_Matrix/apps/backend/src/dicom.ts) — DICOM header parsing using `dcmjs`.
*   [apps/backend/src/storage.ts](file:///home/jemin/Projects/Med_Matrix/apps/backend/src/storage.ts) — S3 helper wrapper managing S3 buckets in MinIO.

### 🖥️ Electron Frontend App (`apps/electron/`)
*   [apps/electron/src/main/index.ts](file:///home/jemin/Projects/Med_Matrix/apps/electron/src/main/index.ts) — Electron main process setup (window creation, IPC handlers, app lifecycle).
*   [apps/electron/src/preload/index.ts](file:///home/jemin/Projects/Med_Matrix/apps/electron/src/preload/index.ts) — Context bridge mapping IPC interfaces.
*   [apps/electron/src/renderer/App.tsx](file:///home/jemin/Projects/Med_Matrix/apps/electron/src/renderer/App.tsx) — Main desktop application GUI in React.
*   [apps/electron/src/renderer/index.css](file:///home/jemin/Projects/Med_Matrix/apps/electron/src/renderer/index.css) — Global application styles implementing the Syngo clinical theme layout.
