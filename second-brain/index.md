# 🧠 KVISION Second Brain Vault Index

Welcome to the **KVISION Developer & AI Agent Second Brain Vault**. This index maps all documentation, memory notes, and core codebase source files for direct accessibility.

---

## 💾 Vault Documentation Index
Use these links to navigate through the persistent memories and attempt logs:

### 🏛️ Core Memories
*   [memories/architecture.md](file:///home/jemin/Projects/MRI/KVISION/second-brain/memories/architecture.md) — Main system architecture, technology stack, data flow, and database schema mappings.
*   [memories/inference.md](file:///home/jemin/Projects/MRI/KVISION/second-brain/memories/inference.md) — Fused S4-CNN model details and C++ ONNX Runtime inference engine.
*   [NeuroScan_AI_Project_Plan-1.txt](file:///home/jemin/Projects/MRI/KVISION/second-brain/NeuroScan_AI_Project_Plan-1.txt) — Comprehensive project plan outlining all 4 phases, timelines, deliverables, and checklists.
*   [NeuroScan_AI_Project_Plan-1.pdf](file:///home/jemin/Projects/MRI/KVISION/second-brain/NeuroScan_AI_Project_Plan-1.pdf) — PDF version of the confidential project plan.
*   [README.md](file:///home/jemin/Projects/MRI/KVISION/second-brain/README.md) — Obsidian quick-start guidelines and vault navigation dashboard.

### 🧪 Task & Attempt Logs
*   [attempts/template.md](file:///home/jemin/Projects/MRI/KVISION/second-brain/attempts/template.md) — Standard markdown template to use when creating new attempt logs.
*   [attempts/phase_1_jemin.md](file:///home/jemin/Projects/MRI/KVISION/second-brain/attempts/phase_1_jemin.md) — Log of Phase 1 task execution (K-Space loader, reconstructions, motion correction, denoiser, and automated testing).
*   [attempts/phase_2_gating_logic.md](file:///home/jemin/Projects/MRI/KVISION/second-brain/attempts/phase_2_gating_logic.md) — Log of Phase 2 task execution (FastAPI `/predict` endpoint, Prisma schemas, Axios Node.js AI client, and BullMQ background worker).

---

## 🛠️ Codebase File Index
This directory links to key files in the KVISION workspace, sorted by monorepo component:

### 🐳 Devops & Environment
*   [docker-compose.yml](file:///home/jemin/Projects/MRI/KVISION/docker-compose.yml) — Backing services orchestration (Postgres, Redis, MinIO, AI-Service, Backend).
*   [package.json](file:///home/jemin/Projects/MRI/KVISION/package.json) — Monorepo entrypoint configuration and execution scripts.
*   [pnpm-workspace.yaml](file:///home/jemin/Projects/MRI/KVISION/pnpm-workspace.yaml) — Defines `pnpm` workspaces for the monorepo.

### 🤖 Python AI Service (`ai-service/`)
*   [ai-service/main.py](file:///home/jemin/Projects/MRI/KVISION/ai-service/main.py) — FastAPI app hosting `/health`, `/reconstruct`, and `/predict` endpoints.
*   [ai-service/models.py](file:///home/jemin/Projects/MRI/KVISION/ai-service/models.py) — Pydantic schemas validating `/predict` request/response types.
*   [ai-service/fused_model.py](file:///home/jemin/Projects/MRI/KVISION/ai-service/fused_model.py) — Hybrid S4-CNN Fused Volumetric Classifier module (complex-valued).
*   [ai-service/fused_model_onnx.py](file:///home/jemin/Projects/MRI/KVISION/ai-service/fused_model_onnx.py) — ONNX-compatible real-valued twin model module.
*   [ai-service/cnn_model.py](file:///home/jemin/Projects/MRI/KVISION/ai-service/cnn_model.py) — 3D CNN branch module.
*   [ai-service/s4_model.py](file:///home/jemin/Projects/MRI/KVISION/ai-service/s4_model.py) — Complex-valued S4 sequence branch module.
*   [ai-service/train.py](file:///home/jemin/Projects/MRI/KVISION/ai-service/train.py) — Multi-GPU training script.
*   [ai-service/export_onnx.py](file:///home/jemin/Projects/MRI/KVISION/ai-service/export_onnx.py) — Export script mapping trained checkpoint weights to the real-valued ONNX model.
*   [ai-service/generate_synthetic_dataset.py](file:///home/jemin/Projects/MRI/KVISION/ai-service/generate_synthetic_dataset.py) — Volumetric K-space synthetic generator script.
*   [ai-service/kspace_reader.py](file:///home/jemin/Projects/MRI/KVISION/ai-service/kspace_reader.py) — Parses Siemens Twix `.dat`, fastMRI `.h5`, and NumPy `.npy` files.
*   [ai-service/reconstruction.py](file:///home/jemin/Projects/MRI/KVISION/ai-service/reconstruction.py) — 2D centered IFFT, RSS coil combination, and zero/first-order phase correction.
*   [ai-service/motion_correction.py](file:///home/jemin/Projects/MRI/KVISION/ai-service/motion_correction.py) — SimpleITK rigid registration using Mean Squares similarity.
*   [ai-service/denoiser.py](file:///home/jemin/Projects/MRI/KVISION/ai-service/denoiser.py) — BM3D baseline and PyTorch residual DnCNN denoiser.
*   [ai-service/artifact_detector.py](file:///home/jemin/Projects/MRI/KVISION/ai-service/artifact_detector.py) — ResNet classifier computing ghosting, wrap-around, and zipper noise scores.
*   [ai-service/test_pipeline.py](file:///home/jemin/Projects/MRI/KVISION/ai-service/test_pipeline.py) — Python integration test verifying full reconstruction & inference flow.
*   [ai-service/test_reconstruction.py](file:///home/jemin/Projects/MRI/KVISION/ai-service/test_reconstruction.py) — Unit tests for the 2D IFFT and RSS coil combination.
*   [ai-service/test_artifact_detector.py](file:///home/jemin/Projects/MRI/KVISION/ai-service/test_artifact_detector.py) — Unit tests for the artifact classifier.

### ⚙️ C++ Inference Engine (`ai-service/inference/`)
*   [ai-service/inference/CMakeLists.txt](file:///home/jemin/Projects/MRI/KVISION/ai-service/inference/CMakeLists.txt) — CMake configuration for compiling the ONNX Runtime-based C++ engine.
*   [ai-service/inference/kvision_inference.h](file:///home/jemin/Projects/MRI/KVISION/ai-service/inference/kvision_inference.h) — C++ header specifying pathology enums, InferenceEngine class API, and result structures.
*   [ai-service/inference/kvision_inference.cpp](file:///home/jemin/Projects/MRI/KVISION/ai-service/inference/kvision_inference.cpp) — Implementation using ONNX Runtime C++ API, enabling CUDA/CPU backends and output post-processing.
*   [ai-service/inference/test_inference.cpp](file:///home/jemin/Projects/MRI/KVISION/ai-service/inference/test_inference.cpp) — Command-line runner for validating inference accuracy and measuring latency.


### 🌐 Express Backend (`apps/backend/`)
*   [apps/backend/prisma/schema.prisma](file:///home/jemin/Projects/MRI/KVISION/apps/backend/prisma/schema.prisma) — Prisma schema definitions for Patient, Study, Report, ModelResult, AnomalyDetection, and GatingDecision.
*   [apps/backend/src/index.ts](file:///home/jemin/Projects/MRI/KVISION/apps/backend/src/index.ts) — Server listener starting Express and initiating the BullMQ worker.
*   [apps/backend/src/ai-client.ts](file:///home/jemin/Projects/MRI/KVISION/apps/backend/src/ai-client.ts) — Axios client bridge calling FastAPI `/predict` and persisting inference records.
*   [apps/backend/src/worker.ts](file:///home/jemin/Projects/MRI/KVISION/apps/backend/src/worker.ts) — BullMQ job processor managing study lifecycle states and triggering the AI client.
*   [apps/backend/src/queue.ts](file:///home/jemin/Projects/MRI/KVISION/apps/backend/src/queue.ts) — Queue manager definition for the `study-processing` BullMQ queue.
*   [apps/backend/src/routes/studies.ts](file:///home/jemin/Projects/MRI/KVISION/apps/backend/src/routes/studies.ts) — Express routes for DICOM uploads and study queries.
*   [apps/backend/src/dicom.ts](file:///home/jemin/Projects/MRI/KVISION/apps/backend/src/dicom.ts) — DICOM header parsing using `dcmjs`.
*   [apps/backend/src/storage.ts](file:///home/jemin/Projects/MRI/KVISION/apps/backend/src/storage.ts) — S3 helper wrapper managing S3 buckets (`kspace-raw` and `reconstructed`) in MinIO.

### 🖥️ Electron Frontend App (`apps/electron/`)
*   [apps/electron/src/main/index.ts](file:///home/jemin/Projects/MRI/KVISION/apps/electron/src/main/index.ts) — Electron main process setup (window creation, IPC handlers, app lifecycle).
*   [apps/electron/src/preload/index.ts](file:///home/jemin/Projects/MRI/KVISION/apps/electron/src/preload/index.ts) — Context bridge mapping IPC interfaces.
*   [apps/electron/src/renderer/App.tsx](file:///home/jemin/Projects/MRI/KVISION/apps/electron/src/renderer/App.tsx) — Main desktop application GUI in React. Handles the Ingest, Archive, Patients, and AI Reports tab screens.
*   [apps/electron/src/renderer/index.css](file:///home/jemin/Projects/MRI/KVISION/apps/electron/src/renderer/index.css) — Global application styles implementing the Syngo clinical theme layout.
