# 🧠 MedMatrix

[![C++ Inference Test Suite](https://img.shields.io/badge/C%2B%2B%20Inference-10%2F10%20Passed-success)](file:///home/jemin/Projects/Med_Matrix/ai-service/inference)
[![SSM Anomaly Test Suite](https://img.shields.io/badge/SSM%20Anomaly-9%2F9%20Passed-success)](file:///home/jemin/Projects/Med_Matrix/ai-service/inference)
[![Node.js v20](https://img.shields.io/badge/Node.js-v20-blue)](file:///home/jemin/Projects/Med_Matrix/apps/backend)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue)](file:///home/jemin/Projects/Med_Matrix/ai-service)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](file:///home/jemin/Projects/Med_Matrix/LICENSE)

**MedMatrix** (also referred to as **KVISION // NeuroScan AI**) is a premium, monorepo-based clinical MRI volumetric analysis, anomaly detection, and automated reporting platform. It integrates state-of-the-art Deep Learning (Hybrid S4 State-Space Models + Spatial Convolutions) with high-performance native engines (C++ ONNX Runtime and Rust reconstruction routines) and a clinical desktop console (Electron + React) to deliver a seamless, high-throughput pipeline for radiologists.

---

## 🏛️ System Architecture & Workflow

MedMatrix operates on an automated, two-tier AI cascade to optimize compute and preserve fine-grained structural details during MRI raw acquisition processing.

```mermaid
graph TD
    A[DICOM / Raw K-Space Upload] --> B[MinIO S3 Storage: 'kspace-raw']
    B --> C[BullMQ Enqueue Processing Job]
    C --> D[TypeScript Background Worker]
    D --> E[Axios AIServiceClient.predict]
    E --> F[FastAPI /predict Endpoint]
    F --> G[Reconstruct Magnitude Image]
    G --> H[CNN Artifact Classifier]
    H --> I{Composite score >= 0.5?}
    
    I -- Yes: Anomaly Detected --> J[Trigger Image Encoder: Motion Correction & Denoising]
    J --> K[Upload reconstructed.npy to 'reconstructed' bucket]
    K --> L[Return full classification scores + report metadata]
    
    I -- No: Clean Scan --> M[Bypass secondary image models to save compute]
    M --> N[Return gating decision: skip image encoder]
    
    L & N --> O[Persist ModelResult, AnomalyDetection & GatingDecision in PostgreSQL via Prisma]
    O --> P[Compile PDF Report draft using Rust Engine]
    P --> Q[Mark Study Status: Complete in Electronic Health Record (EHR)]
```

---

## 🚀 Key Features

* **⚡ Two-Tier AI Gating Cascade:** Automatically identifies corrupted or artifact-heavy scans (ghosting, wrap-around, zipper noise) before executing intensive processing models. Clean scans bypass heavy models to preserve throughput.
* **🧠 Hybrid S4-CNN Volumetric Classifier:** Combines a Diagonal State Space Model (S4D) sequence branch capturing frequency features in the complex domain with a spatial branch mapping reconstructed magnitude slices. Merged via Slice-Level Cross-Attention to classify 11 pathology classes.
* **📈 SSM-Based K-Space Anomaly Estimator:** A State Space Model that ingests multi-coil complex K-space data row-by-row and outputs continuous regression metrics for noise, motion, and phase corruption.
* **🚀 C++ ONNX Runtime Engine:** Native C++ implementations (`kvision::InferenceEngine` and `kvision::AnomalyDetectorEngine`) compiling with CMake, achieving up to 84.8 inferences/sec using CUDA GPU acceleration.
* **🦀 Rust Reconstruction & PDF Compiler:** A native Rust module providing phase-corrected 2D IFFT slice reconstruction and a polished clinical PDF reporting engine.
* **💻 Syngo-Themed Desktop App:** Electron desktop console styled after modern clinical Syngo design systems, hosting raw DICOM ingestion, database archiving, 2D slice viewers, and 3D volumetric visualizers.

---

## 📺 UI/UX Clinical Showcase

### 1. 3D Volumetric Brain & Lesion Visualizer
Provides a real-time, interactive 3D mesh rendering of patient brains, overlaying detected tumor models, hemorrhage volumes, and lesion nodes in three dimensions using Three.js and VTK.js.

![3D Brain Visualizer Interface](file:///home/jemin/Projects/Med_Matrix/assets/3d_visualizer_showcase.png)

### 2. 2D Clinical Slice Viewer
A clinical-grade multi-planar slice viewer powered by Cornerstone3D, allowing radiologists to scroll through reconstructed axial, sagittal, and coronal slices, overlaying segmentation masks and AI-detected pathology logits.

![2D Clinical Slice Viewer Interface](file:///home/jemin/Projects/Med_Matrix/assets/2d_slice_viewer_showcase.png)

---

## 🗺️ Monorepo Navigation Dashboard

The MedMatrix codebase is structured as a monorepo managed with `pnpm` workspaces. Click any of the links below to view the detailed folder documentation:

```
Med_Matrix/
├── 🐳 docker-compose.yml        <- Infrastructure orchestration (Postgres, Redis, MinIO)
├── 🤖 ai-service/                <- Python FastAPI + PyTorch AI microservice
│   └── ⚙️ inference/               <- C++ ONNX Runtime Inference Engine (CMake)
├── 📱 apps/
│   ├── 🖥️ electron/             <- Electron + React + TypeScript clinical app
│   └── 🌐 backend/              <- Node.js Express server + Prisma ORM + BullMQ queue
├── 🦀 rust-mri/                  <- High-speed Rust reconstruction & PDF generator
├── 📦 packages/
│   ├── ⚙️ config/               <- Shared ESLint, Prettier, and TypeScript configurations
│   └── 🧩 shared-types/         <- Shared TS interfaces across electron and backend
└── 🧠 second-brain/              <- Obsidian-compatible developer vault for memory tracking
```

### 🗂️ Documentation Quick Links:
* **[Root README](file:///home/jemin/Projects/Med_Matrix/README.md)** (this file)
* **[ai-service/README.md](file:///home/jemin/Projects/Med_Matrix/ai-service/README.md)** — Python AI microservice, PyTorch pipelines, and model details.
* **[ai-service/inference/README.md](file:///home/jemin/Projects/Med_Matrix/ai-service/inference/README.md)** — High-performance C++ ONNX Runtime engine & builds.
* **[apps/backend/README.md](file:///home/jemin/Projects/Med_Matrix/apps/backend/README.md)** — Express REST API, Prisma schema, PostgreSQL DB, and BullMQ worker.
* **[apps/electron/README.md](file:///home/jemin/Projects/Med_Matrix/apps/electron/README.md)** — Desktop UI client main/preload/renderer structure and visualizations.
* **[rust-mri/README.md](file:///home/jemin/Projects/Med_Matrix/rust-mri/README.md)** — Native Rust FFT slice reconstructions and PDF report compiler.
* **[packages/README.md](file:///home/jemin/Projects/Med_Matrix/packages/README.md)** — Shared configurations and common type interfaces index.
* **[second-brain/README.md](file:///home/jemin/Projects/Med_Matrix/second-brain/README.md)** — Obsidian vault documentation index and developer logs.

---

## 🛠️ Installation & Local Development Setup

### Prerequisites

* **Node.js**: v20+ and **pnpm** installed globally (`npm install -g pnpm`)
* **Python**: v3.10+ with `pip`
* **Docker & Compose**: For running backing database and storage services
* **C++ Compiler**: GCC 16+ or Clang, CMake 3.18+, pkg-config
* **Rust**: Cargo and rustc (edition 2021)
* **ONNX Runtime C++ Shared Libraries**: Installed via system package manager (e.g. `onnxruntime-cuda` on Arch/EndeavourOS)

### 1. Ingest Backing Services (Docker)
Start the PostgreSQL, Redis, and MinIO storage containers:
```bash
docker-compose up -d
```
Verify containers are running:
* **PostgreSQL**: `localhost:5432`
* **Redis**: `localhost:6379`
* **MinIO Console**: `localhost:9001` (S3 API at `localhost:9000`)

### 2. Configure Environment Variables
Copy the root env example:
```bash
cp .env.example .env
```
Ensure configurations match your local development environment credentials.

### 3. Initialize Monorepo Packages
Install Node dependencies and generate the Prisma database client:
```bash
pnpm install
cd apps/backend
npx prisma db push  # Applies schema to the database
cd ../..
```

### 4. Set Up Python AI Service
Initialize a Python virtual environment and install packages:
```bash
cd ai-service
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
To run the FastAPI server:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Compile C++ Inference Engine
To compile the high-performance inference engine:
```bash
cd ai-service/inference
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```
Run the validation tests:
```bash
./test_inference
./test_anomaly_detector_inference
```

### 6. Build the Rust MRI Module
To compile the Rust reconstruction binary and PDF generator:
```bash
cd rust-mri
cargo build --release
```

### 7. Run the Express Backend
From the root directory, start the backend application in development mode (which initiates the BullMQ job worker):
```bash
pnpm --filter backend dev
```

### 8. Run the Electron GUI
Launch the desktop console:
```bash
pnpm --filter electron dev
```

---

## 📄 License
This project is licensed under the Apache License 2.0. See the `LICENSE` file for details.
