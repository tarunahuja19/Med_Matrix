# 🏛️ KVISION Architecture Overview

This document tracks the high-level architecture and technology stack of the **KVISION** workspace.

---

## 🏗️ Repository Layout

KVISION is a monorepo managed using `pnpm` workspaces.

```
KVISION/
├── 🐳 docker-compose.yml     <- Dev services (database, etc.)
├── 🤖 ai-service/             <- Python-based AI microservice
├── 📱 apps/
│   ├── 🖥️ electron/          <- Desktop app frontend (Electron)
│   └── 🌐 backend/           <- Node/TypeScript backend server
└── 📦 packages/
    ├── ⚙️ config/            <- Shared configuration files
    └── 🧩 shared-types/      <- Shared TS types across front & backend
```

---

## 🛠️ Technology Stack

### 1. Applications (`apps/`)
*   **Electron Desktop Application (`apps/electron`)**:
    *   Framework: Electron, TypeScript.
    *   Packaging/Installer: `electron-winstaller`.
*   **Backend Application (`apps/backend`)**:
    *   Runtime: Node.js, TypeScript.
    *   Database ORM: Prisma.

### 2. Services (`ai-service/`)
*   **Python AI Service**:
    *   Entrypoint: `main.py` (FastAPI app orchestrating reconstruction & inference).
    *   Containerization: `Dockerfile`
    *   Dependencies: `requirements.txt` (FastAPI, PyTorch, SimpleITK, h5py, scikit-image, pywavelets).
    *   Core Modules:
        *   `kspace_reader.py`: Binary parser for Siemens Twix `.dat` files, HDF5 reader for fastMRI `.h5` files, NumPy loader, and synthetic K-Space generator.
        *   `reconstruction.py`: 2D IFFT, RSS coil combination, 1D/2D phase correction.
        *   `motion_correction.py`: SimpleITK rigid registration (Euler 2D/3D transforms, Mean Squares, regular step gradient descent).
        *   `denoiser.py`: Non-Local Means (NLM) baseline and PyTorch DnCNN residual model.
        *   `artifact_detector.py`: ResNet-based CNN classifier to detect ghosting, wrap-around, and zipper noise.

### 3. Packages (`packages/`)
*   **config**: Centralized shared config (ESLint, Prettier, TypeScript).
*   **shared-types**: Common type definitions shared between apps/electron and apps/backend.

---

## ⚙️ Package Management & Tooling
*   **Package Manager**: `pnpm` (configured via `pnpm-workspace.yaml`).
*   **Bundling/Compilation**: `esbuild` for speed.
*   **Linters/Formatters**: ESLint and Prettier.
