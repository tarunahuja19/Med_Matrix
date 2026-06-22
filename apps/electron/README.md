# 🖥️ Electron Frontend App (`apps/electron`)

The **MedMatrix Desktop Console** is an Electron and React-based desktop application designed for clinical radiologists. It implements a dark-themed, high-contrast, professional design system resembling modern clinical workstation environments (e.g., Syngo). The application handles patient indexing, raw file ingestion, 2D slice annotation, and real-time 3D brain mesh reconstruction overlays.

---

## 🏛️ Application Architecture

The project leverages Electron's secure architecture separating system execution from rendering logic:

```
            ┌───────────────────────────────────────────────┐
            │               Renderer Process                │
            │           (React 19 + TypeScript)             │
            └───────────────────────────────────────────────┘
                                    │
                         Invokes Bridge Methods
                                    ▼
            ┌───────────────────────────────────────────────┐
            │                Preload Script                 │
            │          (contextBridge Expositions)          │
            └───────────────────────────────────────────────┘
                                    │
                          Sends IPC Messages
                                    ▼
            ┌───────────────────────────────────────────────┐
            │                 Main Process                  │
            │            (Node.js OS Bindings)              │
            └───────────────────────────────────────────────┘
```

---

## 🗺️ File Map & Directory Structure

Inside the [`src/`](file:///home/jemin/Projects/Med_Matrix/apps/electron/src) directory, key modules are organized as follows:

### 1. Main Process (`src/main/`)
* **[`index.ts`](file:///home/jemin/Projects/Med_Matrix/apps/electron/src/main/index.ts):** Configures application window dimensions (`1280x800`), handles window closing, and registers OS-level IPC handlers:
  * `ping`: Direct confirmation loop.
  * `dialog:openFile`: Opens native OS file-picker with extension filters (`.npy`, `.h5`, `.dat`, `.dcm`).
  * `study:upload`: Streams raw files from host storage to the REST backend via `FormData` payloads.
  * `clipboard:copyFile`: Downloads clinical reports from URL resources and copies them to the native OS clipboard as file buffers.

### 2. Preload Script (`src/preload/`)
* **[`index.ts`](file:///home/jemin/Projects/Med_Matrix/apps/electron/src/preload/index.ts):** Binds system APIs safely in the `window.api` global space using Electron `contextBridge`, mapping:
  * `window.api.ping()`
  * `window.api.openFileDialog()`
  * `window.api.uploadStudy(filePath, metadata)`
  * `window.api.copyFileToClipboard(url, filename)`

### 3. Renderer UI App (`src/renderer/`)
* **[`main.tsx`](file:///home/jemin/Projects/Med_Matrix/apps/electron/src/renderer/main.tsx):** Launches React 19 app container.
* **[`App.tsx`](file:///home/jemin/Projects/Med_Matrix/apps/electron/src/renderer/App.tsx):** A clinical React application containing views for Patient Registry, Raw Study Ingestion, Archive Queries, and AI Diagnostic Reports.
* **[`index.css`](file:///home/jemin/Projects/Med_Matrix/apps/electron/src/renderer/index.css):** Clinical CSS stylesheet styling variables, layouts, and high-contrast tables.

---

## 📺 Clinical Visualization Panels

### 1. 2D Clinical Slice Viewer
* **Technology:** Cornerstone3D integration.
* **Functionality:** Axial, sagittal, and coronal slice scrolling with segmentation overlays representing pathology classification confidence.

### 2. 3D Volumetric Brain Visualizer
* **Technology:** WebGL mesh rendering via Three.js / VTK.js.
* **Functionality:** Real-time 3D voxel/mesh reconstruction overlaying tumor segmentation blocks and lesion volumes inside an interactive camera view.

---

## ⚙️ Build & Local Development

### 1. Install dependencies
Ensure you are using `pnpm` from the monorepo root:
```bash
pnpm install
```

### 2. Boot in development mode
Launch the Vite dev server and the Electron GUI concurrently:
```bash
pnpm --filter electron dev
```

### 3. Build & Package (Optional)
To build production installers utilizing `electron-builder`:
```bash
pnpm --filter electron build
```
The packaged outputs will compile into the local `dist/` directory.
