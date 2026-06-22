# 🧩 Shared Types (`packages/shared-types`)

This package houses the shared TypeScript interface definitions and data schemas that establish the communication boundary between the **Electron Desktop GUI** and the **Express Backend Service** in the MedMatrix monorepo.

---

## 🗺️ File Map & Details

### 1. [`src/index.ts`](file:///home/jemin/Projects/Med_Matrix/packages/shared-types/src/index.ts)
* **Purpose:** Entrypoint exporting the API and IPC structures.
* **Exports:** 
  * Re-exports everything from `./ipc`
  * Re-exports everything from `./api`

### 2. [`src/ipc.ts`](file:///home/jemin/Projects/Med_Matrix/packages/shared-types/src/ipc.ts)
* **Purpose:** Defines the Inter-Process Communication (IPC) boundary for Electron's main-to-renderer context bridge.
* **Key Types:**
  * `IPCChannel`: Enum-like type representing supported channels: `'ping'`, `'study:upload'`, `'study:status'`, and `'study:report'`.
  * `StudyUploadPayload`: Typed payload structure containing `filePath`, `patientId`, and `modality`.
  * `StudyStatusPayload`: Contains `studyId`.
  * `StudyReportPayload`: Contains `studyId`.
  * `IPCResponse<T>`: A generic wrapper for IPC replies enforcing `success`, optional `data`, and optional `error` strings.

### 3. [`src/api.ts`](file:///home/jemin/Projects/Med_Matrix/packages/shared-types/src/api.ts)
* **Purpose:** Formulates network request and response interfaces for communicating with the AI microservices and REST endpoints.
* **Key Types:**
  * `KSpaceInferenceRequest` & `KSpaceInferenceResponse`: Structures for raw K-space anomaly detection gating.
  * `ImageInferenceRequest` & `ImageInferenceResponse`: Structures for secondary image-level processing.
  * `Finding`: Represents clinical diagnostic outcomes like `tumor`, `microbleed`, `stroke`, `hemorrhage`, or `ms_lesion` with accompanying confidence, severity, and `BoundingBox` fields.
  * `BoundingBox`: Spatial dimensions (`x`, `y`, `z`, `width`, `height`, `depth`) for overlaying masks on Cornerstone3D or Three.js visualizers.

---

## ⚙️ How it is Used in the Monorepo

Both **Electron** and **Express Backend** consume these types as a workspace dependency:

```json
"dependencies": {
  "@kvision/shared-types": "workspace:*"
}
```

This configuration prevents type mismatch compilation errors and ensures that modifications to API contracts or IPC structures break the build at compile-time rather than causing runtime failures.
