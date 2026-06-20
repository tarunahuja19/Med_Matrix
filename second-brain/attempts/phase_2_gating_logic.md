# 🧪 Attempt Log: Phase 2 AI Gating & System Integration (Tarun & Jemin)

*   **Date**: 2026-06-18
*   **Status**: 🟢 Solved

---

## 🎯 Goal
Implement Phase 2 AI Core Cascade and Infrastructure Integration:
1. **FastAPI AI Gating**: Implement a `/predict` endpoint that processes raw K-space data, runs a ResNet classifier to estimate ghosting, wrap-around, and zipper noise scores, and skips the heavy image encoder (motion correction + denoising) if the composite score is under the threshold (default `0.5`).
2. **Prisma Database Schemas**: Add Prisma model tables for `ModelResult`, `AnomalyDetection`, and `GatingDecision` to store rich, structured metadata for every inference run.
3. **Axios AI Client**: Build a Node.js client bridge (`apps/backend/src/ai-client.ts`) that calls the Python service and writes the gating records to PostgreSQL.
4. **BullMQ Background Worker**: Hook the study-processing worker (`apps/backend/src/worker.ts`) to run this gating pipeline asynchronously.
5. **UI Panels Integration**: Connect the Electron desktop React interface to show gating metrics, S3 storage keys, confidence thresholds, and detailed AI Report findings.

---

## 🗺️ Background / Context
- Workspace: `/home/jemin/Projects/Med_Matrix`
- Frameworks: Electron (v29), Express (Node.js v20), Prisma, BullMQ, Redis, MinIO S3, FastAPI (Python v3.11), PyTorch.

---

## 🔬 Attempts Tracker

### Attempt 1: Schema Migration & Inter-Service API Integration
*   **What was tried**: 
    1. Defined request/response schemas in [ai-service/models.py](file:///home/jemin/Projects/Med_Matrix/ai-service/models.py) and linked the logic in [ai-service/main.py](file:///home/jemin/Projects/Med_Matrix/ai-service/main.py).
    2. Extended [schema.prisma](file:///home/jemin/Projects/Med_Matrix/apps/backend/prisma/schema.prisma) with the 3 new AI-related tables and ran database migrations.
    3. Coded the `AIServiceClient` class using Axios, and hooked it into the BullMQ queue processor in [apps/backend/src/worker.ts](file:///home/jemin/Projects/Med_Matrix/apps/backend/src/worker.ts).
    4. Completed React layouts in [apps/electron/src/renderer/App.tsx](file:///home/jemin/Projects/Med_Matrix/apps/electron/src/renderer/App.tsx) to render Archive study logs and report summaries.
*   **Result**: Services compiled successfully, but runtime execution encountered minor integration issues.

---

## 🛑 Blockers / Issues

### Issue 1: PostgreSQL Host Port Conflict
- **Symptom**: PostgreSQL container in `docker-compose` failed to start with a "port already in use" error.
- **Root Cause**: The local system was already running a PostgreSQL instance on port `5432`.
- **Fix**: Re-mapped PostgreSQL in [docker-compose.yml](file:///home/jemin/Projects/Med_Matrix/docker-compose.yml) to map the host port `5433` to container port `5432`.

### Issue 2: CORS Preflight Gating Failures
- **Symptom**: The Electron frontend failed to load database studies, throwing CORS policy violation errors in the console.
- **Root Cause**: The Express backend was missing appropriate header declarations to allow cross-origin resource sharing from the Electron origins.
- **Fix**: Installed and configured the `cors` package in Express backend ([apps/backend/src/index.ts](file:///home/jemin/Projects/Med_Matrix/apps/backend/src/index.ts)).

### Issue 3: Missing curl/wget in Docker Containers
- **Symptom**: Docker compose marked the backend and AI service containers as `unhealthy` even though the services were fully responsive.
- **Root Cause**: The minimal node/python alpine-like container images lacked the `curl` or `wget` utilities required by the compose healthchecks.
- **Fix**: Rewrote the healthcheck tests to use native Python `urllib.request` and Node.js `http.get` scripts in [docker-compose.yml](file:///home/jemin/Projects/Med_Matrix/docker-compose.yml).

### Issue 4: Key Form Fields Alignment
- **Symptom**: DICOM study uploads failed with a `400 Bad Request` regarding form data keys.
- **Root Cause**: The Electron uploader submitted the multipart file payload with the key name `file`, while the Express upload router route expected the key name `kspace`.
- **Fix**: Aligned the field names to match consistently.

---

## 🏁 Resolution
*   **Root Cause**: Local system port overlapping, missing cross-origin headers, lack of curl/wget utilities in base images, and mismatched request form keys.
*   **Solution**: Re-mapped external port mappings, added `cors` middleware, rewrote healthchecks to run natively, and aligned the uploader field names.
*   **Lessons Learned**:
    1. Avoid binding Docker containers directly to default system ports (like `5432` or `6379`) to ensure developer environment compatibility across different machines.
    2. Write native scripts for healthchecks instead of depending on CLI utilities like `curl` or `wget` to keep Docker image sizes small and secure.
    3. Ensure frontend API calls match backend route parameter expectations.