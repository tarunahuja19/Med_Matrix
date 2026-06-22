# 🌐 Express Backend Service (`apps/backend`)

The **MedMatrix Express Backend Service** is the central orchestrator of the platform. It manages patients and studies, processes uploaded raw DICOM and K-space files, runs an asynchronous processing pipeline via **BullMQ**, interfaces with the Python AI microservices, and formats/delivers clinical and patient-friendly reports (including SMS/WhatsApp notifications via **Twilio**).

---

## 🏛️ System Architecture & Data Flow

```
   ┌───────────────────┐       Upload DICOM / K-Space
   │  Electron Client  │ ─────────────────────────────────┐
   └───────────────────┘                                  │
             ▲                                            ▼
             │ Query API                         ┌─────────────────┐
             │                                   │ Express Backend │
             ▼                                   └─────────────────┘
    ┌─────────────────┐                             │         │
    │  PostgreSQL DB  │ ◄─── Read/Write (Prisma) ───┤         │ Dispatch Job
    └─────────────────┘                             │         ▼
                                                    │   ┌───────────┐
                                                    │   │  BullMQ   │ (Job queue backed
                                                    │   └───────────┘  by Redis)
                                                    │         ▲
                                                    │         │ Fetch & Run
                                                    ▼         │
                                              ┌─────────────────┐
                                              │ Background Job  │
                                              │     Worker      │
                                              └─────────────────┘
                                                │             │
                                    Call API    │             │ Store Magnitude/
                                 (FastAPI /predict)           │ K-Space Files
                                                ▼             ▼
                                        ┌──────────────┐   ┌─────────────┐
                                        │  AI Service  │   │  MinIO S3   │
                                        └──────────────┘   └─────────────┘
```

---

## 🛠️ Stack & Infrastructure

* **Runtime:** Node.js v20 (TypeScript)
* **Framework:** Express
* **Database ORM:** Prisma Client linked to PostgreSQL 16
* **Job Queues:** BullMQ running on Redis
* **Object Storage:** MinIO S3 client (`@aws-sdk/client-s3`)
* **Integrations:** 
  * Twilio SDK (WhatsApp PDF reporting)
  * Axios (FastAPI Python AI microservice interface)
  * `dcmjs` (DICOM header extraction)

---

## 📂 Core File Map & Directory Structure

Inside the [`src/`](file:///home/jemin/Projects/Med_Matrix/apps/backend/src) directory, key modules are organized as follows:

| File Name | Purpose |
| :--- | :--- |
| **[`index.ts`](file:///home/jemin/Projects/Med_Matrix/apps/backend/src/index.ts)** | Main entrypoint. Boots the Express REST API, establishes MinIO buckets, and fires up the background BullMQ worker. Exposes Patient, Study, Report CRUD, PDF compilers, and WhatsApp send handlers. |
| **[`worker.ts`](file:///home/jemin/Projects/Med_Matrix/apps/backend/src/worker.ts)** | Background job processor running on the `study-processing` queue. Manages the lifecycle of study analysis and report drafting. |
| **[`ai-client.ts`](file:///home/jemin/Projects/Med_Matrix/apps/backend/src/ai-client.ts)** | Axios wrapper routing predict payloads to the Python AI service. Persists the resulting `ModelResult`, `AnomalyDetection`, and `GatingDecision` tables. |
| **[`queue.ts`](file:///home/jemin/Projects/Med_Matrix/apps/backend/src/queue.ts)** | Configuration and helper definitions for BullMQ job enqueuing. |
| **[`dicom.ts`](file:///home/jemin/Projects/Med_Matrix/apps/backend/src/dicom.ts)** | DICOM metadata parser utilizing `dcmjs` to extract patient details and study dates. |
| **[`storage.ts`](file:///home/jemin/Projects/Med_Matrix/apps/backend/src/storage.ts)** | AWS S3 client wrapper configuring and initializing MinIO buckets (`kspace-raw`, `reconstructed`, `reports`). |
| **[`prisma/schema.prisma`](file:///home/jemin/Projects/Med_Matrix/apps/backend/prisma/schema.prisma)** | PostgreSQL schema mapping models: `Patient`, `Study`, `Report`, `ModelResult`, `AnomalyDetection`, and `GatingDecision`. |

---

## 🔄 Asynchronous Study Processing Worker Pipeline

The background worker runs jobs through the following progress checkpoints:

1. **`10%` Checkpoint — Initialization:**
   Marks study state as `processing` in PostgreSQL.
2. **`20% - 60%` Checkpoint — AI Cascade Gating:**
   Dispatches the raw K-space file key from MinIO to the Python FastAPI `/predict` endpoint.
   * **Gating Condition:** If `anomalyDetected` is `true` (composite score $\geq 0.5$), the second-stage image encoder runs. If `false` (clean K-space), the image encoder is bypassed to save compute overhead.
3. **`80%` Checkpoint — Draft Report & RAG Ingestion:**
   Stores `ModelResult`, `AnomalyDetection`, and `GatingDecision` states. Retrieves RAG documentation chunks for the predicted pathology from Redis (`med_docs:<disease>`) and invokes the LLM (`gpt-4o-mini`) to compile professional and patient-friendly reports.
4. **`100%` Checkpoint — Finalization:**
   Persists report drafts and flags the study status as `complete`.

---

## 📱 Twilio WhatsApp Integration

When a radiologist triggers WhatsApp report delivery (`/reports/:id/send-whatsapp`):
1. The backend requests the Python AI service to compile a patient-friendly summary layout into a PDF.
2. The resulting PDF buffer is temporarily hosted via `tmpfiles.org`.
3. The Twilio Messaging API sends a message to the patient's phone containing a direct link to the PDF:
   ```
   "Hello [Patient Name], here is your patient-friendly Brain MRI report summary: [Media URL]"
   ```

---

## ⚙️ Setup & Local Development

### 1. Start Backing Infrastructure (Docker)
Run the Postgres, Redis, and MinIO storage containers from the repository root:
```bash
docker-compose up -d
```

### 2. Configure Environment Variables
Copy and fill out the environment file in this folder:
```bash
cp .env.example .env
```
Ensure you provide `DATABASE_URL` (Postgres connection), `REDIS_HOST`, `MINIO_ENDPOINT`, and optional Twilio credentials (`TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_SENDER_NUMBER`).

### 3. Apply Schema migrations
Push the Prisma schemas to the database:
```bash
npx prisma db push
```

### 4. Boot the server
Launch the backend service in hot-reload development mode:
```bash
pnpm run dev
```
The server starts on port `3000` (configurable via `PORT` environment variable) and logs BullMQ initialization.
