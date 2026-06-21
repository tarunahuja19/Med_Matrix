# 🤖 Radiology RAG Agent Overview

This document summarizes our understanding of the **Radiology RAG Agent** architecture, stack, pipelines, and integrations within the MedMatrix workspace.

---

## 🏛️ System Architecture Spec

The Radiology RAG (Retrieval-Augmented Generation) Agent acts as an intelligent assistant for generating standardized radiology reports from clinical indication, patient metadata, and scientific context ingested from research papers.

```mermaid
graph TD
    subgraph Ingest Pipeline (Run Once Per Paper)
        A[PDF Research Paper] --> B[Parse & Split by Section]
        B --> C[Embed Chunks using text-embedding-3-small]
        C --> D[(PostgreSQL + pgvector)]
    end

    subgraph Query Pipeline (Per Request)
        E[Query: disease_name + patient_metadata] --> F[Embed disease_name]
        F --> G[pgvector Similarity Search with Disease Filter]
        G --> H[(PostgreSQL + pgvector)]
        H --> I[Retrieve Top 5 Chunks]
        I --> J[Construct LLM User Message]
        K[Report Template System Prompt] --> L[LLM Call]
        J --> L
        M[Redis Cache Check] -->|Cache Hit| N[Return Cached Report]
        M -->|Cache Miss| L
        L --> O[Format & Return Radiology Report]
        L --> P[Store in Redis Cache]
    end
```

---

## 🛠️ Stack & Infrastructure

1. **RAG Database & Cache**: Upstash Redis (Serverless).
2. **Embeddings**: `text-embedding-3-small` (1536-dimensional vector space) from OpenAI.
3. **LLM**: `gpt-4o-mini` (or GPT-4o) configured with strict system prompt constraints to prevent hallucinations.

---

## 💾 Redis Key-Value Schema

Instead of a relational SQL database, all reference sections and generated reports are stored as Redis keys:

### 1. RAG Document Store
* **Key**: `med_docs:<disease_name>` (e.g. `med_docs:avm`)
* **Value**: A serialized JSON list of all section chunks for that disease:
  ```json
  [
    {
      "section": "Clinical Presentation & Symptoms",
      "content": "...",
      "source": "MedMatrix Disease Reference",
      "embedding": [0.012, -0.045, ..., 0.089]
    },
    ...
  ]
  ```

### 2. LLM Report Cache
* **Key**: `rad_report:<disease_name>:<age_bucket>:<sex>` (e.g. `rad_report:avm:young_adult:male`)
* **Value**: Plain-text markdown report containing placeholders `{name}` and `{date}` to enable reuse within the same age/sex bucket.

---

## 🔄 Core Pipelines

### 1. Ingest Pipeline
*   **Input**: Markdown Reference Files.
*   **Parsing**: Extracts sections by H2 headings (`## 1. Section Title`).
*   **Embedding**: Generates a 1536-dim vector for each section chunk using `text-embedding-3-small` in a single batch request.
*   **Storage**: Serializes and stores the section list in Redis under the key `med_docs:<disease_name>`.

### 2. Query Pipeline
*   **Inputs**: `disease_name` (string), `patient_metadata` (age, sex, symptoms, name, date).
*   **Cache Check**: Looks up `rad_report:<disease_name>:<age_bucket>:<sex>`.
    * If cached -> dynamic replacement of `{name}` and `{date}` placeholder values and returns instantly.
    * If miss -> proceeds to RAG search.
*   **RAG Query**:
    1. Fetches the document chunk list from key `med_docs:<disease_name>`.
    2. Embeds the `disease_name` using `text-embedding-3-small`.
    3. Computes the cosine similarity between the query embedding and each chunk's embedding directly in Python using NumPy.
    4. Sorts chunks by similarity score descending, and takes the top-5 as context.
*   **LLM Invocation**: Calls the LLM with the report template system prompt and user context, caches the formatted template back in Redis, and returns the generated report.

---


## 📝 Report Template

All generated reports must strictly adhere to the following output format:

```text
---
RADIOLOGY REPORT

Patient: {name} | Age: {age} | Sex: {sex}
Date: {date}

CLINICAL INDICATION
[Why the scan was ordered based on symptoms and disease]

TECHNIQUE
[Typical imaging modality and sequences used for this disease]

FINDINGS
[Organ-by-organ observations relevant to this disease]

IMPRESSION
1. [Most likely diagnosis]
2. [Second differential]
3. [Third differential if applicable]

RECOMMENDATION
[Follow-up scan / biopsy / urgent referral as appropriate]
---
```

---

## 🔄 Integration Strategy within MedMatrix

*   **Database Integration**: Since Express backend uses Prisma, we will need to ensure Prisma can execute pgvector raw queries or map the vector data type correctly. Alternatively, raw SQL via Prisma client can run `pgvector` queries.
*   **AI Service (FastAPI)**: The ingest and query pipelines can be exposed as endpoints in the Python `ai-service` (leveraging PyPDF/PyMuPDF for PDF splitting, and standard OpenAI/Gemini SDKs for embeddings/LLM).
*   **UI Integration**: Add a dedicated UI tab or panel in the Electron frontend for uploading research papers (Ingest) and querying/previewing the generated report.
