"""
KVision 4.0 — AI Service Application Entry Point

This module is a thin orchestrator. All business logic lives in:
  - routes/        → API endpoint handlers
  - services/      → Shared services (MinIO, model loading, Grad-CAM, preprocessing)
  - config.py      → Centralized configuration
  - models.py      → Pydantic request/response schemas

Architecture:
  ┌─────────────────────────────────────────────────┐
  │ FastAPI Application                              │
  │   ├── /health            (routes/health.py)      │
  │   ├── /reconstruct       (routes/reconstruct.py) │
  │   ├── /predict           (routes/predict.py)     │
  │   ├── /predict/progression (routes/progression.py)│
  │   ├── /rag/*             (routes/rag.py)         │
  │   └── /system/models     (routes/health.py)      │
  └─────────────────────────────────────────────────┘
"""

from fastapi import FastAPI
from config import logger

# ── Application ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="KVision 4.0 AI Service",
    description="Digital Brain Twin Platform — Physics-Constrained K-Space Intelligence",
    version="4.0.0",
)

# ── Register Route Modules ───────────────────────────────────────────────────
from routes.health import router as health_router
from routes.reconstruct import router as reconstruct_router
from routes.predict import router as predict_router
from routes.rag import router as rag_router
from routes.progression import router as progression_router

app.include_router(health_router)
app.include_router(reconstruct_router)
app.include_router(predict_router)
app.include_router(rag_router)
app.include_router(progression_router)

logger.info("KVision 4.0 AI Service initialized — all route modules registered.")
