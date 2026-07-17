"""
KVision 4.0 — Centralized Configuration
All environment variables, model hyperparameters, and system constants
live here. No magic numbers anywhere else in the codebase.
"""

import os
import logging

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai-service")


# ── MinIO Object Storage ─────────────────────────────────────────────────────
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin123")
MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() == "true"

# Bucket names
BUCKET_KSPACE_RAW = "kspace-raw"
BUCKET_RECONSTRUCTED = "reconstructed"
BUCKET_REPORTS = "reports"


# ── Pathology Classification ─────────────────────────────────────────────────
PATHOLOGY_CLASSES = [
    "Normal",
    "Tumor_Glioma",
    "Ischemia",
    "MS_Lesions",
    "Hydrocephalus",
    "Atrophy",
    "Hemorrhage",
    "Cerebral_Cyst",
    "Edema",
    "AVM",
    "Cerebral_Microbleeds",
]

NUM_CLASSES = len(PATHOLOGY_CLASSES)


# ── Model Hyperparameters ────────────────────────────────────────────────────
# Fused S4-CNN Classifier
FUSED_MODEL_CONFIG = {
    "d_model_s4": 128,
    "d_state_s4": 16,
    "n_layers_s4": 2,
    "d_model_cnn": 128,
    "num_classes": NUM_CLASSES,
    "input_dim_s4": 16 * 128 * 128,  # coils × height × width
    "d_attn": 128,
    "coils": 16,
}

# K-Space Anomaly Estimator (SSM)
ANOMALY_ESTIMATOR_CONFIG = {
    "coils": 16,
    "resolution": 256,
    "d_model": 64,
    "d_state": 16,
    "n_layers": 2,
    "embedding_dim": 32,
}

# Input preprocessing targets
TARGET_RESOLUTION_CLASSIFIER = 128     # Fused S4-CNN expects 128×128
TARGET_RESOLUTION_ANOMALY = 256        # Anomaly estimator expects 256×256
TARGET_SLICES = 8                      # Classifier expects 8 slices
TARGET_COILS = 16                      # All models expect 16 coils
OUTPUT_RESOLUTION = 256                # Reconstructed output resolution


# ── Model Weight Paths ───────────────────────────────────────────────────────
AI_SERVICE_DIR = os.path.dirname(os.path.abspath(__file__))

FUSED_MODEL_WEIGHTS = os.path.join(AI_SERVICE_DIR, "fused_model_128.pt")
ANOMALY_DETECTOR_WEIGHTS = os.path.join(AI_SERVICE_DIR, "anomaly_detector.pt")
ANOMALY_DETECTOR_ONNX = os.path.join(AI_SERVICE_DIR, "anomaly_detector.onnx")
FUSED_MODEL_ONNX = os.path.join(AI_SERVICE_DIR, "fused_model_128.onnx")


# ── Rust Binary Paths ────────────────────────────────────────────────────────
RUST_MRI_BINARY_SEARCH_PATHS = [
    "/usr/local/bin/rust-mri-bin",
    os.path.join(AI_SERVICE_DIR, "rust-mri-bin"),
]

RUST_PDF_BINARY_SEARCH_PATHS = [
    "/usr/local/bin/report_pdf",
    "/app/report_pdf",
    "./report_pdf",
    os.path.join(AI_SERVICE_DIR, "report_pdf"),
]


# ── RAG Configuration ────────────────────────────────────────────────────────
RAG_DATA_DIR = os.getenv("RAG_DATA_DIR", "/app/data")
RAG_DEFAULT_LLM = os.getenv("RAG_DEFAULT_LLM", "gemini-3.1-flash-lite")


# ── Digital Brain Twin (Phase 2) ─────────────────────────────────────────────
BRAIN_TWIN_STATE_DIM = 64              # Latent SDE state dimension
BRAIN_TWIN_TREATMENT_DIM = 8           # Treatment conditioning vector size
BRAIN_TWIN_COVARIATE_DIM = 16          # Patient covariate embedding size
BRAIN_TWIN_SDE_DRIFT_HIDDEN = 128      # Drift network hidden size
BRAIN_TWIN_N_MONTE_CARLO = 50          # MC samples for uncertainty


# ── Progression Model Defaults ───────────────────────────────────────────────
DEFAULT_PATHOLOGY_VOLUMES = {
    "Normal": 0.0,
    "Tumor_Glioma": 15.0,
    "Ischemia": 25.0,
    "MS_Lesions": 8.0,
    "Hydrocephalus": 60.0,
    "Atrophy": 35.0,
    "Hemorrhage": 30.0,
    "Cerebral_Cyst": 12.0,
    "Edema": 10.0,
    "AVM": 18.0,
    "Cerebral_Microbleeds": 2.0,
}

PROGRESSION_TIMELINE_MONTHS = [0, 3, 6, 12, 18, 24]
HEALTHY_BRAIN_VOLUME_CM3 = 1350.0      # Average healthy adult brain volume
