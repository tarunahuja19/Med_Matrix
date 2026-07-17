"""
KVision 4.0 — Health & System Info Routes
"""

from fastapi import APIRouter
from services.model_loader import get_model_info
from services.minio_service import minio_client

router = APIRouter(tags=["System"])


@router.get("/health")
def health():
    """System health check endpoint."""
    return {
        "status": "ok",
        "minio_connected": minio_client is not None,
        "version": "4.0.0",
    }


@router.get("/system/models")
def system_models():
    """Returns information about loaded ML models."""
    return {
        "status": "ok",
        "models": get_model_info(),
    }
