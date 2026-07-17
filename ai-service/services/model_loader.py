"""
KVision 4.0 — Model Loading Service
Centralized model registry for all PyTorch models.
Handles lazy loading, weight loading, device placement, and caching.
"""

import os
import logging
import torch
import torch.nn as nn
from config import (
    FUSED_MODEL_CONFIG, ANOMALY_ESTIMATOR_CONFIG,
    FUSED_MODEL_WEIGHTS, ANOMALY_DETECTOR_WEIGHTS,
    PATHOLOGY_CLASSES,
)

logger = logging.getLogger("ai-service")

# ── Device Selection ─────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Inference device: {DEVICE}")


# ── Singleton Model Cache ────────────────────────────────────────────────────
_model_cache: dict[str, nn.Module] = {}


def _load_state_dict_safe(model: nn.Module, weights_path: str) -> bool:
    """
    Loads a state dict, stripping 'module.' prefixes from DataParallel models.
    Returns True if weights were loaded, False if file not found.
    """
    if not os.path.exists(weights_path):
        logger.warning(f"Model weights not found at {weights_path}")
        return False

    try:
        state_dict = torch.load(weights_path, map_location=DEVICE)
        # Remove DataParallel 'module.' prefix if present
        cleaned = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith("module.") else k
            cleaned[name] = v
        model.load_state_dict(cleaned, strict=False)
        logger.info(f"Loaded model weights from {weights_path}")
        return True
    except Exception as e:
        logger.error(f"Error loading weights from {weights_path}: {e}")
        return False


def get_pathology_model():
    """Returns the Fused S4-CNN pathology classifier (lazy-loaded, cached)."""
    cache_key = "fused_s4_cnn"
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    from fused_model import FusedS4CNNClassifier

    model = FusedS4CNNClassifier(**FUSED_MODEL_CONFIG)
    _load_state_dict_safe(model, FUSED_MODEL_WEIGHTS)

    model = model.to(DEVICE)
    model.eval()
    _model_cache[cache_key] = model
    logger.info(f"Fused S4-CNN classifier ready ({sum(p.numel() for p in model.parameters()):,} params)")
    return model


def get_anomaly_estimator_model():
    """Returns the K-Space Anomaly Estimator (lazy-loaded, cached)."""
    cache_key = "anomaly_estimator"
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    from anomaly_detector_model import KSpaceAnomalyEstimator

    model = KSpaceAnomalyEstimator(**ANOMALY_ESTIMATOR_CONFIG)
    _load_state_dict_safe(model, ANOMALY_DETECTOR_WEIGHTS)

    model = model.to(DEVICE)
    model.eval()
    _model_cache[cache_key] = model
    logger.info(f"K-Space Anomaly Estimator ready ({sum(p.numel() for p in model.parameters()):,} params)")
    return model


def get_model_info() -> dict:
    """Returns information about all loaded models for health/debug endpoints."""
    info = {}
    for name, model in _model_cache.items():
        info[name] = {
            "parameters": sum(p.numel() for p in model.parameters()),
            "device": str(next(model.parameters()).device),
            "training": model.training,
        }
    return info


def get_device() -> torch.device:
    """Returns the current inference device."""
    return DEVICE
