"""
KVision 4.0 — Predict Route (Two-Tier AI Gating Cascade)
Handles /predict endpoint: K-space anomaly detection → gated pathology classification.

Pipeline:
  1. Download raw K-space from MinIO
  2. Reconstruct (IFFT + phase correction)
  3. Compute artifact/anomaly scores
  4. Gating decision (anomaly_score >= threshold?)
  5. If gated: motion correction + denoise + upload enhanced image
  6. Pathology classification via Fused S4-CNN
  7. K-Space Anomaly Estimation via SSM
  8. Grad-CAM explainability heatmaps
"""

import os
import tempfile
import logging
import numpy as np
import torch
from fastapi import APIRouter, HTTPException

from config import (
    PATHOLOGY_CLASSES,
    BUCKET_KSPACE_RAW,
    BUCKET_RECONSTRUCTED,
)
from models import (
    PredictRequest,
    PredictResponse,
    AnomalyScore,
    GatingDecision,
)
from kspace_reader import load_kspace
from reconstruction import reconstruct_kspace
from motion_correction import correct_motion
from denoiser import denoise_image
from artifact_detector import detect_artifacts
from services.minio_service import download_object, upload_object
from services.model_loader import get_pathology_model, get_anomaly_estimator_model, get_device
from services.gradcam import compute_kspace_gradcam, compute_reconstructed_gradcam
from services.preprocessing import (
    resize_volume_to_target,
    prepare_kspace_for_classifier,
    prepare_kspace_for_anomaly_estimator,
)

logger = logging.getLogger("ai-service")

router = APIRouter(tags=["Prediction"])


@router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """Two-tier AI gating cascade: K-space anomaly detection → pathology classification."""

    with tempfile.TemporaryDirectory() as tmpdir:
        _, ext = os.path.splitext(request.kspace_key)
        local_kspace_path = os.path.join(tmpdir, f"kspace_input{ext}")

        # ── Step 1: Download raw K-space ───────────────────────────────────
        try:
            logger.info(f"[predict] Downloading {request.kspace_key}...")
            download_object(BUCKET_KSPACE_RAW, request.kspace_key, local_kspace_path)
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Failed to retrieve K-space: {e}")

        # ── Step 2: Load K-space ───────────────────────────────────────────
        try:
            kspace = load_kspace(local_kspace_path)
            logger.info(f"[predict] K-space shape: {kspace.shape}")
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Failed to parse K-space: {e}")

        # ── Step 3: Reconstruct ────────────────────────────────────────────
        try:
            reconstructed = reconstruct_kspace(kspace, phase_correction=request.phase_correction)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Reconstruction failed: {e}")

        # ── Step 4: Compute anomaly scores ─────────────────────────────────
        try:
            scan_slice = _get_representative_slice(reconstructed)
            raw_scores = detect_artifacts(scan_slice)
            ghosting_score = float(raw_scores["ghosting"])
            wrap_score = float(raw_scores["wrap_around"])
            zipper_score = float(raw_scores["zipper_noise"])
            composite = max(ghosting_score, wrap_score, zipper_score)

            anomaly_scores = AnomalyScore(
                ghosting=ghosting_score,
                wrap_around=wrap_score,
                zipper_noise=zipper_score,
                composite_score=composite,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Anomaly scoring failed: {e}")

        # ── Step 5: Gating decision ────────────────────────────────────────
        anomaly_detected = composite >= request.anomaly_threshold
        reason = (
            f"Composite anomaly score {composite:.4f} >= threshold {request.anomaly_threshold:.4f}. "
            "Image encoder pipeline triggered."
        ) if anomaly_detected else (
            f"Composite anomaly score {composite:.4f} < threshold {request.anomaly_threshold:.4f}. "
            "Image encoder skipped — K-Space clean."
        )

        gating = GatingDecision(
            anomaly_detected=anomaly_detected,
            confidence=composite,
            threshold_used=request.anomaly_threshold,
            image_encoder_triggered=anomaly_detected,
            reason=reason,
        )

        # ── Step 6: Reconstruct & Upload ───────────────────────────────────
        reconstructed_key = f"{request.study_id}/reconstructed.npy"
        local_reconstructed_path = os.path.join(tmpdir, "reconstructed.npy")
        artifact_report = None

        if anomaly_detected:
            artifact_report = _run_full_image_pipeline(
                reconstructed, request, tmpdir, local_reconstructed_path, reconstructed_key
            )
        else:
            _upload_base_reconstruction(
                reconstructed, tmpdir, local_reconstructed_path, reconstructed_key
            )

        # ── Step 7: Pathology Classification ───────────────────────────────
        predicted_pathology = None
        pathology_confidence = None
        pathology_probs = None
        kspace_gradcam_key = None
        kspace_log_mag_key = None
        reconstructed_gradcam_key = None

        try:
            (
                predicted_pathology, pathology_confidence, pathology_probs,
                kspace_gradcam_key, kspace_log_mag_key, reconstructed_gradcam_key
            ) = _run_pathology_classification(kspace, request.study_id, tmpdir)
        except Exception as e:
            logger.error(f"[predict] Pathology classification failed: {e}")

        # ── Step 8: K-Space Anomaly Estimation (SSM) ──────────────────────
        noise_severity, motion_severity, phase_severity = None, None, None
        try:
            noise_severity, motion_severity, phase_severity = _run_anomaly_estimation(kspace)
        except Exception as e:
            logger.error(f"[predict] Anomaly estimation failed: {e}")

        # ── Return response ────────────────────────────────────────────────
        return PredictResponse(
            status="success",
            study_id=request.study_id,
            anomaly_scores=anomaly_scores,
            gating_decision=gating,
            reconstructed_key=reconstructed_key,
            artifact_report=artifact_report,
            predicted_pathology=predicted_pathology,
            pathology_confidence=pathology_confidence,
            pathology_probabilities=pathology_probs,
            kspace_gradcam_key=kspace_gradcam_key,
            kspace_log_mag_key=kspace_log_mag_key,
            reconstructed_gradcam_key=reconstructed_gradcam_key,
            noise_severity=noise_severity,
            motion_severity=motion_severity,
            phase_severity=phase_severity,
            message=reason,
        )


# ── Helper Functions ─────────────────────────────────────────────────────────

def _get_representative_slice(volume: np.ndarray) -> np.ndarray:
    """Gets a representative 2D slice from a multi-dimensional volume."""
    if volume.ndim == 3:
        return np.mean(volume, axis=0)
    elif volume.ndim == 4:
        return np.mean(volume, axis=(0, 1))
    return volume


def _run_full_image_pipeline(
    reconstructed, request, tmpdir, local_path, recon_key
) -> dict | None:
    """Runs motion correction + denoise + artifact detection when anomaly is detected."""
    try:
        corrected = correct_motion(reconstructed)
        denoised = denoise_image(corrected, method=request.denoise_method)
        artifact_img = _get_representative_slice(denoised)
        artifact_report = detect_artifacts(artifact_img)

        denoised_resized = resize_volume_to_target(denoised)
        np.save(local_path, denoised_resized)
        upload_object(BUCKET_RECONSTRUCTED, recon_key, local_path)
        return artifact_report
    except Exception as e:
        logger.error(f"[predict] Image pipeline failed: {e}")
        # Fallback: upload base reconstruction
        _upload_base_reconstruction(reconstructed, tmpdir, local_path, recon_key)
        return {"error": str(e)}


def _upload_base_reconstruction(reconstructed, tmpdir, local_path, recon_key):
    """Uploads the base (un-enhanced) reconstruction."""
    try:
        resized = resize_volume_to_target(reconstructed)
        np.save(local_path, resized)
        upload_object(BUCKET_RECONSTRUCTED, recon_key, local_path)
    except Exception as e:
        logger.error(f"[predict] Base upload failed: {e}")


def _run_pathology_classification(kspace, study_id, tmpdir):
    """Runs the Fused S4-CNN pathology classifier with Grad-CAM."""
    model = get_pathology_model()
    device = get_device()
    x_final = prepare_kspace_for_classifier(kspace).to(device)

    # Forward pass
    if hasattr(model, "forward_with_uncertainty"):
        mean_probs, entropy = model.forward_with_uncertainty(x_final)
        probs = mean_probs.squeeze(0)
        pred_idx = int(torch.argmax(mean_probs, dim=-1).item())
        logger.info(f"[predict] Running MC Dropout uncertainty. Mean Entropy: {entropy.mean().item():.4f}")
    else:
        with torch.no_grad():
            logits = model(x_final)
            probs = torch.softmax(logits, dim=-1).squeeze(0)
            pred_idx = int(torch.argmax(logits, dim=-1).item())

    predicted_pathology = PATHOLOGY_CLASSES[pred_idx]
    pathology_confidence = float(probs[pred_idx].item())
    pathology_probs = {PATHOLOGY_CLASSES[i]: float(probs[i].item()) for i in range(len(PATHOLOGY_CLASSES))}


    logger.info(f"[predict] Pathology: {predicted_pathology} (conf={pathology_confidence:.4f})")

    # Grad-CAM
    kspace_gradcam_key = None
    kspace_log_mag_key = None
    reconstructed_gradcam_key = None

    try:
        gradcam_heatmap = compute_kspace_gradcam(model, x_final, pred_idx)
        recon_gradcam_heatmap = compute_reconstructed_gradcam(model, x_final, pred_idx)

        if gradcam_heatmap is not None:
            # Compute K-space log-magnitude
            kspace_mag = torch.sqrt(torch.sum(torch.abs(x_final) ** 2, dim=2)).squeeze(0)
            kspace_log_mag = torch.log(1.0 + kspace_mag).cpu().numpy()
            # Normalize per slice
            for s in range(kspace_log_mag.shape[0]):
                s_min, s_max = kspace_log_mag[s].min(), kspace_log_mag[s].max()
                d = s_max - s_min
                kspace_log_mag[s] = (kspace_log_mag[s] - s_min) / d if d > 1e-8 else 0.0

            kspace_gradcam_key = f"{study_id}/kspace_gradcam.npy"
            kspace_log_mag_key = f"{study_id}/kspace_log_mag.npy"

            np.save(os.path.join(tmpdir, "kspace_gradcam.npy"), gradcam_heatmap)
            np.save(os.path.join(tmpdir, "kspace_log_mag.npy"), kspace_log_mag)
            upload_object(BUCKET_RECONSTRUCTED, kspace_gradcam_key, os.path.join(tmpdir, "kspace_gradcam.npy"))
            upload_object(BUCKET_RECONSTRUCTED, kspace_log_mag_key, os.path.join(tmpdir, "kspace_log_mag.npy"))

        if recon_gradcam_heatmap is not None:
            reconstructed_gradcam_key = f"{study_id}/reconstructed_gradcam.npy"
            resized_cam = resize_volume_to_target(recon_gradcam_heatmap)
            np.save(os.path.join(tmpdir, "reconstructed_gradcam.npy"), resized_cam)
            upload_object(BUCKET_RECONSTRUCTED, reconstructed_gradcam_key, os.path.join(tmpdir, "reconstructed_gradcam.npy"))

    except Exception as cam_err:
        logger.error(f"[predict] Grad-CAM generation failed: {cam_err}")

    return (
        predicted_pathology, pathology_confidence, pathology_probs,
        kspace_gradcam_key, kspace_log_mag_key, reconstructed_gradcam_key
    )


def _run_anomaly_estimation(kspace):
    """Runs the SSM K-Space Anomaly Estimator for noise/motion/phase severity."""
    model = get_anomaly_estimator_model()
    device = get_device()

    x_input, contrast_tensor = prepare_kspace_for_anomaly_estimator(kspace)
    x_input = x_input.to(device)
    contrast_tensor = contrast_tensor.to(device)

    with torch.no_grad():
        preds = model(x_input, contrast_tensor).cpu().numpy()

    # Aggregate across slices using max-severity
    noise = float(np.max(preds[:, 0]))
    motion = float(np.max(preds[:, 1]))
    phase = float(np.max(preds[:, 2]))

    logger.info(f"[predict] Anomaly estimation: noise={noise:.4f}, motion={motion:.4f}, phase={phase:.4f}")
    return noise, motion, phase
