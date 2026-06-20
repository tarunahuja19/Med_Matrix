import os
import tempfile
import logging
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from minio import Minio

from kspace_reader import load_kspace
from reconstruction import reconstruct_kspace
from motion_correction import correct_motion
from denoiser import denoise_image
from artifact_detector import detect_artifacts
import numpy as np
import torch
from fused_model import FusedS4CNNClassifier

# Phase 2 — Pydantic schemas for /predict
from models import (
    PredictRequest,
    PredictResponse,
    AnomalyScore,
    GatingDecision,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai-service")

app = FastAPI(title="KVISION AI Service", version="1.0.0")

# MinIO client configuration from environment
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin123")
MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() == "true"

try:
    minio_client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=MINIO_SECURE
    )
    logger.info(f"Connected to MinIO at {MINIO_ENDPOINT}")
except Exception as e:
    logger.error(f"Failed to connect to MinIO: {e}")
    minio_client = None


class ReconstructRequest(BaseModel):
    study_id: str
    kspace_key: str
    reconstructed_key: str
    phase_correction: bool = True
    denoise_method: str = "dncnn"  # "dncnn" or "nlm"


class ReconstructResponse(BaseModel):
    status: str
    study_id: str
    reconstructed_key: str
    artifact_report: dict


@app.get("/health")
def health():
    return {
        "status": "ok",
        "minio_connected": minio_client is not None
    }


@app.post("/reconstruct", response_model=ReconstructResponse)
def reconstruct(request: ReconstructRequest):
    if minio_client is None:
        raise HTTPException(status_code=503, detail="MinIO client not configured or connected")

    # Create temporary directory for processing
    with tempfile.TemporaryDirectory() as tmpdir:
        _, ext = os.path.splitext(request.kspace_key)
        local_kspace_path = os.path.join(tmpdir, f"kspace_input{ext}")
        local_reconstructed_path = os.path.join(tmpdir, "reconstructed.npy")

        # 1. Download raw K-space from 'kspace-raw' bucket
        try:
            logger.info(f"Downloading {request.kspace_key} from 'kspace-raw'...")
            minio_client.fget_object(
                bucket_name="kspace-raw",
                object_name=request.kspace_key,
                file_path=local_kspace_path
            )
        except Exception as e:
            logger.error(f"Failed to download kspace file from MinIO: {e}")
            raise HTTPException(
                status_code=404,
                detail=f"Failed to retrieve K-space file {request.kspace_key}: {e}"
            )

        # 2. Load K-space data
        try:
            logger.info("Loading K-space...")
            kspace = load_kspace(local_kspace_path)
            logger.info(f"K-space loaded successfully with shape: {kspace.shape}")
        except Exception as e:
            logger.error(f"Failed to parse K-space file: {e}")
            raise HTTPException(
                status_code=422,
                detail=f"Failed to parse K-space file format: {e}"
            )

        # 3. Perform FFT Reconstruction + RSS Coil Combination + Phase Correction
        try:
            logger.info("Performing IFFT reconstruction...")
            reconstructed = reconstruct_kspace(
                kspace,
                phase_correction=request.phase_correction
            )
            logger.info(f"Reconstruction complete. Shape: {reconstructed.shape}")
        except Exception as e:
            logger.error(f"Reconstruction failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Image reconstruction failed: {e}"
            )

        # 4. Perform Motion Correction
        try:
            logger.info("Applying rigid motion correction...")
            corrected = correct_motion(reconstructed)
            logger.info("Motion correction complete.")
        except Exception as e:
            logger.error(f"Motion correction failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Motion correction failed: {e}"
            )

        # 5. Apply Denoising
        try:
            logger.info(f"Denoising image using method: {request.denoise_method}...")
            denoised = denoise_image(
                corrected,
                method=request.denoise_method
            )
            logger.info("Denoising complete.")
        except Exception as e:
            logger.error(f"Denoising failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Denoising failed: {e}"
            )

        # 6. Perform Artifact Detection
        try:
            logger.info("Analyzing image for artifacts...")
            # For multi-slice/dynamic sequences, detect artifacts on the average or middle slice
            if denoised.ndim == 3:
                # Average across slices/timeframes for global artifact score
                artifact_img = np.mean(denoised, axis=0)
            elif denoised.ndim == 4:
                artifact_img = np.mean(denoised, axis=(0, 1))
            else:
                artifact_img = denoised

            artifact_report = detect_artifacts(artifact_img)
            logger.info(f"Artifact analysis complete: {artifact_report}")
        except Exception as e:
            logger.error(f"Artifact detection failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Artifact detection failed: {e}"
            )

        # 7. Upload Reconstructed image to 'reconstructed' bucket in MinIO
        try:
            logger.info(f"Saving and uploading reconstructed image to 'reconstructed' bucket with key {request.reconstructed_key}...")
            np.save(local_reconstructed_path, denoised)
            
            minio_client.fput_object(
                bucket_name="reconstructed",
                object_name=request.reconstructed_key,
                file_path=local_reconstructed_path
            )
            logger.info("Upload complete.")
        except Exception as e:
            logger.error(f"Failed to upload reconstructed image to MinIO: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to upload reconstructed image: {e}"
            )

        return ReconstructResponse(
            status="success",
            study_id=request.study_id,
            reconstructed_key=request.reconstructed_key,
            artifact_report=artifact_report
        )


# Pathology Classes (11 Types matching kvision_inference.h / index.md)
PATHOLOGY_CLASSES = [
    "Normal",
    "Edema",
    "Tumor_Glioma",
    "Tumor_Meningioma",
    "Hemorrhage",
    "Ischemia",
    "MS_Lesions",
    "Atrophy",
    "Hydrocephalus",
    "AVM",
    "Abscess"
]

_PATHOLOGY_MODEL = None

def get_pathology_model():
    global _PATHOLOGY_MODEL
    if _PATHOLOGY_MODEL is not None:
        return _PATHOLOGY_MODEL
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Look for fused_model.pt in the same folder
    weights_path = os.path.join(os.path.dirname(__file__), 'fused_model.pt')
    
    model = FusedS4CNNClassifier(
        d_model_s4=128,
        d_state_s4=16,
        n_layers_s4=2,
        d_model_cnn=128,
        num_classes=11,
        input_dim_s4=128 * 128,  # Resolution is 128
        d_attn=128
    )
    
    if os.path.exists(weights_path):
        try:
            state_dict = torch.load(weights_path, map_location=device)
            # Remove module. prefix if present (due to nn.DataParallel)
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict, strict=False)
            logger.info(f"Successfully loaded Fused S4-CNN pathology model from {weights_path}")
        except Exception as e:
            logger.error(f"Error loading pathology model weights: {e}")
    else:
        logger.warning(f"Pathology model weights not found at {weights_path}")
        
    model = model.to(device)
    model.eval()
    _PATHOLOGY_MODEL = model
    return _PATHOLOGY_MODEL


# ---------------------------------------------------------------------------
# Phase 2 — /predict
# K-Space anomaly detection + gating logic
#
# Flow:
#   1. Download raw K-space from MinIO
#   2. Load + reconstruct (IFFT + phase correction)
#   3. Compute artifact / anomaly scores via detect_artifacts()
#   4. Derive composite_score = max(ghosting, wrap_around, zipper_noise)
#   5. Apply gating: anomaly_detected = composite_score >= threshold
#   6. If anomaly_detected → run full pipeline (motion correction, denoise,
#      upload) and populate reconstructed_key + artifact_report
#   7. Return PredictResponse
# ---------------------------------------------------------------------------

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """K-Space anomaly detection with gating logic for Phase 2."""
    if minio_client is None:
        raise HTTPException(status_code=503, detail="MinIO client not configured or connected")

    with tempfile.TemporaryDirectory() as tmpdir:
        _, ext = os.path.splitext(request.kspace_key)
        local_kspace_path = os.path.join(tmpdir, f"kspace_input{ext}")

        # ── Step 1: Download raw K-space ───────────────────────────────────
        try:
            logger.info(f"[predict] Downloading {request.kspace_key} from 'kspace-raw'...")
            minio_client.fget_object(
                bucket_name="kspace-raw",
                object_name=request.kspace_key,
                file_path=local_kspace_path,
            )
        except Exception as e:
            logger.error(f"[predict] Failed to download kspace: {e}")
            raise HTTPException(
                status_code=404,
                detail=f"Failed to retrieve K-space file '{request.kspace_key}': {e}",
            )

        # ── Step 2: Load K-space ───────────────────────────────────────────
        try:
            logger.info("[predict] Loading K-space data...")
            kspace = load_kspace(local_kspace_path)
            logger.info(f"[predict] K-space shape: {kspace.shape}")
        except Exception as e:
            logger.error(f"[predict] Failed to parse K-space: {e}")
            raise HTTPException(status_code=422, detail=f"Failed to parse K-space file: {e}")

        # ── Step 3: Reconstruct (IFFT + phase correction) ─────────────────
        try:
            logger.info("[predict] Performing IFFT reconstruction for anomaly scoring...")
            reconstructed = reconstruct_kspace(kspace, phase_correction=request.phase_correction)
        except Exception as e:
            logger.error(f"[predict] Reconstruction failed: {e}")
            raise HTTPException(status_code=500, detail=f"Reconstruction failed: {e}")

        # ── Step 4: Compute anomaly / artifact scores ──────────────────────
        try:
            logger.info("[predict] Computing anomaly scores...")
            if reconstructed.ndim == 3:
                scan_slice = np.mean(reconstructed, axis=0)
            elif reconstructed.ndim == 4:
                scan_slice = np.mean(reconstructed, axis=(0, 1))
            else:
                scan_slice = reconstructed

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
            logger.info(f"[predict] Anomaly scores: {raw_scores} | composite={composite:.4f}")
        except Exception as e:
            logger.error(f"[predict] Anomaly scoring failed: {e}")
            raise HTTPException(status_code=500, detail=f"Anomaly scoring failed: {e}")

        # ── Step 5: Gating decision ────────────────────────────────────────
        anomaly_detected = composite >= request.anomaly_threshold
        image_encoder_triggered = anomaly_detected

        if anomaly_detected:
            reason = (
                f"Composite anomaly score {composite:.4f} >= threshold {request.anomaly_threshold:.4f}. "
                "Image encoder pipeline triggered."
            )
        else:
            reason = (
                f"Composite anomaly score {composite:.4f} < threshold {request.anomaly_threshold:.4f}. "
                "Image encoder skipped — K-Space clean."
            )

        logger.info(f"[predict] Gating: anomaly_detected={anomaly_detected} | reason: {reason}")

        gating = GatingDecision(
            anomaly_detected=anomaly_detected,
            confidence=composite,
            threshold_used=request.anomaly_threshold,
            image_encoder_triggered=image_encoder_triggered,
            reason=reason,
        )

        # ── Step 6 (conditional): Full pipeline if anomaly detected ────────
        reconstructed_key: str | None = None
        artifact_report: dict | None = None

        if anomaly_detected:
            logger.info("[predict] Anomaly detected — running full image pipeline...")
            local_reconstructed_path = os.path.join(tmpdir, "reconstructed.npy")
            reconstructed_key = f"{request.study_id}/reconstructed.npy"

            try:
                corrected = correct_motion(reconstructed)
                denoised = denoise_image(corrected, method=request.denoise_method)

                # Artifact report on denoised image
                if denoised.ndim == 3:
                    artifact_img = np.mean(denoised, axis=0)
                elif denoised.ndim == 4:
                    artifact_img = np.mean(denoised, axis=(0, 1))
                else:
                    artifact_img = denoised

                artifact_report = detect_artifacts(artifact_img)

                # Upload to MinIO
                np.save(local_reconstructed_path, denoised)
                minio_client.fput_object(
                    bucket_name="reconstructed",
                    object_name=reconstructed_key,
                    file_path=local_reconstructed_path,
                )
                logger.info(f"[predict] Reconstructed image uploaded: {reconstructed_key}")
            except Exception as e:
                logger.error(f"[predict] Image pipeline failed: {e}")
                # Non-fatal: return gating result even if image pipeline fails
                artifact_report = {"error": str(e)}
                reconstructed_key = None

        # ── Step 6b: Pathology Prediction (Fused S4-CNN) ───────────────────
        predicted_pathology = None
        pathology_confidence = None
        pathology_probs = None

        try:
            logger.info("[predict] Running Fused S4-CNN pathology classifier...")
            pathology_model = get_pathology_model()

            # Preprocess K-space: raw shape is [slices, coils, height, width]
            # Convert to [1, 64, 1, 128, 128] complex64 tensor

            # Reduce coils to 1 if multiple coils present
            if kspace.ndim == 4:
                kspace_reduced = np.mean(kspace, axis=1, keepdims=True)
            elif kspace.ndim == 3:
                kspace_reduced = np.expand_dims(kspace, axis=1)
            elif kspace.ndim == 2:
                kspace_reduced = np.expand_dims(np.expand_dims(kspace, axis=0), axis=0)
            else:
                kspace_reduced = kspace

            x_complex = kspace_reduced.astype(np.complex64)
            x_tensor = torch.from_numpy(x_complex).unsqueeze(0) # [1, slices, 1, H, W]

            # Resample slice dimension to 64 and spatial dimensions to 128x128
            x_real = torch.real(x_tensor)
            x_imag = torch.imag(x_tensor)

            real_interp = torch.nn.functional.interpolate(
                x_real, size=(64, 128, 128), mode='trilinear', align_corners=False
            )
            imag_interp = torch.nn.functional.interpolate(
                x_imag, size=(64, 128, 128), mode='trilinear', align_corners=False
            )

            # [1, 1, 64, 128, 128]
            x_interp = torch.complex(real_interp, imag_interp)

            # Permute to [B=1, S=64, C=1, H=128, W=128]
            x_final = x_interp.permute(0, 2, 1, 3, 4)

            device = next(pathology_model.parameters()).device
            x_final = x_final.to(device)

            with torch.no_grad():
                logits = pathology_model(x_final)
                probs = torch.softmax(logits, dim=-1).squeeze(0)
                pred_idx = int(torch.argmax(logits, dim=-1).item())

                predicted_pathology = PATHOLOGY_CLASSES[pred_idx]
                pathology_confidence = float(probs[pred_idx].item())
                pathology_probs = {
                    PATHOLOGY_CLASSES[i]: float(probs[i].item())
                    for i in range(len(PATHOLOGY_CLASSES))
                }

            logger.info(f"[predict] Pathology prediction complete: {predicted_pathology} (conf={pathology_confidence:.4f})")
        except Exception as e:
            logger.error(f"[predict] Pathology classification failed: {e}")

        # ── Step 7: Return response ────────────────────────────────────────
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
            message=reason,
        )