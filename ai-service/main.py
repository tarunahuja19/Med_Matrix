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