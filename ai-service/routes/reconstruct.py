"""
KVision 4.0 — K-Space Reconstruction Route
Handles /reconstruct endpoint: IFFT + phase correction + motion correction + denoising.
"""

import os
import tempfile
import logging
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from kspace_reader import load_kspace
from reconstruction import reconstruct_kspace
from motion_correction import correct_motion
from denoiser import denoise_image
from artifact_detector import detect_artifacts
from config import BUCKET_KSPACE_RAW, BUCKET_RECONSTRUCTED
from services.minio_service import download_object, upload_object
from services.preprocessing import resize_volume_to_target

logger = logging.getLogger("ai-service")

router = APIRouter(tags=["Reconstruction"])


class ReconstructRequest(BaseModel):
    study_id: str
    kspace_key: str
    reconstructed_key: str
    phase_correction: bool = True
    denoise_method: str = "dncnn"
    physics_constrained: bool = False



class ReconstructResponse(BaseModel):
    status: str
    study_id: str
    reconstructed_key: str
    artifact_report: dict


@router.post("/reconstruct", response_model=ReconstructResponse)
def reconstruct(request: ReconstructRequest):
    """
    Full k-space → image reconstruction pipeline:
    1. Download raw k-space from MinIO
    2. IFFT + RSS coil combination + phase correction
    3. Rigid motion correction (SimpleITK)
    4. Denoising (NLM or DnCNN)
    5. Artifact detection
    6. Upload reconstructed image to MinIO
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        _, ext = os.path.splitext(request.kspace_key)
        local_kspace_path = os.path.join(tmpdir, f"kspace_input{ext}")
        local_reconstructed_path = os.path.join(tmpdir, "reconstructed.npy")

        # 1. Download raw K-space
        try:
            logger.info(f"Downloading {request.kspace_key} from '{BUCKET_KSPACE_RAW}'...")
            download_object(BUCKET_KSPACE_RAW, request.kspace_key, local_kspace_path)
        except Exception as e:
            logger.error(f"Failed to download kspace: {e}")
            raise HTTPException(status_code=404, detail=f"Failed to retrieve K-space file: {e}")

        # 2. Load K-space
        try:
            kspace = load_kspace(local_kspace_path)
            logger.info(f"K-space loaded: shape={kspace.shape}")
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Failed to parse K-space: {e}")

        # 3. Reconstruct
        try:
            reconstructed = reconstruct_kspace(
                kspace,
                phase_correction=request.phase_correction,
                physics_constrained=request.physics_constrained
            )
            logger.info(f"Reconstruction complete: shape={reconstructed.shape}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Reconstruction failed: {e}")

        # 4. Motion correction
        try:
            corrected = correct_motion(reconstructed)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Motion correction failed: {e}")

        # 5. Denoising
        try:
            denoised = denoise_image(corrected, method=request.denoise_method)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Denoising failed: {e}")

        # 6. Artifact detection
        try:
            if denoised.ndim == 3:
                artifact_img = np.mean(denoised, axis=0)
            elif denoised.ndim == 4:
                artifact_img = np.mean(denoised, axis=(0, 1))
            else:
                artifact_img = denoised
            artifact_report = detect_artifacts(artifact_img)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Artifact detection failed: {e}")

        # 7. Upload
        try:
            denoised_resized = resize_volume_to_target(denoised)
            np.save(local_reconstructed_path, denoised_resized)
            upload_object(BUCKET_RECONSTRUCTED, request.reconstructed_key, local_reconstructed_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to upload reconstructed image: {e}")

        return ReconstructResponse(
            status="success",
            study_id=request.study_id,
            reconstructed_key=request.reconstructed_key,
            artifact_report=artifact_report,
        )
