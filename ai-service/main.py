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


# Pathology Classes (11 Types matching the generator and model training labels)
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
    "Cerebral_Microbleeds"
]

_PATHOLOGY_MODEL = None

def compute_kspace_gradcam(model, x_final, target_class_idx):
    """
    Computes a Grad-CAM heatmap on the K-space encoder branch (relu4 of KSpaceS4Encoder).
    """
    model.eval()
    
    # Store activations and gradients
    activations = []
    gradients = []
    
    def forward_hook(module, input, output):
        activations.append(output)
        
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])
        
    # Conv4 is the final conv layer in KSpaceS4Encoder
    target_layer = model.s4_branch.encoder.conv4
    h_forward = target_layer.register_forward_hook(forward_hook)
    h_backward = target_layer.register_full_backward_hook(backward_hook)
    
    try:
        # Enable grads for backward pass
        with torch.enable_grad():
            # detach and clone input to avoid side effects
            x_input = x_final.detach().clone()
            
            # Forward pass
            logits = model(x_input)
            
            # Backward pass for the target class
            score = logits[0, target_class_idx]
            model.zero_grad()
            score.backward()
    except Exception as e:
        logger.error(f"Error inside compute_kspace_gradcam: {e}")
        return None
    finally:
        # Always remove hooks to prevent memory leaks
        h_forward.remove()
        h_backward.remove()
        
    if not activations or not gradients:
        logger.warning("Grad-CAM hooks failed to capture activations/gradients")
        return None
        
    # act shape: [B * S, 128, H_feat, W_feat] = [8, 128, 8, 8]
    act = activations[0].detach()
    # grad shape: [8, 128, 8, 8]
    grad = gradients[0].detach()
    
    # Global average pooling of gradients along spatial dimensions of features
    # weights shape: [8, 128, 1, 1]
    weights = torch.mean(grad, dim=(2, 3), keepdim=True)
    
    # Weighted combination of feature maps
    # cam shape: [8, 8, 8]
    cam = torch.sum(weights * act, dim=1)
    
    # Apply ReLU
    cam = torch.clamp(cam, min=0)
    
    # Upsample to target K-space resolution (128x128)
    # Target shape: [1, 1, 8, 128, 128]
    cam = cam.unsqueeze(0).unsqueeze(1) # [1, 1, 8, 8, 8]
    cam_upsampled = torch.nn.functional.interpolate(
        cam, size=(8, 128, 128), mode='trilinear', align_corners=False
    ).squeeze(0).squeeze(0) # [8, 128, 128]
    
    # Normalize per-slice to [0, 1]
    cam_np = cam_upsampled.cpu().numpy()
    for s in range(cam_np.shape[0]):
        slice_min = cam_np[s].min()
        slice_max = cam_np[s].max()
        denom = slice_max - slice_min
        if denom > 1e-8:
            cam_np[s] = (cam_np[s] - slice_min) / denom
        else:
            cam_np[s] = np.zeros_like(cam_np[s])
            
    return cam_np

def compute_reconstructed_gradcam(model, x_final, target_class_idx):
    """
    Computes a Grad-CAM heatmap on the reconstructed spatial branch (relu4 of VolumeCNNClassifier).
    """
    model.eval()
    
    # Store activations and gradients
    activations = []
    gradients = []
    
    def forward_hook(module, input, output):
        activations.append(output)
        
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])
        
    # conv4 is the final conv layer in VolumeCNNClassifier
    target_layer = model.cnn_branch.conv4
    h_forward = target_layer.register_forward_hook(forward_hook)
    h_backward = target_layer.register_full_backward_hook(backward_hook)
    
    try:
        # Enable grads for backward pass
        with torch.enable_grad():
            x_input = x_final.detach().clone()
            
            # Forward pass
            logits = model(x_input)
            
            # Backward pass for the target class
            score = logits[0, target_class_idx]
            model.zero_grad()
            score.backward()
    except Exception as e:
        logger.error(f"Error inside compute_reconstructed_gradcam: {e}")
        return None
    finally:
        # Always remove hooks to prevent memory leaks
        h_forward.remove()
        h_backward.remove()
        
    if not activations or not gradients:
        logger.warning("Reconstructed Grad-CAM hooks failed to capture activations/gradients")
        return None
        
    # act shape: [B, 128, S_down, H_down, W_down] = [1, 128, 1, 8, 8]
    act = activations[0].detach()
    # grad shape: [1, 128, 1, 8, 8]
    grad = gradients[0].detach()
    
    # Global average pooling of gradients along spatial dimensions (dim 2, 3, 4)
    # weights shape: [1, 128, 1, 1, 1]
    weights = torch.mean(grad, dim=(2, 3, 4), keepdim=True)
    
    # Weighted combination of feature maps
    # cam shape: [1, S_down, H_down, W_down] = [1, 1, 8, 8]
    cam = torch.sum(weights * act, dim=1)
    
    # Apply ReLU
    cam = torch.clamp(cam, min=0)
    
    # Upsample to target reconstructed image resolution
    # Target shape: [1, 1, slices, height, width]
    cam = cam.unsqueeze(1) # [1, 1, S_down, H_down, W_down]
    
    _, slices, _, height, width = x_final.shape
    
    cam_upsampled = torch.nn.functional.interpolate(
        cam, size=(slices, height, width), mode='trilinear', align_corners=False
    ).squeeze(0).squeeze(0) # [slices, height, width]
    
    # Normalize per-slice to [0, 1]
    cam_np = cam_upsampled.cpu().numpy()
    for s in range(cam_np.shape[0]):
        slice_min = cam_np[s].min()
        slice_max = cam_np[s].max()
        denom = slice_max - slice_min
        if denom > 1e-8:
            cam_np[s] = (cam_np[s] - slice_min) / denom
        else:
            cam_np[s] = np.zeros_like(cam_np[s])
            
    return cam_np


def get_pathology_model():
    global _PATHOLOGY_MODEL
    if _PATHOLOGY_MODEL is not None:
        return _PATHOLOGY_MODEL
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Look for fused_model_128.pt in the same folder
    weights_path = os.path.join(os.path.dirname(__file__), 'fused_model_128.pt')
    
    model = FusedS4CNNClassifier(
        d_model_s4=128,
        d_state_s4=16,
        n_layers_s4=2,
        d_model_cnn=128,
        num_classes=11,
        input_dim_s4=16 * 128 * 128,  # 16 coils * 128 * 128 resolution
        d_attn=128,
        coils=16
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
        # ── Step 6: Save and upload reconstructed volume (always) ────────
        reconstructed_key = f"{request.study_id}/reconstructed.npy"
        local_reconstructed_path = os.path.join(tmpdir, "reconstructed.npy")
        artifact_report: dict | None = None

        if anomaly_detected:
            logger.info("[predict] Anomaly detected — running full image pipeline (motion correction + denoise)...")
            try:
                corrected = correct_motion(reconstructed)
                denoised = denoise_image(corrected, method=request.denoise_method)

                if denoised.ndim == 3:
                    artifact_img = np.mean(denoised, axis=0)
                elif denoised.ndim == 4:
                    artifact_img = np.mean(denoised, axis=(0, 1))
                else:
                    artifact_img = denoised

                artifact_report = detect_artifacts(artifact_img)

                # Save denoised/motion corrected version
                np.save(local_reconstructed_path, denoised)
                minio_client.fput_object(
                    bucket_name="reconstructed",
                    object_name=reconstructed_key,
                    file_path=local_reconstructed_path,
                )
                logger.info(f"[predict] Anomaly-corrected image uploaded: {reconstructed_key}")
            except Exception as e:
                logger.error(f"[predict] Image pipeline failed: {e}")
                artifact_report = {"error": str(e)}
                # Fallback to saving base reconstruction
                try:
                    np.save(local_reconstructed_path, reconstructed)
                    minio_client.fput_object(
                        bucket_name="reconstructed",
                        object_name=reconstructed_key,
                        file_path=local_reconstructed_path,
                    )
                except Exception as ex:
                    logger.error(f"[predict] Fallback base image upload failed: {ex}")
                    reconstructed_key = None
        else:
            logger.info("[predict] Clean scan — uploading base reconstruction directly...")
            try:
                # Save clean base reconstruction directly (skip motion correction and denoise to save compute)
                np.save(local_reconstructed_path, reconstructed)
                minio_client.fput_object(
                    bucket_name="reconstructed",
                    object_name=reconstructed_key,
                    file_path=local_reconstructed_path,
                )
                logger.info(f"[predict] Base reconstructed image uploaded: {reconstructed_key}")
            except Exception as e:
                logger.error(f"[predict] Base image upload failed: {e}")
                reconstructed_key = None

        # ── Step 6b: Pathology Prediction (Fused S4-CNN) ───────────────────
        predicted_pathology = None
        pathology_confidence = None
        pathology_probs = None
        kspace_gradcam_key = None
        kspace_log_mag_key = None
        reconstructed_gradcam_key = None

        try:
            logger.info("[predict] Running Fused S4-CNN pathology classifier...")
            pathology_model = get_pathology_model()

            # Preprocess K-space: raw shape is [slices, coils, height, width]
            slices_in, coils_in, h_in, w_in = kspace.shape

            x_complex = kspace.astype(np.complex64)
            x_tensor = torch.from_numpy(x_complex) # [slices, coils, H, W]

            # Resample spatial dimensions: Crop or Pad K-Space to 128x128
            target_res = 128
            if h_in > target_res:
                sh = (h_in - target_res) // 2
                x_tensor = x_tensor[:, :, sh:sh+target_res, :]
            elif h_in < target_res:
                pad_h = (target_res - h_in) // 2
                pad_tensor = torch.zeros((slices_in, coils_in, target_res, w_in), dtype=x_tensor.dtype, device=x_tensor.device)
                pad_tensor[:, :, pad_h:pad_h+h_in, :] = x_tensor
                x_tensor = pad_tensor

            slices_in, coils_in, h_in, w_in = x_tensor.shape
            if w_in > target_res:
                sw = (w_in - target_res) // 2
                x_tensor = x_tensor[:, :, :, sw:sw+target_res]
            elif w_in < target_res:
                pad_w = (target_res - w_in) // 2
                pad_tensor = torch.zeros((slices_in, coils_in, target_res, target_res), dtype=x_tensor.dtype, device=x_tensor.device)
                pad_tensor[:, :, :, pad_w:pad_w+w_in] = x_tensor
                x_tensor = pad_tensor

            # Separate real and imaginary parts
            x_real = torch.real(x_tensor)
            x_imag = torch.imag(x_tensor)

            # Permute to [coils, 1, slices, 128, 128] to interpolate slices to 8 along Z axis
            x_real_5d = x_real.permute(1, 0, 2, 3).unsqueeze(1) # [coils, 1, slices, 128, 128]
            x_imag_5d = x_imag.permute(1, 0, 2, 3).unsqueeze(1) # [coils, 1, slices, 128, 128]

            real_interp = torch.nn.functional.interpolate(
                x_real_5d, size=(8, 128, 128), mode='trilinear', align_corners=False
            ).squeeze(1) # [coils, 8, 128, 128]

            imag_interp = torch.nn.functional.interpolate(
                x_imag_5d, size=(8, 128, 128), mode='trilinear', align_corners=False
            ).squeeze(1) # [coils, 8, 128, 128]

            # Now adjust coils to exactly 16
            final_real = torch.zeros(16, 8, 128, 128, dtype=torch.float32)
            final_imag = torch.zeros(16, 8, 128, 128, dtype=torch.float32)

            if coils_in <= 16:
                final_real[:coils_in] = real_interp
                final_imag[:coils_in] = imag_interp
            else:
                final_real = real_interp[:16]
                final_imag = imag_interp[:16]

            # Permute back to [8, 16, 128, 128] (slices, coils, H, W)
            final_real = final_real.permute(1, 0, 2, 3)
            final_imag = final_imag.permute(1, 0, 2, 3)

            # Convert back to complex and add batch dimension
            x_final = torch.complex(final_real, final_imag).unsqueeze(0) # [1, 8, 16, 128, 128]

            # Normalize complex k-space signal to unit standard deviation as in training
            norm_factor = torch.std(torch.abs(x_final))
            if norm_factor > 0:
                x_final = x_final / norm_factor

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

            # ── Compute K-space Grad-CAM & Log-magnitude ─────────────────────
            try:
                logger.info(f"[predict] Computing K-Space Grad-CAM for class {predicted_pathology} (idx: {pred_idx})...")
                gradcam_heatmap = compute_kspace_gradcam(pathology_model, x_final, pred_idx)
                
                logger.info(f"[predict] Computing Reconstructed Image Grad-CAM for class {predicted_pathology} (idx: {pred_idx})...")
                reconstructed_gradcam_heatmap = compute_reconstructed_gradcam(pathology_model, x_final, pred_idx)
                
                if gradcam_heatmap is not None:
                    # Compute Log-Magnitude of K-space (coils combined via RSS)
                    kspace_mag = torch.sqrt(torch.sum(torch.abs(x_final)**2, dim=2)).squeeze(0) # [8, 128, 128]
                    kspace_log_mag = torch.log(1.0 + kspace_mag).cpu().numpy()
                    
                    # Normalize K-space log-magnitude to [0, 1] per slice
                    for s in range(kspace_log_mag.shape[0]):
                        slice_min = kspace_log_mag[s].min()
                        slice_max = kspace_log_mag[s].max()
                        denom = slice_max - slice_min
                        if denom > 1e-8:
                            kspace_log_mag[s] = (kspace_log_mag[s] - slice_min) / denom
                        else:
                            kspace_log_mag[s] = np.zeros_like(kspace_log_mag[s])
                            
                    kspace_gradcam_key = f"{request.study_id}/kspace_gradcam.npy"
                    kspace_log_mag_key = f"{request.study_id}/kspace_log_mag.npy"
                    
                    local_gradcam_path = os.path.join(tmpdir, "kspace_gradcam.npy")
                    local_log_mag_path = os.path.join(tmpdir, "kspace_log_mag.npy")
                    
                    np.save(local_gradcam_path, gradcam_heatmap)
                    np.save(local_log_mag_path, kspace_log_mag)
                    
                    minio_client.fput_object(
                        bucket_name="reconstructed",
                        object_name=kspace_gradcam_key,
                        file_path=local_gradcam_path
                    )
                    minio_client.fput_object(
                        bucket_name="reconstructed",
                        object_name=kspace_log_mag_key,
                        file_path=local_log_mag_path
                    )
                    logger.info("[predict] K-Space Grad-CAM and log-magnitude arrays uploaded successfully.")

                if reconstructed_gradcam_heatmap is not None:
                    reconstructed_gradcam_key = f"{request.study_id}/reconstructed_gradcam.npy"
                    local_reconstructed_gradcam_path = os.path.join(tmpdir, "reconstructed_gradcam.npy")
                    np.save(local_reconstructed_gradcam_path, reconstructed_gradcam_heatmap)
                    
                    minio_client.fput_object(
                        bucket_name="reconstructed",
                        object_name=reconstructed_gradcam_key,
                        file_path=local_reconstructed_gradcam_path
                    )
                    logger.info("[predict] Reconstructed Image Grad-CAM array uploaded successfully.")
            except Exception as cam_err:
                logger.error(f"[predict] Grad-CAM generation failed: {cam_err}")

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
            kspace_gradcam_key=kspace_gradcam_key,
            kspace_log_mag_key=kspace_log_mag_key,
            reconstructed_gradcam_key=reconstructed_gradcam_key,
            message=reason,
        )