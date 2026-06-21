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
    ProgressionRequest,
    ProgressionResponse,
    ProgressionPoint,
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
            denoised_resized = resize_volume_to_256(denoised)
            np.save(local_reconstructed_path, denoised_resized)
            
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

def resize_volume_to_256(volume: np.ndarray) -> np.ndarray:
    """
    Resizes a volume or image to [..., 256, 256] using bilinear interpolation in PyTorch.
    Works for 2D [H, W], 3D [slices, H, W], and 4D [time, slices, H, W] arrays.
    """
    orig_shape = volume.shape
    if len(orig_shape) < 2:
        return volume
        
    h, w = orig_shape[-2], orig_shape[-1]
    if h == 256 and w == 256:
        return volume

    # Reshape to 3D: [N, H, W] where N is the product of all outer dimensions
    flat_outer = int(np.prod(orig_shape[:-2])) if len(orig_shape) > 2 else 1
    reshaped = volume.reshape((flat_outer, h, w))
    
    # Convert numpy to torch tensor of shape [N, 1, H, W] for 2D interpolation
    tensor = torch.from_numpy(reshaped.astype(np.float32)).unsqueeze(1)
    
    # Perform bilinear interpolation
    resized_tensor = torch.nn.functional.interpolate(
        tensor,
        size=(256, 256),
        mode='bilinear',
        align_corners=False
    )
    
    # Convert back to numpy, squeeze, and reshape to the original outer dimensions
    resized_flat = resized_tensor.squeeze(1).numpy().astype(volume.dtype)
    if len(orig_shape) == 2:
        return resized_flat[0]
    else:
        return resized_flat.reshape(orig_shape[:-2] + (256, 256))


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
                denoised_resized = resize_volume_to_256(denoised)
                np.save(local_reconstructed_path, denoised_resized)
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
                    reconstructed_resized = resize_volume_to_256(reconstructed)
                    np.save(local_reconstructed_path, reconstructed_resized)
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
                reconstructed_resized = resize_volume_to_256(reconstructed)
                np.save(local_reconstructed_path, reconstructed_resized)
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
                    reconstructed_gradcam_resized = resize_volume_to_256(reconstructed_gradcam_heatmap)
                    np.save(local_reconstructed_gradcam_path, reconstructed_gradcam_resized)
                    
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

# ── RAG Reporting Agent Endpoints ────────────────────────────────────────

class RagQueryRequest(BaseModel):
    disease_name: str
    patient_metadata: dict
    llm_model: str = "gemini-3.5-flash"
    for_patient: bool = False

class GeneratePdfRequest(BaseModel):
    report_text: str
    patient_metadata: dict = {}
    study_id: str | None = None
    for_patient: bool = False

def generate_kspace_log_mag_npy(study_id: str) -> np.ndarray | None:
    try:
        # Find raw K-space key in MinIO kspace-raw
        objects = minio_client.list_objects("kspace-raw", prefix=f"{study_id}/", recursive=True)
        kspace_key = None
        for obj in objects:
            if "kspace_input" in obj.object_name:
                kspace_key = obj.object_name
                break
        
        if not kspace_key:
            logger.warning(f"Could not find raw K-space for study {study_id} to generate log mag.")
            return None
            
        with tempfile.TemporaryDirectory() as tmpdir:
            _, ext = os.path.splitext(kspace_key)
            local_kspace_path = os.path.join(tmpdir, f"kspace{ext}")
            minio_client.fget_object("kspace-raw", kspace_key, local_kspace_path)
            
            kspace = load_kspace(local_kspace_path)
            
            # Combine coils via RSS
            if kspace.ndim == 4:
                kspace_mag = np.sqrt(np.sum(np.abs(kspace)**2, axis=1))
            elif kspace.ndim == 3:
                kspace_mag = np.sqrt(np.sum(np.abs(kspace)**2, axis=0))
                kspace_mag = kspace_mag[np.newaxis, ...]
            else:
                kspace_mag = np.abs(kspace)[np.newaxis, ...]
                
            kspace_log_mag = np.log(1.0 + kspace_mag)
            
            # Normalize per slice
            for s in range(kspace_log_mag.shape[0]):
                slice_min = kspace_log_mag[s].min()
                slice_max = kspace_log_mag[s].max()
                denom = slice_max - slice_min
                if denom > 1e-8:
                    kspace_log_mag[s] = (kspace_log_mag[s] - slice_min) / denom
                else:
                    kspace_log_mag[s] = np.zeros_like(kspace_log_mag[s])
                    
            # Upload to MinIO so it's cached for next time
            local_log_mag_path = os.path.join(tmpdir, "kspace_log_mag.npy")
            np.save(local_log_mag_path, kspace_log_mag)
            
            minio_client.fput_object(
                bucket_name="reconstructed",
                object_name=f"{study_id}/kspace_log_mag.npy",
                file_path=local_log_mag_path
            )
            logger.info(f"Dynamically generated and uploaded kspace_log_mag.npy for study {study_id}")
            return kspace_log_mag
    except Exception as e:
        logger.error(f"Failed to dynamically generate kspace log mag: {e}")
        return None

def compile_pdf_with_rust(report_text: str, patient_metadata: dict, study_id: str | None = None, for_patient: bool = False) -> bytes:
    import subprocess
    import tempfile
    import os
    import numpy as np
    from PIL import Image as PILImage

    binary_path = None
    possible_paths = [
        "/usr/local/bin/report_pdf",
        "/app/report_pdf",
        "./report_pdf",
        os.path.join(os.path.dirname(__file__), "report_pdf"),
    ]
    for p in possible_paths:
        if os.path.exists(p) and os.access(p, os.X_OK):
            binary_path = p
            break
            
    if not binary_path:
        raise FileNotFoundError("report_pdf Rust binary not found or not executable in search paths.")

    f_in_name = None
    f_out_name = None
    mri_png_path = None
    kspace_png_path = None

    try:
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.txt', delete=False) as f_in:
            f_in.write(report_text)
            f_in_name = f_in.name

        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f_out:
            f_out_name = f_out.name

        cmd = [binary_path, "--input", f_in_name, "--output", f_out_name]
        
        if for_patient:
            cmd.append("--for-patient")
        
        if "name" in patient_metadata:
            cmd.extend(["--name", str(patient_metadata["name"])])
        if "age" in patient_metadata:
            cmd.extend(["--age", str(patient_metadata["age"])])
        if "sex" in patient_metadata:
            cmd.extend(["--sex", str(patient_metadata["sex"])])
        if "physician" in patient_metadata:
            cmd.extend(["--physician", str(patient_metadata["physician"])])
        if "report_id" in patient_metadata:
            cmd.extend(["--id", str(patient_metadata["report_id"])])
        if "patient_id" in patient_metadata:
            cmd.extend(["--patient-id", str(patient_metadata["patient_id"])])
        if "study_date" in patient_metadata:
            cmd.extend(["--study-date", str(patient_metadata["study_date"])])
        if "modality" in patient_metadata:
            cmd.extend(["--modality", str(patient_metadata["modality"])])
        if "date" in patient_metadata:
            cmd.extend(["--date", str(patient_metadata["date"])])

        if study_id and minio_client:
            # 1. MRI reconstructed image
            try:
                with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmp_npy:
                    tmp_npy_path = tmp_npy.name
                
                logger.info(f"Downloading reconstructed.npy for study {study_id} from MinIO...")
                minio_client.fget_object(
                    bucket_name="reconstructed",
                    object_name=f"{study_id}/reconstructed.npy",
                    file_path=tmp_npy_path
                )
                
                data = np.load(tmp_npy_path)
                if data.ndim >= 3:
                    slice_data = data[data.shape[0] // 2]
                else:
                    slice_data = data
                
                s_min, s_max = slice_data.min(), slice_data.max()
                denom = s_max - s_min
                normalized = (slice_data - s_min) / denom * 255.0 if denom > 1e-8 else np.zeros_like(slice_data)
                img = PILImage.fromarray(normalized.astype(np.uint8))
                
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_png:
                    mri_png_path = tmp_png.name
                img.save(mri_png_path, "PNG")
                cmd.extend(["--mri", mri_png_path])
                logger.info(f"Successfully generated temporary MRI PNG: {mri_png_path}")
                
                os.remove(tmp_npy_path)
            except Exception as mri_err:
                logger.error(f"Failed to fetch/convert MRI image for PDF: {mri_err}")

            # 2. K-Space log-magnitude image
            if not for_patient:
                kspace_data = None
                try:
                    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmp_npy:
                        tmp_npy_path = tmp_npy.name
                    
                    logger.info(f"Downloading kspace_log_mag.npy for study {study_id} from MinIO...")
                    minio_client.fget_object(
                        bucket_name="reconstructed",
                        object_name=f"{study_id}/kspace_log_mag.npy",
                        file_path=tmp_npy_path
                    )
                    
                    kspace_data = np.load(tmp_npy_path)
                    os.remove(tmp_npy_path)
                except Exception as ksp_err:
                    logger.warning(f"kspace_log_mag.npy not found for study {study_id}, trying dynamic generation...")
                    kspace_data = generate_kspace_log_mag_npy(study_id)
                    if kspace_data is None:
                        logger.error(f"Failed to fetch/convert K-space image for PDF: {ksp_err}")

                if kspace_data is not None:
                    try:
                        if kspace_data.ndim >= 3:
                            slice_data = kspace_data[kspace_data.shape[0] // 2]
                        else:
                            slice_data = kspace_data
                        
                        s_min, s_max = slice_data.min(), slice_data.max()
                        denom = s_max - s_min
                        normalized = (slice_data - s_min) / denom * 255.0 if denom > 1e-8 else np.zeros_like(slice_data)
                        img = PILImage.fromarray(normalized.astype(np.uint8))
                        
                        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_png:
                            kspace_png_path = tmp_png.name
                        img.save(kspace_png_path, "PNG")
                        cmd.extend(["--kspace", kspace_png_path])
                        logger.info(f"Successfully generated temporary K-Space PNG: {kspace_png_path}")
                    except Exception as img_err:
                        logger.error(f"Failed to convert K-space numpy to PNG: {img_err}")

        logger.info(f"Invoking Rust PDF generator: {' '.join(cmd)}")
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info(f"Rust compiler stdout: {res.stdout}")
        
        with open(f_out_name, 'rb') as f_pdf:
            pdf_bytes = f_pdf.read()
            
        return pdf_bytes
    finally:
        for path in [f_in_name, f_out_name, mri_png_path, kspace_png_path]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass

@app.post("/rag/ingest")
def rag_ingest(data_dir: str = "/app/data"):
    """Ingests all Markdown reference documents from the data directory into Upstash Redis."""
    from rag_agent.ingest import run_full_ingestion
    try:
        if not os.path.exists(data_dir):
            raise HTTPException(status_code=404, detail=f"Data directory '{data_dir}' not found.")
        
        success_count = run_full_ingestion(data_dir)
        return {
            "status": "success",
            "message": f"Successfully ingested {success_count} files into Upstash Redis."
        }
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag/query")
def rag_query(request: RagQueryRequest):
    """Queries RAG and generates a radiology report for a patient."""
    from rag_agent.query import generate_radiology_report
    try:
        report = generate_radiology_report(
            disease_name=request.disease_name,
            patient_metadata=request.patient_metadata,
            llm_model=request.llm_model,
            for_patient=request.for_patient
        )
        if report.startswith("Error:"):
            raise HTTPException(status_code=404, detail=report)
        return {
            "status": "success",
            "report": report
        }
    except Exception as e:
        logger.error(f"RAG query generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag/generate-pdf")
def rag_generate_pdf(request: GeneratePdfRequest):
    """Generates a styled radiology report PDF using the compiled Rust binary."""
    from fastapi.responses import Response
    import io
    try:
        pdf_bytes = compile_pdf_with_rust(
            request.report_text,
            request.patient_metadata,
            study_id=request.study_id,
            for_patient=request.for_patient
        )
        
        if request.study_id and minio_client:
            try:
                object_name = f"{request.study_id}/report.pdf"
                logger.info(f"Uploading PDF report to 'reports' bucket with key {object_name}...")
                minio_client.put_object(
                    bucket_name="reports",
                    object_name=object_name,
                    data=io.BytesIO(pdf_bytes),
                    length=len(pdf_bytes),
                    content_type="application/pdf"
                )
            except Exception as upload_err:
                logger.error(f"Failed to upload PDF report to MinIO: {upload_err}")
        
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=radiology_report.pdf"}
        )
    except Exception as e:
        logger.error(f"PDF generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/progression", response_model=ProgressionResponse)
def progression_projection(request: ProgressionRequest):
    """
    Given a pathology class and initial volume, forecasts how the condition
    would evolve over a 24-month period if untreated.
    """
    pathology = request.pathology
    v0 = request.initial_pathology_volume_cm3
    
    # Establish baseline defaults for initial volumes based on pathology
    default_volumes = {
        "Normal": 0.0,
        "Tumor_Glioma": 15.0,
        "Ischemia": 25.0,
        "MS_Lesions": 8.0,
        "Hydrocephalus": 60.0,
        "Atrophy": 35.0, # represented as ventricle volume
        "Hemorrhage": 30.0,
        "Cerebral_Cyst": 12.0,
        "Edema": 10.0,
        "AVM": 18.0,
        "Cerebral_Microbleeds": 2.0
    }
    
    init_vol = v0 if v0 is not None else default_volumes.get(pathology, 5.0)
    
    timeline = []
    months = [0, 3, 6, 12, 18, 24]
    
    for m in months:
        # Defaults
        path_vol = 0.0
        edema_vol = 0.0
        healthy_vol = 1350.0
        cog_impact = 0.0
        severity = "Mild"
        note = "No significant untreated progression expected."
        
        if pathology == "Normal":
            severity = "Normal"
            note = "Patient brain structures remain within normal limits."
            
        elif pathology == "Tumor_Glioma":
            path_vol = init_vol * np.exp(0.08 * m)
            edema_vol = 0.6 * path_vol + 1.0 + (m * 0.5)
            healthy_vol = 1350.0 - 0.7 * (path_vol + edema_vol)
            cog_impact = min(100.0, 10.0 + 0.8 * (path_vol + edema_vol))
            if m == 0:
                note = "Initial focal tumor mass identified with mild surrounding vasogenic edema."
                severity = "Mild" if path_vol < 10 else "Moderate"
            elif m <= 6:
                note = "Progression of neoplastic growth and vasogenic edema. Mild compression on local parenchyma."
                severity = "Moderate"
            elif m <= 18:
                note = "Marked tumor expansion. Local mass effect and midline shift beginning to manifest."
                severity = "Severe"
            else:
                note = "Critical mass effect. Severe herniation risk, heavy midline shift, and significant intracranial pressure."
                severity = "Critical"
                
        elif pathology == "Atrophy":
            # Primary: Ventricles enlarging, healthy brain volume decreasing
            path_vol = init_vol + 1.8 * m  # Ventricle expansion
            healthy_vol = 1200.0 - 2.8 * m  # Cortical volume loss
            cog_impact = min(100.0, 15.0 + 2.2 * m)
            if m == 0:
                note = "Mild enlargement of ventricles and prominent cortical sulci consistent with early atrophy."
                severity = "Mild"
            elif m <= 6:
                note = "Gradual progressive enlargement of CSF spaces, widening of cortical sulci."
                severity = "Moderate"
            elif m <= 18:
                note = "Severe loss of gray matter volume, prominent ventriculomegaly ex-vacuo."
                severity = "Severe"
            else:
                note = "End-stage global cortical atrophy. Marked cognitive impairment and generalized tissue volume loss."
                severity = "Critical"
                
        elif pathology == "MS_Lesions":
            path_vol = init_vol + 0.5 * m + np.sin(m * 0.8) * 1.2
            edema_vol = 1.5 + np.cos(m * 0.8) * 0.8
            healthy_vol = 1300.0 - 0.9 * m
            cog_impact = min(100.0, 12.0 + 1.8 * m)
            if m == 0:
                note = "Disseminated demyelinating plaques in periventricular white matter."
                severity = "Mild"
            elif m <= 6:
                note = "Subacute lesion activity and variable local inflammation. Active demyelination plaques."
                severity = "Moderate"
            elif m <= 18:
                note = "Confluent lesion burden. Secondary axonal loss and brain volume loss beginning to accelerate."
                severity = "Severe"
            else:
                note = "Extensive plaque fusion, severe myelin depletion, and permanent neurological disability markers."
                severity = "Critical"
                
        elif pathology == "Hydrocephalus":
            # Ventricle volume grows rapidly then plateaus
            path_vol = init_vol + 85.0 * (1.0 - np.exp(-0.12 * m))
            healthy_vol = 1250.0 - 0.65 * (path_vol - init_vol)
            cog_impact = min(100.0, 20.0 + 3.2 * m)
            if m == 0:
                note = "Ventriculomegaly with transependymal flow of CSF indicating active pressure elevation."
                severity = "Moderate"
            elif m <= 6:
                note = "Progressive ventriculomegaly. Active compression on adjacent white matter tracts."
                severity = "Severe"
            elif m <= 18:
                note = "Severe ventricular dilation. Thinning of corpus callosum and progressive periventricular damage."
                severity = "Critical"
            else:
                note = "Chronic end-stage hydrocephalus. Severe cognitive, gait, and physiological impairment."
                severity = "Critical"
                
        elif pathology == "Ischemia":
            # Core slowly retracts/scars, edema peaks early and completely resolves
            path_vol = init_vol * np.exp(-0.03 * m)
            edema_vol = 15.0 * np.exp(-0.6 * m) if m > 0 else 15.0
            healthy_vol = 1300.0 - path_vol - edema_vol
            cog_impact = max(20.0, 50.0 - 0.9 * m)
            if m == 0:
                note = "Acute cytotoxic edema and ischemic core. Large region of high risk infarction."
                severity = "Severe"
            elif m <= 6:
                note = "Resolution of cytotoxic edema. Liquefactive necrosis and tissue resorption in progress."
                severity = "Moderate"
            else:
                note = "Residual encephalomalacia, gliosis, and focal volume loss (glial scar tissue)."
                severity = "Mild" if path_vol < 15 else "Moderate"
                
        elif pathology == "Hemorrhage":
            # Hematoma absorbs, edema resolves
            path_vol = init_vol * np.exp(-0.09 * m)
            edema_vol = 20.0 * np.exp(-0.5 * m) if m > 0 else 20.0
            healthy_vol = 1320.0 - path_vol - edema_vol
            cog_impact = max(15.0, 55.0 - 1.2 * m)
            if m == 0:
                note = "Acute intraparenchymal hematoma with significant surrounding vasogenic edema and mass effect."
                severity = "Severe"
            elif m <= 6:
                note = "Clot resorption, breakdown of blood products (hemosiderin deposition), and resolving edema."
                severity = "Moderate"
            else:
                note = "Complete hematoma resorption leaving a cystic cavity (slit-like defect) surrounded by hemosiderin staining."
                severity = "Mild"
                
        else: # Generic default (Cyst, Edema, AVM, Microbleeds)
            path_vol = init_vol + 0.1 * m
            edema_vol = 2.0 + 0.1 * m
            healthy_vol = 1330.0 - path_vol
            cog_impact = min(100.0, 5.0 + 1.0 * m)
            if m == 0:
                note = f"Initial presentation of {pathology.replace('_', ' ')}."
                severity = "Mild"
            elif m <= 12:
                note = f"Stable or extremely slow expansion of {pathology.replace('_', ' ')}."
                severity = "Moderate"
            else:
                note = f"Chronic presentation of {pathology.replace('_', ' ')} with stable cognitive markers."
                severity = "Moderate"

        timeline.append(
            ProgressionPoint(
                month=m,
                pathology_volume_cm3=round(float(path_vol), 2),
                edema_volume_cm3=round(float(edema_vol), 2),
                healthy_brain_volume_cm3=round(float(healthy_vol), 2),
                cognitive_impact_pct=round(float(cog_impact), 1),
                severity_level=severity,
                clinical_note=note
            )
        )
        
    return ProgressionResponse(
        status="success",
        pathology=pathology,
        initial_volume_cm3=round(float(init_vol), 2),
        timeline=timeline
    )


