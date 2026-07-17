"""
KVision 4.0 — Grad-CAM Explainability Service
Computes Grad-CAM heatmaps on both the K-space encoder branch
and the reconstructed spatial branch of the Fused S4-CNN classifier.
"""

import logging
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger("ai-service")


def compute_kspace_gradcam(model: nn.Module, x_final: torch.Tensor, target_class_idx: int) -> np.ndarray | None:
    """
    Computes a Grad-CAM heatmap on the K-space encoder branch (conv4 of KSpaceS4Encoder).

    Args:
        model: The Fused S4-CNN classifier model.
        x_final: Input tensor of shape [1, S, C, H, W] (complex64).
        target_class_idx: Index of the target class for gradient computation.

    Returns:
        Normalized Grad-CAM heatmap of shape [S, 128, 128] or None on failure.
    """
    model.eval()

    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    # Target: final conv layer in the K-space encoder
    target_layer = model.s4_branch.encoder.conv4
    h_forward = target_layer.register_forward_hook(forward_hook)
    h_backward = target_layer.register_full_backward_hook(backward_hook)

    try:
        with torch.enable_grad():
            x_input = x_final.detach().clone()
            logits = model(x_input)
            score = logits[0, target_class_idx]
            model.zero_grad()
            score.backward()
    except Exception as e:
        logger.error(f"Error in compute_kspace_gradcam: {e}")
        return None
    finally:
        h_forward.remove()
        h_backward.remove()

    if not activations or not gradients:
        logger.warning("K-space Grad-CAM hooks failed to capture activations/gradients")
        return None

    # act shape: [B*S, 128, H_feat, W_feat]
    act = activations[0].detach()
    grad = gradients[0].detach()

    # Global average pooling of gradients → channel weights
    weights = torch.mean(grad, dim=(2, 3), keepdim=True)

    # Weighted combination → ReLU → spatial heatmap
    cam = torch.clamp(torch.sum(weights * act, dim=1), min=0)

    # Upsample to target resolution
    cam = cam.unsqueeze(0).unsqueeze(1)
    cam_upsampled = torch.nn.functional.interpolate(
        cam, size=(8, 128, 128), mode="trilinear", align_corners=False
    ).squeeze(0).squeeze(0)

    return _normalize_cam(cam_upsampled.cpu().numpy())


def compute_reconstructed_gradcam(model: nn.Module, x_final: torch.Tensor, target_class_idx: int) -> np.ndarray | None:
    """
    Computes a Grad-CAM heatmap on the reconstructed spatial branch (conv4 of VolumeCNNClassifier).

    Args:
        model: The Fused S4-CNN classifier model.
        x_final: Input tensor of shape [1, S, C, H, W] (complex64).
        target_class_idx: Index of the target class for gradient computation.

    Returns:
        Normalized Grad-CAM heatmap of shape [S, H, W] or None on failure.
    """
    model.eval()

    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    target_layer = model.cnn_branch.conv4
    h_forward = target_layer.register_forward_hook(forward_hook)
    h_backward = target_layer.register_full_backward_hook(backward_hook)

    try:
        with torch.enable_grad():
            x_input = x_final.detach().clone()
            logits = model(x_input)
            score = logits[0, target_class_idx]
            model.zero_grad()
            score.backward()
    except Exception as e:
        logger.error(f"Error in compute_reconstructed_gradcam: {e}")
        return None
    finally:
        h_forward.remove()
        h_backward.remove()

    if not activations or not gradients:
        logger.warning("Reconstructed Grad-CAM hooks failed to capture activations/gradients")
        return None

    act = activations[0].detach()
    grad = gradients[0].detach()

    # 3D global average pooling of gradients
    weights = torch.mean(grad, dim=(2, 3, 4), keepdim=True)
    cam = torch.clamp(torch.sum(weights * act, dim=1), min=0)

    # Upsample to match input volume dimensions
    _, slices, _, height, width = x_final.shape
    cam = cam.unsqueeze(1)
    cam_upsampled = torch.nn.functional.interpolate(
        cam, size=(slices, height, width), mode="trilinear", align_corners=False
    ).squeeze(0).squeeze(0)

    return _normalize_cam(cam_upsampled.cpu().numpy())


def _normalize_cam(cam_np: np.ndarray) -> np.ndarray:
    """Normalizes a Grad-CAM volume per-slice to [0, 1]."""
    for s in range(cam_np.shape[0]):
        s_min = cam_np[s].min()
        s_max = cam_np[s].max()
        denom = s_max - s_min
        if denom > 1e-8:
            cam_np[s] = (cam_np[s] - s_min) / denom
        else:
            cam_np[s] = np.zeros_like(cam_np[s])
    return cam_np
