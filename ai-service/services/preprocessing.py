"""
KVision 4.0 — Preprocessing Utilities
Common volume/k-space preprocessing used across multiple routes.
"""

import logging
import numpy as np
import torch

from config import (
    TARGET_RESOLUTION_CLASSIFIER,
    TARGET_RESOLUTION_ANOMALY,
    TARGET_SLICES,
    TARGET_COILS,
    OUTPUT_RESOLUTION,
)

logger = logging.getLogger("ai-service")


def resize_volume_to_target(volume: np.ndarray, target_h: int = OUTPUT_RESOLUTION, target_w: int = OUTPUT_RESOLUTION) -> np.ndarray:
    """
    Resizes a volume or image to [..., target_h, target_w] using bilinear interpolation.
    Works for 2D [H, W], 3D [slices, H, W], and 4D [time, slices, H, W] arrays.
    """
    orig_shape = volume.shape
    if len(orig_shape) < 2:
        return volume

    h, w = orig_shape[-2], orig_shape[-1]
    if h == target_h and w == target_w:
        return volume

    flat_outer = int(np.prod(orig_shape[:-2])) if len(orig_shape) > 2 else 1
    reshaped = volume.reshape((flat_outer, h, w))

    tensor = torch.from_numpy(reshaped.astype(np.float32)).unsqueeze(1)
    resized_tensor = torch.nn.functional.interpolate(
        tensor, size=(target_h, target_w), mode="bilinear", align_corners=False
    )
    resized_flat = resized_tensor.squeeze(1).numpy().astype(volume.dtype)

    if len(orig_shape) == 2:
        return resized_flat[0]
    else:
        return resized_flat.reshape(orig_shape[:-2] + (target_h, target_w))


def prepare_kspace_for_classifier(kspace: np.ndarray) -> torch.Tensor:
    """
    Preprocesses raw k-space [slices, coils, H, W] into the tensor format
    expected by the Fused S4-CNN classifier: [1, 8, 16, 128, 128] complex64.

    Steps:
        1. Crop/pad spatial dims to 128×128
        2. Interpolate slice dim to 8
        3. Pad/truncate coils to 16
        4. Normalize by magnitude std
    """
    target_res = TARGET_RESOLUTION_CLASSIFIER

    x_complex = kspace.astype(np.complex64)
    x_tensor = torch.from_numpy(x_complex)  # [slices, coils, H, W]
    slices_in, coils_in, h_in, w_in = x_tensor.shape

    # Crop/pad height to target_res
    x_tensor = _crop_or_pad_dim(x_tensor, dim=2, target=target_res)
    # Crop/pad width to target_res
    _, _, h_in, w_in = x_tensor.shape
    x_tensor = _crop_or_pad_dim(x_tensor, dim=3, target=target_res)

    # Separate real/imag for trilinear interpolation
    x_real = torch.real(x_tensor)
    x_imag = torch.imag(x_tensor)

    # Interpolate slices to TARGET_SLICES: [coils, 1, slices, H, W] → [coils, 1, 8, H, W]
    x_real_5d = x_real.permute(1, 0, 2, 3).unsqueeze(1)
    x_imag_5d = x_imag.permute(1, 0, 2, 3).unsqueeze(1)

    real_interp = torch.nn.functional.interpolate(
        x_real_5d, size=(TARGET_SLICES, target_res, target_res), mode="trilinear", align_corners=False
    ).squeeze(1)
    imag_interp = torch.nn.functional.interpolate(
        x_imag_5d, size=(TARGET_SLICES, target_res, target_res), mode="trilinear", align_corners=False
    ).squeeze(1)

    # Pad/truncate coils to TARGET_COILS
    final_real = torch.zeros(TARGET_COILS, TARGET_SLICES, target_res, target_res, dtype=torch.float32)
    final_imag = torch.zeros(TARGET_COILS, TARGET_SLICES, target_res, target_res, dtype=torch.float32)

    c = min(coils_in, TARGET_COILS)
    final_real[:c] = real_interp[:c]
    final_imag[:c] = imag_interp[:c]

    # Permute to [slices, coils, H, W] and recombine as complex
    final_real = final_real.permute(1, 0, 2, 3)
    final_imag = final_imag.permute(1, 0, 2, 3)
    x_final = torch.complex(final_real, final_imag).unsqueeze(0)  # [1, 8, 16, 128, 128]

    # Normalize by magnitude std (as in training)
    norm_factor = torch.std(torch.abs(x_final))
    if norm_factor > 0:
        x_final = x_final / norm_factor

    return x_final


def prepare_kspace_for_anomaly_estimator(kspace: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Preprocesses raw k-space [slices, coils, H, W] into the tensor format
    expected by the SSM Anomaly Estimator: [slices, 32, 256, 256] float32 + contrast tensor.

    Returns:
        (x_input, contrast_tensor): ready for model forward pass
    """
    target_res = TARGET_RESOLUTION_ANOMALY
    x_complex = kspace.astype(np.complex64)
    x_tensor = torch.from_numpy(x_complex)
    slices_in, coils_in, h_in, w_in = x_tensor.shape

    # Crop/pad spatial to 256×256
    x_tensor = _crop_or_pad_dim(x_tensor, dim=2, target=target_res)
    x_tensor = _crop_or_pad_dim(x_tensor, dim=3, target=target_res)

    # Pad/truncate coils
    slices_in, coils_in_adj, _, _ = x_tensor.shape
    if coils_in <= TARGET_COILS:
        final = torch.zeros(slices_in, TARGET_COILS, target_res, target_res, dtype=x_tensor.dtype)
        final[:, :coils_in] = x_tensor
    else:
        final = x_tensor[:, :TARGET_COILS]

    # Per-slice magnitude normalization
    normalized = torch.zeros_like(final)
    for s in range(slices_in):
        slice_abs = torch.abs(final[s])
        nf = torch.std(slice_abs)
        normalized[s] = final[s] / nf if nf > 0 else final[s]

    # Stack real + imag → [slices, 32, 256, 256]
    x_real = torch.real(normalized)
    x_imag = torch.imag(normalized)
    x_input = torch.cat([x_real, x_imag], dim=1)

    # Contrast tensor: 0=T1 for first slice, 1=T2 for rest
    contrast = torch.tensor([0 if s == 0 else 1 for s in range(slices_in)], dtype=torch.long)

    return x_input, contrast


def _crop_or_pad_dim(tensor: torch.Tensor, dim: int, target: int) -> torch.Tensor:
    """Crops or zero-pads a tensor along the specified dimension to the target size."""
    current = tensor.shape[dim]
    if current == target:
        return tensor
    elif current > target:
        start = (current - target) // 2
        return tensor.narrow(dim, start, target)
    else:
        pad_before = (target - current) // 2
        pad_after = target - current - pad_before
        # Build pad tuple (PyTorch pads from last dim backward)
        n_dims = tensor.ndim
        pad_list = [0] * (2 * n_dims)
        pad_idx = 2 * (n_dims - 1 - dim)
        pad_list[pad_idx] = pad_before
        pad_list[pad_idx + 1] = pad_after
        return torch.nn.functional.pad(tensor, pad_list)
