"""
Denoising Module for KVISION AI Service.
Provides a Non-Local Means (NLM) baseline and a PyTorch DnCNN residual model.
"""

import logging
import os
import numpy as np
import torch
import torch.nn as nn
from skimage.restoration import denoise_nl_means, estimate_sigma

logger = logging.getLogger(__name__)


class DnCNN(nn.Module):
    """
    DnCNN (Denoising Convolutional Neural Network) architecture.
    Uses residual learning: predicts the residual noise, which is then
    subtracted from the input image to obtain the denoised image.
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 1, num_layers: int = 17, num_features: int = 64):
        super(DnCNN, self).__init__()
        layers = [
            nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        ]
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(num_features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1, bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Predict the residual noise
        noise = self.dncnn(x)
        # Residual learning: subtract the noise from input
        return x - noise


def train_dncnn(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader = None,
    epochs: int = 5,
    lr: float = 1e-3,
    device: str = None
) -> nn.Module:
    """
    Train or fine-tune the DnCNN model to denoise images.

    Parameters:
    -----------
    model : nn.Module
        The DnCNN model instance.
    train_loader : DataLoader
        DataLoader yielding batches of:
        - (noisy_images, clean_images)
        - or clean_images only (in which case synthetic Gaussian noise is added).
    val_loader : DataLoader, optional
        DataLoader for validation data. Same format as train_loader.
    epochs : int
        Number of training epochs. Default is 5.
    lr : float
        Learning rate for the Adam optimizer. Default is 1e-3.
    device : str, optional
        Device to train on ('cuda', 'cpu', or None for auto-detection).

    Returns:
    --------
    model : nn.Module
        The trained model.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    logger.info(f"Training DnCNN on device: {device}")

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total_samples = 0

        for batch in train_loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                noisy, clean = batch
            else:
                clean = batch
                # Simulate noise: random sigma between 5/255 and 50/255
                noise_sigma = np.random.uniform(5.0, 50.0) / 255.0
                noise = torch.randn_like(clean) * noise_sigma
                noisy = clean + noise

            noisy = noisy.to(device)
            clean = clean.to(device)

            optimizer.zero_grad()
            denoised = model(noisy)
            loss = criterion(denoised, clean)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * noisy.size(0)
            total_samples += noisy.size(0)

        epoch_loss = running_loss / total_samples

        # Validation
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_samples = 0
            with torch.no_grad():
                for batch in val_loader:
                    if isinstance(batch, (list, tuple)) and len(batch) == 2:
                        noisy, clean = batch
                    else:
                        clean = batch
                        # Fixed sigma for validation
                        noise = torch.randn_like(clean) * (25.0 / 255.0)
                        noisy = clean + noise

                    noisy = noisy.to(device)
                    clean = clean.to(device)

                    denoised = model(noisy)
                    loss = criterion(denoised, clean)
                    val_loss += loss.item() * noisy.size(0)
                    val_samples += noisy.size(0)

            val_epoch_loss = val_loss / val_samples
            logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_loss:.6f} | Val Loss: {val_epoch_loss:.6f}")
        else:
            logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_loss:.6f}")

    return model


def _denoise_nlm_2d(image: np.ndarray, **kwargs) -> np.ndarray:
    """
    Helper to denoise a single 2D image (real-valued) using Non-Local Means.
    """
    min_val = image.min()
    max_val = image.max()
    rng = max_val - min_val

    if rng > 1e-8:
        scaled = (image - min_val) / rng
    else:
        return image.copy()

    patch_size = kwargs.get('patch_size', 5)
    patch_distance = kwargs.get('patch_distance', 7)
    fast_mode = kwargs.get('fast_mode', True)
    h = kwargs.get('h', None)

    if h is None:
        try:
            sigma_est = np.mean(estimate_sigma(scaled, channel_axis=None))
        except TypeError:
            # Fallback for older scikit-image versions without channel_axis
            sigma_est = np.mean(estimate_sigma(scaled))
        h_val = 0.8 * sigma_est if sigma_est > 0 else 0.1
    else:
        # Scale user-provided h parameter to fit [0, 1] range
        h_val = h / rng

    denoised_scaled = denoise_nl_means(
        scaled,
        patch_size=patch_size,
        patch_distance=patch_distance,
        h=h_val,
        fast_mode=fast_mode
    )

    return denoised_scaled * rng + min_val


def _denoise_dncnn_2d(image: np.ndarray, model: nn.Module, device: torch.device) -> np.ndarray:
    """
    Helper to denoise a single 2D image (real-valued) using a DnCNN model.
    """
    min_val = image.min()
    max_val = image.max()
    rng = max_val - min_val

    if rng > 1e-8:
        scaled = (image - min_val) / rng
    else:
        return image.copy()

    # Shape: (1, 1, H, W)
    tensor = torch.from_numpy(scaled).float().unsqueeze(0).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        denoised_tensor = model(tensor)

    denoised_scaled = denoised_tensor.squeeze(0).squeeze(0).cpu().numpy()
    return denoised_scaled * rng + min_val


def _apply_2d_denoiser(image: np.ndarray, denoiser_2d_func) -> np.ndarray:
    """
    Recursively apply a 2D denoiser function slice-by-slice across 3D or 4D stacks.
    Supports complex data by denoising real and imaginary channels separately.
    """
    is_complex = np.iscomplexobj(image)

    def process_2d(img2d: np.ndarray) -> np.ndarray:
        if is_complex:
            real_denoised = denoiser_2d_func(np.real(img2d))
            imag_denoised = denoiser_2d_func(np.imag(img2d))
            return real_denoised + 1j * imag_denoised
        else:
            return denoiser_2d_func(img2d)

    ndim = image.ndim
    if ndim == 2:
        return process_2d(image)
    elif ndim == 3:
        slices = []
        for i in range(image.shape[0]):
            slices.append(process_2d(image[i]))
        return np.stack(slices, axis=0)
    elif ndim == 4:
        volumes = []
        for t in range(image.shape[0]):
            slices = []
            for z in range(image.shape[1]):
                slices.append(process_2d(image[t, z]))
            volumes.append(np.stack(slices, axis=0))
        return np.stack(volumes, axis=0)
    else:
        raise ValueError(f"Image must be 2D, 3D, or 4D. Got dimension: {ndim}")


def denoise_image(
    image: np.ndarray,
    method: str = 'dncnn',
    model: nn.Module = None,
    device: str = None,
    **kwargs
) -> np.ndarray:
    """
    Denoise a reconstructed 2D or 3D image using NLM baseline or a DnCNN model.
    Supports complex-valued MRI reconstructions.

    Parameters:
    -----------
    image : np.ndarray
        Reconstructed image to denoise. Shape can be (H, W), (N, H, W) or (T, Z, H, W).
    method : str
        Denoising method: 'nlm' or 'dncnn'. Default is 'dncnn'.
    model : nn.Module, optional
        Pre-trained DnCNN model instance. If None and method is 'dncnn',
        a default DnCNN model will be instantiated. If `dncnn.pth` exists
        in the same directory, its weights will be loaded automatically.
    device : str, optional
        Device to run inference on ('cuda', 'cpu', or None for auto-detection).
    **kwargs :
        Additional arguments passed to the Non-Local Means filter:
        - patch_size : int (default: 5)
        - patch_distance : int (default: 7)
        - h : float (default: estimated using noise sigma)
        - fast_mode : bool (default: True)

    Returns:
    --------
    denoised_image : np.ndarray
        Denoised image of the same shape and type as input.
    """
    method_lower = method.lower()
    if method_lower not in ('nlm', 'dncnn'):
        raise ValueError(f"Unsupported denoising method: {method}. Choose 'nlm' or 'dncnn'.")

    if method_lower == 'nlm':
        def denoiser_func(image: np.ndarray) -> np.ndarray:
            return _denoise_nlm_2d(image, **kwargs)
        return _apply_2d_denoiser(image, denoiser_func)

    elif method_lower == 'dncnn':
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        torch_device = torch.device(device)

        if model is None:
            model = DnCNN(in_channels=1, out_channels=1)
            checkpoint_path = os.path.join(os.path.dirname(__file__), 'dncnn.pth')
            if os.path.exists(checkpoint_path):
                try:
                    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
                    logger.info(f"Loaded pre-trained DnCNN weights from {checkpoint_path}")
                except Exception as e:
                    logger.warning(f"Could not load DnCNN weights from {checkpoint_path}: {e}")
            else:
                logger.warning("No pre-trained DnCNN weights found. Running with randomly initialized weights.")

        model.to(torch_device)

        def denoiser_func(image: np.ndarray) -> np.ndarray:
            return _denoise_dncnn_2d(image, model, torch_device)
        return _apply_2d_denoiser(image, denoiser_func)
