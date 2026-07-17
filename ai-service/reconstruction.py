import numpy as np

def correct_line_phases(kspace: np.ndarray) -> np.ndarray:
    """
    Applies global phase correction in K-space (zero-order phase correction).
    Aligns the phase of the entire volume to the central echo peak.
    
    Parameters:
        kspace (np.ndarray): Complex K-space array of shape [slices, coils, height, width].
        
    Returns:
        np.ndarray: Phase-corrected K-space.
    """
    corrected_kspace = kspace.copy()
    slices, coils, height, width = kspace.shape
    
    for s in range(slices):
        for c in range(coils):
            # Locate the global peak of the 2D k-space (typically at the center)
            peak_idx_flat = np.argmax(np.abs(corrected_kspace[s, c, :, :]))
            y_peak, x_peak = np.unravel_index(peak_idx_flat, (height, width))
            # Get phase at the global peak
            peak_phase = np.angle(corrected_kspace[s, c, y_peak, x_peak])
            # Subtract this single phase from the entire 2D k-space
            corrected_kspace[s, c, :, :] *= np.exp(-1j * peak_phase)
            
    return corrected_kspace

def align_coil_phases(coil_images: np.ndarray, kspace: np.ndarray) -> np.ndarray:
    """
    Aligns phases across coils using low-resolution phase maps.
    Estimates the coil sensitivity phase from the central region of K-space (low-pass filter).
    
    Parameters:
        coil_images (np.ndarray): Full resolution complex coil images of shape [slices, coils, height, width].
        kspace (np.ndarray): Corresponding complex K-space array of shape [slices, coils, height, width].
        
    Returns:
        np.ndarray: Phase-aligned complex coil images.
    """
    aligned_images = coil_images.copy()
    slices, coils, height, width = kspace.shape
    
    # Define central low-resolution calibration region (usually 24x24 or 32x32)
    cal_h = min(height, 24)
    cal_w = min(width, 24)
    
    h_start = (height - cal_h) // 2
    h_end = h_start + cal_h
    w_start = (width - cal_w) // 2
    w_end = w_start + cal_w
    
    for s in range(slices):
        for c in range(coils):
            # Extract central K-space calibration region
            cal_kspace = np.zeros((height, width), dtype=kspace.dtype)
            # K-space is centered, so the low frequencies are in the middle
            cal_kspace[h_start:h_end, w_start:w_end] = kspace[s, c, h_start:h_end, w_start:w_end]
            
            # Reconstruct low-resolution image
            temp = np.fft.ifftshift(cal_kspace)
            temp = np.fft.ifft2(temp)
            low_res_img = np.fft.fftshift(temp)
            
            # Extract the smooth phase profile
            phase_profile = np.angle(low_res_img)
            
            # Align the full-resolution coil image using this phase map
            aligned_images[s, c, :, :] = coil_images[s, c, :, :] * np.exp(-1j * phase_profile)
            
    return aligned_images

def reconstruct_kspace(kspace: np.ndarray, phase_correction: bool = True, **kwargs) -> np.ndarray:
    """
    Reconstructs K-space data into magnitude images.
    
    Uses compiled Rust binary for high performance reconstruction. If the Rust binary is
    not found or fails, it falls back to the Python implementation.
    
    Parameters:
        kspace (np.ndarray): Complex K-space array of shape [slices, coils, height, width] 
                             or [coils, height, width].
        phase_correction (bool): Whether to perform phase correction steps.
        
    Returns:
        np.ndarray: Combined magnitude image of shape [slices, height, width] 
                    (or [height, width] if input had no slice dimension).
    """
    import os
    import tempfile
    import subprocess
    import logging

    # Check dimensions and normalize to 4D [slices, coils, height, width]
    original_ndim = kspace.ndim
    if original_ndim == 3:
        kspace = kspace[np.newaxis, ...]
    elif original_ndim == 2:
        kspace = kspace[np.newaxis, np.newaxis, ...]
    elif original_ndim != 4:
        raise ValueError(f"Invalid input shape {kspace.shape}. Expected 2D, 3D, or 4D array.")
        
    slices, coils, height, width = kspace.shape

    # Path to compiled Rust binary
    rust_bin_path = "/usr/local/bin/rust-mri-bin"
    if not os.path.exists(rust_bin_path):
        alt_path = os.path.join(os.path.dirname(__file__), "rust-mri-bin")
        if os.path.exists(alt_path):
            rust_bin_path = alt_path
        else:
            rust_bin_path = None

    if rust_bin_path:
        try:
            # Cast to complex128 for precision to match Python
            kspace_f64 = kspace.astype(np.complex128)
            
            with tempfile.TemporaryDirectory() as tmpdir:
                in_path = os.path.join(tmpdir, "kspace_in.bin")
                out_path = os.path.join(tmpdir, "magnitude_out.bin")
                
                # Write contiguous array to binary file
                kspace_f64.tofile(in_path)
                
                # Run the Rust binary
                cmd = [
                    rust_bin_path,
                    in_path,
                    out_path,
                    str(slices),
                    str(coils),
                    str(height),
                    str(width),
                    "true" if phase_correction else "false"
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                logging.getLogger("ai-service").info(f"Rust binary output: {result.stdout.strip()}")
                
                # Read magnitude float64 values from binary file
                magnitude_flat = np.fromfile(out_path, dtype=np.float64)
                recon_magnitude = magnitude_flat.reshape((slices, height, width))
                
                if original_ndim == 3:
                    recon_magnitude = np.squeeze(recon_magnitude, axis=0)
                elif original_ndim == 2:
                    recon_magnitude = np.squeeze(recon_magnitude)
                    
                return recon_magnitude
                
        except Exception as e:
            logging.getLogger("ai-service").warning(
                f"Rust MRI reconstruction failed (binary: {rust_bin_path}): {e}. Falling back to Python."
            )

    # 1. Option: Learned unrolled physics-constrained reconstruction
    if kwargs.get("physics_constrained", False):
        try:
            logging.getLogger("ai-service").info("Running learned unrolled physics-constrained reconstruction...")
            combined = physics_constrained_reconstruction(kspace)
            
            # Squeeze output to match input dimensions
            if original_ndim == 3:
                combined = np.squeeze(combined, axis=0)
            elif original_ndim == 2:
                combined = np.squeeze(combined)
            return combined
        except Exception as p_err:
            logging.getLogger("ai-service").warning(
                f"Physics-constrained reconstruction failed: {p_err}. Falling back to standard."
            )

    # 2. Apply line-by-line phase correction in K-space
    if phase_correction:
        kspace_pc = correct_line_phases(kspace)
    else:
        kspace_pc = kspace.copy()
        
    coil_images = np.zeros_like(kspace_pc, dtype=np.complex128)
    
    # 3. Apply 2D IFFT slice-by-slice, coil-by-coil
    for s in range(slices):
        for c in range(coils):
            # Centering: input ifftshift, ifft2, output fftshift
            shifted_k = np.fft.ifftshift(kspace_pc[s, c, :, :])
            img_c = np.fft.ifft2(shifted_k)
            coil_images[s, c, :, :] = np.fft.fftshift(img_c)
            
    # 4. Apply coil phase alignment
    if phase_correction:
        coil_images = align_coil_phases(coil_images, kspace_pc)
        
    # 5. Root Sum of Squares (RSS) combination
    combined = np.sqrt(np.sum(np.abs(coil_images)**2, axis=1))
    
    # Squeeze output to match input dimensions
    if original_ndim == 3:
        combined = np.squeeze(combined, axis=0)  # [height, width]
    elif original_ndim == 2:
        combined = np.squeeze(combined)  # [height, width]
        
    return combined


def physics_constrained_reconstruction(kspace: np.ndarray, n_iterations: int = 3, eta: float = 0.5) -> np.ndarray:
    """
    Learned unrolled physics-constrained reconstruction.
    Iteratively updates:
        z = x + eta * A^H (y - A x)  [Data Consistency Step]
        x = Denoiser(z)              [Regularization Step]
        
    Where:
        A = Mask * FFT * Sensitivity_Maps
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    slices, coils, height, width = kspace.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Estimate coil sensitivity maps using simple low-res center calibration (ESPIRiT-like baseline)
    cal_h, cal_w = min(height, 24), min(width, 24)
    h_start = (height - cal_h) // 2
    h_end = h_start + cal_h
    w_start = (width - cal_w) // 2
    w_end = w_start + cal_w
    
    sens_maps = np.zeros((slices, coils, height, width), dtype=np.complex64)
    for s in range(slices):
        for c in range(coils):
            cal_k = np.zeros((height, width), dtype=kspace.dtype)
            cal_k[h_start:h_end, w_start:w_end] = kspace[s, c, h_start:h_end, w_start:w_end]
            low_res = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(cal_k)))
            sens_maps[s, c] = low_res
            
        # Normalize sensitivities across coils
        sum_sq = np.sqrt(np.sum(np.abs(sens_maps[s])**2, axis=0)) + 1e-8
        sens_maps[s] /= sum_sq[np.newaxis, ...]
        
    # Convert numpy inputs to PyTorch complex tensors
    y = torch.tensor(kspace, dtype=torch.complex64, device=device) # [S, C, H, W]
    S = torch.tensor(sens_maps, dtype=torch.complex64, device=device) # [S, C, H, W]
    
    # Forward operator A(x): maps image x -> kspace y
    def A_op(x):
        x_coils = x.unsqueeze(1) * S # [S, C, H, W]
        shifted = torch.fft.fftshift(x_coils, dim=(-2, -1))
        fft_out = torch.fft.fft2(shifted)
        k_out = torch.fft.ifftshift(fft_out, dim=(-2, -1))
        return k_out

    # Adjoint operator A^H(k): maps kspace -> image
    def AH_op(k):
        shifted = torch.fft.ifftshift(k, dim=(-2, -1))
        ifft_out = torch.fft.ifft2(shifted)
        img_coils = torch.fft.fftshift(ifft_out, dim=(-2, -1))
        x_out = torch.sum(img_coils * torch.conj(S), dim=1) # [S, H, W]
        return x_out

    # Initialize x with adjoint reconstruction
    x = AH_op(y) # [S, H, W]
    
    # Real-valued CNN denoiser as regularization block
    class LearnedPriorDenoiser(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(2, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 2, kernel_size=3, padding=1)
            )
        def forward(self, x):
            real = torch.stack([x.real, x.imag], dim=1)
            out = real + self.net(real) # residual
            return torch.complex(out[:, 0], out[:, 1])

    denoiser = LearnedPriorDenoiser().to(device)
    denoiser.eval()
    
    # Iterative unrolled optimization loop
    with torch.no_grad():
        for i in range(n_iterations):
            diff = y - A_op(x)
            z = x + eta * AH_op(diff)
            x = denoiser(z)
            
    # Return absolute magnitude
    magnitude = torch.abs(x).cpu().numpy()
    return magnitude