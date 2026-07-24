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

def reconstruct_kspace(kspace: np.ndarray, phase_correction: bool = True) -> np.ndarray:
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

    # 1. Apply line-by-line phase correction in K-space
    if phase_correction:
        kspace_pc = correct_line_phases(kspace)
    else:
        kspace_pc = kspace.copy()
        
    coil_images = np.zeros_like(kspace_pc, dtype=np.complex128)
    
    # 2. Apply 2D IFFT slice-by-slice, coil-by-coil
    for s in range(slices):
        for c in range(coils):
            # Centering: input ifftshift, ifft2, output fftshift
            shifted_k = np.fft.ifftshift(kspace_pc[s, c, :, :])
            img_c = np.fft.ifft2(shifted_k)
            coil_images[s, c, :, :] = np.fft.fftshift(img_c)
            
    # 3. Apply coil phase alignment
    if phase_correction:
        coil_images = align_coil_phases(coil_images, kspace_pc)
        
    # 4. Root Sum of Squares (RSS) combination
    combined = np.sqrt(np.sum(np.abs(coil_images)**2, axis=1))
    
    # Squeeze output to match input dimensions
    if original_ndim == 3:
        combined = np.squeeze(combined, axis=0)  # [height, width]
    elif original_ndim == 2:
        combined = np.squeeze(combined)  # [height, width]
        
    return combined


def compute_snr_cnr(
    img: np.ndarray,
    signal_roi: np.ndarray | None = None,
    noise_roi: np.ndarray | None = None,
    contrast_roi: np.ndarray | None = None,
) -> dict:
    """
    Computes Signal-to-Noise Ratio (SNR) and Contrast-to-Noise Ratio (CNR)
    from a reconstructed MRI magnitude image.

    If no ROI masks are provided the function uses automatic estimation:
      - Signal ROI : central 20% of the image (expected to contain brain tissue).
      - Noise ROI  : four 10%-wide corners of the image (expected background).
      - Contrast ROI: central 10% ring around the signal ROI (peri-lesional tissue).

    Parameters
    ----------
    img : np.ndarray
        2-D magnitude image [H, W] or 3-D volume [slices, H, W].  For 3-D
        input the middle slice is used for metric computation.
    signal_roi : np.ndarray | None
        Boolean mask the same spatial size as img (or the selected 2-D slice)
        indicating the primary signal region.
    noise_roi : np.ndarray | None
        Boolean mask for the background / noise region.
    contrast_roi : np.ndarray | None
        Boolean mask for a secondary tissue region used in CNR computation.

    Returns
    -------
    dict with keys:
        snr (float)           : Signal-to-Noise Ratio
        cnr (float)           : Contrast-to-Noise Ratio
        mean_signal (float)   : Mean pixel intensity of signal ROI
        std_noise (float)     : Standard deviation of noise ROI
        mean_contrast (float) : Mean pixel intensity of contrast ROI
        snr_quality (str)     : Qualitative label ("Excellent"/"Good"/"Fair"/"Poor")
        cnr_quality (str)     : Qualitative label ("Excellent"/"Good"/"Fair"/"Poor")
    """
    # Select the 2-D working slice
    if img.ndim == 3:
        working = img[img.shape[0] // 2].astype(np.float64)
    elif img.ndim == 2:
        working = img.astype(np.float64)
    else:
        raise ValueError(f"Expected 2-D or 3-D array, got shape {img.shape}")

    H, W = working.shape

    # ── Build default ROIs if none are supplied ─────────────────────────────
    if signal_roi is None:
        # Central 20% box — expected to cover brain parenchyma
        h0, h1 = int(H * 0.40), int(H * 0.60)
        w0, w1 = int(W * 0.40), int(W * 0.60)
        signal_roi = np.zeros((H, W), dtype=bool)
        signal_roi[h0:h1, w0:w1] = True

    if noise_roi is None:
        # Four 10%-wide corners — expected to be background / air
        corner_h = max(1, int(H * 0.10))
        corner_w = max(1, int(W * 0.10))
        noise_roi = np.zeros((H, W), dtype=bool)
        noise_roi[:corner_h, :corner_w] = True          # top-left
        noise_roi[:corner_h, -corner_w:] = True         # top-right
        noise_roi[-corner_h:, :corner_w] = True         # bottom-left
        noise_roi[-corner_h:, -corner_w:] = True        # bottom-right

    if contrast_roi is None:
        # Peri-central 10% annular ring (between 25% and 35% from centre)
        ch, cw = H // 2, W // 2
        contrast_roi = np.zeros((H, W), dtype=bool)
        inner_h0, inner_h1 = int(H * 0.35), int(H * 0.65)
        inner_w0, inner_w1 = int(W * 0.35), int(W * 0.65)
        outer_h0, outer_h1 = int(H * 0.25), int(H * 0.75)
        outer_w0, outer_w1 = int(W * 0.25), int(W * 0.75)
        contrast_roi[outer_h0:outer_h1, outer_w0:outer_w1] = True
        contrast_roi[inner_h0:inner_h1, inner_w0:inner_w1] = False  # punch hole

    signal_pixels = working[signal_roi]
    noise_pixels = working[noise_roi]
    contrast_pixels = working[contrast_roi]

    # Guard against empty ROIs
    mean_signal = float(np.mean(signal_pixels)) if signal_pixels.size > 0 else 0.0
    std_noise = float(np.std(noise_pixels)) if noise_pixels.size > 0 else 1e-9
    mean_contrast = float(np.mean(contrast_pixels)) if contrast_pixels.size > 0 else 0.0

    # Avoid divide-by-zero
    if std_noise < 1e-9:
        std_noise = 1e-9

    snr = mean_signal / std_noise
    cnr = abs(mean_signal - mean_contrast) / std_noise

    # Qualitative grading (clinical thresholds for brain MRI at 3T)
    def _snr_grade(v: float) -> str:
        if v >= 20:  return "Excellent"
        if v >= 10:  return "Good"
        if v >= 5:   return "Fair"
        return "Poor"

    def _cnr_grade(v: float) -> str:
        if v >= 5:   return "Excellent"
        if v >= 2.5: return "Good"
        if v >= 1:   return "Fair"
        return "Poor"

    return {
        "snr": round(snr, 2),
        "cnr": round(cnr, 2),
        "mean_signal": round(mean_signal, 4),
        "std_noise": round(std_noise, 4),
        "mean_contrast": round(mean_contrast, 4),
        "snr_quality": _snr_grade(snr),
        "cnr_quality": _cnr_grade(cnr),
    }