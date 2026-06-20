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
    
    Steps:
      1. (Optional) Apply 1D line-by-line phase correction in K-space.
      2. Apply 2D IFFT slice-by-slice, coil-by-coil to compute coil images.
      3. (Optional) Align coil phases using low-resolution calibration.
      4. Apply Root Sum of Squares (RSS) coil combination.
      
    Parameters:
        kspace (np.ndarray): Complex K-space array of shape [slices, coils, height, width] 
                             or [coils, height, width].
        phase_correction (bool): Whether to perform phase correction steps.
        
    Returns:
        np.ndarray: Combined magnitude image of shape [slices, height, width] 
                    (or [height, width] if input had no slice dimension).
    """
    # Check dimensions and normalize to 4D [slices, coils, height, width]
    original_ndim = kspace.ndim
    if original_ndim == 3:
        kspace = kspace[np.newaxis, ...]
    elif original_ndim == 2:
        kspace = kspace[np.newaxis, np.newaxis, ...]
    elif original_ndim != 4:
        raise ValueError(f"Invalid input shape {kspace.shape}. Expected 2D, 3D, or 4D array.")
        
    # 1. Apply line-by-line phase correction in K-space
    if phase_correction:
        kspace = correct_line_phases(kspace)
        
    slices, coils, height, width = kspace.shape
    coil_images = np.zeros_like(kspace, dtype=np.complex128)
    
    # 2. Apply 2D IFFT slice-by-slice, coil-by-coil
    for s in range(slices):
        for c in range(coils):
            # Centering: input ifftshift, ifft2, output fftshift
            shifted_k = np.fft.ifftshift(kspace[s, c, :, :])
            img_c = np.fft.ifft2(shifted_k)
            coil_images[s, c, :, :] = np.fft.fftshift(img_c)
            
    # 3. Apply coil phase alignment
    if phase_correction:
        coil_images = align_coil_phases(coil_images, kspace)
        
    # 4. Root Sum of Squares (RSS) combination
    # Combined = sqrt( sum( |coil_images|^2, axis=coil_axis ) )
    # coil_axis is 1 (second dimension in [slices, coils, height, width])
    combined = np.sqrt(np.sum(np.abs(coil_images)**2, axis=1))
    
    # Squeeze output to match input dimensions
    if original_ndim == 3:
        combined = np.squeeze(combined, axis=0)  # [height, width]
    elif original_ndim == 2:
        combined = np.squeeze(combined)  # [height, width]
        
    return combined