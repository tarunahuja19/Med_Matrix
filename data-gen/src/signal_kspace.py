import numpy as np
from typing import Tuple

def generate_spin_echo_image(
    T1_map: np.ndarray,
    T2_map: np.ndarray,
    PD_map: np.ndarray,
    TR: float = 2500.0,  # Repetition time in ms
    TE: float = 85.0     # Echo time in ms
) -> np.ndarray:
    """
    Computes magnitude MR image using closed-form spin-echo signal equation:
    S = PD * (1 - exp(-TR / T1)) * exp(-TE / T2)
    """
    H, W = T1_map.shape
    signal = np.zeros((H, W), dtype=np.float32)
    
    valid_mask = (T1_map > 0) & (T2_map > 0) & (PD_map > 0)
    
    t1_term = 1.0 - np.exp(-TR / np.maximum(T1_map[valid_mask], 1.0))
    t2_term = np.exp(-TE / np.maximum(T2_map[valid_mask], 1.0))
    
    signal[valid_mask] = PD_map[valid_mask] * t1_term * t2_term
    
    # Normalize to [0, 1] range for U-Net input
    max_val = np.max(signal)
    if max_val > 0:
        signal = signal / max_val
        
    return signal.astype(np.float32)

def generate_2d_dipole_kernel(shape: Tuple[int, int], B0_dir: Tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
    """
    Generates a 2D dipole kernel in Fourier domain for susceptibility phase mapping.
    D(k) = 1/3 - (k . B0)^2 / |k|^2
    """
    H, W = shape
    ky = np.fft.fftfreq(H)
    kx = np.fft.fftfreq(W)
    KX, KY = np.meshgrid(kx, ky)
    
    K2 = KX**2 + KY**2
    K2[0, 0] = 1e-12  # Avoid division by zero at DC
    
    # B0 vector along Y axis (0, 1)
    K_dot_B0 = KY * B0_dir[1] + KX * B0_dir[0]
    kernel = (1.0 / 3.0) - (K_dot_B0**2 / K2)
    kernel[0, 0] = 0.0
    
    return kernel.astype(np.float32)

def compute_susceptibility_phase(
    chi_map: np.ndarray,  # in ppm
    TE_sec: float = 0.020,  # 20 ms echo time for gradient echo phase
    B0_Tesla: float = 3.0,
    gamma: float = 42.58e6  # Hz/Tesla
) -> np.ndarray:
    """
    Computes susceptibility-induced phase perturbation phi = 2*pi * gamma * B0 * dB * TE.
    """
    H, W = chi_map.shape
    
    # Dipole convolution via FFT: dB = IFFT2( FFT2(chi) * D_kernel )
    chi_fft = np.fft.fft2(chi_map * 1e-6)  # Convert ppm to absolute ratio
    D_kernel = generate_2d_dipole_kernel((H, W))
    dB_map = np.real(np.fft.ifft2(chi_fft * D_kernel))
    
    phase_map = 2.0 * np.pi * gamma * B0_Tesla * dB_map * TE_sec
    return phase_map.astype(np.float32)

def generate_kspace_from_image(
    magnitude_img: np.ndarray,
    phase_img: np.ndarray,
    target_snr_db: float = 25.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates 2-channel complex k-space (real, imag) from magnitude & phase image, plus Gaussian noise.
    
    Returns:
        kspace_channels: (2, H, W) float32 array where channel 0=real, channel 1=imag.
        reconstructed_magnitude: (H, W) magnitude image reconstructed from noisy k-space.
    """
    H, W = magnitude_img.shape
    
    # Build complex image space S * exp(i * phi)
    complex_image = magnitude_img * np.exp(1j * phase_img)
    
    # 2D FFT to k-space
    kspace_clean = np.fft.fftshift(np.fft.fft2(complex_image))
    
    # Add complex Gaussian noise
    signal_power = np.mean(np.abs(kspace_clean)**2)
    snr_linear = 10.0**(target_snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise_std = np.sqrt(noise_power / 2.0)
    
    noise_real = np.random.normal(0, noise_std, size=(H, W))
    noise_imag = np.random.normal(0, noise_std, size=(H, W))
    
    kspace_noisy = kspace_clean + (noise_real + 1j * noise_imag)
    
    # Reconstruct magnitude image via IFFT
    reconstructed_complex = np.fft.ifft2(np.fft.ifftshift(kspace_noisy))
    reconstructed_magnitude = np.abs(reconstructed_complex).astype(np.float32)
    
    # Format k-space as (2, H, W) real and imaginary channels
    kspace_channels = np.stack([np.real(kspace_noisy), np.imag(kspace_noisy)], axis=0).astype(np.float32)
    
    return kspace_channels, reconstructed_magnitude
