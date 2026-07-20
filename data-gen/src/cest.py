import numpy as np
from typing import Tuple, List

# 20 offsets spanning -5 to +5 ppm, fine spacing around +3.5 ppm (APT peak)
CEST_OFFSETS_PPM = np.array([
    -5.0, -4.0, -3.5, -3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0,
    2.0, 2.5, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 5.0
], dtype=np.float32)

def compute_z_spectrum_slice(
    T1_map: np.ndarray,          # in ms
    T2_map: np.ndarray,          # in ms
    k_sw_map: np.ndarray,        # Exchange rate in Hz
    f_s_map: np.ndarray,         # Solute concentration fraction
    offsets_ppm: np.ndarray = None,
    B1_uT: float = 1.5,          # Saturation B1 power in microTesla
    B0_Tesla: float = 3.0,
    solute_shift_ppm: float = 3.5  # APT amide proton resonance at 3.5 ppm
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes 2-pool steady-state Bloch-McConnell Z-spectra across saturation offsets.
    
    Returns:
        - Z_spectrum stack shape (N_offsets, H, W) float32
        - exchange_rate_map (H, W) float32
        - concentration_map (H, W) float32
    """
    if offsets_ppm is None:
        offsets_ppm = CEST_OFFSETS_PPM
        
    H, W = T1_map.shape
    N_offsets = len(offsets_ppm)
    
    z_spectrum = np.zeros((N_offsets, H, W), dtype=np.float32)
    
    valid_mask = (T1_map > 0) & (T2_map > 0)
    
    # Constants
    gamma_rad = 2.0 * np.pi * 42.58e6  # rad / (s * Tesla)
    B1_Tesla = B1_uT * 1e-6
    w1 = gamma_rad * B1_Tesla           # Nutation frequency rad/s
    freq_3T = 42.58e6 * B0_Tesla        # Water Larmor frequency ~ 127.74 MHz
    
    T1_sec = T1_map / 1000.0
    T2_sec = T2_map / 1000.0
    
    for idx, d_ppm in enumerate(offsets_ppm):
        d_rad = 2.0 * np.pi * d_ppm * 1e-6 * freq_3T
        
        # Steady state water saturation: Z_water = 1 / (1 + (w1 * T1 * T2)^2 / (1 + (d_rad * T2)^2))
        w1_t1_t2 = (w1 * T1_sec * T2_sec)
        d_t2_sq = (d_rad * T2_sec)**2
        
        Z_water = np.zeros((H, W), dtype=np.float32)
        Z_water[valid_mask] = 1.0 / (1.0 + (w1_t1_t2[valid_mask]**2) / (1.0 + d_t2_sq[valid_mask]))
        
        # CEST exchange attenuation near +3.5 ppm
        gamma_solute_ppm = (k_sw_map + 20.0) / (np.pi * 127.74)  # Width in ppm
        gamma_solute_ppm = np.maximum(gamma_solute_ppm, 0.4)
        
        cest_lorentzian = 1.0 / (1.0 + ((d_ppm - solute_shift_ppm) / (gamma_solute_ppm / 2.0))**2)
        
        # Exchange transfer effect proportional to f_s * k_sw * T1w
        cest_attenuation = f_s_map * (k_sw_map * T1_sec) * cest_lorentzian
        
        Z_slice = np.clip(Z_water - cest_attenuation, 0.05, 1.0)
        Z_slice[~valid_mask] = 0.0
        
        z_spectrum[idx] = Z_slice
        
    return z_spectrum, k_sw_map.astype(np.float32), f_s_map.astype(np.float32)
