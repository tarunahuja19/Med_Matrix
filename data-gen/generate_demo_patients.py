#!/usr/bin/env python3
"""
generate_demo_patients.py

Generates a demo_patients.zip containing 10 patient sub-zips.
Each patient zip has:
  - k_space.npy  : synthetic k-space array (256x256 complex float32)
                   + appended sentinel + PNG bytes of the reconstructed MRI image
  - cest.npy     : CEST Z-spectrum array (20 offsets x 64 x 64 float32)
                   + appended sentinel + JSON bytes of disease info + DnCNN noise-reduction result

The appended format is:
  <raw .npy bytes>  ||  MAGIC_SENTINEL (8 bytes)  ||  payload_length (4 bytes uint32 LE)  ||  <payload bytes>

This lets us load the .npy file normally and then seek to EOF - 12 - payload_length
to read the hidden payload.

Patient disease plan:
  patient01  Normal (healthy, no noise)
  patient02  Low-grade Glioma (no noise)
  patient03  High-grade Glioma (Rician noise SNR~25dB)
  patient04  Necrosis (no noise)
  patient05  Edema (Rician noise SNR~20dB)
  patient06  Meningioma (no noise)
  patient07  Microbleed (Rician noise SNR~30dB)
  patient08  Low-grade Glioma + Edema (Rician noise SNR~22dB)
  patient09  High-grade Glioma (no noise)
  patient10  Normal (Rician noise SNR~18dB)
"""

import io
import json
import struct
import zipfile
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# ── Constants ────────────────────────────────────────────────────────────────

MAGIC_SENTINEL = b'\xDE\xAD\xBE\xEF\xCA\xFE\xBA\xBE'  # 8 bytes
IMG_SIZE   = 256   # k-space / image size
CEST_H     = 64    # CEST map spatial resolution
CEST_W     = 64
N_OFFSETS  = 20    # CEST frequency offsets
GAMMA_HZ_T = 42.58e6
B0_TESLA   = 3.0
LARMOR_HZ  = GAMMA_HZ_T * B0_TESLA  # ~127.74 MHz

CEST_OFFSETS_PPM = np.array([
    -5.0, -4.0, -3.5, -3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0,
     2.0,  2.5,  3.0,  3.25, 3.5,  3.75, 4.0,  4.25, 4.5, 5.0
], dtype=np.float32)

# ── Disease Definitions ───────────────────────────────────────────────────────

PATIENTS = [
    {
        "id": "patient01",
        "disease": "Normal",
        "disease_code": 0,
        "noise_snr_db": None,   # No noise
        "description": "Healthy brain with no pathology. Normal white matter, gray matter, and CSF signal distribution.",
        # CEST physics params (3T, WM dominant)
        "ksw_amide": 30.0,   "fs_amide": 0.001,
        "ksw_amine": 200.0,  "fs_amine": 0.0006,
        "T1w_ms": 840.0,     "T2w_ms": 75.0,
        "T1s_ms": 700.0,     "T2s_ms": 15.0,
        "b1_uT": 0.8,        "b0_shift": 0.0,
        # MRI phantom params
        "tumor_T1_ms": None,  "tumor_T2_ms": None,
        "apt_pct": 1.2,  # Normal APT@3.5ppm ~1-1.5%
        "dncnn_noise_level": 0.0,
        "dncnn_denoise_gain_db": 0.0,
    },
    {
        "id": "patient02",
        "disease": "Low-grade Glioma",
        "disease_code": 1,
        "noise_snr_db": None,
        "description": "WHO Grade II glioma. Mildly elevated T1/T2 relaxation times. Moderate APT signal elevation (2.5-3.5%) due to increased mobile protein content.",
        "ksw_amide": 65.0,   "fs_amide": 0.004,
        "ksw_amine": 350.0,  "fs_amine": 0.0012,
        "T1w_ms": 1800.0,    "T2w_ms": 130.0,
        "T1s_ms": 900.0,     "T2s_ms": 18.0,
        "b1_uT": 0.8,        "b0_shift": 0.03,
        "tumor_T1_ms": 1850.0, "tumor_T2_ms": 130.0,
        "apt_pct": 3.1,
        "dncnn_noise_level": 0.0,
        "dncnn_denoise_gain_db": 0.0,
    },
    {
        "id": "patient03",
        "disease": "High-grade Glioma",
        "disease_code": 2,
        "noise_snr_db": 25.0,
        "description": "WHO Grade IV glioblastoma multiforme (GBM). Strongly elevated T1/T2. High APT signal (4-6%) from rapid cellular proliferation and elevated mobile protein/peptide content.",
        "ksw_amide": 150.0,  "fs_amide": 0.012,
        "ksw_amine": 900.0,  "fs_amine": 0.004,
        "T1w_ms": 2600.0,    "T2w_ms": 200.0,
        "T1s_ms": 1100.0,    "T2s_ms": 20.0,
        "b1_uT": 0.8,        "b0_shift": 0.05,
        "tumor_T1_ms": 2600.0, "tumor_T2_ms": 200.0,
        "apt_pct": 5.2,
        "dncnn_noise_level": 0.042,   # σ of additive Gaussian noise added before denoising
        "dncnn_denoise_gain_db": 6.8, # PSNR improvement from DnCNN
    },
    {
        "id": "patient04",
        "disease": "Necrosis",
        "disease_code": 3,
        "noise_snr_db": None,
        "description": "Central necrotic core (radiation necrosis / GBM necrosis). Fluid-like very long T1/T2. Low APT due to breakdown of proteins and absence of living cells.",
        "ksw_amide": 18.0,   "fs_amide": 0.001,
        "ksw_amine": 120.0,  "fs_amine": 0.0003,
        "T1w_ms": 3500.0,    "T2w_ms": 350.0,
        "T1s_ms": 1500.0,    "T2s_ms": 25.0,
        "b1_uT": 0.8,        "b0_shift": -0.02,
        "tumor_T1_ms": 3500.0, "tumor_T2_ms": 350.0,
        "apt_pct": 0.5,   # Necrosis = very low APT
        "dncnn_noise_level": 0.0,
        "dncnn_denoise_gain_db": 0.0,
    },
    {
        "id": "patient05",
        "disease": "Edema",
        "disease_code": 4,
        "noise_snr_db": 20.0,
        "description": "Vasogenic peritumoral edema. Mildly elevated T1/T2 (water infiltration into extracellular space). Low APT signal – diluted amide proton concentration.",
        "ksw_amide": 30.0,   "fs_amide": 0.0015,
        "ksw_amine": 180.0,  "fs_amine": 0.0005,
        "T1w_ms": 1700.0,    "T2w_ms": 145.0,
        "T1s_ms": 850.0,     "T2s_ms": 16.0,
        "b1_uT": 0.8,        "b0_shift": 0.01,
        "tumor_T1_ms": 1700.0, "tumor_T2_ms": 145.0,
        "apt_pct": 1.8,
        "dncnn_noise_level": 0.068,
        "dncnn_denoise_gain_db": 8.3,
    },
    {
        "id": "patient06",
        "disease": "Meningioma",
        "disease_code": 5,
        "noise_snr_db": None,
        "description": "Benign WHO Grade I meningioma (peripheral, dural attachment). Isointense-to-mildly elevated T1/T2. Moderate CEST effect from dense fibrous protein matrix.",
        "ksw_amide": 90.0,   "fs_amide": 0.005,
        "ksw_amine": 500.0,  "fs_amine": 0.0018,
        "T1w_ms": 1500.0,    "T2w_ms": 110.0,
        "T1s_ms": 780.0,     "T2s_ms": 17.0,
        "b1_uT": 0.8,        "b0_shift": 0.02,
        "tumor_T1_ms": 1500.0, "tumor_T2_ms": 110.0,
        "apt_pct": 2.8,
        "dncnn_noise_level": 0.0,
        "dncnn_denoise_gain_db": 0.0,
    },
    {
        "id": "patient07",
        "disease": "Microbleed",
        "disease_code": 6,
        "noise_snr_db": 30.0,
        "description": "Cerebral microbleed (hemosiderin deposit). T2* hypointense focus, strong diamagnetic susceptibility. Normal CEST background; local field inhomogeneity causes B0 offset artifacts.",
        "ksw_amide": 30.0,   "fs_amide": 0.001,
        "ksw_amine": 200.0,  "fs_amine": 0.0006,
        "T1w_ms": 840.0,     "T2w_ms": 75.0,
        "T1s_ms": 700.0,     "T2s_ms": 15.0,
        "b1_uT": 0.8,        "b0_shift": 0.15,  # B0 offset from iron
        "tumor_T1_ms": 600.0, "tumor_T2_ms": 20.0,
        "apt_pct": 1.1,   # Normal CEST, but B0-distorted
        "dncnn_noise_level": 0.025,
        "dncnn_denoise_gain_db": 4.2,
    },
    {
        "id": "patient08",
        "disease": "Low-grade Glioma + Edema",
        "disease_code": 1,
        "noise_snr_db": 22.0,
        "description": "Low-grade glioma with surrounding vasogenic edema. Mixed CEST signature: moderately elevated APT in tumor core (2.5-3%), diluted in edema rim (1.5-2%). High uncertainty at tumor-edema boundary.",
        "ksw_amide": 60.0,   "fs_amide": 0.0035,
        "ksw_amine": 320.0,  "fs_amine": 0.001,
        "T1w_ms": 1750.0,    "T2w_ms": 135.0,
        "T1s_ms": 880.0,     "T2s_ms": 17.0,
        "b1_uT": 0.8,        "b0_shift": 0.04,
        "tumor_T1_ms": 1750.0, "tumor_T2_ms": 140.0,
        "apt_pct": 2.6,
        "dncnn_noise_level": 0.055,
        "dncnn_denoise_gain_db": 7.1,
    },
    {
        "id": "patient09",
        "disease": "High-grade Glioma",
        "disease_code": 2,
        "noise_snr_db": None,
        "description": "Second GBM case — clean acquisition. Very high APT signal (5-7%), heterogeneous necrotic core visible. Ideal case for demonstrating BM-NODE-UQ parameter mapping without noise confounding.",
        "ksw_amide": 170.0,  "fs_amide": 0.014,
        "ksw_amine": 950.0,  "fs_amine": 0.0045,
        "T1w_ms": 2800.0,    "T2w_ms": 220.0,
        "T1s_ms": 1150.0,    "T2s_ms": 22.0,
        "b1_uT": 0.8,        "b0_shift": 0.06,
        "tumor_T1_ms": 2800.0, "tumor_T2_ms": 220.0,
        "apt_pct": 5.8,
        "dncnn_noise_level": 0.0,
        "dncnn_denoise_gain_db": 0.0,
    },
    {
        "id": "patient10",
        "disease": "Normal",
        "disease_code": 0,
        "noise_snr_db": 18.0,
        "description": "Healthy volunteer with heavy thermal noise (low SNR acquisition). DnCNN denoising recovers clean image. Normal CEST signature throughout. Demonstrates model robustness to noisy acquisitions.",
        "ksw_amide": 32.0,   "fs_amide": 0.0011,
        "ksw_amine": 210.0,  "fs_amine": 0.00065,
        "T1w_ms": 850.0,     "T2w_ms": 78.0,
        "T1s_ms": 710.0,     "T2s_ms": 15.0,
        "b1_uT": 0.8,        "b0_shift": 0.0,
        "tumor_T1_ms": None,  "tumor_T2_ms": None,
        "apt_pct": 1.3,
        "dncnn_noise_level": 0.090,
        "dncnn_denoise_gain_db": 9.5,
    },
]

# ── Helper: Append hidden payload to .npy bytes ───────────────────────────────

def embed_payload(npy_bytes: bytes, payload: bytes) -> bytes:
    """
    Append MAGIC_SENTINEL + uint32 payload length + payload to raw .npy bytes.
    The .npy file can still be loaded with np.load() on the leading bytes.
    """
    length_bytes = struct.pack('<I', len(payload))
    return npy_bytes + MAGIC_SENTINEL + length_bytes + payload


def save_npy_to_bytes(array: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, array)
    return buf.getvalue()


# ── Helper: Generate synthetic brain k-space ─────────────────────────────────

def make_brain_phantom(size: int, p: dict, rng: np.random.Generator) -> np.ndarray:
    """
    Generate a synthetic brain MRI phantom as a 2D float image [0,1].
    Uses ellipsoidal tissue compartments with disease-specific signal changes.
    """
    img = np.zeros((size, size), dtype=np.float32)
    cx, cy = size // 2, size // 2

    # Coordinate grids
    yy, xx = np.ogrid[:size, :size]
    dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)

    # ── Skull (low signal ring) ──
    skull_outer = size * 0.47
    skull_inner = size * 0.43
    skull_mask = (dist > skull_inner) & (dist <= skull_outer)
    img[skull_mask] = 0.15

    # ── CSF / ventricles (bright in PD/T2) ──
    brain_r = size * 0.43
    brain_mask = dist <= brain_r

    # White matter (inner ~70%)
    wm_r = brain_r * 0.70
    wm_mask = dist <= wm_r
    img[wm_mask] = 0.65  # WM slightly darker

    # Gray matter ring
    gm_mask = brain_mask & ~wm_mask
    img[gm_mask] = 0.82  # GM brighter

    # Ventricles (small ellipse near center)
    vent_mask = ((xx - cx + size * 0.07)**2 / (size * 0.06)**2 +
                 (yy - cy)**2 / (size * 0.10)**2) <= 1.0
    vent_mask &= brain_mask
    img[vent_mask] = 0.10  # CSF = dark on T1w

    # ── Tumor / pathology ──
    if p.get("tumor_T1_ms") is not None:
        # Tumor positioned in WM, off-center
        t1_ms = p["tumor_T1_ms"]
        t2_ms = p["tumor_T2_ms"]

        # Tumor signal (longer T1→ darker, longer T2→ brighter on T2w but darker on T1w)
        # Simplified: T1w signal ∝ exp(-TR/T1) * (1 - exp(-TR/T1))
        TR_ms = 2500.0; TE_ms = 85.0
        tumor_signal = (1 - np.exp(-TR_ms / t1_ms)) * np.exp(-TE_ms / t2_ms)
        tumor_signal = float(np.clip(tumor_signal, 0.0, 1.0))

        d = p["disease_code"]
        # Place tumor offset from center
        tx = cx + int(size * 0.14)
        ty = cy - int(size * 0.10)

        if d == 2:  # HGG: compact, realistic lesion
            t_r = size * 0.065
        elif d == 3:  # Necrosis: compact necrotic core
            t_r = size * 0.055
        elif d == 6:  # Microbleed: very small
            t_r = size * 0.025
        else:
            t_r = size * 0.090

        if d == 6:
            t_r_y = t_r
        else:
            t_r_y = t_r * rng.uniform(0.85, 1.15)

        tumor_mask = ((xx - tx)**2 / t_r**2 + (yy - ty)**2 / t_r_y**2) <= 1.0
        tumor_mask &= brain_mask

        if d == 6:  # Microbleed: very dark (T2* blooming)
            img[tumor_mask] = 0.02
        elif d in (3, 4):  # Necrosis / edema: bright rim
            img[tumor_mask] = min(1.0, tumor_signal * 1.4)
        else:
            img[tumor_mask] = tumor_signal

        # Perilesional edema ring for gliomas + edema
        if d in (1, 2, 4, 8):
            edema_mask = ((xx - tx)**2 / (t_r * 1.6)**2 + (yy - ty)**2 / (t_r_y * 1.6)**2) <= 1.0
            edema_mask &= brain_mask & ~tumor_mask
            img[edema_mask] = np.minimum(1.0, img[edema_mask] * 1.15)

    # ── Smooth with slight Gaussian blur (physical T2* blur) ──
    # Simple box-blur approximation (avoids scipy dependency)
    from numpy.fft import fft2, ifft2, fftfreq
    fy = fftfreq(size).reshape(-1, 1)
    fx = fftfreq(size).reshape(1, -1)
    sigma_pix = 1.2
    gauss_filt = np.exp(-2 * np.pi**2 * sigma_pix**2 * (fx**2 + fy**2))
    img_f = fft2(img.astype(np.complex64))
    img = np.abs(ifft2(img_f * gauss_filt)).astype(np.float32)
    img = np.clip(img, 0.0, 1.0)

    return img


def make_kspace(img: np.ndarray, snr_db: float | None, rng: np.random.Generator) -> np.ndarray:
    """
    Compute complex k-space from image via 2D FFT.
    Optionally add Rician noise at given SNR.
    Returns complex64 array (2, H, W) with [real, imag] stored separately,
    but we save as complex array of shape (H, W).
    """
    kspace = np.fft.fftshift(np.fft.fft2(img.astype(np.complex64)))

    if snr_db is not None:
        signal_power = np.mean(np.abs(kspace)**2)
        noise_power = signal_power / (10 ** (snr_db / 10.0))
        sigma = np.sqrt(noise_power / 2)
        noise_r = rng.normal(0, sigma, kspace.shape).astype(np.float32)
        noise_i = rng.normal(0, sigma, kspace.shape).astype(np.float32)
        kspace = kspace + (noise_r + 1j * noise_i).astype(np.complex64)

    return kspace.astype(np.complex64)


def reconstruct_image(kspace: np.ndarray) -> np.ndarray:
    """IFFT to reconstruct magnitude image from k-space."""
    img = np.abs(np.fft.ifft2(np.fft.ifftshift(kspace)))
    img = img - img.min()
    if img.max() > 0:
        img = img / img.max()
    return img.astype(np.float32)


def image_to_png_bytes(img: np.ndarray) -> bytes:
    """
    Convert a float32 [0,1] image to PNG bytes using pure numpy (no PIL dependency).
    Uses a minimal uncompressed PNG writer.
    """
    import zlib, struct as st

    h, w = img.shape
    img8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)

    def png_chunk(chunk_type: bytes, data: bytes) -> bytes:
        c = chunk_type + data
        crc = zlib.crc32(c) & 0xFFFFFFFF
        return st.pack('>I', len(data)) + c + st.pack('>I', crc)

    # IHDR
    ihdr_data = st.pack('>IIBBBBB', w, h, 8, 0, 0, 0, 0)  # 8-bit grayscale
    ihdr = png_chunk(b'IHDR', ihdr_data)

    # IDAT: scanline filter byte (0=None) prepended to each row
    raw_rows = b''.join(b'\x00' + bytes(img8[r]) for r in range(h))
    compressed = zlib.compress(raw_rows, level=6)
    idat = png_chunk(b'IDAT', compressed)

    iend = png_chunk(b'IEND', b'')

    return b'\x89PNG\r\n\x1a\n' + ihdr + idat + iend


# ── CEST Z-spectrum generation ────────────────────────────────────────────────

def compute_z_spectrum_voxel(
    offset_ppm: float,
    T1w_ms: float, T2w_ms: float,
    ksw: float, fs: float,
    b1_uT: float, b0_shift: float,
    solute_ppm: float = 3.5,
    T2s_ms: float = 15.0,
) -> float:
    """
    2-pool Bloch-McConnell steady-state Z-spectrum (Zaiss 2013).
    """
    dw = offset_ppm - b0_shift
    w1 = 2 * np.pi * GAMMA_HZ_T * b1_uT * 1e-6  # rad/s
    DC = 0.5
    w1_eff = w1 * np.sqrt(DC)

    T1w = T1w_ms * 1e-3
    T2w = T2w_ms * 1e-3
    R1w = 1.0 / T1w
    R2s = 1.0 / (T2s_ms * 1e-3)

    dw_water = 2 * np.pi * dw * 1e-6 * LARMOR_HZ
    dwT2w = dw_water * T2w
    Z_water = 1.0 / (1.0 + (w1_eff**2 * T1w * T2w) / (1.0 + dwT2w**2))

    # Amide Rex (Zaiss 2013 Eq. 13 with power-broadening)
    dw_amide = 2 * np.pi * (dw - solute_ppm) * 1e-6 * LARMOR_HZ
    pb = (R2s + ksw) * (R2s + ksw + w1_eff**2 / R2s) + dw_amide**2
    Rex = (fs * ksw * w1_eff**2) / pb if pb > 0 else 0.0
    totalRex = Rex

    # NOE pool for realism (-3.5 ppm)
    fs_NOE, k_NOE = 0.0025, 12.0
    dw_NOE = 2 * np.pi * (dw - (-3.5)) * 1e-6 * LARMOR_HZ
    pb_NOE = (R2s + k_NOE) * (R2s + k_NOE + w1_eff**2 / R2s) + dw_NOE**2
    totalRex += (fs_NOE * k_NOE * w1_eff**2) / pb_NOE if pb_NOE > 0 else 0.0

    Z = Z_water / (1.0 + totalRex / R1w)
    return float(np.clip(Z, 0.02, 1.0))


def make_cest_maps(p: dict, rng: np.random.Generator) -> np.ndarray:
    """
    Generate CEST Z-spectrum stack (N_offsets, H, W) with spatially varying
    parameters. Returns float32 array.
    """
    H, W = CEST_H, CEST_W
    cx, cy = W // 2, H // 2
    brain_r = H * 0.44
    tumor_r = H * 0.14

    # Pixel distance map
    yy, xx = np.ogrid[:H, :W]
    dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    brain_mask = dist <= brain_r

    # Tumor region (offset like MRI phantom)
    tx = int(cx + W * 0.15)
    ty = int(cy - H * 0.08)
    tumor_dist = np.sqrt((xx - tx)**2 + (yy - ty)**2)
    tumor_mask = (tumor_dist <= tumor_r) & brain_mask

    # Build parameter maps
    ksw_map  = np.zeros((H, W), dtype=np.float32)
    fs_map   = np.zeros((H, W), dtype=np.float32)
    T1w_map  = np.zeros((H, W), dtype=np.float32)
    T2w_map  = np.zeros((H, W), dtype=np.float32)

    # Background brain tissue
    ksw_map[brain_mask]  = p["ksw_amide"] * 0.65
    fs_map[brain_mask]   = p["fs_amide"]  * 0.65
    T1w_map[brain_mask]  = p["T1w_ms"]
    T2w_map[brain_mask]  = p["T2w_ms"]

    # Tumor region
    if p.get("tumor_T1_ms") is not None:
        ksw_map[tumor_mask]  = p["ksw_amide"]
        fs_map[tumor_mask]   = p["fs_amide"]
        T1w_map[tumor_mask]  = p["tumor_T1_ms"]
        T2w_map[tumor_mask]  = p["tumor_T2_ms"]

    # Add smooth spatial inhomogeneity (B1/B0 field variation)
    xg = np.arange(W) / W
    yg = np.arange(H).reshape(-1, 1) / H
    inhomog = (1.0
               + 0.04 * np.cos(xg * 2 * np.pi)
               + 0.03 * np.sin(yg * 2 * np.pi)
               + 0.02 * np.cos((xg + yg) * 3 * np.pi))
    ksw_map  *= inhomog
    fs_map   *= inhomog

    # Per-voxel noise
    snr_noise = 0.05  # 5% spatial noise
    ksw_map  += rng.normal(0, p["ksw_amide"] * snr_noise, (H, W)).astype(np.float32)
    fs_map   += rng.normal(0, p["fs_amide"]  * snr_noise, (H, W)).astype(np.float32)
    ksw_map   = np.clip(ksw_map, 0, None)
    fs_map    = np.clip(fs_map,  0, None)

    # Compute Z-spectrum per voxel
    z_stack = np.zeros((N_OFFSETS, H, W), dtype=np.float32)
    for i, offset in enumerate(CEST_OFFSETS_PPM):
        for r in range(H):
            for c in range(W):
                if not brain_mask[r, c]:
                    continue
                z_stack[i, r, c] = compute_z_spectrum_voxel(
                    offset_ppm=float(offset),
                    T1w_ms=float(T1w_map[r, c]),
                    T2w_ms=float(T2w_map[r, c]),
                    ksw=float(ksw_map[r, c]),
                    fs=float(fs_map[r, c]),
                    b1_uT=p["b1_uT"],
                    b0_shift=p["b0_shift"],
                    T2s_ms=p["T1s_ms"] * 0.02,  # approximate T2s
                )

    return z_stack


# ── Disease info payload ──────────────────────────────────────────────────────

def make_disease_payload(p: dict) -> bytes:
    """Build JSON payload with disease info + DnCNN denoising results."""
    has_noise = p["noise_snr_db"] is not None
    dncnn_result = {
        "applied": has_noise,
        "noise_sigma_estimated": round(p["dncnn_noise_level"], 4),
        "noise_type": "Rician" if has_noise else "None",
        "input_snr_db": p["noise_snr_db"],
        "output_snr_db": (
            round(p["noise_snr_db"] + p["dncnn_denoise_gain_db"], 1)
            if has_noise else None
        ),
        "psnr_improvement_db": p["dncnn_denoise_gain_db"],
        "model": "DnCNN-B (blind denoising, 20 layers, 3T brain MRI fine-tuned)",
        "inference_ms": round(12.4 + p["dncnn_noise_level"] * 80, 1),
        "residual_map_available": has_noise,
    }

    apt_val = p["apt_pct"]
    # MTRasym @3.5 ppm
    z_neg35 = compute_z_spectrum_voxel(
        -3.5, p["T1w_ms"], p["T2w_ms"],
        p["ksw_amide"], p["fs_amide"],
        p["b1_uT"], p["b0_shift"]
    )
    z_pos35 = compute_z_spectrum_voxel(
        +3.5, p["T1w_ms"], p["T2w_ms"],
        p["ksw_amide"], p["fs_amide"],
        p["b1_uT"], p["b0_shift"]
    )
    mtr_asym_35 = round(max(0, (z_neg35 - z_pos35)) * 100, 2)

    payload = {
        "patient_id": p["id"],
        "disease": p["disease"],
        "disease_code": p["disease_code"],
        "description": p["description"],
        "cest_params": {
            "ksw_amide_hz": p["ksw_amide"],
            "fs_amide": p["fs_amide"],
            "ksw_amine_hz": p["ksw_amine"],
            "fs_amine": p["fs_amine"],
            "T1w_ms": p["T1w_ms"],
            "T2w_ms": p["T2w_ms"],
            "T1s_ms": p["T1s_ms"],
            "T2s_ms": p["T2s_ms"],
            "b1_uT": p["b1_uT"],
            "b0_shift_ppm": p["b0_shift"],
        },
        "biomarkers": {
            "APT_percent": apt_val,
            "MTRasym_at_3p5ppm_percent": mtr_asym_35,
            "NOE_percent": round(1.2 + p["ksw_amide"] * 0.002, 2),
            "T1w_wm_ms": round(p["T1w_ms"] * 0.85, 1),
            "T2w_wm_ms": round(p["T2w_ms"] * 0.90, 1),
        },
        "dncnn_denoising": dncnn_result,
        "acquisition": {
            "field_strength_T": 3.0,
            "scanner": "Siemens Prisma 3T (simulated)",
            "sequence": "Pulsed CEST + SE-EPI",
            "TR_ms": 2500.0,
            "TE_ms": 85.0,
            "B1_saturation_uT": p["b1_uT"],
            "n_offsets": N_OFFSETS,
            "offset_range_ppm": [-5.0, 5.0],
        },
        "bm_node_uq_inference": {
            "model_version": "v0.1-alpha-demo",
            "pool_model": "3-pool",
            "mc_passes": 20,
            "inference_ms": 18.4,
            "confidence_ksw": round(max(0.40, 0.92 - p["dncnn_noise_level"] * 2.0), 3),
            "confidence_fs":  round(max(0.35, 0.89 - p["dncnn_noise_level"] * 2.5), 3),
        }
    }
    return json.dumps(payload, indent=2).encode("utf-8")


# ── Main generation ───────────────────────────────────────────────────────────

def generate_patient_zip(p: dict, rng: np.random.Generator) -> bytes:
    """Build a single patient zip in memory and return its bytes."""
    print(f"  Generating {p['id']}: {p['disease']} (noise SNR={p['noise_snr_db']} dB)")

    # 1) MRI phantom + k-space
    phantom = make_brain_phantom(IMG_SIZE, p, rng)
    kspace  = make_kspace(phantom, p["noise_snr_db"], rng)
    recon_img = reconstruct_image(kspace)

    # k_space.npy: save complex array, then embed PNG of reconstructed image
    kspace_npy = save_npy_to_bytes(kspace)
    png_bytes   = image_to_png_bytes(recon_img)
    kspace_embedded = embed_payload(kspace_npy, png_bytes)

    # 2) CEST data
    print(f"    Computing CEST Z-spectrum maps ({CEST_H}x{CEST_W} x {N_OFFSETS} offsets)...")
    cest_stack = make_cest_maps(p, rng)

    # cest.npy: save Z-spectrum, then embed disease info JSON
    cest_npy        = save_npy_to_bytes(cest_stack)
    disease_payload = make_disease_payload(p)
    cest_embedded   = embed_payload(cest_npy, disease_payload)

    # 3) Pack into zip
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        zf.writestr("k_space.npy", kspace_embedded)
        zf.writestr("cest.npy",    cest_embedded)
    return buf.getvalue()


def main():
    rng = np.random.default_rng(seed=42)

    out_dir = Path(__file__).parent.parent / "demo_data"
    out_dir.mkdir(parents=True, exist_ok=True)

    patients_dir = out_dir / "patients"
    patients_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MED-MATRIX Demo Patient Generator")
    print("=" * 60)
    print(f"Output: {patients_dir}")
    print()

    summary = []
    patient_zips = {}

    for p in PATIENTS:
        patient_zip_bytes = generate_patient_zip(p, rng)
        zip_name = f"{p['id']}.zip"
        zip_path = patients_dir / zip_name
        with open(zip_path, 'wb') as f:
            f.write(patient_zip_bytes)
        patient_zips[zip_name] = patient_zip_bytes
        summary.append({
            "id":      p["id"],
            "disease": p["disease"],
            "noise":   f"{p['noise_snr_db']} dB" if p["noise_snr_db"] else "None",
            "apt_pct": p["apt_pct"],
            "file_kb": round(len(patient_zip_bytes) / 1024, 1),
        })
        print(f"    ✓ Saved {zip_name} ({summary[-1]['file_kb']} KB)")

    # Master bundle zip
    bundle_path = out_dir / "demo_patients.zip"
    print()
    print(f"Creating master bundle: {bundle_path}")
    with zipfile.ZipFile(bundle_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for zip_name, data in patient_zips.items():
            zf.writestr(f"patients/{zip_name}", data)
    print(f"  ✓ Master bundle: {bundle_path} ({bundle_path.stat().st_size // 1024} KB)")

    # Print summary table
    print()
    print("=" * 60)
    print("PATIENT SUMMARY")
    print("=" * 60)
    print(f"{'ID':<12} {'Disease':<30} {'Noise SNR':<12} {'APT%':<8} {'Size'}")
    print("-" * 72)
    for s in summary:
        print(f"{s['id']:<12} {s['disease']:<30} {s['noise']:<12} {s['apt_pct']:<8} {s['file_kb']} KB")

    print()
    print("File format (appended bytes):")
    print("  k_space.npy : complex64 (256x256) k-space + [PNG of reconstructed MRI]")
    print("  cest.npy    : float32 (20x64x64) Z-spectrum + [JSON disease info + DnCNN results]")
    print()
    print("To read hidden payload:")
    print("  data = open('k_space.npy','rb').read()")
    print("  sentinel_pos = data.rfind(MAGIC_SENTINEL)")
    print("  length = struct.unpack('<I', data[sentinel_pos+8:sentinel_pos+12])[0]")
    print("  payload = data[sentinel_pos+12:sentinel_pos+12+length]")
    print()
    print("Done!")


if __name__ == "__main__":
    main()
