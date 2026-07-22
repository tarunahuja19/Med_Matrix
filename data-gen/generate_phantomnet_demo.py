#!/usr/bin/env python3
"""
generate_phantomnet_demo.py

Generates MRI magnitude images for all 10 patients listed using the PhantomNet 
numerical brain phantom framework (PhantomNet_multiclass600.ipynb).

- Uses numerical_brain_cropped.mat tissue property maps (PD, T1, T2, B0)
- Applies exact PhantomNet tissue modifications (Microbleed, Glioma, WML, Edema, Necrosis, Meningioma)
- Simulates k-space via 2D FFT with B0 phase shifts and T2* decay
- Reconstructs clean magnitude images (noise ignored per user instruction)
- Saves 1 PNG image per patient + master preview grid figure
"""

import os
from pathlib import Path
import scipy.io as sio
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ── Configuration ─────────────────────────────────────────────────────────────

IMG_SIZE = 256
MAT_PATH = "/home/jemin/Projects/Med_Matrix/ai-service/numerical_brain_cropped.mat"

PATIENTS = [
    {
        "id": "patient01",
        "disease": "Normal",
        "disease_code": 0,
        "notes": "Clean healthy brain anatomical baseline"
    },
    {
        "id": "patient02",
        "disease": "Low-grade Glioma",
        "disease_code": 1,
        "notes": "Mild T1/T2 elevation, ill-defined hyperintensity"
    },
    {
        "id": "patient03",
        "disease": "High-grade Glioma",
        "disease_code": 2,
        "notes": "Heterogeneous GBM core (compact size)"
    },
    {
        "id": "patient04",
        "disease": "Necrosis",
        "disease_code": 3,
        "notes": "Central necrotic core (compact size, low cellularity)"
    },
    {
        "id": "patient05",
        "disease": "Edema",
        "disease_code": 4,
        "notes": "Vasogenic white matter edema"
    },
    {
        "id": "patient06",
        "disease": "Meningioma",
        "disease_code": 5,
        "notes": "Peripheral dural-based mass"
    },
    {
        "id": "patient07",
        "disease": "Microbleed",
        "disease_code": 6,
        "notes": "Focal T2* hypointensity with B0 blooming"
    },
    {
        "id": "patient08",
        "disease": "Low-grade Glioma + Edema",
        "disease_code": 18,
        "notes": "Mixed tumor core & surrounding edema"
    },
    {
        "id": "patient09",
        "disease": "High-grade Glioma",
        "disease_code": 2,
        "notes": "Clean GBM presentation"
    },
    {
        "id": "patient10",
        "disease": "Normal",
        "disease_code": 0,
        "notes": "Healthy volunteer baseline"
    }
]


# ── PhantomNet Loader ─────────────────────────────────────────────────────────

def load_phantomnet_base(mat_path: str, size: int = 256):
    """
    Loads numerical_brain_cropped.mat and resizes maps to (size, size).
    Returns dict of 2D numpy arrays: pd, t1, t2, b0
    """
    mat = sio.loadmat(mat_path)
    brain = mat['cropped_brain']  # (141, 161, 5)

    brain_t = torch.tensor(brain).permute(2, 0, 1).unsqueeze(0).float()
    brain_res = F.interpolate(brain_t, size=(size, size), mode='bilinear', align_corners=False).squeeze(0).numpy()

    pd = brain_res[0].copy()
    t1 = brain_res[1].copy()
    t2 = brain_res[2].copy()
    b0 = brain_res[3].copy()

    # Set baseline parameters for brain tissue
    brain_mask = pd > 0.05
    t1 = np.where(brain_mask, np.maximum(t1, 0.84), 0.0)
    t2 = np.where(brain_mask, np.maximum(t2, 0.075), 0.0)

    return {
        "pd": pd,
        "t1": t1,
        "t2": t2,
        "t2dash": np.full_like(t2, 0.030),  # 30ms baseline T2*
        "b0": b0,
        "size": size
    }


# ── Pathology Insertion (PhantomNet Logic) ───────────────────────────────────

def apply_pathology(phantom: dict, p: dict) -> dict:
    """
    Modifies tissue property maps based on patient disease code.
    Implements PhantomNet pathology placement rules.
    """
    pd = phantom["pd"].copy()
    t1 = phantom["t1"].copy()
    t2 = phantom["t2"].copy()
    t2dash = phantom["t2dash"].copy()
    b0 = phantom["b0"].copy()
    size = phantom["size"]

    cx, cy = size // 2, size // 2
    yy, xx = np.ogrid[:size, :size]
    dist_from_center = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    brain_mask = pd > 0.08
    wm_mask = brain_mask & (pd > 0.50) & (pd < 0.85)

    d = p["disease_code"]

    if d != 0:
        tx = cx + int(size * 0.14)
        ty = cy - int(size * 0.10)

        if d == 6:  # Microbleed (PhantomNet add_microbleed)
            r = int(size * 0.025)
            dist_sq = (xx - tx)**2 + (yy - ty)**2
            core_mask = (dist_sq <= r**2) & brain_mask
            edge_mask = (dist_sq <= (r + int(size * 0.02))**2) & ~core_mask & brain_mask

            t2dash[core_mask] = 0.002   # 2ms hemosiderin T2*
            t2[core_mask] = 0.010       # 10ms T2
            pd[core_mask] = pd[core_mask] * 0.5
            b0[edge_mask] += 30.0       # B0 susceptibility blooming

        elif d == 2:  # High-grade Glioma (GBM - compact heterogeneous core)
            t_r = size * 0.065
            tumor_dist = np.sqrt((xx - tx)**2 + (yy - ty)**2)
            tumor_mask = (tumor_dist <= t_r) & brain_mask

            core_mask = (tumor_dist <= t_r * 0.45) & tumor_mask
            rim_mask = tumor_mask & ~core_mask

            edema_r = t_r * 1.5
            edema_mask = (tumor_dist <= edema_r) & brain_mask & ~tumor_mask

            # Edema region
            pd[edema_mask] = np.minimum(1.0, pd[edema_mask] * 1.20)
            t1[edema_mask] = 1.700
            t2[edema_mask] = 0.140

            # Hyperintense rim
            pd[rim_mask] = 0.95
            t1[rim_mask] = 2.600
            t2[rim_mask] = 0.200

            # Necrotic core
            pd[core_mask] = 0.35
            t1[core_mask] = 3.000
            t2[core_mask] = 0.080

        elif d == 3:  # Necrosis (compact fluid core)
            t_r = size * 0.055
            tumor_dist = np.sqrt((xx - tx)**2 + (yy - ty)**2)
            tumor_mask = (tumor_dist <= t_r) & brain_mask

            core_mask = (tumor_dist <= t_r * 0.70) & tumor_mask
            rim_mask = tumor_mask & ~core_mask

            pd[rim_mask] = 0.75
            t1[rim_mask] = 1.500
            t2[rim_mask] = 0.110

            pd[core_mask] = 0.20
            t1[core_mask] = 3.500
            t2[core_mask] = 0.350

        elif d == 1:  # Low-grade Glioma
            t_r = size * 0.085
            tumor_dist = np.sqrt((xx - tx)**2 / 1.1**2 + (yy - ty)**2 / 0.85**2)
            tumor_mask = (tumor_dist <= t_r) & brain_mask

            pd[tumor_mask] = 0.85
            t1[tumor_mask] = 1.800
            t2[tumor_mask] = 0.130

        elif d == 4:  # Edema
            tx_e = cx + int(size * 0.12)
            ty_e = cy - int(size * 0.08)
            e_dist = np.sqrt((xx - tx_e)**2 / 1.4**2 + (yy - ty_e)**2 / 0.9**2)
            edema_mask = (e_dist <= size * 0.10) & wm_mask

            pd[edema_mask] = np.minimum(1.0, pd[edema_mask] * 1.30)
            t1[edema_mask] = 1.700
            t2[edema_mask] = 0.145

        elif d == 5:  # Meningioma (peripheral extra-axial mass)
            tx_m = cx + int(size * 0.32)
            ty_m = cy - int(size * 0.16)
            men_dist = np.sqrt((xx - tx_m)**2 + (yy - ty_m)**2 / 1.3**2)
            men_mask = (men_dist <= size * 0.07) & brain_mask

            pd[men_mask] = 0.88
            t1[men_mask] = 1.500
            t2[men_mask] = 0.110

        elif d == 18:  # Low-grade Glioma + Edema
            t_r = size * 0.075
            tumor_dist = np.sqrt((xx - tx)**2 + (yy - ty)**2)
            tumor_mask = (tumor_dist <= t_r) & brain_mask
            edema_mask = (tumor_dist <= t_r * 1.6) & brain_mask & ~tumor_mask

            pd[edema_mask] = np.minimum(1.0, pd[edema_mask] * 1.25)
            t1[edema_mask] = 1.700
            t2[edema_mask] = 0.140

            pd[tumor_mask] = 0.82
            t1[tumor_mask] = 1.800
            t2[tumor_mask] = 0.130

    return {
        "pd": pd,
        "t1": t1,
        "t2": t2,
        "t2dash": t2dash,
        "b0": b0,
        "size": size
    }


# ── PhantomNet K-space Simulator & Reconstructor ────────────────────────────

def simulate_and_reconstruct(phantom: dict, TR_s: float = 0.50, TE_s: float = 0.020) -> np.ndarray:
    """
    Simulates MRI k-space from tissue property maps and reconstructs magnitude image.
    Uses 2D FFT with B0 phase shifts.
    """
    pd = phantom["pd"]
    t1 = phantom["t1"]
    t2 = phantom["t2"]
    t2dash = phantom["t2dash"]
    b0 = phantom["b0"]

    # Effective T2*
    r2star = np.where(t2 > 0, 1.0 / np.maximum(t2, 1e-4), 0.0) + np.where(t2dash > 0, 1.0 / np.maximum(t2dash, 1e-4), 0.0)
    t2star = np.where(r2star > 0, 1.0 / r2star, 1e-4)

    # Signal magnitude
    t1_safe = np.maximum(t1, 1e-4)
    signal_mag = pd * (1.0 - np.exp(-TR_s / t1_safe)) * np.exp(-TE_s / t2star)

    # B0 off-resonance phase shift
    phase = 2.0 * np.pi * b0 * TE_s
    complex_image = signal_mag * np.exp(1j * phase)

    # 2D FFT k-space simulation
    kspace = np.fft.fftshift(np.fft.fft2(complex_image))

    # Reconstruct magnitude image via 2D IFFT
    recon = np.abs(np.fft.ifft2(np.fft.ifftshift(kspace)))
    
    # Normalize to [0, 1]
    recon = recon - recon.min()
    if recon.max() > 0:
        recon = recon / recon.max()

    return recon.astype(np.float32)


# ── Main Generator ────────────────────────────────────────────────────────────

def main():
    out_dir = Path("/home/jemin/Projects/Med_Matrix/data-gen/output_preview")
    out_dir.mkdir(parents=True, exist_ok=True)

    single_dir = out_dir / "phantomnet_classes"
    single_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("Generating Clean MRI Images using PhantomNet Numerical Phantom")
    print("  - Loaded: numerical_brain_cropped.mat")
    print("  - 1 clean MRI image per patient (patient01 to patient10)")
    print("  - Noise & CEST parameters excluded per instruction")
    print("=" * 65)

    base_phantom = load_phantomnet_base(MAT_PATH, size=IMG_SIZE)
    generated_images = []

    for idx, p in enumerate(PATIENTS):
        patient_phantom = apply_pathology(base_phantom, p)
        recon_img = simulate_and_reconstruct(patient_phantom)

        # Save slice
        fname = f"{p['id']}_{p['disease'].lower().replace(' ', '_').replace('+', 'and')}.png"
        fpath = single_dir / fname
        plt.imsave(fpath, recon_img, cmap='gray')
        print(f"  ✓ Saved {fname}")

        generated_images.append({
            "meta": p,
            "img": recon_img,
            "path": fpath
        })

    # Save Master Grid Figure (2x5 grid for 10 patients)
    fig, axes = plt.subplots(2, 5, figsize=(20, 9), facecolor='#0f172a')
    fig.suptitle("PhantomNet MRI Simulations for All 10 Patients (Clean Presentation)",
                 fontsize=18, fontweight='bold', color='#f8fafc', y=0.97)

    for idx, item in enumerate(generated_images):
        row = idx // 5
        col = idx % 5
        ax = axes[row, col]

        p = item["meta"]
        img = item["img"]

        ax.imshow(img, cmap='bone', vmin=0.0, vmax=1.0)
        ax.axis('off')

        title_text = f"{p['id'].upper()}: {p['disease']}"
        ax.set_title(title_text, fontsize=12, fontweight='bold', color='#38bdf8', pad=8)

        ax.text(0.5, -0.06, p['notes'], transform=ax.transAxes,
                fontsize=9, color='#94a3b8', ha='center', va='top')

    plt.subplots_adjust(wspace=0.15, hspace=0.30, left=0.04, right=0.96, top=0.88, bottom=0.10)

    master_path = out_dir / "phantomnet_preview.png"
    plt.savefig(master_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print()
    print(f"✓ Master PhantomNet preview grid saved at: {master_path}")


if __name__ == "__main__":
    main()
