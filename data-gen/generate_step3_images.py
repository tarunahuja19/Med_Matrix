#!/usr/bin/env python3
"""
generate_step3_images.py

Generates Step 3 Spin-Echo MRI magnitude images for each patient class specified:
  - patient01: Normal
  - patient02: Low-grade Glioma
  - patient03: High-grade Glioma (Smaller, realistic lesion size)
  - patient04: Necrosis (Smaller, realistic lesion size)
  - patient05: Edema
  - patient06: Meningioma
  - patient08: Low-grade Glioma + Edema
  - patient09: High-grade Glioma (Clean, smaller size)
  - patient10: Normal

Microbleeds (patient07) is omitted per user instruction.
Noise column is ignored for clean presentation.
"""

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ── Constants ────────────────────────────────────────────────────────────────
IMG_SIZE = 256

PATIENTS_STEP3 = [
    {
        "id": "patient01",
        "disease": "Normal",
        "disease_code": 0,
        "notes": "Clean healthy brain",
        "apt_pct": "1.2%"
    },
    {
        "id": "patient02",
        "disease": "Low-grade Glioma",
        "disease_code": 1,
        "notes": "Moderate APT elevation",
        "apt_pct": "3.1%"
    },
    {
        "id": "patient03",
        "disease": "High-grade Glioma",
        "disease_code": 2,
        "notes": "GBM presentation (compact size)",
        "apt_pct": "5.2%"
    },
    {
        "id": "patient04",
        "disease": "Necrosis",
        "disease_code": 3,
        "notes": "Central necrotic core (compact size)",
        "apt_pct": "0.5%"
    },
    {
        "id": "patient05",
        "disease": "Edema",
        "disease_code": 4,
        "notes": "Vasogenic white matter edema",
        "apt_pct": "1.8%"
    },
    {
        "id": "patient06",
        "disease": "Meningioma",
        "disease_code": 5,
        "notes": "Peripheral dural-based mass",
        "apt_pct": "2.8%"
    },
    {
        "id": "patient07",
        "disease": "Microbleed",
        "disease_code": 6,
        "notes": "Focal T2* hypointensity",
        "apt_pct": "1.1%"
    },
    {
        "id": "patient08",
        "disease": "Low-grade Glioma + Edema",
        "disease_code": 18,  # Mixed
        "notes": "Mixed tumor core & surrounding edema",
        "apt_pct": "2.6%"
    },
    {
        "id": "patient09",
        "disease": "High-grade Glioma",
        "disease_code": 2,
        "notes": "Clean GBM presentation",
        "apt_pct": "5.8%"
    },
    {
        "id": "patient10",
        "disease": "Normal",
        "disease_code": 0,
        "notes": "Healthy volunteer anatomical baseline",
        "apt_pct": "1.3%"
    }
]

def make_detailed_brain_phantom(size: int, p: dict, seed: int = 42) -> np.ndarray:
    """
    Generates a realistic 2D Spin-Echo brain MRI phantom slice [0, 1].
    HGG and Necrosis lesions are scaled down for realistic clinical appearance.
    Noise is ignored for clean presentation.
    """
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size), dtype=np.float32)
    cx, cy = size // 2, size // 2

    yy, xx = np.ogrid[:size, :size]
    dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)

    # 1. Skull & Calvarial bone ring
    skull_outer = size * 0.46
    skull_inner = size * 0.42
    skull_mask = (dist > skull_inner) & (dist <= skull_outer)
    img[skull_mask] = 0.18

    # Scalp subcutaneous fat (bright ring outside skull)
    scalp_mask = (dist > skull_outer) & (dist <= size * 0.485)
    img[scalp_mask] = 0.65

    # 2. Brain parenchymal tissue
    brain_r = skull_inner
    brain_mask = dist <= brain_r

    # White matter core (~70% radius)
    wm_r = brain_r * 0.68
    wm_mask = dist <= wm_r
    img[wm_mask] = 0.58

    # Gray matter cortical ribbon (outer ~32% with subtle cortical undulations)
    gm_mask = brain_mask & ~wm_mask
    cortex_ripple = 1.0 + 0.04 * np.sin(np.arctan2(yy - cy, xx - cx) * 14)
    img[gm_mask] = 0.76 * cortex_ripple[gm_mask]

    # Deep gray matter nuclei (Thalami / Caudate)
    thalamus_l = ((xx - (cx - size * 0.08))**2 / (size * 0.05)**2 + (yy - cy)**2 / (size * 0.09)**2) <= 1.0
    thalamus_r = ((xx - (cx + size * 0.08))**2 / (size * 0.05)**2 + (yy - cy)**2 / (size * 0.09)**2) <= 1.0
    img[thalamus_l | thalamus_r] = 0.70

    # Lateral Ventricles (CSF - dark on T1w/Spin-Echo)
    vent_l = ((xx - (cx - size * 0.05))**2 / (size * 0.035)**2 + (yy - cy)**2 / (size * 0.12)**2) <= 1.0
    vent_r = ((xx - (cx + size * 0.05))**2 / (size * 0.035)**2 + (yy - cy)**2 / (size * 0.12)**2) <= 1.0
    vent_mask = (vent_l | vent_r) & brain_mask
    img[vent_mask] = 0.08

    # 3. Disease Pathology Insertion
    d = p["disease_code"]
    
    if d != 0:  # If not Normal
        # Coordinates for pathology
        tx = cx + int(size * 0.14)
        ty = cy - int(size * 0.10)

        if d == 2:  # High-grade Glioma (HGG) -> SMALLER (t_r ~ 0.065)
            t_r = size * 0.065
            tumor_mask = ((xx - tx)**2 + (yy - ty)**2) <= t_r**2
            tumor_mask &= brain_mask

            # Heterogeneous tumor: Rim + Necrotic Center + Edema
            rim_mask = tumor_mask
            core_r = t_r * 0.45
            core_mask = (((xx - tx)**2 + (yy - ty)**2) <= core_r**2) & tumor_mask

            edema_r = t_r * 1.5
            edema_mask = (((xx - tx)**2 / (edema_r * 1.2)**2 + (yy - ty)**2 / (edema_r * 0.9)**2) <= 1.0) & brain_mask & ~tumor_mask

            img[edema_mask] = np.minimum(1.0, img[edema_mask] * 1.22)
            img[rim_mask] = 0.92  # Bright hyperintense rim
            img[core_mask] = 0.25 # Darker necrotic core

        elif d == 3:  # Necrosis -> SMALLER (t_r ~ 0.055)
            t_r = size * 0.055
            tumor_mask = ((xx - tx)**2 + (yy - ty)**2) <= t_r**2
            tumor_mask &= brain_mask

            core_mask = (((xx - tx)**2 + (yy - ty)**2) <= (t_r * 0.7)**2) & tumor_mask
            rim_mask = tumor_mask & ~core_mask

            img[rim_mask] = 0.72  # Mild gliotic rim
            img[core_mask] = 0.12 # Fluid-like dark necrotic center

        elif d == 1:  # Low-grade Glioma
            t_r = size * 0.09
            tumor_mask = ((xx - tx)**2 / (t_r * 1.1)**2 + (yy - ty)**2 / (t_r * 0.85)**2) <= 1.0
            tumor_mask &= brain_mask
            img[tumor_mask] = 0.82  # Homogeneous mild hyperintensity

        elif d == 4:  # Edema
            tx_e = cx + int(size * 0.12)
            ty_e = cy - int(size * 0.08)
            e_rx = size * 0.14
            e_ry = size * 0.09
            edema_mask = ((xx - tx_e)**2 / e_rx**2 + (yy - ty_e)**2 / e_ry**2) <= 1.0
            edema_mask &= wm_mask  # Edema follows white matter tracts
            img[edema_mask] = np.minimum(1.0, img[edema_mask] * 1.30)

        elif d == 5:  # Meningioma (extra-axial dural based mass)
            tx_m = cx + int(size * 0.33)
            ty_m = cy - int(size * 0.16)
            t_r = size * 0.07
            men_mask = ((xx - tx_m)**2 / t_r**2 + (yy - ty_m)**2 / (t_r * 1.3)**2) <= 1.0
            men_mask &= brain_mask
            img[men_mask] = 0.86  # Iso/hyperintense extra-axial mass

        elif d == 6:  # Microbleed (small focal hypointensity)
            t_r = size * 0.025
            mb_mask = ((xx - tx)**2 + (yy - ty)**2) <= t_r**2
            mb_mask &= brain_mask
            img[mb_mask] = 0.02  # Very dark T2* blooming spot

        elif d == 18:  # Low-grade Glioma + Edema
            t_r = size * 0.08
            tumor_mask = ((xx - tx)**2 + (yy - ty)**2) <= t_r**2
            tumor_mask &= brain_mask
            edema_r = t_r * 1.6
            edema_mask = (((xx - tx)**2 / (edema_r * 1.2)**2 + (yy - ty)**2 / edema_r**2) <= 1.0) & brain_mask & ~tumor_mask
            img[edema_mask] = np.minimum(1.0, img[edema_mask] * 1.25)
            img[tumor_mask] = 0.80

    # 4. Smooth Gaussian blur for realistic point-spread function
    from numpy.fft import fft2, ifft2, fftfreq
    fy = fftfreq(size).reshape(-1, 1)
    fx = fftfreq(size).reshape(1, -1)
    sigma_pix = 1.0
    gauss_filt = np.exp(-2 * np.pi**2 * sigma_pix**2 * (fx**2 + fy**2))
    img_f = fft2(img.astype(np.complex64))
    img_smooth = np.abs(ifft2(img_f * gauss_filt)).astype(np.float32)
    img_smooth = np.clip(img_smooth, 0.0, 1.0)

    return img_smooth

def main():
    out_dir = Path("/home/jemin/Projects/Med_Matrix/data-gen/output_preview")
    out_dir.mkdir(parents=True, exist_ok=True)

    single_dir = out_dir / "step3_classes"
    single_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Generating Clean MRI Images for All 10 Patients")
    print("  - 1 image per patient listed (patient01 to patient10)")
    print("  - Noise ignored for clean presentation")
    print("=" * 60)

    generated_images = []

    for idx, p in enumerate(PATIENTS_STEP3):
        # Generate image
        mri_slice = make_detailed_brain_phantom(IMG_SIZE, p, seed=100 + idx)
        
        # Save individual slice
        fname = f"{p['id']}_{p['disease'].lower().replace(' ', '_').replace('+', 'and')}.png"
        fpath = single_dir / fname
        plt.imsave(fpath, mri_slice, cmap='gray')
        print(f"  ✓ Saved {fname}")
        
        generated_images.append({
            "meta": p,
            "img": mri_slice,
            "path": fpath
        })

    # Save Master Step 3 Preview Grid Figure (2x5 grid for 10 patients)
    fig, axes = plt.subplots(2, 5, figsize=(20, 9), facecolor='#0f172a')
    fig.suptitle("Clean MRI Images for All 10 Patients (Noise & CEST Excluded)",
                 fontsize=18, fontweight='bold', color='#f8fafc', y=0.97)

    for idx, item in enumerate(generated_images):
        row = idx // 5
        col = idx % 5
        ax = axes[row, col]
        
        p = item["meta"]
        img = item["img"]

        ax.imshow(img, cmap='bone', vmin=0.0, vmax=1.0)
        ax.axis('off')

        # Title card with patient ID and disease
        title_text = f"{p['id'].upper()}: {p['disease']}"
        ax.set_title(title_text, fontsize=12, fontweight='bold', color='#38bdf8', pad=8)

        # Subtle bottom caption with notes
        ax.text(0.5, -0.06, p['notes'], transform=ax.transAxes,
                fontsize=9, color='#94a3b8', ha='center', va='top')

    plt.subplots_adjust(wspace=0.15, hspace=0.30, left=0.04, right=0.96, top=0.88, bottom=0.10)
    
    master_path = out_dir / "step3_preview.png"
    plt.savefig(master_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print()
    print(f"✓ Master preview grid updated at: {master_path}")

if __name__ == "__main__":
    main()

