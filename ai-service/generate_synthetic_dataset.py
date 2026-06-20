import os
import argparse
import csv
import numpy as np
from multiprocessing import Pool, cpu_count

# Physiological relaxation times at 3T (in ms) and Proton Densities
T1_CSF, T2_CSF, PD_CSF = 4200.0, 2000.0, 1.0
T1_GM,  T2_GM,  PD_GM  = 1400.0, 90.0,   0.85
T1_WM,  T2_WM,  PD_WM  = 800.0,  70.0,   0.70
T1_BONE,T2_BONE,PD_BONE = 300.0,  15.0,   0.15
T1_BG,  T2_BG,  PD_BG  = 1.0,    1.0,    0.0

DISEASES = {
    0: "Healthy",
    1: "Glioma",
    2: "Stroke",
    3: "Multiple Sclerosis",
    4: "Hydrocephalus",
    5: "Brain Atrophy",
    6: "Subdural Hematoma",
    7: "Cerebral Cyst",
    8: "Localized Edema",
    9: "Cavernous Malformation",
    10: "Cerebral Microbleeds"
}

def get_ellipse_mask(x, y, cx, cy, rx, ry, angle_deg):
    """Generates a boolean mask for an ellipse rotated by angle_deg."""
    angle = np.radians(angle_deg)
    xr = (x - cx) * np.cos(angle) + (y - cy) * np.sin(angle)
    yr = -(x - cx) * np.sin(angle) + (y - cy) * np.cos(angle)
    return (xr / rx)**2 + (yr / ry)**2 <= 1.0

def generate_quantitative_maps(size: int, slice_idx: int, num_slices: int, category_id: int, difficulty: float):
    """
    Generates quantitative T1, T2, and PD maps at 3.0T for a given slice and disease.
    difficulty is a float in [0, 1] scaling the size and contrast of the anomaly.
    """
    y, x = np.ogrid[-1:1:complex(0, size), -1:1:complex(0, size)]
    
    # Scale phantom according to slice index (3D progression simulation)
    z_frac = slice_idx / max(1, num_slices - 1) if num_slices > 1 else 0.5
    scale = np.sin(np.pi * (0.15 + 0.7 * z_frac))
    
    # Initialize background
    t1 = np.ones((size, size), dtype=np.float32) * T1_BG
    t2 = np.ones((size, size), dtype=np.float32) * T2_BG
    pd = np.zeros((size, size), dtype=np.float32)
    
    # Anatomical variation for healthy / base phantoms
    rng_anom = np.random.default_rng(seed=int(difficulty * 100000) + slice_idx + category_id * 7)
    
    # Base scale & rotation
    b_scale = scale * rng_anom.uniform(0.97, 1.03) if category_id == 0 else scale
    b_rot = rng_anom.uniform(-4.0, 4.0) if category_id == 0 else 0.0
    
    # Ventricles radii
    v_rx = 0.12 * b_scale * (1.2 - 0.5 * abs(z_frac - 0.5))
    v_ry = 0.25 * b_scale
    
    # Hydrocephalus (category 4): enlarged ventricles
    if category_id == 4:
        dilation = 2.5 - 1.25 * difficulty
        v_rx *= dilation
        v_ry *= dilation
        
    # Brain Atrophy (category 5): widened ventricles & shrunk brain tissue
    brain_scale = 1.0
    midline_shift = 0.0
    if category_id == 5:
        brain_scale = 0.82 + 0.12 * difficulty
        dilation = 1.6 - 0.45 * difficulty
        v_rx *= dilation
        v_ry *= dilation
    elif category_id == 6:
        # Midline shift due to Subdural Hematoma
        midline_shift = (0.075 - 0.055 * difficulty) * b_scale

    # Base brain masks
    skull_outer_mask = get_ellipse_mask(x, y, 0.0, -0.05, 0.75 * b_scale, 0.85 * b_scale, b_rot)
    skull_inner_mask = get_ellipse_mask(x, y, 0.0, -0.05, 0.71 * b_scale, 0.81 * b_scale, b_rot)
    
    brain_tissue_mask = get_ellipse_mask(x, y, midline_shift, -0.05, 0.70 * b_scale * brain_scale, 0.80 * b_scale * brain_scale, b_rot)
    
    # White matter (Deep tracts)
    wm1_mask = get_ellipse_mask(x, y, -0.25 * b_scale * brain_scale + midline_shift, -0.2 * b_scale * brain_scale, 0.18 * b_scale * brain_scale, 0.25 * b_scale * brain_scale, 15 + b_rot)
    wm2_mask = get_ellipse_mask(x, y, 0.25 * b_scale * brain_scale + midline_shift, -0.2 * b_scale * brain_scale, 0.18 * b_scale * brain_scale, 0.25 * b_scale * brain_scale, -15 + b_rot)
    wm3_mask = get_ellipse_mask(x, y, midline_shift, 0.35 * b_scale * brain_scale, 0.25 * b_scale * brain_scale, 0.15 * b_scale * brain_scale, b_rot)
    
    v1_mask = get_ellipse_mask(x, y, -0.15 * b_scale + midline_shift, 0.05 * b_scale, v_rx, v_ry, -10 + b_rot)
    v2_mask = get_ellipse_mask(x, y, 0.15 * b_scale + midline_shift, 0.05 * b_scale, v_rx, v_ry, 10 + b_rot)
    
    # Fill standard tissue values
    t1[skull_outer_mask] = T1_BONE
    t2[skull_outer_mask] = T2_BONE
    pd[skull_outer_mask] = PD_BONE
    
    t1[skull_inner_mask] = T1_CSF
    t2[skull_inner_mask] = T2_CSF
    pd[skull_inner_mask] = PD_CSF
    
    t1[brain_tissue_mask] = T1_GM
    t2[brain_tissue_mask] = T2_GM
    pd[brain_tissue_mask] = PD_GM
    
    for wm_m in [wm1_mask, wm2_mask, wm3_mask]:
        wm_intersection = wm_m & brain_tissue_mask
        t1[wm_intersection] = T1_WM
        t2[wm_intersection] = T2_WM
        pd[wm_intersection] = PD_WM
        
    t1[v1_mask & brain_tissue_mask] = T1_CSF
    t2[v1_mask & brain_tissue_mask] = T2_CSF
    pd[v1_mask & brain_tissue_mask] = PD_CSF
    t1[v2_mask & brain_tissue_mask] = T1_CSF
    t2[v2_mask & brain_tissue_mask] = T2_CSF
    pd[v2_mask & brain_tissue_mask] = PD_CSF

    # Pathology Injection
    if category_id == 1:
        cx, cy = 0.33, -0.08
        r_tum = (0.16 - 0.11 * difficulty) * b_scale
        r_ede = r_tum + (0.07 - 0.05 * difficulty) * b_scale
        tumor_mask = get_ellipse_mask(x, y, cx, cy, r_tum, r_tum * 0.9, 15) & brain_tissue_mask
        edema_mask = get_ellipse_mask(x, y, cx, cy, r_ede, r_ede * 0.9, 15) & brain_tissue_mask
        
        t1[edema_mask] = 1650 - 250 * difficulty
        t2[edema_mask] = 135 - 45 * difficulty
        pd[edema_mask] = 0.90
        t1[tumor_mask] = 1450 - 80 * difficulty
        t2[tumor_mask] = 115 - 25 * difficulty
        pd[tumor_mask] = 0.91 - 0.04 * difficulty
        
    elif category_id == 2:
        cx, cy = -0.38, 0.02
        r_str = (0.20 - 0.14 * difficulty) * b_scale
        stroke_mask = get_ellipse_mask(x, y, cx, cy, r_str, r_str * 1.25, -25) & brain_tissue_mask
        
        t1[stroke_mask] = 1600 - 180 * difficulty
        t2[stroke_mask] = 130 - 35 * difficulty
        pd[stroke_mask] = 0.94 - 0.08 * difficulty
        
    elif category_id == 3:
        rng = np.random.default_rng(seed=int(difficulty * 450) + slice_idx)
        num_plaques = max(1, int(8 - 6 * difficulty))
        for _ in range(num_plaques):
            px = rng.uniform(0.13, 0.27) * rng.choice([-1.0, 1.0]) * b_scale
            py = rng.uniform(-0.12, 0.18) * b_scale
            p_rad = (0.032 - 0.021 * difficulty) * b_scale
            plaque_mask = get_ellipse_mask(x, y, px, py, p_rad, p_rad * rng.uniform(0.85, 1.15), rng.uniform(-30, 30)) & brain_tissue_mask
            t1[plaque_mask] = 1420 - 180 * difficulty
            t2[plaque_mask] = 112 - 32 * difficulty
            pd[plaque_mask] = 0.84 - 0.04 * difficulty
            
    elif category_id == 6:
        h_outer = get_ellipse_mask(x, y, 0.0, -0.05, 0.71 * b_scale, 0.81 * b_scale, b_rot)
        shift_val = (0.075 - 0.055 * difficulty) * b_scale
        h_inner = get_ellipse_mask(x, y, shift_val, -0.05, 0.71 * b_scale, 0.81 * b_scale, b_rot)
        hematoma_mask = h_outer & (~h_inner) & skull_inner_mask
        
        t1[hematoma_mask] = 880 + 220 * difficulty
        t2[hematoma_mask] = 58 + 22 * difficulty
        pd[hematoma_mask] = 0.81 - 0.04 * difficulty
        
    elif category_id == 7:
        cx, cy = -0.3, -0.25
        r_cy = (0.11 - 0.08 * difficulty) * b_scale
        cyst_mask = get_ellipse_mask(x, y, cx, cy, r_cy, r_cy, 0) & brain_tissue_mask
        
        t1[cyst_mask] = T1_CSF
        t2[cyst_mask] = T2_CSF
        pd[cyst_mask] = PD_CSF
        
    elif category_id == 8:
        cx, cy = 0.20, 0.28
        r_ede = (0.24 - 0.17 * difficulty) * b_scale
        edema_mask = get_ellipse_mask(x, y, cx, cy, r_ede, r_ede * 0.78, -12) & brain_tissue_mask
        
        t1[edema_mask] = 1580 - 140 * difficulty
        t2[edema_mask] = 122 - 28 * difficulty
        pd[edema_mask] = 0.88
        
    elif category_id == 9:
        rng = np.random.default_rng(seed=int(difficulty * 777) + slice_idx)
        cx, cy = 0.08, -0.28
        num_lobules = max(2, int(5 - 3 * difficulty))
        for _ in range(num_lobules):
            ox = rng.uniform(-0.035, 0.035) * b_scale
            oy = rng.uniform(-0.035, 0.035) * b_scale
            r_lob = (0.023 - 0.014 * difficulty) * b_scale
            lob_mask = get_ellipse_mask(x, y, cx + ox, cy + oy, r_lob, r_lob, rng.uniform(0, 180)) & brain_tissue_mask
            t1[lob_mask] = 160.0
            t2[lob_mask] = 5.5
            pd[lob_mask] = 0.06
            
    elif category_id == 10:
        rng = np.random.default_rng(seed=int(difficulty * 999) + slice_idx)
        num_bleeds = max(6, int(20 - 12 * difficulty))  # High-density microbleeds
        for _ in range(num_bleeds):
            r_dist = rng.uniform(0.1, 0.55) * b_scale
            theta = rng.uniform(0, 2 * np.pi)
            bx = r_dist * np.cos(theta) + midline_shift
            by = r_dist * np.sin(theta) - 0.05
            b_rad = (0.012 - 0.006 * difficulty) * b_scale
            bleed_mask = get_ellipse_mask(x, y, bx, by, b_rad, b_rad, 0.0) & brain_tissue_mask
            t1[bleed_mask] = 200.0
            t2[bleed_mask] = 5.0
            pd[bleed_mask] = 0.05
            
    # Clip map values
    t1 = np.clip(t1, 1.0, 5000.0)
    t2 = np.clip(t2, 1.0, 3000.0)
    pd = np.clip(pd, 0.0, 1.0)
    return t1, t2, pd

def simulate_mri_signal(t1: np.ndarray, t2: np.ndarray, pd: np.ndarray, TR: float, TE: float) -> np.ndarray:
    t1_safe = np.where(t1 > 0, t1, 1e-8)
    t2_safe = np.where(t2 > 0, t2, 1e-8)
    return pd * (1.0 - np.exp(-TR / t1_safe)) * np.exp(-TE / t2_safe)

def generate_coil_sensitivities(height: int, width: int, num_coils: int) -> np.ndarray:
    y, x = np.ogrid[-1:1:complex(0, height), -1:1:complex(0, width)]
    coils = []
    r_coil = 1.2
    sigma = 1.5
    for c in range(num_coils):
        angle = 2 * np.pi * c / num_coils
        cx = r_coil * np.cos(angle)
        cy = r_coil * np.sin(angle)
        dist_sq = (x - cx)**2 + (y - cy)**2
        magnitude = np.exp(-dist_sq / (2 * sigma**2))
        phase = np.arctan2(y - cy, x - cx)
        sens = magnitude * np.exp(1j * phase)
        coils.append(sens)
    sens_maps = np.stack(coils, axis=0)
    rss_sens = np.sqrt(np.sum(np.abs(sens_maps)**2, axis=0))
    rss_sens[rss_sens == 0] = 1e-8
    sens_maps = sens_maps / rss_sens[np.newaxis, ...]
    return sens_maps

def generate_single_patient(args):
    """Worker function to generate a patient volume [slices, coils, resolution, resolution]"""
    slices, coils, resolution, category_id, difficulty, output_path = args
    
    patient_volume = np.zeros((slices, coils, resolution, resolution), dtype=np.complex64)
    sens_maps = generate_coil_sensitivities(resolution, resolution, coils)
    
    # Inject pathology in the middle 50% of slices
    pathology_start = slices // 4
    pathology_end = 3 * slices // 4
    
    for s in range(slices):
        if s == 0:
            # T1-weighted localize slice
            TR, TE = 600.0, 15.0
        else:
            # T2-weighted contrast
            TR, TE = 3000.0, 90.0
            
        curr_cat_id = category_id if (s >= pathology_start and s < pathology_end) else 0
        curr_diff = difficulty if (s >= pathology_start and s < pathology_end) else 0.0
        
        t1, t2, pd = generate_quantitative_maps(resolution, s, slices, curr_cat_id, curr_diff)
        img_slice = simulate_mri_signal(t1, t2, pd, TR, TE)
        
        for c in range(coils):
            coil_img = img_slice * sens_maps[c]
            coil_kspace = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(coil_img)))
            patient_volume[s, c, :, :] = coil_kspace.astype(np.complex64)
            
    # Add complex Gaussian noise (~30 dB SNR)
    vol_std = np.std(np.abs(patient_volume))
    noise_std = 0.03 * vol_std
    noise = (np.random.normal(0, noise_std, patient_volume.shape) + 
             1j * np.random.normal(0, noise_std, patient_volume.shape)) / np.sqrt(2)
    patient_volume += noise.astype(np.complex64)
    
    np.save(output_path, patient_volume)
    return True

def main():
    parser = argparse.ArgumentParser(description="Synthetic MRI Patient K-Space Volume Generator.")
    parser.add_argument("--num_patients", type=int, default=800, help="Total number of patients to generate.")
    parser.add_argument("--slices", type=int, default=8, help="Slices per patient.")
    parser.add_argument("--coils", type=int, default=16, help="Coils per patient.")
    parser.add_argument("--resolution", type=int, default=256, help="Resolution.")
    parser.add_argument("--normal_ratio", type=float, default=0.4, help="Ratio of healthy patients.")
    parser.add_argument("--output_dir", type=str, default="/app/data/synthetic_kspace", help="Output directory.")
    parser.add_argument("--num_workers", type=int, default=None, help="Parallel worker process count.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    num_patients = args.num_patients
    num_healthy = int(num_patients * args.normal_ratio)
    num_diseased = num_patients - num_healthy
    
    diseased_categories = list(range(1, 11))
    patients_per_disease = num_diseased // len(diseased_categories)
    
    # Recalculate true counts based on integer division
    num_diseased = patients_per_disease * len(diseased_categories)
    num_patients = num_healthy + num_diseased
    
    print(f"Generating patient-level raw K-space dataset:")
    print(f"  - Total Patients: {num_patients}")
    print(f"  - Healthy Patients: {num_healthy}")
    print(f"  - Diseased Patients: {num_diseased} ({patients_per_disease} per disease category 1-10)")
    print(f"  - Shape per Patient: [{args.slices}, {args.coils}, {args.resolution}, {args.resolution}]")
    print(f"  - Output folder: {args.output_dir}")

    # Prepare patient generation tasks
    tasks = []
    
    # 1. Healthy Patients
    for i in range(num_healthy):
        filename = f"patient_{i:04d}.npy"
        filepath = os.path.join(args.output_dir, filename)
        difficulty = float(i) / max(1, num_healthy - 1)
        tasks.append({
            "patient_index": i,
            "filename": filename,
            "category_id": 0,
            "category_name": DISEASES[0],
            "difficulty": difficulty,
            "is_healthy": True,
            "filepath": filepath
        })
        
    # 2. Diseased Patients
    temp_diseased = []
    for cat_id in diseased_categories:
        for j in range(patients_per_disease):
            difficulty = float(j) / max(1, patients_per_disease - 1)
            temp_diseased.append((cat_id, DISEASES[cat_id], difficulty))
            
    # Sort diseased patients by difficulty
    temp_diseased.sort(key=lambda x: x[2])
    
    for rank, (cat_id, name, diff) in enumerate(temp_diseased):
        idx = num_healthy + rank
        filename = f"patient_{idx:04d}.npy"
        filepath = os.path.join(args.output_dir, filename)
        tasks.append({
            "patient_index": idx,
            "filename": filename,
            "category_id": cat_id,
            "category_name": name,
            "difficulty": diff,
            "is_healthy": False,
            "filepath": filepath
        })

    # Prepare worker args
    worker_args = [
        (args.slices, args.coils, args.resolution, t["category_id"], t["difficulty"], t["filepath"])
        for t in tasks
    ]

    workers = args.num_workers if args.num_workers else cpu_count()
    print(f"Starting patient generation using {workers} parallel processes...")
    
    completed = 0
    with Pool(processes=workers) as pool:
        for _ in pool.imap_unordered(generate_single_patient, worker_args):
            completed += 1
            if completed % max(1, num_patients // 10) == 0 or completed == num_patients:
                print(f"Generated {completed}/{num_patients} patients ({completed * 100 // num_patients}%)...")
                
    print("Writing metadata manifest...")
    manifest_path = os.path.join(args.output_dir, "dataset_manifest.csv")
    with open(manifest_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["patient_index", "filename", "category_id", "category_name", "difficulty", "is_healthy"])
        for t in tasks:
            writer.writerow([
                t["patient_index"],
                t["filename"],
                t["category_id"],
                t["category_name"],
                f"{t['difficulty']:.6f}",
                "1" if t["is_healthy"] else "0"
            ])
            
    print(f"Success! {num_patients} patients generated. Manifest saved to {manifest_path}.")

if __name__ == "__main__":
    main()
