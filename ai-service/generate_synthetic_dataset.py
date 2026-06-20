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
    Generates quantitative T1, T2, and PD maps at 3.0T by loading and interpolating
    the realistic numerical brain phantom from numerical_brain_cropped.mat.
    """
    import scipy.io
    import scipy.ndimage

    mat_path = os.path.join(os.path.dirname(__file__), "numerical_brain_cropped.mat")
    if not os.path.exists(mat_path):
        mat_path = "numerical_brain_cropped.mat"
        
    mat = scipy.io.loadmat(mat_path)
    raw_brain = mat['cropped_brain'] # shape (141, 161, 5)
    
    # Extract 2D channel maps
    pd_raw = raw_brain[..., 0]
    t1_raw = raw_brain[..., 1]
    t2_raw = raw_brain[..., 2]
    b0_raw = raw_brain[..., 3]
    
    # Rescale each 2D channel to (size, size) using linear interpolation
    zoom_y = size / pd_raw.shape[0]
    zoom_x = size / pd_raw.shape[1]
    
    pd = scipy.ndimage.zoom(pd_raw, (zoom_y, zoom_x), order=1)
    t1 = scipy.ndimage.zoom(t1_raw, (zoom_y, zoom_x), order=1) * 1000.0  # to ms
    t2 = scipy.ndimage.zoom(t2_raw, (zoom_y, zoom_x), order=1) * 1000.0  # to ms
    b0 = scipy.ndimage.zoom(b0_raw, (zoom_y, zoom_x), order=1)
    
    # Initialize t2dash map (default 30ms for brain tissue, 1e8 for CSF/background)
    t2dash = np.where((t2 > 1500.0) | (pd < 0.1), 1e8, 30.0)
    
    # Brain tissue mask (where PD > 0.1)
    brain_mask = pd > 0.1

    y, x = np.ogrid[-1:1:complex(0, size), -1:1:complex(0, size)]
    
    # Hydrocephalus (category 4): enlarged ventricles
    if category_id == 4 or category_id == 5:
        # Ventricles are roughly centered at x in [-0.25, 0.25] and y in [-0.2, 0.35]
        y_grid, x_grid = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size), indexing='ij')
        ventricle_coord_mask = (x_grid > -0.25) & (x_grid < 0.25) & (y_grid > -0.2) & (y_grid < 0.35)
        ventricle_mask = ventricle_coord_mask & (t2 > 1000.0) & brain_mask
        
        if category_id == 4:
            # Hydrocephalus: dilate ventricles
            dilation_pixels = int((12 - 6 * difficulty) * (size / 256.0))
            dilated_ventricles = scipy.ndimage.binary_dilation(ventricle_mask, iterations=max(1, dilation_pixels))
            t1[dilated_ventricles] = T1_CSF
            t2[dilated_ventricles] = T2_CSF
            pd[dilated_ventricles] = PD_CSF
            t2dash[dilated_ventricles] = 1e8
            
        elif category_id == 5:
            # Brain Atrophy: dilate ventricles and erode outer brain tissue
            dilation_pixels = int((5 - 2 * difficulty) * (size / 256.0))
            dilated_ventricles = scipy.ndimage.binary_dilation(ventricle_mask, iterations=max(1, dilation_pixels))
            t1[dilated_ventricles] = T1_CSF
            t2[dilated_ventricles] = T2_CSF
            pd[dilated_ventricles] = PD_CSF
            t2dash[dilated_ventricles] = 1e8
            
            # Erode brain tissue boundary to simulate widened sulci (outer CSF space)
            tissue_mask = (pd > 0.3) & (t2 < 500.0)
            erosion_pixels = int((6 - 3 * difficulty) * (size / 256.0))
            eroded_tissue = scipy.ndimage.binary_erosion(tissue_mask, iterations=max(1, erosion_pixels))
            atrophy_mask = tissue_mask & ~eroded_tissue
            t1[atrophy_mask] = T1_CSF
            t2[atrophy_mask] = T2_CSF
            pd[atrophy_mask] = PD_CSF
            t2dash[atrophy_mask] = 1e8
 
    # Pathology Injection
    if category_id == 1:
        cx, cy = 0.33, -0.08
        r_tum = (0.16 - 0.11 * difficulty)
        r_ede = r_tum + (0.07 - 0.05 * difficulty)
        tumor_mask = get_ellipse_mask(x, y, cx, cy, r_tum, r_tum * 0.9, 15) & brain_mask
        edema_mask = get_ellipse_mask(x, y, cx, cy, r_ede, r_ede * 0.9, 15) & brain_mask
        
        t1[edema_mask] = 1650 - 250 * difficulty
        t2[edema_mask] = 135 - 45 * difficulty
        t2dash[edema_mask] = 70.0
        pd[edema_mask] = 0.90
        
        t1[tumor_mask] = 1450 - 80 * difficulty
        t2[tumor_mask] = 115 - 25 * difficulty
        t2dash[tumor_mask] = 60.0
        pd[tumor_mask] = 0.91 - 0.04 * difficulty
        
    elif category_id == 2:
        cx, cy = -0.38, 0.02
        r_str = (0.20 - 0.14 * difficulty)
        stroke_mask = get_ellipse_mask(x, y, cx, cy, r_str, r_str * 1.25, -25) & brain_mask
        
        # Edema tissue properties (ischemic stroke / cytotoxic edema)
        t1[stroke_mask] = 1500.0  # acute ischemia (notebook: 1500ms)
        t2[stroke_mask] = 200.0   # very prolonged (notebook: 200ms)
        t2dash[stroke_mask] = 50.0 # T2* slightly elevated (notebook: 50ms)
        pd[stroke_mask] = pd[stroke_mask] * 1.2  # elevated (notebook: * 1.2)
        
        # B0 distortion at boundaries (notebook: B0 = B0 + 10.0)
        r_str_edge = r_str + 0.04
        edge_mask = get_ellipse_mask(x, y, cx, cy, r_str_edge, r_str_edge * 1.25, -25) & brain_mask & ~stroke_mask
        b0[edge_mask] = b0[edge_mask] + 10.0
        
    elif category_id == 3:
        rng = np.random.default_rng(seed=int(difficulty * 450) + slice_idx)
        num_plaques = max(1, int(8 - 6 * difficulty))
        for _ in range(num_plaques):
            px = rng.uniform(0.13, 0.27) * rng.choice([-1.0, 1.0])
            py = rng.uniform(-0.12, 0.18)
            p_rad = (0.032 - 0.021 * difficulty)
            plaque_mask = get_ellipse_mask(x, y, px, py, p_rad, p_rad * rng.uniform(0.85, 1.15), rng.uniform(-30, 30)) & brain_mask
            
            # Demyelination tissue properties (WML)
            t1[plaque_mask] = 1200.0   # prolonged (notebook: 1200ms)
            t2[plaque_mask] = 150.0    # very prolonged (notebook: 150ms)
            t2dash[plaque_mask] = 80.0  # T2* prolonged (notebook: 80ms)
            pd[plaque_mask] = pd[plaque_mask] * 1.1  # slightly elevated
            
    elif category_id == 6:
        shift_val = (0.075 - 0.055 * difficulty)
        h_outer = get_ellipse_mask(x, y, 0.0, -0.05, 0.71, 0.81, 0.0)
        h_inner = get_ellipse_mask(x, y, shift_val, -0.05, 0.71, 0.81, 0.0)
        hematoma_mask = h_outer & (~h_inner)
        
        t1[hematoma_mask] = 880 + 220 * difficulty
        t2[hematoma_mask] = 58 + 22 * difficulty
        t2dash[hematoma_mask] = 15.0  # blood breakdown products (low T2*)
        pd[hematoma_mask] = 0.81 - 0.04 * difficulty
        
    elif category_id == 7:
        cx, cy = -0.3, -0.25
        r_cy = (0.11 - 0.08 * difficulty)
        cyst_mask = get_ellipse_mask(x, y, cx, cy, r_cy, r_cy, 0) & brain_mask
        
        t1[cyst_mask] = T1_CSF
        t2[cyst_mask] = T2_CSF
        pd[cyst_mask] = PD_CSF
        t2dash[cyst_mask] = 1e8
        
    elif category_id == 8:
        cx, cy = 0.20, 0.28
        r_ede = (0.24 - 0.17 * difficulty)
        edema_mask = get_ellipse_mask(x, y, cx, cy, r_ede, r_ede * 0.78, -12) & brain_mask
        
        t1[edema_mask] = 1580 - 140 * difficulty
        t2[edema_mask] = 122 - 28 * difficulty
        t2dash[edema_mask] = 60.0
        pd[edema_mask] = 0.88
        
    elif category_id == 9:
        rng = np.random.default_rng(seed=int(difficulty * 777) + slice_idx)
        cx, cy = 0.08, -0.28
        num_lobules = max(2, int(5 - 3 * difficulty))
        for _ in range(num_lobules):
            ox = rng.uniform(-0.035, 0.035)
            oy = rng.uniform(-0.035, 0.035)
            r_lob = (0.023 - 0.014 * difficulty)
            lob_mask = get_ellipse_mask(x, y, cx + ox, cy + oy, r_lob, r_lob, rng.uniform(0, 180)) & brain_mask
            
            # Hemosiderin tissue properties
            t1[lob_mask] = 200.0
            t2[lob_mask] = 10.0
            t2dash[lob_mask] = 2.0  # 2ms hemosiderin T2* (notebook: 2ms)
            pd[lob_mask] = 0.05
            
            # B0 distortion at the boundary
            r_lob_edge = r_lob + 0.015
            lob_edge_mask = get_ellipse_mask(x, y, cx + ox, cy + oy, r_lob_edge, r_lob_edge, rng.uniform(0, 180)) & brain_mask & ~lob_mask
            b0[lob_edge_mask] = b0[lob_edge_mask] + 20.0
            
    elif category_id == 10:
        rng = np.random.default_rng(seed=int(difficulty * 999) + slice_idx)
        num_bleeds = max(6, int(20 - 12 * difficulty))  # High-density microbleeds
        for _ in range(num_bleeds):
            r_dist = rng.uniform(0.1, 0.55)
            theta = rng.uniform(0, 2 * np.pi)
            bx = r_dist * np.cos(theta)
            by = r_dist * np.sin(theta) - 0.05
            b_rad = (0.012 - 0.006 * difficulty)
            bleed_mask = get_ellipse_mask(x, y, bx, by, b_rad, b_rad, 0.0) & brain_mask
            
            # Hemosiderin tissue properties (microbleed)
            t1[bleed_mask] = 200.0
            t2[bleed_mask] = 10.0
            t2dash[bleed_mask] = 2.0  # 2ms hemosiderin T2* (notebook: 2ms)
            pd[bleed_mask] = pd[bleed_mask] * 0.5  # halved (notebook: * 0.5)
            
            # B0 distortion at the boundaries for blooming artifact (notebook: B0 = B0 + 30.0)
            b_rad_edge = b_rad + 0.01
            bleed_edge_mask = get_ellipse_mask(x, y, bx, by, b_rad_edge, b_rad_edge, 0.0) & brain_mask & ~bleed_mask
            b0[bleed_edge_mask] = b0[bleed_edge_mask] + 30.0
            
    # Clip map values
    t1 = np.clip(t1, 1.0, 5000.0)
    t2 = np.clip(t2, 1.0, 3000.0)
    pd = np.clip(pd, 0.0, 1.0)
    t2dash = np.clip(t2dash, 1.0, 1e8)
    return t1, t2, pd, t2dash, b0

def simulate_mri_signal(t1: np.ndarray, t2: np.ndarray, pd: np.ndarray, t2dash: np.ndarray, b0: np.ndarray, TR: float, TE: float) -> np.ndarray:
    t1_safe = np.where(t1 > 0, t1, 1e-8)
    t2_safe = np.where(t2 > 0, t2, 1e-8)
    
    # Calculate t2_star: 1/t2_star = 1/t2 + 1/t2dash
    # CSF has t2dash = infinity (represented as 1e8, so 1/t2dash = 0)
    t2dash_inv = np.where(t2dash > 0, 1.0 / t2dash, 0.0)
    t2_star = 1.0 / (1.0 / t2_safe + t2dash_inv)
    
    # Steady state GRE/FLASH signal equation
    # S = pd * sin(alpha) * (1 - exp(-TR/T1)) / (1 - cos(alpha)*exp(-TR/T1)) * exp(-TE/T2*)
    # flip angle alpha = 10 degrees (same as notebook: 10 * pi / 180 = 0.1745 rad)
    alpha = np.radians(10.0)
    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)
    
    exp_tr_t1 = np.exp(-TR / t1_safe)
    gre_magnitude = pd * sin_alpha * (1.0 - exp_tr_t1) / (1.0 - cos_alpha * exp_tr_t1) * np.exp(-TE / t2_star)
    
    # Phase shift from B0 off-resonance (TE is in ms, so convert to seconds: TE / 1000.0)
    phase = 2.0 * np.pi * b0 * (TE / 1000.0)
    
    return gre_magnitude * np.exp(1j * phase)

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
        
        t1, t2, pd, t2dash, b0 = generate_quantitative_maps(resolution, s, slices, curr_cat_id, curr_diff)
        img_slice = simulate_mri_signal(t1, t2, pd, t2dash, b0, TR, TE)
        
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
    parser.add_argument("--num_patients", type=int, default=2000, help="Total number of patients to generate.")
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
