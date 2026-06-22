import os
import argparse
import csv
import numpy as np
from multiprocessing import Pool, cpu_count

def corrupt_kspace_slice_numpy(kspace: np.ndarray, p_noise: float, p_motion: float, p_phase: float, rng=None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
        
    C, H, W = kspace.shape
    kspace_corrupted = kspace.copy()
    
    # 1. Apply Phase Error (in Image Domain)
    if p_phase > 0:
        shifted_k = np.fft.ifftshift(kspace_corrupted, axes=(-2, -1))
        img = np.fft.ifft2(shifted_k, axes=(-2, -1))
        img_shift = np.fft.fftshift(img, axes=(-2, -1))
        
        y_grid, x_grid = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, W), indexing='ij')
        phi_global = rng.uniform(-np.pi, np.pi)
        phi_spatial = (
            rng.uniform(-1, 1) * np.sin(np.pi * y_grid) +
            rng.uniform(-1, 1) * np.cos(np.pi * x_grid) +
            rng.uniform(-0.5, 0.5) * np.sin(2 * np.pi * (x_grid + y_grid))
        )
        phase_map = p_phase * (phi_global + 2.0 * phi_spatial)
        
        img_corrupted = img_shift * np.exp(1j * phase_map[np.newaxis, :, :])
        
        shifted_img = np.fft.ifftshift(img_corrupted, axes=(-2, -1))
        kspace_corrupted = np.fft.fft2(shifted_img, axes=(-2, -1))
        kspace_corrupted = np.fft.fftshift(kspace_corrupted, axes=(-2, -1))

    # 2. Apply Motion (Position Shifts in K-Space Domain)
    if p_motion > 0:
        num_events = rng.integers(1, 4)
        y_coords = np.arange(H) - H // 2
        x_coords = np.arange(W) - W // 2
        
        max_shift = 12.0
        
        for _ in range(num_events):
            start_line = rng.integers(0, H - 5)
            end_line = rng.integers(start_line + 5, H)
            
            dx = p_motion * rng.uniform(-max_shift, max_shift)
            dy = p_motion * rng.uniform(-max_shift, max_shift)
            
            ramp_y = y_coords[start_line:end_line, np.newaxis] * dy / H
            ramp_x = x_coords[np.newaxis, :] * dx / W
            phase_ramp = np.exp(-2j * np.pi * (ramp_y + ramp_x))
            
            kspace_corrupted[:, start_line:end_line, :] *= phase_ramp[np.newaxis, :, :]

    # 3. Apply Rician Noise
    if p_noise > 0:
        vol_std = np.std(np.abs(kspace_corrupted))
        max_noise = 0.15
        noise_std = p_noise * max_noise * vol_std
        noise = (rng.normal(0, noise_std, kspace_corrupted.shape) + 
                 1j * rng.normal(0, noise_std, kspace_corrupted.shape)) / np.sqrt(2)
        kspace_corrupted += noise
        
    return kspace_corrupted

def reconstruct_kspace_numpy(kspace: np.ndarray) -> np.ndarray:
    shifted_k = np.fft.ifftshift(kspace, axes=(-2, -1))
    img_c = np.fft.ifft2(shifted_k, axes=(-2, -1))
    coil_images = np.fft.fftshift(img_c, axes=(-2, -1))
    
    magnitude = np.sqrt(np.sum(np.abs(coil_images)**2, axis=0) + 1e-8)
    
    mean = np.mean(magnitude)
    std = np.std(magnitude)
    magnitude = (magnitude - mean) / (std + 1e-8)
    
    return magnitude

def process_patient(args):
    patient_idx, filepath, is_train = args
    try:
        vol = np.load(filepath) # [8, coils, H, W] complex
        num_slices = vol.shape[0]
        
        patient_images = []
        patient_targets = []
        patient_contrasts = []
        
        for slice_idx in range(num_slices):
            kspace_slice = vol[slice_idx]
            contrast_type = 0 if slice_idx == 0 else 1  # 0: T1, 1: T2
            
            # Replicating dataset sampling logic
            if not is_train:
                # Deterministic corruptions for validation
                rng = np.random.default_rng(seed=patient_idx * 8 + slice_idx + 42)
                p_noise = rng.uniform(0.0, 1.0)
                p_motion = rng.uniform(0.0, 1.0)
                p_phase = rng.uniform(0.0, 1.0)
                
                if rng.uniform(0.0, 1.0) < 0.15: p_noise = 0.0
                if rng.uniform(0.0, 1.0) < 0.15: p_motion = 0.0
                if rng.uniform(0.0, 1.0) < 0.15: p_phase = 0.0
            else:
                # Random training corruptions
                rng = np.random.default_rng()
                p_noise = rng.uniform(0.0, 1.0)
                p_motion = rng.uniform(0.0, 1.0)
                p_phase = rng.uniform(0.0, 1.0)
                
                if rng.uniform(0.0, 1.0) < 0.15: p_noise = 0.0
                if rng.uniform(0.0, 1.0) < 0.15: p_motion = 0.0
                if rng.uniform(0.0, 1.0) < 0.15: p_phase = 0.0
                
            corrupted = corrupt_kspace_slice_numpy(kspace_slice, p_noise, p_motion, p_phase, rng=rng)
            mag_img = reconstruct_kspace_numpy(corrupted).astype(np.float32)
            
            patient_images.append(mag_img)
            patient_targets.append([p_noise, p_motion, p_phase])
            patient_contrasts.append(contrast_type)
            
        return patient_idx, np.stack(patient_images), np.array(patient_targets, dtype=np.float32), np.array(patient_contrasts, dtype=np.int64)
    except Exception as e:
        print(f"Error processing patient {patient_idx}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Pre-reconstruct corrupted K-space into magnitude dataset.")
    parser.add_argument("--data_dir", type=str, required=True, help="Data directory containing dataset_manifest.csv")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory to save recon arrays")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    manifest_csv = os.path.join(args.data_dir, "dataset_manifest.csv")
    
    samples = []
    with open(manifest_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append({
                'patient_index': int(row['patient_index']),
                'filename': row['filename']
            })
            
    num_patients = len(samples)
    print(f"Pre-reconstructing {num_patients} patients ({num_patients * 8} slices)...")
    
    # We will build train and validation split exactly as in train_anomaly_detector.py
    import random
    patient_indices = list(range(num_patients))
    random.Random(42).shuffle(patient_indices)
    num_train = int(0.8 * num_patients)
    train_set_indices = set(patient_indices[:num_train])
    
    worker_args = []
    for s in samples:
        idx = s['patient_index']
        filepath = os.path.join(args.data_dir, s['filename'])
        is_train = idx in train_set_indices
        worker_args.append((idx, filepath, is_train))
        
    workers = cpu_count()
    print(f"Processing in parallel using {workers} processes...")
    
    # Pre-allocate containers
    # Shape: [num_patients, 8, 256, 256]
    images_all = np.zeros((num_patients, 8, 256, 256), dtype=np.float32)
    targets_all = np.zeros((num_patients, 8, 3), dtype=np.float32)
    contrasts_all = np.zeros((num_patients, 8), dtype=np.int64)
    
    completed = 0
    with Pool(processes=workers) as pool:
        for res in pool.imap_unordered(process_patient, worker_args):
            if res is not None:
                idx, imgs, targs, contrs = res
                images_all[idx] = imgs
                targets_all[idx] = targs
                contrasts_all[idx] = contrs
                
            completed += 1
            if completed % 100 == 0 or completed == num_patients:
                print(f"Processed {completed}/{num_patients} patients...")
                
    # Flatten slices dimension
    images_flat = images_all.reshape(-1, 256, 256)
    targets_flat = targets_all.reshape(-1, 3)
    contrasts_flat = contrasts_all.reshape(-1)
    
    print("Saving arrays to disk...")
    np.save(os.path.join(args.output_dir, "recon_images.npy"), images_flat)
    np.save(os.path.join(args.output_dir, "recon_targets.npy"), targets_flat)
    np.save(os.path.join(args.output_dir, "recon_contrasts.npy"), contrasts_flat)
    print("Success! Pre-reconstruction complete.")

if __name__ == "__main__":
    main()
