import os
import argparse
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Import our anomaly detector model
from anomaly_detector_model import KSpaceAnomalyEstimator

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


class KSpaceAnomalyDataset(Dataset):
    """
    Dataset that loads raw patient-level complex K-space volumes,
    applies simulated corruptions on the fly for each of the 8 slices,
    normalizes the K-space, and returns the stacked real/imaginary parts.
    """
    def __init__(self, samples: list, data_dir: str, coils: int = 16, is_train: bool = True, preload: bool = False):
        self.samples = samples
        self.data_dir = data_dir
        self.coils = coils
        self.is_train = is_train
        self.preload = preload
        
        if self.preload:
            print(f"Preloading {len(self.samples)} patient volumes into memory...")
            self.preloaded_volumes = []
            for i, s in enumerate(self.samples):
                filepath = os.path.join(self.data_dir, s['filename'])
                vol = np.load(filepath) # [8, coils, H, W] complex
                self.preloaded_volumes.append(vol)
                if (i + 1) % max(1, len(self.samples) // 5) == 0 or i + 1 == len(self.samples):
                    print(f"  - Preloaded {i+1}/{len(self.samples)} patients...")
            print("Preloading complete.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        patient_idx = sample['patient_index']
        
        # 1. Load patient complex volume [8, coils, H, W]
        if self.preload:
            vol = self.preloaded_volumes[idx].copy()
        else:
            filepath = os.path.join(self.data_dir, sample['filename'])
            vol = np.load(filepath)
            
        num_slices = vol.shape[0]
        coils = vol.shape[1]
        H, W = vol.shape[2], vol.shape[3]
        
        stacked_slices = []
        contrasts = []
        targets = []
        
        # 2. Corrupt each slice
        for s in range(num_slices):
            kspace_slice = vol[s] # [coils, H, W]
            contrast_type = 0 if s == 0 else 1  # 0: T1, 1: T2
            
            # Replicating corruption parameter selection
            if not self.is_train:
                # Deterministic corruptions for validation based on patient/slice index
                rng = np.random.default_rng(seed=patient_idx * 8 + s + 42)
                p_noise = rng.uniform(0.0, 1.0)
                p_motion = rng.uniform(0.0, 1.0)
                p_phase = rng.uniform(0.0, 1.0)
                
                if rng.uniform(0.0, 1.0) < 0.15: p_noise = 0.0
                if rng.uniform(0.0, 1.0) < 0.15: p_motion = 0.0
                if rng.uniform(0.0, 1.0) < 0.15: p_phase = 0.0
            else:
                rng = np.random.default_rng()
                p_noise = rng.uniform(0.0, 1.0)
                p_motion = rng.uniform(0.0, 1.0)
                p_phase = rng.uniform(0.0, 1.0)
                
                if rng.uniform(0.0, 1.0) < 0.15: p_noise = 0.0
                if rng.uniform(0.0, 1.0) < 0.15: p_motion = 0.0
                if rng.uniform(0.0, 1.0) < 0.15: p_phase = 0.0
                
            corrupted = corrupt_kspace_slice_numpy(kspace_slice, p_noise, p_motion, p_phase, rng=rng)
            
            # Standard deviation normalization based on magnitude
            norm_factor = np.std(np.abs(corrupted))
            if norm_factor > 0:
                corrupted = corrupted / norm_factor
                
            # Stack real and imaginary parts: shape [2 * coils, H, W]
            x_real = np.real(corrupted)
            x_imag = np.imag(corrupted)
            x_stacked = np.concatenate([x_real, x_imag], axis=0) # [2 * coils, H, W]
            
            stacked_slices.append(x_stacked)
            contrasts.append(contrast_type)
            targets.append([p_noise, p_motion, p_phase])
            
        # Return patient volume representation:
        # stacked_slices: [8, 2 * coils, H, W]
        # contrasts: [8]
        # targets: [8, 3]
        return (
            torch.from_numpy(np.stack(stacked_slices)).float(),
            torch.tensor(contrasts, dtype=torch.long),
            torch.tensor(targets, dtype=torch.float32)
        )


def get_status(val: float):
    """Calibrates threshold logic to map estimates to status reports."""
    if val > 0.5:
        return "SEVERE", "Rescan required before patient leaves"
    elif val > 0.15:
        return "MILD", "Acceptable, flag for radiologist awareness"
    else:
        return "NONE", "Acceptable"


def print_patient_report(patient_idx: int, preds: np.ndarray, targets: np.ndarray):
    """
    Prints a clinical anomaly report for a single patient (8 slices).
    - Slice 0 is T1
    - Slices 1-7 are T2 (using max severity aggregation)
    """
    t1_preds = preds[0]
    t1_targets = targets[0]
    
    t2_preds = preds[1:]
    t2_targets = targets[1:]
    
    t2_preds_max = np.max(t2_preds, axis=0)
    t2_targets_max = np.max(t2_targets, axis=0)
    
    print(f"\n========================================================")
    print(f"=== CLINICAL ANOMALY DIAGNOSTIC REPORT: PATIENT {patient_idx:04d} ===")
    print(f"========================================================")
    
    # T1 Section
    t1_n_class, t1_n_act = get_status(t1_preds[0])
    t1_m_class, t1_m_act = get_status(t1_preds[1])
    t1_p_class, t1_p_act = get_status(t1_preds[2])
    
    # Aggregate T1
    t1_acts = [t1_n_act, t1_m_act, t1_p_act]
    if "Rescan required before patient leaves" in t1_acts:
        t1_overall_act = "Rescan required before patient leaves"
        t1_overall_sev = "SEVERE"
    elif "Acceptable, flag for radiologist awareness" in t1_acts:
        t1_overall_act = "Acceptable, flag for radiologist awareness"
        t1_overall_sev = "MILD"
    else:
        t1_overall_act = "Acceptable"
        t1_overall_sev = "NONE"
        
    print(f"T1-weighted scan (Slice 0):")
    print(f"  - Noise:  {t1_n_class:<7} (est: {t1_preds[0]:.2f}, gt: {t1_targets[0]:.2f}) -> \"{t1_n_act}\"")
    print(f"  - Motion: {t1_m_class:<7} (est: {t1_preds[1]:.2f}, gt: {t1_targets[1]:.2f}) -> \"{t1_m_act}\"")
    print(f"  - Phase:  {t1_p_class:<7} (est: {t1_preds[2]:.2f}, gt: {t1_targets[2]:.2f}) -> \"{t1_p_act}\"")
    print(f"  >> Overall T1 Status: {t1_overall_sev} -> \"{t1_overall_act}\"\n")
    
    # T2 Section
    t2_n_class, t2_n_act = get_status(t2_preds_max[0])
    t2_m_class, t2_m_act = get_status(t2_preds_max[1])
    t2_p_class, t2_p_act = get_status(t2_preds_max[2])
    
    # Aggregate T2
    t2_acts = [t2_n_act, t2_m_act, t2_p_act]
    if "Rescan required before patient leaves" in t2_acts:
        t2_overall_act = "Rescan required before patient leaves"
        t2_overall_sev = "SEVERE"
    elif "Acceptable, flag for radiologist awareness" in t2_acts:
        t2_overall_act = "Acceptable, flag for radiologist awareness"
        t2_overall_sev = "MILD"
    else:
        t2_overall_act = "Acceptable"
        t2_overall_sev = "NONE"
        
    print(f"T2-weighted scan (Slices 1-7, worst case):")
    print(f"  - Noise:  {t2_n_class:<7} (est max: {t2_preds_max[0]:.2f}, gt max: {t2_targets_max[0]:.2f}) -> \"{t2_n_act}\"")
    print(f"  - Motion: {t2_m_class:<7} (est max: {t2_preds_max[1]:.2f}, gt max: {t2_targets_max[1]:.2f}) -> \"{t2_m_act}\"")
    print(f"  - Phase:  {t2_p_class:<7} (est max: {t2_preds_max[2]:.2f}, gt max: {t2_targets_max[2]:.2f}) -> \"{t2_p_act}\"")
    print(f"  >> Overall T2 Status: {t2_overall_sev} -> \"{t2_overall_act}\"")
    print(f"========================================================\n")


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_work_dir = "/kaggle/working" if os.path.exists("/kaggle/working") else base_dir
    default_data_dir = os.path.join(default_work_dir, "data", "synthetic_kspace")
    default_onnx_path = os.path.join(default_work_dir, "anomaly_detector.onnx")
    default_plot_path = os.path.join(default_work_dir, "anomaly_training_results.png")
    default_checkpoint_path = os.path.join(default_work_dir, "anomaly_detector.pt")
    
    parser = argparse.ArgumentParser(description="Train K-space MRI Anomaly Estimation and Classification model directly on K-space.")
    parser.add_argument("--data_dir", type=str, default=default_data_dir, help="Directory containing dataset_manifest.csv")
    parser.add_argument("--coils", type=int, default=16, help="Number of coils in K-space")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (number of patients per batch)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--onnx_path", type=str, default=default_onnx_path, help="ONNX output model path")
    parser.add_argument("--plot_path", type=str, default=default_plot_path, help="Validation performance plot path")
    parser.add_argument("--checkpoint_path", type=str, default=default_checkpoint_path, help="Model checkpoint path")
    parser.add_argument("--no_preload", action="store_true", help="Disable preloading dataset into memory")
    parser.add_argument("--limit_samples", type=int, default=250, help="Limit dataset size for quick testing")
    parser.add_argument("--d_model", type=int, default=64, help="Model latent dimension")
    args = parser.parse_args()

    # Verify manifest exists
    manifest_csv = os.path.join(args.data_dir, "dataset_manifest.csv")
    if not os.path.exists(manifest_csv):
        raise FileNotFoundError(f"Manifest not found at {manifest_csv}. Please generate data first.")

    # 1. Load manifest and setup samples
    samples = []
    with open(manifest_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append({
                'patient_index': int(row['patient_index']),
                'filename': row['filename']
            })
            
    if args.limit_samples is not None:
        import random
        random.Random(42).shuffle(samples)
        samples = samples[:args.limit_samples]
        
    num_patients = len(samples)
    print(f"Loaded: {num_patients} patients ({num_patients * 8} slices)")
    
    # Train/validation split (80% train, 20% validation)
    import random
    patient_indices = list(range(num_patients))
    random.Random(42).shuffle(patient_indices)
    num_train = int(0.8 * num_patients)
    
    train_indices = patient_indices[:num_train]
    val_indices = patient_indices[num_train:]
    
    train_samples = [samples[i] for i in train_indices]
    val_samples = [samples[i] for i in val_indices]
    
    # Instantiate dataset classes
    preload = not args.no_preload
    train_dataset = KSpaceAnomalyDataset(train_samples, args.data_dir, coils=args.coils, is_train=True, preload=preload)
    val_dataset = KSpaceAnomalyDataset(val_samples, args.data_dir, coils=args.coils, is_train=False, preload=preload)
    
    # Fast parallel loader for CPU/GPU data transfer (RAM -> GPU)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    print(f"Dataset Split:")
    print(f"  - Train Patients: {len(train_dataset)} ({len(train_dataset) * 8} slices)")
    print(f"  - Validation Patients: {len(val_dataset)} ({len(val_dataset) * 8} slices)")
    
    # 2. Model & Optimizer
    sample_img, _, _ = train_dataset[0]
    resolution = sample_img.shape[3]
    print(f"Dynamic Resolution Detected: {resolution}x{resolution}")
    model = KSpaceAnomalyEstimator(coils=args.coils, resolution=resolution, d_model=args.d_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training...")
        model = nn.DataParallel(model)
    model = model.to(device)
    
    # MSE loss is ideal for continuous regression outputs
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 3. Training Loop
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_noise_mse': [],
        'val_motion_mse': [],
        'val_phase_mse': []
    }
    
    print("\nStarting training loop...")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for x, contrast, targets in train_loader:
            B_patient, S, C2, H, W = x.shape
            # Flatten patients and slices dimension to fit model input
            x = x.view(B_patient * S, C2, H, W)
            contrast = contrast.view(B_patient * S)
            targets = targets.view(B_patient * S, 3)
            
            x, contrast, targets = x.to(device), contrast.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(x, contrast)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * x.size(0)
            
        train_loss /= (len(train_dataset) * 8)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_targets_all = []
        val_preds_all = []
        
        with torch.no_grad():
            for x, contrast, targets in val_loader:
                B_patient, S, C2, H, W = x.shape
                x = x.view(B_patient * S, C2, H, W)
                contrast = contrast.view(B_patient * S)
                targets = targets.view(B_patient * S, 3)
                
                x, contrast, targets = x.to(device), contrast.to(device), targets.to(device)
                outputs = model(x, contrast)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * x.size(0)
                
                val_targets_all.append(targets.cpu().numpy())
                val_preds_all.append(outputs.cpu().numpy())
                
        val_loss /= (len(val_dataset) * 8)
        val_targets_all = np.concatenate(val_targets_all, axis=0)
        val_preds_all = np.concatenate(val_preds_all, axis=0)
        
        # Compute MSE per parameter
        mses = np.mean((val_preds_all - val_targets_all)**2, axis=0)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_noise_mse'].append(mses[0])
        history['val_motion_mse'].append(mses[1])
        history['val_phase_mse'].append(mses[2])
        
        print(f"Epoch {epoch+1:02d}/{args.epochs:02d} | "
              f"Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | "
              f"MSE - Noise: {mses[0]:.4f}, Motion: {mses[1]:.4f}, Phase: {mses[2]:.4f}")
        
        scheduler.step()
        
    print("Training complete! Saving checkpoint...")
    state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save(state_dict, args.checkpoint_path)
    print(f"Model saved to {args.checkpoint_path}")
    
    # 4. Generate Final Reports & Metric Assessments
    val_targets_class = np.zeros_like(val_targets_all, dtype=np.int32)
    val_preds_class = np.zeros_like(val_preds_all, dtype=np.int32)
    
    # Classify threshold: Severe (>0.5) is 2, Mild (>0.15) is 1, None <= 0.15 is 0
    val_targets_class[val_targets_all > 0.15] = 1
    val_targets_class[val_targets_all > 0.5] = 2
    
    val_preds_class[val_preds_all > 0.15] = 1
    val_preds_class[val_preds_all > 0.5] = 2
    
    noise_acc = np.mean(val_targets_class[:, 0] == val_preds_class[:, 0])
    motion_acc = np.mean(val_targets_class[:, 1] == val_preds_class[:, 1])
    phase_acc = np.mean(val_targets_class[:, 2] == val_preds_class[:, 2])
    
    print("\n=================== CLASSIFICATION ACCURACY ===================")
    print(f"Noise Severity Classification Accuracy:  {noise_acc*100:.2f}%")
    print(f"Motion Severity Classification Accuracy: {motion_acc*100:.2f}%")
    print(f"Phase Severity Classification Accuracy:  {phase_acc*100:.2f}%")
    print("================================================================\n")
    
    # Print diagnostic reports for first 2 patients in validation set
    num_val_patients = len(val_samples)
    for p in range(min(2, num_val_patients)):
        # Extract slices index for validation patient p
        p_slice_preds = val_preds_all[p*8 : (p+1)*8]
        p_slice_gts = val_targets_all[p*8 : (p+1)*8]
        print_patient_report(val_samples[p]['patient_index'], p_slice_preds, p_slice_gts)
        
    # 5. Plot training Curves
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        epochs_range = range(1, args.epochs + 1)
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, history['train_loss'], label='Train MSE Loss')
        plt.plot(epochs_range, history['val_loss'], label='Val MSE Loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('Overall Training & Validation Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, history['val_noise_mse'], label='Noise Parameter MSE')
        plt.plot(epochs_range, history['val_motion_mse'], label='Motion Parameter MSE')
        plt.plot(epochs_range, history['val_phase_mse'], label='Phase Parameter MSE')
        plt.xlabel('Epoch')
        plt.ylabel('Parameter MSE')
        plt.title('Individual Corruption Parameter Errors')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(args.plot_path)
        print(f"Training plots saved to {args.plot_path}")
    except Exception as e:
        print(f"Failed to plot validation performance: {e}")
        
    # 6. Export to ONNX
    print("\nExporting model to ONNX...")
    onnx_model = KSpaceAnomalyEstimator(coils=args.coils, resolution=resolution, d_model=args.d_model)
    onnx_model.load_state_dict(state_dict)
    onnx_model.eval()
    
    # Trace model with raw k-space shape [1, 2 * coils, resolution, resolution]
    dummy_img = torch.randn(1, 2 * args.coils, resolution, resolution, dtype=torch.float32)
    dummy_contrast = torch.tensor([0], dtype=torch.long)
    
    try:
        traced_model = torch.jit.trace(onnx_model, (dummy_img, dummy_contrast))
        torch.onnx.export(
            traced_model,
            (dummy_img, dummy_contrast),
            args.onnx_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
            input_names=['kspace', 'contrast'],
            output_names=['predictions'],
            dynamic_axes={
                'kspace': {0: 'batch_size'},
                'contrast': {0: 'batch_size'},
                'predictions': {0: 'batch_size'}
            }
        )
        print(f"ONNX Model successfully exported to {args.onnx_path}!")
    except Exception as e:
        print(f"Failed to export ONNX model: {e}")


if __name__ == "__main__":
    main()
