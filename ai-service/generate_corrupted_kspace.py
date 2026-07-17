import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

# Local imports from the project
from generate_synthetic_dataset import generate_quantitative_maps, simulate_mri_signal, generate_coil_sensitivities
from pre_reconstruct_dataset import corrupt_kspace_slice_numpy
from reconstruction import reconstruct_kspace
from anomaly_detector_model import KSpaceAnomalyEstimator

def generate_and_corrupt(
    p_noise: float,
    p_motion: float,
    p_phase: float,
    pathology_id: int,
    difficulty: float,
    output_path: str,
    visualize_path: str
):
    slices = 8
    coils = 16
    resolution = 256
    
    print("\n" + "="*60)
    print("STEP 1: Generating clean brain phantom K-space volume...")
    print("="*60)
    print(f"Dimensions: {slices} slices, {coils} coils, {resolution}x{resolution} resolution")
    print(f"Pathology: Category {pathology_id} (difficulty={difficulty:.2f})")
    
    clean_kspace = np.zeros((slices, coils, resolution, resolution), dtype=np.complex64)
    sens_maps = generate_coil_sensitivities(resolution, resolution, coils)
    
    pathology_start = slices // 4
    pathology_end = 3 * slices // 4
    
    for s in range(slices):
        if s == 0:
            TR, TE = 600.0, 15.0 # T1 contrast
        else:
            TR, TE = 3000.0, 90.0 # T2 contrast
            
        curr_cat_id = pathology_id if (s >= pathology_start and s < pathology_end) else 0
        curr_diff = difficulty if (s >= pathology_start and s < pathology_end) else 0.0
        
        t1, t2, pd, t2dash, b0 = generate_quantitative_maps(resolution, s, slices, curr_cat_id, curr_diff)
        img_slice = simulate_mri_signal(t1, t2, pd, t2dash, b0, TR, TE)
        
        for c in range(coils):
            coil_img = img_slice * sens_maps[c]
            # Center K-space (FFT shift)
            coil_kspace = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(coil_img)))
            clean_kspace[s, c, :, :] = coil_kspace.astype(np.complex64)

    # Standard complex Gaussian background noise (around 30 dB SNR) added to the clean k-space base
    vol_std = np.std(np.abs(clean_kspace))
    bg_noise = (np.random.normal(0, 0.03 * vol_std, clean_kspace.shape) + 
                1j * np.random.normal(0, 0.03 * vol_std, clean_kspace.shape)) / np.sqrt(2)
    clean_kspace += bg_noise.astype(np.complex64)
    
    print(f"Clean K-space shape: {clean_kspace.shape}")
    
    print("\n" + "="*60)
    print("STEP 2: Applying simulated physical corruptions...")
    print("="*60)
    print(f"Requested parameters:")
    print(f"  - Noise Level: {p_noise:.2f}")
    print(f"  - Motion Shift: {p_motion:.2f}")
    print(f"  - Phase Error: {p_phase:.2f}")
    
    corrupted_kspace = np.zeros_like(clean_kspace)
    stacked_slices = []
    
    for s in range(slices):
        kspace_slice = clean_kspace[s]
        rng = np.random.default_rng(seed=42 + s)
        
        # Corrupt the slice using project's corruption function
        corrupted = corrupt_kspace_slice_numpy(kspace_slice, p_noise, p_motion, p_phase, rng=rng)
        corrupted_kspace[s] = corrupted
        
        # Normalize as in KSpaceAnomalyDataset
        norm_factor = np.std(np.abs(corrupted))
        if norm_factor > 0:
            corrupted_norm = corrupted / norm_factor
        else:
            corrupted_norm = corrupted
            
        x_real = np.real(corrupted_norm)
        x_imag = np.imag(corrupted_norm)
        x_stacked = np.concatenate([x_real, x_imag], axis=0) # [32, 256, 256]
        stacked_slices.append(x_stacked)
        
    np.save(output_path, corrupted_kspace)
    print(f"Successfully saved corrupted K-space to {output_path} (shape: {corrupted_kspace.shape})")
    
    print("\n" + "="*60)
    print("STEP 3: Running Estimator Model Inference...")
    print("="*60)
    checkpoint_path = "anomaly_detector.pt"
    if not os.path.exists(checkpoint_path):
        print(f"Error: {checkpoint_path} not found in current directory.")
        print("Cannot run model inference, but K-space was successfully generated and saved!")
    else:
        try:
            # Instantiate model
            model = KSpaceAnomalyEstimator(coils=coils, resolution=resolution, d_model=64)
            model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
            model.eval()
            
            # Prepare tensors
            x_tensor = torch.from_numpy(np.stack(stacked_slices)).float() # [8, 32, 256, 256]
            contrast_tensor = torch.tensor([0 if s == 0 else 1 for s in range(slices)], dtype=torch.long) # [8]
            
            with torch.no_grad():
                preds = model(x_tensor, contrast_tensor).numpy() # [8, 3]
                
            print("\nModel Anomaly Severity Predictions per Slice (range [0, 1]):")
            print(f"{'Slice':<8}{'Noise (Est / GT)':<22}{'Motion (Est / GT)':<22}{'Phase (Est / GT)':<22}")
            print("-" * 75)
            
            for s in range(slices):
                print(f"Slice {s}:  "
                      f"{preds[s, 0]:.3f} / {p_noise:.2f}     "
                      f"{preds[s, 1]:.3f} / {p_motion:.2f}     "
                      f"{preds[s, 2]:.3f} / {p_phase:.2f}")
                      
            # Aggregate predictions (Max severity across slices)
            max_preds = np.max(preds, axis=0)
            print("-" * 75)
            print(f"Aggregated Volume Severity (Max):")
            print(f"  - Noise Severity:  {max_preds[0]:.3f} (GT: {p_noise:.2f})")
            print(f"  - Motion Severity: {max_preds[1]:.3f} (GT: {p_motion:.2f})")
            print(f"  - Phase Severity:  {max_preds[2]:.3f} (GT: {p_phase:.2f})")
            
        except Exception as e:
            print(f"Failed during model inference: {e}")
            
    print("\n" + "="*60)
    print("STEP 4: Reconstructing magnitude images & plotting...")
    print("="*60)
    print("Reconstructing clean volume...")
    recon_clean = reconstruct_kspace(clean_kspace, phase_correction=True)
    print("Reconstructing corrupted volume...")
    recon_corrupted = reconstruct_kspace(corrupted_kspace, phase_correction=True)
    
    # We will plot slice 4 (middle slice) which has the injected pathology
    plot_slice_idx = 4
    
    img_clean = recon_clean[plot_slice_idx]
    img_corr = recon_corrupted[plot_slice_idx]
    img_diff = np.abs(img_clean - img_corr)
    
    # Create matplotlib plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    im0 = axes[0].imshow(img_clean, cmap='gray')
    axes[0].set_title(f"Clean Reconstruction (Slice {plot_slice_idx})")
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    
    im1 = axes[1].imshow(img_corr, cmap='gray')
    axes[1].set_title(f"Corrupted Reconstruction (Slice {plot_slice_idx})\nNoise={p_noise:.1f}, Motion={p_motion:.1f}, Phase={p_phase:.1f}")
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    im2 = axes[2].imshow(img_diff, cmap='hot')
    axes[2].set_title("Absolute Difference Map")
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.suptitle("MRI Reconstruction Corruption Comparison", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(visualize_path, dpi=150)
    print(f"Saved comparison visualization plot to {visualize_path}")
    print("="*60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate corrupted kspace from physical models and test with anomaly estimator.")
    parser.add_argument("--noise", type=float, default=0.4, help="Target noise level [0, 1]")
    parser.add_argument("--motion", type=float, default=0.5, help="Target motion shift level [0, 1]")
    parser.add_argument("--phase", type=float, default=0.3, help="Target phase error level [0, 1]")
    parser.add_argument("--pathology", type=int, default=10, help="Pathology Category ID (e.g. 10=Microbleeds, 1=Glioma)")
    parser.add_argument("--difficulty", type=float, default=0.3, help="Pathology difficulty / size parameter [0, 1]")
    parser.add_argument("--output", type=str, default="corrupted_kspace.npy", help="Output path for corrupted K-space")
    parser.add_argument("--visualize", type=str, default="corruption_comparison.png", help="Output path for comparison plot")
    
    args = parser.parse_args()
    generate_and_corrupt(
        p_noise=args.noise,
        p_motion=args.motion,
        p_phase=args.phase,
        pathology_id=args.pathology,
        difficulty=args.difficulty,
        output_path=args.output,
        visualize_path=args.visualize
    )
