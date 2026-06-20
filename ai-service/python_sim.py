import os
import numpy as np
import matplotlib.pyplot as plt

# Import from our local files
from generate_synthetic_dataset import generate_quantitative_maps, simulate_mri_signal, generate_coil_sensitivities
from reconstruction import reconstruct_kspace

def main():
    resolution = 256
    mid_slice = 0
    slices = 1
    coils = 1
    difficulty = 0.5

    # 1. Generate Microbleed K-Space in Python using exact teammate code
    print("Simulating Microbleed brain in Python...")
    t1_mb, t2_mb, pd_mb, t2dash_mb, b0_mb = generate_quantitative_maps(
        resolution, mid_slice, slices, category_id=10, difficulty=difficulty
    )
    img_slice_mb = simulate_mri_signal(t1_mb, t2_mb, pd_mb, t2dash_mb, b0_mb, TR=3000.0, TE=90.0)
    
    sens_maps = generate_coil_sensitivities(resolution, resolution, coils)
    kspace_mb = np.zeros((1, coils, resolution, resolution), dtype=np.complex128)
    for c in range(coils):
        coil_img = img_slice_mb * sens_maps[c]
        kspace_mb[0, c, :, :] = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(coil_img)))

    # Save Python raw K-space to binary file (resolution * resolution * 2 f64 values)
    kspace_flat = np.zeros((resolution * resolution * 2,), dtype=np.float64)
    kspace_flat[0::2] = kspace_mb[0, 0].real.flatten()
    kspace_flat[1::2] = kspace_mb[0, 0].imag.flatten()
    kspace_flat_path = 'kspace_python.bin'
    kspace_flat.tofile(kspace_flat_path)
    print(f"Saved Python raw k-space to {kspace_flat_path}")

    # 2. Reconstruct in Python
    print("Reconstructing in Python...")
    recon_py = reconstruct_kspace(kspace_mb, phase_correction=True)[0]
    recon_py_path = 'python_recon.bin'
    recon_py.astype(np.float64).tofile(recon_py_path)
    print(f"Saved Python reconstructed image to {recon_py_path}")

    # 3. Read Rust reconstruction output
    rust_recon_path = '../rust-mri/rust_recon.bin'
    if os.path.exists(rust_recon_path):
        # Load Rust reconstructed magnitude
        rust_recon = np.fromfile(rust_recon_path, dtype=np.float64).reshape((resolution, resolution))
        
        # Calculate maximum absolute difference
        diff = np.max(np.abs(recon_py - rust_recon))
        print("\nReconstruction Comparison:")
        print(f"  - Max Absolute Difference: {diff:.2e}")
        
        # Verify perfect match
        assert diff < 1e-10, f"Reconstruction difference too large: {diff:.2e}"
        print("Verification SUCCESS! The Rust reconstruction matches the Python reconstruction perfectly.")
        
        # Plot and save comparison
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        im0 = axes[0].imshow(recon_py, cmap='gray')
        axes[0].set_title("Python Reconstruction")
        axes[0].axis('off')
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        
        im1 = axes[1].imshow(rust_recon, cmap='gray')
        axes[1].set_title("Rust Reconstruction")
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        im2 = axes[2].imshow(np.abs(recon_py - rust_recon), cmap='hot')
        axes[2].set_title("Absolute Difference")
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        
        plt.suptitle("MRI Reconstruction Comparison: Python vs Rust", fontsize=14)
        plt.tight_layout()
        
        comparison_plot_path = '../rust_vs_python_comparison.png'
        plt.savefig(comparison_plot_path, dpi=150)
        print(f"Saved comparison plot to {comparison_plot_path}")
    else:
        print("\nRust reconstruction output not found. Please run the Rust cargo binary first.")

if __name__ == '__main__':
    main()
