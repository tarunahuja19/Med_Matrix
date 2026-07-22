import os
import sys
import numpy as np
import torch

from fused_model import FusedS4CNNClassifier
from main import get_pathology_model, PATHOLOGY_CLASSES

def main():
    print("=" * 60)
    print("VERIFYING MODEL LOADING AND INFERENCE")
    print("=" * 60)
    
    # 1. Load test K-space data
    kspace_path = os.path.join(os.path.dirname(__file__), "test_kspace.npy")
    if not os.path.exists(kspace_path):
        print(f"Error: test K-space data not found at {kspace_path}")
        sys.exit(1)
        
    kspace = np.load(kspace_path)
    print(f"Loaded test_kspace.npy: shape = {kspace.shape}, dtype = {kspace.dtype}")
    
    # 2. Load model
    print("\nLoading pathology model...")
    try:
        model = get_pathology_model()
        print("Pathology model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
        
    # 3. Preprocess
    print("\nPreprocessing input K-space data...")
    try:
        # If the test kspace is 2D or 3D, simulate slices/coils to match standard raw loading
        if kspace.ndim == 2:
            # [H, W] -> [1, 1, H, W]
            kspace = kspace[np.newaxis, np.newaxis, ...]
        elif kspace.ndim == 3:
            # [coils, H, W] -> [1, coils, H, W]
            kspace = kspace[np.newaxis, ...]
            
        slices_in, coils_in, h_in, w_in = kspace.shape
        print(f"Input shape for preprocessing: slices={slices_in}, coils={coils_in}, resolution={h_in}x{w_in}")
        
        x_complex = kspace.astype(np.complex64)
        x_tensor = torch.from_numpy(x_complex)
        
        x_real = torch.real(x_tensor)
        x_imag = torch.imag(x_tensor)
        
        # Permute to [coils, 1, slices, H, W]
        x_real_5d = x_real.permute(1, 0, 2, 3).unsqueeze(1)
        x_imag_5d = x_imag.permute(1, 0, 2, 3).unsqueeze(1)
        
        # Interpolate slices to 8 and spatial to 128x128
        real_interp = torch.nn.functional.interpolate(
            x_real_5d, size=(8, 128, 128), mode='trilinear', align_corners=False
        ).squeeze(1)
        
        imag_interp = torch.nn.functional.interpolate(
            x_imag_5d, size=(8, 128, 128), mode='trilinear', align_corners=False
        ).squeeze(1)
        
        # Adjust coils to 16
        final_real = torch.zeros(16, 8, 128, 128, dtype=torch.float32)
        final_imag = torch.zeros(16, 8, 128, 128, dtype=torch.float32)
        
        if coils_in <= 16:
            final_real[:coils_in] = real_interp
            final_imag[:coils_in] = imag_interp
        else:
            final_real = real_interp[:16]
            final_imag = imag_interp[:16]
            
        final_real = final_real.permute(1, 0, 2, 3)
        final_imag = final_imag.permute(1, 0, 2, 3)
        
        x_final = torch.complex(final_real, final_imag).unsqueeze(0)
        print(f"Preprocessed input tensor shape: {x_final.shape}")
        assert x_final.shape == (1, 8, 16, 128, 128), f"Expected shape (1, 8, 16, 128, 128), got {x_final.shape}"
        
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        sys.exit(1)
        
    # 4. Inference
    print("\nRunning inference...")
    try:
        device = next(model.parameters()).device
        x_final = x_final.to(device)
        
        with torch.no_grad():
            logits = model(x_final)
            probs = torch.softmax(logits, dim=-1).squeeze(0)
            pred_idx = int(torch.argmax(logits, dim=-1).item())
            
            predicted_class = PATHOLOGY_CLASSES[pred_idx]
            confidence = float(probs[pred_idx].item())
            
        print("\n✓ Inference completed successfully!")
        print(f"Predicted Pathology: {predicted_class}")
        print(f"Confidence: {confidence:.4f}")
        print("\nAll probabilities:")
        for idx, cls_name in enumerate(PATHOLOGY_CLASSES):
            print(f"  - {cls_name}: {probs[idx].item():.4f}")
            
    except Exception as e:
        print(f"Error during inference: {e}")
        sys.exit(1)
        
    print("=" * 60)
    print("VERIFICATION SUCCESSFUL")
    print("=" * 60)

if __name__ == "__main__":
    main()
