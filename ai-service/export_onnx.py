import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
from fused_model_onnx import FusedS4CNNClassifierONNX

def main():
    print("=" * 60)
    print("ONNX EXPORT PIPELINE")
    print("=" * 60)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    checkpoint_path = os.path.join(base_dir, "fused_model.pt")
    onnx_path = os.path.join(base_dir, "fused_model.onnx")

    # ── Step 1: Load checkpoint ──
    print("\n[1/4] Loading PyTorch checkpoint...")
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    print(f"  Loaded {len(state_dict)} parameter tensors.")

    # ── Step 2: Build ONNX-compatible model ──
    print("\n[2/4] Instantiating ONNX-compatible model...")
    model = FusedS4CNNClassifierONNX(
        d_model_s4=128,
        d_state_s4=16,
        n_layers_s4=2,
        d_model_cnn=128,
        num_classes=11,
        input_dim_s4=128 * 128,
        d_attn=128,
        resolution=128
    )

    # ── Step 3: Load weights ──
    print("\n[3/4] Loading weights (strict=False to skip DFT buffers)...")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys (expected for DFT buffers): {len(missing)}")
        for k in missing[:5]:
            print(f"    - {k}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")
        for k in unexpected[:5]:
            print(f"    - {k}")
    model.eval()

    # ── Step 4: ONNX export ──
    # Dummy input: [B=1, S=64, C=1, H=128, W=128, 2]
    dummy_input = torch.randn(1, 64, 1, 128, 128, 2, dtype=torch.float32)

    # Verify forward pass works
    print("\n  Verifying forward pass...")
    with torch.no_grad():
        out = model(dummy_input)
    print(f"  Forward pass OK → output shape: {out.shape}")

    print(f"\n[4/4] Exporting to ONNX at {onnx_path}...")

    # Approach 1: Legacy TorchScript-based export (most compatible)
    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=False,
            dynamo=False,
            input_names=['kspace_real_imag'],
            output_names=['logits'],
        )
        print("\n✓ ONNX export succeeded!")
    except Exception as e:
        print(f"\n✗ Standard export failed: {e}")
        print("\nAttempting fallback: trace then legacy export...")
        traced = torch.jit.trace(model, dummy_input)
        torch.onnx.export(
            traced,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=False,
            dynamo=False,
            input_names=['kspace_real_imag'],
            output_names=['logits'],
        )
        print("\n✓ Fallback ONNX export succeeded!")

    # ── Validate ──
    try:
        import onnx
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print(f"\n✓ ONNX validation passed!")
        import os
        size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        print(f"  Model size: {size_mb:.1f} MB")
    except ImportError:
        print("\n  (onnx package not installed; skipping validation)")
    except Exception as e:
        print(f"\n⚠ ONNX validation warning: {e}")

    print("\n" + "=" * 60)
    print("EXPORT COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()

