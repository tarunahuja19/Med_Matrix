import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
from fused_model_onnx import FusedS4CNNClassifierONNX

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    checkpoint_path = os.path.join(base_dir, "fused_model.pt")
    onnx_path = os.path.join(base_dir, "fused_model.onnx")

    print("Loading PyTorch checkpoint from:", checkpoint_path)
    state_dict = torch.load(checkpoint_path, map_location='cpu')

    C = 16
    H = 256
    W = 256
    S = 8
    
    print(f"Instantiating ONNX-compatible model (S={S}, C={C}, H={H}, W={W})")
    onnx_model = FusedS4CNNClassifierONNX(
        d_model_s4=128,
        d_state_s4=16,
        n_layers_s4=2,
        d_model_cnn=128,
        num_classes=11,
        input_dim_s4=C * H * W,
        d_attn=128,
        resolution=H
    )

    # Load weights
    onnx_model.load_state_dict(state_dict, strict=False)
    onnx_model.eval()

    # Dummy input representing shape [B=1, S=8, C=16, H=256, W=256, 2]
    dummy_input = torch.randn(1, S, C, H, W, 2, dtype=torch.float32)

    # Trace model
    print("Tracing model with TorchScript...")
    traced_model = torch.jit.trace(onnx_model, dummy_input)
    print("Model traced successfully!")

    print(f"Exporting model to ONNX at {onnx_path}...")
    torch.onnx.export(
        traced_model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        dynamo=False, # Explicitly use legacy TorchScript exporter
        input_names=['kspace_real_imag'],
        output_names=['logits'],
        dynamic_axes={
            'kspace_real_imag': {0: 'batch_size'},
            'logits': {0: 'batch_size'}
        }
    )
    print("ONNX Model successfully exported via legacy TorchScript exporter!")

if __name__ == "__main__":
    main()
