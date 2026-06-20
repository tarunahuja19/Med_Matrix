import torch
import torch.nn as nn
import sys

# Add path so we can import model modules
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from fused_model import FusedS4CNNClassifier

def test_batch_size(batch_size):
    try:
        model = FusedS4CNNClassifier(
            d_model_s4=128,
            d_state_s4=16,
            n_layers_s4=2,
            d_model_cnn=128,
            num_classes=11,
            input_dim_s4=16 * 256 * 256,
            d_attn=128
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model = model.to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        # Dummy inputs representing [B, S, C, H, W, 2]
        x = torch.randn(batch_size, 8, 16, 256, 256, 2, dtype=torch.float32, device=device)
        y = torch.randint(0, 11, (batch_size,), dtype=torch.long, device=device)
        
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        # Clean up memory
        del x, y, logits, loss, model, optimizer, criterion
        torch.cuda.empty_cache()
        return True
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            return False
        else:
            raise e

def main():
    batch_sizes = [16, 32, 64, 128, 256]
    working_batch = 16
    for b in batch_sizes:
        print(f"Testing batch size {b}...")
        if test_batch_size(b):
            print(f"Batch size {b} works!")
            working_batch = b
        else:
            print(f"Batch size {b} failed due to OOM.")
            break
    print(f"Max working batch size is: {working_batch}")

if __name__ == "__main__":
    main()
