import os
import argparse
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

# Import our models
from fused_model import FusedS4CNNClassifier
from fused_model_onnx import FusedS4CNNClassifierONNX

class KSpaceDataset(Dataset):
    def __init__(self, manifest_csv, data_dir):
        self.data_dir = data_dir
        self.samples = []
        with open(manifest_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append({
                    'filename': row['filename'],
                    'category_id': int(row['category_id'])
                })
                
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        sample = self.samples[idx]
        filepath = os.path.join(self.data_dir, sample['filename'])
        vol = np.load(filepath) # Shape: [slices, coils, height, width] (complex64)
        x = torch.from_numpy(vol)
        x_real_imag = torch.view_as_real(x) # Shape: [slices, coils, height, width, 2]
        y = torch.tensor(sample['category_id'], dtype=torch.long)
        return x_real_imag, y

def main():
    parser = argparse.ArgumentParser(description="Multi-GPU Training for K-space Fused S4-CNN Classifier.")
    parser.add_argument("--data_dir", type=str, default="/kaggle/working/data/synthetic_kspace", help="Data folder.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per training step.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--d_model_s4", type=int, default=128, help="S4 latent features.")
    parser.add_argument("--d_state_s4", type=int, default=16, help="S4 diagonal state expansion.")
    parser.add_argument("--n_layers_s4", type=int, default=2, help="S4 layers count.")
    parser.add_argument("--d_model_cnn", type=int, default=128, help="CNN features.")
    parser.add_argument("--d_attn", type=int, default=128, help="Cross-Attention dimensions.")
    parser.add_argument("--onnx_path", type=str, default="/kaggle/working/fused_model.onnx", help="Exported ONNX path.")
    parser.add_argument("--plot_path", type=str, default="/kaggle/working/training_results.png", help="Training plot path.")
    args = parser.parse_args()

    manifest_csv = os.path.join(args.data_dir, "dataset_manifest.csv")
    if not os.path.exists(manifest_csv):
        raise FileNotFoundError(f"Manifest not found at {manifest_csv}. Please generate data first.")

    # 1. Dataset & Dataloaders
    dataset = KSpaceDataset(manifest_csv, args.data_dir)
    total_len = len(dataset)
    train_len = int(0.8 * total_len)
    val_len = total_len - train_len
    
    # Deterministic split
    train_set, val_set = random_split(dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Dataset Loaded:")
    print(f"  - Total: {total_len}")
    print(f"  - Train: {train_len}")
    print(f"  - Validation: {val_len}")
    print(f"  - Device Count: {torch.cuda.device_count()}")

    # 2. Model Instantiation & GPU wrapping
    # We obtain the spatial resolution H, W from the first dataset sample
    sample_x, _ = dataset[0]
    S, C, H, W, _ = sample_x.shape
    
    model = FusedS4CNNClassifier(
        d_model_s4=args.d_model_s4,
        d_state_s4=args.d_state_s4,
        n_layers_s4=args.n_layers_s4,
        d_model_cnn=args.d_model_cnn,
        num_classes=11,
        input_dim_s4=H * W,
        d_attn=args.d_attn
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print(f"Wrapping model in nn.DataParallel over {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
        
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # 3. Training Loop
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': []
    }
    
    print("Starting Training Loop...")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            
        train_loss /= train_len
        
        # Validation evaluation
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item() * x.size(0)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == y).sum().item()
                
        val_loss /= val_len
        val_acc = correct / val_len
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc*100:.2f}%")
    print("Training finished successfully.")
    
    # Save the PyTorch checkpoint
    checkpoint_path = "/kaggle/working/fused_model.pt"
    state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save(state_dict, checkpoint_path)
    print(f"PyTorch checkpoint saved to {checkpoint_path}.")

    # 4. Save and Plot Results
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        epochs_range = range(1, args.epochs + 1)
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        color = 'tab:red'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color=color)
        ax1.plot(epochs_range, history['train_loss'], label='Train Loss', color='darkred', linestyle='dashed')
        ax1.plot(epochs_range, history['val_loss'], label='Val Loss', color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Accuracy (%)', color=color)
        ax2.plot(epochs_range, [acc * 100 for acc in history['val_acc']], label='Val Acc', color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title('Fused S4-CNN Cross-Attention Model Training Results')
        fig.tight_layout()
        plt.savefig(args.plot_path)
        print(f"Training plots saved to {args.plot_path}.")
    except Exception as e:
        print(f"Failed to plot results: {e}")

    # 5. Export to ONNX (using real-valued wrapper)
    print("Preparing ONNX model export...")
    onnx_model = FusedS4CNNClassifierONNX(
        d_model_s4=args.d_model_s4,
        d_state_s4=args.d_state_s4,
        n_layers_s4=args.n_layers_s4,
        d_model_cnn=args.d_model_cnn,
        num_classes=11,
        input_dim_s4=H * W,
        d_attn=args.d_attn,
        resolution=H
    )
    
    # Load state dict (unwrap DataParallel if needed)
    state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    onnx_model.load_state_dict(state_dict, strict=False)
    onnx_model.eval()
    
    # Dummy input representing shape [B, S, C, H, W, 2]
    # For exporting, batch size = 1 is standard
    dummy_input = torch.randn(1, S, C, H, W, 2, dtype=torch.float32)
    
    # Export
    print(f"Exporting model to ONNX at {args.onnx_path}...")
    torch.onnx.export(
        onnx_model,
        dummy_input,
        args.onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['kspace_real_imag'],
        output_names=['logits'],
        dynamic_axes={
            'kspace_real_imag': {0: 'batch_size'},
            'logits': {0: 'batch_size'}
        }
    )
    print(f"ONNX Model successfully exported to {args.onnx_path}!")

if __name__ == "__main__":
    main()
