import os
import torch
from torch.utils.data import Dataset, DataLoader
from denoiser import DnCNN, train_dncnn
from artifact_detector import generate_phantom_image

class SyntheticDenoiseDataset(Dataset):
    def __init__(self, num_samples=200, size=256):
        self.num_samples = num_samples
        self.size = size
        self.data = []
        print(f"Pre-generating {num_samples} phantoms of size {size}x{size}...")
        for i in range(num_samples):
            img = generate_phantom_image(size=size)
            tensor = torch.from_numpy(img).float().unsqueeze(0)
            self.data.append(tensor)
            if (i + 1) % 50 == 0:
                print(f"  - Generated {i + 1}/{num_samples}...")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]

def main():
    print("=" * 60)
    print("TRAINING DNCNN DENOISER MODEL")
    print("=" * 60)

    # 1. Prepare Datasets & Loaders
    train_dataset = SyntheticDenoiseDataset(num_samples=160, size=256)
    val_dataset = SyntheticDenoiseDataset(num_samples=40, size=256)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # 2. Instantiate model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = DnCNN(in_channels=1, out_channels=1)

    # 3. Train
    print("Starting training (5 epochs)...")
    train_dncnn(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=5,
        lr=1e-3,
        device=str(device)
    )

    # 4. Save model weights
    save_path = os.path.join(os.path.dirname(__file__), 'dncnn.pth')
    torch.save(model.state_dict(), save_path)
    print("=" * 60)
    print(f"Success! Training finished! Model weights saved to {save_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
