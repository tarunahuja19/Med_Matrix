import os
import torch
from artifact_detector import (
    get_model,
    train_model,
    generate_synthetic_dataset,
    MRIArtifactDataset
)
from torch.utils.data import DataLoader

def main():
    print("=" * 60)
    print("TRAINING ARTIFACT DETECTOR MODEL")
    print("=" * 60)
    
    # 1. Generate synthetic dataset
    print("Generating synthetic dataset (250 samples)...")
    images, labels = generate_synthetic_dataset(num_samples=250, size=256)
    
    # Split into train/val
    train_images, train_labels = images[:200], labels[:200]
    val_images, val_labels = images[200:], labels[200:]
    
    # Create datasets & dataloaders
    train_dataset = MRIArtifactDataset(train_images, train_labels)
    val_dataset = MRIArtifactDataset(val_images, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    # 2. Instantiate model (resnet18 model type is default)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Instantiating ResNet-18 ArtifactDetectorCNN...")
    model = get_model(model_type='resnet18', device=device)
    
    # 3. Train model
    save_path = os.path.join(os.path.dirname(__file__), 'artifact_detector.pth')
    print(f"Starting training (5 epochs). Best model will be saved to {save_path}...")
    
    train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        epochs=5,
        lr=1e-3,
        device=str(device),
        save_path=save_path
    )
    
    print("=" * 60)
    print(f"Success! Training finished! Model weights saved to {save_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
