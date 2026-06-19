import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models

# Global model instance for default inference
_MODEL_INSTANCE = None


class CustomCNN(nn.Module):
    """
    A lightweight custom CNN for artifact detection on single-channel MRI images.
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # (H/2, W/2)
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # (H/4, W/4)
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # (H/8, W/8)
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # (H/16, W/16)
            
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, 3)  # Output scores for: ghosting, wrap_around, zipper_noise
        )
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class ArtifactDetectorCNN(nn.Module):
    """
    ResNet-18 based classifier adapted for 3-class multi-label MRI artifact detection.
    """
    def __init__(self, pretrained: bool = False):
        super().__init__()
        if pretrained:
            try:
                from torchvision.models import ResNet18_Weights
                self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
            except Exception:
                self.model = models.resnet18(weights=None)
        else:
            self.model = models.resnet18(weights=None)
        
        # Modify the first conv layer to accept 1 channel instead of 3
        self.model.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        # Modify the fully connected layer to output scores for 3 categories
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 3)

    def forward(self, x):
        return self.model(x)


def get_model(model_type: str = 'resnet18', weights_path: str = None, device: torch.device = None) -> nn.Module:
    """
    Instantiates the selected model architecture and optionally loads pre-trained weights.
    
    Args:
        model_type (str): 'resnet18' or 'custom'.
        weights_path (str, optional): Path to the saved weights file (.pth).
        device (torch.device, optional): Device to transfer the model to.
        
    Returns:
        nn.Module: The instantiated model.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    if model_type == 'resnet18':
        model = ArtifactDetectorCNN(pretrained=False)
    elif model_type == 'custom':
        model = CustomCNN()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    if weights_path and os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
        
    model = model.to(device)
    model.eval()
    return model


def preprocess_image(image: np.ndarray, target_size: tuple = (256, 256)) -> torch.Tensor:
    """
    Preprocesses a numpy array MRI image to a standardized PyTorch tensor of shape (1, H, W).
    Handles raw format, dimensions, normalization, and resizing.
    
    Args:
        image (np.ndarray): Input image, can be 2D, 3D (multi-channel or slice), or 4D.
        target_size (tuple): Desired (H, W) spatial dimensions.
        
    Returns:
        torch.Tensor: Normalized single-channel tensor of shape (1, target_size[0], target_size[1]).
    """
    if not isinstance(image, np.ndarray):
        image = np.asarray(image)
        
    # Ensure float32 representation
    image = image.astype(np.float32)
    
    # 1. Reduce multi-dimensional images to 2D
    if image.ndim == 3:
        # Check if first or last dimension is channel dimension
        if image.shape[0] in [1, 3]:
            image = np.mean(image, axis=0)
        elif image.shape[2] in [1, 3]:
            image = np.mean(image, axis=-1)
        else:
            # Multi-slice volume, take the middle slice
            image = image[image.shape[0] // 2]
    elif image.ndim == 4:
        # batch, channel, H, W -> take first slice
        image = image[0, 0]
    elif image.ndim != 2:
        raise ValueError(f"Unsupported image shape for preprocessing: {image.shape}")
        
    # 2. Normalize intensity to [0, 1] range
    img_min = image.min()
    img_max = image.max()
    denom = img_max - img_min
    if denom > 1e-8:
        image = (image - img_min) / denom
    else:
        image = np.zeros_like(image)
        
    # Convert to Tensor: (1, H, W)
    tensor = torch.from_numpy(image).unsqueeze(0)
    
    # 3. Resize to target size: add batch dim, interpolate, then squeeze back
    tensor = tensor.unsqueeze(0)
    tensor = nn.functional.interpolate(tensor, size=target_size, mode='bilinear', align_corners=False)
    tensor = tensor.squeeze(0)
    
    return tensor


def detect_artifacts(
    image: np.ndarray, 
    model_path: str = None, 
    model_type: str = 'resnet18', 
    device: str = None
) -> dict:
    """
    Detects artifact probabilities/scores in a single MRI image.
    
    Args:
        image (np.ndarray): Input 2D or 3D MRI image.
        model_path (str, optional): Path to loaded model weights. If None, looks for
                                    'artifact_detector.pth' in the same directory, 
                                    or initializes a fresh instance.
        model_type (str, optional): 'resnet18' or 'custom'. Default is 'resnet18'.
        device (str, optional): Device to run inference on. If None, auto-selects GPU if available.
        
    Returns:
        dict: Probabilities/scores for ghosting, wrap_around, and zipper_noise.
              e.g., { 'ghosting': float, 'wrap_around': float, 'zipper_noise': float }
    """
    global _MODEL_INSTANCE
    
    if device is None:
        device_obj = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device_obj = torch.device(device)
        
    # Determine the model source
    if model_path is not None:
        model = get_model(model_type=model_type, weights_path=model_path, device=device_obj)
    else:
        if _MODEL_INSTANCE is None:
            # Check default path
            default_weights = os.path.join(os.path.dirname(__file__), 'artifact_detector.pth')
            if os.path.exists(default_weights):
                _MODEL_INSTANCE = get_model(model_type=model_type, weights_path=default_weights, device=device_obj)
            else:
                _MODEL_INSTANCE = get_model(model_type=model_type, weights_path=None, device=device_obj)
        model = _MODEL_INSTANCE
        
    # Preprocess image
    tensor = preprocess_image(image)
    # Add batch dimension -> shape (1, 1, H, W)
    tensor = tensor.unsqueeze(0).to(device_obj)
    
    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
        
    return {
        'ghosting': float(probs[0]),
        'wrap_around': float(probs[1]),
        'zipper_noise': float(probs[2])
    }


class MRIArtifactDataset(Dataset):
    """
    PyTorch Dataset for loading MRI images and their multi-label artifact targets.
    """
    def __init__(self, images: list, labels: list, transform=None):
        """
        Args:
            images (list of np.ndarray or list of str paths): The MRI images.
            labels (list of lists/np.ndarray): Multilabel targets of shape (N, 3).
                                              Columns: [ghosting, wrap_around, zipper_noise]
            transform (callable, optional): Optional additional transforms.
        """
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if isinstance(img, str):
            if img.endswith('.npy'):
                img_data = np.load(img)
            else:
                raise ValueError(f"Unsupported image file format: {img}")
        else:
            img_data = img
            
        tensor = preprocess_image(img_data)
        
        if self.transform:
            tensor = self.transform(tensor)
            
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return tensor, label


def train_model(
    train_loader: DataLoader,
    val_loader: DataLoader = None,
    model: nn.Module = None,
    epochs: int = 5,
    lr: float = 1e-3,
    device: str = None,
    save_path: str = None
) -> nn.Module:
    """
    Trains or fine-tunes the artifact detector model.
    
    Args:
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader, optional): DataLoader for the validation set.
        model (nn.Module, optional): Model to train. If None, instantiates a default resnet18.
        epochs (int): Number of epochs.
        lr (float): Learning rate.
        device (str, optional): Device to train on.
        save_path (str, optional): Path to save the best model weights.
        
    Returns:
        nn.Module: Trained model.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    device_obj = torch.device(device)
    
    if model is None:
        model = get_model(model_type='resnet18', device=device_obj)
    else:
        model = model.to(device_obj)
        
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images = images.to(device_obj)
            labels = labels.to(device_obj)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        
        val_loss_str = ""
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device_obj)
                    labels = labels.to(device_obj)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * images.size(0)
            epoch_val_loss = val_loss / len(val_loader.dataset)
            val_loss_str = f", Val Loss: {epoch_val_loss:.4f}"
            
            # Save best model based on validation loss
            if epoch_val_loss < best_loss:
                best_loss = epoch_val_loss
                if save_path:
                    torch.save(model.state_dict(), save_path)
        else:
            # Save best model based on training loss
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                if save_path:
                    torch.save(model.state_dict(), save_path)
                    
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss:.4f}{val_loss_str}")
        
    return model


# ==========================================
# Artifact Simulation Helpers (for Training Data Generation)
# ==========================================

def add_ghosting_artifact(image: np.ndarray, num_ghosts: int = 3, intensity: float = 0.25, axis: int = 0) -> np.ndarray:
    """
    Simulates motion/phase-encode ghosting by adding shifted, reduced-intensity copies of the image.
    """
    ghosted = image.copy()
    h, w = image.shape
    step = h // (num_ghosts + 1) if axis == 0 else w // (num_ghosts + 1)
    
    for i in range(1, num_ghosts + 1):
        shift = i * step
        shifted = np.roll(image, shift, axis=axis)
        ghosted += shifted * (intensity / i)
        
    return np.clip(ghosted, 0.0, 1.0)


def add_wrap_around_artifact(image: np.ndarray, wrap_fraction: float = 0.25, axis: int = 1) -> np.ndarray:
    """
    Simulates wrap-around (aliasing) by folding over the edges of the image.
    """
    aliased = image.copy()
    size = image.shape[axis]
    wrap_pixels = int(size * wrap_fraction)
    
    if axis == 0:
        top_part = image[:wrap_pixels, :].copy()
        bottom_part = image[-wrap_pixels:, :].copy()
        aliased[-wrap_pixels:, :] += top_part
        aliased[:wrap_pixels, :] += bottom_part
    else:
        left_part = image[:, :wrap_pixels].copy()
        right_part = image[:, -wrap_pixels:].copy()
        aliased[:, -wrap_pixels:] += left_part
        aliased[:, :wrap_pixels] += right_part
        
    return np.clip(aliased, 0.0, 1.0)


def add_zipper_noise_artifact(image: np.ndarray, spike_intensity: float = 5.0, num_spikes: int = 1) -> np.ndarray:
    """
    Simulates RF zipper noise by inserting spike noise in k-space.
    """
    kspace = np.fft.fft2(image)
    kspace_shifted = np.fft.fftshift(kspace)
    
    h, w = kspace_shifted.shape
    
    for _ in range(num_spikes):
        ry = np.random.randint(0, h)
        rx = np.random.randint(0, w)
        # Avoid direct center (low frequencies) to create clean stripes
        if abs(ry - h//2) < 5 and abs(rx - w//2) < 5:
            ry = (ry + 20) % h
            rx = (rx + 20) % w
        
        kspace_shifted[ry, rx] += spike_intensity * np.max(np.abs(kspace_shifted))
        
    kspace_inverse_shifted = np.fft.ifftshift(kspace_shifted)
    corrupted = np.fft.ifft2(kspace_inverse_shifted)
    
    return np.clip(np.abs(corrupted), 0.0, 1.0)


def generate_phantom_image(size: int = 256) -> np.ndarray:
    """
    Generates a simple geometric phantom resembling a head cross-section.
    """
    image = np.zeros((size, size), dtype=np.float32)
    y, x = np.ogrid[:size, :size]
    center = size // 2
    
    # Outer skull
    r_skull_y, r_skull_x = int(size * 0.45), int(size * 0.38)
    skull_mask = ((y - center) / r_skull_y)**2 + ((x - center) / r_skull_x)**2 <= 1.0
    image[skull_mask] = 0.2
    
    # Brain tissue
    r_brain_y, r_brain_x = int(size * 0.41), int(size * 0.34)
    brain_mask = ((y - center) / r_brain_y)**2 + ((x - center) / r_brain_x)**2 <= 1.0
    image[brain_mask] = 0.6
    
    # Ventricles
    v1_mask = ((y - (center - size//8)) / (size//12))**2 + ((x - (center - size//12)) / (size//16))**2 <= 1.0
    image[v1_mask] = 0.1
    v2_mask = ((y - (center - size//8)) / (size//12))**2 + ((x - (center + size//12)) / (size//16))**2 <= 1.0
    image[v2_mask] = 0.1
    
    # Internal brain structures
    gm_mask = ((y - (center + size//8)) / (size//8))**2 + ((x - center) / (size//6))**2 <= 1.0
    image[gm_mask] = 0.8
    
    # Gaussian noise
    noise = np.random.normal(0, 0.02, size=(size, size)).astype(np.float32)
    image = np.clip(image + noise, 0.0, 1.0)
    
    return image


def generate_synthetic_dataset(num_samples: int = 100, size: int = 256) -> tuple:
    """
    Generates a synthetic dataset of MRI phantoms with varying combinations of artifacts.
    
    Returns:
        images (list of np.ndarray): List of synthetic MRI images.
        labels (list of np.ndarray): Multilabel indicators of shape (N, 3).
                                     Columns: [ghosting, wrap_around, zipper_noise]
    """
    images = []
    labels = []
    
    for _ in range(num_samples):
        img = generate_phantom_image(size=size)
        
        has_ghosting = np.random.rand() > 0.5
        has_wrap = np.random.rand() > 0.5
        has_zipper = np.random.rand() > 0.5
        
        if has_ghosting:
            axis = np.random.choice([0, 1])
            intensity = np.random.uniform(0.15, 0.35)
            img = add_ghosting_artifact(img, axis=axis, intensity=intensity)
            
        if has_wrap:
            axis = np.random.choice([0, 1])
            fraction = np.random.uniform(0.15, 0.3)
            img = add_wrap_around_artifact(img, axis=axis, wrap_fraction=fraction)
            
        if has_zipper:
            intensity = np.random.uniform(3.0, 7.0)
            img = add_zipper_noise_artifact(img, spike_intensity=intensity)
            
        images.append(img)
        labels.append(np.array([float(has_ghosting), float(has_wrap), float(has_zipper)], dtype=np.float32))
        
    return images, labels


if __name__ == '__main__':
    # Run a self-test to verify the module works
    print("Running self-test for artifact_detector...")
    
    # 1. Generate synthetic dataset
    print("Generating synthetic dataset (20 samples for quick test)...")
    images, labels = generate_synthetic_dataset(num_samples=20, size=128)
    
    # Split into train/val
    train_images, train_labels = images[:15], labels[:15]
    val_images, val_labels = images[15:], labels[15:]
    
    # Create datasets & dataloaders
    train_dataset = MRIArtifactDataset(train_images, train_labels)
    val_dataset = MRIArtifactDataset(val_images, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    # 2. Instantiate and train model (1 epoch)
    print("Instantiating CustomCNN and training for 1 epoch...")
    custom_model = get_model(model_type='custom')
    temp_weights_path = 'temp_test_model.pth'
    
    train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        model=custom_model,
        epochs=1,
        lr=1e-3,
        save_path=temp_weights_path
    )
    
    # 3. Verify detect_artifacts
    print("Testing detect_artifacts with untrained/fresh model...")
    test_img = generate_phantom_image(size=128)
    
    # Test on default model (will auto initialize/load)
    probs = detect_artifacts(test_img, model_type='custom')
    print("Probabilities on test image:", probs)
    assert 'ghosting' in probs
    assert 'wrap_around' in probs
    assert 'zipper_noise' in probs
    for k, v in probs.items():
        assert 0.0 <= v <= 1.0, f"Probability for {k} is out of bounds: {v}"
        
    # Test loading from the temporary saved weights
    print("Testing detect_artifacts loading from saved weights...")
    probs_saved = detect_artifacts(test_img, model_path=temp_weights_path, model_type='custom')
    print("Probabilities with loaded weights:", probs_saved)
    
    # Clean up temporary file
    if os.path.exists(temp_weights_path):
        os.remove(temp_weights_path)
        print(f"Cleaned up temporary weight file: {temp_weights_path}")
        
    print("Self-test completed successfully!")
