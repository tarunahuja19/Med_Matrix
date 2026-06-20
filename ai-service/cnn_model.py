import torch
import torch.nn as nn
import torch.fft

class VolumeCNNClassifier(nn.Module):
    """
    3D CNN Volumetric Classifier for raw complex K-space data.
    1. Reconstructs complex K-space to magnitude images on the fly.
    2. Applies 3D Convolutional layers to classify the volume.
    """
    def __init__(self, num_classes: int = 11):
        super().__init__()
        
        # Volumetric 3D CNN layers
        # Input shape to Conv3D: [B, 1, slices, height, width] = [B, 1, 64, 128, 128]
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=16, kernel_size=3, stride=(1, 2, 2), padding=1)
        self.bn1 = nn.BatchNorm3d(16)
        self.relu1 = nn.ReLU(inplace=True)
        
        # Output of Conv1: [B, 16, 64, 64, 64]
        self.conv2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=(2, 2, 2), padding=1)
        self.bn2 = nn.BatchNorm3d(32)
        self.relu2 = nn.ReLU(inplace=True)
        
        # Output of Conv2: [B, 32, 32, 32, 32]
        self.conv3 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=(2, 2, 2), padding=1)
        self.bn3 = nn.BatchNorm3d(64)
        self.relu3 = nn.ReLU(inplace=True)
        
        # Output of Conv3: [B, 64, 16, 16, 16]
        self.conv4 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=(2, 2, 2), padding=1)
        self.bn4 = nn.BatchNorm3d(128)
        self.relu4 = nn.ReLU(inplace=True)
        
        # Output of Conv4: [B, 128, 8, 8, 8]
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Classification projection
        self.head = nn.Linear(128, num_classes)
        
    def reconstruct_kspace(self, kspace: torch.Tensor) -> torch.Tensor:
        """
        Differentiable on-the-fly reconstruction of complex K-space to magnitude images.
        Input shape: [B, S, C, H, W] (complex64)
        Output shape: [B, 1, S, H, W] (float32 magnitude volume)
        """
        # 1. 2D IFFT on spatial dimensions: shift -> IFFT -> shift back
        shifted_k = torch.fft.ifftshift(kspace, dim=(-2, -1))
        img_c = torch.fft.ifft2(shifted_k, dim=(-2, -1))
        coil_images = torch.fft.fftshift(img_c, dim=(-2, -1))
        
        # 2. RSS combination along the coil channel (dim 2)
        # combined shape: [B, S, H, W]
        combined = torch.sqrt(torch.sum(torch.abs(coil_images)**2, dim=2))
        
        # 3. Add channel dimension: [B, 1, S, H, W]
        return combined.unsqueeze(1)
        
    def forward(self, x: torch.Tensor, return_features: bool = False, return_sequence: bool = False) -> torch.Tensor:
        # x is complex64 raw K-space of shape [B, S, C, H, W]
        # Reconstruct into magnitude image
        mag = self.reconstruct_kspace(x) # [B, 1, S, H, W]
        B, _, S, H_vol, W_vol = mag.shape
        
        # 3D CNN forward propagation
        x_feat = self.relu1(self.bn1(self.conv1(mag)))
        x_feat = self.relu2(self.bn2(self.conv2(x_feat)))
        x_feat = self.relu3(self.bn3(self.conv3(x_feat)))
        x_feat = self.relu4(self.bn4(self.conv4(x_feat)))
        
        if return_sequence:
            # Spatial average pool to 1x1
            x_spat_pool = nn.functional.avg_pool3d(x_feat, kernel_size=(1, x_feat.shape[-2], x_feat.shape[-1]))
            # Squeeze spatial dimensions: [B, 128, S_down]
            x_spat_pool = x_spat_pool.squeeze(-1).squeeze(-1)
            # Interpolate sequence length to S (64)
            x_seq = nn.functional.interpolate(x_spat_pool, size=S, mode='linear', align_corners=False)
            # Transpose to [B, S, 128]
            return x_seq.transpose(1, 2)
            
        # Global pooling and flattening
        x_pool = self.pool(x_feat) # [B, 128, 1, 1, 1]
        x_flat = torch.flatten(x_pool, 1) # [B, 128]
        
        if return_features:
            return x_flat
            
        # Predict logits
        logits = self.head(x_flat) # [B, num_classes]
        return logits
