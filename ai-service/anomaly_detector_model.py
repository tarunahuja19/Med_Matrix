import torch
import torch.nn as nn
import numpy as np

class RealDiagonalSSM(nn.Module):
    """
    Real-valued Diagonal State Space Model layer.
    Computes:
      h_k = dA * h_{k-1} + dB * x_k
      y_k = C * h_k + D * x_k
    """
    def __init__(self, d_model: int, d_state: int = 16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # A is a diagonal matrix. To guarantee stability, Re(A) must be negative.
        # We learn log(-A) where A_ii < 0.
        self.log_a = nn.Parameter(torch.log(0.5 * torch.ones(d_model, d_state)))
        self.b = nn.Parameter(torch.randn(d_model, d_state) / np.sqrt(d_state))
        self.c = nn.Parameter(torch.randn(d_model, d_state) / np.sqrt(d_state))
        self.log_dt = nn.Parameter(torch.log(torch.exp(torch.rand(d_model) * (np.log(0.1) - np.log(0.001)) + np.log(0.001))))
        self.d = nn.Parameter(torch.ones(d_model))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, S, D] (real-valued)
        B, S, D = x.shape
        device = x.device
        
        # Discretize continuous A and B
        a = -torch.exp(self.log_a) # negative A ensures stability
        dt = torch.exp(self.log_dt) # [D]
        
        # Bilinear transform (trapezoidal rule)
        dt_a = dt.unsqueeze(1) * a # [D, N]
        dA = (1.0 + 0.5 * dt_a) / (1.0 - 0.5 * dt_a) # [D, N]
        dB = (dt.unsqueeze(1) * self.b) / (1.0 - 0.5 * dt_a) # [D, N]
        
        # Recurrent loop
        h = torch.zeros(B, D, self.d_state, dtype=torch.float32, device=device)
        y = torch.zeros(B, S, D, dtype=torch.float32, device=device)
        
        dA_uns = dA.unsqueeze(0)
        dB_uns = dB.unsqueeze(0)
        c_uns = self.c.unsqueeze(0)
        d_uns = self.d.unsqueeze(0)
        
        for k in range(S):
            x_k = x[:, k, :].unsqueeze(2) # [B, D, 1]
            h = dA_uns * h + dB_uns * x_k
            y_k = torch.sum(c_uns * h, dim=2) + d_uns * x[:, k, :]
            y[:, k, :] = y_k
            
        return y


class RealSSMBlock(nn.Module):
    """
    Standard LayerNorm -> RealDiagonalSSM -> Dropout -> Linear + Residual
    """
    def __init__(self, d_model: int, d_state: int = 16, dropout: float = 0.1):
        super().__init__()
        self.ln = nn.ModuleList([nn.LayerNorm(d_model)])  # Use ModuleList for dynamic ONNX tracing compatibility if needed
        self.ln_single = nn.LayerNorm(d_model)
        self.ssm = RealDiagonalSSM(d_model, d_state=d_state)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(d_model, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x_norm = self.ln_single(x)
        x_ssm = self.ssm(x_norm)
        x_out = self.proj(self.dropout(x_ssm))
        return residual + x_out


class KSpaceAnomalyEstimator(nn.Module):
    """
    State Space Model (SSM) based MRI Anomaly Estimator.
    Processes raw multi-coil complex K-space slices row-by-row (sequence length H).
    No 2D/3D Convolutional layers are used.
    """
    def __init__(self, coils: int = 16, resolution: int = 256, d_model: int = 64, d_state: int = 16, n_layers: int = 2, embedding_dim: int = 32):
        super().__init__()
        self.coils = coils
        self.resolution = resolution
        self.d_model = d_model
        
        # Input features per step: 2 * coils * resolution
        self.input_projection = nn.Linear(2 * coils * resolution, d_model)
        
        # SSM sequential layers
        self.blocks = nn.ModuleList([
            RealSSMBlock(d_model, d_state=d_state)
            for _ in range(n_layers)
        ])
        
        # Contrast conditioning embedding
        # 0: T1, 1: T2
        self.contrast_embedding = nn.Embedding(2, embedding_dim)
        
        # Regression Head
        self.fc = nn.Sequential(
            nn.Linear(d_model + embedding_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 3)  # Output estimates for: [noise_severity, motion_severity, phase_severity]
        )

        
    def forward(self, x: torch.Tensor, contrast: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Stacked raw K-space real/imag parts, shape [B, 2 * coils, H, W]
            contrast (torch.Tensor): Contrast types, shape [B] (values 0 for T1, 1 for T2)
            
        Returns:
            torch.Tensor: Predicted continuous corruption parameters in [0, 1]^3, shape [B, 3]
        """
        B, C2, H, W = x.shape
        
        # Reshape to treat row dimension H as the sequence dimension:
        # [B, C2, H, W] -> [B, H, C2, W] -> [B, H, C2 * W]
        x_seq = x.permute(0, 2, 1, 3).reshape(B, H, C2 * W)
        
        # Project each row sequence step to latent d_model dimension
        feat = self.input_projection(x_seq) # [B, H, d_model]
        
        # Pass sequence through the diagonal State Space Model (SSM) blocks
        for block in self.blocks:
            feat = block(feat) # [B, H, d_model]
            
        # Global average pool over sequence dimension (H rows)
        feat_pooled = torch.mean(feat, dim=1) # [B, d_model]
        
        # Embed contrast condition
        contrast_emb = self.contrast_embedding(contrast) # [B, embedding_dim]
        
        # Concatenate spatial sequence features with contrast embedding
        combined = torch.cat([feat_pooled, contrast_emb], dim=1) # [B, d_model + embedding_dim]
        logits = self.fc(combined) # [B, 3]
        
        # Output bounded between [0, 1] matching parameter targets
        return torch.sigmoid(logits)


if __name__ == '__main__':
    # Simple self-test to verify tensor dimensions
    print("Testing KSpaceAnomalyEstimator (SSM) shape matches...")
    model = KSpaceAnomalyEstimator(coils=16, resolution=256)
    
    # Dummy input: Batch size 4, 2 * 16 channels, 256x256 image
    dummy_img = torch.randn(4, 32, 256, 256, dtype=torch.float32)
    dummy_contrast = torch.tensor([0, 1, 1, 0], dtype=torch.long)
    
    outputs = model(dummy_img, dummy_contrast)
    print("Output shape:", outputs.shape)
    assert outputs.shape == (4, 3)
    print("Success! Dimensions match expected shapes.")
