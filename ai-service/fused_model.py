import torch
import torch.nn as nn
from s4_model import KSpaceS4Classifier
from cnn_model import VolumeCNNClassifier

class FusedS4CNNClassifier(nn.Module):
    """
    Hybrid Fused Volumetric Classifier via Slice-Level Cross-Attention.
    Combines:
      - S4 Frequency Branch: extracts sequence features from complex K-space directly.
      - CNN Spatial Branch: extracts spatial features from reconstructed magnitude volumes on the fly.
    Aligns both branches at slice-level, performs Cross-Attention (CNN as Query, S4 as Key/Value),
    pools the attended representation, and projects to classification logits.
    """
    def __init__(self, d_model_s4: int = 128, d_state_s4: int = 16, n_layers_s4: int = 2, d_model_cnn: int = 128, num_classes: int = 11, input_dim_s4: int = 16384, d_attn: int = 128):
        super().__init__()
        
        # 1. Frequency branch
        self.s4_branch = KSpaceS4Classifier(
            d_model=d_model_s4,
            d_state=d_state_s4,
            n_layers=n_layers_s4,
            num_classes=num_classes,
            input_dim=input_dim_s4
        )
        
        # 2. Spatial branch
        self.cnn_branch = VolumeCNNClassifier(
            num_classes=num_classes
        )
        
        # 3. Cross-Attention Projections
        self.q_proj = nn.Linear(d_model_cnn, d_attn)
        self.k_proj = nn.Linear(d_model_s4, d_attn)
        self.v_proj = nn.Linear(d_model_s4, d_attn)
        
        # Residual projection for Query (CNN)
        self.res_proj = nn.Linear(d_model_cnn, d_attn)
        
        # 4. Joint classification head
        self.head = nn.Linear(d_attn, num_classes)
        
        self.d_attn = d_attn
        
    def forward(self, x: torch.Tensor, return_attention: bool = False) -> torch.Tensor:
        # Input shape: [B, S, C, H, W] (complex64) or [B, S, C, H, W, 2] (float32)
        if x.dtype == torch.float32 and x.shape[-1] == 2:
            x = torch.complex(x[..., 0], x[..., 1])
            
        # 1. Extract slice-aligned sequences
        # z_s4 shape: [B, S, d_model_s4] (magnitude sequence)
        z_s4 = self.s4_branch(x, return_sequence=True)
        
        # z_cnn shape: [B, S, d_model_cnn] (spatial sequence tokens)
        z_cnn = self.cnn_branch(x, return_sequence=True)
        
        # 2. Compute Queries, Keys, and Values
        # Q: [B, S, d_attn]
        # K: [B, S, d_attn]
        # V: [B, S, d_attn]
        Q = self.q_proj(z_cnn)
        K = self.k_proj(z_s4)
        V = self.v_proj(z_s4)
        
        # 3. Scaled Dot-Product Attention
        # Scores: [B, S, S]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_attn ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1) # [B, S, S]
        
        # Attended representation: [B, S, d_attn]
        z_attn = torch.matmul(attn_weights, V)
        
        # 4. Residual projection of Query (CNN sequence)
        z_fused = z_attn + self.res_proj(z_cnn) # [B, S, d_attn]
        
        # 5. Global Average Pooling (GAP) along slice dimension S
        f_pool = torch.mean(z_fused, dim=1) # [B, d_attn]
        
        # 6. Classification logits
        logits = self.head(f_pool) # [B, num_classes]
        
        if return_attention:
            return logits, attn_weights
            
        return logits

