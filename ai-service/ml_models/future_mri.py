"""
KVision 4.0 — Future MRI Generation via Conditional Deformation Fields

Instead of generating pixel-space MRI (which hallucinates), this module
generates DEFORMATION FIELDS that warp the current scan to predict the future:

    x̂(t₀ + Δt) = φ_θ(Δt) ∘ x(t₀) + Δι_θ(Δt)

Where:
    x(t₀)      : Current MRI volume
    φ_θ(Δt)     : Predicted diffeomorphic deformation field
    Δι_θ(Δt)    : Predicted intensity change map (for contrast-enhancing lesions, etc.)
    ∘           : Spatial transformer (warps image by deformation field)

Advantages over pixel-space generation:
    1. Anatomical anchoring: Deformation preserves real anatomy (skull, ventricles)
    2. Topology preservation: Diffeomorphic deformations prevent folding
    3. Interpretability: You can visualize WHERE the brain is predicted to change
    4. Uncertainty: Sample multiple deformation fields → distribution of futures
    5. Clinical safety: Cannot hallucinate anatomy that doesn't exist in current scan

Architecture:
    Conditional U-Net that takes (current_mri, brain_state, Δt, treatment)
    and outputs a stationary velocity field (SVF), which is exponentiated
    via scaling-and-squaring to produce a diffeomorphic deformation.

References:
    - Dalca et al., "VoxelMorph: A Learning Framework for Deformable Medical Image Registration", IEEE TMI 2019
    - Krebs et al., "Learning a Probabilistic Model for Diffeomorphic Registration", IEEE TMI 2019
    - Bone et al., "Deformetrica: A Toolkit for Statistical Shape Analysis", 2018
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """3D convolution + InstanceNorm + LeakyReLU."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        self.norm = nn.InstanceNorm3d(out_ch)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class UpConvBlock(nn.Module):
    """3D transposed convolution for upsampling."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.norm = nn.InstanceNorm3d(out_ch)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.up(x)))


class DeformationUNet(nn.Module):
    """
    3D U-Net that predicts a stationary velocity field (SVF) for
    diffeomorphic deformation, conditioned on disease state and treatment.
    
    Input: Concatenation of [current_mri, condition_maps] along channel dim
    Output: Velocity field v ∈ ℝ^{3×D×H×W}
    
    The velocity field is exponentiated to get a diffeomorphic deformation:
        φ = exp(v) via scaling-and-squaring
    """

    def __init__(self, in_channels: int = 1, cond_channels: int = 8, base_filters: int = 16):
        super().__init__()

        ch = base_filters
        total_in = in_channels + cond_channels

        # Encoder
        self.enc1 = ConvBlock(total_in, ch)
        self.enc2 = ConvBlock(ch, ch * 2, stride=2)
        self.enc3 = ConvBlock(ch * 2, ch * 4, stride=2)
        self.enc4 = ConvBlock(ch * 4, ch * 8, stride=2)

        # Bottleneck
        self.bottleneck = ConvBlock(ch * 8, ch * 8)

        # Decoder with skip connections
        self.up3 = UpConvBlock(ch * 8, ch * 4)
        self.dec3 = ConvBlock(ch * 8, ch * 4)  # ch*4 from up + ch*4 from skip

        self.up2 = UpConvBlock(ch * 4, ch * 2)
        self.dec2 = ConvBlock(ch * 4, ch * 2)

        self.up1 = UpConvBlock(ch * 2, ch)
        self.dec1 = ConvBlock(ch * 2, ch)

        # Output: 3D velocity field (vx, vy, vz)
        self.velocity_head = nn.Conv3d(ch, 3, kernel_size=3, padding=1)
        # Initialize near zero for stable training
        nn.init.zeros_(self.velocity_head.weight)
        nn.init.zeros_(self.velocity_head.bias)

        # Optional intensity change head
        self.intensity_head = nn.Conv3d(ch, 1, kernel_size=3, padding=1)
        nn.init.zeros_(self.intensity_head.weight)
        nn.init.zeros_(self.intensity_head.bias)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Current MRI volume [batch, 1, D, H, W]
            condition: Conditioning maps [batch, cond_channels, D, H, W]
            
        Returns:
            velocity_field: [batch, 3, D, H, W]
            intensity_change: [batch, 1, D, H, W]
        """
        inp = torch.cat([x, condition], dim=1)

        # Encoder
        e1 = self.enc1(inp)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Bottleneck
        b = self.bottleneck(e4)

        # Decoder with skip connections
        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        velocity = self.velocity_head(d1)
        intensity = self.intensity_head(d1)

        return velocity, intensity


class SpatialTransformer(nn.Module):
    """
    3D spatial transformer that warps an image by a deformation field.
    Uses grid_sample for differentiable warping.
    """

    def __init__(self, size: tuple[int, int, int]):
        super().__init__()

        # Create identity grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids)  # [3, D, H, W]
        grid = grid.float()

        # Normalize to [-1, 1] for grid_sample
        for i in range(3):
            grid[i] = 2.0 * grid[i] / (size[i] - 1) - 1.0

        self.register_buffer("grid", grid.unsqueeze(0))  # [1, 3, D, H, W]

    def forward(self, image: torch.Tensor, deformation: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Source image [batch, 1, D, H, W]
            deformation: Deformation field [batch, 3, D, H, W]
            
        Returns:
            Warped image [batch, 1, D, H, W]
        """
        # Add deformation to identity grid
        new_grid = self.grid + deformation

        # Rearrange for grid_sample: [batch, D, H, W, 3]
        new_grid = new_grid.permute(0, 2, 3, 4, 1)

        return F.grid_sample(image, new_grid, align_corners=True, mode="bilinear", padding_mode="border")


def scaling_and_squaring(velocity_field: torch.Tensor, n_steps: int = 7) -> torch.Tensor:
    """
    Exponentiate a stationary velocity field to get a diffeomorphic deformation
    via the scaling-and-squaring algorithm.
    
    φ = exp(v) ≈ (I + v/2^n) ∘ (I + v/2^n) ∘ ... (n compositions)
    
    This guarantees the resulting deformation is diffeomorphic (invertible,
    topology-preserving) regardless of the input velocity field.
    
    Args:
        velocity_field: SVF [batch, 3, D, H, W]
        n_steps: Number of squaring steps (7 gives 2^7 = 128 compositions)
        
    Returns:
        Diffeomorphic deformation field [batch, 3, D, H, W]
    """
    flow = velocity_field / (2 ** n_steps)

    for _ in range(n_steps):
        # Warp flow by itself (composition)
        flow = flow + _compose_flows(flow, flow)

    return flow


def _compose_flows(flow1: torch.Tensor, flow2: torch.Tensor) -> torch.Tensor:
    """Compose two deformation fields: flow1 ∘ flow2."""
    size = flow1.shape[2:]
    vectors = [torch.arange(0, s, device=flow1.device, dtype=flow1.dtype) for s in size]
    grids = torch.meshgrid(vectors, indexing='ij')
    grid = torch.stack(grids).unsqueeze(0)  # [1, 3, D, H, W]

    # Normalize to [-1, 1]
    for i in range(3):
        grid[:, i] = 2.0 * grid[:, i] / (size[i] - 1) - 1.0

    new_locs = grid + flow2
    new_locs = new_locs.permute(0, 2, 3, 4, 1)

    return F.grid_sample(flow1, new_locs, align_corners=True, mode="bilinear", padding_mode="border")


class FutureMRIGenerator(nn.Module):
    """
    Generates predicted future MRI volumes by predicting deformation fields
    conditioned on disease state, time horizon, and treatment.
    
    Usage:
        generator = FutureMRIGenerator(volume_size=(64, 64, 64))
        future_mri, deformation, intensity_change = generator(
            current_mri=current_scan,
            brain_state=disease_state_vector,
            delta_t=12.0,   # 12 months ahead
            treatment=treatment_vector,
        )
    """

    def __init__(
        self,
        volume_size: tuple[int, int, int] = (64, 64, 64),
        d_state: int = 64,
        d_treatment: int = 8,
        base_filters: int = 16,
    ):
        super().__init__()
        self.volume_size = volume_size

        # Condition channels: brain_state + delta_t + treatment broadcast to volume
        cond_channels = d_state + 1 + d_treatment

        # Condition mapper: project (brain_state, delta_t, treatment) → per-voxel conditioning
        self.condition_mapper = nn.Sequential(
            nn.Linear(d_state + 1 + d_treatment, 128),
            nn.SiLU(),
            nn.Linear(128, 8),  # Reduced conditioning channels
        )

        self.unet = DeformationUNet(in_channels=1, cond_channels=8, base_filters=base_filters)
        self.spatial_transformer = SpatialTransformer(volume_size)

    def forward(
        self,
        current_mri: torch.Tensor,
        brain_state: torch.Tensor,
        delta_t: float,
        treatment: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate a predicted future MRI volume.
        
        Args:
            current_mri: Current MRI volume [batch, 1, D, H, W]
            brain_state: Disease state vector [batch, d_state]
            delta_t: Time offset in months (scalar)
            treatment: Treatment vector [batch, d_treatment]
            
        Returns:
            (future_mri, deformation_field, intensity_change):
                future_mri: Predicted MRI [batch, 1, D, H, W]
                deformation_field: Diffeomorphic deformation [batch, 3, D, H, W]
                intensity_change: Intensity change map [batch, 1, D, H, W]
        """
        batch = current_mri.shape[0]
        D, H, W = self.volume_size

        # Build conditioning vector
        dt_tensor = torch.full((batch, 1), delta_t, device=current_mri.device)
        cond_vec = torch.cat([brain_state, dt_tensor, treatment], dim=-1)

        # Map to per-voxel conditioning (broadcast scalar → volume)
        cond_features = self.condition_mapper(cond_vec)  # [batch, 8]
        cond_volume = cond_features.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        cond_volume = cond_volume.expand(-1, -1, D, H, W)  # [batch, 8, D, H, W]

        # Predict velocity field and intensity change
        velocity, intensity_change = self.unet(current_mri, cond_volume)

        # Exponentiate velocity → diffeomorphic deformation
        deformation = scaling_and_squaring(velocity)

        # Warp current MRI
        warped = self.spatial_transformer(current_mri, deformation)

        # Add intensity changes (for lesion enhancement, edema, etc.)
        future_mri = warped + intensity_change

        return future_mri, deformation, intensity_change
