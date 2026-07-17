"""
KVision 4.0 — Perceiver-Style Multimodal Fusion

Fuses heterogeneous brain data modalities into a unified brain state
vector using a Perceiver-style cross-attention architecture.

Why Perceiver:
    Different modalities have radically different shapes:
    - MRI volume features: 512-dim vector from CNN/SSM encoder
    - Demographics: 5-10 scalar values (age, sex, etc.)
    - Clinical notes: 768-dim BERT embedding
    - Connectome graph: 128-dim GATv2 embedding
    - Genetic markers: variable-length binary vector
    
    Standard concatenation or averaging loses modality-specific structure.
    Perceiver uses a fixed-size learned latent array that cross-attends
    to ANY modality regardless of its shape, handling missing modalities
    gracefully via masking.

Architecture:
    1. Project each modality into a common d_latent space
    2. Concatenate all projected modality tokens into a key-value set
    3. Learned latent queries cross-attend to the modality tokens
    4. Self-attention among latent queries for inter-modality reasoning
    5. Pool the latent array to a single brain state vector

References:
    - Jaegle et al., "Perceiver: General Perception with Iterative Attention", ICML 2021
    - Jaegle et al., "Perceiver IO", ICML 2022
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ModalityProjection(nn.Module):
    """Projects a single modality into the shared latent dimension with positional encoding."""

    def __init__(self, d_input: int, d_latent: int, n_tokens: int = 1, modality_id: int = 0, n_modalities: int = 5):
        super().__init__()
        self.projection = nn.Linear(d_input, d_latent)
        self.modality_embedding = nn.Embedding(n_modalities, d_latent)
        self.modality_id = modality_id
        self.norm = nn.LayerNorm(d_latent)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Modality features [batch, d_input] or [batch, n_tokens, d_input]
        Returns:
            Projected tokens [batch, n_tokens, d_latent]
        """
        if x.ndim == 2:
            x = x.unsqueeze(1)  # [batch, 1, d_input]

        projected = self.projection(x)  # [batch, n_tokens, d_latent]

        # Add modality-type embedding
        mod_emb = self.modality_embedding(
            torch.tensor(self.modality_id, device=x.device)
        )
        projected = self.norm(projected + mod_emb)

        return projected


class CrossAttentionBlock(nn.Module):
    """
    Cross-attention: Latent queries attend to modality key-value pairs.
    
    Q = latent queries (learned, fixed size)
    K, V = concatenated modality tokens (variable size)
    """

    def __init__(self, d_latent: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_latent,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_latent)
        self.ffn = nn.Sequential(
            nn.Linear(d_latent, d_latent * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_latent * 4, d_latent),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_latent)

    def forward(
        self,
        latents: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            latents: Query tokens [batch, n_latents, d_latent]
            context: Key-value tokens [batch, n_context, d_latent]
            context_mask: Optional mask [batch, n_context] (True = ignore)
            
        Returns:
            Updated latent tokens [batch, n_latents, d_latent]
        """
        # Cross-attention
        attended, _ = self.cross_attn(
            query=latents,
            key=context,
            value=context,
            key_padding_mask=context_mask,
        )
        latents = self.norm1(latents + attended)

        # Feed-forward
        latents = self.norm2(latents + self.ffn(latents))

        return latents


class SelfAttentionBlock(nn.Module):
    """Self-attention among latent queries for inter-modality reasoning."""

    def __init__(self, d_latent: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_latent,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_latent)
        self.ffn = nn.Sequential(
            nn.Linear(d_latent, d_latent * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_latent * 4, d_latent),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_latent)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        attended, _ = self.self_attn(latents, latents, latents)
        latents = self.norm1(latents + attended)
        latents = self.norm2(latents + self.ffn(latents))
        return latents


class PerceiverFusion(nn.Module):
    """
    Perceiver-style multimodal fusion for the Digital Brain Twin.
    
    Fuses up to 5 modalities:
        0: MRI volume features (from CNN/SSM encoder)
        1: Demographics (age, sex, APOE4, etc.)
        2: Clinical text (from BioClinicalBERT)
        3: Connectome embedding (from GATv2)
        4: Genomic features (optional)
    
    Missing modalities are handled gracefully via attention masking —
    the model learns to extract maximum information from available data.
    """

    def __init__(
        self,
        d_imaging: int = 512,
        d_demographics: int = 16,
        d_clinical_text: int = 768,
        d_connectome: int = 128,
        d_genomic: int = 64,
        d_latent: int = 256,
        d_output: int = 64,
        n_latents: int = 32,
        n_cross_attn_layers: int = 2,
        n_self_attn_layers: int = 2,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_latent = d_latent
        self.n_latents = n_latents

        # Learned latent queries
        self.latent_queries = nn.Parameter(torch.randn(n_latents, d_latent) * 0.02)

        # Per-modality projection heads
        self.proj_imaging = ModalityProjection(d_imaging, d_latent, modality_id=0)
        self.proj_demographics = ModalityProjection(d_demographics, d_latent, modality_id=1)
        self.proj_clinical = ModalityProjection(d_clinical_text, d_latent, modality_id=2)
        self.proj_connectome = ModalityProjection(d_connectome, d_latent, modality_id=3)
        self.proj_genomic = ModalityProjection(d_genomic, d_latent, modality_id=4)

        # Cross-attention + Self-attention blocks
        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionBlock(d_latent, n_heads, dropout)
            for _ in range(n_cross_attn_layers)
        ])
        self.self_attn_layers = nn.ModuleList([
            SelfAttentionBlock(d_latent, n_heads, dropout)
            for _ in range(n_self_attn_layers)
        ])

        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(d_latent),
            nn.Linear(d_latent, d_output),
        )

    def forward(
        self,
        imaging_features: Optional[torch.Tensor] = None,
        demographics: Optional[torch.Tensor] = None,
        clinical_text: Optional[torch.Tensor] = None,
        connectome: Optional[torch.Tensor] = None,
        genomics: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Fuse available modalities into a unified brain state vector.
        
        Args:
            imaging_features: [batch, d_imaging] — MRI volume encoder output
            demographics: [batch, d_demographics] — age, sex, etc.
            clinical_text: [batch, d_clinical_text] — BioClinicalBERT embedding
            connectome: [batch, d_connectome] — GATv2 graph embedding
            genomics: [batch, d_genomic] — genetic markers
            
        Returns:
            Brain state vector [batch, d_output]
            
        Note: At least ONE modality must be provided. Missing modalities
        are simply excluded from the key-value set.
        """
        # Collect available modality tokens
        tokens = []

        if imaging_features is not None:
            tokens.append(self.proj_imaging(imaging_features))
        if demographics is not None:
            tokens.append(self.proj_demographics(demographics))
        if clinical_text is not None:
            tokens.append(self.proj_clinical(clinical_text))
        if connectome is not None:
            tokens.append(self.proj_connectome(connectome))
        if genomics is not None:
            tokens.append(self.proj_genomic(genomics))

        assert len(tokens) > 0, "At least one modality must be provided"

        # Concatenate all modality tokens into key-value context
        context = torch.cat(tokens, dim=1)  # [batch, n_total_tokens, d_latent]

        # Expand learned latent queries for the batch
        batch_size = context.shape[0]
        latents = self.latent_queries.unsqueeze(0).expand(batch_size, -1, -1)

        # Cross-attention: latents attend to modality tokens
        for cross_layer in self.cross_attn_layers:
            latents = cross_layer(latents, context)

        # Self-attention: latents reason about inter-modality relationships
        for self_layer in self.self_attn_layers:
            latents = self_layer(latents)

        # Pool latent array → single vector
        brain_state = self.output_proj(latents.mean(dim=1))  # [batch, d_output]

        return brain_state
