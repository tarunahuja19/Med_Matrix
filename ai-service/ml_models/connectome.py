"""
KVision 4.0 — Connectome Explorer (GATv2 Graph Attention Network)

Encodes structural brain connectivity graphs using GATv2 (Graph Attention Network v2),
producing a fixed-size graph embedding that captures the patient's brain network state.

Why GATv2 instead of Graph Mamba:
    Brain connectome graphs are SMALL (68–360 nodes). At this scale, full attention
    is computationally trivial (O(N²) where N=360 → 130K operations). Mamba's
    linear scaling advantage only matters for N >> 1000. GATv2 is the established
    baseline for brain graph analysis (BrainGNN, BrainGB) and provides learnable
    attention over edges — exactly what's needed for heterogeneous brain networks.

Input:
    - Adjacency matrix A ∈ ℝ^{N×N}: Edge weights from structural connectivity
      (streamline counts from DWI tractography) or functional connectivity
      (Pearson correlation from resting-state fMRI)
    - Node features X ∈ ℝ^{N×d}: Per-region features
      (regional volume, mean cortical thickness, fractional anisotropy, mean diffusivity)

Output:
    - Graph embedding z ∈ ℝ^{d_out}: Fixed-size representation for downstream tasks

Parcellation Schemes Supported:
    - Desikan-Killiany (68 regions) — FreeSurfer default
    - Schaefer-100/200/400 — Functional parcellation
    - Glasser HCP-MMP (360 regions) — Multi-modal parcellation

References:
    - Brody et al., "How Attentive are Graph Attention Networks?", ICLR 2022 (GATv2)
    - Li et al., "BrainGNN", Medical Image Analysis 2021
    - Said et al., "BrainGB", NeurIPS 2022 (Brain Graph Benchmark)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class GATv2Layer(nn.Module):
    """
    Graph Attention Network v2 layer (Brody et al., ICLR 2022).
    
    Unlike GATv1 which computes attention with a static key,
    GATv2 computes attention dynamically:
        e_ij = a^T · LeakyReLU(W·[h_i || h_j])
    
    This allows the attention to be a universal approximator over the
    node feature pairs, not just a monotonic function of Wh_i + Wh_j.
    """

    def __init__(self, d_in: int, d_out: int, n_heads: int = 4, dropout: float = 0.1, residual: bool = True):
        super().__init__()
        self.n_heads = n_heads
        self.d_out = d_out
        self.d_head = d_out // n_heads
        self.residual = residual

        assert d_out % n_heads == 0, f"d_out ({d_out}) must be divisible by n_heads ({n_heads})"

        # Shared linear transformation
        self.W = nn.Linear(d_in, d_out, bias=False)

        # Attention mechanism: learns e_ij = a^T · LeakyReLU(W[h_i||h_j])
        self.attn = nn.Parameter(torch.FloatTensor(n_heads, self.d_head))
        nn.init.xavier_uniform_(self.attn.unsqueeze(0))

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

        # Optional residual connection
        if residual and d_in != d_out:
            self.res_proj = nn.Linear(d_in, d_out, bias=False)
        else:
            self.res_proj = None

        self.norm = nn.LayerNorm(d_out)

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        edge_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Node features [N, d_in]
            adj: Adjacency matrix [N, N] (binary or weighted)
            edge_weights: Optional edge features [N, N] (used to modulate attention)
            
        Returns:
            Updated node features [N, d_out]
        """
        N = x.shape[0]
        h = self.W(x)  # [N, d_out]
        h = h.view(N, self.n_heads, self.d_head)  # [N, heads, d_head]

        # Compute pairwise attention scores
        # GATv2: e_ij = a^T · LeakyReLU(W_h_i + W_h_j)
        # Expand for broadcasting: [N, 1, heads, d_head] + [1, N, heads, d_head]
        h_i = h.unsqueeze(1).expand(-1, N, -1, -1)  # [N, N, heads, d_head]
        h_j = h.unsqueeze(0).expand(N, -1, -1, -1)  # [N, N, heads, d_head]

        e = self.leaky_relu(h_i + h_j)  # [N, N, heads, d_head]
        e = (e * self.attn.unsqueeze(0).unsqueeze(0)).sum(dim=-1)  # [N, N, heads]

        # Mask non-edges (set attention to -inf for non-connected nodes)
        mask = (adj == 0).unsqueeze(-1).expand_as(e)
        e = e.masked_fill(mask, float("-inf"))

        # Modulate by edge weights if provided
        if edge_weights is not None:
            e = e + edge_weights.unsqueeze(-1).expand_as(e)

        # Softmax over neighbors
        alpha = F.softmax(e, dim=1)  # [N, N, heads]
        alpha = self.dropout(alpha)

        # Weighted aggregation
        out = torch.einsum("ijh,jhd->ihd", alpha, h)  # [N, heads, d_head]
        out = out.reshape(N, self.d_out)  # [N, d_out]

        # Residual connection
        if self.residual:
            residual = self.res_proj(x) if self.res_proj is not None else x
            out = out + residual

        return self.norm(out)


class ConnectomeEncoder(nn.Module):
    """
    Multi-layer GATv2 encoder for brain connectivity graphs.
    
    Produces a fixed-size graph embedding from an adjacency matrix
    and per-region node features.
    """

    def __init__(
        self,
        d_node: int = 8,
        d_hidden: int = 64,
        d_out: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.layers = nn.ModuleList()
        d_in = d_node
        for i in range(n_layers):
            self.layers.append(
                GATv2Layer(d_in, d_hidden, n_heads=n_heads, dropout=dropout, residual=True)
            )
            d_in = d_hidden

        # Graph-level readout
        self.readout = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_out),
        )

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        edge_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Node features [N, d_node]
            adj: Adjacency matrix [N, N]
            edge_weights: Optional edge weights [N, N]
            
        Returns:
            Graph embedding [d_out]
        """
        h = x
        for layer in self.layers:
            h = F.elu(layer(h, adj, edge_weights))

        # Global readout: mean + max pooling (captures both average and extreme patterns)
        h_mean = h.mean(dim=0)  # [d_hidden]
        h_max = h.max(dim=0).values  # [d_hidden]

        # Combine via concatenation → project (or just use mean for simplicity)
        graph_embedding = self.readout(h_mean)  # [d_out]

        return graph_embedding


class TemporalConnectomeODE(nn.Module):
    """
    Models the evolution of brain connectivity over time using a learned ODE.
    
    dA/dt = G_θ(A(t), z(t), u(t))
    
    Where:
        A(t): Adjacency matrix at time t
        z(t): Disease state from the Latent SDE
        u(t): Treatment vector
        G_θ: Graph neural network predicting edge weight changes
    
    Constraints:
        - A(t) remains symmetric (enforce A = (A + A^T) / 2)
        - Edge weights remain non-negative (apply ReLU)
        - Changes are sparse (L1 penalty on dA/dt)
    """

    def __init__(self, n_nodes: int = 360, d_state: int = 64, d_treatment: int = 8):
        super().__init__()

        # Predicts change rate for each edge
        self.edge_dynamics = nn.Sequential(
            nn.Linear(d_state + d_treatment + 1, 128),  # +1 for current edge weight
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
            nn.Tanh(),  # Output in [-1, 1] as relative change rate
        )

        self.change_scale = nn.Parameter(torch.tensor(0.01))  # Small changes per step

    def forward(
        self,
        adj: torch.Tensor,
        disease_state: torch.Tensor,
        treatment: torch.Tensor,
        dt: float = 1.0,
    ) -> torch.Tensor:
        """
        Predict the adjacency matrix at time t + dt.
        
        Args:
            adj: Current adjacency [N, N]
            disease_state: Latent disease state [d_state]
            treatment: Treatment vector [d_treatment]
            dt: Time step (months)
            
        Returns:
            Updated adjacency [N, N]
        """
        N = adj.shape[0]

        # Broadcast disease state and treatment to each edge
        z_expanded = disease_state.unsqueeze(0).unsqueeze(0).expand(N, N, -1)
        u_expanded = treatment.unsqueeze(0).unsqueeze(0).expand(N, N, -1)
        edge_feat = torch.cat([z_expanded, u_expanded, adj.unsqueeze(-1)], dim=-1)

        # Predict relative change for each edge
        delta = self.edge_dynamics(edge_feat).squeeze(-1)  # [N, N]

        # Apply change (multiplicative)
        new_adj = adj + self.change_scale * delta * adj * dt

        # Enforce constraints
        new_adj = F.relu(new_adj)                      # Non-negative
        new_adj = (new_adj + new_adj.t()) / 2          # Symmetric
        new_adj.fill_diagonal_(0)                       # No self-loops

        return new_adj
