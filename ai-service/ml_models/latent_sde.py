"""
KVision 4.0 — Latent Stochastic Differential Equation (SDE) for Disease Dynamics

This module implements the core of the Digital Brain Twin: a learned disease
progression model that replaces the hardcoded parametric curves in routes/progression.py.

Mathematical Formulation:
    ds/dt = f_θ(s(t), u(t), c) + σ_θ(s(t)) · dW_t

Where:
    s(t) ∈ ℝ^d     : Latent brain state (encodes pathology volume, texture, edema, etc.)
    u(t) ∈ ℝ^k     : Treatment vector (binary: surgery, radiation, chemo, immunotherapy, etc.)
    c   ∈ ℝ^p      : Patient covariates (age, sex, genetic markers, APOE4, IDH status)
    σ_θ(s(t))       : State-dependent diffusion (captures irreducible stochasticity)
    W_t             : Standard Wiener process

Training Strategy:
    1. Encode each longitudinal MRI scan at time t_i into latent z(t_i)
    2. Integrate SDE from t_i to t_{i+1} using adaptive solver (Dormand-Prince)
    3. Score predicted latent against encoded observation at t_{i+1}
    4. Loss = reconstruction + KL divergence + trajectory matching

Target Datasets:
    - ADNI (Alzheimer's Disease Neuroimaging Initiative): 2000+ subjects, 2-8 visits
    - OASIS-3: 1098 subjects, up to 30 years follow-up
    - BraTS-LFPS: ~400 subjects with pre/post treatment pairs
    - UK Biobank: 50,000+ repeat imaging subjects

References:
    - Rubanova et al., "Latent ODEs for Irregularly-Sampled Time Series", NeurIPS 2019
    - Kidger et al., "Neural SDEs Made Easy", 2021
    - De Brouwer et al., "GRU-ODE-Bayes", NeurIPS 2019
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentSDEDrift(nn.Module):
    """
    Drift function f_θ(s, u, c) of the Latent SDE.
    
    Models the deterministic component of disease evolution,
    conditioned on treatment and patient covariates.
    
    Architecture: 3-layer MLP with SiLU activations and residual connection.
    """

    def __init__(self, d_state: int = 64, d_treatment: int = 8, d_covariate: int = 16, d_hidden: int = 128):
        super().__init__()
        d_input = d_state + d_treatment + d_covariate

        self.net = nn.Sequential(
            nn.Linear(d_input, d_hidden),
            nn.SiLU(),
            nn.LayerNorm(d_hidden),
            nn.Linear(d_hidden, d_hidden),
            nn.SiLU(),
            nn.LayerNorm(d_hidden),
            nn.Linear(d_hidden, d_state),
        )

        # Residual scaling (initialized near zero for stable training)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, s: torch.Tensor, u: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            s: Brain state [batch, d_state]
            u: Treatment vector [batch, d_treatment]
            c: Patient covariates [batch, d_covariate]
        Returns:
            Drift vector ds/dt [batch, d_state]
        """
        x = torch.cat([s, u, c], dim=-1)
        return self.residual_scale * self.net(x)


class LatentSDEDiffusion(nn.Module):
    """
    Diffusion function σ_θ(s) of the Latent SDE.
    
    Models the stochastic (uncertain) component of disease evolution.
    Outputs a diagonal diffusion matrix (per-dimension noise scale).
    
    Constrained to be positive via Softplus activation.
    """

    def __init__(self, d_state: int = 64, d_hidden: int = 64, min_sigma: float = 0.01):
        super().__init__()
        self.min_sigma = min_sigma

        self.net = nn.Sequential(
            nn.Linear(d_state, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_state),
            nn.Softplus(),  # Ensure positive noise scale
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """
        Args:
            s: Brain state [batch, d_state]
        Returns:
            Diffusion scale σ [batch, d_state] (positive)
        """
        return self.net(s) + self.min_sigma


class LatentSDEDynamics(nn.Module):
    """
    Complete Latent SDE dynamics module combining drift and diffusion.
    
    Used by the SDE solver to evolve the brain state forward in time:
        ds = f_θ(s, u, c) · dt + σ_θ(s) · dW
    """

    def __init__(
        self,
        d_state: int = 64,
        d_treatment: int = 8,
        d_covariate: int = 16,
        d_hidden: int = 128,
    ):
        super().__init__()
        self.d_state = d_state
        self.d_treatment = d_treatment
        self.d_covariate = d_covariate

        self.drift = LatentSDEDrift(d_state, d_treatment, d_covariate, d_hidden)
        self.diffusion = LatentSDEDiffusion(d_state, d_hidden=d_hidden // 2)

    def forward(
        self,
        s: torch.Tensor,
        u: torch.Tensor,
        c: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes both drift and diffusion at the current state.
        
        Returns:
            (drift, diffusion): Both of shape [batch, d_state]
        """
        return self.drift(s, u, c), self.diffusion(s)

    def sde_step(
        self,
        s: torch.Tensor,
        u: torch.Tensor,
        c: torch.Tensor,
        dt: float,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Euler-Maruyama SDE integration step.
        
        s_{t+dt} = s_t + f(s_t, u_t, c) * dt + σ(s_t) * sqrt(dt) * ε
        
        Args:
            s: Current state [batch, d_state]
            u: Treatment vector [batch, d_treatment]
            c: Covariates [batch, d_covariate]
            dt: Time step size
            noise: Optional pre-generated noise [batch, d_state]
        """
        f, sigma = self.forward(s, u, c)

        if noise is None:
            noise = torch.randn_like(s)

        return s + f * dt + sigma * math.sqrt(dt) * noise


class BrainStateEncoder(nn.Module):
    """
    Encodes observable brain measurements into the latent state space.
    
    Takes MRI-derived features (volumes, intensities, texture descriptors)
    and maps them to the Latent SDE state vector s(t).
    
    This is the q_φ(z(t) | x(t)) inference network in the VAE framework.
    """

    def __init__(self, d_observation: int = 32, d_state: int = 64, d_hidden: int = 128):
        super().__init__()

        # Posterior mean
        self.mu_net = nn.Sequential(
            nn.Linear(d_observation, d_hidden),
            nn.SiLU(),
            nn.LayerNorm(d_hidden),
            nn.Linear(d_hidden, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_state),
        )

        # Posterior log-variance
        self.logvar_net = nn.Sequential(
            nn.Linear(d_observation, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_state),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Observed brain features [batch, d_observation]
        Returns:
            (mu, logvar): Posterior parameters for z ~ N(mu, exp(logvar))
        """
        return self.mu_net(x), self.logvar_net(x)

    def sample(self, x: torch.Tensor) -> torch.Tensor:
        """Reparameterized sampling: z = mu + exp(0.5*logvar) * eps"""
        mu, logvar = self.forward(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps


class BrainStateDecoder(nn.Module):
    """
    Decodes the latent state into clinically meaningful observables.
    
    This is the p_θ(x(t) | z(t)) generative model.
    
    Outputs:
        - Pathology volume (cm³)
        - Edema volume (cm³)
        - Healthy brain volume (cm³)
        - Cognitive impact (0-100%)
        - Severity logits (4-class: Mild/Moderate/Severe/Critical)
    """

    def __init__(self, d_state: int = 64, d_hidden: int = 128):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(d_state, d_hidden),
            nn.SiLU(),
            nn.LayerNorm(d_hidden),
        )

        # Volume heads (positive via Softplus)
        self.volume_head = nn.Sequential(
            nn.Linear(d_hidden, 64),
            nn.SiLU(),
            nn.Linear(64, 3),       # pathology_vol, edema_vol, healthy_vol
            nn.Softplus(),          # Volumes must be non-negative
        )

        # Cognitive impact head (0-1 via Sigmoid, scaled to 0-100 at output)
        self.cognitive_head = nn.Sequential(
            nn.Linear(d_hidden, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # Severity classification head
        self.severity_head = nn.Sequential(
            nn.Linear(d_hidden, 32),
            nn.SiLU(),
            nn.Linear(32, 4),       # Mild, Moderate, Severe, Critical
        )

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            z: Latent state [batch, d_state]
        Returns:
            Dict with keys: volumes, cognitive_impact, severity_logits
        """
        h = self.shared(z)

        volumes = self.volume_head(h)              # [batch, 3]
        cognitive = self.cognitive_head(h) * 100.0  # [batch, 1] → scaled to 0-100%
        severity = self.severity_head(h)            # [batch, 4]

        return {
            "volumes": volumes,                     # [pathology, edema, healthy]
            "cognitive_impact": cognitive.squeeze(-1),
            "severity_logits": severity,
        }


class DigitalBrainTwin(nn.Module):
    """
    Complete Digital Brain Twin: Encoder + SDE Dynamics + Decoder.
    
    Given:
        - A patient's current brain observation x(t₀)
        - Treatment plan u(t)
        - Patient covariates c
        - Forecast horizon [t₀, t₀ + Δt]
    
    Produces:
        - Trajectory of brain states s(t₀), s(t₁), ..., s(t_N)
        - Decoded clinical observables at each timepoint
        - Uncertainty bands via Monte Carlo SDE sampling
    
    Example usage:
        twin = DigitalBrainTwin()
        trajectory = twin.forecast(
            observation=brain_features,        # Current MRI features
            treatment=treatment_vector,         # Treatment plan
            covariates=patient_info,           # Age, sex, genetics
            time_horizon=24.0,                 # 24 months
            n_steps=48,                        # 2-week resolution
            n_samples=50,                      # MC samples for uncertainty
        )
    """

    SEVERITY_LABELS = ["Mild", "Moderate", "Severe", "Critical"]

    def __init__(
        self,
        d_observation: int = 32,
        d_state: int = 64,
        d_treatment: int = 8,
        d_covariate: int = 16,
        d_hidden: int = 128,
    ):
        super().__init__()

        self.encoder = BrainStateEncoder(d_observation, d_state, d_hidden)
        self.dynamics = LatentSDEDynamics(d_state, d_treatment, d_covariate, d_hidden)
        self.decoder = BrainStateDecoder(d_state, d_hidden)

    def forecast(
        self,
        observation: torch.Tensor,
        treatment: torch.Tensor,
        covariates: torch.Tensor,
        time_horizon: float = 24.0,
        n_steps: int = 48,
        n_samples: int = 50,
    ) -> dict:
        """
        Generate Monte Carlo forecast trajectories.
        
        Args:
            observation: Brain features at t=0 [batch, d_observation]
            treatment: Treatment vector [batch, d_treatment]
            covariates: Patient covariates [batch, d_covariate]
            time_horizon: Forecast length in months
            n_steps: Number of integration steps
            n_samples: Number of MC samples for uncertainty
            
        Returns:
            Dict with:
                - mean_trajectory: Mean decoded observables at each step
                - ci_lower: 5th percentile (90% CI lower)
                - ci_upper: 95th percentile (90% CI upper)
                - time_points: Array of time values
        """
        self.eval()
        dt = time_horizon / n_steps
        batch = observation.shape[0]

        all_trajectories = []

        with torch.no_grad():
            for sample_idx in range(n_samples):
                # Encode initial state (stochastic via reparameterization)
                s = self.encoder.sample(observation)  # [batch, d_state]

                trajectory = []
                for step in range(n_steps + 1):
                    # Decode current state
                    decoded = self.decoder(s)
                    trajectory.append({
                        "volumes": decoded["volumes"].cpu(),
                        "cognitive_impact": decoded["cognitive_impact"].cpu(),
                        "severity_logits": decoded["severity_logits"].cpu(),
                    })

                    # Integrate one SDE step (except at last step)
                    if step < n_steps:
                        s = self.dynamics.sde_step(s, treatment, covariates, dt)

                all_trajectories.append(trajectory)

        # Aggregate across MC samples
        time_points = [i * dt for i in range(n_steps + 1)]
        return self._aggregate_trajectories(all_trajectories, time_points)

    def _aggregate_trajectories(self, all_trajectories: list, time_points: list) -> dict:
        """Computes mean and confidence intervals across MC trajectory samples."""
        n_samples = len(all_trajectories)
        n_steps = len(time_points)

        # Stack volumes: [n_samples, n_steps, batch, 3]
        volumes = torch.stack([
            torch.stack([all_trajectories[s][t]["volumes"] for t in range(n_steps)])
            for s in range(n_samples)
        ])

        cognitive = torch.stack([
            torch.stack([all_trajectories[s][t]["cognitive_impact"] for t in range(n_steps)])
            for s in range(n_samples)
        ])

        severity = torch.stack([
            torch.stack([all_trajectories[s][t]["severity_logits"] for t in range(n_steps)])
            for s in range(n_samples)
        ])

        return {
            "time_points": time_points,
            "volumes_mean": volumes.mean(dim=0),
            "volumes_ci_lower": volumes.quantile(0.05, dim=0),
            "volumes_ci_upper": volumes.quantile(0.95, dim=0),
            "cognitive_mean": cognitive.mean(dim=0),
            "cognitive_ci_lower": cognitive.quantile(0.05, dim=0),
            "cognitive_ci_upper": cognitive.quantile(0.95, dim=0),
            "severity_mean": F.softmax(severity.mean(dim=0), dim=-1),
        }

    def compute_loss(
        self,
        observations: list[torch.Tensor],
        time_points: list[float],
        treatment: torch.Tensor,
        covariates: torch.Tensor,
        beta_kl: float = 0.1,
    ) -> dict[str, torch.Tensor]:
        """
        Training loss for longitudinal sequences.
        
        Given a sequence of observations {x(t₁), x(t₂), ..., x(t_K)} at
        irregular time points, compute:
        
        L = Σ_k ||decode(s(t_k)) - x(t_k)||² + β · KL[q(z|x) || N(0,I)]
        
        Args:
            observations: List of K observation tensors [batch, d_obs]
            time_points: List of K time values (months)
            treatment: Treatment vector [batch, d_treatment]
            covariates: Patient covariates [batch, d_covariate]
            beta_kl: Weight for KL divergence term
            
        Returns:
            Dict with total_loss, recon_loss, kl_loss, trajectory_loss
        """
        self.train()

        # Encode first observation
        mu, logvar = self.encoder(observations[0])
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        s = mu + std * eps

        # KL divergence: KL[N(mu, sigma²) || N(0, I)]
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()

        # Integrate and accumulate reconstruction loss
        recon_loss = torch.tensor(0.0, device=s.device)
        trajectory_loss = torch.tensor(0.0, device=s.device)

        for k in range(len(observations)):
            # Decode current state
            decoded = self.decoder(s)
            # Reconstruction loss (MSE on volumes)
            target_vols = observations[k][:, :3]  # First 3 features are volumes
            recon_loss = recon_loss + F.mse_loss(decoded["volumes"], target_vols)

            # Integrate to next timepoint
            if k < len(observations) - 1:
                dt = time_points[k + 1] - time_points[k]
                n_substeps = max(1, int(dt / 0.5))  # Sub-step every ~2 weeks
                sub_dt = dt / n_substeps
                for _ in range(n_substeps):
                    s = self.dynamics.sde_step(s, treatment, covariates, sub_dt)

                # Trajectory matching: encode next obs, compare to predicted state
                mu_next, _ = self.encoder(observations[k + 1])
                trajectory_loss = trajectory_loss + F.mse_loss(s, mu_next)

        recon_loss = recon_loss / len(observations)
        trajectory_loss = trajectory_loss / max(1, len(observations) - 1)

        total_loss = recon_loss + beta_kl * kl_loss + trajectory_loss

        return {
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "trajectory_loss": trajectory_loss,
        }
