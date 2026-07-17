"""
KVision 4.0 — Treatment Simulator (What-If Engine)

Uses the Latent SDE dynamics to simulate disease trajectories under
different treatment scenarios, enabling clinicians to compare outcomes
before committing to a treatment plan.

This is NOT a separate model. It wraps the DigitalBrainTwin and runs
multiple forward simulations with different treatment vectors.

Clinical Use Cases:
    1. "What if we do surgery + radiation vs. chemo alone?"
    2. "What is the 12-month cognitive trajectory with/without treatment?"
    3. "Which treatment minimizes tumor volume at 18 months?"
    4. "What is the probability of progression to 'Severe' by month 12?"

Output:
    For each treatment scenario, produces:
    - Mean trajectory with 90% confidence intervals
    - Probability of each severity level at each timepoint
    - Treatment comparison table (which is best on each metric)
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from ml_models.latent_sde import DigitalBrainTwin


@dataclass
class TreatmentPlan:
    """Represents a treatment scenario for simulation."""
    name: str
    description: str
    # Binary treatment flags: [surgery, radiation, chemo, immunotherapy,
    #                          targeted_therapy, corticosteroids, anticoagulant, monitoring_only]
    vector: list[float] = field(default_factory=lambda: [0.0] * 8)

    def to_tensor(self, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        return torch.tensor(self.vector, dtype=torch.float32, device=device).unsqueeze(0)


# ── Pre-defined Treatment Plans ──────────────────────────────────────────────
STANDARD_TREATMENTS = {
    "no_treatment": TreatmentPlan(
        name="No Treatment (Natural History)",
        description="Observation only — no intervention",
        vector=[0, 0, 0, 0, 0, 0, 0, 1],
    ),
    "surgery_only": TreatmentPlan(
        name="Surgical Resection Only",
        description="Maximal safe resection without adjuvant therapy",
        vector=[1, 0, 0, 0, 0, 0, 0, 0],
    ),
    "stupp_protocol": TreatmentPlan(
        name="Stupp Protocol (Surgery + TMZ + RT)",
        description="Standard of care for GBM: surgery + temozolomide + radiation",
        vector=[1, 1, 1, 0, 0, 1, 0, 0],
    ),
    "immunotherapy": TreatmentPlan(
        name="Immunotherapy (Checkpoint Inhibitor)",
        description="Anti-PD-1/PD-L1 immunotherapy",
        vector=[0, 0, 0, 1, 0, 0, 0, 0],
    ),
    "chemoradiation": TreatmentPlan(
        name="Chemoradiation",
        description="Concurrent chemotherapy and radiation without surgery",
        vector=[0, 1, 1, 0, 0, 1, 0, 0],
    ),
    "ms_dmt": TreatmentPlan(
        name="Disease-Modifying Therapy (MS)",
        description="DMT for multiple sclerosis (e.g., ocrelizumab, natalizumab)",
        vector=[0, 0, 0, 1, 1, 0, 0, 0],
    ),
    "ad_antiamyloid": TreatmentPlan(
        name="Anti-Amyloid Therapy (AD)",
        description="Aducanumab/Lecanemab for Alzheimer's disease",
        vector=[0, 0, 0, 1, 1, 0, 0, 0],
    ),
}


class TreatmentSimulator:
    """
    Runs the Digital Brain Twin forward under different treatment scenarios
    and compares outcomes.
    
    Example:
        simulator = TreatmentSimulator(brain_twin_model)
        results = simulator.compare_treatments(
            observation=current_brain_features,
            covariates=patient_info,
            treatment_names=["no_treatment", "stupp_protocol", "immunotherapy"],
            time_horizon=24.0,
        )
    """

    def __init__(self, model: DigitalBrainTwin):
        self.model = model

    def simulate_single(
        self,
        observation: torch.Tensor,
        covariates: torch.Tensor,
        treatment: TreatmentPlan,
        time_horizon: float = 24.0,
        n_steps: int = 48,
        n_samples: int = 50,
    ) -> dict:
        """
        Simulate a single treatment scenario.
        
        Returns dict with mean/CI trajectories for volumes, cognitive impact, severity.
        """
        treatment_tensor = treatment.to_tensor(device=observation.device)

        trajectory = self.model.forecast(
            observation=observation,
            treatment=treatment_tensor,
            covariates=covariates,
            time_horizon=time_horizon,
            n_steps=n_steps,
            n_samples=n_samples,
        )

        return {
            "treatment": treatment.name,
            "description": treatment.description,
            **trajectory,
        }

    def compare_treatments(
        self,
        observation: torch.Tensor,
        covariates: torch.Tensor,
        treatment_names: list[str] | None = None,
        custom_treatments: list[TreatmentPlan] | None = None,
        time_horizon: float = 24.0,
        n_steps: int = 48,
        n_samples: int = 50,
    ) -> dict:
        """
        Compare multiple treatment scenarios head-to-head.
        
        Args:
            observation: Current brain features [1, d_obs]
            covariates: Patient covariates [1, d_cov]
            treatment_names: List of keys from STANDARD_TREATMENTS
            custom_treatments: Optional list of custom TreatmentPlan objects
            time_horizon: Forecast horizon in months
            
        Returns:
            Dict with:
                - scenarios: List of per-treatment results
                - comparison: Head-to-head comparison table
                - recommendation: Best treatment on each metric
        """
        treatments = []

        # Add standard treatments
        if treatment_names:
            for name in treatment_names:
                if name in STANDARD_TREATMENTS:
                    treatments.append(STANDARD_TREATMENTS[name])

        # Add custom treatments
        if custom_treatments:
            treatments.extend(custom_treatments)

        if not treatments:
            treatments = [STANDARD_TREATMENTS["no_treatment"], STANDARD_TREATMENTS["stupp_protocol"]]

        # Run simulations
        scenarios = []
        for treatment in treatments:
            result = self.simulate_single(
                observation, covariates, treatment,
                time_horizon, n_steps, n_samples,
            )
            scenarios.append(result)

        # Build comparison table
        comparison = self._build_comparison(scenarios, time_horizon)

        return {
            "scenarios": scenarios,
            "comparison": comparison,
            "time_points": scenarios[0]["time_points"],
        }

    def _build_comparison(self, scenarios: list[dict], horizon: float) -> dict:
        """Builds a comparison table across treatment scenarios at key timepoints."""
        comparison = {}

        for scenario in scenarios:
            name = scenario["treatment"]
            vols = scenario["volumes_mean"]     # [n_steps, batch, 3]
            cog = scenario["cognitive_mean"]    # [n_steps, batch]

            # Final timepoint metrics
            final_pathology_vol = float(vols[-1, 0, 0].item())
            final_cognitive = float(cog[-1, 0].item())

            # Area under the volume curve (lower = better)
            dt = horizon / (len(vols) - 1)
            auc_pathology = float(torch.sum(vols[:, 0, 0]).item() * dt)

            comparison[name] = {
                "final_pathology_volume_cm3": round(final_pathology_vol, 2),
                "final_cognitive_impact_pct": round(final_cognitive, 1),
                "auc_pathology_volume": round(auc_pathology, 2),
            }

        return comparison
