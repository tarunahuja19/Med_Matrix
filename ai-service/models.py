"""
Pydantic request/response schemas for the KVISION AI-service Phase 2 endpoints.

Defines:
  - PredictRequest   – input to /predict
  - AnomalyScore     – per-artifact scores from K-Space analysis
  - GatingDecision   – whether image encoder should be triggered
  - PredictResponse  – full response from /predict
"""
from pydantic import BaseModel, Field
from typing import Optional


# ---------------------------------------------------------------------------
# /predict  — Request
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    study_id: str = Field(..., description="UUID of the study being analysed")
    kspace_key: str = Field(..., description="MinIO object key for the raw K-space file")
    anomaly_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Confidence threshold above which K-Space anomaly is considered detected. "
            "If anomaly_score >= threshold → image encoder is triggered."
        ),
    )
    phase_correction: bool = Field(default=True, description="Apply phase correction before reconstruction")
    denoise_method: str = Field(default="nlm", description="Denoising method: 'nlm' or 'dncnn'")


# ---------------------------------------------------------------------------
# /predict  — Internal sub-schemas
# ---------------------------------------------------------------------------
class AnomalyScore(BaseModel):
    """Artifact probabilities returned by the K-Space analysis CNN."""
    ghosting: float = Field(..., ge=0.0, le=1.0)
    wrap_around: float = Field(..., ge=0.0, le=1.0)
    zipper_noise: float = Field(..., ge=0.0, le=1.0)
    composite_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Max of individual artifact probabilities — used for gating decision.",
    )


class GatingDecision(BaseModel):
    """Records the gating logic outcome."""
    anomaly_detected: bool = Field(
        ...,
        description="True when composite_score >= anomaly_threshold",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Composite anomaly confidence score",
    )
    threshold_used: float = Field(..., description="Threshold applied for this request")
    image_encoder_triggered: bool = Field(
        ...,
        description="Whether the image encoder pipeline was invoked",
    )
    reason: str = Field(..., description="Human-readable explanation of the gating decision")


# ---------------------------------------------------------------------------
# /predict  — Response
# ---------------------------------------------------------------------------
class PredictResponse(BaseModel):
    status: str = Field(..., description="'success' or 'error'")
    study_id: str
    anomaly_scores: AnomalyScore
    gating_decision: GatingDecision
    reconstructed_key: Optional[str] = Field(
        default=None,
        description="MinIO key where the reconstructed image is stored (populated when image encoder ran)",
    )
    artifact_report: Optional[dict] = Field(
        default=None,
        description="Full artifact report dict (only present when image encoder ran)",
    )
    predicted_pathology: Optional[str] = Field(default=None, description="Predicted pathology class name")
    pathology_confidence: Optional[float] = Field(default=None, description="Softmax confidence score for pathology prediction")
    pathology_probabilities: Optional[dict] = Field(default=None, description="Full map of pathology class probabilities")
    message: Optional[str] = None