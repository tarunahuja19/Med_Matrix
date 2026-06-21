/**
 * ai-client.ts — Phase 2 Node.js AI-service bridge
 *
 * Responsibilities:
 *  1. POST to FastAPI /predict with study + K-space info
 *  2. Parse the PredictResponse (anomaly scores + gating decision)
 *  3. Persist ModelResult, AnomalyDetection, and GatingDecision rows via Prisma
 *
 * Tech: Axios, Prisma
 */

import axios, { AxiosInstance } from 'axios'
import { PrismaClient } from '@prisma/client'

const prisma = new PrismaClient()

// ---------------------------------------------------------------------------
// Request / Response type mirrors (matches ai-service/models.py)
// ---------------------------------------------------------------------------

export interface PredictRequest {
  study_id: string
  kspace_key: string
  anomaly_threshold?: number   // default 0.5
  phase_correction?: boolean   // default true
  denoise_method?: string      // default "nlm"
}

export interface AnomalyScore {
  ghosting: number
  wrap_around: number
  zipper_noise: number
  composite_score: number
}

export interface GatingDecisionPayload {
  anomaly_detected: boolean
  confidence: number
  threshold_used: number
  image_encoder_triggered: boolean
  reason: string
}

export interface PredictResponse {
  status: string
  study_id: string
  anomaly_scores: AnomalyScore
  gating_decision: GatingDecisionPayload
  reconstructed_key: string | null
  artifact_report: Record<string, number> | null
  predicted_pathology: string | null
  pathology_confidence: number | null
  pathology_probabilities: Record<string, number> | null
  kspace_gradcam_key: string | null
  kspace_log_mag_key: string | null
  reconstructed_gradcam_key: string | null
  message: string | null
}

// ---------------------------------------------------------------------------
// Stored result shape returned to the caller (worker.ts)
// ---------------------------------------------------------------------------
export interface AIInferenceResult {
  modelResultId: string
  anomalyDetected: boolean
  confidence: number
  imageEncoderTriggered: boolean
  reconstructedKey: string | null
  artifactReport: Record<string, number> | null
  predictedPathology: string | null
  pathologyConfidence: number | null
  pathologyProbabilities: Record<string, number> | null
  kspaceGradcamKey: string | null
  kspaceLogMagKey: string | null
  reconstructedGradcamKey: string | null
}

// ---------------------------------------------------------------------------
// AI Client class
// ---------------------------------------------------------------------------

export class AIServiceClient {
  private readonly http: AxiosInstance

  constructor(baseURL?: string) {
    this.http = axios.create({
      baseURL: baseURL ?? process.env.AI_SERVICE_URL ?? 'http://localhost:8000',
      timeout: 120_000, // 2 min — inference can be slow on large MRIs
      headers: { 'Content-Type': 'application/json' },
    })
  }

  /**
   * Run K-Space anomaly detection + gating on a study.
   * Persists ModelResult, AnomalyDetection and GatingDecision to the DB.
   *
   * @returns AIInferenceResult with key IDs and gating outcome
   * @throws on HTTP error or DB failure
   */
  async predict(req: PredictRequest): Promise<AIInferenceResult> {
    // ── 1. Call /predict on the FastAPI AI-service ─────────────────────────
    let data: PredictResponse
    try {
      const res = await this.http.post<PredictResponse>('/predict', {
        study_id: req.study_id,
        kspace_key: req.kspace_key,
        anomaly_threshold: req.anomaly_threshold ?? 0.5,
        phase_correction: req.phase_correction ?? true,
        denoise_method: req.denoise_method ?? 'nlm',
      })
      data = res.data
    } catch (err: unknown) {
      if (axios.isAxiosError(err)) {
        throw new Error(
          `AI-service /predict failed [${err.response?.status}]: ${JSON.stringify(err.response?.data)}`
        )
      }
      throw new Error(`AI-service /predict network error: ${String(err)}`)
    }

    const {
      anomaly_scores,
      gating_decision,
      reconstructed_key,
      artifact_report,
      predicted_pathology,
      pathology_confidence,
      pathology_probabilities,
      kspace_gradcam_key,
      kspace_log_mag_key,
      reconstructed_gradcam_key,
    } = data

    // ── 2. Persist ModelResult (Anomaly Detection) ──────────────────────────
    const modelResult = await prisma.modelResult.create({
      data: {
        studyId: req.study_id,
        modelName: 'kspace-anomaly',
        modelVersion: 'v1',
        rawScores: JSON.stringify(anomaly_scores),
        confidenceScore: anomaly_scores.composite_score,
        reconstructedKey: reconstructed_key ?? null,
      },
    })

    // ── 3. Persist AnomalyDetection ────────────────────────────────────────
    await prisma.anomalyDetection.create({
      data: {
        studyId: req.study_id,
        modelResultId: modelResult.id,
        ghostingScore: anomaly_scores.ghosting,
        wrapAroundScore: anomaly_scores.wrap_around,
        zipperScore: anomaly_scores.zipper_noise,
        compositeScore: anomaly_scores.composite_score,
        anomalyDetected: gating_decision.anomaly_detected,
        threshold: gating_decision.threshold_used,
      },
    })

    // ── 4. Persist GatingDecision ──────────────────────────────────────────
    await prisma.gatingDecision.create({
      data: {
        studyId: req.study_id,
        modelResultId: modelResult.id,
        imageEncoderTriggered: gating_decision.image_encoder_triggered,
        confidence: gating_decision.confidence,
        reason: gating_decision.reason,
      },
    })

    // ── 5. Persist Pathology ModelResult (fused-s4-cnn) ──────────────────────
    if (predicted_pathology !== null) {
      await prisma.modelResult.create({
        data: {
          studyId: req.study_id,
          modelName: 'fused-s4-cnn',
          modelVersion: 'v1',
          rawScores: JSON.stringify({
            predictedPathology: predicted_pathology,
            probabilities: pathology_probabilities,
          }),
          confidenceScore: pathology_confidence ?? 0.0,
          reconstructedKey: reconstructed_key ?? null, // Link to the same reconstructed slice
        },
      })
    }

    return {
      modelResultId: modelResult.id,
      anomalyDetected: gating_decision.anomaly_detected,
      confidence: gating_decision.confidence,
      imageEncoderTriggered: gating_decision.image_encoder_triggered,
      reconstructedKey: reconstructed_key ?? null,
      artifactReport: artifact_report ?? null,
      predictedPathology: predicted_pathology,
      pathologyConfidence: pathology_confidence,
      pathologyProbabilities: pathology_probabilities,
      kspaceGradcamKey: kspace_gradcam_key ?? null,
      kspaceLogMagKey: kspace_log_mag_key ?? null,
      reconstructedGradcamKey: reconstructed_gradcam_key ?? null,
    }
  }

  /**
   * Generates a radiology report from clinical indication, patient metadata,
   * and scientific context via the FastAPI RAG agent query endpoint.
   */
  async generateRagReport(
    diseaseName: string,
    patientMetadata: {
      name: string
      age: number
      gender: string
      dateOfBirth: string
      symptoms?: string
      studyDate?: string
    },
    forPatient?: boolean
  ): Promise<string | null> {
    try {
      const res = await this.http.post<{ status: string; report: string }>('/rag/query', {
        disease_name: diseaseName,
        patient_metadata: patientMetadata,
        llm_model: 'gemini-3.5-flash',
        for_patient: forPatient ?? false,
      })
      if (res.data && res.data.status === 'success') {
        return res.data.report
      }
      return null
    } catch (err: any) {
      console.error(`AI-service /rag/query failed: ${err.message}`)
      return null
    }
  }

  /**
   * Fetches the forecasted untreated progression projection for a given pathology.
   */
  async getProgressionProjection(
    pathology: string,
    initialVolume?: number
  ): Promise<ProgressionResponse | null> {
    try {
      const res = await this.http.post<ProgressionResponse>('/predict/progression', {
        pathology,
        initial_pathology_volume_cm3: initialVolume ?? null,
      })
      if (res.data && res.data.status === 'success') {
        return res.data
      }
      return null
    } catch (err: any) {
      console.error(`AI-service /predict/progression failed: ${err.message}`)
      return null
    }
  }
}

export interface ProgressionPoint {
  month: number
  pathology_volume_cm3: number
  edema_volume_cm3: number
  healthy_brain_volume_cm3: number
  cognitive_impact_pct: number
  severity_level: string
  clinical_note: string
}

export interface ProgressionResponse {
  status: string
  pathology: string
  initial_volume_cm3: number
  timeline: ProgressionPoint[]
}

// Singleton export so worker and routes share one instance
export const aiClient = new AIServiceClient()