/**
 * worker.ts — Phase 2 BullMQ Worker
 *
 * Queue: "study-processing"
 *
 * Pipeline per job:
 *  10%  Mark study as "processing"
 *  20%  Call AI-service /predict (K-Space anomaly detection + gating)
 *  60%  Store ModelResult, AnomalyDetection, GatingDecision via aiClient
 *  80%  Create draft Report (includes AI findings + gating decision)
 *  100% Mark study as "complete"
 *
 * Gating logic:
 *  - If anomaly_detected=true  → image encoder was run inside /predict,
 *    reconstructed_key is populated, full artifact_report is available.
 *  - If anomaly_detected=false → K-Space was clean, image encoder skipped.
 *    Report still created but findings note "no anomaly detected".
 */

import { Worker, Job } from 'bullmq'
import { PrismaClient } from '@prisma/client'
import { StudyJobData } from './queue'
import { aiClient } from './ai-client'

const prisma = new PrismaClient()

const connection = {
  host: process.env.REDIS_HOST ?? 'localhost',
  port: Number(process.env.REDIS_PORT ?? 6379),
}

// ---------------------------------------------------------------------------
// Core job processor
// ---------------------------------------------------------------------------
async function processStudy(job: Job<StudyJobData>): Promise<void> {
  const { studyId, kspaceKey, phaseCorrection, denoiseMethod } = job.data

  // ── 10%: Mark study as processing ────────────────────────────────────────
  await prisma.study.update({
    where: { id: studyId },
    data: { status: 'processing' },
  })
  await job.updateProgress(10)

  // ── 20–60%: Run AI inference (K-Space anomaly + gating) ──────────────────
  const inferenceResult = await aiClient.predict({
    study_id: studyId,
    kspace_key: kspaceKey,
    anomaly_threshold: 0.5, // configurable per pathology in Phase 3
    phase_correction: phaseCorrection,
    denoise_method: denoiseMethod,
  })
  await job.updateProgress(60)

  // ── 80%: Create draft report ──────────────────────────────────────────────
  // Build findings string from inference result
  const findingsSummary = JSON.stringify({
    anomalyDetected: inferenceResult.anomalyDetected,
    confidence: inferenceResult.confidence,
    imageEncoderTriggered: inferenceResult.imageEncoderTriggered,
    reconstructedKey: inferenceResult.reconstructedKey,
    artifactScores: inferenceResult.artifactReport,
    modelResultId: inferenceResult.modelResultId,
    predictedPathology: inferenceResult.predictedPathology,
    pathologyConfidence: inferenceResult.pathologyConfidence,
    pathologyProbabilities: inferenceResult.pathologyProbabilities,
    kspaceGradcamKey: inferenceResult.kspaceGradcamKey,
    kspaceLogMagKey: inferenceResult.kspaceLogMagKey,
    reconstructedGradcamKey: inferenceResult.reconstructedGradcamKey,
    noiseSeverity: inferenceResult.noiseSeverity,
    motionSeverity: inferenceResult.motionSeverity,
    phaseSeverity: inferenceResult.phaseSeverity,
    note: inferenceResult.anomalyDetected
      ? undefined
      : 'K-Space anomaly score below threshold — image encoder not triggered.',
  })

  // ── 80%: Query RAG Reporting Agent and Create draft report ────────────────
  const study = await prisma.study.findUnique({
    where: { id: studyId },
    include: { patient: true },
  })
  const patient = study?.patient

  let generatedReport: string | null = null
  let generatedPatientReport: string | null = null
  if (patient && inferenceResult.predictedPathology) {
    try {
      const dob = new Date(patient.dateOfBirth)
      const today = new Date()
      let age = today.getFullYear() - dob.getFullYear()
      const m = today.getMonth() - dob.getMonth()
      if (m < 0 || (m === 0 && today.getDate() < dob.getDate())) {
        age--
      }

      const patientMetadata = {
        name: patient.name,
        age: age,
        gender: patient.gender,
        dateOfBirth: patient.dateOfBirth.toISOString(),
        symptoms: inferenceResult.predictedPathology === 'Normal'
          ? 'Routine check'
          : `Suspected ${inferenceResult.predictedPathology.replace(/_/g, ' ')}`,
        studyDate: study.studyDate.toISOString(),
      }

      generatedReport = await aiClient.generateRagReport(
        inferenceResult.predictedPathology,
        patientMetadata,
        false
      )

      generatedPatientReport = await aiClient.generateRagReport(
        inferenceResult.predictedPathology,
        patientMetadata,
        true
      )
    } catch (err: any) {
      console.error(`[worker] Failed to generate RAG report: ${err.message}`)
    }
  }

  await prisma.report.create({
    data: {
      studyId,
      findings: findingsSummary,
      impression: generatedReport,
      patientImpression: generatedPatientReport,
      status: 'draft',
    },
  })
  await job.updateProgress(80)

  // ── 100%: Mark study as complete ─────────────────────────────────────────
  await prisma.study.update({
    where: { id: studyId },
    data: { status: 'complete' },
  })
  await job.updateProgress(100)
}

// ---------------------------------------------------------------------------
// Worker factory
// ---------------------------------------------------------------------------
export function startWorker() {
  const worker = new Worker<StudyJobData>('study-processing', processStudy, {
    connection,
    concurrency: 2,
  })

  worker.on('active', (job) => {
    console.log(`⚙  [worker] Processing study ${job.data.studyId} (job ${job.id})`)
  })

  worker.on('completed', (job) => {
    console.log(`✓  [worker] Study ${job.data.studyId} complete (job ${job.id})`)
  })

  worker.on('failed', async (job, err) => {
    console.error(`✗  [worker] Job ${job?.id} failed:`, err.message)
    if (job?.data.studyId) {
      await prisma.study
        .update({ where: { id: job.data.studyId }, data: { status: 'failed' } })
        .catch(() => {})
    }
  })

  console.log('✓  BullMQ worker started (concurrency: 2) — Phase 2 AI gating enabled')
  return worker
}