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
  const findingsSummary = inferenceResult.anomalyDetected
    ? JSON.stringify({
        anomalyDetected: true,
        confidence: inferenceResult.confidence,
        imageEncoderTriggered: inferenceResult.imageEncoderTriggered,
        reconstructedKey: inferenceResult.reconstructedKey,
        artifactScores: inferenceResult.artifactReport,
        modelResultId: inferenceResult.modelResultId,
      })
    : JSON.stringify({
        anomalyDetected: false,
        confidence: inferenceResult.confidence,
        imageEncoderTriggered: false,
        note: 'K-Space anomaly score below threshold — image encoder not triggered.',
        modelResultId: inferenceResult.modelResultId,
      })

  await prisma.report.create({
    data: {
      studyId,
      findings: findingsSummary,
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