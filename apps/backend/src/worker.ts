import { Worker, Job } from 'bullmq'
import { PrismaClient } from '@prisma/client'
import { StudyJobData } from './queue'

const prisma = new PrismaClient()

const connection = {
  host: process.env.REDIS_HOST ?? 'localhost',
  port: Number(process.env.REDIS_PORT ?? 6379),
}

const AI_SERVICE_URL = process.env.AI_SERVICE_URL ?? 'http://localhost:8000'

async function processStudy(job: Job<StudyJobData>): Promise<void> {
  const { studyId, kspaceKey, phaseCorrection, denoiseMethod } = job.data

  // Mark study as processing
  await prisma.study.update({
    where: { id: studyId },
    data: { status: 'processing' },
  })

  await job.updateProgress(10)

  // Build the reconstructed output key
  const reconstructedKey = `${studyId}/reconstructed.npy`

  // Call AI service
  const response = await fetch(`${AI_SERVICE_URL}/reconstruct`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      study_id: studyId,
      kspace_key: kspaceKey,
      reconstructed_key: reconstructedKey,
      phase_correction: phaseCorrection,
      denoise_method: denoiseMethod,
    }),
  })

  if (!response.ok) {
    const detail = await response.text()
    throw new Error(`AI service returned ${response.status}: ${detail}`)
  }

  const result = await response.json() as {
    status: string
    artifact_report: Record<string, number>
  }

  await job.updateProgress(80)

  // Store findings in a draft report
  await prisma.report.create({
    data: {
      studyId,
      findings: JSON.stringify(result.artifact_report),
      status: 'draft',
    },
  })

  // Mark study as complete
  await prisma.study.update({
    where: { id: studyId },
    data: { status: 'complete' },
  })

  await job.updateProgress(100)
}

export function startWorker() {
  const worker = new Worker<StudyJobData>('study-processing', processStudy, {
    connection,
    concurrency: 2,
  })

  worker.on('active', (job) => {
    console.log(`⚙  Processing study ${job.data.studyId} (job ${job.id})`)
  })

  worker.on('completed', (job) => {
    console.log(`✓  Study ${job.data.studyId} complete (job ${job.id})`)
  })

  worker.on('failed', async (job, err) => {
    console.error(`✗  Job ${job?.id} failed:`, err.message)
    if (job?.data.studyId) {
      await prisma.study
        .update({ where: { id: job.data.studyId }, data: { status: 'failed' } })
        .catch(() => {})
    }
  })

  console.log('✓  BullMQ worker started (concurrency: 2)')
  return worker
}