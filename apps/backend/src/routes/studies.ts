import { Router } from 'express'
import multer from 'multer'
import { Readable } from 'stream'
import { PrismaClient } from '@prisma/client'
import { studyQueue } from '../queue'
import { minioClient } from '../storage'

const router = Router()
const prisma = new PrismaClient()

// Memory storage is fine for dev; swap to disk storage for large MRI files in prod
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 500 * 1024 * 1024 }, // 500 MB cap
})

// ---------------------------------------------------------------------------
// POST /studies/upload
// Body (multipart/form-data):
//   kspace       - the raw K-space file (.npy / .h5 / .dat)
//   patientId    - UUID of an existing patient
//   modality     - e.g. "MRI"
//   studyDate    - ISO date string
//   phaseCorrection  (optional, default true)
//   denoiseMethod    (optional, default "nlm")
// ---------------------------------------------------------------------------
router.post('/upload', upload.single('kspace'), async (req, res) => {
  const file = req.file
  if (!file) {
    res.status(400).json({ error: 'No kspace file provided' })
    return
  }

  const { patientId, modality, studyDate, phaseCorrection, denoiseMethod } = req.body

  if (!patientId || !modality || !studyDate) {
    res.status(400).json({ error: 'patientId, modality, and studyDate are required' })
    return
  }

  // Verify patient exists
  const patient = await prisma.patient.findUnique({ where: { id: patientId } })
  if (!patient) {
    res.status(404).json({ error: `Patient ${patientId} not found` })
    return
  }

  // Create study row (status: pending)
  const study = await prisma.study.create({
    data: {
      patientId,
      modality,
      studyDate: new Date(studyDate),
      status: 'pending',
    },
  })

  // Upload file to MinIO kspace-raw bucket
  const ext = file.originalname.split('.').pop() ?? 'npy'
  const kspaceKey = `${study.id}/kspace_input.${ext}`

  await minioClient.putObject(
    'kspace-raw',
    kspaceKey,
    Readable.from(file.buffer),
    file.size,
    { 'Content-Type': file.mimetype ?? 'application/octet-stream' }
  )

  // Enqueue processing job
  const job = await studyQueue.add(
    'process-study',
    {
      studyId: study.id,
      kspaceKey,
      modality,
      phaseCorrection: phaseCorrection !== 'false',
      denoiseMethod: denoiseMethod ?? 'nlm',
    },
    {
      attempts: 3,
      backoff: { type: 'exponential', delay: 5000 },
    }
  )

  res.status(201).json({
    studyId: study.id,
    jobId: job.id,
    status: 'pending',
    message: 'Study queued for processing',
  })
})

// ---------------------------------------------------------------------------
// GET /studies/:id/status
// ---------------------------------------------------------------------------
router.get('/:id/status', async (req, res) => {
  const study = await prisma.study.findUnique({
    where: { id: req.params.id },
    include: { reports: { orderBy: { createdAt: 'desc' }, take: 1 } },
  })

  if (!study) {
    res.status(404).json({ error: 'Study not found' })
    return
  }

  // Try to fetch live job progress if a jobId is provided as query param
  let progress: number | null = null
  const { jobId } = req.query
  if (typeof jobId === 'string') {
    const job = await studyQueue.getJob(jobId)
    if (job) progress = job.progress as number
  }

  res.json({
    studyId: study.id,
    status: study.status,
    modality: study.modality,
    studyDate: study.studyDate,
    progress,
    latestReport: study.reports[0] ?? null,
  })
})

export default router