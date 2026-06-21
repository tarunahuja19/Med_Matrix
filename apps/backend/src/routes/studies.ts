import { Router } from 'express'
import multer from 'multer'
import { Readable } from 'stream'
import { PrismaClient } from '@prisma/client'
import { studyQueue } from '../queue'
import { minioClient } from '../storage'
import { aiClient } from '../ai-client'

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

// ---------------------------------------------------------------------------
// GET /studies/:id/reconstructed
// Streams the reconstructed binary .npy file from MinIO reconstructed bucket
// ---------------------------------------------------------------------------
router.get('/:id/reconstructed', async (req, res) => {
  const { id } = req.params

  try {
    // 1. Find the model result for this study that has a reconstructed key
    const modelResult = await prisma.modelResult.findFirst({
      where: {
        studyId: id,
        reconstructedKey: { not: null },
      },
      orderBy: {
        createdAt: 'desc',
      },
    })

    if (!modelResult || !modelResult.reconstructedKey) {
      res.status(404).json({ error: 'Reconstructed image not found for this study' })
      return
    }

    // 2. Fetch the reconstructed image stream from MinIO 'reconstructed' bucket
    const stream = await minioClient.getObject('reconstructed', modelResult.reconstructedKey)

    // 3. Set content headers and pipe the stream to client
    res.setHeader('Content-Type', 'application/octet-stream')
    stream.pipe(res)
  } catch (err: any) {
    console.error(`Failed to fetch reconstructed image for study ${id}: ${err.message}`)
    res.status(500).json({ error: `Failed to fetch reconstructed image: ${err.message}` })
  }
})

// ---------------------------------------------------------------------------
// GET /studies/:id/kspace-log-mag
// Streams the log-magnitude K-space binary .npy file from MinIO
// ---------------------------------------------------------------------------
router.get('/:id/kspace-log-mag', async (req, res) => {
  const { id } = req.params
  try {
    const stream = await minioClient.getObject('reconstructed', `${id}/kspace_log_mag.npy`)
    res.setHeader('Content-Type', 'application/octet-stream')
    stream.pipe(res)
  } catch (err: any) {
    console.error(`Failed to fetch kspace log-magnitude for study ${id}: ${err.message}`)
    res.status(500).json({ error: `Failed to fetch kspace log-magnitude: ${err.message}` })
  }
})

// ---------------------------------------------------------------------------
// GET /studies/:id/kspace-gradcam
// Streams the K-space Grad-CAM overlay binary .npy file from MinIO
// ---------------------------------------------------------------------------
router.get('/:id/kspace-gradcam', async (req, res) => {
  const { id } = req.params
  try {
    const stream = await minioClient.getObject('reconstructed', `${id}/kspace_gradcam.npy`)
    res.setHeader('Content-Type', 'application/octet-stream')
    stream.pipe(res)
  } catch (err: any) {
    console.error(`Failed to fetch kspace Grad-CAM for study ${id}: ${err.message}`)
    res.status(500).json({ error: `Failed to fetch kspace Grad-CAM: ${err.message}` })
  }
})

// ---------------------------------------------------------------------------
// GET /studies/:id/reconstructed-gradcam
// Streams the reconstructed image Grad-CAM overlay binary .npy file from MinIO
// ---------------------------------------------------------------------------
router.get('/:id/reconstructed-gradcam', async (req, res) => {
  const { id } = req.params
  try {
    const stream = await minioClient.getObject('reconstructed', `${id}/reconstructed_gradcam.npy`)
    res.setHeader('Content-Type', 'application/octet-stream')
    stream.pipe(res)
  } catch (err: any) {
    console.error(`Failed to fetch reconstructed Grad-CAM for study ${id}: ${err.message}`)
    res.status(500).json({ error: `Failed to fetch reconstructed Grad-CAM: ${err.message}` })
  }
})

// ---------------------------------------------------------------------------
// GET /studies/:id/progression
// Fetches progression projection based on pathology from Fused S4-CNN
// ---------------------------------------------------------------------------
router.get('/:id/progression', async (req, res) => {
  const { id } = req.params
  const initialVolume = req.query.initialVolume ? Number(req.query.initialVolume) : undefined

  try {
    // Find the latest completed model result for this study
    const modelResult = await prisma.modelResult.findFirst({
      where: {
        studyId: id,
        modelName: 'fused-s4-cnn',
      },
      orderBy: {
        createdAt: 'desc',
      },
    })

    let pathology = 'Normal'
    if (modelResult) {
      try {
        const scores = JSON.parse(modelResult.rawScores)
        if (scores && scores.predictedPathology) {
          pathology = scores.predictedPathology
        }
      } catch (e) {
        // Ignore parsing error
      }
    }

    const projection = await aiClient.getProgressionProjection(pathology, initialVolume)
    if (!projection) {
      res.status(502).json({ error: 'Failed to generate progression projection from AI service' })
      return
    }

    res.json(projection)
  } catch (err: any) {
    console.error(`Failed to fetch progression projection for study ${id}: ${err.message}`)
    res.status(500).json({ error: `Failed to fetch progression projection: ${err.message}` })
  }
})

export default router