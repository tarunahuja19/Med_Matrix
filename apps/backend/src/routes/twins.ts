import { Router } from 'express'
import { PrismaClient } from '@prisma/client'
import axios from 'axios'

const router = Router()
const prisma = new PrismaClient()

const AI_SERVICE_URL = process.env.AI_SERVICE_URL ?? 'http://localhost:8000'

// Helper for logging audit actions
async function logAudit(userId: string, action: string, resourceType: string, resourceId: string, metadata?: any) {
  try {
    await prisma.auditLog.create({
      data: {
        userId,
        action,
        resourceType,
        resourceId,
        metadata: metadata ? (metadata as any) : undefined,
      },
    })
  } catch (err: any) {
    console.error(`Failed to write audit log: ${err.message}`)
  }
}

// ---------------------------------------------------------------------------
// GET /twins/patient/:patientId
// Fetches the active brain twin with its trajectories and simulations
// ---------------------------------------------------------------------------
router.get('/patient/:patientId', async (req, res) => {
  const { patientId } = req.params
  try {
    const twin = await prisma.brainTwin.findFirst({
      where: { patientId },
      include: {
        trajectories: { orderBy: { createdAt: 'desc' }, take: 1 },
        simulations: { orderBy: { createdAt: 'desc' } },
      },
      orderBy: { createdAt: 'desc' },
    })

    if (!twin) {
      res.status(404).json({ error: `No brain twin found for patient ${patientId}` })
      return
    }

    res.json(twin)
  } catch (err: any) {
    res.status(500).json({ error: err.message })
  }
})

// ---------------------------------------------------------------------------
// POST /twins/patient/:patientId/initialize
// Initializes a patient's BrainTwin using findings from their latest study
// ---------------------------------------------------------------------------
router.post('/patient/:patientId/initialize', async (req, res) => {
  const { patientId } = req.params
  const userId = req.headers['x-user-id'] as string ?? 'system'

  try {
    // 1. Find latest complete study with classification results
    const latestStudy = await prisma.study.findFirst({
      where: { patientId, status: 'complete' },
      include: {
        modelResults: {
          where: { modelName: 'fused-s4-cnn' },
          orderBy: { createdAt: 'desc' },
          take: 1,
        },
        anomalyDetections: {
          orderBy: { createdAt: 'desc' },
          take: 1,
        },
      },
      orderBy: { studyDate: 'desc' },
    })

    if (!latestStudy) {
      res.status(404).json({ error: 'No completed studies found to initialize brain twin' })
      return
    }

    // 2. Extract baseline pathology values
    let pathology = 'Normal'
    let pathologyConfidence = 1.0
    let pathologyVol = 0.0

    const modelResult = latestStudy.modelResults[0]
    if (modelResult) {
      try {
        const scores = JSON.parse(modelResult.rawScores)
        pathology = scores.predictedPathology ?? 'Normal'
        pathologyConfidence = modelResult.confidenceScore
      } catch (e) {}
    }

    // Initial pathology volume defaults based on predicted class
    const defaultVols: Record<string, number> = {
      Normal: 0.0,
      Tumor_Glioma: 15.0,
      Ischemia: 25.0,
      MS_Lesions: 8.0,
      Hydrocephalus: 60.0,
      Atrophy: 35.0,
      Hemorrhage: 30.0,
      Cerebral_Cyst: 12.0,
      Edema: 10.0,
      AVM: 18.0,
      Cerebral_Microbleeds: 2.0,
    }
    pathologyVol = defaultVols[pathology] ?? 5.0

    // Construct 32-dim observation vector for Latent SDE encoder
    // format: [pathologyVol, edemaVol, healthyVol, ...zeros]
    const observation = new Array(32).fill(0.0)
    observation[0] = pathologyVol
    observation[1] = pathology === 'Tumor_Glioma' ? 10.0 : pathology === 'MS_Lesions' ? 1.5 : 0.0 // edema volume
    observation[2] = 1350.0 - (observation[0] + observation[1]) // healthy brain volume

    // 3. Request state vector initialization from Python AI service
    const response = await axios.post(`${AI_SERVICE_URL}/twin/initialize`, {
      observation,
    })

    const { state_vector, state_timestamp } = response.data

    // Clean up any stale brain twins, trajectories, and simulations for this patient
    const oldTwins = await prisma.brainTwin.findMany({ where: { patientId } })
    const oldTwinIds = oldTwins.map(t => t.id)
    if (oldTwinIds.length > 0) {
      await prisma.trajectory.deleteMany({ where: { brainTwinId: { in: oldTwinIds } } })
      await prisma.treatmentSimulation.deleteMany({ where: { brainTwinId: { in: oldTwinIds } } })
      await prisma.brainTwin.deleteMany({ where: { patientId } })
    }

    // 4. Create BrainTwin in database
    const twin = await prisma.brainTwin.create({
      data: {
        patientId,
        stateVector: Buffer.from(new Float32Array(state_vector).buffer),
        stateTimestamp: new Date(state_timestamp),
        version: 1,
      },
    })

    await logAudit(userId, 'initialize_twin', 'BrainTwin', twin.id, { pathology })

    // 5. Generate default baseline (no treatment) trajectory forecast
    const forecastResponse = await axios.post(`${AI_SERVICE_URL}/twin/forecast`, {
      state_vector,
      time_horizon: 24.0,
      n_steps: 24,
      treatment: [0, 0, 0, 0, 0, 0, 0, 1], // Standard "no treatment" vector
      covariates: [55.0, 1.0, 0.0, ...new Array(13).fill(0.0)], // Default age 55, female
    })

    const trajectory = await prisma.trajectory.create({
      data: {
        brainTwinId: twin.id,
        timepoints: JSON.stringify(forecastResponse.data.time_points.map((t: number, idx: number) => ({
          month: t,
          pathology_volume: forecastResponse.data.volumes_mean[idx][0],
          edema_volume: forecastResponse.data.volumes_mean[idx][1],
          healthy_volume: forecastResponse.data.volumes_mean[idx][2],
          cognitive_impact: forecastResponse.data.cognitive_mean[idx],
          severity: forecastResponse.data.severity_mean[idx],
        }))),
        uncertaintyBand: JSON.stringify({
          volumes_lower: forecastResponse.data.volumes_ci_lower,
          volumes_upper: forecastResponse.data.volumes_ci_upper,
          cognitive_lower: forecastResponse.data.cognitive_ci_lower,
          cognitive_upper: forecastResponse.data.cognitive_ci_upper,
        }),
      },
    })

    const populatedTwin = await prisma.brainTwin.findUnique({
      where: { id: twin.id },
      include: {
        trajectories: true,
        simulations: true,
      }
    })

    res.status(201).json({
      twin: populatedTwin,
      baselineTrajectory: trajectory,
    })
  } catch (err: any) {
    console.error(`Initialize Twin failed: ${err.message}`)
    res.status(500).json({ error: err.message })
  }
})

// ---------------------------------------------------------------------------
// POST /twins/:twinId/forecast
// Generates and stores a new custom forecasting trajectory
// ---------------------------------------------------------------------------
router.post('/:twinId/forecast', async (req, res) => {
  const { twinId } = req.params
  const { timeHorizon, treatmentVector, covariates } = req.body
  const userId = req.headers['x-user-id'] as string ?? 'system'

  try {
    const twin = await prisma.brainTwin.findUnique({ where: { id: twinId } })
    if (!twin) {
      res.status(404).json({ error: `BrainTwin ${twinId} not found` })
      return
    }

    // Convert Buffer back to Float32Array
    const state_vector = Array.from(new Float32Array(
      twin.stateVector.buffer,
      twin.stateVector.byteOffset,
      twin.stateVector.length / 4
    ))

    // Call python service
    const response = await axios.post(`${AI_SERVICE_URL}/twin/forecast`, {
      state_vector,
      time_horizon: timeHorizon ?? 24.0,
      n_steps: 24,
      treatment: treatmentVector ?? [0, 0, 0, 0, 0, 0, 0, 1],
      covariates: covariates ?? [55.0, 1.0, 0.0, ...new Array(13).fill(0.0)],
    })

    const trajectory = await prisma.trajectory.create({
      data: {
        brainTwinId: twinId,
        timepoints: JSON.stringify(response.data.time_points.map((t: number, idx: number) => ({
          month: t,
          pathology_volume: response.data.volumes_mean[idx][0],
          edema_volume: response.data.volumes_mean[idx][1],
          healthy_volume: response.data.volumes_mean[idx][2],
          cognitive_impact: response.data.cognitive_mean[idx],
          severity: response.data.severity_mean[idx],
        }))),
        uncertaintyBand: JSON.stringify({
          volumes_lower: response.data.volumes_ci_lower,
          volumes_upper: response.data.volumes_ci_upper,
          cognitive_lower: response.data.cognitive_ci_lower,
          cognitive_upper: response.data.cognitive_ci_upper,
        }),
      },
    })

    await logAudit(userId, 'run_forecast', 'BrainTwin', twinId, { timeHorizon })

    res.json(trajectory)
  } catch (err: any) {
    res.status(500).json({ error: err.message })
  }
})

// ---------------------------------------------------------------------------
// POST /twins/:twinId/simulate
// Runs multi-treatment simulation comparisons
// ---------------------------------------------------------------------------
router.post('/:twinId/simulate', async (req, res) => {
  const { twinId } = req.params
  const { treatmentNames, covariates, timeHorizon } = req.body
  const userId = req.headers['x-user-id'] as string ?? 'system'

  try {
    const twin = await prisma.brainTwin.findUnique({ where: { id: twinId } })
    if (!twin) {
      res.status(404).json({ error: `BrainTwin ${twinId} not found` })
      return
    }

    const state_vector = Array.from(new Float32Array(
      twin.stateVector.buffer,
      twin.stateVector.byteOffset,
      twin.stateVector.length / 4
    ))

    const response = await axios.post(`${AI_SERVICE_URL}/twin/simulate`, {
      state_vector,
      covariates: covariates ?? [55.0, 1.0, 0.0, ...new Array(13).fill(0.0)],
      treatment_names: treatmentNames ?? ['no_treatment', 'stupp_protocol'],
      time_horizon: timeHorizon ?? 24.0,
    })

    const simulation = await prisma.treatmentSimulation.create({
      data: {
        brainTwinId: twinId,
        treatmentPlan: JSON.stringify(treatmentNames),
        predictedOutcome: JSON.stringify({
          scenarios: response.data.scenarios,
          comparison: response.data.comparison,
          time_points: response.data.time_points,
        }),
      },
    })

    await logAudit(userId, 'run_treatment_simulation', 'BrainTwin', twinId, { treatmentNames })

    res.json(simulation)
  } catch (err: any) {
    res.status(500).json({ error: err.message })
  }
})

// ---------------------------------------------------------------------------
// POST /twins/:twinId/connectome
// Evolved connectome graph attention scores based on active model
// ---------------------------------------------------------------------------
router.post('/:twinId/connectome', async (req, res) => {
  const { twinId } = req.params
  const { dt, treatmentVector } = req.body

  try {
    const twin = await prisma.brainTwin.findUnique({ where: { id: twinId } })
    if (!twin) {
      res.status(404).json({ error: `BrainTwin ${twinId} not found` })
      return
    }

    const state_vector = Array.from(new Float32Array(
      twin.stateVector.buffer,
      twin.stateVector.byteOffset,
      twin.stateVector.length / 4
    ))

    const response = await axios.post(`${AI_SERVICE_URL}/twin/connectome`, {
      state_vector,
      treatment: treatmentVector ?? [0, 0, 0, 0, 0, 0, 0, 1],
      dt: dt ?? 0.0,
    })

    res.json(response.data)
  } catch (err: any) {
    res.status(500).json({ error: err.message })
  }
})

// ---------------------------------------------------------------------------
// POST /twins/:twinId/future-mri
// Dynamic SVF U-Net deformation grid mappings for visual warping
// ---------------------------------------------------------------------------
router.post('/:twinId/future-mri', async (req, res) => {
  const { twinId } = req.params
  const { deltaT, treatmentVector } = req.body

  try {
    const twin = await prisma.brainTwin.findUnique({ where: { id: twinId } })
    if (!twin) {
      res.status(404).json({ error: `BrainTwin ${twinId} not found` })
      return
    }

    const state_vector = Array.from(new Float32Array(
      twin.stateVector.buffer,
      twin.stateVector.byteOffset,
      twin.stateVector.length / 4
    ))

    const response = await axios.post(`${AI_SERVICE_URL}/twin/future-mri`, {
      state_vector,
      treatment: treatmentVector ?? [0, 0, 0, 0, 0, 0, 0, 1],
      delta_t: deltaT ?? 0.0,
    })

    res.json(response.data)
  } catch (err: any) {
    res.status(500).json({ error: err.message })
  }
})

// ---------------------------------------------------------------------------
// POST /reports/:reportId/signoff
// Workflow: Radiologist signs off on a report with optional amendments
// ---------------------------------------------------------------------------
router.post('/reports/:reportId/signoff', async (req, res) => {
  const { reportId } = req.params
  const { radiologistName, status, amendments } = req.body
  const userId = req.headers['x-user-id'] as string ?? 'radiologist'

  if (!radiologistName || !status) {
    res.status(400).json({ error: 'radiologistName and status are required' })
    return
  }

  try {
    const signoff = await prisma.radiologistSignOff.create({
      data: {
        reportId,
        radiologistName,
        status,
        amendments: amendments ?? null,
        signedAt: new Date(),
      },
    })

    // Update Report status
    await prisma.report.update({
      where: { id: reportId },
      data: { status: status === 'approved' ? 'final' : 'draft' },
    })

    await logAudit(userId, 'sign_off_report', 'Report', reportId, { status, radiologistName })

    res.json({ success: true, signoff })
  } catch (err: any) {
    res.status(500).json({ error: err.message })
  }
})

// ---------------------------------------------------------------------------
// GET /twins/patient/:patientId/similar
// Retrieves similar patients from database using Python vector matches
// ---------------------------------------------------------------------------
router.get('/patient/:patientId/similar', async (req, res) => {
  const { patientId } = req.params
  try {
    // 1. Get latest complete study to determine pathology
    const study = await prisma.study.findFirst({
      where: { patientId, status: 'complete' },
      include: {
        modelResults: {
          where: { modelName: 'fused-s4-cnn' },
          orderBy: { createdAt: 'desc' },
          take: 1,
        },
      },
      orderBy: { studyDate: 'desc' },
    })

    const pathology = study?.modelResults[0] 
      ? (JSON.parse(study.modelResults[0].rawScores).predictedPathology || 'Normal')
      : 'Normal'

    // 2. Query Python AI service for similar patient cases
    const response = await axios.post(`${AI_SERVICE_URL}/rag/similar-patients`, {
      patient_id: patientId,
      pathology,
      limit: 3,
    })

    res.json(response.data.similar_patients || [])
  } catch (err: any) {
    console.error(`Similar patient fetch failed: ${err.message}`)
    res.status(500).json({ error: err.message })
  }
})

// ---------------------------------------------------------------------------
// GET /audit-logs
// Fetch system audit trail for security and clinical trace compliance
// ---------------------------------------------------------------------------
router.get('/audit-logs', async (req, res) => {
  try {
    const logs = await prisma.auditLog.findMany({
      orderBy: { createdAt: 'desc' },
      take: 100,
    })
    res.json(logs)
  } catch (err: any) {
    res.status(500).json({ error: err.message })
  }
})

export default router
