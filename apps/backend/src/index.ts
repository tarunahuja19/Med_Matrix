import 'dotenv/config'
import express from 'express'
import cors from 'cors'
import { PrismaClient } from '@prisma/client'
import { initBuckets } from './storage'
import { startWorker } from './worker'
import studiesRouter from './routes/studies'
import { aiClient } from './ai-client'
import axios from 'axios'

const app = express()
const prisma = new PrismaClient()
const PORT = process.env.PORT ?? 3000

app.use(cors())
app.use(express.json())

// ── Health ──────────────────────────────────────────────────────────────────
app.get('/health', (_req, res) => {
  res.json({ status: 'ok' })
})

// ── Patients ─────────────────────────────────────────────────────────────────
app.get('/patients', async (_req, res) => {
  const patients = await prisma.patient.findMany()
  res.json(patients)
})

app.post('/patients', async (req, res) => {
  const { name, dateOfBirth, gender } = req.body

  if (!name || !dateOfBirth || !gender) {
    res.status(400).json({ error: 'name, dateOfBirth, and gender are required' })
    return
  }

  const patient = await prisma.patient.create({
    data: {
      name,
      dateOfBirth: new Date(dateOfBirth),
      gender,
    },
  })

  res.status(201).json(patient)
})

// ── Studies ──────────────────────────────────────────────────────────────────
// GET  /studies        → list all
// POST /studies/upload → upload + queue
// GET  /studies/:id/status → status + progress
app.get('/studies', async (_req, res) => {
  const studies = await prisma.study.findMany({ include: { patient: true } })
  res.json(studies)
})

app.use('/studies', studiesRouter)

// ── Reports ──────────────────────────────────────────────────────────────────
app.get('/reports', async (_req, res) => {
  const reports = await prisma.report.findMany({
    include: {
      study: {
        include: {
          patient: true,
        },
      },
    },
  })
  res.json(reports)
})

app.post('/reports/backfill', async (_req, res) => {
  try {
    const reportsToBackfill = await prisma.report.findMany({
      where: {
        OR: [
          { impression: null },
          { impression: '' },
        ],
      },
      include: {
        study: {
          include: {
            patient: true,
          },
        },
      },
    })

    let successCount = 0
    let failureCount = 0
    const results = []

    for (const report of reportsToBackfill) {
      const study = report.study
      const patient = study?.patient

      if (!study || !patient) {
        results.push({ reportId: report.id, status: 'error', reason: 'Missing study or patient' })
        failureCount++
        continue
      }

      // Parse predictedPathology from findings
      let predictedPathology: string | null = null
      if (report.findings) {
        try {
          const findingsObj = JSON.parse(report.findings)
          predictedPathology = findingsObj.predictedPathology
        } catch (err) {
          // Fallback or ignore
        }
      }

      if (!predictedPathology) {
        results.push({ reportId: report.id, status: 'error', reason: 'No predicted pathology in findings' })
        failureCount++
        continue
      }

      const pathologyName = predictedPathology

      try {
        const dob = new Date(patient.dateOfBirth)
        const today = new Date()
        let age = today.getFullYear() - dob.getFullYear()
        const m = today.getMonth() - dob.getMonth()
        if (m < 0 || (m === 0 && today.getDate() < dob.getDate())) {
          age--
        }

        const generatedReport = await aiClient.generateRagReport(
          pathologyName,
          {
            name: patient.name,
            age: age,
            gender: patient.gender,
            dateOfBirth: patient.dateOfBirth.toISOString(),
            symptoms: pathologyName === 'Normal'
              ? 'Routine check'
              : `Suspected ${pathologyName.replace(/_/g, ' ')}`,
            studyDate: study.studyDate.toISOString(),
          }
        )

        if (generatedReport) {
          await prisma.report.update({
            where: { id: report.id },
            data: { impression: generatedReport },
          })
          results.push({ reportId: report.id, status: 'success' })
          successCount++
        } else {
          results.push({ reportId: report.id, status: 'failed_generation' })
          failureCount++
        }
      } catch (err: any) {
        results.push({ reportId: report.id, status: 'error', reason: err.message })
        failureCount++
      }
    }

    res.json({
      message: `Backfill process finished. Success: ${successCount}, Failures: ${failureCount}`,
      results,
    })
  } catch (err: any) {
    res.status(500).json({ error: err.message })
  }
})

app.put('/reports/:id', async (req, res) => {
  const { id } = req.params
  const { impression, status } = req.body

  try {
    const updatedReport = await prisma.report.update({
      where: { id },
      data: {
        impression,
        status: status ?? 'draft',
      },
      include: {
        study: {
          include: {
            patient: true,
          },
        },
      },
    })
    res.json(updatedReport)
  } catch (err: any) {
    res.status(500).json({ error: err.message })
  }
})

app.get('/reports/:id/pdf', async (req, res) => {
  const { id } = req.params
  try {
    const report = await prisma.report.findUnique({
      where: { id },
      include: {
        study: {
          include: {
            patient: true,
          },
        },
      },
    })

    if (!report || !report.study || !report.study.patient) {
      res.status(404).json({ error: 'Report not found' })
      return
    }

    const { study } = report
    const { patient } = study

    // Calculate age
    const dob = new Date(patient.dateOfBirth)
    const today = new Date()
    let age = today.getFullYear() - dob.getFullYear()
    const m = today.getMonth() - dob.getMonth()
    if (m < 0 || (m === 0 && today.getDate() < dob.getDate())) {
      age--
    }

    // Call ai-service generate-pdf
    const aiServiceUrl = process.env.AI_SERVICE_URL ?? 'http://ai-service:8000'

    const response = await axios.post(`${aiServiceUrl}/rag/generate-pdf`, {
      report_text: report.impression ?? '',
      patient_metadata: {
        name: patient.name,
        age: age,
        sex: patient.gender === 'M' ? 'M' : patient.gender === 'F' ? 'F' : 'Other',
        physician: 'Dr. Tarun Ahuja, MD',
        report_id: report.id.substring(0, 8),
        patient_id: patient.id.substring(0, 8),
        study_date: new Date(study.studyDate).toLocaleDateString('en-US'),
        modality: study.modality === 'MRI' ? 'MRI (3T)' : study.modality,
        date: new Date(report.createdAt).toLocaleDateString('en-US')
      },
      study_id: report.studyId
    }, {
      responseType: 'arraybuffer'
    })

    res.setHeader('Content-Type', 'application/pdf')
    res.setHeader('Content-Disposition', `attachment; filename=radiology_report_${patient.name.replace(/\s+/g, '_')}_${report.id.substring(0, 8)}.pdf`)
    res.send(response.data)
  } catch (err: any) {
    console.error('[backend] Failed to generate report PDF:', err.message)
    res.status(500).json({ error: `PDF generation failed: ${err.message}` })
  }
})

// ── Boot ─────────────────────────────────────────────────────────────────────
app.listen(PORT, async () => {
  await initBuckets()
  startWorker()
  console.log(`✓  Backend running on http://localhost:${PORT}`)
})