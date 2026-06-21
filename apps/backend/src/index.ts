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
  const { name, dateOfBirth, gender, phone } = req.body

  if (!name || !dateOfBirth || !gender) {
    res.status(400).json({ error: 'name, dateOfBirth, and gender are required' })
    return
  }

  const patient = await prisma.patient.create({
    data: {
      name,
      dateOfBirth: new Date(dateOfBirth),
      gender,
      phone: phone || null,
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
          { patientImpression: null },
          { patientImpression: '' },
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

        const patientMetadata = {
          name: patient.name,
          age: age,
          gender: patient.gender,
          dateOfBirth: patient.dateOfBirth.toISOString(),
          symptoms: pathologyName === 'Normal'
            ? 'Routine check'
            : `Suspected ${pathologyName.replace(/_/g, ' ')}`,
          studyDate: study.studyDate.toISOString(),
        }

        const dataToUpdate: any = {}

        if (!report.impression) {
          const generatedReport = await aiClient.generateRagReport(
            pathologyName,
            patientMetadata,
            false
          )
          if (generatedReport) {
            dataToUpdate.impression = generatedReport
          }
        }

        if (!report.patientImpression) {
          const generatedPatientReport = await aiClient.generateRagReport(
            pathologyName,
            patientMetadata,
            true
          )
          if (generatedPatientReport) {
            dataToUpdate.patientImpression = generatedPatientReport
          }
        }

        if (Object.keys(dataToUpdate).length > 0) {
          await prisma.report.update({
            where: { id: report.id },
            data: dataToUpdate,
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
  const { impression, patientImpression, status } = req.body

  try {
    const updatedReport = await prisma.report.update({
      where: { id },
      data: {
        impression,
        patientImpression,
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

// Helper to generate the PDF buffer for a report (Professional or Patient summary)
async function generateReportPdfBuffer(reportId: string, forPatient: boolean): Promise<Buffer> {
  const report = await prisma.report.findUnique({
    where: { id: reportId },
    include: {
      study: {
        include: {
          patient: true,
        },
      },
    },
  })

  if (!report || !report.study || !report.study.patient) {
    throw new Error('Report or patient details not found')
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
  const reportText = forPatient ? (report.patientImpression ?? '') : (report.impression ?? '')

  const response = await axios.post(`${aiServiceUrl}/rag/generate-pdf`, {
    report_text: reportText,
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
    study_id: report.studyId,
    for_patient: forPatient
  }, {
    responseType: 'arraybuffer'
  })

  return Buffer.from(response.data)
}

app.post('/reports/:id/send-whatsapp', async (req, res) => {
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

    if (!report) {
      res.status(404).json({ error: 'Report not found' })
      return
    }

    const patient = report.study?.patient
    if (!patient) {
      res.status(404).json({ error: 'Patient not found' })
      return
    }

    // Fallback to default test phone number if not registered
    const rawPhone = patient.phone || '+919974202309'
    let cleanPhone = rawPhone.replace(/\D/g, '')
    if (!cleanPhone.startsWith('+')) {
      cleanPhone = `+${cleanPhone}`
    }

    const patientName = patient.name.replace(/\s+/g, '_')
    const filename = `patient_report_${patientName}_${report.id.substring(0, 8)}.pdf`

    console.log(`[Twilio WhatsApp] Generating PDF report for ${patient.name}...`)
    const pdfBuffer = await generateReportPdfBuffer(id, true)

    console.log(`[Twilio WhatsApp] Uploading PDF buffer to tmpfiles.org...`)
    const formData = new globalThis.FormData()
    formData.append('file', new Blob([new Uint8Array(pdfBuffer)], { type: 'application/pdf' }), filename)

    const uploadResponse = await fetch('https://tmpfiles.org/api/v1/upload', {
      method: 'POST',
      body: formData,
    })

    const responseData = await uploadResponse.json()
    if (responseData.status !== 'success' || !responseData.data?.url) {
      throw new Error(`Failed to upload file to tmpfiles.org: ${JSON.stringify(responseData)}`)
    }

    const publicMediaUrl = responseData.data.url.replace('https://tmpfiles.org/', 'https://tmpfiles.org/dl/')
    console.log(`[Twilio WhatsApp] PDF successfully uploaded. Public Link: ${publicMediaUrl}`)

    // Twilio configurations
    const accountSid = process.env.TWILIO_ACCOUNT_SID
    const authToken = process.env.TWILIO_AUTH_TOKEN
    const fromNumber = process.env.TWILIO_SENDER_NUMBER

    if (!accountSid || !authToken || !fromNumber) {
      throw new Error('Twilio configurations (TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, or TWILIO_SENDER_NUMBER) are missing from environment variables')
    }

    console.log(`[Twilio WhatsApp] Sending PDF document to ${cleanPhone} via Twilio...`)

    const twilioUrl = `https://api.twilio.com/2010-04-01/Accounts/${accountSid}/Messages.json`
    const authHeader = 'Basic ' + Buffer.from(`${accountSid}:${authToken}`).toString('base64')

    const params = new URLSearchParams()
    params.append('From', `whatsapp:${fromNumber}`)
    params.append('To', `whatsapp:${cleanPhone}`)
    params.append('MediaUrl', publicMediaUrl)
    params.append('Body', `Hello ${patient.name}, here is your patient-friendly Brain MRI report summary.`)

    const twilioResponse = await axios.post(twilioUrl, params.toString(), {
      headers: {
        'Authorization': authHeader,
        'Content-Type': 'application/x-www-form-urlencoded',
      },
    })

    console.log(`[Twilio WhatsApp] Delivered! Twilio SID: ${twilioResponse.data.sid}`)

    res.json({ success: true, phone: cleanPhone, filename, sid: twilioResponse.data.sid })
  } catch (err: any) {
    const errorDetails = err.response?.data || err.message
    console.error('[Twilio WhatsApp] Error:', errorDetails)
    res.status(500).json({ error: err.message, details: errorDetails })
  }
})

app.get('/reports/:id/pdf', async (req, res) => {
  const { id } = req.params
  const forPatient = req.query.forPatient === 'true'
  try {
    const pdfBuffer = await generateReportPdfBuffer(id, forPatient)
    
    const report = await prisma.report.findUnique({
      where: { id },
      include: { study: { include: { patient: true } } }
    })
    const patientName = report?.study?.patient?.name.replace(/\s+/g, '_') || 'patient'

    const filename = forPatient
      ? `patient_report_${patientName}_${id.substring(0, 8)}.pdf`
      : `radiology_report_${patientName}_${id.substring(0, 8)}.pdf`

    res.setHeader('Content-Type', 'application/pdf')
    res.setHeader('Content-Disposition', `attachment; filename=${filename}`)
    res.send(pdfBuffer)
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