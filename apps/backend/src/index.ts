import 'dotenv/config'
import express from 'express'
import { PrismaClient } from '@prisma/client'
import { initBuckets } from './storage'
import { startWorker } from './worker'
import studiesRouter from './routes/studies'

const app = express()
const prisma = new PrismaClient()
const PORT = process.env.PORT ?? 3000

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
  const reports = await prisma.report.findMany({ include: { study: true } })
  res.json(reports)
})

// ── Boot ─────────────────────────────────────────────────────────────────────
app.listen(PORT, async () => {
  await initBuckets()
  startWorker()
  console.log(`✓  Backend running on http://localhost:${PORT}`)
})