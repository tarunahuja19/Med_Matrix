import 'dotenv/config'
import express from 'express'
import { PrismaClient } from '@prisma/client'
import { initBuckets } from './storage'



const app = express()
const prisma = new PrismaClient()
const PORT = 3000

app.use(express.json())

app.get('/health', (req, res) => {
  res.json({ status: 'ok' })
})

app.get('/patients', async (req, res) => {
  const patients = await prisma.patient.findMany()
  res.json(patients)
})

app.get('/studies', async (req, res) => {
  const studies = await prisma.study.findMany({ include: { patient: true } })
  res.json(studies)
})

app.get('/reports', async (req, res) => {
  const reports = await prisma.report.findMany({ include: { study: true } })
  res.json(reports)
})


app.listen(PORT, async () => {
  await initBuckets()
  console.log(`Backend running on http://localhost:${PORT}`)
})