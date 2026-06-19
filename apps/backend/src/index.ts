import 'dotenv/config'
import express from 'express'
import { PrismaClient } from '@prisma/client'
import { initBuckets, minioClient } from './storage'
import multer from 'multer'
import { parseDicomMetadata } from './dicom'

const app = express()
const prisma = new PrismaClient()
const PORT = 3000

// Configure Multer to store uploaded files in memory
const upload = multer({ storage: multer.memoryStorage() })

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

// DICOM upload endpoint
app.post('/studies/upload', upload.single('file'), async (req, res) => {
  try {
    if (!req.file) {
      res.status(400).json({ error: 'No file uploaded. Please upload a DICOM file in the "file" field.' })
      return
    }

    // Parse DICOM metadata using helper
    const metadata = parseDicomMetadata(req.file.buffer)

    // Check if patient already exists (match by name and DOB)
    let patient = await prisma.patient.findFirst({
      where: {
        name: metadata.patientName,
        dateOfBirth: metadata.patientDob
      }
    })

    if (!patient) {
      // Create new Patient if not found
      patient = await prisma.patient.create({
        data: {
          name: metadata.patientName,
          dateOfBirth: metadata.patientDob,
          gender: metadata.gender
        }
      })
    }

    // Create a new Study referencing the patient (no dicomKey yet)
    let study = await prisma.study.create({
      data: {
        patientId: patient.id,
        modality: metadata.modality,
        studyDate: metadata.studyDate,
        status: 'pending'
      },
      include: { patient: true }
    })

    // Upload the raw DICOM file to MinIO 'dicom-files' bucket
    const originalName = req.file.originalname || 'upload.dcm'
    const dicomKey = `${study.id}/${originalName}`
    const fileBuffer = req.file.buffer

    await new Promise<void>((resolve, reject) => {
      minioClient.putObject(
        'dicom-files',
        dicomKey,
        fileBuffer,
        fileBuffer.length,
        { 'Content-Type': 'application/dicom' },
        (err) => { if (err) reject(err); else resolve() }
      )
    })

    // Update Study record with dicomKey
    study = await prisma.study.update({
      where: { id: study.id },
      data: { dicomKey },
      include: { patient: true }
    })

    res.status(201).json({
      message: 'DICOM study uploaded and metadata saved successfully',
      study
    })
  } catch (error: any) {
    console.error('Error processing DICOM upload:', error)
    res.status(500).json({
      error: 'Failed to process DICOM upload',
      details: error.message
    })
  }
})

app.listen(PORT, async () => {
  await initBuckets()
  console.log(`Backend running on http://localhost:${PORT}`)
})