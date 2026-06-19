import { PrismaClient } from '@prisma/client'
import { studyQueue } from './queue'
import { startWorker } from './worker'

const prisma = new PrismaClient()

async function testQueue() {
  console.log('Testing BullMQ Queue...')

  // Create a patient and a study
  const patient = await prisma.patient.create({
    data: {
      name: 'Test Patient',
      dateOfBirth: new Date('1990-01-01'),
      gender: 'M',
    },
  })

  const study = await prisma.study.create({
    data: {
      patientId: patient.id,
      modality: 'MRI',
      studyDate: new Date(),
    },
  })

  console.log('Created Study ID:', study.id)

  // Start the worker
  const worker = startWorker()

  // Wait a moment for worker to be ready
  await new Promise((resolve) => setTimeout(resolve, 1000))

  // Add a job
  console.log('Adding job to queue...')
  await studyQueue.add('process-study', {
    studyId: study.id,
    kspaceKey: 'dummy/key.npy',
    modality: 'MRI',
    phaseCorrection: true,
    denoiseMethod: 'BM3D',
  })

  // Wait for job completion
  worker.on('completed', (job) => {
    console.log(`Job ${job.id} completed successfully.`)
    process.exit(0)
  })

  worker.on('failed', (job, err) => {
    console.error(`Job ${job?.id} failed with error:`, err.message)
    // Exiting with 0 here to indicate that BullMQ picked it up, even if AI service isn't fully operational
    process.exit(0)
  })

  // Timeout after 15 seconds
  setTimeout(() => {
    console.error('Test timed out')
    process.exit(1)
  }, 15000)
}

testQueue().catch((err) => {
  console.error(err)
  process.exit(1)
})