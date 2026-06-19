import { Queue } from 'bullmq'

const connection = {
  host: process.env.REDIS_HOST ?? 'localhost',
  port: Number(process.env.REDIS_PORT ?? 6379),
}

export const studyQueue = new Queue('study-processing', { connection })

export type StudyJobData = {
  studyId: string
  kspaceKey: string
  modality: string
  phaseCorrection: boolean
  denoiseMethod: string
}