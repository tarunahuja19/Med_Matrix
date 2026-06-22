import { PrismaClient } from '@prisma/client'

const prisma = new PrismaClient()

async function main() {
  console.log('Clearing database...')
  await prisma.gatingDecision.deleteMany()
  await prisma.anomalyDetection.deleteMany()
  await prisma.modelResult.deleteMany()
  await prisma.report.deleteMany()
  await prisma.study.deleteMany()
  await prisma.patient.deleteMany()
  console.log('Database cleared successfully.')
}

main()
  .catch(console.error)
  .finally(() => prisma.$disconnect())
