-- CreateTable
CREATE TABLE "ModelResult" (
    "id" TEXT NOT NULL,
    "studyId" TEXT NOT NULL,
    "modelName" TEXT NOT NULL,
    "modelVersion" TEXT NOT NULL DEFAULT 'v1',
    "rawScores" TEXT NOT NULL,
    "confidenceScore" DOUBLE PRECISION NOT NULL,
    "reconstructedKey" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "ModelResult_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "AnomalyDetection" (
    "id" TEXT NOT NULL,
    "studyId" TEXT NOT NULL,
    "modelResultId" TEXT NOT NULL,
    "ghostingScore" DOUBLE PRECISION NOT NULL,
    "wrapAroundScore" DOUBLE PRECISION NOT NULL,
    "zipperScore" DOUBLE PRECISION NOT NULL,
    "compositeScore" DOUBLE PRECISION NOT NULL,
    "anomalyDetected" BOOLEAN NOT NULL,
    "threshold" DOUBLE PRECISION NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "AnomalyDetection_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "GatingDecision" (
    "id" TEXT NOT NULL,
    "studyId" TEXT NOT NULL,
    "modelResultId" TEXT NOT NULL,
    "imageEncoderTriggered" BOOLEAN NOT NULL,
    "confidence" DOUBLE PRECISION NOT NULL,
    "reason" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "GatingDecision_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "AnomalyDetection_modelResultId_key" ON "AnomalyDetection"("modelResultId");

-- CreateIndex
CREATE UNIQUE INDEX "GatingDecision_modelResultId_key" ON "GatingDecision"("modelResultId");

-- AddForeignKey
ALTER TABLE "ModelResult" ADD CONSTRAINT "ModelResult_studyId_fkey" FOREIGN KEY ("studyId") REFERENCES "Study"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "AnomalyDetection" ADD CONSTRAINT "AnomalyDetection_studyId_fkey" FOREIGN KEY ("studyId") REFERENCES "Study"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "AnomalyDetection" ADD CONSTRAINT "AnomalyDetection_modelResultId_fkey" FOREIGN KEY ("modelResultId") REFERENCES "ModelResult"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "GatingDecision" ADD CONSTRAINT "GatingDecision_studyId_fkey" FOREIGN KEY ("studyId") REFERENCES "Study"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "GatingDecision" ADD CONSTRAINT "GatingDecision_modelResultId_fkey" FOREIGN KEY ("modelResultId") REFERENCES "ModelResult"("id") ON DELETE RESTRICT ON UPDATE CASCADE;