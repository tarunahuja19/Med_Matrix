// AI Service request types
export interface KSpaceInferenceRequest {
  studyId: string
  kspacePath: string
  modality: string
}

export interface KSpaceInferenceResponse {
  studyId: string
  anomalyDetected: boolean
  anomalyScore: number
  confidence: number
}

export interface ImageInferenceRequest {
  studyId: string
  reconstructedPath: string
}

export interface ImageInferenceResponse {
  studyId: string
  findings: Finding[]
}

export interface Finding {
  type: 'tumor' | 'microbleed' | 'stroke' | 'hemorrhage' | 'ms_lesion'
  confidence: number
  boundingBox?: BoundingBox
  severity: number
}

export interface BoundingBox {
  x: number
  y: number
  z: number
  width: number
  height: number
  depth: number
}