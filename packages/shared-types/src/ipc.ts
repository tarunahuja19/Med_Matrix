// IPC channels
export type IPCChannel = 
  | 'ping'
  | 'study:upload'
  | 'study:status'
  | 'study:report'

// IPC payloads
export interface StudyUploadPayload {
  filePath: string
  patientId: string
  modality: string
}

export interface StudyStatusPayload {
  studyId: string
}

export interface StudyReportPayload {
  studyId: string
}

// IPC responses
export interface IPCResponse<T> {
  success: boolean
  data?: T
  error?: string
}