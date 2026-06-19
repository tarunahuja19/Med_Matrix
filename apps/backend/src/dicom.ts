import * as dcmjs from 'dcmjs'

export interface ParsedDicomMetadata {
  patientName: string
  patientDob: Date
  gender: string
  modality: string
  studyDate: Date
}

export function parseDicomMetadata(buffer: Buffer): ParsedDicomMetadata {
  // Convert Node Buffer to ArrayBuffer as required by dcmjs
  const arrayBuffer = buffer.buffer.slice(
    buffer.byteOffset,
    buffer.byteOffset + buffer.byteLength
  )

  const dicomDict = dcmjs.data.DicomMessage.readFile(arrayBuffer)
  const dataset = dcmjs.data.DicomMetaDictionary.naturalizeDataset(dicomDict.dict)

  // 1. Patient Name
  let patientName = 'Anonymous'
  if (dataset.PatientName) {
    if (typeof dataset.PatientName === 'object' && dataset.PatientName !== null) {
      patientName = dataset.PatientName.Alphabetic || 'Anonymous'
    } else if (typeof dataset.PatientName === 'string') {
      patientName = dataset.PatientName
    }
  }
  // Sanitize DICOM patient name format (usually separated by ^ caret)
  patientName = patientName.replace(/\^/g, ' ').trim() || 'Anonymous'

  // 2. Date of Birth
  let patientDob = new Date('1970-01-01')
  if (typeof dataset.PatientBirthDate === 'string' && dataset.PatientBirthDate.length === 8) {
    const year = parseInt(dataset.PatientBirthDate.substring(0, 4), 10)
    const month = parseInt(dataset.PatientBirthDate.substring(4, 6), 10) - 1
    const day = parseInt(dataset.PatientBirthDate.substring(6, 8), 10)
    const d = new Date(year, month, day)
    if (!isNaN(d.getTime())) {
      patientDob = d
    }
  }

  // 3. Gender / PatientSex
  let gender = 'Unknown'
  if (typeof dataset.PatientSex === 'string') {
    const sex = dataset.PatientSex.trim().toUpperCase()
    if (sex === 'M') {
      gender = 'Male'
    } else if (sex === 'F') {
      gender = 'Female'
    } else if (sex === 'O') {
      gender = 'Other'
    } else {
      gender = sex
    }
  }

  // 4. Modality
  let modality = 'MRI'
  if (typeof dataset.Modality === 'string' && dataset.Modality.trim()) {
    modality = dataset.Modality.trim()
  }

  // 5. Study Date
  let studyDate = new Date()
  if (typeof dataset.StudyDate === 'string' && dataset.StudyDate.length === 8) {
    const year = parseInt(dataset.StudyDate.substring(0, 4), 10)
    const month = parseInt(dataset.StudyDate.substring(4, 6), 10) - 1
    const day = parseInt(dataset.StudyDate.substring(6, 8), 10)
    let hour = 0, minute = 0, second = 0
    if (typeof dataset.StudyTime === 'string' && dataset.StudyTime.length >= 6) {
      hour = parseInt(dataset.StudyTime.substring(0, 2), 10)
      minute = parseInt(dataset.StudyTime.substring(2, 4), 10)
      second = parseInt(dataset.StudyTime.substring(4, 6), 10)
    }
    const d = new Date(year, month, day, hour, minute, second)
    if (!isNaN(d.getTime())) {
      studyDate = d
    }
  }

  return {
    patientName,
    patientDob,
    gender,
    modality,
    studyDate
  }
}
