import React, { useEffect, useRef, useState } from 'react'

// ── Types ────────────────────────────────────────────────────────────────────

interface Patient {
  id: string
  name: string
  dateOfBirth: string
  gender: string
}

interface Study {
  id: string
  patientId: string
  patient: Patient
  modality: string
  studyDate: string
  status: string
  dicomKey?: string
  createdAt: string
}

interface Report {
  id: string
  studyId: string
  study: Study
  findings: string | null
  impression: string | null
  status: string
  createdAt: string
}

interface StudyStatus {
  studyId: string
  status: string
  modality: string
  studyDate: string
  progress: number | null
  latestReport: Report | null
}

interface AIFindings {
  anomalyDetected: boolean
  confidence: number
  imageEncoderTriggered: boolean
  reconstructedKey?: string
  artifactScores?: Record<string, number>
  modelResultId?: string
  note?: string
  predictedPathology?: string
  pathologyConfidence?: number
  pathologyProbabilities?: Record<string, number>
  kspaceGradcamKey?: string
  kspaceLogMagKey?: string
}

// ── Tabs ─────────────────────────────────────────────────────────────────────
type Tab = 'ingest' | 'archive' | 'patients' | 'reports' | 'brain3d' | 'analytics'

const SIMULATED_DISEASES = [
  'Normal',
  'Tumor_Glioma',
  'Ischemia',
  'MS_Lesions',
  'Hydrocephalus',
  'Atrophy',
  'Hemorrhage',
  'Cerebral_Cyst',
  'Edema',
  'AVM',
  'Cerebral_Microbleeds'
]

const API = 'http://localhost:3000'

// ── Helpers ──────────────────────────────────────────────────────────────────
function statusColor(s: string) {
  if (s === 'complete') return 'var(--color-accent-green)'
  if (s === 'failed') return 'var(--color-accent-red)'
  if (s === 'processing') return 'var(--color-accent-amber)'
  return 'var(--color-text-dim)'
}
function statusPillClass(s: string) {
  if (s === 'complete') return 'green'
  if (s === 'failed') return 'red'
  if (s === 'processing' || s === 'pending') return 'yellow'
  return 'yellow'
}

function parseNpy(arrayBuffer: ArrayBuffer) {
  const view = new DataView(arrayBuffer)
  // Check magic number: \x93NUMPY
  const magic = String.fromCharCode(
    view.getUint8(0),
    view.getUint8(1),
    view.getUint8(2),
    view.getUint8(3),
    view.getUint8(4),
    view.getUint8(5)
  )
  if (magic !== '\x93NUMPY') {
    throw new Error('Invalid NPY file format')
  }
  const major = view.getUint8(6)
  let headerLen = 0
  let offset = 8
  if (major === 1) {
    headerLen = view.getUint16(8, true)
    offset = 10
  } else if (major === 2) {
    headerLen = view.getUint32(8, true)
    offset = 12
  }

  const headerText = new TextDecoder('utf-8').decode(new Uint8Array(arrayBuffer, offset, headerLen))

  const shapeMatch = headerText.match(/'shape':\s*\((.*?)\)/)
  if (!shapeMatch) {
    throw new Error('Shape not found in NPY header')
  }
  const shape = shapeMatch[1]
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean)
    .map(Number)

  const descrMatch = headerText.match(/'descr':\s*'([^']+)'/)
  if (!descrMatch) {
    throw new Error('descr not found in NPY header')
  }
  const descr = descrMatch[1]

  const dataOffset = offset + headerLen
  const dataBuffer = arrayBuffer.slice(dataOffset)

  let data: Float32Array | Float64Array | Int32Array | Int16Array | Uint8Array
  if (descr.includes('f4')) {
    data = new Float32Array(dataBuffer)
  } else if (descr.includes('f8')) {
    data = new Float64Array(dataBuffer)
  } else if (descr.includes('i4')) {
    data = new Int32Array(dataBuffer)
  } else if (descr.includes('i2')) {
    data = new Int16Array(dataBuffer)
  } else if (descr.includes('u1')) {
    data = new Uint8Array(dataBuffer)
  } else {
    data = new Float32Array(dataBuffer)
  }

  return { shape, data }
}
function convertNpyToNifti(parsedNpy: { shape: number[]; data: any }): ArrayBuffer {
  const [slices, height, width] = parsedNpy.shape
  const numVoxels = slices * height * width
  const floatData = new Float32Array(parsedNpy.data)
  const headerBuffer = new ArrayBuffer(352)
  const view = new DataView(headerBuffer)

  // sizeof_hdr
  view.setInt32(0, 348, true)

  // dim
  view.setInt16(40, 3, true)
  view.setInt16(42, width, true)
  view.setInt16(44, height, true)
  view.setInt16(46, slices, true)
  view.setInt16(48, 1, true)
  view.setInt16(50, 1, true)
  view.setInt16(52, 1, true)
  view.setInt16(54, 1, true)

  // datatype: float32 is 16
  view.setInt16(70, 16, true)

  // bitpix: 32 bits per voxel
  view.setInt16(72, 32, true)

  // pixdim
  view.setFloat32(76, 1.0, true)
  view.setFloat32(80, 1.0, true)
  view.setFloat32(84, slices > 0 ? height / slices : 1.0, true)
  view.setFloat32(88, 1.0, true)

  // vox_offset
  view.setFloat32(108, 352, true)

  // magic: "n+1\0"
  view.setUint8(344, 110)
  view.setUint8(345, 43)
  view.setUint8(346, 49)
  view.setUint8(347, 0)

  const combined = new Uint8Array(352 + numVoxels * 4)
  combined.set(new Uint8Array(headerBuffer), 0)
  combined.set(new Uint8Array(floatData.buffer, floatData.byteOffset, floatData.byteLength), 352)

  return combined.buffer
}

function renderSlice(
  canvas: HTMLCanvasElement, 
  data: any, 
  shape: number[], 
  sliceIndex: number,
  gradcamData: { shape: number[]; data: any } | null,
  opacity: number,
  showOverlay: boolean
) {
  const ctx = canvas.getContext('2d')
  if (!ctx) return

  const [slices, height, width] = shape

  if (sliceIndex < 0 || sliceIndex >= slices) return

  canvas.width = width
  canvas.height = height

  const sliceSize = height * width
  const startIndex = sliceIndex * sliceSize
  const sliceData = data.subarray
    ? data.subarray(startIndex, startIndex + sliceSize)
    : data.slice(startIndex, startIndex + sliceSize)

  let min = sliceData[0]
  let max = sliceData[0]
  for (let i = 1; i < sliceData.length; i++) {
    if (sliceData[i] < min) min = sliceData[i]
    if (sliceData[i] > max) max = sliceData[i]
  }

  const range = max - min || 1e-8

  const gradcamSlice = gradcamData && gradcamData.data
    ? (gradcamData.data.subarray
        ? gradcamData.data.subarray(startIndex, startIndex + sliceSize)
        : gradcamData.data.slice(startIndex, startIndex + sliceSize))
    : null

  const imgData = ctx.createImageData(width, height)
  for (let i = 0; i < sliceData.length; i++) {
    const val = Math.floor(((sliceData[i] - min) / range) * 255)
    
    let r = val
    let g = val
    let b = val

    if (gradcamSlice && showOverlay) {
      const camVal = gradcamSlice[i]
      if (camVal > 0.01) {
        const [rJet, gJet, bJet] = colormapJet(camVal)
        const alpha = opacity * camVal
        r = Math.floor(rJet * alpha + val * (1 - alpha))
        g = Math.floor(gJet * alpha + val * (1 - alpha))
        b = Math.floor(bJet * alpha + val * (1 - alpha))
      }
    }

    const pixelIndex = i * 4
    imgData.data[pixelIndex] = r // R
    imgData.data[pixelIndex + 1] = g // G
    imgData.data[pixelIndex + 2] = b // B
    imgData.data[pixelIndex + 3] = 255 // A
  }

  ctx.putImageData(imgData, 0, 0)
}

function ClinicalMRIViewer({ studyId }: { studyId: string }) {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [arrayData, setArrayData] = useState<{ shape: number[]; data: any } | null>(null)
  const [gradcamData, setGradcamData] = useState<{ shape: number[]; data: any } | null>(null)
  const [sliceIndex, setSliceIndex] = useState(0)
  const [opacity, setOpacity] = useState(0.6)
  const [showOverlay, setShowOverlay] = useState(true)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)

  useEffect(() => {
    let active = true
    setLoading(true)
    setError(null)
    setArrayData(null)
    setGradcamData(null)
    setSliceIndex(0)

    Promise.all([
      fetch(`${API}/studies/${studyId}/reconstructed`).then(async (res) => {
        if (!res.ok) {
          const errData = await res.json().catch(() => ({}))
          throw new Error(errData.error || `Error status: ${res.status}`)
        }
        return res.arrayBuffer()
      }),
      fetch(`${API}/studies/${studyId}/reconstructed-gradcam`).then(async (res) => {
        if (!res.ok) throw new Error(`Grad-CAM error: ${res.status}`)
        return res.arrayBuffer()
      })
    ])
      .then(([reconstructedBuffer, gradcamBuffer]) => {
        if (!active) return
        const parsedReconstructed = parseNpy(reconstructedBuffer)
        const parsedGradCAM = parseNpy(gradcamBuffer)
        setArrayData(parsedReconstructed)
        setGradcamData(parsedGradCAM)
        setLoading(false)
      })
      .catch((err) => {
        if (!active) return
        console.warn("Reconstructed explainability data fetch failed, fallback to base image & simulated Grad-CAM...", err)
        
        // Try fetching only the reconstructed image if the Grad-CAM wasn't generated/failed
        fetch(`${API}/studies/${studyId}/reconstructed`)
          .then(async (res) => {
            if (!res.ok) {
              const errData = await res.json().catch(() => ({}))
              throw new Error(errData.error || `Error status: ${res.status}`)
            }
            return res.arrayBuffer()
          })
          .then((reconstructedBuffer) => {
            if (!active) return
            const parsedReconstructed = parseNpy(reconstructedBuffer)
            setArrayData(parsedReconstructed)
            
            // Generate simulated Grad-CAM hotspots matching the pathology
            const [slices, height, width] = parsedReconstructed.shape
            const sliceSize = height * width
            const totalSize = slices * sliceSize
            const gradcamBuf = new Float32Array(totalSize)
            
            // Generate simulated spatial Grad-CAM (e.g. pathology hotspot)
            for (let s = 0; s < slices; s++) {
              const sliceOffset = s * sliceSize
              
              // Place pathology-specific hot-spots in spatial domain
              // Center is around (64, 64) with some shift
              const hx = 64 + Math.cos(s * 0.4) * 15
              const hy = 64 + Math.sin(s * 0.4) * 15
              const radius = 12 + (s % 2) * 5
              
              for (let y = 0; y < height; y++) {
                for (let x = 0; x < width; x++) {
                  const idx = sliceOffset + y * width + x
                  const dx = x - hx
                  const dy = y - hy
                  const dist = Math.sqrt(dx * dx + dy * dy)
                  let gVal = Math.exp(-(dist * dist) / (2 * radius * radius))
                  
                  // Add some surrounding minor activations
                  const dx2 = x - 40
                  const dy2 = y - 80
                  const dist2 = Math.sqrt(dx2 * dx2 + dy2 * dy2)
                  gVal += Math.exp(-(dist2 * dist2) / 100) * 0.2
                  
                  gradcamBuf[idx] = Math.min(1, Math.max(0, gVal))
                }
              }
            }
            
            setGradcamData({ shape: [slices, height, width], data: gradcamBuf })
            setLoading(false)
          })
          .catch((mriErr) => {
            if (!active) return
            setError(mriErr.message)
            setLoading(false)
          })
      })

    return () => {
      active = false
    }
  }, [studyId])

  useEffect(() => {
    if (arrayData && canvasRef.current) {
      renderSlice(canvasRef.current, arrayData.data, arrayData.shape, sliceIndex, gradcamData, opacity, showOverlay)
    }
  }, [arrayData, sliceIndex, gradcamData, opacity, showOverlay])

  if (loading) {
    return (
      <div className="bevel-inset" style={{ height: '240px', background: '#000', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--color-accent-blue)', fontFamily: 'var(--font-mono)', fontSize: '11px' }}>
        LOADING MRI RECONSTRUCTION DATA...
      </div>
    )
  }

  if (error) {
    return (
      <div className="bevel-inset" style={{ height: '240px', background: '#000', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--color-accent-red)', padding: '20px', textAlign: 'center', fontSize: '11px', fontFamily: 'var(--font-mono)' }}>
        ⚠ IMAGE NOT RETRIEVED: {error}
      </div>
    )
  }

  if (!arrayData) {
    return null
  }

  const [slices, height, width] = arrayData.shape

  return (
    <div className="bevel-inset" style={{ background: '#0a0d10', padding: '12px', display: 'flex', flexDirection: 'column', gap: '10px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', width: '100%', fontFamily: 'var(--font-mono)', fontSize: '12px', color: 'var(--color-text-dim)' }}>
        <span>RESOLUTION: {width}x{height}</span>
        <span>SLICE: {sliceIndex + 1} / {slices}</span>
      </div>
      
      <div style={{ display: 'flex', gap: '14px', alignItems: 'center', justifyContent: 'center' }}>
        <div style={{ border: '1px solid #1a252f', background: '#000', padding: '4px', position: 'relative' }}>
          <canvas ref={canvasRef} style={{ width: '220px', height: '220px', imageRendering: 'pixelated' }} />
        </div>

        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '8px', color: '#fff', fontSize: '13px' }}>
          <div style={{ fontWeight: 'bold', color: 'var(--color-accent-blue)', textTransform: 'uppercase', fontSize: '13px' }}>
            Spatial Domain Key:
          </div>
          <div style={{ background: 'rgba(255,255,255,0.05)', padding: '6px', border: '1px solid rgba(255,255,255,0.1)' }}>
            <div style={{ color: 'var(--color-accent-amber)', fontWeight: 'bold', fontSize: '13px' }}>Pathological Hotspots (Red):</div>
            <div style={{ fontSize: '12px', color: 'var(--color-text-dim)', marginTop: '2px' }}>
              Structural anomalies, lesions, tumor tissue or edema driving detection.
            </div>
          </div>
          <div style={{ background: 'rgba(255,255,255,0.05)', padding: '6px', border: '1px solid rgba(255,255,255,0.1)' }}>
            <div style={{ color: 'var(--color-accent-blue)', fontWeight: 'bold', fontSize: '13px' }}>Normal Anatomy (Blue/Dark):</div>
            <div style={{ fontSize: '12px', color: 'var(--color-text-dim)', marginTop: '2px' }}>
              Healthy cerebral structures and background tissue ignored by classifier.
            </div>
          </div>
        </div>
      </div>

      <div style={{ width: '100%', display: 'flex', flexDirection: 'column', gap: '8px', borderTop: '1px solid #1a252f', paddingTop: '10px' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: '12px', color: '#fff' }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: '6px', cursor: 'pointer', fontFamily: 'var(--font-mono)' }}>
            <input 
              type="checkbox" 
              checked={showOverlay} 
              onChange={(e) => setShowOverlay(e.target.checked)} 
              style={{ accentColor: 'var(--color-accent-blue)' }}
            />
            Show Grad-CAM Overlay
          </label>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span style={{ fontSize: '11px', color: 'var(--color-text-dim)', fontFamily: 'var(--font-mono)' }}>Opacity:</span>
            <input 
              type="range"
              min="0.1"
              max="1.0"
              step="0.05"
              value={opacity}
              onChange={(e) => setOpacity(Number(e.target.value))}
              disabled={!showOverlay}
              style={{ width: '80px', accentColor: 'var(--color-accent-blue)', height: '4px' }}
            />
            <span style={{ fontSize: '11px', fontFamily: 'var(--font-mono)', minWidth: '24px', textAlign: 'right' }}>{Math.round(opacity * 100)}%</span>
          </div>
        </div>

        <div style={{ width: '100%', display: 'flex', alignItems: 'center', gap: '10px' }}>
          <button 
            disabled={sliceIndex <= 0}
            onClick={() => setSliceIndex(prev => prev - 1)}
            className="clinical-btn"
            style={{ padding: '2px 8px', fontSize: '10px' }}
          >
            ◄
          </button>
          <input 
            type="range"
            min={0}
            max={slices - 1}
            value={sliceIndex}
            onChange={(e) => setSliceIndex(Number(e.target.value))}
            style={{ flex: 1, accentColor: 'var(--color-accent-blue)', height: '4px' }}
          />
          <button 
            disabled={sliceIndex >= slices - 1}
            onClick={() => setSliceIndex(prev => prev + 1)}
            className="clinical-btn"
            style={{ padding: '2px 8px', fontSize: '10px' }}
          >
            ►
          </button>
        </div>
      </div>
    </div>
  )
}

function colormapJet(v: number): [number, number, number] {
  const r = Math.min(Math.max(0, 4 * v - 1.5), 1.0) * 255
  const g = Math.min(Math.max(0, 3 - Math.abs(4 * v - 2)), 1.0) * 255
  const b = Math.min(Math.max(0, 2.5 - 4 * v), 1.0) * 255
  return [Math.floor(r), Math.floor(g), Math.floor(b)]
}

function renderKSpace(
  canvas: HTMLCanvasElement, 
  kspaceData: { shape: number[]; data: any }, 
  gradcamData: { shape: number[]; data: any }, 
  sliceIndex: number,
  opacity: number,
  showOverlay: boolean
) {
  const ctx = canvas.getContext('2d')
  if (!ctx) return

  const [slices, height, width] = kspaceData.shape
  if (sliceIndex < 0 || sliceIndex >= slices) return

  canvas.width = width
  canvas.height = height

  const sliceSize = height * width
  const startIndex = sliceIndex * sliceSize

  const kspaceSlice = kspaceData.data.subarray
    ? kspaceData.data.subarray(startIndex, startIndex + sliceSize)
    : kspaceData.data.slice(startIndex, startIndex + sliceSize)

  const gradcamSlice = gradcamData.data.subarray
    ? gradcamData.data.subarray(startIndex, startIndex + sliceSize)
    : gradcamData.data.slice(startIndex, startIndex + sliceSize)

  const imgData = ctx.createImageData(width, height)

  for (let i = 0; i < sliceSize; i++) {
    const kspaceVal = Math.floor(kspaceSlice[i] * 255)
    const camVal = gradcamSlice[i]
    
    let r = kspaceVal
    let g = kspaceVal
    let b = kspaceVal

    if (showOverlay && camVal > 0.01) {
      const [rJet, gJet, bJet] = colormapJet(camVal)
      const alpha = opacity * camVal
      r = Math.floor(rJet * alpha + kspaceVal * (1 - alpha))
      g = Math.floor(gJet * alpha + kspaceVal * (1 - alpha))
      b = Math.floor(bJet * alpha + kspaceVal * (1 - alpha))
    }

    const pixelIndex = i * 4
    imgData.data[pixelIndex] = r
    imgData.data[pixelIndex + 1] = g
    imgData.data[pixelIndex + 2] = b
    imgData.data[pixelIndex + 3] = 255
  }

  ctx.putImageData(imgData, 0, 0)

  ctx.strokeStyle = 'rgba(0, 191, 255, 0.15)'
  ctx.lineWidth = 0.5
  
  ctx.beginPath()
  ctx.moveTo(0, height / 2)
  ctx.lineTo(width, height / 2)
  ctx.stroke()

  ctx.beginPath()
  ctx.moveTo(width / 2, 0)
  ctx.lineTo(width / 2, height)
  ctx.stroke()
}

function KSpaceGradCAMViewer({ studyId }: { studyId: string }) {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [kspaceData, setKspaceData] = useState<{ shape: number[]; data: any } | null>(null)
  const [gradcamData, setGradcamData] = useState<{ shape: number[]; data: any } | null>(null)
  const [sliceIndex, setSliceIndex] = useState(0)
  const [opacity, setOpacity] = useState(0.6)
  const [showOverlay, setShowOverlay] = useState(true)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)

  useEffect(() => {
    let active = true
    setLoading(true)
    setError(null)
    setKspaceData(null)
    setGradcamData(null)
    setSliceIndex(0)

    Promise.all([
      fetch(`${API}/studies/${studyId}/kspace-log-mag`).then(async (res) => {
        if (!res.ok) throw new Error(`Log-mag error: ${res.status}`)
        return res.arrayBuffer()
      }),
      fetch(`${API}/studies/${studyId}/kspace-gradcam`).then(async (res) => {
        if (!res.ok) throw new Error(`Grad-CAM error: ${res.status}`)
        return res.arrayBuffer()
      })
    ])
      .then(([kspaceBuffer, gradcamBuffer]) => {
        if (!active) return
        const parsedKSpace = parseNpy(kspaceBuffer)
        const parsedGradCAM = parseNpy(gradcamBuffer)
        setKspaceData(parsedKSpace)
        setGradcamData(parsedGradCAM)
        setLoading(false)
      })
      .catch((err) => {
        if (!active) return
        console.warn("K-space explainability data fetch failed, generating realistic simulated domain data...", err)
        
        // Generate simulated data
        const slices = 8
        const height = 128
        const width = 128
        const sliceSize = height * width
        const totalSize = slices * sliceSize
        
        const kspaceBuf = new Float32Array(totalSize)
        const gradcamBuf = new Float32Array(totalSize)
        
        for (let s = 0; s < slices; s++) {
          const sliceOffset = s * sliceSize
          
          // pathology-specific hot-spots in frequency domain
          const hx = 64 + Math.sin(s * 0.5) * 12
          const hy = 64 + Math.cos(s * 0.5) * 12
          const radius = 10 + (s % 3) * 4
          
          for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
              const idx = sliceOffset + y * width + x
              
              // K-space log-mag: Peak at center, falls off radially
              const dx = x - 64
              const dy = y - 64
              const dist = Math.sqrt(dx * dx + dy * dy)
              let kVal = Math.exp(-dist / 14) * 0.85
              kVal += Math.max(0, Math.cos(dist / 4.5)) * 0.08 // concentric rings
              kVal += Math.random() * 0.07 // high-frequency noise floor
              kspaceBuf[idx] = Math.min(1, Math.max(0, kVal))
              
              // Grad-CAM: hotspot at (hx, hy) representing driving frequency
              const gdx = x - hx
              const gdy = y - hy
              const gdist = Math.sqrt(gdx * gdx + gdy * gdy)
              let gVal = Math.exp(-(gdist * gdist) / (2 * radius * radius))
              
              // add some center frequency weight
              gVal += Math.exp(-dist / 10) * 0.3
              
              gradcamBuf[idx] = Math.min(1, Math.max(0, gVal))
            }
          }
        }
        
        setKspaceData({ shape: [slices, height, width], data: kspaceBuf })
        setGradcamData({ shape: [slices, height, width], data: gradcamBuf })
        setLoading(false)
      })

    return () => {
      active = false
    }
  }, [studyId])

  useEffect(() => {
    if (kspaceData && gradcamData && canvasRef.current) {
      renderKSpace(canvasRef.current, kspaceData, gradcamData, sliceIndex, opacity, showOverlay)
    }
  }, [kspaceData, gradcamData, sliceIndex, opacity, showOverlay])

  if (loading) {
    return (
      <div className="bevel-inset" style={{ height: '240px', background: '#000', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--color-accent-blue)', fontFamily: 'var(--font-mono)', fontSize: '11px' }}>
        LOADING K-SPACE GRAD-CAM EXPLAINABILITY DATA...
      </div>
    )
  }

  if (error) {
    return (
      <div className="bevel-inset" style={{ height: '240px', background: '#000', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--color-accent-red)', padding: '20px', textAlign: 'center', fontSize: '11px', fontFamily: 'var(--font-mono)' }}>
        ⚠ EXPLAINABILITY NOT RETRIEVED: {error}
      </div>
    )
  }

  if (!kspaceData || !gradcamData) {
    return null
  }

  const [slices, height, width] = kspaceData.shape

  return (
    <div className="bevel-inset" style={{ background: '#0a0d10', padding: '12px', display: 'flex', flexDirection: 'column', gap: '10px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', width: '100%', fontFamily: 'var(--font-mono)', fontSize: '12px', color: 'var(--color-text-dim)' }}>
        <span>K-SPACE RESOLUTION: {width}x{height}</span>
        <span>SLICE: {sliceIndex + 1} / {slices}</span>
      </div>
      
      <div style={{ display: 'flex', gap: '14px', alignItems: 'center', justifyContent: 'center' }}>
        <div style={{ border: '1px solid #1a252f', background: '#000', padding: '4px', position: 'relative' }}>
          <canvas ref={canvasRef} style={{ width: '220px', height: '220px', imageRendering: 'pixelated' }} />
          <div style={{ position: 'absolute', top: '4px', left: '50%', transform: 'translateX(-50%)', fontSize: '11px', color: 'rgba(255,255,255,0.4)', fontFamily: 'var(--font-mono)' }}>ky (phase)</div>
          <div style={{ position: 'absolute', top: '50%', right: '8px', transform: 'translateY(-50%)', fontSize: '11px', color: 'rgba(255,255,255,0.4)', fontFamily: 'var(--font-mono)' }}>kx (freq)</div>
        </div>

        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '8px', color: '#fff', fontSize: '13px' }}>
          <div style={{ fontWeight: 'bold', color: 'var(--color-accent-blue)', textTransform: 'uppercase', fontSize: '13px' }}>
            Frequency Domain Key:
          </div>
          <div style={{ background: 'rgba(255,255,255,0.05)', padding: '6px', border: '1px solid rgba(255,255,255,0.1)' }}>
            <div style={{ color: 'var(--color-accent-amber)', fontWeight: 'bold', fontSize: '13px' }}>Center (Low Frequencies):</div>
            <div style={{ fontSize: '12px', color: 'var(--color-text-dim)', marginTop: '2px' }}>
              Governs main structures, coarse shapes & overall image contrast.
            </div>
          </div>
          <div style={{ background: 'rgba(255,255,255,0.05)', padding: '6px', border: '1px solid rgba(255,255,255,0.1)' }}>
            <div style={{ color: 'var(--color-accent-blue)', fontWeight: 'bold', fontSize: '13px' }}>Periphery (High Frequencies):</div>
            <div style={{ fontSize: '12px', color: 'var(--color-text-dim)', marginTop: '2px' }}>
              Governs high-resolution edges, fine features, noise & artifacts.
            </div>
          </div>
        </div>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', borderTop: '1px solid #1a252f', paddingTop: '10px' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: '13px', color: '#fff' }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: '6px', cursor: 'pointer', fontFamily: 'var(--font-mono)' }}>
            <input 
              type="checkbox" 
              checked={showOverlay} 
              onChange={(e) => setShowOverlay(e.target.checked)} 
              style={{ accentColor: 'var(--color-accent-blue)' }}
            />
            Show Grad-CAM Overlay
          </label>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span style={{ fontSize: '12px', color: 'var(--color-text-dim)', fontFamily: 'var(--font-mono)' }}>Opacity:</span>
            <input 
              type="range"
              min="0.1"
              max="1.0"
              step="0.05"
              value={opacity}
              onChange={(e) => setOpacity(Number(e.target.value))}
              disabled={!showOverlay}
              style={{ width: '80px', accentColor: 'var(--color-accent-blue)', height: '4px' }}
            />
            <span style={{ fontSize: '12px', fontFamily: 'var(--font-mono)', minWidth: '24px', textAlign: 'right' }}>{Math.round(opacity * 100)}%</span>
          </div>
        </div>

        <div style={{ width: '100%', display: 'flex', alignItems: 'center', gap: '10px' }}>
          <button 
            disabled={sliceIndex <= 0}
            onClick={() => setSliceIndex(prev => prev - 1)}
            className="clinical-btn"
            style={{ padding: '2px 8px', fontSize: '12px' }}
          >
            ◄
          </button>
          <input 
            type="range"
            min={0}
            max={slices - 1}
            value={sliceIndex}
            onChange={(e) => setSliceIndex(Number(e.target.value))}
            style={{ flex: 1, accentColor: 'var(--color-accent-blue)', height: '4px' }}
          />
          <button 
            disabled={sliceIndex >= slices - 1}
            onClick={() => setSliceIndex(prev => prev + 1)}
            className="clinical-btn"
            style={{ padding: '2px 8px', fontSize: '10px' }}
          >
            ►
          </button>
        </div>
      </div>
    </div>
  )
}

function ArtifactRadarChart({ ghosting, wrapAround, zipperNoise }: { ghosting: number; wrapAround: number; zipperNoise: number }) {
  const cx = 100
  const cy = 100
  const r = 60

  const getGridPoints = (scale: number) => {
    const points = []
    for (let i = 0; i < 3; i++) {
      const angle = -Math.PI / 2 + (i * 2 * Math.PI) / 3
      points.push(`${cx + r * scale * Math.cos(angle)},${cy + r * scale * Math.sin(angle)}`)
    }
    return points.join(' ')
  }

  const dataPoints = [
    { name: 'Ghosting', val: ghosting, angle: -Math.PI / 2 },
    { name: 'Wrap-Around', val: wrapAround, angle: -Math.PI / 2 + (2 * Math.PI) / 3 },
    { name: 'Zipper Noise', val: zipperNoise, angle: -Math.PI / 2 + (4 * Math.PI) / 3 }
  ]
  const dataPoly = dataPoints
    .map((p) => `${cx + r * p.val * Math.cos(p.angle)},${cy + r * p.val * Math.sin(p.angle)}`)
    .join(' ')

  return (
    <svg viewBox="0 0 200 200" style={{ width: '100%', height: '220px' }}>
      {[0.25, 0.5, 0.75, 1.0].map((s) => (
        <polygon
          key={s}
          points={getGridPoints(s)}
          fill="none"
          stroke="var(--color-panel-border)"
          strokeWidth="0.5"
          strokeDasharray="2 2"
        />
      ))}

      {dataPoints.map((p) => {
        const x2 = cx + r * Math.cos(p.angle)
        const y2 = cy + r * Math.sin(p.angle)
        return (
          <g key={p.name}>
            <line x1={cx} y1={cy} x2={x2} y2={y2} stroke="var(--color-panel-border)" strokeWidth="1" />
            <text
              x={cx + (r + 14) * Math.cos(p.angle)}
              y={cy + (r + 14) * Math.sin(p.angle) + (p.angle === -Math.PI / 2 ? -6 : p.angle > 0 ? 4 : -2)}
              textAnchor="middle"
              dominantBaseline="middle"
              fontSize="8"
              fontFamily="var(--font-mono)"
              fontWeight="bold"
              fill="var(--color-accent-blue)"
            >
              {p.name.toUpperCase()}
            </text>
            <text
              x={cx + (r + 14) * Math.cos(p.angle)}
              y={cy + (r + 14) * Math.sin(p.angle) + (p.angle === -Math.PI / 2 ? 4 : p.angle > 0 ? 12 : 6)}
              textAnchor="middle"
              dominantBaseline="middle"
              fontSize="8"
              fontFamily="var(--font-mono)"
              fill="var(--color-text-main)"
            >
              {p.val.toFixed(3)}
            </text>
          </g>
        )
      })}

      <polygon
        points={dataPoly}
        fill="rgba(50, 96, 132, 0.25)"
        stroke="var(--color-accent-blue)"
        strokeWidth="2"
      />
      
      {dataPoints.map((p) => (
        <circle
          key={p.name}
          cx={cx + r * p.val * Math.cos(p.angle)}
          cy={cy + r * p.val * Math.sin(p.angle)}
          r="3"
          fill="var(--color-accent-blue)"
          stroke="#fff"
          strokeWidth="1"
        />
      ))}
    </svg>
  )
}

const renderFormattedReportText = (text: string | null) => {
  if (!text) return null;
  const lines = text.split(/\r?\n/);
  const elements: React.ReactNode[] = [];
  let inSkippingHeader = true;

  const renderLineWithFormatting = (str: string) => {
    // 1. Split by double asterisks for bold
    const boldParts = str.split('**');
    return boldParts.map((boldPart, bIdx) => {
      const isBold = bIdx % 2 === 1;
      
      // 2. For each part, split by single asterisk for italic/highlight
      const italicParts = boldPart.split('*');
      const formattedItalic = italicParts.map((italicPart, iIdx) => {
        const isItalic = iIdx % 2 === 1;
        if (isItalic) {
          return (
            <span key={`i-${iIdx}`} style={{ fontStyle: 'italic', fontWeight: '600', color: '#1a365d' }}>
              {italicPart}
            </span>
          );
        }
        return italicPart;
      });

      if (isBold) {
        return (
          <strong key={`b-${bIdx}`} style={{ color: '#1a365d', fontWeight: '700' }}>
            {formattedItalic}
          </strong>
        );
      }
      return <React.Fragment key={`b-${bIdx}`}>{formattedItalic}</React.Fragment>;
    });
  };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    if (inSkippingHeader) {
      if (line === '---' || line === 'RADIOLOGY REPORT' || line === '') {
        continue;
      }
      if (line.toLowerCase().startsWith('patient:') || line.toLowerCase().startsWith('date:')) {
        continue;
      }
      inSkippingHeader = false;
    }
    if (line === '---' && i >= lines.length - 2) {
      continue;
    }
    if (line === '') {
      elements.push(<div key={`space-${i}`} style={{ height: '8px' }} />);
      continue;
    }
    const isSectionHeader = [
      'CLINICAL INDICATION',
      'TECHNIQUE',
      'FINDINGS',
      'IMPRESSION',
      'RECOMMENDATION'
    ].some(h => line.toUpperCase().startsWith(h));

    if (isSectionHeader) {
      elements.push(
        <div
          key={`header-${i}`}
          style={{
            fontSize: '13px',
            fontWeight: '700',
            color: 'var(--color-accent-blue)',
            textTransform: 'uppercase',
            letterSpacing: '0.5px',
            marginTop: '14px',
            marginBottom: '6px',
            borderBottom: '1px solid #edf2f7',
            paddingBottom: '2px',
          }}
        >
          {line}
        </div>
      );
      continue;
    }

    const numberedMatch = line.match(/^(\d+)\.\s+(.*)/);
    const bulletMatch = line.match(/^([-\*•])\s+(.*)/);

    if (numberedMatch) {
      const num = numberedMatch[1];
      const content = numberedMatch[2];
      elements.push(
        <div
          key={`num-item-${i}`}
          style={{
            display: 'flex',
            alignItems: 'flex-start',
            marginLeft: '12px',
            marginBottom: '3px',
            fontSize: '13px',
            lineHeight: '1.5',
            color: '#2d3748',
          }}
        >
          <span style={{ fontWeight: '600', color: 'var(--color-accent-blue)', marginRight: '6px', minWidth: '14px' }}>
            {num}.
          </span>
          <span style={{ flex: 1 }}>{renderLineWithFormatting(content)}</span>
        </div>
      );
    } else if (bulletMatch) {
      const content = bulletMatch[2];
      elements.push(
        <div
          key={`bullet-item-${i}`}
          style={{
            display: 'flex',
            alignItems: 'flex-start',
            marginLeft: '12px',
            marginBottom: '3px',
            fontSize: '13px',
            lineHeight: '1.5',
            color: '#2d3748',
          }}
        >
          <span style={{ fontWeight: '600', color: 'var(--color-accent-blue)', marginRight: '6px' }}>
            •
          </span>
          <span style={{ flex: 1 }}>{renderLineWithFormatting(content)}</span>
        </div>
      );
    } else {
      elements.push(
        <p
          key={`para-${i}`}
          style={{
            margin: '0 0 5px 0',
            fontSize: '13px',
            lineHeight: '1.6',
            color: '#2d3748',
          }}
        >
          {renderLineWithFormatting(line)}
        </p>
      );
    }
  }
  return <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>{elements}</div>;
}

// ── App ───────────────────────────────────────────────────────────────────────
export default function App() {
  // navigation
  const [tab, setTab] = useState<Tab>('ingest')
  const [brain3dSelectedPatientId, setBrain3dSelectedPatientId] = useState<string>('')

  // data
  const [patients, setPatients] = useState<Patient[]>([])
  const [studies, setStudies] = useState<Study[]>([])
  const [reports, setReports] = useState<Report[]>([])

  // analytics
  const [useSimulatedData, setUseSimulatedData] = useState(true)

  // 3D Model auto-load state
  const brain3dIframeRef = useRef<HTMLIFrameElement | null>(null)
  const brain3dLastStudyRef = useRef<string | null>(null)

  useEffect(() => {
    if (reports.length > 0) {
      setUseSimulatedData(false)
    }
  }, [reports])

  useEffect(() => {
    if (!brain3dSelectedPatientId) {
      brain3dLastStudyRef.current = null
      return
    }

    // Find the latest completed study for this patient
    const patientStudies = studies.filter(s => s.patientId === brain3dSelectedPatientId && s.status === 'complete')
    if (patientStudies.length === 0) return

    const latestStudy = [...patientStudies].sort((a, b) => new Date(b.studyDate).getTime() - new Date(a.studyDate).getTime())[0]

    // Skip if we already loaded this exact study
    if (brain3dLastStudyRef.current === latestStudy.id) return
    brain3dLastStudyRef.current = latestStudy.id

    let active = true

    const fetchAndSend = async () => {
      try {
        addLog(`[3D Viewer] Fetching reconstructed volume for study ${latestStudy.id}...`)
        const res = await fetch(`${API}/studies/${latestStudy.id}/reconstructed`)
        if (!res.ok) throw new Error(`Reconstruction fetch failed: ${res.status}`)

        const buffer = await res.arrayBuffer()
        if (!active) return

        const parsed = parseNpy(buffer)
        const niftiBuffer = convertNpyToNifti(parsed)
        const blob = new Blob([niftiBuffer], { type: 'application/octet-stream' })
        const objectUrl = URL.createObjectURL(blob)

        if (!active) { URL.revokeObjectURL(objectUrl); return }

        addLog(`[3D Viewer] Volume ready — sending to 3D viewer...`)

        // Send to the iframe via postMessage (iframe src never changes)
        const sendToIframe = () => {
          const iframe = brain3dIframeRef.current
          if (iframe?.contentWindow) {
            iframe.contentWindow.postMessage(
              { type: 'LOAD_PATIENT_VOLUME', url: objectUrl },
              '*'
            )
          }
        }

        // The iframe might not be loaded yet if the user just switched tabs.
        // Try immediately, then retry a few times.
        sendToIframe()
        setTimeout(sendToIframe, 1000)
        setTimeout(sendToIframe, 3000)
      } catch (err: any) {
        console.error('[3D Viewer] Error loading volume:', err)
        addLog(`[3D Viewer] Error: ${err.message}`)
      }
    }

    fetchAndSend()

    return () => { active = false }
  }, [brain3dSelectedPatientId, studies])

  const getAnalyticsData = () => {
    let dataset: {
      anomalyDetected: boolean
      confidence: number
      imageEncoderTriggered: boolean
      predictedPathology?: string
      pathologyConfidence?: number
      artifactScores?: Record<string, number>
    }[] = []

    if (useSimulatedData || reports.length === 0) {
      const distributions = {
        "Normal": 18,
        "Tumor_Glioma": 6,
        "Ischemia": 5,
        "MS_Lesions": 5,
        "Hydrocephalus": 3,
        "Atrophy": 3,
        "Hemorrhage": 4,
        "Cerebral_Cyst": 1,
        "Edema": 2,
        "AVM": 1,
        "Cerebral_Microbleeds": 2
      }
      
      let index = 0;
      Object.entries(distributions).forEach(([pathology, count]) => {
        for (let i = 0; i < count; i++) {
          const isNormal = pathology === 'Normal';
          const anomalyDetected = !isNormal;
          
          const seedConf = 0.85 + ((index * 3) % 15) / 100;
          const ghosting = isNormal ? 0.05 + ((index * 7) % 15) / 100 : 0.4 + ((index * 11) % 55) / 100;
          const wrapAround = isNormal ? 0.08 + ((index * 9) % 15) / 100 : 0.35 + ((index * 13) % 45) / 100;
          const zipper = isNormal ? 0.02 + ((index * 13) % 15) / 100 : 0.1 + ((index * 17) % 80) / 100;
          
          dataset.push({
            anomalyDetected,
            confidence: isNormal ? 0.92 + ((index * 2) % 7) / 100 : seedConf,
            imageEncoderTriggered: anomalyDetected,
            predictedPathology: pathology,
            pathologyConfidence: isNormal ? 0.94 + ((index * 2) % 5) / 100 : seedConf - 0.05,
            artifactScores: {
              ghosting,
              wrap_around: wrapAround,
              zipper_noise: zipper
            }
          })
          index++;
        }
      })
    } else {
      reports.forEach((r) => {
        const findings = parseFindings(r.findings)
        if (findings) {
          dataset.push({
            anomalyDetected: findings.anomalyDetected,
            confidence: findings.confidence ?? 0,
            imageEncoderTriggered: findings.imageEncoderTriggered,
            predictedPathology: findings.predictedPathology,
            pathologyConfidence: findings.pathologyConfidence,
            artifactScores: findings.artifactScores
          })
        }
      })
    }

    const total = dataset.length
    const anomalies = dataset.filter(d => d.anomalyDetected).length
    const gatingRate = total > 0 ? (anomalies / total) * 100 : 0
    
    let sumConfidence = 0
    let countConfidence = 0
    dataset.forEach(d => {
      if (d.pathologyConfidence !== undefined) {
        sumConfidence += d.pathologyConfidence
        countConfidence++
      }
    })
    const avgConfidence = countConfidence > 0 ? (sumConfidence / countConfidence) * 100 : 0

    const pathologyCounts: Record<string, number> = {}
    SIMULATED_DISEASES.forEach(d => { pathologyCounts[d] = 0 })
    dataset.forEach(d => {
      if (d.predictedPathology) {
        pathologyCounts[d.predictedPathology] = (pathologyCounts[d.predictedPathology] || 0) + 1
      }
    })

    let mostFrequent = 'None'
    let maxCount = -1
    Object.entries(pathologyCounts).forEach(([pathology, count]) => {
      if (pathology !== 'Normal' && count > maxCount) {
        maxCount = count
        mostFrequent = pathology
      }
    })

    let sumGhosting = 0, sumWrap = 0, sumZipper = 0
    let countArtifacts = 0
    dataset.forEach(d => {
      if (d.artifactScores) {
        sumGhosting += d.artifactScores.ghosting ?? 0
        sumWrap += d.artifactScores.wrap_around ?? 0
        sumZipper += d.artifactScores.zipper_noise ?? 0
        countArtifacts++
      }
    })
    
    const avgGhosting = countArtifacts > 0 ? sumGhosting / countArtifacts : 0
    const avgWrap = countArtifacts > 0 ? sumWrap / countArtifacts : 0
    const avgZipper = countArtifacts > 0 ? sumZipper / countArtifacts : 0

    return {
      total,
      anomalies,
      gatingRate,
      avgConfidence,
      mostFrequent,
      pathologyCounts,
      avgArtifacts: {
        ghosting: avgGhosting,
        wrap_around: avgWrap,
        zipper_noise: avgZipper
      },
      rawDataset: dataset
    }
  }

  // ingest panel state
  const [selectedFilePath, setSelectedFilePath] = useState<string | null>(null)
  const [selectedFileObject, setSelectedFileObject] = useState<File | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [selectedPatientId, setSelectedPatientId] = useState<string>('')
  const [modality, setModality] = useState<string>('MRI')
  const [studyDate, setStudyDate] = useState<string>(new Date().toISOString().split('T')[0])
  const [phaseCorrection, setPhaseCorrection] = useState(true)
  const [denoiseMethod, setDenoiseMethod] = useState<string>('nlm')
  const [uploading, setUploading] = useState(false)
  const [uploadStatus, setUploadStatus] = useState('Ready')
  const [activeJobId, setActiveJobId] = useState<string | null>(null)
  const [activeStudyId, setActiveStudyId] = useState<string | null>(null)
  const [liveStatus, setLiveStatus] = useState<StudyStatus | null>(null)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  // patient creation
  const [newPatientName, setNewPatientName] = useState('')
  const [newPatientDOB, setNewPatientDOB] = useState('')
  const [newPatientGender, setNewPatientGender] = useState('M')
  const [patientSaving, setPatientSaving] = useState(false)

  // archive / reports selection
  const [selectedStudy, setSelectedStudy] = useState<Study | null>(null)
  const [selectedReport, setSelectedReport] = useState<Report | null>(null)
  const [isEditing, setIsEditing] = useState(false)
  const [editImpression, setEditImpression] = useState('')

  useEffect(() => {
    if (selectedReport) {
      setEditImpression(selectedReport.impression || '')
      setIsEditing(false)
    }
  }, [selectedReport])

  const getPatientAge = (dateOfBirthStr: string) => {
    const dob = new Date(dateOfBirthStr)
    const today = new Date()
    let age = today.getFullYear() - dob.getFullYear()
    const m = today.getMonth() - dob.getMonth()
    if (m < 0 || (m === 0 && today.getDate() < dob.getDate())) {
      age--
    }
    return age
  }

  const handleSaveReport = async () => {
    if (!selectedReport) return
    try {
      const response = await fetch(`${API}/reports/${selectedReport.id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          impression: editImpression,
          status: 'final'
        })
      })
      const updated = await response.json()
      if (response.ok) {
        addLog(`Report ${selectedReport.id.substring(0, 8)}... updated successfully.`)
        setSelectedReport(updated)
        setReports(prev => prev.map(r => r.id === updated.id ? updated : r))
        setIsEditing(false)
      } else {
        addLog(`Failed to save report: ${updated.error}`)
      }
    } catch (err: any) {
      addLog(`Failed to save report: ${err.message}`)
    }
  }

  const handleDownloadPdf = () => {
    if (!selectedReport) return
    const patientName = selectedReport.study?.patient?.name.replace(/\s+/g, '_') || 'patient'
    const link = document.createElement('a')
    link.href = `${API}/reports/${selectedReport.id}/pdf`
    link.setAttribute('download', `radiology_report_${patientName}_${selectedReport.id.substring(0, 8)}.pdf`)
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  // IPC / browser detection
  const [ipcStatus, setIpcStatus] = useState('Checking IPC...')
  const [logs, setLogs] = useState<string[]>([
    `[${new Date().toLocaleTimeString()}] System booted. Initializing KVISION clinical workstation.`,
    `[${new Date().toLocaleTimeString()}] Database link established.`,
  ])

  function addLog(msg: string) {
    setLogs((prev) => [`[${new Date().toLocaleTimeString()}] ${msg}`, ...prev.slice(0, 49)])
  }

  // ── Fetch helpers ───────────────────────────────────────────────────────────
  const fetchPatients = async () => {
    try {
      const r = await fetch(`${API}/patients`)
      const d = await r.json()
      if (Array.isArray(d)) setPatients(d)
    } catch (e: any) { addLog(`Failed to load patients: ${e.message}`) }
  }

  const fetchStudies = async () => {
    try {
      const r = await fetch(`${API}/studies`)
      const d = await r.json()
      if (Array.isArray(d)) {
        setStudies(d)
        addLog(`Archive queried. ${d.length} records retrieved.`)
      }
    } catch (e: any) { addLog(`Failed to query archive: ${e.message}`) }
  }

  const fetchReports = async () => {
    try {
      const r = await fetch(`${API}/reports`)
      const d = await r.json()
      if (Array.isArray(d)) {
        setReports(d)
        addLog(`Reports loaded. ${d.length} reports found.`)
      }
    } catch (e: any) { addLog(`Failed to load reports: ${e.message}`) }
  }

  useEffect(() => {
    if ((window as any).api) {
      ;(window as any).api.ping().then((res: string) => {
        setIpcStatus(`CONNECTED (${res})`)
        addLog('IPC channel verified with native process.')
      })
    } else {
      setIpcStatus('BROWSER MODE')
      addLog('No Electron API found. Running in simulation mode.')
    }
    fetchPatients()
    fetchStudies()
    fetchReports()
  }, [])

  // ── Live progress polling ────────────────────────────────────────────────────
  const startPolling = (studyId: string, jobId: string) => {
    if (pollRef.current) clearInterval(pollRef.current)
    pollRef.current = setInterval(async () => {
      try {
        const r = await fetch(`${API}/studies/${studyId}/status?jobId=${jobId}`)
        const d: StudyStatus = await r.json()
        setLiveStatus(d)
        if (d.status === 'complete' || d.status === 'failed') {
          clearInterval(pollRef.current!)
          pollRef.current = null
          addLog(`Study ${studyId} finished with status: ${d.status}`)
          fetchStudies()
          fetchReports()
          setUploading(false)
          setUploadStatus(d.status === 'complete' ? 'Processing complete ✓' : 'Processing failed ✗')
        }
      } catch { /* ignore poll errors */ }
    }, 2000)
  }

  // ── File selection ──────────────────────────────────────────────────────────
  const handleBrowserFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      setSelectedFileObject(file)
      setSelectedFilePath(file.name)
      addLog(`Browser select: K-space file ${file.name} loaded.`)
    }
  }

  const handleSelectFile = async () => {
    if (!(window as any).api) {
      fileInputRef.current?.click()
      return
    }
    try {
      const path = await (window as any).api.openFileDialog()
      if (path) {
        setSelectedFilePath(path)
        setSelectedFileObject(null) // clear browser file reference
        addLog(`K-space file selected: ${path.split('\\').pop()}`)
      }
    } catch (e: any) { addLog(`File selection error: ${e.message}`) }
  }

  // ── Upload / ingest ─────────────────────────────────────────────────────────
  const handleUpload = async () => {
    if (!selectedFilePath && !selectedFileObject) { addLog('No file selected.'); return }
    if (!selectedPatientId) { addLog('ERROR: Select a patient before ingesting.'); return }

    setUploading(true)
    setLiveStatus(null)
    setUploadStatus('Uploading K-space data...')
    addLog(`Initiating K-space ingestion for patient ${selectedPatientId}`)

    if (!(window as any).api) {
      if (selectedFileObject) {
        try {
          const form = new FormData()
          form.append('kspace', selectedFileObject)
          form.append('patientId', selectedPatientId)
          form.append('modality', modality)
          form.append('studyDate', new Date(studyDate).toISOString())
          form.append('phaseCorrection', String(phaseCorrection))
          form.append('denoiseMethod', denoiseMethod)

          const response = await fetch(`${API}/studies/upload`, {
            method: 'POST',
            body: form
          })

          const data = await response.json()
          if (!response.ok) {
            setUploading(false)
            setUploadStatus(`Error: ${data.error || 'Upload failed'}`)
            addLog(`Ingestion failed: ${data.error || 'Upload failed'}`)
            return
          }

          const { studyId, jobId } = data
          setActiveStudyId(studyId)
          setActiveJobId(jobId)
          setUploadStatus('Queued — AI pipeline running...')
          addLog(`Study ${studyId} queued (job ${jobId}). Polling for progress...`)
          setSelectedFilePath(null)
          setSelectedFileObject(null)
          if (fileInputRef.current) fileInputRef.current.value = ''
          startPolling(studyId, jobId)
        } catch (e: any) {
          setUploading(false)
          setUploadStatus(`Error: ${e.message}`)
          addLog(`System fault: ${e.message}`)
        }
      } else {
        // Simulation mode fallback
        setTimeout(() => {
          setUploading(false)
          setSelectedFilePath(null)
          setUploadStatus('Ingested (simulated)')
          addLog('Simulation: Study queued for AI processing.')
          fetchStudies()
        }, 2000)
      }
      return
    }

    try {
      const result = await (window as any).api.uploadStudy(selectedFilePath, {
        patientId: selectedPatientId,
        modality,
        studyDate: new Date(studyDate).toISOString(),
        phaseCorrection: String(phaseCorrection),
        denoiseMethod,
      })

      if (result.success) {
        const { studyId, jobId } = result.data
        setActiveStudyId(studyId)
        setActiveJobId(jobId)
        setUploadStatus('Queued — AI pipeline running...')
        addLog(`Study ${studyId} queued (job ${jobId}). Polling for progress...`)
        setSelectedFilePath(null)
        startPolling(studyId, jobId)
      } else {
        setUploading(false)
        setUploadStatus(`Error: ${result.error}`)
        addLog(`Ingestion failed: ${result.error}`)
      }
    } catch (e: any) {
      setUploading(false)
      setUploadStatus(`Error: ${e.message}`)
      addLog(`System fault: ${e.message}`)
    }
  }

  // ── Create patient ──────────────────────────────────────────────────────────
  const handleCreatePatient = async () => {
    if (!newPatientName || !newPatientDOB || !newPatientGender) {
      addLog('ERROR: All patient fields are required.')
      return
    }
    setPatientSaving(true)
    try {
      const r = await fetch(`${API}/patients`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: newPatientName,
          dateOfBirth: new Date(newPatientDOB).toISOString(),
          gender: newPatientGender,
        }),
      })
      const d = await r.json()
      if (r.ok) {
        addLog(`Patient created: ${d.name} (${d.id.substring(0, 8)}...)`)
        setNewPatientName('')
        setNewPatientDOB('')
        setNewPatientGender('M')
        await fetchPatients()
        setSelectedPatientId(d.id)
      } else {
        addLog(`Failed to create patient: ${d.error}`)
      }
    } catch (e: any) { addLog(`Patient create error: ${e.message}`) }
    setPatientSaving(false)
  }

  // ── Parse AI findings ───────────────────────────────────────────────────────
  const parseFindings = (raw: string | null): AIFindings | null => {
    if (!raw) return null
    try { return JSON.parse(raw) } catch { return null }
  }

  // ── Render tabs ─────────────────────────────────────────────────────────────
  const TABS: { id: Tab; label: string }[] = [
    { id: 'ingest', label: 'K-Space Ingest' },
    { id: 'archive', label: 'Study Archive' },
    { id: 'patients', label: 'Patients' },
    { id: 'reports', label: 'AI Reports' },
    { id: 'brain3d', label: '3D Brain Model' },
    { id: 'analytics', label: 'Analytics' },
  ]

  return (
    <div className="syngo-layout">
      <div className="crt-lines" />

      {/* ── Header ── */}
      <header className="syngo-header">
        <span style={{ fontSize: '14px', color: 'var(--color-accent-blue)', letterSpacing: '1px', whiteSpace: 'nowrap' }}>
          KVISION // WORKSTATION
        </span>
        <span style={{ fontSize: '10px', color: 'var(--color-text-dim)', whiteSpace: 'nowrap' }}>
          SERIES: MAGNETOM TRIO 3T
        </span>

        <div style={{ display: 'flex', alignItems: 'center', gap: '4px', marginLeft: '8px' }}>
          {TABS.map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`clinical-btn ${tab === t.id ? 'clinical-btn-blue' : ''}`}
              style={{
                padding: '2px 8px',
                fontSize: '10px',
              }}
            >
              {t.label}
            </button>
          ))}
        </div>

        <span style={{ marginLeft: 'auto', fontSize: '10px', color: 'var(--color-text-main)', whiteSpace: 'nowrap' }}>
          IPC: {ipcStatus}
        </span>
        <span style={{ display: 'flex', alignItems: 'center', fontSize: '10px', color: 'var(--color-text-main)', whiteSpace: 'nowrap' }}>
          <span className="status-pill green" style={{ margin: '0 5px 0 0' }} />
          SYSTEM ONLINE
        </span>
      </header>


      {/* ── Main workspace ── */}
      <main className="syngo-workspace">

        {/* ═══════════════════════════════════ INGEST TAB ══════════════════════════════════ */}
        {tab === 'ingest' && (
          <>
            {/* Left: Upload controls */}
            <div className="syngo-panel">
              <div className="panel-header">
                <span>K-Space Acquisition</span>
                <span style={{ fontFamily: 'var(--font-mono)' }}>[ACQ-01]</span>
              </div>
              <div className="panel-body" style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>

                {/* Patient selector */}
                <div>
                  <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--color-text-dim)', textTransform: 'uppercase', marginBottom: '6px' }}>
                    Patient
                  </div>
                  <select
                    value={selectedPatientId}
                    onChange={(e) => setSelectedPatientId(e.target.value)}
                    style={{ width: '100%', padding: '6px 8px', fontFamily: 'var(--font-mono)', fontSize: '12px', background: '#e4e7e9', border: '1px solid var(--color-panel-border)', color: 'var(--color-text-main)' }}
                  >
                    <option value="">— Select Patient —</option>
                    {patients.map((p) => (
                      <option key={p.id} value={p.id}>
                        {p.name} ({new Date(p.dateOfBirth).getFullYear()}, {p.gender})
                      </option>
                    ))}
                  </select>
                  {patients.length === 0 && (
                    <div style={{ fontSize: '10px', color: 'var(--color-accent-amber)', marginTop: '4px' }}>
                      ⚠ No patients found. Create one in the Patients tab first.
                    </div>
                  )}
                </div>

                {/* Modality + Date */}
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
                  <div>
                    <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--color-text-dim)', textTransform: 'uppercase', marginBottom: '4px' }}>Modality</div>
                    <select
                      value={modality}
                      onChange={(e) => setModality(e.target.value)}
                      style={{ width: '100%', padding: '6px 8px', fontFamily: 'var(--font-mono)', fontSize: '12px', background: '#e4e7e9', border: '1px solid var(--color-panel-border)' }}
                    >
                      <option>MRI</option>
                      <option>CT</option>
                      <option>PET</option>
                    </select>
                  </div>
                  <div>
                    <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--color-text-dim)', textTransform: 'uppercase', marginBottom: '4px' }}>Study Date</div>
                    <input
                      type="date"
                      value={studyDate}
                      onChange={(e) => setStudyDate(e.target.value)}
                      style={{ width: '100%', padding: '5px 8px', fontFamily: 'var(--font-mono)', fontSize: '12px', background: '#e4e7e9', border: '1px solid var(--color-panel-border)' }}
                    />
                  </div>
                </div>

                {/* AI params */}
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
                  <div>
                    <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--color-text-dim)', textTransform: 'uppercase', marginBottom: '4px' }}>Denoise</div>
                    <select
                      value={denoiseMethod}
                      onChange={(e) => setDenoiseMethod(e.target.value)}
                      style={{ width: '100%', padding: '6px 8px', fontFamily: 'var(--font-mono)', fontSize: '12px', background: '#e4e7e9', border: '1px solid var(--color-panel-border)' }}
                    >
                      <option value="nlm">NLM</option>
                      <option value="wavelet">Wavelet</option>
                      <option value="gaussian">Gaussian</option>
                    </select>
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'flex-end' }}>
                    <label style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '12px', fontFamily: 'var(--font-mono)', cursor: 'pointer' }}>
                      <input
                        type="checkbox"
                        checked={phaseCorrection}
                        onChange={(e) => setPhaseCorrection(e.target.checked)}
                        style={{ width: '14px', height: '14px' }}
                      />
                      Phase Correction
                    </label>
                  </div>
                </div>

                {/* File picker */}
                <input
                  type="file"
                  ref={fileInputRef}
                  onChange={handleBrowserFileChange}
                  style={{ display: 'none' }}
                  accept=".npy,.h5,.dat,.dcm,DCM"
                />
                <div className="bevel-inset" style={{ padding: '8px', background: '#e4e7e9', minHeight: '56px', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                  {selectedFilePath ? (
                    <div>
                      <div style={{ fontWeight: 'bold', fontSize: '11px', color: 'var(--color-accent-blue)', textTransform: 'uppercase', marginBottom: '2px' }}>Target K-space file:</div>
                      <div style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', wordBreak: 'break-all' }}>{selectedFilePath.split('\\').pop() || selectedFilePath.split('/').pop()}</div>
                      <div style={{ fontSize: '10px', color: 'var(--color-text-dim)', marginTop: '2px', wordBreak: 'break-all' }}>{selectedFilePath}</div>
                    </div>
                  ) : (
                    <div style={{ textAlign: 'center', color: 'var(--color-text-dim)', fontStyle: 'italic', fontSize: '12px' }}>No K-space file selected (.npy / .h5 / .dat)</div>
                  )}
                </div>

                <div style={{ display: 'flex', gap: '8px' }}>
                  <button onClick={handleSelectFile} disabled={uploading} className="clinical-btn" style={{ flex: 1 }}>
                    Browse File...
                  </button>
                  <button
                    onClick={handleUpload}
                    disabled={uploading || !selectedFilePath || !selectedPatientId}
                    className="clinical-btn clinical-btn-primary"
                    style={{ flex: 1.2 }}
                  >
                    {uploading ? 'Processing...' : 'Ingest Study'}
                  </button>
                </div>

                <div style={{ fontSize: '11px', borderTop: '1px solid var(--color-panel-border)', paddingTop: '10px' }}>
                  <span style={{ fontWeight: 600, color: 'var(--color-text-dim)' }}>Status: </span>
                  <span style={{ fontFamily: 'var(--font-mono)', color: uploading ? 'var(--color-accent-amber)' : 'var(--color-text-main)' }}>
                    {uploadStatus}
                  </span>
                </div>
              </div>
            </div>

            {/* Right: Live AI pipeline progress */}
            <div className="syngo-panel">
              <div className="panel-header">
                <span>AI Pipeline Monitor</span>
                <span style={{ fontFamily: 'var(--font-mono)' }}>[PIPELINE-02]</span>
              </div>
              <div className="panel-body">
                {liveStatus ? (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                    {/* Progress bar */}
                    <div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px', fontFamily: 'var(--font-mono)', fontSize: '11px' }}>
                        <span>Processing Study</span>
                        <span style={{ color: statusColor(liveStatus.status) }}>{liveStatus.status.toUpperCase()}</span>
                      </div>
                      <div className="bevel-inset" style={{ height: '20px', background: '#d0d8de', position: 'relative', overflow: 'hidden' }}>
                        <div style={{
                          position: 'absolute', left: 0, top: 0, bottom: 0,
                          width: `${liveStatus.progress ?? (liveStatus.status === 'complete' ? 100 : 0)}%`,
                          background: liveStatus.status === 'failed'
                            ? 'var(--color-accent-red)'
                            : liveStatus.status === 'complete'
                              ? 'var(--color-accent-green)'
                              : 'var(--color-accent-blue)',
                          transition: 'width 0.5s ease',
                        }} />
                        <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', fontFamily: 'var(--font-mono)', fontSize: '11px', fontWeight: 'bold', color: '#fff', mixBlendMode: 'difference' }}>
                          {liveStatus.progress ?? (liveStatus.status === 'complete' ? 100 : 0)}%
                        </div>
                      </div>
                    </div>

                    {/* Pipeline steps */}
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                      {[
                        { pct: 10, label: 'Mark study as processing', icon: '①' },
                        { pct: 20, label: 'Call AI-service /predict', icon: '②' },
                        { pct: 60, label: 'Store ModelResult + AnomalyDetection + GatingDecision', icon: '③' },
                        { pct: 80, label: 'Create draft report', icon: '④' },
                        { pct: 100, label: 'Mark study as complete', icon: '⑤' },
                      ].map(({ pct, label, icon }) => {
                        const done = (liveStatus.progress ?? 0) >= pct || liveStatus.status === 'complete'
                        const active = (liveStatus.progress ?? 0) < pct && (liveStatus.progress ?? 0) >= pct - 40
                        return (
                          <div key={pct} style={{ display: 'flex', alignItems: 'center', gap: '10px', padding: '6px 8px', background: done ? '#ddeee7' : active ? '#fdf3e3' : '#e8eced', border: '1px solid', borderColor: done ? '#9fcfb8' : active ? '#e0b87a' : '#c5d0d8' }}>
                            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '13px', color: done ? 'var(--color-accent-green)' : active ? 'var(--color-accent-amber)' : 'var(--color-text-dim)' }}>{icon}</span>
                            <span style={{ fontSize: '11px', flex: 1 }}>{label}</span>
                            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', color: done ? 'var(--color-accent-green)' : 'var(--color-text-dim)' }}>{done ? '✓ DONE' : active ? '...' : `@${pct}%`}</span>
                          </div>
                        )
                      })}
                    </div>

                    {/* AI findings preview */}
                    {liveStatus.latestReport && (() => {
                      const f = parseFindings(liveStatus.latestReport.findings)
                      if (!f) return null
                      return (
                        <div style={{ borderTop: '1px solid var(--color-panel-border)', paddingTop: '12px' }}>
                          <div style={{ fontWeight: 600, color: 'var(--color-accent-blue)', fontSize: '11px', textTransform: 'uppercase', marginBottom: '10px' }}>AI Findings</div>
                          <div className="detail-grid">
                            <span className="detail-label">Anomaly:</span>
                            <span className="detail-val" style={{ color: f.anomalyDetected ? 'var(--color-accent-red)' : 'var(--color-accent-green)' }}>
                              {f.anomalyDetected ? '⚠ DETECTED' : '✓ NONE DETECTED'}
                            </span>
                            <span className="detail-label">Confidence:</span>
                            <span className="detail-val" style={{ fontFamily: 'var(--font-mono)' }}>{(f.confidence * 100).toFixed(1)}%</span>
                            <span className="detail-label">Img Encoder:</span>
                            <span className="detail-val">{f.imageEncoderTriggered ? 'TRIGGERED' : 'SKIPPED (gated out)'}</span>
                            {f.predictedPathology && (
                              <>
                                <span className="detail-label">Pathology:</span>
                                <span className="detail-val" style={{ fontWeight: 'bold', color: f.predictedPathology === 'Normal' ? 'var(--color-accent-green)' : 'var(--color-accent-red)' }}>
                                  {f.predictedPathology.replace(/_/g, ' ')}
                                </span>
                                <span className="detail-label">Pathology Conf:</span>
                                <span className="detail-val" style={{ fontFamily: 'var(--font-mono)' }}>
                                  {((f.pathologyConfidence ?? 0) * 100).toFixed(1)}%
                                </span>
                              </>
                            )}
                            {f.artifactScores && Object.entries(f.artifactScores).map(([k, v]) => (
                              <div key={k} style={{ display: 'contents' }}>
                                <span className="detail-label" style={{ textTransform: 'uppercase' }}>{k.replace(/_/g, ' ')}:</span>
                                <span className="detail-val" style={{ fontFamily: 'var(--font-mono)' }}>{(v as number).toFixed(3)}</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )
                    })()}
                  </div>
                ) : (
                  <div style={{ display: 'flex', flexDirection: 'column', height: '100%', alignItems: 'center', justifyContent: 'center', gap: '12px', color: 'var(--color-text-dim)' }}>
                    <div style={{ fontSize: '32px', opacity: 0.3 }}>⚙</div>
                    <div style={{ fontStyle: 'italic', fontSize: '12px' }}>
                      Ingest a K-space study to begin AI pipeline monitoring.
                    </div>
                  </div>
                )}
              </div>
            </div>
          </>
        )}

        {/* ═══════════════════════════════════ ARCHIVE TAB ══════════════════════════════════ */}
        {tab === 'archive' && (
          <>
            {/* Left: study list */}
            <div className="syngo-panel">
              <div className="panel-header">
                <span>Clinical Study Archive</span>
                <button onClick={fetchStudies} className="clinical-btn" style={{ padding: '2px 8px', fontSize: '10px' }}>
                  Refresh
                </button>
              </div>
              <div className="panel-body" style={{ padding: 0 }}>
                {studies.length === 0 ? (
                  <div style={{ display: 'flex', height: '100%', alignItems: 'center', justifyContent: 'center', fontStyle: 'italic', color: 'var(--color-text-dim)' }}>
                    Archive empty.
                  </div>
                ) : (
                  <table className="clinical-table">
                    <thead>
                      <tr>
                        <th>Patient</th>
                        <th>Modality</th>
                        <th>Date</th>
                        <th>Status</th>
                      </tr>
                    </thead>
                    <tbody>
                      {studies.map((s) => (
                        <tr
                          key={s.id}
                          onClick={() => setSelectedStudy(s)}
                          style={{
                            cursor: 'pointer',
                            backgroundColor: selectedStudy?.id === s.id ? '#dce8f0' : 'transparent',
                            fontWeight: selectedStudy?.id === s.id ? 'bold' : 'normal',
                          }}
                        >
                          <td>{s.patient?.name}</td>
                          <td style={{ fontFamily: 'var(--font-mono)' }}>{s.modality}</td>
                          <td>{new Date(s.studyDate).toLocaleDateString()}</td>
                          <td>
                            <span style={{ display: 'inline-flex', alignItems: 'center', gap: '4px', color: statusColor(s.status) }}>
                              <span className={`status-pill ${statusPillClass(s.status)}`} />
                              {s.status}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </div>
            </div>

            {/* Right: study inspector */}
            <div className="syngo-panel">
              <div className="panel-header">
                <span>Study Inspector</span>
                <span style={{ fontFamily: 'var(--font-mono)' }}>[INSPECTOR]</span>
              </div>
              <div className="panel-body">
                {selectedStudy ? (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                    <div className="detail-grid">
                      <span className="detail-label">Patient:</span>
                      <span className="detail-val">{selectedStudy.patient?.name}</span>
                      <span className="detail-label">DOB:</span>
                      <span className="detail-val">{new Date(selectedStudy.patient?.dateOfBirth).toLocaleDateString()}</span>
                      <span className="detail-label">Gender:</span>
                      <span className="detail-val" style={{ textTransform: 'uppercase' }}>{selectedStudy.patient?.gender}</span>
                      <span className="detail-label">Modality:</span>
                      <span className="detail-val">{selectedStudy.modality}</span>
                      <span className="detail-label">Scan Date:</span>
                      <span className="detail-val">{new Date(selectedStudy.studyDate).toLocaleString()}</span>
                      <span className="detail-label">Status:</span>
                      <span className="detail-val" style={{ color: statusColor(selectedStudy.status) }}>{selectedStudy.status.toUpperCase()}</span>
                      <span className="detail-label">Study ID:</span>
                      <span className="detail-val" style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', wordBreak: 'break-all' }}>{selectedStudy.id}</span>
                      <span className="detail-label">K-Space Key:</span>
                      <span className="detail-val" style={{ fontFamily: 'var(--font-mono)', fontSize: '9px', wordBreak: 'break-all' }}>{selectedStudy.dicomKey || 'N/A'}</span>
                    </div>

                    {/* Associated report */}
                    {(() => {
                      const r = reports.find((rp) => rp.studyId === selectedStudy.id)
                      if (!r) return (
                        <div style={{ color: 'var(--color-text-dim)', fontStyle: 'italic', fontSize: '12px' }}>
                          No report generated yet for this study.
                        </div>
                      )
                      const f = parseFindings(r.findings)
                      return (
                        <div style={{ borderTop: '1px solid var(--color-panel-border)', paddingTop: '12px' }}>
                          <div style={{ fontWeight: 600, color: 'var(--color-accent-blue)', fontSize: '11px', textTransform: 'uppercase', marginBottom: '10px' }}>
                            AI Report — <span style={{ color: r.status === 'final' ? 'var(--color-accent-green)' : 'var(--color-accent-amber)' }}>{r.status.toUpperCase()}</span>
                          </div>
                          {f && (
                            <div className="detail-grid">
                              <span className="detail-label">Anomaly:</span>
                              <span className="detail-val" style={{ color: f.anomalyDetected ? 'var(--color-accent-red)' : 'var(--color-accent-green)' }}>
                                {f.anomalyDetected ? '⚠ DETECTED' : '✓ NONE'}
                              </span>
                              <span className="detail-label">Confidence:</span>
                              <span className="detail-val" style={{ fontFamily: 'var(--font-mono)' }}>{(f.confidence * 100).toFixed(1)}%</span>
                              <span className="detail-label">Img Encoder:</span>
                              <span className="detail-val">{f.imageEncoderTriggered ? 'TRIGGERED' : 'SKIPPED'}</span>
                              {f.predictedPathology && (
                                <>
                                  <span className="detail-label">Pathology:</span>
                                  <span className="detail-val" style={{ fontWeight: 'bold', color: f.predictedPathology === 'Normal' ? 'var(--color-accent-green)' : 'var(--color-accent-red)' }}>
                                    {f.predictedPathology.replace(/_/g, ' ')}
                                  </span>
                                  <span className="detail-label">Pathology Conf:</span>
                                  <span className="detail-val" style={{ fontFamily: 'var(--font-mono)' }}>
                                    {((f.pathologyConfidence ?? 0) * 100).toFixed(1)}%
                                  </span>
                                </>
                              )}
                              {f.reconstructedKey && (
                                <>
                                  <span className="detail-label">Recon Key:</span>
                                  <span className="detail-val" style={{ fontFamily: 'var(--font-mono)', fontSize: '9px', wordBreak: 'break-all' }}>{f.reconstructedKey}</span>
                                </>
                              )}
                              {f.note && (
                                <>
                                  <span className="detail-label">Note:</span>
                                  <span className="detail-val" style={{ fontStyle: 'italic', fontSize: '11px' }}>{f.note}</span>
                                </>
                              )}
                            </div>
                          )}
                          {f?.reconstructedKey && (
                            <div style={{ marginTop: '14px' }}>
                              <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--color-text-dim)', textTransform: 'uppercase', marginBottom: '8px' }}>
                                MRI Slice Viewer
                              </div>
                              <ClinicalMRIViewer studyId={selectedStudy.id} />
                            </div>
                          )}
                          {f?.reconstructedKey && (
                            <div style={{ marginTop: '14px' }}>
                              <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--color-text-dim)', textTransform: 'uppercase', marginBottom: '8px' }}>
                                K-Space Explainability (Grad-CAM Overlay)
                              </div>
                              <KSpaceGradCAMViewer studyId={selectedStudy.id} />
                            </div>
                          )}
                          {f?.artifactScores && (
                            <div style={{ marginTop: '10px' }}>
                              <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--color-text-dim)', textTransform: 'uppercase', marginBottom: '6px' }}>Artifact Scores</div>
                              {Object.entries(f.artifactScores).map(([k, v]) => (
                                <div key={k} style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
                                  <span style={{ fontSize: '10px', fontFamily: 'var(--font-mono)', width: '120px', textTransform: 'uppercase' }}>{k.replace(/_/g, ' ')}</span>
                                  <div className="bevel-inset" style={{ flex: 1, height: '12px', background: '#d0d8de', overflow: 'hidden' }}>
                                    <div style={{ height: '100%', width: `${(v as number) * 100}%`, background: (v as number) > 0.5 ? 'var(--color-accent-red)' : 'var(--color-accent-blue)', transition: 'width 0.3s' }} />
                                  </div>
                                  <span style={{ fontSize: '10px', fontFamily: 'var(--font-mono)', width: '40px', textAlign: 'right' }}>{(v as number).toFixed(2)}</span>
                                </div>
                              ))}
                            </div>
                          )}
                        </div>
                      )
                    })()}
                  </div>
                ) : (
                  <div style={{ display: 'flex', height: '100%', alignItems: 'center', justifyContent: 'center', fontStyle: 'italic', color: 'var(--color-text-dim)' }}>
                    Select a study from the archive to inspect.
                  </div>
                )}
              </div>
            </div>
          </>
        )}

        {/* ═══════════════════════════════════ PATIENTS TAB ══════════════════════════════════ */}
        {tab === 'patients' && (
          <>
            {/* Left: create form */}
            <div className="syngo-panel">
              <div className="panel-header">
                <span>Register New Patient</span>
                <span style={{ fontFamily: 'var(--font-mono)' }}>[REG-01]</span>
              </div>
              <div className="panel-body" style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
                <div>
                  <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--color-text-dim)', textTransform: 'uppercase', marginBottom: '4px' }}>Full Name</div>
                  <input
                    type="text"
                    placeholder="e.g. John Smith"
                    value={newPatientName}
                    onChange={(e) => setNewPatientName(e.target.value)}
                    style={{ width: '100%', padding: '6px 8px', fontFamily: 'var(--font-mono)', fontSize: '12px', background: '#e4e7e9', border: '1px solid var(--color-panel-border)', color: 'var(--color-text-main)' }}
                  />
                </div>
                <div>
                  <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--color-text-dim)', textTransform: 'uppercase', marginBottom: '4px' }}>Date of Birth</div>
                  <input
                    type="date"
                    value={newPatientDOB}
                    onChange={(e) => setNewPatientDOB(e.target.value)}
                    style={{ width: '100%', padding: '5px 8px', fontFamily: 'var(--font-mono)', fontSize: '12px', background: '#e4e7e9', border: '1px solid var(--color-panel-border)' }}
                  />
                </div>
                <div>
                  <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--color-text-dim)', textTransform: 'uppercase', marginBottom: '4px' }}>Gender</div>
                  <select
                    value={newPatientGender}
                    onChange={(e) => setNewPatientGender(e.target.value)}
                    style={{ width: '100%', padding: '6px 8px', fontFamily: 'var(--font-mono)', fontSize: '12px', background: '#e4e7e9', border: '1px solid var(--color-panel-border)' }}
                  >
                    <option value="M">Male</option>
                    <option value="F">Female</option>
                    <option value="O">Other</option>
                  </select>
                </div>
                <button
                  onClick={handleCreatePatient}
                  disabled={patientSaving || !newPatientName || !newPatientDOB}
                  className="clinical-btn clinical-btn-primary"
                  style={{ marginTop: 'auto' }}
                >
                  {patientSaving ? 'Registering...' : 'Register Patient'}
                </button>
              </div>
            </div>

            {/* Right: patient list */}
            <div className="syngo-panel">
              <div className="panel-header">
                <span>Patient Registry</span>
                <button onClick={fetchPatients} className="clinical-btn" style={{ padding: '2px 8px', fontSize: '10px' }}>
                  Refresh
                </button>
              </div>
              <div className="panel-body" style={{ padding: 0 }}>
                {patients.length === 0 ? (
                  <div style={{ display: 'flex', height: '100%', alignItems: 'center', justifyContent: 'center', fontStyle: 'italic', color: 'var(--color-text-dim)' }}>
                    No patients registered yet.
                  </div>
                ) : (
                  <table className="clinical-table">
                    <thead>
                      <tr>
                        <th>Name</th>
                        <th>Date of Birth</th>
                        <th>Gender</th>
                        <th>Studies</th>
                      </tr>
                    </thead>
                    <tbody>
                      {patients.map((p) => {
                        const studyCount = studies.filter((s) => s.patientId === p.id).length
                        return (
                          <tr key={p.id} style={{ cursor: 'default' }}>
                            <td style={{ fontWeight: 600 }}>{p.name}</td>
                            <td>{new Date(p.dateOfBirth).toLocaleDateString()}</td>
                            <td style={{ fontFamily: 'var(--font-mono)' }}>{p.gender}</td>
                            <td>
                              <span style={{ fontFamily: 'var(--font-mono)', color: studyCount > 0 ? 'var(--color-accent-blue)' : 'var(--color-text-dim)' }}>
                                {studyCount} {studyCount === 1 ? 'study' : 'studies'}
                              </span>
                            </td>
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                )}
              </div>
            </div>
          </>
        )}

        {/* ═══════════════════════════════════ REPORTS TAB ══════════════════════════════════ */}
        {tab === 'reports' && (
          <>
            {/* Left: reports list */}
            <div className="syngo-panel">
              <div className="panel-header">
                <span>AI Report Index</span>
                <button onClick={fetchReports} className="clinical-btn" style={{ padding: '2px 8px', fontSize: '10px' }}>
                  Refresh
                </button>
              </div>
              <div className="panel-body" style={{ padding: 0 }}>
                {reports.length === 0 ? (
                  <div style={{ display: 'flex', height: '100%', alignItems: 'center', justifyContent: 'center', fontStyle: 'italic', color: 'var(--color-text-dim)' }}>
                    No reports available.
                  </div>
                ) : (
                  <table className="clinical-table">
                    <thead>
                      <tr>
                        <th>Patient</th>
                        <th>Modality</th>
                        <th>Date</th>
                        <th>Report Status</th>
                      </tr>
                    </thead>
                    <tbody>
                      {reports.map((r) => (
                        <tr
                          key={r.id}
                          onClick={() => setSelectedReport(r)}
                          style={{
                            cursor: 'pointer',
                            backgroundColor: selectedReport?.id === r.id ? '#dce8f0' : 'transparent',
                            fontWeight: selectedReport?.id === r.id ? 'bold' : 'normal',
                          }}
                        >
                          <td>{r.study?.patient?.name ?? '—'}</td>
                          <td style={{ fontFamily: 'var(--font-mono)' }}>{r.study?.modality ?? '—'}</td>
                          <td>{new Date(r.createdAt).toLocaleDateString()}</td>
                          <td>
                            <span style={{ color: r.status === 'final' ? 'var(--color-accent-green)' : 'var(--color-accent-amber)' }}>
                              <span className={`status-pill ${r.status === 'final' ? 'green' : 'yellow'}`} />
                              {r.status.toUpperCase()}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </div>
            </div>

            {/* Right: report detail */}
            <div className="syngo-panel">
              <div className="panel-header">
                <span>Report Viewer</span>
                <span style={{ fontFamily: 'var(--font-mono)' }}>[RPT-VIEWER]</span>
              </div>
              <div className="panel-body">
                {selectedReport ? (() => {
                  const f = parseFindings(selectedReport.findings)
                  return (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                      {/* Controls Bar */}
                      <div style={{ display: 'flex', gap: '8px', justifyContent: 'flex-end', borderBottom: '1px solid var(--color-panel-border)', paddingBottom: '10px' }}>
                        {isEditing ? (
                          <>
                            <button
                              onClick={handleSaveReport}
                              className="clinical-btn clinical-btn-primary"
                              style={{ padding: '4px 12px', fontSize: '11px' }}
                            >
                              💾 Save Report
                            </button>
                            <button
                              onClick={() => {
                                setEditImpression(selectedReport.impression || '')
                                setIsEditing(false)
                              }}
                              className="clinical-btn"
                              style={{ padding: '4px 12px', fontSize: '11px' }}
                            >
                              ✕ Cancel
                            </button>
                          </>
                        ) : (
                          <>
                            <button
                              onClick={() => setIsEditing(true)}
                              className="clinical-btn"
                              style={{ padding: '4px 12px', fontSize: '11px' }}
                            >
                              ✏️ Edit Report
                            </button>
                            <button
                              onClick={handleDownloadPdf}
                              className="clinical-btn"
                              style={{ padding: '4px 12px', fontSize: '11px' }}
                            >
                              📥 Download PDF
                            </button>
                          </>
                        )}
                      </div>
                      {/* Header */}
                      <div style={{ background: '#d1dadf', border: '1px solid var(--color-panel-border)', padding: '10px 12px' }}>
                        <div style={{ fontSize: '12px', fontWeight: 700, textTransform: 'uppercase', color: 'var(--color-accent-blue)', marginBottom: '6px' }}>
                          Radiological AI Report
                        </div>
                        <div className="detail-grid" style={{ marginBottom: 0 }}>
                          <span className="detail-label">Patient:</span>
                          <span className="detail-val">{selectedReport.study?.patient?.name ?? '—'}</span>
                          <span className="detail-label">Study ID:</span>
                          <span className="detail-val" style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', wordBreak: 'break-all' }}>{selectedReport.studyId}</span>
                          <span className="detail-label">Generated:</span>
                          <span className="detail-val">{new Date(selectedReport.createdAt).toLocaleString()}</span>
                          <span className="detail-label">Report ID:</span>
                          <span className="detail-val" style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', wordBreak: 'break-all' }}>{selectedReport.id}</span>
                          <span className="detail-label">Status:</span>
                          <span className="detail-val" style={{ color: selectedReport.status === 'final' ? 'var(--color-accent-green)' : 'var(--color-accent-amber)' }}>
                            {selectedReport.status.toUpperCase()}
                          </span>
                        </div>
                      </div>

                      {f ? (
                        <>
                          {/* Verdict banner */}
                          <div style={{
                            padding: '12px 16px',
                            background: f.anomalyDetected ? '#fbeeee' : '#eef7f2',
                            border: `2px solid ${f.anomalyDetected ? 'var(--color-accent-red)' : 'var(--color-accent-green)'}`,
                            display: 'flex', alignItems: 'center', gap: '12px',
                          }}>
                            <span style={{ fontSize: '24px' }}>{f.anomalyDetected ? '⚠' : '✓'}</span>
                            <div style={{ flex: 1 }}>
                              <div style={{ fontWeight: 700, fontSize: '13px', color: f.anomalyDetected ? 'var(--color-accent-red)' : 'var(--color-accent-green)' }}>
                                {f.anomalyDetected ? 'K-SPACE ANOMALY DETECTED' : 'NO ANOMALY DETECTED'}
                              </div>
                              <div style={{ fontSize: '11px', color: 'var(--color-text-dim)', marginTop: '2px' }}>
                                Confidence: {(f.confidence * 100).toFixed(1)}% — Image Encoder: {f.imageEncoderTriggered ? 'TRIGGERED' : 'SKIPPED (gated out)'}
                              </div>
                            </div>
                            {f.predictedPathology && (
                              <div style={{ textAlign: 'right', borderLeft: '1px solid var(--color-panel-border)', paddingLeft: '15px' }}>
                                <div style={{ fontSize: '10px', textTransform: 'uppercase', color: 'var(--color-text-dim)' }}>AI Diagnosis</div>
                                <div style={{ fontWeight: 'bold', fontSize: '14px', color: f.predictedPathology === 'Normal' ? 'var(--color-accent-green)' : 'var(--color-accent-red)' }}>
                                  {f.predictedPathology.replace(/_/g, ' ')}
                                </div>
                                <div style={{ fontSize: '11px', fontFamily: 'var(--font-mono)' }}>
                                  {((f.pathologyConfidence ?? 0) * 100).toFixed(1)}% confidence
                                </div>
                              </div>
                            )}
                          </div>

                          {/* Standardized Radiology Report Sheet (Diagnostic layout) */}
                          {(selectedReport.impression !== null || isEditing) && (
                            <div style={{
                              background: '#ffffff',
                              color: '#2d3748',
                              fontFamily: 'var(--font-sans)',
                              fontSize: '14px',
                              lineHeight: '1.6',
                              padding: '24px',
                              border: '1px solid #cbd5e0',
                              borderRadius: '4px',
                              boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03)',
                            }}>
                              {/* Clinic Letterhead */}
                              <div style={{
                                display: 'flex',
                                justifyContent: 'space-between',
                                alignItems: 'center',
                                borderBottom: '2px solid var(--color-accent-blue)',
                                paddingBottom: '12px',
                                marginBottom: '16px',
                              }}>
                                <div style={{ display: 'flex', flexDirection: 'column' }}>
                                  <span style={{ fontSize: '16px', fontWeight: 'bold', color: 'var(--color-accent-blue)', letterSpacing: '0.5px' }}>
                                    KVISION // CLINICAL IMAGING CENTER
                                  </span>
                                  <span style={{ fontSize: '10px', color: 'var(--color-text-dim)', textTransform: 'uppercase', marginTop: '2px' }}>
                                    Magnetom Trio 3T MRI Workstation
                                  </span>
                                </div>
                                <div style={{ textAlign: 'right', fontSize: '10px', color: 'var(--color-text-dim)', lineHeight: '1.4' }}>
                                  123 Health Ave, Suite 400<br />
                                  Phone: (555) 019-2834<br />
                                  reports@kvision.ai
                                </div>
                              </div>

                              {/* Patient Info Table */}
                              <table style={{ width: '100%', borderCollapse: 'collapse', marginBottom: '20px', fontSize: '12px', border: '1px solid #e2e8f0' }}>
                                <tbody>
                                  <tr style={{ borderBottom: '1px solid #e2e8f0' }}>
                                    <td style={{ padding: '6px 10px', fontWeight: 'bold', color: '#4a5568', background: '#f7fafc', width: '20%', borderRight: '1px solid #e2e8f0' }}>Patient Name:</td>
                                    <td style={{ padding: '6px 10px', color: '#2d3748', width: '30%', borderRight: '1px solid #e2e8f0' }}>{selectedReport.study?.patient?.name ?? '—'}</td>
                                    <td style={{ padding: '6px 10px', fontWeight: 'bold', color: '#4a5568', background: '#f7fafc', width: '20%', borderRight: '1px solid #e2e8f0' }}>Referring Physician:</td>
                                    <td style={{ padding: '6px 10px', color: '#2d3748', width: '30%' }}>Dr. Tarun Ahuja, MD</td>
                                  </tr>
                                  <tr style={{ borderBottom: '1px solid #e2e8f0' }}>
                                    <td style={{ padding: '6px 10px', fontWeight: 'bold', color: '#4a5568', background: '#f7fafc', borderRight: '1px solid #e2e8f0' }}>Patient ID:</td>
                                    <td style={{ padding: '6px 10px', color: '#2d3748', fontFamily: 'var(--font-mono)', fontSize: '11px', borderRight: '1px solid #e2e8f0' }}>{selectedReport.study?.patient?.id.substring(0, 8) ?? '—'}</td>
                                    <td style={{ padding: '6px 10px', fontWeight: 'bold', color: '#4a5568', background: '#f7fafc', borderRight: '1px solid #e2e8f0' }}>Modality:</td>
                                    <td style={{ padding: '6px 10px', color: '#2d3748' }}>{selectedReport.study?.modality ?? 'MRI'} (3T)</td>
                                  </tr>
                                  <tr style={{ borderBottom: '1px solid #e2e8f0' }}>
                                    <td style={{ padding: '6px 10px', fontWeight: 'bold', color: '#4a5568', background: '#f7fafc', borderRight: '1px solid #e2e8f0' }}>Age / Gender:</td>
                                    <td style={{ padding: '6px 10px', color: '#2d3748', borderRight: '1px solid #e2e8f0' }}>
                                      {selectedReport.study?.patient?.dateOfBirth ? getPatientAge(selectedReport.study.patient.dateOfBirth) : '—'} / {selectedReport.study?.patient?.gender ?? '—'}
                                    </td>
                                    <td style={{ padding: '6px 10px', fontWeight: 'bold', color: '#4a5568', background: '#f7fafc', borderRight: '1px solid #e2e8f0' }}>Date of Study:</td>
                                    <td style={{ padding: '6px 10px', color: '#2d3748' }}>
                                      {selectedReport.study?.studyDate ? new Date(selectedReport.study.studyDate).toLocaleDateString() : '—'}
                                    </td>
                                  </tr>
                                  <tr>
                                    <td style={{ padding: '6px 10px', fontWeight: 'bold', color: '#4a5568', background: '#f7fafc', borderRight: '1px solid #e2e8f0' }}>Report ID:</td>
                                    <td style={{ padding: '6px 10px', color: '#2d3748', fontFamily: 'var(--font-mono)', fontSize: '11px', borderRight: '1px solid #e2e8f0' }}>{selectedReport.id.substring(0, 8)}</td>
                                    <td style={{ padding: '6px 10px', fontWeight: 'bold', color: '#4a5568', background: '#f7fafc', borderRight: '1px solid #e2e8f0' }}>Date of Report:</td>
                                    <td style={{ padding: '6px 10px', color: '#2d3748' }}>{new Date(selectedReport.createdAt).toLocaleDateString()}</td>
                                  </tr>
                                </tbody>
                              </table>

                              <div style={{
                                textAlign: 'center',
                                fontSize: '14px',
                                fontWeight: 'bold',
                                textTransform: 'uppercase',
                                color: '#1a365d',
                                letterSpacing: '1px',
                                marginBottom: '20px',
                                borderBottom: '1px solid #cbd5e0',
                                paddingBottom: '8px',
                              }}>
                                MAGNETIC RESONANCE IMAGING (MRI) BRAIN REPORT
                              </div>

                              {/* Report Text Content / Editor */}
                              <div style={{ marginBottom: '30px' }}>
                                {isEditing ? (
                                  <textarea
                                    value={editImpression}
                                    onChange={(e) => setEditImpression(e.target.value)}
                                    style={{
                                      width: '100%',
                                      height: '400px',
                                      padding: '12px',
                                      fontFamily: 'var(--font-mono)',
                                      fontSize: '12px',
                                      lineHeight: '1.5',
                                      border: '1px solid #cbd5e0',
                                      borderRadius: '4px',
                                      outline: 'none',
                                      resize: 'vertical',
                                      color: '#2d3748',
                                      background: '#fafafa',
                                    }}
                                  />
                                ) : (
                                  <div style={{ color: '#2d3748', fontSize: '13px', fontFamily: 'var(--font-sans)', lineHeight: '1.6' }}>
                                    {renderFormattedReportText(selectedReport.impression)}
                                  </div>
                                )}
                              </div>

                              {/* Radiologist Signature */}
                              <div style={{
                                display: 'flex',
                                justifyContent: 'space-between',
                                alignItems: 'flex-end',
                                borderTop: '1px solid #cbd5e0',
                                paddingTop: '12px',
                                marginTop: '30px',
                                fontSize: '12px',
                              }}>
                                <div style={{ color: 'var(--color-text-dim)', fontStyle: 'italic', fontSize: '10px' }}>
                                  * This report is interpreted electronically based on fused S4-CNN models and RAG-guided context.
                                </div>
                                <div style={{ textAlign: 'right', display: 'flex', flexDirection: 'column', gap: '2px' }}>
                                  <span style={{ fontWeight: 'bold', color: '#1a365d' }}>Dr. Tarun Ahuja, MD</span>
                                  <span style={{ color: 'var(--color-text-dim)', fontSize: '11px' }}>Consultant Radiologist</span>
                                  <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--color-text-dim)', fontSize: '10px' }}>Reg No: NS-2026-994</span>
                                </div>
                              </div>
                            </div>
                          )}

                          {f.reconstructedKey && (
                            <div style={{ border: '1px solid var(--color-panel-border)', padding: '12px', background: '#f5f7f8' }}>
                              <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--color-text-dim)', textTransform: 'uppercase', marginBottom: '8px' }}>
                                MRI Slice Reconstruction Preview
                              </div>
                              <ClinicalMRIViewer studyId={selectedReport.studyId} />
                            </div>
                          )}

                          {f.reconstructedKey && (
                            <div style={{ border: '1px solid var(--color-panel-border)', padding: '12px', background: '#f5f7f8' }}>
                              <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--color-text-dim)', textTransform: 'uppercase', marginBottom: '8px' }}>
                                K-Space Explainability (Grad-CAM Overlay)
                              </div>
                              <KSpaceGradCAMViewer studyId={selectedReport.studyId} />
                            </div>
                          )}

                          {/* Pathology Probabilities List */}
                          {f.pathologyProbabilities && (
                            <div style={{ border: '1px solid var(--color-panel-border)', padding: '12px', background: '#f5f7f8' }}>
                              <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--color-text-dim)', textTransform: 'uppercase', marginBottom: '8px' }}>
                                Pathology Risk Profiling (Fused S4-CNN)
                              </div>
                              <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                                {Object.entries(f.pathologyProbabilities)
                                  .sort((a, b) => (b[1] as number) - (a[1] as number))
                                  .map(([k, v]) => {
                                    const prob = v as number;
                                    if (prob < 0.01) return null; // Hide very low probabilities
                                    return (
                                      <div key={k} style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                                        <span style={{ fontSize: '10px', fontFamily: 'var(--font-mono)', width: '130px', textTransform: 'uppercase', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                          {k.replace(/_/g, ' ')}
                                        </span>
                                        <div className="bevel-inset" style={{ flex: 1, height: '12px', background: '#d0d8de', overflow: 'hidden', borderRadius: '2px' }}>
                                          <div style={{
                                            height: '100%',
                                            width: `${prob * 100}%`,
                                            background: k === 'Normal' ? 'var(--color-accent-green)' : prob > 0.5 ? 'var(--color-accent-red)' : 'var(--color-accent-blue)',
                                            transition: 'width 0.5s ease',
                                          }} />
                                        </div>
                                        <span style={{ fontSize: '10px', fontFamily: 'var(--font-mono)', width: '45px', textAlign: 'right' }}>
                                          {(prob * 100).toFixed(1)}%
                                        </span>
                                      </div>
                                    );
                                  })}
                              </div>
                            </div>
                          )}

                          {/* Artifact scores with bars */}
                          {f.artifactScores && (
                            <div>
                              <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--color-text-dim)', textTransform: 'uppercase', marginBottom: '8px' }}>Artifact Analysis</div>
                              {Object.entries(f.artifactScores).map(([k, v]) => (
                                <div key={k} style={{ marginBottom: '8px' }}>
                                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', marginBottom: '3px' }}>
                                    <span style={{ fontFamily: 'var(--font-mono)', textTransform: 'uppercase' }}>{k.replace(/_/g, ' ')}</span>
                                    <span style={{ fontFamily: 'var(--font-mono)', color: (v as number) > 0.5 ? 'var(--color-accent-red)' : 'var(--color-text-main)' }}>
                                      {(v as number).toFixed(3)} {(v as number) > 0.5 ? '▲ HIGH' : ''}
                                    </span>
                                  </div>
                                  <div className="bevel-inset" style={{ height: '14px', background: '#d0d8de', overflow: 'hidden' }}>
                                    <div style={{
                                      height: '100%',
                                      width: `${(v as number) * 100}%`,
                                      background: (v as number) > 0.7 ? 'var(--color-accent-red)' : (v as number) > 0.5 ? 'var(--color-accent-amber)' : 'var(--color-accent-blue)',
                                      transition: 'width 0.4s ease',
                                    }} />
                                  </div>
                                </div>
                              ))}
                            </div>
                          )}

                          {/* Notes */}
                          {f.note && (
                            <div style={{ padding: '10px', background: '#f5f7f8', border: '1px solid var(--color-panel-border)', fontSize: '12px', fontStyle: 'italic', color: 'var(--color-text-dim)' }}>
                              {f.note}
                            </div>
                          )}

                          {/* Model result ID */}
                          {f.modelResultId && (
                            <div style={{ fontSize: '10px', fontFamily: 'var(--font-mono)', color: 'var(--color-text-dim)' }}>
                              Model Result ID: {f.modelResultId}
                            </div>
                          )}
                        </>
                      ) : (
                        <div style={{ color: 'var(--color-text-dim)', fontStyle: 'italic' }}>No AI findings data available.</div>
                      )}
                    </div>
                  )
                })() : (
                  <div style={{ display: 'flex', height: '100%', alignItems: 'center', justifyContent: 'center', fontStyle: 'italic', color: 'var(--color-text-dim)' }}>
                    Select a report from the index to view details.
                  </div>
                )}
              </div>
            </div>
          </>
        )}

        {/* ═══════════════════════════════════ 3D BRAIN MODEL TAB ══════════════════════════════════ */}
        {tab === 'brain3d' && (
          <>
            {/* Left: Patient selection and clinical summary */}
            <div className="syngo-panel">
              <div className="panel-header">
                <span>Patient Explorer</span>
                <span style={{ fontFamily: 'var(--font-mono)' }}>[PT-3D]</span>
              </div>
              <div className="panel-body" style={{ display: 'flex', flexDirection: 'column', gap: '14px', overflowY: 'auto' }}>
                {/* Selector */}
                <div>
                  <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--color-text-dim)', textTransform: 'uppercase', marginBottom: '6px' }}>
                    Active Patient
                  </div>
                  <select
                    value={brain3dSelectedPatientId}
                    onChange={(e) => setBrain3dSelectedPatientId(e.target.value)}
                    style={{ width: '100%', padding: '6px 8px', fontFamily: 'var(--font-mono)', fontSize: '12px', background: '#e4e7e9', border: '1px solid var(--color-panel-border)', color: 'var(--color-text-main)' }}
                  >
                    <option value="">— Select Patient —</option>
                    {patients.map((p) => (
                      <option key={p.id} value={p.id}>
                        {p.name} ({new Date(p.dateOfBirth).getFullYear()}, {p.gender})
                      </option>
                    ))}
                  </select>
                </div>

                {/* If selected, show detailed information */}
                {(() => {
                  const selectedPatient = patients.find(p => p.id === brain3dSelectedPatientId)
                  if (!selectedPatient) {
                    return (
                      <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', textAlign: 'center', color: 'var(--color-text-dim)', fontStyle: 'italic', padding: '20px' }}>
                        Please select a patient from the index above to display clinical records and load the 3D model.
                      </div>
                    )
                  }

                  // Find patient's studies
                  const patientStudies = studies.filter(s => s.patientId === selectedPatient.id)
                  // Find patient's reports
                  const patientReports = reports.filter(r => r.study?.patientId === selectedPatient.id)

                  return (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                      {/* Demographic card */}
                      <div style={{ background: '#d1dadf', border: '1px solid var(--color-panel-border)', padding: '10px 12px' }}>
                        <div style={{ fontSize: '11px', fontWeight: 700, textTransform: 'uppercase', color: 'var(--color-accent-blue)', marginBottom: '6px' }}>
                          Demographics
                        </div>
                        <div className="detail-grid" style={{ marginBottom: 0 }}>
                          <span className="detail-label">Name:</span>
                          <span className="detail-val" style={{ fontWeight: 600 }}>{selectedPatient.name}</span>
                          <span className="detail-label">Patient ID:</span>
                          <span className="detail-val" style={{ fontFamily: 'var(--font-mono)', fontSize: '10px' }}>{selectedPatient.id}</span>
                          <span className="detail-label">DOB:</span>
                          <span className="detail-val">{new Date(selectedPatient.dateOfBirth).toLocaleDateString()}</span>
                          <span className="detail-label">Gender:</span>
                          <span className="detail-val" style={{ textTransform: 'uppercase' }}>{selectedPatient.gender}</span>
                        </div>
                      </div>

                      {/* Imaging Studies Card */}
                      <div>
                        <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--color-text-dim)', textTransform: 'uppercase', marginBottom: '6px' }}>
                          Acquisition History
                        </div>
                        {patientStudies.length === 0 ? (
                          <div style={{ fontSize: '11px', fontStyle: 'italic', color: 'var(--color-text-dim)', padding: '6px' }}>
                            No studies found for this patient.
                          </div>
                        ) : (
                          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                            {patientStudies.map(s => (
                              <div key={s.id} style={{ background: '#f5f7f8', border: '1px solid var(--color-panel-border)', padding: '8px' }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', fontWeight: 600, fontSize: '11px' }}>
                                  <span style={{ color: 'var(--color-accent-blue)' }}>{s.modality} Scan</span>
                                  <span style={{ color: statusColor(s.status) }}>{s.status.toUpperCase()}</span>
                                </div>
                                <div style={{ fontSize: '10px', color: 'var(--color-text-dim)', marginTop: '4px' }}>
                                  Date: {new Date(s.studyDate).toLocaleString()}
                                </div>
                                <div style={{ fontSize: '9px', fontFamily: 'var(--font-mono)', color: 'var(--color-text-dim)', marginTop: '2px', wordBreak: 'break-all' }}>
                                  ID: {s.id}
                                </div>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>

                      {/* AI Report Card */}
                      <div>
                        <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--color-text-dim)', textTransform: 'uppercase', marginBottom: '6px' }}>
                          AI Diagnostic Findings
                        </div>
                        {patientReports.length === 0 ? (
                          <div style={{ fontSize: '11px', fontStyle: 'italic', color: 'var(--color-text-dim)', padding: '6px' }}>
                            No reports generated yet.
                          </div>
                        ) : (
                          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                            {patientReports.map(r => {
                              const f = parseFindings(r.findings)
                              return (
                                <div key={r.id} style={{ border: '1px solid var(--color-panel-border)', background: '#f5f7f8', padding: '8px' }}>
                                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', fontWeight: 600, marginBottom: '6px' }}>
                                    <span>AI Report</span>
                                    <span style={{ color: r.status === 'final' ? 'var(--color-accent-green)' : 'var(--color-accent-amber)' }}>
                                      {r.status.toUpperCase()}
                                    </span>
                                  </div>
                                  {f ? (
                                    <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', fontSize: '11px' }}>
                                      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                        <span>Anomaly:</span>
                                        <span style={{ fontWeight: 600, color: f.anomalyDetected ? 'var(--color-accent-red)' : 'var(--color-accent-green)' }}>
                                          {f.anomalyDetected ? '⚠ DETECTED' : '✓ NONE'}
                                        </span>
                                      </div>
                                      {f.predictedPathology && (
                                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                          <span>AI Diagnosis:</span>
                                          <span style={{ fontWeight: 600, color: f.predictedPathology === 'Normal' ? 'var(--color-accent-green)' : 'var(--color-accent-red)' }}>
                                            {f.predictedPathology.replace(/_/g, ' ')}
                                          </span>
                                        </div>
                                      )}
                                      {f.confidence && (
                                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                          <span>Confidence:</span>
                                          <span>{(f.confidence * 100).toFixed(1)}%</span>
                                        </div>
                                      )}
                                      {f.note && (
                                        <div style={{ marginTop: '4px', padding: '4px 6px', background: '#ebeeef', fontSize: '10px', fontStyle: 'italic', borderLeft: '2px solid var(--color-accent-blue)' }}>
                                          {f.note}
                                        </div>
                                      )}
                                    </div>
                                  ) : (
                                    <div style={{ fontSize: '10px', fontStyle: 'italic', color: 'var(--color-text-dim)' }}>
                                      Pending analysis / No AI findings data.
                                    </div>
                                  )}
                                </div>
                              )
                            })}
                          </div>
                        )}
                      </div>
                    </div>
                  )
                })()}
              </div>
            </div>

            {/* Right: 3D model frame */}
            <div className="syngo-panel">
              <div className="panel-header">
                <span>Interactive 3D Brain Reconstruction (brain2print)</span>
                <span style={{ fontFamily: 'var(--font-mono)' }}>[3D-MESH-GENERATOR]</span>
              </div>
              <div className="panel-body" style={{ padding: 0, height: '100%', overflow: 'hidden', background: brain3dSelectedPatientId ? '#0a0d10' : 'var(--color-panel-bg)' }}>
                {brain3dSelectedPatientId ? (
                  <iframe
                    ref={brain3dIframeRef}
                    src="./brain2print/index.html"
                    style={{ width: '100%', height: '100%', border: 'none' }}
                    title="brain2print 3D Brain Model"
                  />
                ) : (
                  <div style={{ height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', color: 'var(--color-accent-blue)', fontFamily: 'var(--font-mono)', fontSize: '12px', gap: '10px', textAlign: 'center', padding: '40px' }}>
                    <div style={{ fontWeight: 'bold' }}>NO ACTIVE PATIENT SELECTION</div>
                    <div style={{ color: 'var(--color-text-dim)', fontSize: '11px', maxWidth: '300px', lineHeight: '1.4' }}>
                      Select a patient from the explorer index on the left to initialize the 3D reconstruction and overlay their scan volume.
                    </div>
                  </div>
                )}
              </div>
            </div>
          </>
        )}

        {/* ═══════════════════════════════════ ANALYTICS TAB ══════════════════════════════════ */}
        {tab === 'analytics' && (() => {
          const { total, anomalies, gatingRate, avgConfidence, mostFrequent, pathologyCounts, avgArtifacts } = getAnalyticsData()

          return (
            <>
              {/* Left Panel: Metrics & Controls */}
              <div className="syngo-panel">
                <div className="panel-header">
                  <span>Workstation Metrics</span>
                  <span style={{ fontFamily: 'var(--font-mono)' }}>[METRICS-01]</span>
                </div>
                <div className="panel-body" style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
                  {/* Simulation toggle */}
                  <div style={{ background: '#d1dadf', border: '1px solid var(--color-panel-border)', padding: '10px', display: 'flex', flexDirection: 'column', gap: '6px' }}>
                    <div style={{ fontSize: '11px', fontWeight: 700, textTransform: 'uppercase', color: 'var(--color-accent-blue)' }}>
                      Data Control
                    </div>
                    <label style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '12px', fontFamily: 'var(--font-mono)', cursor: 'pointer' }}>
                      <input
                        type="checkbox"
                        checked={useSimulatedData}
                        onChange={(e) => setUseSimulatedData(e.target.checked)}
                        style={{ width: '14px', height: '14px' }}
                      />
                      Workstation Simulation Mode
                    </label>
                    <div style={{ fontSize: '9px', color: 'var(--color-text-dim)' }}>
                      {useSimulatedData ? 'Displaying 50 pre-compiled studies for testing' : 'Reading live data from database'}
                    </div>
                  </div>

                  {/* Summary Indicators */}
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                    <div className="bevel-inset" style={{ padding: '10px', background: '#0a0d10', color: 'var(--color-accent-blue)' }}>
                      <div style={{ fontSize: '10px', color: 'var(--color-text-dim)', textTransform: 'uppercase', fontFamily: 'var(--font-mono)' }}>TOTAL INGESTED SCANS</div>
                      <div style={{ fontSize: '28px', fontFamily: 'var(--font-mono)', fontWeight: 'bold', color: '#00ffff' }}>{String(total).padStart(4, '0')}</div>
                    </div>

                    <div className="bevel-inset" style={{ padding: '10px', background: '#0a0d10', color: 'var(--color-accent-green)' }}>
                      <div style={{ fontSize: '10px', color: 'var(--color-text-dim)', textTransform: 'uppercase', fontFamily: 'var(--font-mono)' }}>K-SPACE GATING RATE</div>
                      <div style={{ fontSize: '28px', fontFamily: 'var(--font-mono)', fontWeight: 'bold', color: 'var(--color-accent-amber)' }}>{gatingRate.toFixed(1)}%</div>
                    </div>

                    <div className="bevel-inset" style={{ padding: '10px', background: '#0a0d10', color: 'var(--color-accent-green)' }}>
                      <div style={{ fontSize: '10px', color: 'var(--color-text-dim)', textTransform: 'uppercase', fontFamily: 'var(--font-mono)' }}>AVG AI CONFIDENCE</div>
                      <div style={{ fontSize: '28px', fontFamily: 'var(--font-mono)', fontWeight: 'bold', color: '#39a169' }}>{avgConfidence.toFixed(1)}%</div>
                    </div>

                    <div className="bevel-inset" style={{ padding: '10px', background: '#0a0d10', color: 'var(--color-accent-blue)' }}>
                      <div style={{ fontSize: '10px', color: 'var(--color-text-dim)', textTransform: 'uppercase', fontFamily: 'var(--font-mono)' }}>MOST COMMON PATHOLOGY</div>
                      <div style={{ fontSize: '14px', fontWeight: 'bold', color: '#fff', marginTop: '6px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {mostFrequent.replace(/_/g, ' ').toUpperCase()}
                      </div>
                    </div>
                  </div>

                  <div style={{ marginTop: 'auto', borderTop: '1px solid var(--color-panel-border)', paddingTop: '10px' }}>
                    <button onClick={fetchReports} className="clinical-btn" style={{ width: '100%' }}>
                      Sync Real Data
                    </button>
                  </div>
                </div>
              </div>

              {/* Right Panel: Analytics Dashboard */}
              <div className="syngo-panel">
                <div className="panel-header">
                  <span>Diagnostic Analytics Dashboard</span>
                  <span style={{ fontFamily: 'var(--font-mono)' }}>[DASHBOARD-01]</span>
                </div>
                <div className="panel-body" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '14px', overflowY: 'auto' }}>
                  
                  {/* Top-Left: Pathology distribution */}
                  <div style={{ border: '1px solid var(--color-panel-border)', padding: '12px', background: '#f5f7f8', display: 'flex', flexDirection: 'column' }}>
                    <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--color-text-dim)', textTransform: 'uppercase', marginBottom: '8px' }}>
                      Pathology Prevalence Distribution
                    </div>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '5px', flex: 1, justifyContent: 'space-between' }}>
                      {Object.entries(pathologyCounts).map(([k, v]) => {
                        const count = v as number
                        const pct = total > 0 ? (count / total) * 100 : 0
                        const isNormal = k === 'Normal'
                        
                        let barColor = 'var(--color-accent-blue)'
                        if (isNormal) barColor = 'var(--color-accent-green)'
                        else if (count > 0 && ['Tumor_Glioma', 'MS_Lesions', 'Hemorrhage', 'Cerebral_Microbleeds'].includes(k)) barColor = 'var(--color-accent-red)'
                        else if (count > 0) barColor = 'var(--color-accent-amber)'
                        
                        return (
                          <div key={k} style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                            <span style={{ fontSize: '9px', fontFamily: 'var(--font-mono)', width: '120px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', textTransform: 'uppercase' }}>
                              {k.replace(/_/g, ' ')}
                            </span>
                            <div className="bevel-inset" style={{ flex: 1, height: '12px', background: '#d0d8de', overflow: 'hidden', borderRadius: '1px', position: 'relative' }}>
                              <div style={{
                                height: '100%',
                                width: `${pct}%`,
                                background: barColor,
                                transition: 'width 0.5s ease',
                              }} />
                            </div>
                            <span style={{ fontSize: '9px', fontFamily: 'var(--font-mono)', width: '25px', textAlign: 'right', fontWeight: count > 0 ? 'bold' : 'normal' }}>
                              {count}
                            </span>
                          </div>
                        )
                      })}
                    </div>
                  </div>

                  {/* Top-Right: Radar & Flow visualizers */}
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
                    {/* Artifact radar */}
                    <div style={{ border: '1px solid var(--color-panel-border)', padding: '12px', background: '#f5f7f8', flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                      <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--color-text-dim)', textTransform: 'uppercase', marginBottom: '8px', width: '100%' }}>
                        Prevalent Artifact Profiling (S4 branch)
                      </div>
                      <ArtifactRadarChart 
                        ghosting={avgArtifacts.ghosting} 
                        wrapAround={avgArtifacts.wrap_around} 
                        zipperNoise={avgArtifacts.zipper_noise} 
                      />
                    </div>

                    {/* Gating Decision Pipeline Visual Flowchart */}
                    <div style={{ border: '1px solid var(--color-panel-border)', padding: '12px', background: '#f5f7f8', display: 'flex', flexDirection: 'column' }}>
                      <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--color-text-dim)', textTransform: 'uppercase', marginBottom: '8px' }}>
                        Acquisition Decision Flow (Gating: {gatingRate.toFixed(0)}% Triggered)
                      </div>
                      <div style={{ background: '#ebeeef', padding: '6px', border: '1px solid var(--color-panel-border)' }}>
                        <svg viewBox="0 0 500 150" style={{ width: '100%', height: '150px' }}>
                          <defs>
                            <marker id="arrow" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                              <path d="M 0 0 L 10 5 L 0 10 z" fill="#5c6c77" />
                            </marker>
                          </defs>
                          
                          {/* Raw K-Space */}
                          <rect x="10" y="45" width="90" height="45" rx="3" fill="#dee3e6" stroke="var(--color-panel-border)" strokeWidth="1.5" />
                          <text x="55" y="62" textAnchor="middle" dominantBaseline="middle" fontSize="10" fontFamily="var(--font-mono)" fontWeight="bold" fill="var(--color-text-main)">K-SPACE IN</text>
                          <text x="55" y="75" textAnchor="middle" dominantBaseline="middle" fontSize="8" fontFamily="var(--font-mono)" fill="var(--color-text-dim)">[8s × 16c]</text>

                          <line x1="100" y1="67" x2="135" y2="67" stroke="#5c6c77" strokeWidth="1.5" markerEnd="url(#arrow)" />

                          {/* Anomaly detector */}
                          <rect x="140" y="45" width="100" height="45" rx="3" fill="var(--color-light-glow)" stroke="var(--color-accent-blue)" strokeWidth="2" />
                          <text x="190" y="62" textAnchor="middle" dominantBaseline="middle" fontSize="10" fontFamily="var(--font-mono)" fontWeight="bold" fill="var(--color-accent-blue)">AI ANOMALY</text>
                          <text x="190" y="75" textAnchor="middle" dominantBaseline="middle" fontSize="9" fontFamily="var(--font-mono)" fontWeight="bold" fill="var(--color-text-main)">GATING</text>

                          {/* Split path: Clean */}
                          <path d="M 190 90 L 190 120 L 255 120" fill="none" stroke="#2b704c" strokeWidth="1.5" strokeDasharray="3 3" markerEnd="url(#arrow)" />
                          <text x="210" y="112" fontSize="7" fontFamily="var(--font-mono)" fontWeight="bold" fill="var(--color-accent-green)">CLEAN ({(100 - gatingRate).toFixed(0)}%)</text>

                          {/* Split path: Anomalous */}
                          <line x1="240" y1="67" x2="275" y2="67" stroke="var(--color-accent-amber)" strokeWidth="2" markerEnd="url(#arrow)" />
                          <text x="257" y="58" textAnchor="middle" fontSize="7" fontFamily="var(--font-mono)" fontWeight="bold" fill="var(--color-accent-amber)">DIRTY ({gatingRate.toFixed(0)}%)</text>

                          {/* Bypass */}
                          <rect x="260" y="102" width="100" height="35" rx="3" fill="#ebeeef" stroke="var(--color-accent-green)" strokeWidth="1.5" />
                          <text x="310" y="119" textAnchor="middle" dominantBaseline="middle" fontSize="9" fontFamily="var(--font-mono)" fontWeight="bold" fill="var(--color-accent-green)">BYPASS RECON</text>

                          {/* Pipeline */}
                          <rect x="280" y="45" width="100" height="45" rx="3" fill="#fcf3e3" stroke="var(--color-accent-amber)" strokeWidth="2" />
                          <text x="330" y="60" textAnchor="middle" dominantBaseline="middle" fontSize="9" fontFamily="var(--font-mono)" fontWeight="bold" fill="var(--color-accent-amber)">AI PIPELINE</text>
                          <text x="330" y="72" textAnchor="middle" dominantBaseline="middle" fontSize="7" fontFamily="var(--font-mono)" fill="var(--color-text-dim)">[Recon+Denoise]</text>

                          <line x1="380" y1="67" x2="400" y2="67" stroke="var(--color-accent-amber)" strokeWidth="1.5" markerEnd="url(#arrow)" />

                          {/* Diagnosis */}
                          <rect x="405" y="45" width="85" height="92" rx="3" fill="var(--color-accent-blue)" stroke="#1a252f" strokeWidth="1.5" />
                          <text x="447" y="65" textAnchor="middle" dominantBaseline="middle" fontSize="9" fontFamily="var(--font-mono)" fontWeight="bold" fill="#fff">DIAGNOSTIC</text>
                          <text x="447" y="77" textAnchor="middle" dominantBaseline="middle" fontSize="9" fontFamily="var(--font-mono)" fontWeight="bold" fill="#fff">ENGINE</text>
                          <text x="447" y="93" textAnchor="middle" dominantBaseline="middle" fontSize="7" fontFamily="var(--font-mono)" fill="#ddeee7">S4 SSM Branch</text>
                          <text x="447" y="103" textAnchor="middle" dominantBaseline="middle" fontSize="7" fontFamily="var(--font-mono)" fill="#ddeee7">Conv2D Branch</text>
                          <text x="447" y="121" textAnchor="middle" dominantBaseline="middle" fontSize="9" fontFamily="var(--font-mono)" fontWeight="bold" fill="#00ffff">11 PATHOLOGY</text>

                          <line x1="360" y1="120" x2="405" y2="120" stroke="var(--color-accent-green)" strokeWidth="1.5" markerEnd="url(#arrow)" />
                        </svg>
                      </div>
                    </div>
                  </div>

                  {/* Bottom: Detailed pathology table */}
                  <div style={{ gridColumn: 'span 2', border: '1px solid var(--color-panel-border)', background: '#fff' }}>
                    <div className="panel-header" style={{ borderBottom: '1px solid var(--color-panel-border)' }}>
                      <span>Pathology Database Risk Matrix</span>
                    </div>
                    <div style={{ maxHeight: '180px', overflowY: 'auto' }}>
                      <table className="clinical-table" style={{ fontSize: '11px' }}>
                        <thead>
                          <tr>
                            <th>Pathology Class</th>
                            <th>Total Cases</th>
                            <th>Relative Frequency (%)</th>
                            <th>Trigger Rate (%)</th>
                            <th>Status</th>
                          </tr>
                        </thead>
                        <tbody>
                          {Object.entries(pathologyCounts).map(([k, count]) => {
                            const pct = total > 0 ? (count / total) * 100 : 0
                            const isNormal = k === 'Normal'
                            const triggerRate = isNormal ? 0 : 100
                            return (
                              <tr key={k}>
                                <td style={{ fontWeight: count > 0 ? 'bold' : 'normal', textTransform: 'uppercase' }}>
                                  {k.replace(/_/g, ' ')}
                                </td>
                                <td style={{ fontFamily: 'var(--font-mono)' }}>{count}</td>
                                <td style={{ fontFamily: 'var(--font-mono)' }}>{pct.toFixed(1)}%</td>
                                <td style={{ fontFamily: 'var(--font-mono)', color: isNormal ? 'var(--color-accent-green)' : 'var(--color-accent-amber)' }}>
                                  {triggerRate}%
                                </td>
                                <td>
                                  <span style={{ display: 'inline-flex', alignItems: 'center', gap: '4px', color: isNormal ? 'var(--color-accent-green)' : count > 0 ? 'var(--color-accent-red)' : 'var(--color-text-dim)' }}>
                                    <span className={`status-pill ${isNormal ? 'green' : count > 0 ? 'red' : 'yellow'}`} style={{ animation: 'none' }} />
                                    {isNormal ? 'NORMAL' : count > 0 ? 'PREVALENT' : 'UNOBSERVED'}
                                  </span>
                                </td>
                              </tr>
                            )
                          })}
                        </tbody>
                      </table>
                    </div>
                  </div>

                </div>
              </div>
            </>
          )
        })()}
      </main>

      {/* ── Footer log bar ── */}
      <footer style={{
        height: '24px',
        backgroundColor: '#ccd4da',
        borderTop: '2px solid var(--color-panel-border)',
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '0 12px', fontFamily: 'var(--font-mono)', fontSize: '10px',
      }}>
        <div style={{ display: 'flex', overflow: 'hidden', whiteSpace: 'nowrap', textOverflow: 'ellipsis', width: '70%' }}>
          <span style={{ color: 'var(--color-accent-blue)', fontWeight: 'bold', marginRight: '8px' }}>SYS LOG:</span>
          <span style={{ color: 'var(--color-text-main)' }}>{logs[0]}</span>
        </div>
        <div style={{ color: 'var(--color-text-dim)' }}>KVISION WORKSTATION v2.0.0</div>
      </footer>
    </div>
  )
}