import { useEffect, useRef, useState } from 'react'

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
}

// ── Tabs ─────────────────────────────────────────────────────────────────────
type Tab = 'ingest' | 'archive' | 'patients' | 'reports' | 'brain3d'

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

// ── App ───────────────────────────────────────────────────────────────────────
export default function App() {
  // navigation
  const [tab, setTab] = useState<Tab>('ingest')
  const [brain3dSelectedPatientId, setBrain3dSelectedPatientId] = useState<string>('')

  // data
  const [patients, setPatients] = useState<Patient[]>([])
  const [studies, setStudies] = useState<Study[]>([])
  const [reports, setReports] = useState<Report[]>([])

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

  // IPC / browser detection
  const [ipcStatus, setIpcStatus] = useState('Checking IPC...')
  const [logs, setLogs] = useState<string[]>([
    `[${new Date().toLocaleTimeString()}] System booted. Initializing KVISION clinical workstation.`,
    `[${new Date().toLocaleTimeString()}] Database link established.`,
  ])

  const addLog = (msg: string) =>
    setLogs((prev) => [`[${new Date().toLocaleTimeString()}] ${msg}`, ...prev.slice(0, 49)])

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
  ]

  return (
    <div className="syngo-layout">
      <div className="crt-lines" />

      {/* ── Header ── */}
      <header className="syngo-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
          <span style={{ fontSize: '15px', color: 'var(--color-accent-blue)', letterSpacing: '1px' }}>
            KVISION // WORKSTATION
          </span>
          <span style={{ fontSize: '11px', color: 'var(--color-text-dim)' }}>
            SERIES: MAGNETOM TRIO 3T
          </span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '20px', fontSize: '11px' }}>
          {/* Tabs */}
          {TABS.map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className="clinical-btn"
              style={{
                padding: '2px 10px',
                fontSize: '10px',
                background: tab === t.id ? 'var(--color-accent-blue)' : undefined,
                color: tab === t.id ? '#fff' : undefined,
                borderColor: tab === t.id ? '#1e4f70' : undefined,
              }}
            >
              {t.label}
            </button>
          ))}
          <span>IPC: {ipcStatus}</span>
          <span style={{ display: 'flex', alignItems: 'center' }}>
            <span className="status-pill green" />
            SYSTEM ONLINE
          </span>
        </div>
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
              <div className="panel-body" style={{ padding: 0, height: '100%', overflow: 'hidden' }}>
                <iframe
                  src="./brain2print/index.html"
                  style={{ width: '100%', height: '100%', border: 'none' }}
                  title="brain2print 3D Brain Model"
                />
              </div>
            </div>
          </>
        )}
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