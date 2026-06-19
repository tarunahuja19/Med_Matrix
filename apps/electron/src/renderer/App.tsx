import { useEffect, useState } from 'react'

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

function App() {
  const [studies, setStudies] = useState<Study[]>([])
  const [loading, setLoading] = useState(false)
  const [selectedFilePath, setSelectedFilePath] = useState<string | null>(null)
  const [uploadStatus, setUploadStatus] = useState<string>('Ready')
  const [uploading, setUploading] = useState(false)
  const [selectedStudy, setSelectedStudy] = useState<Study | null>(null)
  const [ipcStatus, setIpcStatus] = useState('Checking IPC...')
  const [logs, setLogs] = useState<string[]>([
    `[${new Date().toLocaleTimeString()}] System booted. Initializing clinical workstation.`,
    `[${new Date().toLocaleTimeString()}] Database link established.`
  ])

  const addLog = (msg: string) => {
    setLogs((prev) => [`[${new Date().toLocaleTimeString()}] ${msg}`, ...prev.slice(0, 19)])
  }

  const fetchStudies = async () => {
    setLoading(true)
    try {
      const response = await fetch('http://localhost:3000/studies')
      const data = await response.json()
      if (Array.isArray(data)) {
        setStudies(data)
        addLog(`Archive queried successfully. ${data.length} records retrieved.`)
      }
    } catch (err: any) {
      addLog(`Failed to query database archive: ${err.message}`)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if ((window as any).api) {
      (window as any).api.ping().then((res: string) => {
        setIpcStatus(`CONNECTED (pong: ${res})`)
        addLog('IPC channel verified with native process.')
      })
    } else {
      setIpcStatus('RUNNING IN BROWSER')
      addLog('No Electron API context found. Running in simulation mode.')
    }
    fetchStudies()
  }, [])

  const handleSelectFile = async () => {
    if (!(window as any).api) {
      addLog('Simulation: Open file dialog requested.')
      setSelectedFilePath('C:\\Simulation\\DICOM\\test_brain_t2.dcm')
      return
    }
    try {
      const path = await (window as any).api.openFileDialog()
      if (path) {
        setSelectedFilePath(path)
        setUploadStatus('Ready to Ingest')
        addLog(`Selected file for ingestion: ${path}`)
      }
    } catch (err: any) {
      addLog(`File selection error: ${err.message}`)
    }
  }

  const handleUpload = async () => {
    if (!selectedFilePath) return
    setUploading(true)
    setUploadStatus('Uploading raw DICOM data...')
    addLog(`Initiating raw DICOM ingestion sequence for: ${selectedFilePath}`)

    if (!(window as any).api) {
      // Simulation mode
      setTimeout(() => {
        setUploading(false)
        setSelectedFilePath(null)
        setUploadStatus('Ingested successfully (simulated)')
        addLog('Simulation: Study uploaded to MinIO & registered in database.')
        fetchStudies()
      }, 2000)
      return
    }

    try {
      const result = await (window as any).api.uploadStudy(selectedFilePath)
      if (result.success) {
        setUploadStatus('Ingested successfully')
        setSelectedFilePath(null)
        addLog(`Ingestion sequence completed. New Study ID: ${result.data.study?.id || 'Unknown'}`)
        fetchStudies()
      } else {
        setUploadStatus(`Error: ${result.error}`)
        addLog(`Ingestion sequence aborted: ${result.error}`)
      }
    } catch (err: any) {
      setUploadStatus(`Error: ${err.message}`)
      addLog(`System fault during transmission: ${err.message}`)
    } finally {
      setUploading(false)
    }
  }

  return (
    <div className="syngo-layout">
      {/* CRT scanlines effect */}
      <div className="crt-lines"></div>

      {/* Header */}
      <header className="syngo-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
          <span style={{ fontSize: '15px', color: 'var(--color-accent-blue)', letterSpacing: '1px' }}>
            KVISION // WORKSTATION
          </span>
          <span style={{ fontSize: '11px', color: 'var(--color-text-dim)' }}>
            SERIES: MAGNETOM TRIO 3T
          </span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '15px', fontSize: '11px' }}>
          <span>IPC: {ipcStatus}</span>
          <span style={{ display: 'flex', alignItems: 'center' }}>
            <span className="status-pill green"></span>
            SYSTEM ONLINE
          </span>
        </div>
      </header>

      {/* Main Workspace */}
      <main className="syngo-workspace">
        {/* Sidebar Controls */}
        <div className="syngo-panel">
          <div className="panel-header">
            <span>DICOM Acquisition</span>
            <span style={{ fontFamily: 'var(--font-mono)' }}>[ACQ-01]</span>
          </div>
          <div className="panel-body" style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            <div style={{ fontSize: '12px', lineHeight: '1.5', color: 'var(--color-text-dim)' }}>
              Select a raw DICOM (.dcm) series from local disk to parse metadata, upload to MinIO 'dicom-files' storage, and register in database.
            </div>

            <div className="bevel-inset" style={{ padding: '8px', background: '#e4e7e9', minHeight: '60px', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
              {selectedFilePath ? (
                <div>
                  <div style={{ fontWeight: 'bold', fontSize: '11px', color: 'var(--color-accent-blue)', textTransform: 'uppercase', marginBottom: '4px' }}>Target file:</div>
                  <div style={{ fontFamily: 'var(--font-mono)', wordBreak: 'break-all', fontSize: '11px' }}>{selectedFilePath.split('\\').pop()}</div>
                  <div style={{ fontSize: '10px', color: 'var(--color-text-dim)', marginTop: '2px', wordBreak: 'break-all' }}>{selectedFilePath}</div>
                </div>
              ) : (
                <div style={{ textAlign: 'center', color: 'var(--color-text-dim)', fontStyle: 'italic' }}>
                  No DICOM file selected
                </div>
              )}
            </div>

            <div style={{ display: 'flex', gap: '8px' }}>
              <button 
                onClick={handleSelectFile} 
                disabled={uploading}
                className="clinical-btn"
                style={{ flex: 1 }}
              >
                Browse File...
              </button>
              <button 
                onClick={handleUpload} 
                disabled={uploading || !selectedFilePath}
                className="clinical-btn clinical-btn-primary"
                style={{ flex: 1.2 }}
              >
                {uploading ? 'Ingesting...' : 'Ingest Study'}
              </button>
            </div>

            <div style={{ fontSize: '11px', borderTop: '1px solid var(--color-panel-border)', paddingTop: '12px' }}>
              <span style={{ fontWeight: '600', color: 'var(--color-text-dim)' }}>Ingest Status: </span>
              <span style={{ fontFamily: 'var(--font-mono)', color: uploading ? 'var(--color-accent-amber)' : 'var(--color-text-main)' }}>
                {uploadStatus}
              </span>
            </div>

            {/* Selected study detailed view */}
            <div style={{ marginTop: 'auto', borderTop: '1px solid var(--color-panel-border)', paddingTop: '12px' }}>
              <div style={{ fontWeight: '600', color: 'var(--color-accent-blue)', fontSize: '11px', textTransform: 'uppercase', marginBottom: '8px' }}>
                Study Inspector
              </div>
              {selectedStudy ? (
                <div className="detail-grid">
                  <span className="detail-label">Patient:</span>
                  <span className="detail-val">{selectedStudy.patient?.name}</span>
                  
                  <span className="detail-label">DOB:</span>
                  <span className="detail-val">
                    {new Date(selectedStudy.patient?.dateOfBirth).toLocaleDateString()}
                  </span>

                  <span className="detail-label">Gender:</span>
                  <span className="detail-val" style={{ textTransform: 'uppercase' }}>
                    {selectedStudy.patient?.gender}
                  </span>

                  <span className="detail-label">Modality:</span>
                  <span className="detail-val">{selectedStudy.modality}</span>

                  <span className="detail-label">Scan Date:</span>
                  <span className="detail-val">
                    {new Date(selectedStudy.studyDate).toLocaleString()}
                  </span>

                  <span className="detail-label">Status:</span>
                  <span className="detail-val" style={{ color: selectedStudy.status === 'complete' ? 'var(--color-accent-green)' : 'var(--color-accent-amber)' }}>
                    {selectedStudy.status}
                  </span>

                  <span className="detail-label">MinIO Key:</span>
                  <span className="detail-val" style={{ fontFamily: 'var(--font-mono)', fontSize: '9px', wordBreak: 'break-all' }}>
                    {selectedStudy.dicomKey || 'N/A'}
                  </span>
                </div>
              ) : (
                <div style={{ fontSize: '11px', color: 'var(--color-text-dim)', fontStyle: 'italic' }}>
                  Select a study from the archive list to inspect details.
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Database Grid */}
        <div className="syngo-panel">
          <div className="panel-header">
            <span>Clinical Study Archive</span>
            <button 
              onClick={fetchStudies} 
              disabled={loading}
              className="clinical-btn" 
              style={{ padding: '2px 8px', fontSize: '10px' }}
            >
              {loading ? 'Refreshing...' : 'Refresh Archive'}
            </button>
          </div>
          <div className="panel-body" style={{ padding: 0 }}>
            {loading && studies.length === 0 ? (
              <div style={{ display: 'flex', height: '100%', alignItems: 'center', justifyContent: 'center', fontStyle: 'italic', color: 'var(--color-text-dim)' }}>
                Querying database archive...
              </div>
            ) : studies.length === 0 ? (
              <div style={{ display: 'flex', height: '100%', alignItems: 'center', justifyContent: 'center', fontStyle: 'italic', color: 'var(--color-text-dim)' }}>
                Archive empty. Perform DICOM Ingestion to register records.
              </div>
            ) : (
              <table className="clinical-table">
                <thead>
                  <tr>
                    <th>Patient Name</th>
                    <th>Modality</th>
                    <th>Acquisition Date</th>
                    <th>Status</th>
                    <th>Storage Key</th>
                  </tr>
                </thead>
                <tbody>
                  {studies.map((study) => (
                    <tr 
                      key={study.id} 
                      onClick={() => setSelectedStudy(study)}
                      style={{ 
                        cursor: 'pointer',
                        backgroundColor: selectedStudy?.id === study.id ? '#e4e9ed' : 'transparent',
                        fontWeight: selectedStudy?.id === study.id ? 'bold' : 'normal'
                      }}
                    >
                      <td>{study.patient?.name}</td>
                      <td style={{ fontFamily: 'var(--font-mono)' }}>{study.modality}</td>
                      <td>{new Date(study.studyDate).toLocaleString()}</td>
                      <td>
                        <span style={{ display: 'inline-flex', alignItems: 'center', gap: '4px' }}>
                          <span className={`status-pill ${study.status === 'complete' ? 'green' : 'yellow'}`}></span>
                          {study.status}
                        </span>
                      </td>
                      <td style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--color-text-dim)' }}>
                        {study.dicomKey ? study.dicomKey.substring(0, 32) + (study.dicomKey.length > 32 ? '...' : '') : 'N/A'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </div>
      </main>

      {/* Footer Log Output */}
      <footer style={{
        height: '24px',
        backgroundColor: '#ccd4da',
        borderTop: '2px solid var(--color-panel-border)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '0 12px',
        fontFamily: 'var(--font-mono)',
        fontSize: '10px'
      }}>
        <div style={{ display: 'flex', overflow: 'hidden', whiteSpace: 'nowrap', textOverflow: 'ellipsis', width: '70%' }}>
          <span style={{ color: 'var(--color-accent-blue)', fontWeight: 'bold', marginRight: '8px' }}>SYSTEM LOGS:</span>
          <span style={{ color: 'var(--color-text-main)' }}>{logs[0]}</span>
        </div>
        <div style={{ color: 'var(--color-text-dim)' }}>
          KVISION WORKSTATION v1.0.0
        </div>
      </footer>
    </div>
  )
}

export default App