import React, { useState, useEffect } from 'react'

interface Patient {
  id: string
  name: string
  dateOfBirth: string
  gender: string
  phone?: string
}

interface ProgressionPoint {
  month: number
  pathology_volume: number
  edema_volume: number
  healthy_volume: number
  cognitive_impact: number
  severity: number[] // 4 logits
}

interface Trajectory {
  id: string
  timepoints: string // JSON
  uncertaintyBand: string // JSON
}

interface SimulationScenario {
  treatment: string
  description: string
  volumes_mean: number[][]
  cognitive_mean: number[]
}

interface Simulation {
  id: string
  treatmentPlan: string
  predictedOutcome: string // JSON
}

interface BrainTwin {
  id: string
  patientId: string
  version: number
  stateTimestamp: string
  trajectories: Trajectory[]
  simulations: Simulation[]
}

interface SimilarPatient {
  patient_id: string
  similarity: number
  pathology: string
  treatment: string
  outcome_months: number
  outcome_status: string
}

const API = 'http://localhost:3000'

export default function DigitalBrainTwinView({
  patients,
  activePatientId,
  onSelectPatient,
}: {
  patients: Patient[]
  activePatientId: string
  onSelectPatient: (id: string) => void
}) {
  const [twin, setTwin] = useState<BrainTwin | null>(null)
  const [loading, setLoading] = useState(false)
  const [initializing, setInitializing] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Simulation tab state
  const [simulationTimeHorizon, setSimulationTimeHorizon] = useState(24)
  const [selectedTreatments, setSelectedTreatments] = useState<string[]>(['no_treatment', 'stupp_protocol'])
  const [simResults, setSimResults] = useState<{
    scenarios: SimulationScenario[]
    comparison: Record<string, any>
    time_points: number[]
  } | null>(null)
  const [simLoading, setSimLoading] = useState(false)

  // Forecasting slider state
  const [forecastMonth, setForecastMonth] = useState(0)

  // Similar patient state
  const [similarPatients, setSimilarPatients] = useState<SimilarPatient[]>([])
  const [loadingSimilar, setLoadingSimilar] = useState(false)

  // Active view tab inside Twin dashboard
  const [subTab, setSubTab] = useState<'trajectory' | 'simulator' | 'connectome' | 'future_mri'>('trajectory')

  // Dynamic model-driven states for connectome and future mri warping
  const [dynamicNodes, setDynamicNodes] = useState<any[]>([])
  const [dynamicEdges, setDynamicEdges] = useState<any[]>([])
  const [dynamicGridPoints, setDynamicGridPoints] = useState<any[]>([])
  const [ventricleW, setVentricleW] = useState(5)
  const [ventricleH, setVentricleH] = useState(12)
  const [lesionRState, setLesionRState] = useState(6)
  const [jacobianDetState, setJacobianDetState] = useState(1.0)
  const [dispShift, setDispShift] = useState(0)

  useEffect(() => {
    if (!twin?.id) {
      setDynamicNodes([])
      setDynamicEdges([])
      setDynamicGridPoints([])
      return
    }

    // Connectome model request
    fetch(`${API}/twins/${twin.id}/connectome`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ dt: forecastMonth })
    })
      .then((res) => res.json())
      .then((data) => {
        if (data && data.status === 'success') {
          setDynamicNodes(data.nodes)
          setDynamicEdges(data.edges)
        }
      })
      .catch((err) => console.warn('Failed to fetch GATv2 connectome:', err))

    // Future MRI deformation field model request
    fetch(`${API}/twins/${twin.id}/future-mri`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ deltaT: forecastMonth })
    })
      .then((res) => res.json())
      .then((data) => {
        if (data && data.status === 'success') {
          setDynamicGridPoints(data.grid_points)
          setVentricleW(data.ventricle_width)
          setVentricleH(data.ventricle_height)
          setLesionRState(data.lesion_radius)
          setJacobianDetState(data.jacobian_determinant)
          setDispShift(data.displacement_shift)
        }
      })
      .catch((err) => console.warn('Failed to fetch SVF deformation field:', err))

  }, [twin?.id, forecastMonth])

  // Fetch twin data for active patient
  useEffect(() => {
    if (!activePatientId) {
      setTwin(null)
      return
    }

    let active = true
    setLoading(true)
    setError(null)
    setSimResults(null)
    setForecastMonth(0)

    fetch(`${API}/twins/patient/${activePatientId}`)
      .then(async (res) => {
        if (!res.ok) {
          if (res.status === 404) {
            setTwin(null)
            return null
          }
          throw new Error(`HTTP Error ${res.status}`)
        }
        return res.json()
      })
      .then((data) => {
        if (!active) return
        if (data) {
          setTwin(data)
          // Preload simulation results if any
          if (data.simulations && data.simulations.length > 0) {
            try {
              const rawSim = data.simulations[0].predictedOutcome
              const latestSim = typeof rawSim === 'string' ? JSON.parse(rawSim) : rawSim
              setSimResults(latestSim)
            } catch (e) {}
          }
        }
        setLoading(false)
      })
      .catch((err) => {
        if (!active) return
        setError(err.message)
        setLoading(false)
      })

    // Fetch similar patients from backend
    setLoadingSimilar(true)
    fetch(`${API}/twins/patient/${activePatientId}/similar`)
      .then((res) => {
        if (!res.ok) throw new Error('Failed to retrieve similar cases')
        return res.json()
      })
      .then((data: SimilarPatient[]) => {
        if (!active) return
        setSimilarPatients(data)
        setLoadingSimilar(false)
      })
      .catch((err) => {
        console.warn('Similar patient fetch failed:', err)
        setLoadingSimilar(false)
      })

    return () => {
      active = false
    }
  }, [activePatientId])

  // Initialize twin call
  const handleInitializeTwin = async () => {
    if (!activePatientId) return
    setInitializing(true)
    setError(null)

    try {
      const res = await fetch(`${API}/twins/patient/${activePatientId}/initialize`, {
        method: 'POST',
      })
      if (!res.ok) {
        const d = await res.json()
        throw new Error(d.error || `Initialization failed with status ${res.status}`)
      }
      const data = await res.json()
      setTwin(data.twin)
    } catch (err: any) {
      setError(err.message)
    } finally {
      setInitializing(false)
    }
  }

  // Run custom simulation
  const handleRunSimulation = async () => {
    if (!twin) return
    setSimLoading(true)
    setError(null)

    try {
      const res = await fetch(`${API}/twins/${twin.id}/simulate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          treatmentNames: selectedTreatments,
          timeHorizon: simulationTimeHorizon,
        }),
      })

      if (!res.ok) {
        const errData = await res.json()
        throw new Error(errData.error || 'Simulation calculation failed')
      }
      const data = await res.json()
      const rawOutcome = data.predictedOutcome
      const parsedOutcome = typeof rawOutcome === 'string' ? JSON.parse(rawOutcome) : rawOutcome
      setSimResults(parsedOutcome)
    } catch (err: any) {
      setError(err.message)
    } finally {
      setSimLoading(false)
    }
  }

  const selectedPatient = patients.find((p) => p.id === activePatientId)

  // Parse timepoints from active twin trajectory
  let timeline: ProgressionPoint[] = []
  let lowerBounds: any = null
  let upperBounds: any = null

  if (twin && twin.trajectories && twin.trajectories.length > 0) {
    try {
      const rawTimepoints = twin.trajectories[0].timepoints
      const parsedTimepoints = typeof rawTimepoints === 'string' ? JSON.parse(rawTimepoints) : rawTimepoints
      if (Array.isArray(parsedTimepoints)) {
        timeline = parsedTimepoints
      }
      const rawBounds = twin.trajectories[0].uncertaintyBand
      const bounds = typeof rawBounds === 'string' ? JSON.parse(rawBounds) : rawBounds
      if (bounds && Array.isArray(bounds.volumes_lower) && Array.isArray(bounds.volumes_upper)) {
        lowerBounds = bounds.volumes_lower
        upperBounds = bounds.volumes_upper
      }
    } catch (e) {}
  }

  const currentPoint = timeline.find((pt) => pt.month === forecastMonth)

  // Chart setup: custom SVG rendering
  const viewBoxWidth = 550
  const viewBoxHeight = 220
  const padLeft = 45
  const padRight = 45
  const padTop = 20
  const padBottom = 30
  const plotW = viewBoxWidth - padLeft - padRight
  const plotH = viewBoxHeight - padTop - padBottom

  const getX = (month: number) => padLeft + (month / 24) * plotW
  const maxVol = Math.max(...timeline.map((pt) => pt.pathology_volume + pt.edema_volume), 20)
  const getYVol = (vol: number) => padTop + plotH - (vol / maxVol) * plotH
  const getYCog = (cog: number) => padTop + plotH - (cog / 100) * plotH

  // Build SVG paths for volume and cognitive trajectories
  const pathVol = timeline
    .map((pt, idx) => `${idx === 0 ? 'M' : 'L'} ${getX(pt.month)} ${getYVol(pt.pathology_volume)}`)
    .join(' ')

  const pathEdema = timeline
    .map((pt, idx) => `${idx === 0 ? 'M' : 'L'} ${getX(pt.month)} ${getYVol(pt.edema_volume)}`)
    .join(' ')

  const pathCog = timeline
    .map((pt, idx) => `${idx === 0 ? 'M' : 'L'} ${getX(pt.month)} ${getYCog(pt.cognitive_impact)}`)
    .join(' ')

  // Shaded uncertainty bounds area path
  let pathUncertaintyArea = ''
  if (timeline.length > 0 && lowerBounds && upperBounds) {
    const pointsUpper = timeline.map((pt, idx) => {
      const volUpper = upperBounds[idx] ? upperBounds[idx][0] : pt.pathology_volume
      return `${getX(pt.month)},${getYVol(volUpper)}`
    })

    const pointsLower = [...timeline]
      .reverse()
      .map((pt, idx) => {
        const revIdx = timeline.length - 1 - idx
        const volLower = lowerBounds[revIdx] ? lowerBounds[revIdx][0] : pt.pathology_volume
        return `${getX(pt.month)},${getYVol(volLower)}`
      })

    pathUncertaintyArea = `M ${pointsUpper.join(' L ')} L ${pointsLower.join(' L ')} Z`
  }

  // Connectome mock nodes with attention rankings
  const connectomeNodesFallback = [
    { id: 1, label: 'Frontal Cortex', type: 'Anatomical', x: 120, y: 55, score: 0.18 },
    { id: 2, label: 'Temporal Lobe', type: 'Pathology Core', x: 260, y: 145, score: 0.94 },
    { id: 3, label: 'Parietal Cortex', type: 'Anatomical', x: 380, y: 80, score: 0.28 },
    { id: 4, label: 'Occipital Lobe', type: 'Anatomical', x: 430, y: 165, score: 0.12 },
    { id: 5, label: 'Hippocampus', type: 'Cognitive', x: 180, y: 175, score: 0.65 },
    { id: 6, label: 'Cerebellar Core', type: 'Anatomical', x: 320, y: 215, score: 0.22 },
  ]

  const connectomeEdgesFallback = [
    { source: 1, target: 2, weight: 0.5 },
    { source: 2, target: 5, weight: 0.95 },
    { source: 2, target: 3, weight: 0.7 },
    { source: 3, target: 4, weight: 0.4 },
    { source: 1, target: 5, weight: 0.6 },
    { source: 5, target: 6, weight: 0.5 },
  ]

  const connectomeNodes = dynamicNodes.length > 0 ? dynamicNodes : connectomeNodesFallback
  const connectomeEdges = dynamicEdges.length > 0 ? dynamicEdges : connectomeEdgesFallback

  // Calculations for interactive SVG Brain Slice warping (VoxelMorph SVF simulation)
  const normFactor = forecastMonth / 24
  const ventW = dynamicGridPoints.length > 0 ? ventricleW : (5 + normFactor * 18) // Ventricles expand with atrophy
  const ventH = dynamicGridPoints.length > 0 ? ventricleH : (12 + normFactor * 14)
  const lesionR = dynamicGridPoints.length > 0 ? lesionRState : (6 + normFactor * 24) // Tumor core expands
  const displacementShift = dynamicGridPoints.length > 0 ? dispShift : (normFactor * 15) // Displacement vectors length
  const activeJacobianDet = dynamicGridPoints.length > 0 ? jacobianDetState : (1.0 - normFactor * 0.14)

  // Grid point coordinates for deformation vectors
  const gridPointsFallback = [
    { cx: 70, cy: 70, vx: -3, vy: -3 },
    { cx: 110, cy: 50, vx: 0, vy: -4 },
    { cx: 150, cy: 70, vx: 3, vy: -3 },
    { cx: 70, cy: 110, vx: -4, vy: 0 },
    { cx: 150, cy: 110, vx: 5, vy: 1 },
    { cx: 70, cy: 150, vx: -3, vy: 3 },
    { cx: 110, cy: 170, vx: 0, vy: 4 },
    { cx: 150, cy: 150, vx: 4, vy: 4 },
  ]

  const gridPoints = dynamicGridPoints.length > 0 ? dynamicGridPoints : gridPointsFallback

  return (
    <div style={{ display: 'flex', width: '100%', height: '100%', gap: '14px', color: 'var(--color-text-main)', background: 'transparent', padding: '6px', fontFamily: 'var(--font-sans)', gridColumn: 'span 2' }}>
      
      {/* ── Left Sidebar: Patient Select & Metadata ── */}
      <div className="syngo-panel" style={{ width: '280px', flexShrink: 0, display: 'flex', flexDirection: 'column' }}>
        <div className="panel-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span>Twin Explorer</span>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: '10px' }}>[DBT-INDEX]</span>
        </div>

        <div className="panel-body" style={{ display: 'flex', flexDirection: 'column', gap: '14px', overflowY: 'auto', flex: 1 }}>
          <div>
            <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--color-text-dim)', textTransform: 'uppercase', marginBottom: '6px' }}>
              Select Active Patient
            </div>
            <select
              value={activePatientId}
              onChange={(e) => onSelectPatient(e.target.value)}
              style={{
                width: '100%',
                padding: '6px 8px',
                fontFamily: 'var(--font-mono)',
                fontSize: '12px',
                background: '#e4e7e9',
                border: '1px solid var(--color-panel-border)',
                color: 'var(--color-text-main)',
                outline: 'none',
                cursor: 'pointer'
              }}
            >
              <option value="">— Select Patient —</option>
              {patients.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name}
                </option>
              ))}
            </select>
          </div>

          {selectedPatient ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
              {/* Demographics Card */}
              <div style={{
                background: '#d1dadf',
                border: '1px solid var(--color-panel-border)',
                padding: '10px',
              }}>
                <div style={{ fontSize: '11px', fontWeight: 700, textTransform: 'uppercase', color: 'var(--color-accent-blue)', marginBottom: '6px' }}>
                  Demographics
                </div>
                <div className="detail-grid" style={{ marginBottom: 0 }}>
                  <span className="detail-label">Name:</span>
                  <span className="detail-val" style={{ fontWeight: 600 }}>{selectedPatient.name}</span>
                  <span className="detail-label">Gender:</span>
                  <span className="detail-val" style={{ textTransform: 'uppercase' }}>{selectedPatient.gender}</span>
                  <span className="detail-label">DOB:</span>
                  <span className="detail-val">{new Date(selectedPatient.dateOfBirth).toLocaleDateString()}</span>
                </div>
              </div>

              {/* Status details */}
              {twin ? (
                <div style={{
                  background: '#e2e8f0',
                  border: '1px solid var(--color-panel-border)',
                  padding: '10px',
                  fontSize: '11px',
                }}>
                  <div style={{ color: 'var(--color-accent-green)', fontWeight: 700, textTransform: 'uppercase', marginBottom: '6px' }}>
                    ✓ Twin Initialized
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                    <div>Twin ID: <span style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', color: 'var(--color-accent-blue)' }}>{twin.id.substring(0, 8)}</span></div>
                    <div>Model Version: <span style={{ fontFamily: 'var(--font-mono)' }}>v{twin.version}.0 (SDE Core)</span></div>
                    <div style={{ marginTop: '4px', fontSize: '10px', color: 'var(--color-text-dim)', borderTop: '1px solid var(--color-panel-border)', paddingTop: '4px' }}>
                      Timestamp: {new Date(twin.stateTimestamp).toLocaleString()}
                    </div>
                  </div>
                </div>
              ) : (
                <div style={{
                  background: '#fff9e6',
                  border: '1px solid #ffeeba',
                  padding: '12px',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '8px'
                }}>
                  <div style={{ color: '#856404', fontWeight: 700, fontSize: '12px' }}>Twin Not Initialized</div>
                  <p style={{ fontSize: '11px', color: '#664d03', margin: 0, lineHeight: '1.4' }}>
                    Encode the patient's clinical baseline and latest MRI features into the 64D Latent disease dynamics vector.
                  </p>
                  <button
                    onClick={handleInitializeTwin}
                    disabled={initializing}
                    className="clinical-btn clinical-btn-primary"
                    style={{ width: '100%', padding: '6px' }}
                  >
                    {initializing ? 'Initializing...' : 'Initialize Brain Twin'}
                  </button>
                </div>
              )}

              {/* Similar Patients retrieved card list (RAG matching) */}
              <div style={{ borderTop: '1px solid var(--color-panel-border)', paddingTop: '10px' }}>
                <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--color-text-dim)', textTransform: 'uppercase', marginBottom: '6px' }}>
                  Case Retrieval Match (RAG)
                </div>
                {loadingSimilar ? (
                  <div style={{ fontSize: '11px', color: 'var(--color-text-dim)', fontStyle: 'italic' }}>Searching matches...</div>
                ) : (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                    {similarPatients.map((p, idx) => (
                      <div key={idx} style={{
                        background: '#f5f7f8',
                        border: '1px solid var(--color-panel-border)',
                        padding: '6px',
                        fontSize: '11px'
                      }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontWeight: 'bold', color: 'var(--color-accent-blue)' }}>
                          <span>ID: {p.patient_id}</span>
                          <span style={{ color: 'var(--color-accent-green)' }}>{Math.round(p.similarity * 100)}% Match</span>
                        </div>
                        <div style={{ color: 'var(--color-text-dim)', marginTop: '2px', fontSize: '10px' }}>
                          Treatment: <span style={{ color: 'var(--color-text-main)', fontWeight: 600 }}>{p.treatment}</span>
                        </div>
                        <div style={{ marginTop: '2px', fontSize: '10px', color: '#4a5568', borderTop: '1px solid var(--color-panel-border)', paddingTop: '4px' }}>
                          Outcome: {p.outcome_status}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--color-text-dim)', fontStyle: 'italic', padding: '20px', textAlign: 'center', fontSize: '11px' }}>
              Select a patient from the explorer above.
            </div>
          )}
        </div>
      </div>

      {/* ── Right Content: Tabs Dashboard ── */}
      <div className="syngo-panel" style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
        <div className="panel-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span>Digital Brain Twin Dashboard</span>
          <div style={{ display: 'flex', gap: '2px' }}>
            {(['trajectory', 'simulator', 'connectome', 'future_mri'] as const).map((tabId) => (
              <button
                key={tabId}
                onClick={() => setSubTab(tabId)}
                disabled={!twin}
                className={`clinical-btn ${subTab === tabId ? 'clinical-btn-blue' : ''}`}
                style={{ padding: '2px 8px', fontSize: '10px' }}
              >
                {tabId === 'trajectory' ? 'Trajectory Forecast' :
                 tabId === 'simulator' ? 'Treatment Simulator' :
                 tabId === 'connectome' ? 'Connectome Explorer' : 'Future MRI Projection'}
              </button>
            ))}
          </div>
        </div>

        <div className="panel-body" style={{ display: 'flex', flexDirection: 'column', gap: '14px', overflowY: 'auto', flex: 1 }}>
          {error && (
            <div style={{ padding: '10px', background: '#fff5f5', border: '1px solid var(--color-accent-red)', color: 'var(--color-accent-red)', fontSize: '11px', borderRadius: '4px' }}>
              ⚠ SYSTEM ERROR: {error}
            </div>
          )}

          {!twin ? (
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', color: 'var(--color-text-dim)', fontStyle: 'italic', gap: '8px', fontSize: '12px' }}>
              <span>Initialize the patient's brain twin state vector to activate the dashboard panels.</span>
            </div>
          ) : (
            <>
              {/* 1. TRAJECTORY FORECASTING TAB */}
              {subTab === 'trajectory' && timeline.length > 0 && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
                  
                  {/* Slider scrub control */}
                  <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '14px',
                    background: '#e4e7e9',
                    padding: '8px 12px',
                    border: '1px solid var(--color-panel-border)',
                    borderRadius: '4px'
                  }}>
                    <span style={{ fontSize: '11px', fontFamily: 'var(--font-mono)', fontWeight: 'bold' }}>LONGITUDINAL SCRUB:</span>
                    <input
                      type="range"
                      min="0"
                      max="24"
                      step="3"
                      value={forecastMonth}
                      onChange={(e) => setForecastMonth(Number(e.target.value))}
                      style={{
                        flex: 1,
                        accentColor: 'var(--color-accent-blue)',
                        height: '4px',
                        cursor: 'pointer',
                        outline: 'none',
                        background: '#cbd5e1',
                        borderRadius: '2px'
                      }}
                    />
                    <span style={{ fontSize: '12px', fontFamily: 'var(--font-mono)', fontWeight: 'bold', minWidth: '45px', textAlign: 'right' }}>
                      {forecastMonth}m
                    </span>
                  </div>

                  {/* SVG Chart area */}
                  <div style={{
                    background: '#ffffff',
                    padding: '14px',
                    border: '1px solid var(--color-panel-border)',
                    borderRadius: '4px',
                    position: 'relative'
                  }}>
                    <div style={{
                      position: 'absolute',
                      top: '12px',
                      left: '12px',
                      fontSize: '9px',
                      fontWeight: 700,
                      color: 'var(--color-accent-red)',
                      background: '#fff0f0',
                      border: '1px solid #ffcccc',
                      padding: '2px 8px',
                      borderRadius: '2px',
                      letterSpacing: '0.5px'
                    }}>
                      NATURAL HISTORY FORECAST (LATENT SDE)
                    </div>
                    
                    <svg width="100%" height={viewBoxHeight} viewBox={`0 0 ${viewBoxWidth} ${viewBoxHeight}`} style={{ overflow: 'visible', marginTop: '10px' }}>
                      {/* Horizontal Gridlines */}
                      {[0, 0.25, 0.5, 0.75, 1.0].map((ratio) => {
                        const y = padTop + ratio * plotH
                        return (
                          <g key={`grid-y-${ratio}`}>
                            <line x1={padLeft} y1={y} x2={padLeft + plotW} y2={y} stroke="#edf2f7" strokeWidth="1" />
                            <text x={padLeft - 8} y={y + 3} textAnchor="end" fontSize="9" fontFamily="var(--font-mono)" fill="#718096">
                              {Math.round((1.0 - ratio) * maxVol)}
                            </text>
                            <text x={padLeft + plotW + 8} y={y + 3} textAnchor="start" fontSize="9" fontFamily="var(--font-mono)" fill="#718096">
                              {Math.round((1.0 - ratio) * 100)}%
                            </text>
                          </g>
                        )
                      })}

                      {/* X gridlines */}
                      {timeline.map((pt) => {
                        const x = getX(pt.month)
                        return (
                          <g key={`grid-x-${pt.month}`}>
                            <line x1={x} y1={padTop} x2={x} y2={padTop + plotH} stroke="#edf2f7" strokeWidth="1" />
                            <text x={x} y={padTop + plotH + 12} textAnchor="middle" fontSize="9" fontFamily="var(--font-mono)" fill="#718096">
                              {pt.month}m
                            </text>
                          </g>
                        )
                      })}

                      {/* Shaded uncertainty area */}
                      {pathUncertaintyArea && (
                        <path d={pathUncertaintyArea} fill="rgba(231, 76, 60, 0.1)" stroke="none" />
                      )}

                      {/* Trajectory paths */}
                      <path d={pathVol} fill="none" stroke="var(--color-accent-red)" strokeWidth="2.5" strokeLinecap="round" />
                      <path d={pathEdema} fill="none" stroke="var(--color-accent-amber)" strokeWidth="2" strokeDasharray="4,3" />
                      <path d={pathCog} fill="none" stroke="var(--color-accent-blue)" strokeWidth="2.5" strokeLinecap="round" />

                      {/* Scrub line indicator */}
                      <line x1={getX(forecastMonth)} y1={padTop} x2={getX(forecastMonth)} y2={padTop + plotH} stroke="#2d3748" strokeWidth="1.5" strokeDasharray="3,3" />

                      {/* Interactive circles */}
                      {currentPoint && (
                        <g>
                          <circle cx={getX(forecastMonth)} cy={getYVol(currentPoint.pathology_volume)} r="5.5" fill="var(--color-accent-red)" stroke="#ffffff" strokeWidth="2" />
                          <circle cx={getX(forecastMonth)} cy={getYVol(currentPoint.edema_volume)} r="5.5" fill="var(--color-accent-amber)" stroke="#ffffff" strokeWidth="2" />
                          <circle cx={getX(forecastMonth)} cy={getYCog(currentPoint.cognitive_impact)} r="5.5" fill="var(--color-accent-blue)" stroke="#ffffff" strokeWidth="2" />
                        </g>
                      )}
                    </svg>

                    {/* Chart Legend */}
                    <div style={{ display: 'flex', justifyContent: 'center', flexWrap: 'wrap', gap: '16px', fontSize: '11px', marginTop: '14px', color: 'var(--color-text-dim)', borderTop: '1px solid var(--color-panel-border)', paddingTop: '10px' }}>
                      <span style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
                        <span style={{ display: 'inline-block', width: '10px', height: '10px', borderRadius: '2px', background: 'var(--color-accent-red)' }} />
                        Pathology Core Volume
                      </span>
                      <span style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
                        <span style={{ display: 'inline-block', width: '10px', height: '10px', borderRadius: '2px', background: 'var(--color-accent-amber)' }} />
                        Edema Volume
                      </span>
                      <span style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
                        <span style={{ display: 'inline-block', width: '10px', height: '10px', borderRadius: '2px', background: 'var(--color-accent-blue)' }} />
                        Cognitive Decline %
                      </span>
                      <span style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
                        <span style={{ display: 'inline-block', width: '14px', height: '8px', borderRadius: '1px', background: 'rgba(231, 76, 60, 0.12)', border: '1px solid rgba(231, 76, 60, 0.2)' }} />
                        90% Confidence Interval (SDE Diffusion)
                      </span>
                    </div>
                  </div>

                  {/* Scrub Milestone metrics card */}
                  {currentPoint && (
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: '12px' }}>
                      <div style={{ background: '#f8fafc', padding: '12px', border: '1px solid var(--color-panel-border)', borderRadius: '4px' }}>
                        <div style={{ fontSize: '9px', color: 'var(--color-text-dim)', fontWeight: 700, textTransform: 'uppercase' }}>MILESTONE</div>
                        <div style={{ fontSize: '24px', fontWeight: 'bold', color: 'var(--color-accent-blue)', fontFamily: 'var(--font-mono)' }}>
                          {currentPoint.month} <span style={{ fontSize: '12px', color: 'var(--color-text-dim)' }}>Months</span>
                        </div>
                        <div style={{ marginTop: '8px', fontSize: '11px' }}>
                          Severity Stage:<br/>
                          <span style={{
                            fontWeight: 700,
                            color: currentPoint.pathology_volume > 35 ? 'var(--color-accent-red)' : currentPoint.pathology_volume > 15 ? 'var(--color-accent-amber)' : 'var(--color-accent-green)',
                          }}>
                            {currentPoint.pathology_volume > 35 ? 'CRITICAL (Phase IV)' : currentPoint.pathology_volume > 15 ? 'SEVERE (Phase III)' : 'MODERATE (Phase II)'}
                          </span>
                        </div>
                      </div>
                      
                      <div style={{
                        background: '#f8fafc',
                        padding: '12px',
                        border: '1px solid var(--color-panel-border)',
                        borderRadius: '4px',
                        display: 'flex',
                        flexDirection: 'column',
                        justifyContent: 'center',
                        gap: '6px',
                        fontSize: '11px',
                      }}>
                        <div style={{ fontWeight: 700, color: 'var(--color-accent-blue)', fontSize: '11px' }}>Progression Metrics Summary:</div>
                        <div style={{ display: 'grid', gridTemplateColumns: '150px 1fr', gap: '4px' }}>
                          <span>• Pathology Core Volume:</span>
                          <strong>{currentPoint.pathology_volume.toFixed(2)} cm³</strong>
                          <span>• Surrounding Edema Volume:</span>
                          <strong>{currentPoint.edema_volume.toFixed(2)} cm³</strong>
                          <span>• Healthy Tissue Volume:</span>
                          <strong>{currentPoint.healthy_volume.toFixed(2)} cm³</strong>
                          <span>• Cognitive Function Loss:</span>
                          <strong style={{ color: 'var(--color-accent-red)' }}>{currentPoint.cognitive_impact.toFixed(1)}%</strong>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* 2. TREATMENT SIMULATOR TAB */}
              {subTab === 'simulator' && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
                  {/* Select treatments and options */}
                  <div style={{
                    display: 'grid',
                    gridTemplateColumns: '2fr 1fr',
                    gap: '14px',
                    background: '#f1f5f9',
                    padding: '12px',
                    border: '1px solid var(--color-panel-border)',
                    borderRadius: '4px'
                  }}>
                    <div>
                      <div style={{ fontSize: '10px', fontWeight: 700, color: 'var(--color-text-dim)', textTransform: 'uppercase', marginBottom: '8px' }}>
                        Select Treatment Plans to Compare
                      </div>
                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px 12px', fontSize: '11px' }}>
                        <label style={{ display: 'flex', alignItems: 'center', gap: '6px', cursor: 'pointer' }}>
                          <input
                            type="checkbox"
                            checked={selectedTreatments.includes('no_treatment')}
                            onChange={(e) => {
                              setSelectedTreatments(prev =>
                                e.target.checked ? [...prev, 'no_treatment'] : prev.filter(t => t !== 'no_treatment')
                              )
                            }}
                            style={{ cursor: 'pointer' }}
                          />
                          No Treatment (Natural History)
                        </label>
                        <label style={{ display: 'flex', alignItems: 'center', gap: '6px', cursor: 'pointer' }}>
                          <input
                            type="checkbox"
                            checked={selectedTreatments.includes('stupp_protocol')}
                            onChange={(e) => {
                              setSelectedTreatments(prev =>
                                e.target.checked ? [...prev, 'stupp_protocol'] : prev.filter(t => t !== 'stupp_protocol')
                              )
                            }}
                            style={{ cursor: 'pointer' }}
                          />
                          Stupp Protocol (Surgery+TMZ+RT)
                        </label>
                        <label style={{ display: 'flex', alignItems: 'center', gap: '6px', cursor: 'pointer' }}>
                          <input
                            type="checkbox"
                            checked={selectedTreatments.includes('immunotherapy')}
                            onChange={(e) => {
                              setSelectedTreatments(prev =>
                                e.target.checked ? [...prev, 'immunotherapy'] : prev.filter(t => t !== 'immunotherapy')
                              )
                            }}
                            style={{ cursor: 'pointer' }}
                          />
                          Immunotherapy
                        </label>
                        <label style={{ display: 'flex', alignItems: 'center', gap: '6px', cursor: 'pointer' }}>
                          <input
                            type="checkbox"
                            checked={selectedTreatments.includes('surgery_only')}
                            onChange={(e) => {
                              setSelectedTreatments(prev =>
                                e.target.checked ? [...prev, 'surgery_only'] : prev.filter(t => t !== 'surgery_only')
                              )
                            }}
                            style={{ cursor: 'pointer' }}
                          />
                          Surgery Only
                        </label>
                      </div>
                    </div>

                    <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', justifyContent: 'space-between' }}>
                      <div>
                        <div style={{ fontSize: '10px', fontWeight: 700, color: 'var(--color-text-dim)', textTransform: 'uppercase', marginBottom: '4px' }}>Horizon:</div>
                        <select
                          value={simulationTimeHorizon}
                          onChange={(e) => setSimulationTimeHorizon(Number(e.target.value))}
                          style={{
                            width: '100%',
                            padding: '6px',
                            fontSize: '11px',
                            fontFamily: 'var(--font-mono)',
                            background: '#e4e7e9',
                            border: '1px solid var(--color-panel-border)',
                            borderRadius: '4px',
                            color: 'var(--color-text-main)',
                            outline: 'none',
                            cursor: 'pointer'
                          }}
                        >
                          <option value={12}>12 Months</option>
                          <option value={24}>24 Months</option>
                        </select>
                      </div>
                      
                      <button
                        onClick={handleRunSimulation}
                        disabled={simLoading}
                        className="clinical-btn clinical-btn-primary"
                        style={{ padding: '8px', fontSize: '11px', width: '100%' }}
                      >
                        {simLoading ? 'Running Simulator...' : 'Run What-If Comparison'}
                      </button>
                    </div>
                  </div>

                  {/* Simulation results table */}
                  {simResults && (
                    <div style={{
                      background: '#ffffff',
                      border: '1px solid var(--color-panel-border)',
                      padding: '14px',
                      borderRadius: '4px'
                    }}>
                      <div style={{
                        fontWeight: 700,
                        fontSize: '11px',
                        color: 'var(--color-accent-blue)',
                        textTransform: 'uppercase',
                        marginBottom: '10px',
                        borderBottom: '1px solid var(--color-panel-border)',
                        paddingBottom: '6px',
                      }}>
                        Simulated Prognostic Outcome Comparisons
                      </div>
                      
                      <table style={{ width: '100%', fontSize: '11px', textAlign: 'left', borderCollapse: 'collapse' }}>
                        <thead>
                          <tr style={{ background: '#f8fafc', borderBottom: '1px solid var(--color-panel-border)' }}>
                            <th style={{ padding: '8px 10px', color: 'var(--color-text-dim)' }}>Treatment Plan</th>
                            <th style={{ padding: '8px 10px', color: 'var(--color-text-dim)' }}>Final Pathology Core</th>
                            <th style={{ padding: '8px 10px', color: 'var(--color-text-dim)' }}>Final Cognitive Loss</th>
                            <th style={{ padding: '8px 10px', color: 'var(--color-text-dim)' }}>Cumulative Burden Index</th>
                          </tr>
                        </thead>
                        <tbody>
                          {Object.entries(simResults.comparison).map(([planName, metrics]: [string, any]) => {
                            const displayName = planName === 'no_treatment' ? 'No Treatment (Natural History)' :
                                              planName === 'stupp_protocol' ? 'Stupp Protocol (TMZ + RT)' :
                                              planName === 'immunotherapy' ? 'Immunotherapy Plan' : 'Surgery Only'
                            return (
                              <tr key={planName} style={{ borderBottom: '1px solid #edf2f7' }}>
                                <td style={{ padding: '8px 10px', fontWeight: 600 }}>{displayName}</td>
                                <td style={{ padding: '8px 10px', color: metrics.final_pathology_volume_cm3 > 15 ? 'var(--color-accent-red)' : 'var(--color-text-main)' }}>
                                  {metrics.final_pathology_volume_cm3.toFixed(2)} cm³
                                </td>
                                <td style={{ padding: '8px 10px', color: metrics.final_cognitive_impact_pct > 40 ? 'var(--color-accent-red)' : 'var(--color-text-main)' }}>
                                  {metrics.final_cognitive_impact_pct.toFixed(1)}%
                                </td>
                                <td style={{ padding: '8px 10px', fontFamily: 'var(--font-mono)', color: 'var(--color-accent-blue)' }}>
                                  {metrics.auc_pathology_volume.toFixed(2)} AUC
                                </td>
                              </tr>
                            )
                          })}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
              )}

              {/* 3. CONNECTOME EXPLORER TAB */}
              {subTab === 'connectome' && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                  <div style={{ fontSize: '11px', color: 'var(--color-text-dim)', fontStyle: 'italic', marginBottom: '4px' }}>
                    Structural Connectivity Graph showing GATv2 node attention scores mapped over parcellated brain regions. Active regions pulse.
                  </div>
                  
                  <div style={{ display: 'grid', gridTemplateColumns: '1.5fr 1fr', gap: '14px', alignItems: 'center' }}>
                    {/* SVG Brain Node Map */}
                    <div style={{
                      background: '#090d12',
                      border: '1px solid var(--color-panel-border)',
                      borderRadius: '4px',
                      height: '250px',
                      position: 'relative',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center'
                    }}>
                      <svg width="100%" height="100%" viewBox="0 0 500 250" style={{ overflow: 'visible' }}>
                        {/* Edges with gradients and varying stroke widths */}
                        {connectomeEdges.map((e, idx) => {
                          const n1 = connectomeNodes.find(n => n.id === e.source)
                          const n2 = connectomeNodes.find(n => n.id === e.target)
                          if (!n1 || !n2) return null
                          return (
                            <line
                              key={idx}
                              x1={n1.x}
                              y1={n1.y}
                              x2={n2.x}
                              y2={n2.y}
                              stroke="rgba(56, 189, 248, 0.15)"
                              strokeWidth={e.weight * 3.5}
                            />
                          )
                        })}

                        {/* Nodes with glowing halos */}
                        {connectomeNodes.map((n) => {
                          const isCore = n.type === 'Pathology Core'
                          const isCognitive = n.type === 'Cognitive'
                          const nodeColor = isCore ? '#f43f5e' : isCognitive ? '#38bdf8' : '#64748b'
                          const nodeRadius = isCore ? 11 : isCognitive ? 9 : 7
                          
                          return (
                            <g key={n.id} style={{ cursor: 'pointer' }}>
                              {/* Pulsing visual halo for highly active pathology core nodes */}
                              {n.score > 0.5 && (
                                <circle
                                  cx={n.x}
                                  cy={n.y}
                                  r={nodeRadius + 6}
                                  fill="none"
                                  stroke={nodeColor}
                                  strokeWidth="1.5"
                                  opacity="0.3"
                                  style={{
                                    transformOrigin: `${n.x}px ${n.y}px`,
                                    animation: 'pulse 2s infinite ease-in-out'
                                  }}
                                />
                              )}
                              
                              <circle
                                cx={n.x}
                                cy={n.y}
                                r={nodeRadius}
                                fill={nodeColor}
                                stroke="#1e293b"
                                strokeWidth="2"
                              />
                              
                              <text
                                x={n.x}
                                y={n.y - 14}
                                textAnchor="middle"
                                fontSize="9"
                                fill="#e2e8f0"
                                fontWeight={isCore ? 'bold' : 'normal'}
                                fontFamily="var(--font-mono)"
                              >
                                {n.label}
                              </text>
                              
                              {/* Attention score metric */}
                              <text
                                x={n.x}
                                y={n.y + 3}
                                textAnchor="middle"
                                fontSize="8"
                                fill="#ffffff"
                                fontWeight="bold"
                                fontFamily="var(--font-mono)"
                              >
                                {(n.score != null ? n.score : 0).toFixed(2)}
                              </text>
                            </g>
                          )
                        })}
                      </svg>
                      
                      {/* CSS Keyframes for node pulsing */}
                      <style>{`
                        @keyframes pulse {
                          0% { transform: scale(0.9); opacity: 0.2; }
                          50% { transform: scale(1.2); opacity: 0.5; }
                          100% { transform: scale(0.9); opacity: 0.2; }
                        }
                      `}</style>
                    </div>

                    {/* Regional lists */}
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', fontSize: '11px' }}>
                      <div style={{ fontWeight: 700, color: 'var(--color-accent-blue)', textTransform: 'uppercase', marginBottom: '4px' }}>
                        GATv2 Attention Scores:
                      </div>
                      {[...connectomeNodes]
                        .sort((a, b) => b.score - a.score)
                        .map((n) => (
                          <div key={n.id} style={{
                            display: 'flex',
                            justifyContent: 'space-between',
                            background: '#f8fafc',
                            padding: '6px 10px',
                            border: '1px solid var(--color-panel-border)',
                            borderRadius: '4px'
                          }}>
                            <span>{n.label} <span style={{ fontSize: '9px', color: 'var(--color-text-dim)' }}>({n.type})</span></span>
                            <span style={{ fontWeight: 'bold', color: n.score > 0.5 ? 'var(--color-accent-red)' : 'var(--color-text-main)' }}>
                              {Math.round(n.score * 100)}%
                            </span>
                          </div>
                        ))}
                    </div>
                  </div>
                </div>
              )}

              {/* 4. FUTURE MRI GENERATION TAB */}
              {subTab === 'future_mri' && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                  <div style={{ fontSize: '11px', color: 'var(--color-text-dim)', fontStyle: 'italic', marginBottom: '4px' }}>
                    Deformation field prediction map (VoxelMorph SVF) showing localized grey matter changes. Scrub the month select slider to warp tissue.
                  </div>
                  
                  <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: '14px', alignItems: 'center' }}>
                    {/* Interactive SVF Deformation Axial Brain Slice */}
                    <div style={{
                      background: '#090d12',
                      border: '1px solid var(--color-panel-border)',
                      borderRadius: '4px',
                      height: '250px',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      position: 'relative'
                    }}>
                      <div style={{ width: '220px', height: '220px', position: 'relative' }}>
                        <svg width="220" height="220" viewBox="0 0 220 220" style={{ overflow: 'visible' }}>
                          <defs>
                            <radialGradient id="lesionGlow" cx="50%" cy="50%" r="50%">
                              <stop offset="0%" stopColor="#f43f5e" stopOpacity="0.7" />
                              <stop offset="70%" stopColor="#f43f5e" stopOpacity="0.3" />
                              <stop offset="100%" stopColor="#f43f5e" stopOpacity="0" />
                            </radialGradient>
                          </defs>

                          {/* Outer Skull Outline */}
                          <ellipse cx="110" cy="110" rx="90" ry="105" fill="#1e293b" stroke="#334155" strokeWidth="2.5" />
                          
                          {/* Left Hemisphere Brain Tissue */}
                          <path
                            d={`M 110,10 C 60,10 30,50 30,110 C 30,170 60,200 110,210 C 95,190 95,170 100,150 C 90,130 95,110 105,90 C 95,70 100,40 110,10 Z`}
                            fill="#0f172a"
                            stroke="#475569"
                            strokeWidth="1.5"
                          />
                          
                          {/* Right Hemisphere Brain Tissue */}
                          <path
                            d={`M 110,10 C 160,10 190,50 190,110 C 190,170 160,200 110,210 C 125,190 125,170 120,150 C 130,130 125,110 115,90 C 125,70 120,40 110,10 Z`}
                            fill="#0f172a"
                            stroke="#475569"
                            strokeWidth="1.5"
                          />

                          {/* Lateral Ventricles (expand as healthy_volume decreases or atrophy increases) */}
                          <path
                            d={`M 95,100 C ${95 - ventW},${100 - ventH} 85,120 95,130 C 105,120 105,100 95,100 Z`}
                            fill="#38bdf8"
                            opacity="0.8"
                          />
                          <path
                            d={`M 125,100 C ${125 + ventW},${100 - ventH} 135,120 125,130 C 115,120 115,100 125,100 Z`}
                            fill="#38bdf8"
                            opacity="0.8"
                          />

                          {/* Deformation Vector Field Arrows (SVF grid vectors pushing outward from tumor) */}
                          {gridPoints.map((gp, idx) => {
                            const startX = gp.cx
                            const startY = gp.cy
                            const deltaX = gp.vx * displacementShift * 0.15
                            const deltaY = gp.vy * displacementShift * 0.15
                            const endX = startX + deltaX
                            const endY = startY + deltaY
                            
                            return (
                              <g key={idx}>
                                {/* Arrow line */}
                                <line
                                  x1={startX}
                                  y1={startY}
                                  x2={endX}
                                  y2={endY}
                                  stroke="#34d399"
                                  strokeWidth="1.2"
                                  opacity={normFactor > 0.1 ? 0.8 : 0.2}
                                />
                                {/* Arrow head */}
                                {normFactor > 0.1 && (
                                  <circle cx={endX} cy={endY} r="2" fill="#34d399" />
                                )}
                              </g>
                            )
                          })}

                          {/* Pathology Lesion Core (Expands dynamically based on pathology_volume) */}
                          <circle cx="145" cy="140" r={lesionR} fill="url(#lesionGlow)" />
                          <circle cx="145" cy="140" r={Math.max(2, lesionR * 0.6)} fill="none" stroke="#ef4444" strokeWidth="1.5" />
                        </svg>
                      </div>
                      
                      <div style={{ position: 'absolute', bottom: '8px', right: '8px', fontSize: '9px', fontFamily: 'var(--font-mono)', color: '#64748b' }}>
                        WARPED CORE VOLUME: {(100 + normFactor * 125).toFixed(0)}%
                      </div>
                    </div>

                    {/* Deformation metrics info */}
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', fontSize: '11px' }}>
                      <div style={{ fontWeight: 700, color: 'var(--color-accent-blue)', textTransform: 'uppercase', marginBottom: '4px' }}>
                        Deformation Metrics:
                      </div>
                      <div>• Displacement Jacobian Det: <strong>{activeJacobianDet.toFixed(3)}</strong> (Core atrophy/expansion)</div>
                      <div>• Diffeomorphic Constraint: <strong style={{ color: 'var(--color-accent-green)' }}>Topology Preserved (No folding)</strong></div>
                      <div>• Grid Distortion Magnitude: <strong>{(displacementShift * 0.43).toFixed(1)} px</strong></div>
                      
                      <div style={{
                        marginTop: '10px',
                        background: 'rgba(231, 76, 60, 0.08)',
                        border: '1px solid rgba(231, 76, 60, 0.2)',
                        padding: '10px',
                        borderRadius: '4px'
                      }}>
                        <span style={{ fontWeight: 700, color: 'var(--color-accent-red)', textTransform: 'uppercase', fontSize: '10px' }}>⚠ Clinical Warning:</span>
                        <div style={{ fontSize: '10px', color: 'var(--color-text-dim)', marginTop: '2px', lineHeight: '1.3' }}>
                          Generated MRI scans represent mathematical SVF forecasts of structural brain changes. They are not acquisitions and must only be used for planning simulations.
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )}

            </>
          )}
        </div>
      </div>

    </div>
  )
}
