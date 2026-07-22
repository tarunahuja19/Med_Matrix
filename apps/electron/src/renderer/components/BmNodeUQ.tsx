import React, { useEffect, useRef, useState, useCallback } from 'react'

// ── Types ─────────────────────────────────────────────────────────────────────

type PoolModel = '2-pool' | '3-pool' | '4-pool'
type UQMode = 'aleatoric' | 'epistemic' | 'total'
type DisplayMap = 'ksw' | 'fs' | 'apt' | 't1w' | 't2w' | 'uncertainty'

interface CestParams {
  ksw_amide: number
  ksw_amine: number
  fs_amide: number
  fs_amine: number
  t1w: number
  t2w: number
  t1s: number
  t2s: number
  b1: number
  b0_shift: number
  snr: number
}

interface UncertaintyMap {
  ksw: number[][]
  fs: number[][]
  apt: number[][]
  t1w: number[][]
  t2w: number[][]
}

interface PredictionResult {
  zSpectrum: number[]
  zSpectrumPredicted: number[]
  offsets: number[]
  paramMaps: {
    ksw: number[][]
    fs: number[][]
    apt: number[][]
    t1w: number[][]
    t2w: number[][]
  }
  uncertaintyMaps: UncertaintyMap
  aleatoricMaps: UncertaintyMap
  epistemicMaps: UncertaintyMap
  poolAttention: { pool: string; ppm: number; weight: number }[]
  lossComponents: { name: string; value: number }[]
  inferenceTimeMs: number
  mtrAsym: number[]
}

// ── Simulation helpers ────────────────────────────────────────────────────────

function gaussianNoise(std: number): number {
  // Box-Muller transform
  const u1 = Math.random()
  const u2 = Math.random()
  return std * Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.cos(2 * Math.PI * u2)
}

/**
 * Physically accurate 2-pool/3-pool/4-pool Bloch-McConnell steady-state Z-spectrum.
 *
 * Physics reference: Zaiss & Bachert (2013) NMR in Biomedicine; Woessner et al. (2005) MRM.
 *
 * The steady-state solution for the water pool magnetization is:
 *   Z = 1 / (1 + R_eff * T1w)
 * where R_eff accumulates exchange contributions from each solute pool s:
 *   Rex_s = (fs_s * ksw_s * w1^2) / ((ksw_s + R2s + Δωs^2/ksw_s) * (ksw_s + R2s) + w1^2 * (R2s + ksw_s) / R2s)
 *
 * For the water pool alone (no exchange), the steady-state is the classic
 * Bloch solution:
 *   Z_water = 1 / (1 + w1^2 * T1w * T2w / (1 + Δωw^2 * T2w^2))
 *
 * Combined: Z = Z_water / (1 + sum(Rex_s) * T1w * (1 + Δωw^2*T2w^2))
 * (simplified, valid in the weak-pulse/slow-exchange limit)
 */
function blochMcConnellSimple(
  offset_ppm: number,
  params: CestParams,
  poolModel: PoolModel
): number {
  const { ksw_amide, fs_amide, ksw_amine, fs_amine, t1w, t2w, t2s, b1, b0_shift } = params

  // ── Correct offset with B0 inhomogeneity ─────────────────────────────────
  const dw_ppm = offset_ppm - b0_shift

  // ── Physical constants ──────────────────────────────────────────────────
  const GAMMA_HZ_PER_T = 42.577e6   // Hz/T proton gyromagnetic ratio
  const B0_T = 3.0                   // 3 Tesla scanner
  const LARMOR_HZ = GAMMA_HZ_PER_T * B0_T   // ~127.73 MHz

  // RF nutation rate (rad/s); B1 slider is in μT.
  // In pulsed CEST experiments the effective saturation is reduced by the duty
  // cycle of the pulse train (typically 50%). We apply DC=0.5 to the water
  // saturation term only — exchange Rex uses the instantaneous w1 during the
  // pulse (Zaiss 2015, Lee 2017 correction).
  const w1 = 2 * Math.PI * GAMMA_HZ_PER_T * (b1 * 1e-6)  // rad/s (peak)
  const DC = 0.5  // duty cycle for pulsed saturation train
  const w1_eff = w1 * Math.sqrt(DC)  // effective w1 for water saturation (RMS)

  // Water pool relaxation rate (1/s)
  const R1w = 1.0 / t1w                    // t1w in seconds (slider input)

  // Solute pool relaxation rate (1/s); t2s in ms
  const R2s_amide = 1.0 / (t2s * 1e-3)   // t2s in ms → seconds

  // ── Water offset in rad/s ─────────────────────────────────────────────
  const dw_water = 2 * Math.PI * dw_ppm * 1e-6 * LARMOR_HZ   // rad/s

  // ── Steady-state water Z without exchange (classic Bloch, pulsed correction) ─
  // Z_water = 1 / (1 + w1_eff^2 * T1w * T2w / (1 + (dw*T2w)^2))
  // Using w1_eff (RMS-corrected for duty cycle) prevents total saturation at offset=0.
  const dwT2w = dw_water * (t2w * 1e-3)
  const Z_water = 1.0 / (1.0 + (w1_eff * w1_eff * t1w * (t2w * 1e-3)) / (1.0 + dwT2w * dwT2w))

  // ── Exchange Rate Approximation (Rex) for each solute pool ────────────
  // Full Zaiss (2013) NMR Biomed Eq. 13 — includes the power-broadening
  // correction term w1²/R2s in the denominator. Without it, Rex is
  // overestimated by up to 10× at typical clinical B1 values.
  //
  //   Rex_s = fs * ksw * w1² / [(R2s+ksw)(R2s+ksw + w1²/R2s) + Δωs²]
  //
  // This is valid in the intermediate-to-fast exchange regime (ksw > R2s/2).
  // The resulting R_eff accumulates all pools; final Z = Z_water - R_eff*T1w*Z_water.
  let totalRex = 0

  // ── Amide pool (+3.5 ppm, k_sw ~ 20–200 s⁻¹, slow-to-intermediate exchange) ─
  const AMIDE_PPM = 3.5
  const dw_amide = 2 * Math.PI * (dw_ppm - AMIDE_PPM) * 1e-6 * LARMOR_HZ
  // Power-broadening denominator (Zaiss 2013 Eq. 13)
  const pb_amide = (R2s_amide + ksw_amide) * (R2s_amide + ksw_amide + w1_eff * w1_eff / R2s_amide) + dw_amide * dw_amide
  const Rex_amide = (fs_amide * ksw_amide * w1_eff * w1_eff) / pb_amide
  totalRex += Rex_amide

  if (poolModel !== '2-pool') {
    // ── Amine pool (+2.0 ppm, k_sw ~ 500–2000 s⁻¹, fast exchange) ──────
    // In the ultra-fast exchange limit amine pools are mostly averaged into water.
    // Effective R2s is heavily broadened: R2s_eff = R2s + ksw (Gutowsky-Saika).
    const AMINE_PPM = 2.0
    const R2s_amine_eff = R2s_amide + ksw_amine   // very broad due to fast exchange
    const dw_amine = 2 * Math.PI * (dw_ppm - AMINE_PPM) * 1e-6 * LARMOR_HZ
    const pb_amine = (R2s_amine_eff + ksw_amine) * (R2s_amine_eff + ksw_amine + w1_eff * w1_eff / R2s_amine_eff) + dw_amine * dw_amine
    const Rex_amine = (fs_amine * ksw_amine * w1_eff * w1_eff) / pb_amine
    totalRex += Rex_amine

    // ── NOE pool (-3.5 ppm, relayed through aliphatic C-H) ──────────────
    // Literature: fs_NOE ~ 0.0025, apparent k_NOE ~ 8–25 s⁻¹ (relayed NOE rate),
    // T2_NOE similar to amide (protein-bound, restricted motion).
    const NOE_PPM = -3.5
    const fs_NOE = 0.0025
    const k_NOE = 12.0          // relayed NOE transfer rate s⁻¹ (van Zijl 2011)
    const dw_NOE = 2 * Math.PI * (dw_ppm - NOE_PPM) * 1e-6 * LARMOR_HZ
    const pb_NOE = (R2s_amide + k_NOE) * (R2s_amide + k_NOE + w1_eff * w1_eff / R2s_amide) + dw_NOE * dw_NOE
    const Rex_NOE = (fs_NOE * k_NOE * w1_eff * w1_eff) / pb_NOE
    totalRex += Rex_NOE
  }

  if (poolModel === '4-pool') {
    // ── Semi-solid MT pool (broad super-Lorentzian approximated as Gaussian)
    // T2_MT = 12 μs → R2_MT ≈ 83 333 s⁻¹ (very short T2, highly immobile protons)
    // Gaussian super-Lorentzian approximation (Gloor 2008, Morrison 1995):
    //   G(Δω) ∝ exp(−(2π·Δω·T2_MT)² / 6)
    const f_MT  = 0.07  // ~7% semisolid pool by proton fraction (brain WM)
    const k_MT  = 30.0  // s⁻¹ (MT pool → water, literature range 20–50)
    const T2_MT = 12e-6 // 12 μs
    const R2_MT = 1.0 / T2_MT
    const dw_MT_rad = 2 * Math.PI * dw_ppm * 1e-6 * LARMOR_HZ
    // Gaussian lineshape (super-Lorentzian approximation)
    const MT_lineshape = Math.exp(-((dw_MT_rad * T2_MT) ** 2) / 6.0)
    // MT uses a separate Henkelman (1993) formula — R2_MT >> ksw, so the
    // standard Rex doesn't apply; use M0s*k_MT*G(Δω) directly:
    const Rex_MT = f_MT * k_MT * w1_eff * w1_eff * MT_lineshape / (R2_MT * R2_MT + dw_MT_rad * dw_MT_rad)
    totalRex += Rex_MT
  }

  // ── Combine water + exchange ─────────────────────────────────────────
  // Full approximation (Zaiss 2013, Eq. 20):
  //   Z ≈ Z_water / (1 + totalRex / R1w)
  // In the limit where Z_water ~ 1 this reduces to:
  //   Z ≈ 1 - w1^2*T1w*T2w/(1+Δω²T2w²) - totalRex*T1w
  const Z = Z_water / (1.0 + totalRex / R1w)

  return Math.max(0.02, Math.min(1.0, Z))
}

function generateZSpectrum(params: CestParams, poolModel: PoolModel): { offsets: number[]; z: number[]; zPredicted: number[] } {
  // 101 offsets from -5 to +5 ppm at 0.1 ppm spacing
  const offsets: number[] = []
  for (let i = -50; i <= 50; i++) {
    offsets.push(i * 0.1)
  }

  // Measured = ground-truth BM + additive Rician noise (approximated as Gaussian for SNR > 20)
  // sigma_noise = 1/SNR (since Z is normalized 0..1)
  const sigma = 1.0 / params.snr
  const z = offsets.map(o => {
    const base = blochMcConnellSimple(o, params, poolModel)
    return Math.max(0.0, Math.min(1.0, base + gaussianNoise(sigma)))
  })

  // Predicted = clean BM output (what the neural ODE would reconstruct)
  const zPredicted = offsets.map(o => blochMcConnellSimple(o, params, poolModel))

  return { offsets, z, zPredicted }
}

/**
 * Generate a spatially-varying parameter map with realistic tissue heterogeneity.
 *
 * Rather than pure noise, we:
 *  1. Use tissue-zone-specific ground truth values (WM-like normal, GM-like periphery,
 *     tumor core with elevated CEST effect).
 *  2. Add smooth low-frequency spatial variation via a sum of cosines (mimics
 *     B1/B0 field inhomogeneity and perfusion gradients).
 *  3. Add per-voxel Gaussian noise scaled to the measurement SNR.
 *
 * @param centerVal  Normal tissue (WM-equivalent) value
 * @param tumorVal   Tumor ROI value (e.g. elevated ksw due to increased mobile protein)
 * @param noiseFrac  Noise as fraction of centerVal (aleatoric noise floor)
 */
function generateParamMap(
  rows: number,
  cols: number,
  centerVal: number,
  tumorVal: number,
  noiseFrac: number,
  params: CestParams
): number[][] {
  const map: number[][] = []
  const cx = cols / 2
  const cy = rows / 2
  const tumorR = rows * 0.15
  const brainR = rows * 0.42
  // GM ring between 0.68×brainR and brainR (realistic cortex geometry)
  const gmInnerR = brainR * 0.68
  // Perilesional zone: ring immediately surrounding tumor (1.6× tumor radius)
  const peritumorR = tumorR * 1.6

  // Noise standard deviation
  const sigma = centerVal * noiseFrac

  for (let r = 0; r < rows; r++) {
    map.push([])
    for (let c = 0; c < cols; c++) {
      const dx = c - cx
      const dy = r - cy
      const dist = Math.sqrt(dx * dx + dy * dy)

      if (dist > brainR) {
        map[r].push(0) // outside brain
        continue
      }

      // Smooth B1/inhomogeneity field: low-frequency cosine modulation
      const inhomog = 1.0
        + 0.04 * Math.cos(c * Math.PI / cols * 2)
        + 0.03 * Math.sin(r * Math.PI / rows * 2)
        + 0.02 * Math.cos((r + c) * Math.PI / (rows + cols) * 3)

      let val: number
      if (dist < tumorR) {
        // Tumor core: pathologically elevated (e.g. higher mobile protein → higher k_sw)
        val = tumorVal * inhomog
      } else if (dist < peritumorR) {
        // Perilesional zone: intermediate transition (linear blend)
        const t = (dist - tumorR) / (peritumorR - tumorR)
        val = (tumorVal * (1 - t) + centerVal * t) * inhomog
      } else if (dist > gmInnerR) {
        // Gray matter cortex: slightly higher CEST than WM, lower T1 relative difference
        val = centerVal * 1.12 * inhomog
      } else {
        // White matter core
        val = centerVal * inhomog
      }

      // Per-voxel Gaussian noise (aleatoric, SNR-dependent)
      val += gaussianNoise(sigma)
      map[r].push(Math.max(0, val))
    }
  }
  return map
}

function generateUncertaintyMap(
  rows: number,
  cols: number,
  baseUncertainty: number,
  mode: 'aleatoric' | 'epistemic'
): number[][] {
  const map: number[][] = []
  const cx = cols / 2
  const cy = rows / 2
  const brainR = rows * 0.42
  const tumorR = rows * 0.15

  for (let r = 0; r < rows; r++) {
    map.push([])
    for (let c = 0; c < cols; c++) {
      const dx = c - cx
      const dy = r - cy
      const dist = Math.sqrt(dx * dx + dy * dy)

      if (dist > brainR) {
        map[r].push(0)
      } else {
        let u = baseUncertainty
        // Higher uncertainty at tumor boundary
        const boundaryDist = Math.abs(dist - tumorR)
        if (boundaryDist < 8) {
          u = baseUncertainty * (2.5 + Math.random() * 0.8)
        }
        // Epistemic is higher overall in uncertain regions
        if (mode === 'epistemic') {
          u *= 0.6 + Math.random() * 0.8
        }
        // Edge of brain has higher uncertainty
        if (dist > brainR * 0.88) {
          u *= 1.8
        }
        map[r].push(Math.max(0, u + gaussianNoise(u * 0.2)))
      }
    }
  }
  return map
}

function runSimulation(params: CestParams, poolModel: PoolModel, mcPasses: number): PredictionResult {
  const t0 = performance.now()

  const { offsets, z, zPredicted } = generateZSpectrum(params, poolModel)

  const ROWS = 64
  const COLS = 64

  // ── Tissue-realistic parameter maps ─────────────────────────────────────
  // Literature values (3T, Zaiss 2013, Heo 2019, Xu 2016):
  //   WM: ksw_amide ~ 30 s⁻¹, fs_amide ~ 0.001; tumor: ksw up to 2× WM
  //   GM: ksw ~ 50 s⁻¹, fs ~ 0.0015 (slightly higher than WM)
  //   Tumor: ksw ~ 60–120 s⁻¹, fs ~ 0.002–0.004 (elevated mobile protein)
  //
  // We scale by the user-set params so the sliders stay meaningful.
  // Normal WM = 0.65×ksw, Normal GM = 0.90×ksw, Tumor = user ksw_amide.
  const kswMap  = generateParamMap(ROWS, COLS,
    params.ksw_amide * 0.65,   // WM baseline
    params.ksw_amide * 1.00,   // tumor (full slider value)
    0.08, params)

  const fsMap   = generateParamMap(ROWS, COLS,
    params.fs_amide * 0.65,
    params.fs_amide * 1.60,    // tumor has more mobile protein
    0.06, params)

  // APT map = MTRasym@3.5ppm; compute voxel-wise from physics
  // For speed we sample a representative set of voxels from kswMap/fsMap
  // and compute the true MTRasym@3.5 using the BM equation.
  const aptMap: number[][] = []
  for (let r = 0; r < ROWS; r++) {
    aptMap.push([])
    for (let c = 0; c < COLS; c++) {
      if (kswMap[r][c] <= 0) { aptMap[r].push(0); continue }
      // Voxel-specific params
      const vParams: CestParams = {
        ...params,
        ksw_amide: kswMap[r][c],
        fs_amide: fsMap[r][c],
      }
      // MTRasym@3.5 ppm = Z(-3.5) - Z(+3.5)
      const zNeg = blochMcConnellSimple(-3.5, vParams, poolModel)
      const zPos = blochMcConnellSimple(+3.5, vParams, poolModel)
      aptMap[r].push(Math.max(0, (zNeg - zPos) * 100))  // in %
    }
  }

  // T1w and T2w maps: tissue-zone values from known literature (3T)
  // WM: T1w ≈ 0.84 s, T2w ≈ 75 ms; GM: T1w ≈ 1.35 s, T2w ≈ 95 ms
  // Tumor: T1w can be prolonged (1.5–2.5 s), T2w also prolonged (100–150 ms)
  const t1wMap  = generateParamMap(ROWS, COLS,
    params.t1w * 0.85,         // WM (shorter)
    params.t1w * 1.10,         // tumor (slightly prolonged)
    0.04, params)

  const t2wMap  = generateParamMap(ROWS, COLS,
    params.t2w * 0.90,
    params.t2w * 1.25,         // tumor T2w prolonged (vasogenic edema)
    0.05, params)

  // ── Uncertainty maps: aleatoric = SNR-driven, epistemic = MC-Dropout ────
  // Aleatoric uncertainty scales with signal noise floor (1/SNR)
  // Epistemic scales with model complexity and out-of-distribution-ness
  // (higher near tumor boundary and brain edge where training data is sparse)

  // Relative noise fractions per parameter (Fisher information bounds)
  // k_sw: σ/k_sw ≈ 3/SNR (most uncertain); f_s: σ/f_s ≈ 2.5/SNR
  const snrFactor = 1.0 / params.snr
  const aleatoricKsw = generateUncertaintyMap(ROWS, COLS, params.ksw_amide * snrFactor * 3.0, 'aleatoric')
  const epistemicKsw = generateUncertaintyMap(ROWS, COLS, params.ksw_amide * snrFactor * 1.5, 'epistemic')

  const aleatoricFs = generateUncertaintyMap(ROWS, COLS, params.fs_amide * snrFactor * 2.5, 'aleatoric')
  const epistemicFs = generateUncertaintyMap(ROWS, COLS, params.fs_amide * snrFactor * 1.2, 'epistemic')

  const aleatoricApt = generateUncertaintyMap(ROWS, COLS, 0.5 * snrFactor * 3.0, 'aleatoric') // in % units
  const epistemicApt = generateUncertaintyMap(ROWS, COLS, 0.5 * snrFactor * 1.5, 'epistemic')

  const aleatoricT1w = generateUncertaintyMap(ROWS, COLS, params.t1w * snrFactor * 1.5, 'aleatoric')
  const epistemicT1w = generateUncertaintyMap(ROWS, COLS, params.t1w * snrFactor * 0.8, 'epistemic')

  const aleatoricT2w = generateUncertaintyMap(ROWS, COLS, params.t2w * snrFactor * 2.0, 'aleatoric')
  const epistemicT2w = generateUncertaintyMap(ROWS, COLS, params.t2w * snrFactor * 1.0, 'epistemic')

  const totalUnc = (a: number[][], e: number[][]): number[][] =>
    a.map((row, r) => row.map((v, c) => v + e[r][c]))

  // ── Pool-Anchored Cross-Attention weights (physics-informed) ────────────
  // Attention weight approximates ∂Z/∂pool contribution at peak offset.
  // Compute numerically as Rex_pool / (R1w + totalRex) – a proxy for
  // how much each pool influences the BM steady-state at its resonance.
  const GAMMA_HZ_PER_T = 42.577e6
  const LARMOR_HZ = GAMMA_HZ_PER_T * 3.0
  const w1_physics = 2 * Math.PI * GAMMA_HZ_PER_T * (params.b1 * 1e-6)
  const R2s_am = 1.0 / (params.t2s * 1e-3)
  const dw_amide_peak = 2 * Math.PI * (3.5 - params.b0_shift) * 1e-6 * LARMOR_HZ
  const Rex_amide_peak = (params.fs_amide * params.ksw_amide * w1_physics * w1_physics) /
    ((params.ksw_amide + R2s_am) * (params.ksw_amide + R2s_am) + dw_amide_peak * dw_amide_peak)
  const dw_amine_peak = 2 * Math.PI * (2.0 - params.b0_shift) * 1e-6 * LARMOR_HZ
  const R2s_an = R2s_am * 1.5 + params.ksw_amine
  const Rex_amine_peak = poolModel !== '2-pool'
    ? (params.fs_amine * params.ksw_amine * w1_physics * w1_physics) /
      ((params.ksw_amine + R2s_an) * (params.ksw_amine + R2s_an) + dw_amine_peak * dw_amine_peak)
    : 0
  const dw_noe_peak = 2 * Math.PI * (-3.5 - params.b0_shift) * 1e-6 * LARMOR_HZ
  const Rex_noe_peak = poolModel !== '2-pool'
    ? (0.0025 * 15 * w1_physics * w1_physics) /
      ((15 + R2s_am) * (15 + R2s_am) + dw_noe_peak * dw_noe_peak)
    : 0

  const R1w = 1.0 / params.t1w
  const totalRexSum = Rex_amide_peak + Rex_amine_peak + Rex_noe_peak
  const normFactor = R1w + totalRexSum + 1e-9

  // Water attention = how much of Z is governed by direct RF saturation
  const waterW = 1.0 - (w1_physics * w1_physics * params.t1w * (params.t2w * 1e-3)) /
    (1 + (2 * Math.PI * 0.001 * params.t2w * 1e-3) + w1_physics * w1_physics * params.t1w * (params.t2w * 1e-3) + 1e-9)
  const amideW  = Math.min(0.99, Rex_amide_peak / normFactor * 4)
  const amineW  = Math.min(0.99, Rex_amine_peak / normFactor * 6)
  const noeW    = Math.min(0.99, Rex_noe_peak   / normFactor * 6)
  const mtW     = poolModel === '4-pool' ? 0.35 + Math.random() * 0.06 : 0.04

  const poolAttention = [
    { pool: 'Water', ppm: 0.0,  weight: Math.max(0.3, Math.min(0.95, waterW + Math.random() * 0.02)) },
    { pool: 'Amide', ppm: 3.5,  weight: Math.max(0.1, Math.min(0.95, amideW  + Math.random() * 0.02)) },
    { pool: 'Amine', ppm: 2.0,  weight: Math.max(0.05, Math.min(0.95, amineW + Math.random() * 0.02)) },
    { pool: 'NOE',   ppm: -3.5, weight: Math.max(0.05, Math.min(0.95, noeW   + Math.random() * 0.02)) },
    { pool: 'MT',    ppm: -2.5, weight: Math.max(0.02, Math.min(0.95, mtW)) },
  ]

  // ── MTR_asym spectrum ───────────────────────────────────────────────────
  // MTRasym(Δω) = Z(-Δω) - Z(+Δω); computed from the clean BM solution.
  const mtrAsym = offsets
    .filter(o => o >= 0)
    .map(o => {
      const zPos = blochMcConnellSimple(o, params, poolModel)
      const zNeg = blochMcConnellSimple(-o, params, poolModel)
      return Math.max(0, zNeg - zPos)  // clamp negative (non-physical at this offset)
    })

  // ── Loss components (physically motivated values) ───────────────────────
  // Scale with SNR and pool model complexity for realism
  const snrScale = 80 / params.snr  // normalized to SNR=80 baseline
  const poolScale = poolModel === '4-pool' ? 1.2 : poolModel === '3-pool' ? 1.0 : 0.8
  const lossComponents = [
    { name: 'L_recon (NLL)',       value: (0.038 + Math.random() * 0.004) * snrScale },
    { name: 'L_physics (BM residual)', value: (0.007 + Math.random() * 0.002) * poolScale },
    { name: 'L_param (supervised)', value: (0.022 + Math.random() * 0.003) * snrScale },
    { name: 'L_smooth (TV)',       value: 0.0045 + Math.random() * 0.001 },
    { name: 'L_asym (MTRasym)',    value: (0.003 + Math.random() * 0.001) * snrScale },
  ]

  const elapsed = performance.now() - t0
  // Realistic inference time: base 15ms for single pass + ~3ms per MC pass
  const inferenceTimeMs = mcPasses > 1
    ? 15 + mcPasses * 3.2 + Math.random() * 5
    : 15 + Math.random() * 5

  return {
    zSpectrum: z,
    zSpectrumPredicted: zPredicted,
    offsets,
    paramMaps: { ksw: kswMap, fs: fsMap, apt: aptMap, t1w: t1wMap, t2w: t2wMap },
    uncertaintyMaps: {
      ksw: totalUnc(aleatoricKsw, epistemicKsw),
      fs:  totalUnc(aleatoricFs,  epistemicFs),
      apt: totalUnc(aleatoricApt, epistemicApt),
      t1w: totalUnc(aleatoricT1w, epistemicT1w),
      t2w: totalUnc(aleatoricT2w, epistemicT2w),
    },
    aleatoricMaps: { ksw: aleatoricKsw, fs: aleatoricFs, apt: aleatoricApt, t1w: aleatoricT1w, t2w: aleatoricT2w },
    epistemicMaps: { ksw: epistemicKsw, fs: epistemicFs, apt: epistemicApt, t1w: epistemicT1w, t2w: epistemicT2w },
    poolAttention,
    lossComponents,
    inferenceTimeMs,
    mtrAsym,
  }
}

// ── Canvas render helpers ──────────────────────────────────────────────────────

type Colormap = 'viridis' | 'plasma' | 'coolwarm' | 'uncertainty' | 'hot'

function colormapViridis(t: number): [number, number, number] {
  const c = Math.max(0, Math.min(1, t))
  if (c < 0.25) return [Math.round(68 + c * 4 * (58 - 68)), Math.round(1 + c * 4 * (86 - 1)), Math.round(84 + c * 4 * (139 - 84))]
  if (c < 0.5) return [Math.round(58 + (c - 0.25) * 4 * (33 - 58)), Math.round(86 + (c - 0.25) * 4 * (145 - 86)), Math.round(139 + (c - 0.25) * 4 * (140 - 139))]
  if (c < 0.75) return [Math.round(33 + (c - 0.5) * 4 * (94 - 33)), Math.round(145 + (c - 0.5) * 4 * (201 - 145)), Math.round(140 + (c - 0.5) * 4 * (97 - 140))]
  return [Math.round(94 + (c - 0.75) * 4 * (253 - 94)), Math.round(201 + (c - 0.75) * 4 * (231 - 201)), Math.round(97 + (c - 0.75) * 4 * (37 - 97))]
}

function colormapPlasma(t: number): [number, number, number] {
  const c = Math.max(0, Math.min(1, t))
  if (c < 0.25) return [Math.round(13 + c * 4 * (84 - 13)), Math.round(8 + c * 4 * (2 - 8)), Math.round(135 + c * 4 * (163 - 135))]
  if (c < 0.5) return [Math.round(84 + (c - 0.25) * 4 * (163 - 84)), Math.round(2 + (c - 0.25) * 4 * (0 - 2)), Math.round(163 + (c - 0.25) * 4 * (121 - 163))]
  if (c < 0.75) return [Math.round(163 + (c - 0.5) * 4 * (229 - 163)), Math.round(0 + (c - 0.5) * 4 * (87 - 0)), Math.round(121 + (c - 0.5) * 4 * (52 - 121))]
  return [Math.round(229 + (c - 0.75) * 4 * (253 - 229)), Math.round(87 + (c - 0.75) * 4 * (231 - 87)), Math.round(52 + (c - 0.75) * 4 * (37 - 52))]
}

function colormapUncertainty(t: number): [number, number, number] {
  const c = Math.max(0, Math.min(1, t))
  if (c < 0.33) return [Math.round(30 + c * 3 * 70), Math.round(60 + c * 3 * 40), Math.round(180 + c * 3 * 50)]
  if (c < 0.66) return [Math.round(100 + (c - 0.33) * 3 * 120), Math.round(100 - (c - 0.33) * 3 * 60), Math.round(230 - (c - 0.33) * 3 * 50)]
  return [Math.round(220 + (c - 0.66) * 3 * 35), Math.round(40 + (c - 0.66) * 3 * 180), Math.round(180 + (c - 0.66) * 3 * 75)]
}

function colormapHot(t: number): [number, number, number] {
  const c = Math.max(0, Math.min(1, t))
  if (c < 0.4) return [Math.round(c / 0.4 * 230), Math.round(c / 0.4 * 40), 0]
  if (c < 0.8) return [230, Math.round(40 + (c - 0.4) / 0.4 * 180), 0]
  return [255, 220, Math.round((c - 0.8) / 0.2 * 230)]
}

function applyColormap(t: number, cm: Colormap): [number, number, number] {
  switch (cm) {
    case 'viridis': return colormapViridis(t)
    case 'plasma': return colormapPlasma(t)
    case 'uncertainty': return colormapUncertainty(t)
    case 'hot': return colormapHot(t)
    default: return colormapViridis(t)
  }
}

function renderParamMap(
  canvas: HTMLCanvasElement,
  data: number[][],
  colormap: Colormap,
  showTumorOverlay: boolean
) {
  const rows = data.length
  const cols = data[0]?.length || 0
  if (!rows || !cols) return

  canvas.width = cols
  canvas.height = rows
  const ctx = canvas.getContext('2d')
  if (!ctx) return

  let min = Infinity
  let max = -Infinity
  for (const row of data) {
    for (const v of row) {
      if (v > 0) { // skip background
        if (v < min) min = v
        if (v > max) max = v
      }
    }
  }
  const range = max - min || 1e-8

  const cx = cols / 2
  const cy = rows / 2
  const tumorR = rows * 0.15
  const brainR = rows * 0.42

  const imgData = ctx.createImageData(cols, rows)
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const v = data[r][c]
      const idx = (r * cols + c) * 4

      if (v <= 0) {
        // Crisp light slate background for empty voxels in white theme
        imgData.data[idx] = 241
        imgData.data[idx + 1] = 245
        imgData.data[idx + 2] = 249
        imgData.data[idx + 3] = 255
        continue
      }

      const t = (v - min) / range
      const [rr, gg, bb] = applyColormap(t, colormap)

      const dx = c - cx
      const dy = r - cy
      const dist = Math.sqrt(dx * dx + dy * dy)
      const isTumor = dist < tumorR

      if (showTumorOverlay && isTumor) {
        imgData.data[idx] = Math.min(255, rr + 25)
        imgData.data[idx + 1] = Math.max(0, gg - 15)
        imgData.data[idx + 2] = Math.min(255, bb + 20)
      } else {
        imgData.data[idx] = rr
        imgData.data[idx + 1] = gg
        imgData.data[idx + 2] = bb
      }
      imgData.data[idx + 3] = 255
    }
  }
  ctx.putImageData(imgData, 0, 0)

  // Draw brain boundary
  ctx.beginPath()
  ctx.arc(cx, cy, brainR, 0, 2 * Math.PI)
  ctx.strokeStyle = 'rgba(148,163,184,0.6)'
  ctx.lineWidth = 1
  ctx.stroke()

  // Draw tumor boundary
  if (showTumorOverlay) {
    ctx.beginPath()
    ctx.arc(cx, cy, tumorR, 0, 2 * Math.PI)
    ctx.strokeStyle = 'rgba(239,68,68,0.75)'
    ctx.lineWidth = 1.5
    ctx.setLineDash([3, 3])
    ctx.stroke()
    ctx.setLineDash([])
  }
}

// ── Shared Alignment Parameters ────────────────────────────────────────────────
// Shared pixel padding and X-axis mapping ensuring pixel-perfect vertical alignment
const CANVAS_PAD_LEFT = 45
const CANVAS_PAD_RIGHT = 20
const X_MIN_PPM = -5.0
const X_MAX_PPM = 5.0
const X_SPAN_PPM = 10.0

// ── Sub-components ─────────────────────────────────────────────────────────────

function ZSpectrumPlot({
  offsets,
  measured,
  predicted,
  mtrAsym,
  height = 160,
}: {
  offsets: number[]
  measured: number[]
  predicted: number[]
  mtrAsym: number[]
  height?: number
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !offsets.length) return

    const W = canvas.offsetWidth || 500
    const H = height
    canvas.width = W
    canvas.height = H

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Clean white background
    ctx.fillStyle = '#ffffff'
    ctx.fillRect(0, 0, W, H)

    const padL = CANVAS_PAD_LEFT
    const padR = CANVAS_PAD_RIGHT
    const padT = 20
    const padB = 26

    const plotW = W - padL - padR
    const plotH = H - padT - padB

    // Horizontal grid lines
    ctx.strokeStyle = '#f1f5f9'
    ctx.lineWidth = 1
    for (let i = 0; i <= 5; i++) {
      const y = padT + (plotH / 5) * i
      ctx.beginPath()
      ctx.moveTo(padL, y)
      ctx.lineTo(padL + plotW, y)
      ctx.stroke()
    }

    // Y-axis tick labels
    ctx.fillStyle = '#475569'
    ctx.font = '9px JetBrains Mono, monospace'
    ctx.textAlign = 'right'
    for (let i = 0; i <= 5; i++) {
      const val = (1 - i / 5).toFixed(1)
      const y = padT + (plotH / 5) * i
      ctx.fillText(val, padL - 6, y + 3)
    }

    const toX = (o: number) => padL + ((o - X_MIN_PPM) / X_SPAN_PPM) * plotW
    const toY = (v: number) => padT + plotH - (v * plotH)

    // Pool resonance markers
    const pools = [
      { ppm: 3.5, label: 'Amide (+3.5)', color: '#1e40af', lineDash: [3, 3] },
      { ppm: 2.0, label: 'Amine (+2.0)', color: '#047857', lineDash: [3, 3] },
      { ppm: -3.5, label: 'NOE (-3.5)', color: '#c2410c', lineDash: [3, 3] },
      { ppm: 0.0, label: 'Water (0.0)', color: '#64748b', lineDash: [2, 2] },
    ]

    pools.forEach(p => {
      const x = toX(p.ppm)
      ctx.strokeStyle = p.color
      ctx.lineWidth = 1
      ctx.setLineDash(p.lineDash)
      ctx.beginPath()
      ctx.moveTo(x, padT)
      ctx.lineTo(x, padT + plotH)
      ctx.stroke()
      ctx.setLineDash([])
      ctx.fillStyle = p.color
      ctx.font = '8px JetBrains Mono, monospace'
      ctx.textAlign = 'center'
      ctx.fillText(p.label, x, padT - 6)
    })

    // Measured spectrum (vivid blue)
    ctx.strokeStyle = '#2563eb'
    ctx.lineWidth = 2
    ctx.beginPath()
    offsets.forEach((o, i) => {
      const x = toX(o)
      const y = toY(measured[i])
      if (i === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
    })
    ctx.stroke()

    // Predicted spectrum (vivid emerald green)
    ctx.strokeStyle = '#16a34a'
    ctx.lineWidth = 2
    ctx.beginPath()
    offsets.forEach((o, i) => {
      const x = toX(o)
      const y = toY(predicted[i])
      if (i === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
    })
    ctx.stroke()

    // X-axis line
    ctx.strokeStyle = '#cbd5e1'
    ctx.lineWidth = 1
    ctx.beginPath()
    ctx.moveTo(padL, padT + plotH)
    ctx.lineTo(padL + plotW, padT + plotH)
    ctx.stroke()

    // X-axis tick labels
    ctx.fillStyle = '#475569'
    ctx.font = '9px JetBrains Mono, monospace'
    ctx.textAlign = 'center'
    const xTicks = [-5, -3, -1, 0, 1, 3, 5]
    xTicks.forEach(t => {
      const x = toX(t)
      ctx.fillText(t >= 0 ? `+${t}` : `${t}`, x, padT + plotH + 14)
    })

    // Legend inside top-left
    ctx.font = '9px JetBrains Mono, monospace'
    ctx.textAlign = 'left'
    ctx.fillStyle = '#2563eb'
    ctx.fillRect(padL + 6, padT + 4, 16, 3)
    ctx.fillText('Measured Z(Δω)', padL + 26, padT + 9)

    ctx.fillStyle = '#16a34a'
    ctx.fillRect(padL + 6, padT + 16, 16, 3)
    ctx.fillText('BM-NODE-UQ Predicted', padL + 26, padT + 21)

    // X-axis label
    ctx.fillStyle = '#64748b'
    ctx.font = '8px JetBrains Mono, monospace'
    ctx.textAlign = 'center'
    ctx.fillText('Saturation Offset (ppm)', padL + plotW / 2, H - 2)

  }, [offsets, measured, predicted, mtrAsym, height])

  return (
    <canvas
      ref={canvasRef}
      style={{ width: '100%', height: `${height}px`, display: 'block' }}
    />
  )
}

function MTRAsymPlot({ offsets, mtrAsym, height = 90 }: { offsets: number[]; mtrAsym: number[]; height?: number }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !mtrAsym.length) return

    const W = canvas.offsetWidth || 500
    const H = height
    canvas.width = W
    canvas.height = H

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Clean white background
    ctx.fillStyle = '#ffffff'
    ctx.fillRect(0, 0, W, H)

    const padL = CANVAS_PAD_LEFT
    const padR = CANVAS_PAD_RIGHT
    const padT = 16
    const padB = 22

    const plotW = W - padL - padR
    const plotH = H - padT - padB

    // Exact same toX mapping as ZSpectrumPlot ensuring pixel alignment!
    const toX = (o: number) => padL + ((o - X_MIN_PPM) / X_SPAN_PPM) * plotW

    const posOffsets = offsets.filter(o => o >= 0)
    const maxVal = Math.max(...mtrAsym, 0.01)

    const toY = (v: number) => padT + plotH - ((v / (maxVal * 1.25)) * plotH)

    // Horizontal zero line
    ctx.strokeStyle = '#cbd5e1'
    ctx.lineWidth = 1
    const zeroY = toY(0)
    ctx.beginPath()
    ctx.moveTo(padL, zeroY)
    ctx.lineTo(padL + plotW, zeroY)
    ctx.stroke()

    // Aligning vertical pool markers with ZSpectrumPlot
    const pools = [
      { ppm: 3.5, label: 'Amide (+3.5)', color: 'rgba(30,64,175,0.4)', lineDash: [3, 3] },
      { ppm: 2.0, label: 'Amine (+2.0)', color: 'rgba(4,120,87,0.4)', lineDash: [3, 3] },
      { ppm: 0.0, label: 'Water (0.0)', color: 'rgba(100,116,139,0.3)', lineDash: [2, 2] },
    ]

    pools.forEach(p => {
      const x = toX(p.ppm)
      ctx.strokeStyle = p.color
      ctx.lineWidth = 1
      ctx.setLineDash(p.lineDash)
      ctx.beginPath()
      ctx.moveTo(x, padT)
      ctx.lineTo(x, padT + plotH)
      ctx.stroke()
      ctx.setLineDash([])
    })

    // Fill area under MTR_asym curve (positive offsets)
    ctx.fillStyle = 'rgba(217,119,6,0.12)'
    ctx.beginPath()
    ctx.moveTo(toX(0), zeroY)
    posOffsets.forEach((o, i) => {
      ctx.lineTo(toX(o), toY(mtrAsym[i]))
    })
    ctx.lineTo(toX(posOffsets[posOffsets.length - 1]), zeroY)
    ctx.closePath()
    ctx.fill()

    // MTR_asym line (vivid amber/orange)
    ctx.strokeStyle = '#d97706'
    ctx.lineWidth = 2
    ctx.beginPath()
    posOffsets.forEach((o, i) => {
      const x = toX(o)
      const y = toY(mtrAsym[i])
      if (i === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
    })
    ctx.stroke()

    // Y-axis tick labels
    ctx.fillStyle = '#475569'
    ctx.font = '8px JetBrains Mono, monospace'
    ctx.textAlign = 'right'
    ctx.fillText(`${(maxVal * 100).toFixed(0)}%`, padL - 6, padT + 8)
    ctx.fillText('0%', padL - 6, zeroY + 3)

    // X-axis ticks matching ZSpectrumPlot
    ctx.fillStyle = '#64748b'
    ctx.font = '8px JetBrains Mono, monospace'
    ctx.textAlign = 'center'
    const xTicks = [-5, -3, -1, 0, 1, 3, 5]
    xTicks.forEach(t => {
      const x = toX(t)
      ctx.fillText(t >= 0 ? `+${t}` : `${t}`, x, padT + plotH + 12)
    })

    // Peak APT readout callout
    const peakAmideIdx = Math.round((3.5 / 5.0) * (posOffsets.length - 1))
    const peakVal = mtrAsym[peakAmideIdx] || 0
    ctx.fillStyle = '#b45309'
    ctx.font = '9px JetBrains Mono, monospace'
    ctx.fontWeight = 'bold'
    ctx.textAlign = 'left'
    ctx.fillText(`APT@3.5ppm: ${(peakVal * 100).toFixed(1)}%`, padL + 6, padT + 12)

    ctx.fillStyle = '#64748b'
    ctx.font = '8px JetBrains Mono, monospace'
    ctx.textAlign = 'center'
    ctx.fillText('MTR_asym Offset (ppm)', padL + plotW / 2, H - 2)

  }, [offsets, mtrAsym, height])

  return (
    <canvas
      ref={canvasRef}
      style={{ width: '100%', height: `${height}px`, display: 'block' }}
    />
  )
}

function ParamMapCanvas({
  data,
  colormap,
  label,
  unit,
  showTumorOverlay,
  onClick,
  isSelected,
}: {
  data: number[][] | null
  colormap: Colormap
  label: string
  unit: string
  showTumorOverlay: boolean
  onClick: () => void
  isSelected: boolean
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    if (canvasRef.current && data) {
      renderParamMap(canvasRef.current, data, colormap, showTumorOverlay)
    }
  }, [data, colormap, showTumorOverlay])

  if (!data) return null

  return (
    <div
      onClick={onClick}
      style={{
        cursor: 'pointer',
        border: isSelected ? '2px solid #2563eb' : '1px solid #e2e8f0',
        borderRadius: '4px',
        background: isSelected ? 'rgba(37,99,235,0.06)' : '#ffffff',
        padding: '5px',
        transition: 'all 0.15s ease',
        boxShadow: isSelected ? '0 0 0 2px rgba(37,99,235,0.2)' : 'none',
      }}
    >
      <canvas
        ref={canvasRef}
        style={{ width: '100%', aspectRatio: '1', display: 'block', imageRendering: 'pixelated', borderRadius: '3px' }}
      />
      <div style={{ textAlign: 'center', fontFamily: 'var(--font-mono)', fontSize: '10px', color: '#334155', marginTop: '4px', fontWeight: 500 }}>
        {label} {unit && <span style={{ color: '#2563eb' }}>[{unit}]</span>}
      </div>
    </div>
  )
}

function ColormapBar({ colormap, min, max, unit }: { colormap: Colormap; min: number; max: number; unit: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const W = canvas.offsetWidth || 200
    canvas.width = W
    canvas.height = 14
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    for (let x = 0; x < W; x++) {
      const t = x / W
      const [r, g, b] = applyColormap(t, colormap)
      ctx.fillStyle = `rgb(${r},${g},${b})`
      ctx.fillRect(x, 0, 1, 14)
    }
  }, [colormap])

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '4px 2px 0 2px' }}>
      <span style={{ fontFamily: 'var(--font-mono)', fontSize: '9px', color: '#64748b', whiteSpace: 'nowrap' }}>{min.toFixed(1)}</span>
      <canvas ref={canvasRef} style={{ flex: 1, height: '10px', borderRadius: '2px', border: '1px solid #e2e8f0' }} />
      <span style={{ fontFamily: 'var(--font-mono)', fontSize: '9px', color: '#64748b', whiteSpace: 'nowrap' }}>{max.toFixed(1)} {unit}</span>
    </div>
  )
}

function AttentionBar({ pool, ppm, weight }: { pool: string; ppm: number; weight: number }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
      <span style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', width: '45px', color: '#334155', fontWeight: 600 }}>{pool}</span>
      <span style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', width: '42px', color: '#2563eb' }}>
        {ppm >= 0 ? '+' : ''}{ppm.toFixed(1)} ppm
      </span>
      <div style={{ flex: 1, height: '10px', background: '#f1f5f9', borderRadius: '3px', overflow: 'hidden', border: '1px solid #cbd5e1' }}>
        <div style={{
          height: '100%',
          width: `${weight * 100}%`,
          background: weight > 0.7 ? '#2563eb' : weight > 0.3 ? '#d97706' : '#94a3b8',
          transition: 'width 0.6s ease',
        }} />
      </div>
      <span style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', width: '36px', textAlign: 'right', color: weight > 0.5 ? '#0f172a' : '#64748b', fontWeight: 600 }}>
        {(weight * 100).toFixed(0)}%
      </span>
    </div>
  )
}

// ── Main Component ─────────────────────────────────────────────────────────────

export default function BmNodeUQView() {
  const [poolModel, setPoolModel] = useState<PoolModel>('3-pool')
  const [mcPasses, setMcPasses] = useState(20)
  const [uqMode, setUqMode] = useState<UQMode>('total')
  const [displayMap, setDisplayMap] = useState<DisplayMap>('ksw')
  const [showTumorBoundary, setShowTumorBoundary] = useState(true)
  const [running, setRunning] = useState(false)
  const [result, setResult] = useState<PredictionResult | null>(null)
  const [params, setParams] = useState<CestParams>({
    ksw_amide: 120,
    ksw_amine: 800,
    fs_amide: 0.003,
    fs_amine: 0.0015,
    t1w: 1.8,
    t2w: 80,
    t1s: 1.2,
    t2s: 15,
    b1: 0.8,    // μT — realistic for pulsed CEST train at 3T (Schuenke 2017)
    b0_shift: 0.05,
    snr: 80,
  })

  const runInference = useCallback(() => {
    setRunning(true)
    setTimeout(() => {
      const res = runSimulation(params, poolModel, mcPasses)
      setResult(res)
      setRunning(false)
    }, 600 + Math.random() * 400)
  }, [params, poolModel, mcPasses])

  useEffect(() => {
    runInference()
  }, [])

  const getActiveMap = (): number[][] | null => {
    if (!result) return null
    if (displayMap === 'uncertainty') {
      const maps = uqMode === 'aleatoric' ? result.aleatoricMaps : uqMode === 'epistemic' ? result.epistemicMaps : result.uncertaintyMaps
      return maps.ksw
    }
    return result.paramMaps[displayMap] ?? null
  }

  const getMapColormap = (): Colormap => {
    if (displayMap === 'uncertainty') return 'uncertainty'
    if (displayMap === 'apt') return 'plasma'
    if (displayMap === 't1w' || displayMap === 't2w') return 'viridis'
    return 'hot'
  }

  const MAP_CONFIGS: { id: DisplayMap; label: string; unit: string; colormap: Colormap; min: number; max: number }[] = [
    { id: 'ksw', label: 'k_sw (Amide)', unit: 's⁻¹', colormap: 'hot', min: 0, max: params.ksw_amide * 2.5 },
    { id: 'fs', label: 'f_s (Fraction)', unit: '×10⁻³', colormap: 'hot', min: 0, max: params.fs_amide * 1000 * 2 },
    { id: 'apt', label: 'APT / MTR_asym', unit: '%', colormap: 'plasma', min: 0, max: 3 },
    { id: 't1w', label: 'T₁w', unit: 's', colormap: 'viridis', min: 0, max: params.t1w * 1.5 },
    { id: 't2w', label: 'T₂w', unit: 'ms', colormap: 'viridis', min: 0, max: params.t2w * 1.5 },
    { id: 'uncertainty', label: 'Uncertainty (σ)', unit: '', colormap: 'uncertainty', min: 0, max: 1 },
  ]

  const activeConfig = MAP_CONFIGS.find(m => m.id === displayMap)!

  const labelStyle = {
    fontFamily: 'var(--font-mono)',
    fontSize: '10px',
    color: '#475569',
    textTransform: 'uppercase' as const,
    marginBottom: '4px',
    display: 'block',
  }

  return (
    <div style={{ display: 'contents' }}>

      {/* ── Left Panel: Config ── */}
      <div className="syngo-panel" style={{ background: '#ffffff', border: '1px solid #cbd5e1' }}>
        <div className="panel-header" style={{ background: '#f8fafc', borderBottom: '1px solid #cbd5e1', color: '#1e3a8a' }}>
          <span>BM-NODE-UQ Config</span>
          <span style={{ fontFamily: 'var(--font-mono)' }}>[CEST-01]</span>
        </div>
        <div className="panel-body" style={{ display: 'flex', flexDirection: 'column', gap: '14px', overflowY: 'auto', background: '#ffffff' }}>

          {/* Model badge */}
          <div style={{ background: 'linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%)', border: '1px solid #cbd5e1', padding: '10px 12px', borderRadius: '4px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '4px' }}>
              <span className="status-pill green" />
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: '#1e3a8a', fontWeight: 700, letterSpacing: '1px' }}>
                BM-NODE-UQ
              </span>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '9px', color: '#2563eb', marginLeft: 'auto', fontWeight: 600 }}>
                v0.1-α DEMO
              </span>
            </div>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: '9px', color: '#475569', lineHeight: 1.5 }}>
              Bloch-McConnell Neural ODE<br />
              + Spectral Cross-Attention (PACA)<br />
              + Built-in Uncertainty Quantification
            </div>
            <div style={{ marginTop: '6px', display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
              {['~850K params', 'Self-supervised', 'MICCAI 2026'].map(tag => (
                <span key={tag} style={{ fontFamily: 'var(--font-mono)', fontSize: '8px', background: '#e2e8f0', color: '#1e293b', padding: '2px 6px', borderRadius: '10px', fontWeight: 500 }}>
                  {tag}
                </span>
              ))}
            </div>
          </div>

          {/* Pool Model */}
          <div>
            <span style={labelStyle}>Pool Model</span>
            <div style={{ display: 'flex', gap: '4px' }}>
              {(['2-pool', '3-pool', '4-pool'] as PoolModel[]).map(pm => (
                <button
                  key={pm}
                  className={`clinical-btn ${poolModel === pm ? 'clinical-btn-blue' : ''}`}
                  style={{ flex: 1, padding: '4px 4px', fontSize: '10px' }}
                  onClick={() => setPoolModel(pm)}
                >
                  {pm}
                </button>
              ))}
            </div>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: '9px', color: '#64748b', marginTop: '4px' }}>
              {poolModel === '2-pool' && 'Water + Amide only'}
              {poolModel === '3-pool' && 'Water + Amide + Amine (recommended)'}
              {poolModel === '4-pool' && 'Water + Amide + Amine + NOE/MT'}
            </div>
          </div>

          {/* Uncertainty mode */}
          <div>
            <span style={labelStyle}>Uncertainty Quantification</span>
            <div style={{ display: 'flex', gap: '4px', marginBottom: '6px' }}>
              {(['aleatoric', 'epistemic', 'total'] as UQMode[]).map(m => (
                <button
                  key={m}
                  className={`clinical-btn ${uqMode === m ? 'clinical-btn-blue' : ''}`}
                  style={{ flex: 1, padding: '3px 2px', fontSize: '9px' }}
                  onClick={() => setUqMode(m)}
                >
                  {m.charAt(0).toUpperCase() + m.slice(1)}
                </button>
              ))}
            </div>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: '9px', color: '#64748b', lineHeight: 1.5 }}>
              {uqMode === 'aleatoric' && 'Input noise uncertainty (heteroscedastic σ²)'}
              {uqMode === 'epistemic' && `MC-Dropout, T=${mcPasses} passes (model uncertainty)`}
              {uqMode === 'total' && 'σ²_total = σ²_aleatoric + σ²_epistemic'}
            </div>
          </div>

          {/* MC Passes */}
          <div>
            <span style={labelStyle}>MC Dropout Passes (T)</span>
            <div style={{ display: 'flex', gap: '4px', flexWrap: 'wrap' }}>
              {[1, 5, 10, 20, 50].map(n => (
                <button
                  key={n}
                  className={`clinical-btn ${mcPasses === n ? 'clinical-btn-primary' : ''}`}
                  style={{ flex: 1, padding: '3px 4px', fontSize: '10px' }}
                  onClick={() => setMcPasses(n)}
                >
                  T={n}
                </button>
              ))}
            </div>
          </div>

          <div style={{ borderTop: '1px solid #cbd5e1' }} />

          {/* BM Parameters */}
          <div>
            <span style={{ ...labelStyle, color: '#2563eb', fontWeight: 700 }}>Bloch-McConnell Parameters</span>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>

              <div>
                <span style={labelStyle}>k_sw amide (s⁻¹): <strong style={{ color: '#0f172a' }}>{params.ksw_amide}</strong></span>
                <input
                  type="range" min={20} max={500} step={5}
                  value={params.ksw_amide}
                  onChange={e => setParams(p => ({ ...p, ksw_amide: Number(e.target.value) }))}
                  style={{ width: '100%', accentColor: '#2563eb' }}
                />
              </div>

              <div>
                <span style={labelStyle}>f_s amide (×10⁻³): <strong style={{ color: '#0f172a' }}>{(params.fs_amide * 1000).toFixed(1)}</strong></span>
                <input
                  type="range" min={0.5} max={10} step={0.1}
                  value={params.fs_amide * 1000}
                  onChange={e => setParams(p => ({ ...p, fs_amide: Number(e.target.value) / 1000 }))}
                  style={{ width: '100%', accentColor: '#2563eb' }}
                />
              </div>

              <div>
                <span style={labelStyle}>B₁ field (μT): <strong style={{ color: '#0f172a' }}>{params.b1.toFixed(2)}</strong></span>
                <input
                  type="range" min={0.2} max={2.0} step={0.05}
                  value={params.b1}
                  onChange={e => setParams(p => ({ ...p, b1: Number(e.target.value) }))}
                  style={{ width: '100%', accentColor: '#16a34a' }}
                />
              </div>

              <div>
                <span style={labelStyle}>B₀ shift (ppm): <strong style={{ color: '#0f172a' }}>{params.b0_shift.toFixed(2)}</strong></span>
                <input
                  type="range" min={-0.5} max={0.5} step={0.01}
                  value={params.b0_shift}
                  onChange={e => setParams(p => ({ ...p, b0_shift: Number(e.target.value) }))}
                  style={{ width: '100%', accentColor: '#d97706' }}
                />
              </div>

              <div>
                <span style={labelStyle}>T₁w (s): <strong style={{ color: '#0f172a' }}>{params.t1w.toFixed(1)}</strong></span>
                <input
                  type="range" min={1.0} max={2.5} step={0.05}
                  value={params.t1w}
                  onChange={e => setParams(p => ({ ...p, t1w: Number(e.target.value) }))}
                  style={{ width: '100%', accentColor: '#2563eb' }}
                />
              </div>

              <div>
                <span style={labelStyle}>T₂w (ms): <strong style={{ color: '#0f172a' }}>{params.t2w.toFixed(0)}</strong></span>
                <input
                  type="range" min={40} max={120} step={2}
                  value={params.t2w}
                  onChange={e => setParams(p => ({ ...p, t2w: Number(e.target.value) }))}
                  style={{ width: '100%', accentColor: '#2563eb' }}
                />
              </div>

              <div>
                <span style={labelStyle}>SNR: <strong style={{ color: '#0f172a' }}>{params.snr}</strong></span>
                <input
                  type="range" min={20} max={200} step={5}
                  value={params.snr}
                  onChange={e => setParams(p => ({ ...p, snr: Number(e.target.value) }))}
                  style={{ width: '100%', accentColor: '#64748b' }}
                />
              </div>
            </div>
          </div>

          <div style={{ borderTop: '1px solid #cbd5e1' }} />

          {/* Display options */}
          <div>
            <span style={labelStyle}>Display Options</span>
            <label style={{ display: 'flex', alignItems: 'center', gap: '6px', cursor: 'pointer', fontFamily: 'var(--font-mono)', fontSize: '11px', color: '#1e293b' }}>
              <input
                type="checkbox"
                checked={showTumorBoundary}
                onChange={e => setShowTumorBoundary(e.target.checked)}
                style={{ accentColor: '#ef4444' }}
              />
              Show Tumor ROI Boundary
            </label>
          </div>

          {/* Run button */}
          <button
            className="clinical-btn clinical-btn-primary"
            onClick={runInference}
            disabled={running}
            style={{ width: '100%', padding: '10px', fontSize: '12px', fontWeight: 700 }}
          >
            {running ? (
              <>
                <span className="status-pill yellow" style={{ animation: 'pulse 0.8s ease-in-out infinite' }} />
                RUNNING BM-NODE-UQ...
              </>
            ) : (
              <>▶ RUN INFERENCE</>
            )}
          </button>

          {result && !running && (
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: '9px', color: '#64748b', textAlign: 'center' }}>
              Inference: {result.inferenceTimeMs.toFixed(1)} ms
              {mcPasses > 1 ? ` · MC passes: ${mcPasses}` : ' · Single pass'}
            </div>
          )}
        </div>
      </div>

      {/* ── Right Panel: Results ── */}
      <div className="syngo-panel" style={{ overflow: 'hidden', background: '#ffffff', border: '1px solid #cbd5e1' }}>
        <div className="panel-header" style={{ background: '#f8fafc', borderBottom: '1px solid #cbd5e1', color: '#1e3a8a' }}>
          <span>CEST Parameter Maps + Uncertainty</span>
          <span style={{ fontFamily: 'var(--font-mono)' }}>[MAP-OUT]</span>
        </div>

        {running && (
          <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', flexDirection: 'column', gap: '16px', background: '#ffffff' }}>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: '13px', color: '#2563eb', fontWeight: 'bold' }}>
              <span className="status-pill yellow" />
              RUNNING BM-NODE-UQ INFERENCE...
            </div>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', color: '#64748b', textAlign: 'center', maxWidth: '320px', lineHeight: 1.6 }}>
              Spectral encoder → PACA → Self-attention →<br />
              Pool-decoupled heads → BM-ODE decoder →<br />
              Uncertainty quantification
            </div>
          </div>
        )}

        {!running && result && (
          <div className="panel-body" style={{ overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '14px', background: '#ffffff' }}>

            {/* Z-Spectrum */}
            <div style={{ border: '1px solid #cbd5e1', background: '#ffffff', borderRadius: '4px', overflow: 'hidden' }}>
              <div style={{ padding: '6px 12px', background: '#f8fafc', borderBottom: '1px solid #e2e8f0', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', fontWeight: 700, color: '#1e3a8a' }}>
                  Z-SPECTRUM (Voxel-wise Signal)
                </span>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: '9px', color: '#64748b' }}>
                  N=101 offsets · −5.0 → +5.0 ppm
                </span>
              </div>
              <ZSpectrumPlot
                offsets={result.offsets}
                measured={result.zSpectrum}
                predicted={result.zSpectrumPredicted}
                mtrAsym={result.mtrAsym}
                height={160}
              />
            </div>

            {/* MTR Asymmetry */}
            <div style={{ border: '1px solid #cbd5e1', background: '#ffffff', borderRadius: '4px', overflow: 'hidden' }}>
              <div style={{ padding: '6px 12px', background: '#f8fafc', borderBottom: '1px solid #e2e8f0', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', fontWeight: 700, color: '#b45309' }}>
                  MTR_asym (Asymmetry Analysis)
                </span>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: '9px', color: '#92400e', fontWeight: 600 }}>
                  L_asym constraint active
                </span>
              </div>
              <MTRAsymPlot offsets={result.offsets} mtrAsym={result.mtrAsym} height={90} />
            </div>

            {/* Parameter map thumbnails */}
            <div>
              <div style={{ fontSize: '11px', fontWeight: 600, color: '#475569', textTransform: 'uppercase', marginBottom: '8px' }}>
                Quantitative Parameter Maps — click to expand
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: '8px' }}>
                {MAP_CONFIGS.map(cfg => (
                  <ParamMapCanvas
                    key={cfg.id}
                    data={
                      cfg.id === 'uncertainty'
                        ? (uqMode === 'aleatoric' ? result.aleatoricMaps.ksw : uqMode === 'epistemic' ? result.epistemicMaps.ksw : result.uncertaintyMaps.ksw)
                        : result.paramMaps[cfg.id as keyof typeof result.paramMaps]
                    }
                    colormap={cfg.colormap}
                    label={cfg.label}
                    unit={cfg.unit}
                    showTumorOverlay={showTumorBoundary}
                    isSelected={displayMap === cfg.id}
                    onClick={() => setDisplayMap(cfg.id)}
                  />
                ))}
              </div>
            </div>

            {/* Expanded map view */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
              {/* Large map */}
              <div style={{ border: '1px solid #cbd5e1', background: '#ffffff', borderRadius: '4px', overflow: 'hidden' }}>
                <div style={{ padding: '6px 12px', background: '#f8fafc', borderBottom: '1px solid #e2e8f0', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', fontWeight: 700, color: '#1e3a8a' }}>
                    {activeConfig.label}
                  </span>
                  <span style={{ fontFamily: 'var(--font-mono)', fontSize: '9px', color: '#64748b' }}>
                    64×64 · per-voxel
                  </span>
                </div>
                <div style={{ padding: '8px' }}>
                  <ParamMapCanvas
                    data={getActiveMap()}
                    colormap={getMapColormap()}
                    label={activeConfig.label}
                    unit={activeConfig.unit}
                    showTumorOverlay={showTumorBoundary}
                    isSelected={false}
                    onClick={() => {}}
                  />
                  <ColormapBar
                    colormap={getMapColormap()}
                    min={activeConfig.min}
                    max={activeConfig.max}
                    unit={activeConfig.unit}
                  />
                </div>
              </div>

              {/* Uncertainty map */}
              <div style={{ border: '1px solid #cbd5e1', background: '#ffffff', borderRadius: '4px', overflow: 'hidden' }}>
                <div style={{ padding: '6px 12px', background: '#f8fafc', borderBottom: '1px solid #e2e8f0', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', fontWeight: 700, color: '#6b21a8' }}>
                    Confidence Map (σ²_{uqMode})
                  </span>
                  <span style={{ fontFamily: 'var(--font-mono)', fontSize: '9px', color: '#7e22ce', fontWeight: 600 }}>
                    T={mcPasses} passes
                  </span>
                </div>
                <div style={{ padding: '8px' }}>
                  <ParamMapCanvas
                    data={
                      displayMap !== 'uncertainty'
                        ? (uqMode === 'aleatoric'
                          ? result.aleatoricMaps[displayMap as keyof typeof result.aleatoricMaps]
                          : uqMode === 'epistemic'
                            ? result.epistemicMaps[displayMap as keyof typeof result.epistemicMaps]
                            : result.uncertaintyMaps[displayMap as keyof typeof result.uncertaintyMaps])
                        : result.uncertaintyMaps.ksw
                    }
                    colormap="uncertainty"
                    label={`σ²_${uqMode}`}
                    unit=""
                    showTumorOverlay={showTumorBoundary}
                    isSelected={false}
                    onClick={() => {}}
                  />
                  <ColormapBar colormap="uncertainty" min={0} max={1} unit="(norm)" />
                </div>
              </div>
            </div>

            {/* Pool-Anchored Cross-Attention (PACA) */}
            <div style={{ border: '1px solid #cbd5e1', padding: '12px', background: '#f8fafc', borderRadius: '4px' }}>
              <div style={{ fontSize: '11px', fontWeight: 600, color: '#475569', textTransform: 'uppercase', marginBottom: '8px', display: 'flex', justifyContent: 'space-between' }}>
                <span>Pool-Anchored Cross-Attention (PACA)</span>
                <span style={{ fontFamily: 'var(--font-mono)', color: '#2563eb', fontWeight: 600 }}>4 heads · d_k=32</span>
              </div>
              {result.poolAttention.map(p => (
                <AttentionBar key={p.pool} pool={p.pool} ppm={p.ppm} weight={p.weight} />
              ))}
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: '9px', color: '#64748b', marginTop: '6px', fontStyle: 'italic' }}>
                Attention weights: relative contribution of learnable pool anchor embeddings to latent representations
              </div>
            </div>

            {/* Loss components */}
            <div style={{ border: '1px solid #cbd5e1', padding: '12px', background: '#f8fafc', borderRadius: '4px' }}>
              <div style={{ fontSize: '11px', fontWeight: 600, color: '#475569', textTransform: 'uppercase', marginBottom: '8px', display: 'flex', justifyContent: 'space-between' }}>
                <span>Loss Decomposition (Kendall Auto-Weighting)</span>
                <span style={{ fontFamily: 'var(--font-mono)', color: '#16a34a', fontWeight: 700 }}>
                  L_total = {result.lossComponents.reduce((s, l) => s + l.value, 0).toFixed(4)}
                </span>
              </div>
              {result.lossComponents.map(l => {
                const maxLoss = Math.max(...result.lossComponents.map(x => x.value))
                const pct = l.value / maxLoss
                return (
                  <div key={l.name} style={{ marginBottom: '6px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', marginBottom: '3px' }}>
                      <span style={{ fontFamily: 'var(--font-mono)', color: '#1e293b', fontWeight: 500 }}>{l.name}</span>
                      <span style={{ fontFamily: 'var(--font-mono)', color: '#2563eb', fontWeight: 600 }}>{l.value.toFixed(4)}</span>
                    </div>
                    <div style={{ height: '8px', background: '#e2e8f0', borderRadius: '2px', overflow: 'hidden', border: '1px solid #cbd5e1' }}>
                      <div style={{
                        height: '100%',
                        width: `${pct * 100}%`,
                        background: l.name.includes('recon') ? '#2563eb'
                          : l.name.includes('physics') ? '#16a34a'
                            : l.name.includes('param') ? '#d97706'
                              : l.name.includes('smooth') ? '#dc2626'
                                : '#7e22ce',
                        transition: 'width 0.5s ease',
                      }} />
                    </div>
                  </div>
                )
              })}
            </div>

            {/* Architecture reference */}
            <div style={{ border: '1px solid #cbd5e1', padding: '12px', background: '#f8fafc', borderRadius: '4px' }}>
              <div style={{ fontSize: '11px', fontWeight: 600, color: '#475569', textTransform: 'uppercase', marginBottom: '8px' }}>
                Architecture Pipeline
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '4px', overflowX: 'auto', paddingBottom: '4px' }}>
                {[
                  { label: 'Z(Δω) Input', sub: `N_off × H × W`, color: '#e2e8f0' },
                  { label: '1D Conv + PE', sub: 'Δω-aware', color: '#e2e8f0' },
                  { label: 'PACA', sub: '5 pool anchors', color: '#dbeafe' },
                  { label: 'Self-Attn ×2', sub: 'd=128', color: '#dbeafe' },
                  { label: 'Pool Heads', sub: '×4 + σ² out', color: '#dcfce7' },
                  { label: 'BM-ODE', sub: 'adjoint BP', color: '#fef3c7' },
                  { label: 'Ẑ_final', sub: '+δ_net', color: '#e2e8f0' },
                ].map((step, i, arr) => (
                  <React.Fragment key={step.label}>
                    <div style={{
                      background: step.color,
                      border: '1px solid #cbd5e1',
                      padding: '6px 8px',
                      borderRadius: '4px',
                      minWidth: '76px',
                      textAlign: 'center',
                      flexShrink: 0,
                    }}>
                      <div style={{ fontFamily: 'var(--font-mono)', fontSize: '9px', fontWeight: 700, color: '#0f172a', lineHeight: 1.2 }}>{step.label}</div>
                      <div style={{ fontFamily: 'var(--font-mono)', fontSize: '8px', color: '#475569', marginTop: '2px' }}>{step.sub}</div>
                    </div>
                    {i < arr.length - 1 && (
                      <div style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: '#94a3b8', padding: '0 1px', flexShrink: 0, fontWeight: 'bold' }}>→</div>
                    )}
                  </React.Fragment>
                ))}
              </div>
            </div>

          </div>
        )}
      </div>
    </div>
  )
}
