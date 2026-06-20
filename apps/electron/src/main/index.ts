import { app, BrowserWindow, ipcMain, dialog } from 'electron'
import path from 'path'
import fs from 'fs'

function createWindow() {
  const win = new BrowserWindow({
    width: 1280,
    height: 800,
    webPreferences: {
      preload: path.join(__dirname, '../preload/index.js'),
      contextIsolation: true,
    }
  })
  setTimeout(() => {
    win.loadURL('http://localhost:5173')
  }, 1500)
}

// IPC: ping
ipcMain.handle('ping', () => 'pong')

// IPC: open native file picker filtered to K-space / DICOM files
ipcMain.handle('dialog:openFile', async () => {
  const result = await dialog.showOpenDialog({
    title: 'Select K-Space or DICOM File',
    filters: [
      { name: 'K-Space Files', extensions: ['npy', 'h5', 'dat'] },
      { name: 'DICOM Files', extensions: ['dcm', 'DCM'] },
      { name: 'All Files', extensions: ['*'] }
    ],
    properties: ['openFile']
  })
  if (result.canceled || result.filePaths.length === 0) return null
  return result.filePaths[0]
})

// IPC: upload K-space study — reads file from disk and POSTs to backend
// Accepts optional metadata: patientId, modality, studyDate, phaseCorrection, denoiseMethod
ipcMain.handle('study:upload', async (_event, filePath: string, meta?: { patientId?: string; modality?: string; studyDate?: string; phaseCorrection?: string; denoiseMethod?: string }) => {
  try {
    const fileBuffer = fs.readFileSync(filePath)
    const fileName = path.basename(filePath)
    const fileBlob = new Blob([fileBuffer])

    const form = new globalThis.FormData()
    // Backend expects field name 'kspace'
    form.append('kspace', fileBlob, fileName)
    // Required fields — use sensible defaults if not provided
    form.append('patientId', meta?.patientId ?? '')
    form.append('modality', meta?.modality ?? 'MRI')
    form.append('studyDate', meta?.studyDate ?? new Date().toISOString())
    if (meta?.phaseCorrection !== undefined) {
      form.append('phaseCorrection', meta.phaseCorrection)
    }
    if (meta?.denoiseMethod !== undefined) {
      form.append('denoiseMethod', meta.denoiseMethod)
    }

    const response = await fetch('http://localhost:3000/studies/upload', {
      method: 'POST',
      body: form,
    })

    const data = await response.json()
    if (!response.ok) {
      return { success: false, error: data.error || 'Upload failed', details: data.details }
    }
    return { success: true, data }
  } catch (err: any) {
    return { success: false, error: err.message }
  }
})

app.whenReady().then(createWindow)
app.on('window-all-closed', () => { if (process.platform !== 'darwin') app.quit() })