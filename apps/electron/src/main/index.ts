import { app, BrowserWindow, ipcMain, dialog, clipboard } from 'electron'
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

// IPC: copy file from URL to local clipboard as native file object
ipcMain.handle('clipboard:copyFile', async (_event, url: string, fileName: string) => {
  try {
    const res = await fetch(url)
    if (!res.ok) throw new Error(`Failed to fetch file: ${res.statusText}`)
    const arrayBuffer = await res.arrayBuffer()
    const buffer = Buffer.from(arrayBuffer)
    
    const tempPath = path.join(app.getPath('temp'), fileName)
    fs.writeFileSync(tempPath, buffer)
    
    if (process.platform === 'win32') {
      const clipboardBuffer = Buffer.concat([
        Buffer.from(tempPath, 'ucs-2'),
        Buffer.from([0, 0])
      ])
      clipboard.writeBuffer('FileNameW', clipboardBuffer)
    } else if (process.platform === 'darwin') {
      clipboard.writeBuffer('public.file-url', Buffer.from(`file://${tempPath}`))
    } else {
      clipboard.writeText(tempPath)
    }
    
    return { success: true, filePath: tempPath }
  } catch (err: any) {
    return { success: false, error: err.message }
  }
})

app.whenReady().then(createWindow)
app.on('window-all-closed', () => { if (process.platform !== 'darwin') app.quit() })