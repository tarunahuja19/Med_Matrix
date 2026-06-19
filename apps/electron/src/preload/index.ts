import { contextBridge, ipcRenderer } from 'electron'

contextBridge.exposeInMainWorld('api', {
  ping: () => ipcRenderer.invoke('ping'),
  openFileDialog: () => ipcRenderer.invoke('dialog:openFile'),
  uploadStudy: (filePath: string, meta?: Record<string, string>) =>
    ipcRenderer.invoke('study:upload', filePath, meta),
})