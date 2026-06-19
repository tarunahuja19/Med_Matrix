import { app, BrowserWindow, ipcMain } from 'electron'
import path from 'path'

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

// IPC bridge skeleton
ipcMain.handle('ping', () => 'pong')

app.whenReady().then(createWindow)
app.on('window-all-closed', () => { if (process.platform !== 'darwin') app.quit() })