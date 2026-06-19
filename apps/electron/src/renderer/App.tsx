import { useEffect, useState } from 'react'

function App() {
  const [status, setStatus] = useState('loading...')

  useEffect(() => {
    if ((window as any).api) {
      (window as any).api.ping().then(setStatus)
    } else {
      setStatus('running in browser')
    }
  }, [])

  return (
    <div>
      <h1>NeuroScan AI</h1>
      <p>IPC Status: {status}</p>
    </div>
  )
}

export default App