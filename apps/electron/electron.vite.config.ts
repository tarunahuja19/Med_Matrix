import { defineConfig } from 'electron-vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  main: {
    build: {
      target: 'node20',
      rollupOptions: {
        input: 'src/main/index.ts',
        output: {
          format: 'cjs',
          entryFileNames: 'index.js'
        }
      }
    }
  },
  preload: {
    build: {
      target: 'node20',
      rollupOptions: {
        input: 'src/preload/index.ts',
        output: {
          format: 'cjs',
          entryFileNames: 'index.js'
        }
      }
    }
  },
  renderer: {
    build: {
      target: 'chrome120'
    },
    plugins: [react()]
  }
})