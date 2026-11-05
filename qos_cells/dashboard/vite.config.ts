import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// base './' so that dist/index.html also works when opened straight from disk (file://), offline, during the talk.
export default defineConfig({
  plugins: [react()],
  base: './',
})
