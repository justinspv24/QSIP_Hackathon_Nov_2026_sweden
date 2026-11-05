import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'
import { viteSingleFile } from 'vite-plugin-singlefile'

// One self-contained dist/index.html (JS, CSS and KaTeX fonts inlined), so it opens straight from disk (file://),
// offline, during the talk. A normal multi-file build is blocked by the browser's CORS rule for module scripts on file://.
export default defineConfig({
  plugins: [react(), viteSingleFile()],
  base: './',
  build: { assetsInlineLimit: 100_000_000, chunkSizeWarningLimit: 5000 },
})
