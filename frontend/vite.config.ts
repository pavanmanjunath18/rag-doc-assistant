import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// The UI calls /api/...; in development Vite forwards those requests to the FastAPI server,
// so the browser only ever talks to one origin and the API needs no CORS setup.
// In Docker, nginx does the same job (see frontend/nginx.conf).
const apiTarget = process.env.API_URL ?? 'http://127.0.0.1:8000'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: apiTarget,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})
