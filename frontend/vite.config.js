import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],

  server: {
    proxy: {
      // HTTP API
      '/analyze': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },

      // WebSocket progress channel
      '/ws': {
        target: 'ws://localhost:8000',
        ws: true,
      },
    },
  },

  optimizeDeps: {
    include: ['plotly.js-dist-min'],
  },

  build: {
    commonjsOptions: {
      include: [/plotly\.js-dist-min/],
    },
  },
})
