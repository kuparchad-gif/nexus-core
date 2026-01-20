import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'
// FIX: Import fileURLToPath to construct __dirname in an ES module context.
import { fileURLToPath } from 'node:url'

// FIX: __dirname is not available in ES modules. This creates an equivalent.
const __dirname = path.dirname(fileURLToPath(import.meta.url));

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './'),
    },
  },
})
