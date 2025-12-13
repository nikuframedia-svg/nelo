import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // Proxy removido: o frontend agora comunica diretamente com o backend via VITE_API_BASE_URL
  },
})

