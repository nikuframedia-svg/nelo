import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import { Toaster } from 'react-hot-toast'
import App from './App.tsx'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
    <Toaster
      position="bottom-right"
      toastOptions={{
        style: {
          background: '#121212',
          color: '#EAEAEA',
          border: '1px solid #262626',
          boxShadow: '0 4px 20px rgba(0, 230, 118, 0.08)',
        },
        success: {
          iconTheme: {
            primary: '#00E676',
            secondary: '#0B0B0B',
          },
        },
      }}
    />
  </React.StrictMode>,
)

