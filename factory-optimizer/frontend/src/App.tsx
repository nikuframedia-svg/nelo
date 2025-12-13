import { useState } from 'react'
import { NavLink, Route, Routes, Navigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { DataUploader } from './components/DataUploader'
import { 
  Upload, 
  MessageCircle, 
  Factory, 
  Package, 
  Leaf, 
  Cpu, 
  Brain, 
  FlaskConical,
  ChevronDown,
  X
} from 'lucide-react'

// Main Modules
import Prodplan from './pages/Prodplan'
import SmartInventory from './pages/SmartInventory'
import DupliosPage from './pages/Duplios'

// Digital Twin
import DigitalTwinMachines from './pages/DigitalTwinMachines'
import XAIDTProduct from './pages/XAIDTProduct'

// Intelligence
import { WhatIf } from './pages/WhatIf'
import CausalAnalysis from './pages/CausalAnalysis'
import OptimizationDashboard from './pages/OptimizationDashboard'

// R&D
import { Research } from './pages/Research'

// Chat
import { Chat } from './pages/Chat'

// Legacy - kept for backwards compatibility but integrated into main modules
import Shopfloor from './pages/Shopfloor'
import PDMDashboard from './pages/PDMDashboard'
import MRPDashboard from './pages/MRPDashboard'
import WorkInstructions from './pages/WorkInstructions'
import PreventionGuard from './pages/PreventionGuard'

const queryClient = new QueryClient()

/**
 * ════════════════════════════════════════════════════════════════════════════════════════════
 * NAVEGAÇÃO REORGANIZADA - CONTRATO 15
 * ════════════════════════════════════════════════════════════════════════════════════════════
 * 
 * ESTRUTURA PRINCIPAL (6 módulos + Chat):
 * ─────────────────────────────────────
 * 
 * 1. PRODPLAN - Cockpit de Produção
 *    └─ Planeamento, Dashboards, Colaboradores, Gargalos, Sugestões, Shopfloor, Instruções, Prevenção
 * 
 * 2. SMARTINVENTORY - Digital Twin de Inventário  
 *    └─ Stock Real-Time, ABC/XYZ, Forecast & ROP, MRP, Parâmetros, BOM Explosion, Dados Operacionais
 * 
 * 3. DUPLIOS - Passaportes Digitais de Produto
 *    └─ Visão Geral, PDM, Impacto (LCA), Compliance, Identidade, Fornecedores, ESG Analytics
 * 
 * 4. DIGITAL TWIN - Saúde & Qualidade
 *    └─ Máquinas (SHI-DT), Produto (XAI-DT)
 * 
 * 5. INTELIGÊNCIA - IA & Otimização
 *    └─ Causal Analysis, Otimização, What-If Avançado
 * 
 * 6. R&D - Investigação & Desenvolvimento
 *    └─ Overview, WP1-WP4, Relatórios SIFIDE
 * 
 * 7. CHAT (botão flutuante)
 *    └─ Assistente IA Latif
 */

// Navigation structure - organized by main modules
const mainModules = [
  { 
    path: '/prodplan', 
    label: 'ProdPlan', 
    icon: Factory,
    description: 'Planeamento & Produção'
  },
  { 
    path: '/inventory', 
    label: 'SmartInventory', 
    icon: Package,
    description: 'Digital Twin de Inventário'
  },
  { 
    path: '/duplios', 
    label: 'Duplios', 
    icon: Leaf,
    description: 'Passaportes Digitais'
  },
  { 
    path: '/digital-twin', 
    label: 'Digital Twin', 
    icon: Cpu,
    description: 'Saúde & Qualidade'
  },
  { 
    path: '/intelligence', 
    label: 'Inteligência', 
    icon: Brain,
    description: 'IA & Otimização'
  },
  { 
    path: '/research', 
    label: 'R&D', 
    icon: FlaskConical,
    description: 'Investigação'
  },
]

// Sub-navigation for Digital Twin
const digitalTwinSubNav = [
  { path: '/digital-twin/machines', label: 'Máquinas (SHI-DT)' },
  { path: '/digital-twin/product', label: 'Produto (XAI-DT)' },
]

// Sub-navigation for Intelligence
const intelligenceSubNav = [
  { path: '/intelligence/causal', label: 'Análise Causal' },
  { path: '/intelligence/optimization', label: 'Otimização' },
  { path: '/intelligence/whatif', label: 'What-If Avançado' },
]

// Chat floating button component
function ChatButton() {
  const [isOpen, setIsOpen] = useState(false)
  
  return (
    <>
      {/* Floating Chat Button */}
      <motion.button
        onClick={() => setIsOpen(true)}
        className="fixed bottom-6 right-6 z-50 flex items-center gap-2 px-4 py-3 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white rounded-full shadow-lg shadow-cyan-500/20"
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
      >
        <MessageCircle className="w-5 h-5" />
        <span className="font-medium">Latif AI</span>
      </motion.button>
      
      {/* Chat Modal */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 bg-black/50 flex items-center justify-center p-4"
            onClick={() => setIsOpen(false)}
          >
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              className="w-full max-w-4xl h-[80vh] bg-slate-900 rounded-2xl border border-slate-700 overflow-hidden"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex items-center justify-between p-4 border-b border-slate-700">
                <div className="flex items-center gap-2">
                  <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center">
                    <MessageCircle className="w-4 h-4 text-white" />
                  </div>
                  <div>
                    <h3 className="font-semibold text-white">Latif AI Assistant</h3>
                    <p className="text-xs text-slate-400">Copilot de Produção</p>
                  </div>
                </div>
                <button 
                  onClick={() => setIsOpen(false)}
                  className="p-2 text-slate-400 hover:text-white hover:bg-slate-800 rounded-lg"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
              <div className="h-[calc(100%-60px)]">
                <Chat />
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  )
}

// Digital Twin wrapper page
function DigitalTwinPage() {
  return (
    <div className="space-y-6">
      <header className="space-y-2">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-gradient-to-br from-purple-500/20 to-pink-500/20 rounded-xl border border-purple-500/30">
            <Cpu className="w-6 h-6 text-purple-400" />
          </div>
          <div>
            <p className="text-xs uppercase tracking-[0.4em] text-slate-500 font-medium">Digital Twin</p>
            <h2 className="text-2xl font-semibold text-white">Saúde & Qualidade</h2>
          </div>
        </div>
      </header>
      
      {/* Sub-navigation */}
      <div className="flex gap-2">
        {digitalTwinSubNav.map((sub) => (
          <NavLink
            key={sub.path}
            to={sub.path}
            className={({ isActive }) =>
              `px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                isActive 
                  ? 'bg-purple-500/20 text-purple-400 border border-purple-500/30' 
                  : 'text-slate-400 hover:text-white hover:bg-slate-800/50'
              }`
            }
          >
            {sub.label}
          </NavLink>
        ))}
      </div>
      
      <Routes>
        <Route index element={<Navigate to="machines" replace />} />
        <Route path="machines" element={<DigitalTwinMachines />} />
        <Route path="product" element={<XAIDTProduct />} />
      </Routes>
    </div>
  )
}

// Intelligence wrapper page
function IntelligencePage() {
  return (
    <div className="space-y-6">
      <header className="space-y-2">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-gradient-to-br from-amber-500/20 to-orange-500/20 rounded-xl border border-amber-500/30">
            <Brain className="w-6 h-6 text-amber-400" />
          </div>
          <div>
            <p className="text-xs uppercase tracking-[0.4em] text-slate-500 font-medium">Inteligência</p>
            <h2 className="text-2xl font-semibold text-white">IA & Otimização</h2>
          </div>
        </div>
      </header>
      
      {/* Sub-navigation */}
      <div className="flex gap-2">
        {intelligenceSubNav.map((sub) => (
          <NavLink
            key={sub.path}
            to={sub.path}
            className={({ isActive }) =>
              `px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                isActive 
                  ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30' 
                  : 'text-slate-400 hover:text-white hover:bg-slate-800/50'
              }`
            }
          >
            {sub.label}
          </NavLink>
        ))}
      </div>
      
      <Routes>
        <Route index element={<Navigate to="causal" replace />} />
        <Route path="causal" element={<CausalAnalysis />} />
        <Route path="optimization" element={<OptimizationDashboard />} />
        <Route path="whatif" element={<WhatIf />} />
      </Routes>
    </div>
  )
}

function Shell() {
  const [showUploader, setShowUploader] = useState(false)

  return (
    <div className="min-h-screen bg-background text-text-body">
      {/* Header */}
      <header className="sticky top-0 z-30 border-b border-border bg-background/80 backdrop-blur-xl">
        <div className="container flex items-center justify-between py-4">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-nikufra/10 text-xl text-nikufra">
              ⚙️
            </div>
            <div>
              <p className="text-xs uppercase tracking-[0.4em] text-text-muted">Nikufra Ops</p>
              <h1 className="text-lg font-semibold text-text-primary">ProdPlan 4.0</h1>
            </div>
          </div>
          <button
            onClick={() => setShowUploader(true)}
            className="flex items-center gap-2 px-4 py-2 bg-nikufra hover:bg-nikufra/90 text-white rounded-lg transition text-sm font-medium"
          >
            <Upload className="w-4 h-4" />
            <span>Carregar Dados</span>
          </button>
        </div>
        
        {/* Main Navigation - Clean & Minimal */}
        <nav className="border-t border-border">
          <div className="container flex items-center gap-1 py-3 overflow-x-auto scrollbar-thin scrollbar-thumb-slate-700">
            {mainModules.map((module) => (
              <NavLink
                key={module.path}
                to={module.path}
                className={({ isActive }) =>
                  `group relative flex items-center gap-2 rounded-xl px-4 py-2.5 text-sm font-medium transition-all whitespace-nowrap ${
                    isActive 
                      ? 'bg-cyan-600/20 text-cyan-400 border border-cyan-500/30' 
                      : 'text-text-muted hover:text-text-primary hover:bg-slate-700/30'
                  }`
                }
              >
                {({ isActive }) => (
                  <>
                    <module.icon className={`w-4 h-4 ${isActive ? 'text-cyan-400' : 'text-slate-500 group-hover:text-slate-300'}`} />
                    {module.label}
                    {isActive && (
                      <motion.span
                        layoutId="nav-underline-main"
                        className="absolute -bottom-3 left-2 right-2 h-0.5 rounded-full bg-cyan-500"
                      />
                    )}
                  </>
                )}
              </NavLink>
            ))}
          </div>
        </nav>
      </header>

      {/* Main Content */}
      <main className="container pb-24 pt-8">
        <Routes>
          {/* Redirect root to Prodplan */}
          <Route path="/" element={<Navigate to="/prodplan" replace />} />
          
          {/* Main Modules */}
          <Route path="/prodplan" element={<Prodplan />} />
          <Route path="/inventory" element={<SmartInventory />} />
          <Route path="/duplios/*" element={<DupliosPage />} />
          <Route path="/digital-twin/*" element={<DigitalTwinPage />} />
          <Route path="/intelligence/*" element={<IntelligencePage />} />
          <Route path="/research" element={<Research />} />
          
          {/* Legacy routes - redirect to integrated modules */}
          <Route path="/shopfloor" element={<Shopfloor />} />
          <Route path="/shi-dt" element={<Navigate to="/digital-twin/machines" replace />} />
          <Route path="/xai-dt" element={<Navigate to="/digital-twin/product" replace />} />
          <Route path="/pdm" element={<PDMDashboard />} />
          <Route path="/mrp" element={<MRPDashboard />} />
          <Route path="/work-instructions" element={<WorkInstructions />} />
          <Route path="/optimization" element={<Navigate to="/intelligence/optimization" replace />} />
          <Route path="/prevention-guard" element={<PreventionGuard />} />
          <Route path="/whatif" element={<Navigate to="/intelligence/whatif" replace />} />
          <Route path="/causal" element={<Navigate to="/intelligence/causal" replace />} />
          <Route path="/chat" element={<Chat />} />
          
          {/* More legacy redirects */}
          <Route path="/smart-inventory" element={<Navigate to="/inventory" replace />} />
          <Route path="/advanced" element={<Navigate to="/prodplan" replace />} />
          <Route path="/dashboards" element={<Navigate to="/prodplan" replace />} />
          <Route path="/reports" element={<Navigate to="/prodplan" replace />} />
          <Route path="/projects" element={<Navigate to="/prodplan" replace />} />
          <Route path="/workforce" element={<Navigate to="/prodplan" replace />} />
          <Route path="/bottlenecks" element={<Navigate to="/prodplan" replace />} />
          <Route path="/suggestions" element={<Navigate to="/prodplan" replace />} />
        </Routes>
      </main>
      
      {/* Floating Chat Button */}
      <ChatButton />
      
      {/* Data Uploader Modal */}
      <DataUploader isOpen={showUploader} onClose={() => setShowUploader(false)} />
    </div>
  )
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <Shell />
    </QueryClientProvider>
  )
}

export default App
