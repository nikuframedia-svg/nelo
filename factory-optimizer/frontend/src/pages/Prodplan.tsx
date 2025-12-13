/**
 * ════════════════════════════════════════════════════════════════════════════════════════════
 * PRODPLAN MODULE - COCKPIT DE PLANEAMENTO E ANÁLISE DE PRODUÇÃO
 * ════════════════════════════════════════════════════════════════════════════════════════════
 * 
 * Contrato 16 - Reorganização Visual Completa
 *
 * Módulo principal que agrega:
 * 1. PLANEAMENTO - Gantt, timeline, modos de planeamento
 * 2. DASHBOARDS - Heatmaps, OEE, projeções, relatórios
 * 3. COLABORADORES - Performance, produtividade, saturação
 * 4. GARGALOS - Deteção, análise, mitigação
 * 5. SUGESTÕES IA - Recomendações inteligentes de produção
 * 6. MÁQUINAS - Productive Care, manutenção, paragens, SHI, RUL (NOVO)
 * 7. FERRAMENTAS AVANÇADAS - Data Quality, MILP, Prevention Guard
 */

import React, { useState, lazy, Suspense } from 'react'
import { motion } from 'framer-motion'
import {
  Calendar,
  BarChart3,
  Users,
  AlertTriangle,
  Lightbulb,
  Factory,
  Cpu,
  Settings,
  Loader2,
  ChevronDown,
} from 'lucide-react'

// Import submodules
import ProdplanPlanning from './ProdplanPlanning'
import ProdplanDashboards from './ProdplanDashboards'
import ProdplanWorkforce from './ProdplanWorkforce'
import ProdplanBottlenecks from './ProdplanBottlenecks'
import ProdplanSuggestions from './ProdplanSuggestions'
import ProdplanDigitalTwin from './ProdplanDigitalTwin'

// Lazy load advanced panels
const MachinesPanel = lazy(() => import('../components/MachinesPanel'))
const DataQualityPanel = lazy(() => import('../components/DataQualityPanel'))
const MILPOptimizationPanel = lazy(() => import('../components/MILPOptimizationPanel'))
const PreventionGuardPanel = lazy(() => import('../components/PreventionGuardPanel'))
const SHIDTTrainingPanel = lazy(() => import('../components/SHIDTTrainingPanel'))
const WorkInstructionsPanel = lazy(() => import('./WorkInstructions'))

// Tab types
type MainTabId = 'planning' | 'dashboards' | 'workforce' | 'bottlenecks' | 'suggestions' | 'machines' | 'tools'
type ToolsSubTabId = 'digital-twin' | 'data-quality' | 'milp' | 'prevention' | 'shi-training' | 'work-instructions'

// Main tabs - consolidated structure
const mainTabs: { id: MainTabId; label: string; icon: React.ReactNode; description: string }[] = [
  {
    id: 'planning',
    label: 'Planeamento',
    icon: <Calendar className="w-4 h-4" />,
    description: 'Gantt, Timeline, Modos de Planeamento',
  },
  {
    id: 'dashboards',
    label: 'Dashboards',
    icon: <BarChart3 className="w-4 h-4" />,
    description: 'Heatmaps, OEE, Projeções, Relatórios',
  },
  {
    id: 'workforce',
    label: 'Colaboradores',
    icon: <Users className="w-4 h-4" />,
    description: 'Performance, Saturação, Competências',
  },
  {
    id: 'bottlenecks',
    label: 'Gargalos',
    icon: <AlertTriangle className="w-4 h-4" />,
    description: 'Deteção, Carga, Mitigação',
  },
  {
    id: 'suggestions',
    label: 'Sugestões IA',
    icon: <Lightbulb className="w-4 h-4" />,
    description: 'Recomendações Inteligentes',
  },
  {
    id: 'machines',
    label: 'Máquinas',
    icon: <Cpu className="w-4 h-4" />,
    description: 'Productive Care, Manutenção, Paragens',
  },
  {
    id: 'tools',
    label: 'Ferramentas',
    icon: <Settings className="w-4 h-4" />,
    description: 'Digital Twin, Otimização, Prevenção',
  },
]

// Tools sub-tabs
const toolsSubTabs: { id: ToolsSubTabId; label: string }[] = [
  { id: 'digital-twin', label: 'Digital Twin' },
  { id: 'data-quality', label: 'Qualidade Dados' },
  { id: 'milp', label: 'Otimização MILP' },
  { id: 'prevention', label: 'Prevention Guard' },
  { id: 'shi-training', label: 'SHI-DT Training' },
  { id: 'work-instructions', label: 'Instruções de Trabalho' },
]

const LoadingFallback = () => (
  <div className="flex items-center justify-center py-24">
    <Loader2 className="w-8 h-8 text-cyan-400 animate-spin" />
  </div>
)

const Prodplan: React.FC = () => {
  const [activeTab, setActiveTab] = useState<MainTabId>('planning')
  const [activeToolsSubTab, setActiveToolsSubTab] = useState<ToolsSubTabId>('digital-twin')
  const [showToolsMenu, setShowToolsMenu] = useState(false)

  return (
    <div className="space-y-6">
      {/* Header */}
      <header className="space-y-2">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-gradient-to-br from-cyan-500/20 to-blue-500/20 rounded-xl border border-cyan-500/30">
            <Factory className="w-6 h-6 text-cyan-400" />
          </div>
          <div>
            <p className="text-xs uppercase tracking-[0.4em] text-slate-500 font-medium">
              Prodplan
            </p>
            <h2 className="text-2xl font-semibold text-white">
              Cockpit de Produção
            </h2>
          </div>
        </div>
        <p className="max-w-3xl text-sm text-slate-400 ml-14">
          Gestão integrada de planeamento, máquinas, colaboradores e otimização de produção.
        </p>
      </header>

      {/* Tab Navigation - Consolidated */}
      <div className="border-b border-slate-700/50">
        <nav className="flex gap-1 -mb-px overflow-x-auto pb-px scrollbar-thin scrollbar-thumb-slate-700">
          {mainTabs.map((tab) => (
            <div key={tab.id} className="relative">
              {tab.id === 'tools' ? (
                // Tools tab with dropdown
                <div className="relative">
                  <button
                    onClick={() => {
                      if (activeTab === 'tools') {
                        setShowToolsMenu(!showToolsMenu)
                      } else {
                        setActiveTab('tools')
                        setShowToolsMenu(true)
                      }
                    }}
                    className={`
                      group flex items-center gap-2 px-4 py-3 text-sm font-medium 
                      border-b-2 transition-all duration-200 whitespace-nowrap
                      ${activeTab === 'tools'
                        ? 'border-cyan-500 text-cyan-400'
                        : 'border-transparent text-slate-400 hover:text-white hover:border-slate-600'
                      }
                    `}
                    title={tab.description}
                  >
                    <span className={`${activeTab === 'tools' ? 'text-cyan-400' : 'text-slate-500 group-hover:text-slate-300'}`}>
                      {tab.icon}
                    </span>
                    {tab.label}
                    <ChevronDown className={`w-3 h-3 transition-transform ${showToolsMenu && activeTab === 'tools' ? 'rotate-180' : ''}`} />
                  </button>
                  
                  {/* Dropdown menu */}
                  {showToolsMenu && activeTab === 'tools' && (
                    <motion.div
                      initial={{ opacity: 0, y: -10 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="absolute top-full left-0 mt-1 z-50 bg-slate-800 rounded-lg border border-slate-700 shadow-xl py-1 min-w-[180px]"
                    >
                      {toolsSubTabs.map((subTab) => (
                        <button
                          key={subTab.id}
                          onClick={() => {
                            setActiveToolsSubTab(subTab.id)
                            setShowToolsMenu(false)
                          }}
                          className={`w-full text-left px-4 py-2 text-sm transition-colors ${
                            activeToolsSubTab === subTab.id
                              ? 'bg-cyan-500/20 text-cyan-400'
                              : 'text-slate-300 hover:bg-slate-700'
                          }`}
                        >
                          {subTab.label}
                        </button>
                      ))}
                    </motion.div>
                  )}
                </div>
              ) : (
                // Regular tabs
                <button
                  onClick={() => {
                    setActiveTab(tab.id)
                    setShowToolsMenu(false)
                  }}
                  className={`
                    group flex items-center gap-2 px-4 py-3 text-sm font-medium 
                    border-b-2 transition-all duration-200 whitespace-nowrap
                    ${activeTab === tab.id
                      ? 'border-cyan-500 text-cyan-400'
                      : 'border-transparent text-slate-400 hover:text-white hover:border-slate-600'
                    }
                  `}
                  title={tab.description}
                >
                  <span className={`${activeTab === tab.id ? 'text-cyan-400' : 'text-slate-500 group-hover:text-slate-300'}`}>
                    {tab.icon}
                  </span>
                  {tab.label}
                  {tab.id === 'machines' && (
                    <span className="px-1.5 py-0.5 text-[10px] font-bold bg-purple-500/20 text-purple-400 rounded">NOVO</span>
                  )}
                </button>
              )}
            </div>
          ))}
        </nav>
      </div>

      {/* Tools Sub-tab indicator */}
      {activeTab === 'tools' && (
        <div className="flex items-center gap-2 text-sm">
          <span className="text-slate-500">Ferramenta:</span>
          <span className="text-cyan-400 font-medium">
            {toolsSubTabs.find(t => t.id === activeToolsSubTab)?.label}
          </span>
        </div>
      )}

      {/* Tab Content */}
      <motion.div
        key={activeTab + (activeTab === 'tools' ? activeToolsSubTab : '')}
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.2 }}
        onClick={() => setShowToolsMenu(false)}
      >
        {/* Main tabs content */}
        {activeTab === 'planning' && <ProdplanPlanning />}
        {activeTab === 'dashboards' && <ProdplanDashboards />}
        {activeTab === 'workforce' && <ProdplanWorkforce />}
        {activeTab === 'bottlenecks' && <ProdplanBottlenecks />}
        {activeTab === 'suggestions' && <ProdplanSuggestions />}
        
        {/* NEW: Machines tab */}
        {activeTab === 'machines' && (
          <Suspense fallback={<LoadingFallback />}>
            <MachinesPanel />
          </Suspense>
        )}
        
        {/* Tools tab content */}
        {activeTab === 'tools' && (
          <>
            {activeToolsSubTab === 'digital-twin' && <ProdplanDigitalTwin />}
            {activeToolsSubTab === 'data-quality' && (
              <Suspense fallback={<LoadingFallback />}>
                <DataQualityPanel />
              </Suspense>
            )}
            {activeToolsSubTab === 'milp' && (
              <Suspense fallback={<LoadingFallback />}>
                <MILPOptimizationPanel />
              </Suspense>
            )}
            {activeToolsSubTab === 'prevention' && (
              <Suspense fallback={<LoadingFallback />}>
                <PreventionGuardPanel />
              </Suspense>
            )}
            {activeToolsSubTab === 'shi-training' && (
              <Suspense fallback={<LoadingFallback />}>
                <SHIDTTrainingPanel />
              </Suspense>
            )}
            {activeToolsSubTab === 'work-instructions' && (
              <Suspense fallback={<LoadingFallback />}>
                <WorkInstructionsPanel />
              </Suspense>
            )}
          </>
        )}
      </motion.div>
    </div>
  )
}

export default Prodplan
