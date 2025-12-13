/**
 * ════════════════════════════════════════════════════════════════════════════════════════════
 * PRODPLAN DASHBOARDS & REPORTS SUBMODULE
 * ════════════════════════════════════════════════════════════════════════════════════════════
 *
 * Integra dashboards e relatórios:
 * - Heatmap de Utilização
 * - OEE de Máquinas
 * - Células de Produção
 * - Projeção Anual
 * - Relatórios Comparativos
 */

import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  BarChart3,
  FileText,
} from 'lucide-react'

// Import existing components
import Dashboards from './Dashboards'
import Reports from './Reports'

type DashboardView = 'dashboards' | 'reports'

const views = [
  {
    id: 'dashboards' as DashboardView,
    label: 'Dashboards',
    icon: <BarChart3 className="w-4 h-4" />,
    description: 'Heatmaps, OEE, Projeções',
  },
  {
    id: 'reports' as DashboardView,
    label: 'Relatórios',
    icon: <FileText className="w-4 h-4" />,
    description: 'Comparações, Explicações Técnicas',
  },
]

const ProdplanDashboards: React.FC = () => {
  const [activeView, setActiveView] = useState<DashboardView>('dashboards')

  return (
    <div className="space-y-6">
      {/* View Selector */}
      <div className="flex items-center gap-4">
        <div className="flex gap-1 p-1 bg-slate-800/50 rounded-lg">
          {views.map((view) => (
            <button
              key={view.id}
              onClick={() => setActiveView(view.id)}
              className={`
                flex items-center gap-2 px-3 py-2 rounded-md text-sm font-medium transition-all
                ${activeView === view.id
                  ? 'bg-cyan-600 text-white'
                  : 'text-slate-400 hover:text-white hover:bg-slate-700/50'
                }
              `}
              title={view.description}
            >
              {view.icon}
              <span>{view.label}</span>
            </button>
          ))}
        </div>
        
        <div className="text-xs text-slate-500">
          {views.find(v => v.id === activeView)?.description}
        </div>
      </div>

      {/* Content */}
      <AnimatePresence mode="wait">
        <motion.div
          key={activeView}
          initial={{ opacity: 0, x: -10 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: 10 }}
          transition={{ duration: 0.15 }}
        >
          {activeView === 'dashboards' && <Dashboards />}
          {activeView === 'reports' && <Reports />}
        </motion.div>
      </AnimatePresence>
    </div>
  )
}

export default ProdplanDashboards

