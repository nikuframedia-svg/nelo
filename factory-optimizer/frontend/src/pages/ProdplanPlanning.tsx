/**
 * ════════════════════════════════════════════════════════════════════════════════════════════
 * PRODPLAN PLANNING SUBMODULE
 * ════════════════════════════════════════════════════════════════════════════════════════════
 *
 * Integra todas as funcionalidades de planeamento:
 * - Vista Principal (Gantt, Timeline)
 * - Planeamento Avançado (Modos, Motores)
 * - Projetos (KPIs, Vista por Projeto)
 */

import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  LayoutGrid,
  Settings,
  FolderKanban,
} from 'lucide-react'

// Import existing components
import { Planning } from './Planning'
import AdvancedPlanning from './AdvancedPlanning'
import ProjectPlanning from './ProjectPlanning'

type PlanningView = 'main' | 'advanced' | 'projects'

const views = [
  {
    id: 'main' as PlanningView,
    label: 'Principal',
    icon: <LayoutGrid className="w-4 h-4" />,
    description: 'Gantt e Timeline',
  },
  {
    id: 'advanced' as PlanningView,
    label: 'Avançado',
    icon: <Settings className="w-4 h-4" />,
    description: 'Modos e Motores',
  },
  {
    id: 'projects' as PlanningView,
    label: 'Projetos',
    icon: <FolderKanban className="w-4 h-4" />,
    description: 'Vista por Projeto',
  },
]

const ProdplanPlanning: React.FC = () => {
  const [activeView, setActiveView] = useState<PlanningView>('main')

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
          {activeView === 'main' && <Planning />}
          {activeView === 'advanced' && <AdvancedPlanning />}
          {activeView === 'projects' && <ProjectPlanning />}
        </motion.div>
      </AnimatePresence>
    </div>
  )
}

export default ProdplanPlanning
