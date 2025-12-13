/**
 * ════════════════════════════════════════════════════════════════════════════════════════════════════
 * PRODPLAN DIGITAL TWIN SUBMODULE
 * ════════════════════════════════════════════════════════════════════════════════════════════════════
 *
 * Wrapper para integrar o Digital Twin no módulo Prodplan.
 *
 * Sub-sections:
 * - Máquinas (PdM-IPS): RUL, Health Indicators, manutenção preditiva
 * - Produtos (XAI-DT): Conformidade, análise de scans, golden runs
 */

import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { Cpu, Box } from 'lucide-react'
import DigitalTwin from './DigitalTwin'
import DigitalTwinProduct from './DigitalTwinProduct'

type SubView = 'machines' | 'products'

const ProdplanDigitalTwin: React.FC = () => {
  const [activeView, setActiveView] = useState<SubView>('machines')

  return (
    <div className="space-y-6">
      {/* Sub-navigation */}
      <div className="flex items-center gap-2 p-1 bg-slate-800/50 rounded-xl w-fit">
        <button
          onClick={() => setActiveView('machines')}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
            activeView === 'machines'
              ? 'bg-cyan-600 text-white'
              : 'text-slate-400 hover:text-white hover:bg-slate-700/50'
          }`}
        >
          <Cpu className="w-4 h-4" />
          Máquinas (PdM-IPS)
        </button>
        <button
          onClick={() => setActiveView('products')}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
            activeView === 'products'
              ? 'bg-purple-600 text-white'
              : 'text-slate-400 hover:text-white hover:bg-slate-700/50'
          }`}
        >
          <Box className="w-4 h-4" />
          Produtos (XAI-DT)
        </button>
      </div>

      {/* Content */}
      <motion.div
        key={activeView}
        initial={{ opacity: 0, x: 10 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.2 }}
      >
        {activeView === 'machines' ? <DigitalTwin /> : <DigitalTwinProduct />}
      </motion.div>
    </div>
  )
}

export default ProdplanDigitalTwin
