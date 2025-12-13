/**
 * ════════════════════════════════════════════════════════════════════════════════════════════
 * PRODPLAN BOTTLENECKS (GARGALOS) SUBMODULE
 * ════════════════════════════════════════════════════════════════════════════════════════════
 *
 * Deteção e gestão de gargalos:
 * - Máquinas críticas
 * - Análise de carga
 * - Saturação
 * - Sugestões de mitigação
 */

import React from 'react'

// Import existing component
import { Bottlenecks } from './Bottlenecks'

const ProdplanBottlenecks: React.FC = () => {
  return (
    <div className="space-y-6">
      {/* The Bottlenecks component already has all the needed functionality */}
      <Bottlenecks />
    </div>
  )
}

export default ProdplanBottlenecks

