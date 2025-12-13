/**
 * ════════════════════════════════════════════════════════════════════════════════════════════
 * PRODPLAN WORKFORCE (COLABORADORES) SUBMODULE
 * ════════════════════════════════════════════════════════════════════════════════════════════
 *
 * Gestão de colaboradores e performance:
 * - KPIs de produtividade
 * - Saturação de equipas
 * - Competências
 * - Recomendações de alocação
 */

import React from 'react'

// Import existing component
import WorkforcePerformance from './WorkforcePerformance'

const ProdplanWorkforce: React.FC = () => {
  return (
    <div className="space-y-6">
      {/* The WorkforcePerformance component already has all the needed functionality */}
      <WorkforcePerformance />
    </div>
  )
}

export default ProdplanWorkforce



