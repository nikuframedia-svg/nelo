/**
 * ════════════════════════════════════════════════════════════════════════════════════════════
 * PRODPLAN 4.0 — ADVANCED DASHBOARDS PAGE
 * ════════════════════════════════════════════════════════════════════════════════════════════
 *
 * Dashboards avançados:
 * - Gantt Comparativo
 * - Heatmap de Utilização
 * - Dashboard de Operadores
 * - Dashboard de Máquinas/OEE
 * - Performance de Células
 * - Projeção de Capacidade
 */

import React, { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { motion, AnimatePresence } from 'framer-motion'
import {
  BarChart3,
  Calendar,
  Users,
  Settings,
  GitBranch,
  TrendingUp,
  AlertTriangle,
  CheckCircle,
  Clock,
  Activity,
  Layers,
  Grid,
} from 'lucide-react'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000'

// ═══════════════════════════════════════════════════════════════════════════════════════════
// API FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════════════════════

async function fetchHeatmap() {
  const res = await fetch(`${API_BASE_URL}/dashboards/utilization-heatmap`)
  if (!res.ok) throw new Error('Failed to fetch heatmap')
  return res.json()
}

async function fetchOperatorDashboard() {
  const res = await fetch(`${API_BASE_URL}/dashboards/operator`)
  if (!res.ok) throw new Error('Failed to fetch operator dashboard')
  return res.json()
}

async function fetchMachineOEE() {
  const res = await fetch(`${API_BASE_URL}/dashboards/machine-oee`)
  if (!res.ok) throw new Error('Failed to fetch OEE')
  return res.json()
}

async function fetchCellPerformance() {
  const res = await fetch(`${API_BASE_URL}/dashboards/cell-performance`)
  if (!res.ok) throw new Error('Failed to fetch cell performance')
  return res.json()
}

async function fetchCapacityProjection() {
  const res = await fetch(`${API_BASE_URL}/dashboards/capacity-projection`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ forecast_months: 12, demand_growth_monthly: 0.02 }),
  })
  if (!res.ok) throw new Error('Failed to fetch projection')
  return res.json()
}

// ═══════════════════════════════════════════════════════════════════════════════════════════
// HEATMAP COMPONENT
// ═══════════════════════════════════════════════════════════════════════════════════════════

const UtilizationHeatmap: React.FC<{ data: any }> = ({ data }) => {
  if (!data?.machines?.length) return <div className="text-slate-500">Sem dados</div>

  const hours = data.hours.filter((_: number, i: number) => i % 2 === 0) // Show every 2 hours

  return (
    <div className="overflow-x-auto">
      <div className="min-w-[800px]">
        {/* Header - Hours */}
        <div className="flex mb-2">
          <div className="w-24 flex-shrink-0" />
          {hours.map((h: number) => (
            <div key={h} className="w-8 text-center text-xs text-slate-400">{h}h</div>
          ))}
        </div>

        {/* Machine rows */}
        {data.machines.map((machine: string) => (
          <div key={machine} className="flex items-center mb-1">
            <div className="w-24 flex-shrink-0 text-sm text-slate-300 truncate">{machine}</div>
            <div className="flex gap-px">
              {hours.map((h: number) => {
                const key = `0_${h}` // Using Monday (day 0) for now
                const cell = data.matrix?.[machine]?.[key]
                const util = cell?.utilization ?? 0
                
                return (
                  <div
                    key={h}
                    className="w-8 h-6 rounded-sm transition-colors cursor-pointer hover:ring-1 hover:ring-white/50"
                    style={{ backgroundColor: cell?.color ?? '#1e293b' }}
                    title={`${machine} ${h}:00 - ${util.toFixed(0)}%`}
                  />
                )
              })}
            </div>
          </div>
        ))}

        {/* Legend */}
        <div className="flex items-center gap-4 mt-4 text-xs text-slate-400">
          <span>Utilização:</span>
          <div className="flex items-center gap-1">
            <div className="w-4 h-4 rounded" style={{ backgroundColor: '#1e40af' }} />
            <span>0%</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-4 h-4 rounded" style={{ backgroundColor: '#22c55e' }} />
            <span>50%</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-4 h-4 rounded" style={{ backgroundColor: '#7f1d1d' }} />
            <span>100%</span>
          </div>
        </div>
      </div>
    </div>
  )
}

// ═══════════════════════════════════════════════════════════════════════════════════════════
// OPERATOR DASHBOARD COMPONENT
// ═══════════════════════════════════════════════════════════════════════════════════════════

const OperatorDashboardView: React.FC<{ data: any }> = ({ data }) => {
  if (!data?.operators?.length) return <div className="text-slate-500">Sem dados</div>

  return (
    <div className="space-y-4">
      {/* Summary cards */}
      <div className="grid grid-cols-4 gap-3">
        <div className="p-3 bg-slate-800/50 rounded-lg">
          <p className="text-sm text-slate-400">Total</p>
          <p className="text-2xl font-bold text-white">{data.summary.total_operators}</p>
        </div>
        <div className="p-3 bg-red-900/30 rounded-lg">
          <p className="text-sm text-slate-400">Sobrecarregados</p>
          <p className="text-2xl font-bold text-red-400">{data.summary.overloaded_count}</p>
        </div>
        <div className="p-3 bg-amber-900/30 rounded-lg">
          <p className="text-sm text-slate-400">Subutilizados</p>
          <p className="text-2xl font-bold text-amber-400">{data.summary.underutilized_count}</p>
        </div>
        <div className="p-3 bg-cyan-900/30 rounded-lg">
          <p className="text-sm text-slate-400">Gaps de Skill</p>
          <p className="text-2xl font-bold text-cyan-400">{data.summary.critical_gaps}</p>
        </div>
      </div>

      {/* Operator list */}
      <div className="space-y-2">
        {data.operators.slice(0, 8).map((op: any) => (
          <div
            key={op.operator_id}
            className={`p-3 rounded-lg border ${
              op.status === 'overloaded' ? 'bg-red-900/20 border-red-700/50' :
              op.status === 'underutilized' ? 'bg-amber-900/20 border-amber-700/50' :
              'bg-slate-800/50 border-slate-700/50'
            }`}
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="font-medium text-white">{op.name}</p>
                <p className="text-xs text-slate-400">{op.skills?.length ?? 0} competências</p>
              </div>
              <div className="text-right">
                <p className="text-lg font-bold text-white">{op.hours_allocated.toFixed(0)}h</p>
                <p className="text-xs text-slate-400">/ {op.hours_available}h</p>
              </div>
              <div className="w-24">
                <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full ${
                      op.utilization_pct > 100 ? 'bg-red-500' :
                      op.utilization_pct < 50 ? 'bg-amber-500' :
                      'bg-emerald-500'
                    }`}
                    style={{ width: `${Math.min(100, op.utilization_pct)}%` }}
                  />
                </div>
                <p className="text-xs text-center text-slate-400 mt-1">{op.utilization_pct.toFixed(0)}%</p>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

// ═══════════════════════════════════════════════════════════════════════════════════════════
// OEE DASHBOARD COMPONENT
// ═══════════════════════════════════════════════════════════════════════════════════════════

const OEEDashboardView: React.FC<{ data: any }> = ({ data }) => {
  if (!data?.machines?.length) return <div className="text-slate-500">Sem dados</div>

  const statusColors: Record<string, string> = {
    excellent: 'bg-emerald-500',
    good: 'bg-green-500',
    acceptable: 'bg-yellow-500',
    poor: 'bg-orange-500',
    critical: 'bg-red-500',
  }

  return (
    <div className="space-y-4">
      {/* Overall OEE */}
      <div className="p-4 bg-gradient-to-r from-cyan-900/30 to-blue-900/30 rounded-lg border border-cyan-700/30">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm text-slate-400">OEE Global</p>
            <p className="text-4xl font-bold text-white">{data.overall_oee.toFixed(1)}%</p>
          </div>
          <div className="text-right">
            <p className="text-sm text-slate-400">Gargalo</p>
            <p className="text-lg font-medium text-cyan-400">{data.bottleneck_machine || '-'}</p>
          </div>
        </div>
      </div>

      {/* Machine cards */}
      <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
        {data.machines.slice(0, 9).map((m: any) => (
          <div
            key={m.machine_id}
            className={`p-3 rounded-lg border ${
              m.is_bottleneck ? 'bg-amber-900/20 border-amber-700/50 ring-2 ring-amber-500/50' :
              m.status === 'critical' ? 'bg-red-900/20 border-red-700/50' :
              'bg-slate-800/50 border-slate-700/50'
            }`}
          >
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-white">{m.machine_id}</span>
              <span className={`w-2 h-2 rounded-full ${statusColors[m.status] || 'bg-slate-500'}`} />
            </div>
            <div className="text-2xl font-bold text-white">{m.oee.total.toFixed(0)}%</div>
            <div className="grid grid-cols-3 gap-1 mt-2 text-xs">
              <div>
                <p className="text-slate-500">A</p>
                <p className="text-slate-300">{m.oee.availability.toFixed(0)}%</p>
              </div>
              <div>
                <p className="text-slate-500">P</p>
                <p className="text-slate-300">{m.oee.performance.toFixed(0)}%</p>
              </div>
              <div>
                <p className="text-slate-500">Q</p>
                <p className="text-slate-300">{m.oee.quality.toFixed(0)}%</p>
              </div>
            </div>
            {m.is_bottleneck && (
              <div className="mt-2 text-xs text-amber-400 flex items-center gap-1">
                <AlertTriangle className="w-3 h-3" />
                Gargalo
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}

// ═══════════════════════════════════════════════════════════════════════════════════════════
// CELL PERFORMANCE COMPONENT
// ═══════════════════════════════════════════════════════════════════════════════════════════

const CellPerformanceView: React.FC<{ data: any }> = ({ data }) => {
  if (!data?.cells?.length) return <div className="text-slate-500">Sem células detectadas</div>

  return (
    <div className="space-y-4">
      {data.cells.slice(0, 4).map((cell: any) => (
        <div
          key={cell.cell_id}
          className={`p-4 rounded-lg border ${
            cell.status === 'healthy' ? 'bg-emerald-900/20 border-emerald-700/30' :
            cell.status === 'critical' ? 'bg-red-900/20 border-red-700/30' :
            'bg-amber-900/20 border-amber-700/30'
          }`}
        >
          <div className="flex items-center justify-between mb-3">
            <h4 className="font-medium text-white">{cell.name}</h4>
            <span className={`px-2 py-0.5 rounded text-xs ${
              cell.status === 'healthy' ? 'bg-emerald-500/20 text-emerald-400' :
              cell.status === 'critical' ? 'bg-red-500/20 text-red-400' :
              'bg-amber-500/20 text-amber-400'
            }`}>
              {cell.status}
            </span>
          </div>

          {/* Flow visualization */}
          <div className="flex items-center gap-2 mb-3 overflow-x-auto pb-2">
            {cell.stages.map((stage: any, idx: number) => (
              <React.Fragment key={stage.machine_id}>
                <div className={`px-3 py-2 rounded border text-center ${
                  stage.is_bottleneck 
                    ? 'bg-amber-900/40 border-amber-600/50 ring-2 ring-amber-500/30' 
                    : 'bg-slate-800/50 border-slate-700/50'
                }`}>
                  <p className="text-xs text-slate-400">{stage.machine_id}</p>
                  <p className="text-sm font-medium text-white">{stage.utilization_pct.toFixed(0)}%</p>
                </div>
                {idx < cell.stages.length - 1 && (
                  <div className="text-slate-500">→</div>
                )}
              </React.Fragment>
            ))}
          </div>

          {/* Metrics */}
          <div className="grid grid-cols-4 gap-2 text-sm">
            <div>
              <p className="text-slate-500">Lead Time</p>
              <p className="text-white">{cell.lead_time_hours.toFixed(1)}h</p>
            </div>
            <div>
              <p className="text-slate-500">WIP</p>
              <p className="text-white">{cell.wip.total.toFixed(0)} un</p>
            </div>
            <div>
              <p className="text-slate-500">Throughput/dia</p>
              <p className="text-white">{cell.throughput_per_day.toFixed(0)}</p>
            </div>
            <div>
              <p className="text-slate-500">Gargalo</p>
              <p className="text-amber-400">{cell.bottleneck.machine}</p>
            </div>
          </div>
        </div>
      ))}
    </div>
  )
}

// ═══════════════════════════════════════════════════════════════════════════════════════════
// CAPACITY PROJECTION COMPONENT
// ═══════════════════════════════════════════════════════════════════════════════════════════

const CapacityProjectionView: React.FC<{ data: any }> = ({ data }) => {
  if (!data?.projections?.length) return <div className="text-slate-500">Sem dados</div>

  const maxValue = Math.max(
    ...data.projections.map((p: any) => Math.max(p.demand.units, p.capacity.units))
  )

  return (
    <div className="space-y-4">
      {/* Summary */}
      <div className="grid grid-cols-4 gap-3">
        <div className="p-3 bg-slate-800/50 rounded-lg">
          <p className="text-xs text-slate-400">Utilização Média</p>
          <p className="text-xl font-bold text-white">{data.totals.avg_utilization_pct.toFixed(0)}%</p>
        </div>
        <div className="p-3 bg-red-900/30 rounded-lg">
          <p className="text-xs text-slate-400">Meses em Risco</p>
          <p className="text-xl font-bold text-red-400">{data.gaps.months_undercapacity}</p>
        </div>
        <div className="p-3 bg-blue-900/30 rounded-lg">
          <p className="text-xs text-slate-400">Sobrecapacidade</p>
          <p className="text-xl font-bold text-blue-400">{data.gaps.months_overcapacity}</p>
        </div>
        <div className="p-3 bg-emerald-900/30 rounded-lg">
          <p className="text-xs text-slate-400">Crescimento Anual</p>
          <p className="text-xl font-bold text-emerald-400">{data.growth.annual_demand_growth_pct.toFixed(0)}%</p>
        </div>
      </div>

      {/* Bar chart */}
      <div className="flex items-end gap-2 h-48">
        {data.projections.map((p: any) => {
          const demandHeight = (p.demand.units / maxValue) * 100
          const capacityHeight = (p.capacity.units / maxValue) * 100
          
          return (
            <div key={`${p.year}-${p.month}`} className="flex-1 flex flex-col items-center">
              <div className="relative w-full h-40 flex items-end justify-center gap-0.5">
                {/* Demand bar */}
                <div
                  className={`w-3 rounded-t ${
                    p.status === 'undercapacity' ? 'bg-red-500' :
                    p.status === 'overcapacity' ? 'bg-blue-500' :
                    'bg-emerald-500'
                  }`}
                  style={{ height: `${demandHeight}%` }}
                  title={`Procura: ${p.demand.units.toFixed(0)}`}
                />
                {/* Capacity line */}
                <div
                  className="w-3 border-2 border-dashed border-slate-400 rounded-t bg-transparent"
                  style={{ height: `${capacityHeight}%` }}
                  title={`Capacidade: ${p.capacity.units.toFixed(0)}`}
                />
              </div>
              <p className="text-xs text-slate-500 mt-1">{p.month_name.substring(0, 3)}</p>
            </div>
          )
        })}
      </div>

      {/* Legend */}
      <div className="flex items-center gap-6 text-xs text-slate-400">
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 bg-emerald-500 rounded" />
          <span>Procura (equilibrado)</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 bg-red-500 rounded" />
          <span>Procura (subcapacidade)</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 border-2 border-dashed border-slate-400 rounded" />
          <span>Capacidade</span>
        </div>
      </div>

      {/* Recommendations */}
      {data.recommendations?.length > 0 && (
        <div className="p-4 bg-amber-900/20 border border-amber-700/30 rounded-lg">
          <h4 className="font-medium text-amber-400 mb-2">Recomendações</h4>
          <ul className="space-y-1">
            {data.recommendations.map((r: string, i: number) => (
              <li key={i} className="text-sm text-slate-300 flex items-start gap-2">
                <AlertTriangle className="w-4 h-4 mt-0.5 text-amber-500 flex-shrink-0" />
                {r}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}

// ═══════════════════════════════════════════════════════════════════════════════════════════
// MAIN PAGE
// ═══════════════════════════════════════════════════════════════════════════════════════════

const Dashboards: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'heatmap' | 'operator' | 'oee' | 'cell' | 'projection'>('heatmap')

  const { data: heatmapData } = useQuery({
    queryKey: ['heatmap'],
    queryFn: fetchHeatmap,
    enabled: activeTab === 'heatmap',
  })

  const { data: operatorData } = useQuery({
    queryKey: ['operatorDashboard'],
    queryFn: fetchOperatorDashboard,
    enabled: activeTab === 'operator',
  })

  const { data: oeeData } = useQuery({
    queryKey: ['machineOEE'],
    queryFn: fetchMachineOEE,
    enabled: activeTab === 'oee',
  })

  const { data: cellData } = useQuery({
    queryKey: ['cellPerformance'],
    queryFn: fetchCellPerformance,
    enabled: activeTab === 'cell',
  })

  const { data: projectionData } = useQuery({
    queryKey: ['capacityProjection'],
    queryFn: fetchCapacityProjection,
    enabled: activeTab === 'projection',
  })

  const tabs = [
    { id: 'heatmap', label: 'Heatmap', icon: <Grid className="w-4 h-4" /> },
    { id: 'operator', label: 'Operadores', icon: <Users className="w-4 h-4" /> },
    { id: 'oee', label: 'OEE Máquinas', icon: <Settings className="w-4 h-4" /> },
    { id: 'cell', label: 'Células', icon: <GitBranch className="w-4 h-4" /> },
    { id: 'projection', label: 'Projeção Anual', icon: <TrendingUp className="w-4 h-4" /> },
  ]

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-white">📈 Dashboards Avançados</h1>
        <p className="text-slate-400 mt-1">
          Visualizações de utilização, OEE, células e projeções
        </p>
      </div>

      {/* Tabs */}
      <div className="flex gap-2 p-1 bg-slate-800/50 rounded-lg overflow-x-auto">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id as typeof activeTab)}
            className={`px-4 py-2 rounded-md text-sm font-medium transition-all flex items-center gap-2 whitespace-nowrap ${
              activeTab === tab.id
                ? 'bg-cyan-600 text-white'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            {tab.icon}
            {tab.label}
          </button>
        ))}
      </div>

      {/* Content */}
      <AnimatePresence mode="wait">
        <motion.div
          key={activeTab}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -20 }}
          className="p-6 bg-slate-800/30 border border-slate-700/50 rounded-xl"
        >
          {activeTab === 'heatmap' && (
            <>
              <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                <Grid className="w-5 h-5 text-cyan-400" />
                Heatmap de Utilização Horária
              </h2>
              <UtilizationHeatmap data={heatmapData} />
            </>
          )}

          {activeTab === 'operator' && (
            <>
              <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                <Users className="w-5 h-5 text-cyan-400" />
                Dashboard de Operadores
              </h2>
              <OperatorDashboardView data={operatorData} />
            </>
          )}

          {activeTab === 'oee' && (
            <>
              <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                <Settings className="w-5 h-5 text-cyan-400" />
                Dashboard de Máquinas & OEE
              </h2>
              <OEEDashboardView data={oeeData} />
            </>
          )}

          {activeTab === 'cell' && (
            <>
              <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                <GitBranch className="w-5 h-5 text-cyan-400" />
                Performance de Células Encadeadas
              </h2>
              <CellPerformanceView data={cellData} />
            </>
          )}

          {activeTab === 'projection' && (
            <>
              <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                <TrendingUp className="w-5 h-5 text-cyan-400" />
                Projeção Anual: Capacidade vs Procura
              </h2>
              <CapacityProjectionView data={projectionData} />
            </>
          )}
        </motion.div>
      </AnimatePresence>
    </div>
  )
}

export default Dashboards



