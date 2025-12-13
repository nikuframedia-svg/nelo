/**
 * ════════════════════════════════════════════════════════════════════════════════════════════
 * PRODPLAN 4.0 — ADVANCED PLANNING PAGE
 * ════════════════════════════════════════════════════════════════════════════════════════════
 *
 * Interface para planeamento avançado:
 * - Convencional vs Encadeado (Flow Shop)
 * - Comparação de modos
 * - Configuração de cadeias de máquinas
 * - Visualização de resultados
 */

import React, { useState, useMemo } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Settings,
  Play,
  GitBranch,
  Layers,
  ArrowRight,
  Clock,
  TrendingDown,
  TrendingUp,
  AlertTriangle,
  CheckCircle,
  RefreshCw,
  Plus,
  X,
  Zap,
  Target,
  BarChart3,
  Calendar,
} from 'lucide-react'
import { toast } from 'sonner'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000'

// ═══════════════════════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════════════════════

interface PlanningMode {
  id: string
  name: string
  description: string
  use_case: string
  dispatching_rules?: string[]
  solvers?: string[]
  features?: string[]
}

interface PlanningResult {
  mode: string
  makespan_hours: number
  tardiness_hours: number
  setup_hours: number
  throughput_units: number
  utilization_pct: number
  bottleneck_machine: string | null
  bottleneck_utilization: number
  machine_metrics: Record<string, Record<string, number>>
  solver_status: string
  plan_rows: number
  message: string
}

interface ComparisonResult {
  baseline: {
    mode: string
    makespan_hours: number
    tardiness_hours: number
    setup_hours: number
  }
  scenario: {
    mode: string
    makespan_hours: number
    tardiness_hours: number
    setup_hours: number
  }
  deltas: {
    makespan_hours: number
    makespan_pct: number
    tardiness_hours: number
    tardiness_pct: number
    setup_hours: number
    setup_pct: number
  }
  recommendation: {
    mode: string
    reason: string
  }
  message: string
}

// ═══════════════════════════════════════════════════════════════════════════════════════════
// API FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════════════════════

async function fetchPlanningModes(): Promise<{ available_modes: PlanningMode[] }> {
  const res = await fetch(`${API_BASE_URL}/planning/modes`)
  if (!res.ok) throw new Error('Failed to fetch planning modes')
  return res.json()
}

async function executePlanningConventional(config: {
  dispatching_rule: string
  group_by_setup_family: boolean
}): Promise<PlanningResult> {
  const res = await fetch(`${API_BASE_URL}/planning/conventional`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  })
  if (!res.ok) throw new Error('Failed to execute conventional planning')
  return res.json()
}

async function executePlanningChained(config: {
  chains: string[][]
  buffers: Record<string, number>
  default_buffer_min: number
  synchronize_flow: boolean
}): Promise<PlanningResult> {
  const res = await fetch(`${API_BASE_URL}/planning/chained`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  })
  if (!res.ok) throw new Error('Failed to execute chained planning')
  return res.json()
}

async function comparePlanningModes(config: {
  modes: string[]
  chained_config?: {
    chains: string[][]
    buffers: Record<string, number>
    default_buffer_min: number
  }
}): Promise<ComparisonResult> {
  const res = await fetch(`${API_BASE_URL}/planning/compare`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  })
  if (!res.ok) throw new Error('Failed to compare planning modes')
  return res.json()
}

async function fetchMachines(): Promise<{ machine_id: string }[]> {
  const res = await fetch(`${API_BASE_URL}/machines`)
  if (!res.ok) throw new Error('Failed to fetch machines')
  return res.json()
}

// ═══════════════════════════════════════════════════════════════════════════════════════════
// COMPONENTS
// ═══════════════════════════════════════════════════════════════════════════════════════════

const MetricCard: React.FC<{
  title: string
  value: string | number
  subtitle?: string
  icon: React.ReactNode
  trend?: 'up' | 'down' | 'neutral'
  highlight?: boolean
}> = ({ title, value, subtitle, icon, trend, highlight }) => (
  <div className={`p-4 rounded-lg border ${
    highlight 
      ? 'bg-gradient-to-br from-cyan-900/40 to-blue-900/40 border-cyan-500/50' 
      : 'bg-slate-800/50 border-slate-700/50'
  }`}>
    <div className="flex items-center justify-between mb-2">
      <span className="text-slate-400 text-sm">{title}</span>
      <span className="text-slate-500">{icon}</span>
    </div>
    <div className="flex items-end gap-2">
      <span className="text-2xl font-bold text-white">{value}</span>
      {trend && (
        <span className={`text-sm ${
          trend === 'down' ? 'text-emerald-400' : trend === 'up' ? 'text-red-400' : 'text-slate-400'
        }`}>
          {trend === 'down' ? <TrendingDown className="w-4 h-4" /> : 
           trend === 'up' ? <TrendingUp className="w-4 h-4" /> : null}
        </span>
      )}
    </div>
    {subtitle && <p className="text-xs text-slate-500 mt-1">{subtitle}</p>}
  </div>
)

const ChainBuilder: React.FC<{
  machines: string[]
  chains: string[][]
  onChange: (chains: string[][]) => void
}> = ({ machines, chains, onChange }) => {
  const addChain = () => {
    onChange([...chains, []])
  }
  
  const removeChain = (index: number) => {
    onChange(chains.filter((_, i) => i !== index))
  }
  
  const addMachineToChain = (chainIndex: number, machine: string) => {
    const newChains = [...chains]
    if (!newChains[chainIndex].includes(machine)) {
      newChains[chainIndex] = [...newChains[chainIndex], machine]
      onChange(newChains)
    }
  }
  
  const removeMachineFromChain = (chainIndex: number, machineIndex: number) => {
    const newChains = [...chains]
    newChains[chainIndex] = newChains[chainIndex].filter((_, i) => i !== machineIndex)
    onChange(newChains)
  }
  
  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium text-slate-300">Cadeias de Máquinas</h3>
        <button
          onClick={addChain}
          className="flex items-center gap-1 px-2 py-1 text-xs bg-cyan-600 hover:bg-cyan-500 text-white rounded transition-colors"
        >
          <Plus className="w-3 h-3" />
          Nova Cadeia
        </button>
      </div>
      
      {chains.length === 0 ? (
        <div className="text-center py-8 text-slate-500 border border-dashed border-slate-700 rounded-lg">
          <GitBranch className="w-8 h-8 mx-auto mb-2 opacity-50" />
          <p className="text-sm">Nenhuma cadeia definida</p>
          <p className="text-xs">Clique em "Nova Cadeia" para começar</p>
        </div>
      ) : (
        <div className="space-y-3">
          {chains.map((chain, chainIndex) => (
            <div key={chainIndex} className="p-3 bg-slate-900/50 border border-slate-700/50 rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs text-slate-400">Cadeia {chainIndex + 1}</span>
                <button
                  onClick={() => removeChain(chainIndex)}
                  className="text-slate-500 hover:text-red-400 transition-colors"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
              
              <div className="flex items-center gap-2 flex-wrap mb-2">
                {chain.map((machine, machineIndex) => (
                  <React.Fragment key={machine}>
                    <div className="flex items-center gap-1 px-2 py-1 bg-cyan-900/50 border border-cyan-700/50 rounded text-sm">
                      <span className="text-cyan-300">{machine}</span>
                      <button
                        onClick={() => removeMachineFromChain(chainIndex, machineIndex)}
                        className="text-cyan-500 hover:text-red-400"
                      >
                        <X className="w-3 h-3" />
                      </button>
                    </div>
                    {machineIndex < chain.length - 1 && (
                      <ArrowRight className="w-4 h-4 text-slate-500" />
                    )}
                  </React.Fragment>
                ))}
              </div>
              
              <select
                className="w-full px-2 py-1 bg-slate-800 border border-slate-700 rounded text-sm text-white"
                value=""
                onChange={(e) => {
                  if (e.target.value) {
                    addMachineToChain(chainIndex, e.target.value)
                  }
                }}
              >
                <option value="">+ Adicionar máquina...</option>
                {machines
                  .filter((m) => !chain.includes(m))
                  .map((m) => (
                    <option key={m} value={m}>{m}</option>
                  ))}
              </select>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

const ComparisonView: React.FC<{ comparison: ComparisonResult }> = ({ comparison }) => {
  const isImprovement = (delta: number) => delta < 0
  
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-4"
    >
      {/* Recommendation banner */}
      <div className={`p-4 rounded-lg border ${
        comparison.recommendation.mode === 'chained'
          ? 'bg-gradient-to-r from-emerald-900/30 to-cyan-900/30 border-emerald-500/30'
          : 'bg-gradient-to-r from-blue-900/30 to-slate-900/30 border-blue-500/30'
      }`}>
        <div className="flex items-center gap-3">
          <CheckCircle className="w-6 h-6 text-emerald-400" />
          <div>
            <p className="font-medium text-white">
              Recomendação: {comparison.recommendation.mode === 'chained' ? 'Planeamento Encadeado' : 'Planeamento Convencional'}
            </p>
            <p className="text-sm text-slate-300">{comparison.recommendation.reason}</p>
          </div>
        </div>
      </div>
      
      {/* Comparison grid */}
      <div className="grid grid-cols-3 gap-4">
        {/* Baseline */}
        <div className="p-4 bg-slate-800/50 border border-slate-700/50 rounded-lg">
          <h4 className="text-sm font-medium text-slate-400 mb-3">Convencional</h4>
          <div className="space-y-2">
            <div className="flex justify-between">
              <span className="text-slate-500 text-sm">Makespan</span>
              <span className="text-white font-medium">{comparison.baseline.makespan_hours.toFixed(1)}h</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500 text-sm">Atrasos</span>
              <span className="text-white font-medium">{comparison.baseline.tardiness_hours.toFixed(1)}h</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500 text-sm">Setups</span>
              <span className="text-white font-medium">{comparison.baseline.setup_hours.toFixed(1)}h</span>
            </div>
          </div>
        </div>
        
        {/* Deltas */}
        <div className="p-4 bg-slate-900/50 border border-cyan-700/30 rounded-lg">
          <h4 className="text-sm font-medium text-cyan-400 mb-3">Diferença</h4>
          <div className="space-y-2">
            <div className="flex justify-between items-center">
              <span className="text-slate-500 text-sm">Makespan</span>
              <span className={`font-medium ${isImprovement(comparison.deltas.makespan_pct) ? 'text-emerald-400' : 'text-red-400'}`}>
                {comparison.deltas.makespan_pct > 0 ? '+' : ''}{comparison.deltas.makespan_pct.toFixed(1)}%
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-slate-500 text-sm">Atrasos</span>
              <span className={`font-medium ${isImprovement(comparison.deltas.tardiness_pct) ? 'text-emerald-400' : 'text-red-400'}`}>
                {comparison.deltas.tardiness_pct > 0 ? '+' : ''}{comparison.deltas.tardiness_pct.toFixed(1)}%
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-slate-500 text-sm">Setups</span>
              <span className={`font-medium ${isImprovement(comparison.deltas.setup_pct) ? 'text-emerald-400' : 'text-red-400'}`}>
                {comparison.deltas.setup_pct > 0 ? '+' : ''}{comparison.deltas.setup_pct.toFixed(1)}%
              </span>
            </div>
          </div>
        </div>
        
        {/* Scenario */}
        <div className="p-4 bg-slate-800/50 border border-slate-700/50 rounded-lg">
          <h4 className="text-sm font-medium text-slate-400 mb-3">Encadeado</h4>
          <div className="space-y-2">
            <div className="flex justify-between">
              <span className="text-slate-500 text-sm">Makespan</span>
              <span className="text-white font-medium">{comparison.scenario.makespan_hours.toFixed(1)}h</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500 text-sm">Atrasos</span>
              <span className="text-white font-medium">{comparison.scenario.tardiness_hours.toFixed(1)}h</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500 text-sm">Setups</span>
              <span className="text-white font-medium">{comparison.scenario.setup_hours.toFixed(1)}h</span>
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  )
}

// ═══════════════════════════════════════════════════════════════════════════════════════════
// MAIN PAGE
// ═══════════════════════════════════════════════════════════════════════════════════════════

const AdvancedPlanning: React.FC = () => {
  const queryClient = useQueryClient()
  
  // State
  const [activeMode, setActiveMode] = useState<'conventional' | 'chained'>('conventional')
  const [dispatchingRule, setDispatchingRule] = useState('EDD')
  const [chains, setChains] = useState<string[][]>([])
  const [defaultBuffer, setDefaultBuffer] = useState(30)
  const [lastResult, setLastResult] = useState<PlanningResult | null>(null)
  const [comparisonResult, setComparisonResult] = useState<ComparisonResult | null>(null)
  
  // Queries
  const { data: modesData } = useQuery({
    queryKey: ['planningModes'],
    queryFn: fetchPlanningModes,
  })
  
  const { data: machinesData } = useQuery({
    queryKey: ['machines'],
    queryFn: fetchMachines,
  })
  
  const machines = useMemo(() => 
    machinesData?.map(m => m.machine_id) ?? [],
    [machinesData]
  )
  
  // Mutations
  const conventionalMutation = useMutation({
    mutationFn: executePlanningConventional,
    onSuccess: (data) => {
      setLastResult(data)
      toast.success('Planeamento convencional concluído!')
      queryClient.invalidateQueries({ queryKey: ['plan'] })
    },
    onError: () => toast.error('Erro no planeamento convencional'),
  })
  
  const chainedMutation = useMutation({
    mutationFn: executePlanningChained,
    onSuccess: (data) => {
      setLastResult(data)
      toast.success('Planeamento encadeado concluído!')
      queryClient.invalidateQueries({ queryKey: ['plan'] })
    },
    onError: () => toast.error('Erro no planeamento encadeado'),
  })
  
  const compareMutation = useMutation({
    mutationFn: comparePlanningModes,
    onSuccess: (data) => {
      setComparisonResult(data)
      toast.success('Comparação concluída!')
    },
    onError: () => toast.error('Erro na comparação'),
  })
  
  // Handlers
  const handleExecute = () => {
    if (activeMode === 'conventional') {
      conventionalMutation.mutate({
        dispatching_rule: dispatchingRule,
        group_by_setup_family: true,
      })
    } else {
      if (chains.length === 0 || chains.every(c => c.length < 2)) {
        toast.error('Configure pelo menos uma cadeia com 2+ máquinas')
        return
      }
      chainedMutation.mutate({
        chains,
        buffers: {},
        default_buffer_min: defaultBuffer,
        synchronize_flow: true,
      })
    }
  }
  
  const handleCompare = () => {
    if (chains.length === 0 || chains.every(c => c.length < 2)) {
      toast.error('Configure pelo menos uma cadeia para comparar')
      return
    }
    compareMutation.mutate({
      modes: ['conventional', 'chained'],
      chained_config: {
        chains,
        buffers: {},
        default_buffer_min: defaultBuffer,
      },
    })
  }
  
  const isLoading = conventionalMutation.isPending || chainedMutation.isPending || compareMutation.isPending
  
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white">⚙️ Planeamento Avançado</h1>
          <p className="text-slate-400 mt-1">
            Convencional vs. Encadeado (Flow Shop)
          </p>
        </div>
      </div>
      
      {/* Mode selector */}
      <div className="flex gap-2 p-1 bg-slate-800/50 rounded-lg w-fit">
        <button
          onClick={() => setActiveMode('conventional')}
          className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${
            activeMode === 'conventional'
              ? 'bg-cyan-600 text-white'
              : 'text-slate-400 hover:text-white'
          }`}
        >
          <div className="flex items-center gap-2">
            <Layers className="w-4 h-4" />
            Convencional
          </div>
        </button>
        <button
          onClick={() => setActiveMode('chained')}
          className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${
            activeMode === 'chained'
              ? 'bg-cyan-600 text-white'
              : 'text-slate-400 hover:text-white'
          }`}
        >
          <div className="flex items-center gap-2">
            <GitBranch className="w-4 h-4" />
            Encadeado
          </div>
        </button>
      </div>
      
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Configuration panel */}
        <div className="p-6 bg-slate-800/30 border border-slate-700/50 rounded-xl">
          <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <Settings className="w-5 h-5 text-cyan-400" />
            Configuração
          </h2>
          
          <AnimatePresence mode="wait">
            {activeMode === 'conventional' ? (
              <motion.div
                key="conventional"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
                className="space-y-4"
              >
                <div>
                  <label className="block text-sm text-slate-400 mb-2">
                    Regra de Dispatching
                  </label>
                  <select
                    value={dispatchingRule}
                    onChange={(e) => setDispatchingRule(e.target.value)}
                    className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-white"
                  >
                    <option value="EDD">EDD - Earliest Due Date</option>
                    <option value="SPT">SPT - Shortest Processing Time</option>
                    <option value="FIFO">FIFO - First In First Out</option>
                    <option value="CR">CR - Critical Ratio</option>
                    <option value="WSPT">WSPT - Weighted SPT</option>
                  </select>
                </div>
                
                <div className="p-4 bg-slate-900/50 rounded-lg border border-slate-700/50">
                  <h4 className="text-sm font-medium text-slate-300 mb-2">Descrição</h4>
                  <p className="text-sm text-slate-500">
                    {dispatchingRule === 'EDD' && 'Prioriza encomendas com data de entrega mais próxima.'}
                    {dispatchingRule === 'SPT' && 'Prioriza operações com menor tempo de processamento.'}
                    {dispatchingRule === 'FIFO' && 'Processa na ordem de chegada.'}
                    {dispatchingRule === 'CR' && 'Rácio entre folga restante e tempo de processamento.'}
                    {dispatchingRule === 'WSPT' && 'SPT ponderado pela prioridade da encomenda.'}
                  </p>
                </div>
              </motion.div>
            ) : (
              <motion.div
                key="chained"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
                className="space-y-4"
              >
                <ChainBuilder
                  machines={machines}
                  chains={chains}
                  onChange={setChains}
                />
                
                <div>
                  <label className="block text-sm text-slate-400 mb-2">
                    Buffer entre etapas (minutos)
                  </label>
                  <input
                    type="number"
                    value={defaultBuffer}
                    onChange={(e) => setDefaultBuffer(Number(e.target.value))}
                    className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-white"
                    min={0}
                    max={120}
                  />
                  <p className="text-xs text-slate-500 mt-1">
                    Tempo de espera/transferência entre máquinas consecutivas
                  </p>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
          
          {/* Action buttons */}
          <div className="flex gap-3 mt-6">
            <button
              onClick={handleExecute}
              disabled={isLoading}
              className="flex-1 flex items-center justify-center gap-2 px-4 py-3 bg-cyan-600 hover:bg-cyan-500 disabled:bg-slate-700 text-white rounded-lg font-medium transition-colors"
            >
              {isLoading ? (
                <RefreshCw className="w-5 h-5 animate-spin" />
              ) : (
                <Play className="w-5 h-5" />
              )}
              Executar {activeMode === 'conventional' ? 'Convencional' : 'Encadeado'}
            </button>
            
            {activeMode === 'chained' && chains.length > 0 && (
              <button
                onClick={handleCompare}
                disabled={isLoading}
                className="flex items-center gap-2 px-4 py-3 bg-slate-700 hover:bg-slate-600 disabled:bg-slate-800 text-white rounded-lg font-medium transition-colors"
              >
                <BarChart3 className="w-5 h-5" />
                Comparar
              </button>
            )}
          </div>
        </div>
        
        {/* Results panel */}
        <div className="p-6 bg-slate-800/30 border border-slate-700/50 rounded-xl">
          <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <Target className="w-5 h-5 text-cyan-400" />
            Resultados
          </h2>
          
          {lastResult ? (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-3">
                <MetricCard
                  title="Makespan"
                  value={`${lastResult.makespan_hours.toFixed(1)}h`}
                  icon={<Clock className="w-4 h-4" />}
                />
                <MetricCard
                  title="Throughput"
                  value={`${Math.round(lastResult.throughput_units)}`}
                  subtitle="unidades"
                  icon={<Zap className="w-4 h-4" />}
                />
                <MetricCard
                  title="Setup Total"
                  value={`${(lastResult.setup_hours ?? 0).toFixed(1)}h`}
                  icon={<Settings className="w-4 h-4" />}
                />
                <MetricCard
                  title="Utilização"
                  value={`${(lastResult.utilization_pct ?? 0).toFixed(0)}%`}
                  icon={<BarChart3 className="w-4 h-4" />}
                  highlight={(lastResult.utilization_pct ?? 0) > 80}
                />
              </div>
              
              {lastResult.bottleneck_machine && (
                <div className="p-3 bg-amber-900/20 border border-amber-700/30 rounded-lg">
                  <div className="flex items-center gap-2 text-amber-400">
                    <AlertTriangle className="w-4 h-4" />
                    <span className="text-sm font-medium">Gargalo: {lastResult.bottleneck_machine}</span>
                    <span className="text-xs text-amber-500">
                      ({(lastResult.bottleneck_utilization ?? 0).toFixed(0)}% utilização)
                    </span>
                  </div>
                </div>
              )}
              
              <div className="p-3 bg-slate-900/50 rounded-lg">
                <p className="text-sm text-slate-300">{lastResult.message}</p>
              </div>
            </div>
          ) : (
            <div className="text-center py-12 text-slate-500">
              <Calendar className="w-12 h-12 mx-auto mb-3 opacity-50" />
              <p>Execute um planeamento para ver os resultados</p>
            </div>
          )}
        </div>
      </div>
      
      {/* Comparison results */}
      {comparisonResult && (
        <div className="p-6 bg-slate-800/30 border border-slate-700/50 rounded-xl">
          <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <BarChart3 className="w-5 h-5 text-cyan-400" />
            Comparação: Convencional vs. Encadeado
          </h2>
          <ComparisonView comparison={comparisonResult} />
        </div>
      )}
      
      {/* Planning modes info */}
      {modesData && (
        <div className="p-6 bg-slate-800/30 border border-slate-700/50 rounded-xl">
          <h2 className="text-lg font-semibold text-white mb-4">
            Modos de Planeamento Disponíveis
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {modesData.available_modes.map((mode) => (
              <div
                key={mode.id}
                className="p-4 bg-slate-900/50 border border-slate-700/50 rounded-lg"
              >
                <h3 className="font-medium text-white mb-1">{mode.name}</h3>
                <p className="text-xs text-slate-400 mb-2">{mode.description}</p>
                <p className="text-xs text-cyan-400">{mode.use_case}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

export default AdvancedPlanning



