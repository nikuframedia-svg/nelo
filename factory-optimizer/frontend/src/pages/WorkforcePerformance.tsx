/**
 * ════════════════════════════════════════════════════════════════════════════════════════════
 * PRODPLAN 4.0 — WORKFORCE PERFORMANCE VIEW
 * ════════════════════════════════════════════════════════════════════════════════════════════
 *
 * Análise e planeamento de performance de colaboradores.
 * - Métricas de produtividade, eficiência, saturação
 * - Skill scores e curvas de aprendizagem
 * - Previsão de performance (ARIMA)
 * - Recomendações de alocação (MILP)
 */

import React, { useState, useMemo } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Users,
  TrendingUp,
  Activity,
  Award,
  AlertTriangle,
  ChevronDown,
  ChevronUp,
  RefreshCw,
  Target,
  Clock,
  Zap,
  BarChart3,
  LineChart,
} from 'lucide-react'
import { toast } from 'sonner'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000'

// ═══════════════════════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════════════════════

interface WorkerMetrics {
  worker_id: string
  worker_name: string | null
  productivity: number
  efficiency: number
  saturation: number
  skill_score: number
  total_time_hours: number
  total_units: number
  total_operations: number
  snr_performance: number
  snr_level: string
  consistency_score: number
  performance_level: string
  saturation_level: string
  learning_curve: {
    a_asymptote: number
    b_initial_gap: number
    c_learning_rate: number
    r_squared: number
    time_to_90pct_days: number | null
  } | null
}

interface WorkerPerformance {
  worker_id: string
  metrics: WorkerMetrics
  productivity_history: number[]
  dates: string[]
  recommendations: string[]
}

interface PerformanceResponse {
  workers: WorkerPerformance[]
  total_workers: number
  global_metrics: {
    avg_productivity: number
    avg_efficiency: number
    avg_skill_score: number
    avg_snr: number
  }
  performance_distribution: Record<string, number>
}

interface ForecastPoint {
  date: string
  value: number
  lower_bound: number
  upper_bound: number
}

interface WorkerForecast {
  worker_id: string
  model_type: string
  forecast_values: ForecastPoint[]
  forecast_mean: number
  forecast_trend: string
  mape: number
  snr_forecast: number
  snr_level: string
  confidence_score: number
}

interface ForecastResponse {
  forecasts: WorkerForecast[]
  horizon_days: number
  model_type: string
}

// ═══════════════════════════════════════════════════════════════════════════════════════════
// API FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════════════════════

async function fetchPerformance(): Promise<PerformanceResponse> {
  const res = await fetch(`${API_BASE_URL}/workforce/performance`)
  if (!res.ok) throw new Error('Erro ao carregar performance')
  return res.json()
}

async function fetchForecast(horizonDays: number, modelType: string): Promise<ForecastResponse> {
  const res = await fetch(`${API_BASE_URL}/workforce/forecast`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ horizon_days: horizonDays, model_type: modelType }),
  })
  if (!res.ok) throw new Error('Erro ao gerar previsão')
  return res.json()
}

async function fetchAssignment(useMilp: boolean): Promise<any> {
  const res = await fetch(`${API_BASE_URL}/workforce/assign`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ use_milp: useMilp }),
  })
  if (!res.ok) throw new Error('Erro ao otimizar alocação')
  return res.json()
}

// ═══════════════════════════════════════════════════════════════════════════════════════════
// COMPONENTS
// ═══════════════════════════════════════════════════════════════════════════════════════════

// Performance level badge
const PerformanceBadge: React.FC<{ level: string }> = ({ level }) => {
  const colors: Record<string, string> = {
    excellent: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30',
    good: 'bg-green-500/20 text-green-300 border-green-500/30',
    average: 'bg-slate-500/20 text-slate-300 border-slate-500/30',
    below_average: 'bg-amber-500/20 text-amber-300 border-amber-500/30',
    needs_improvement: 'bg-red-500/20 text-red-300 border-red-500/30',
  }
  
  const labels: Record<string, string> = {
    excellent: 'Excelente',
    good: 'Bom',
    average: 'Médio',
    below_average: 'Abaixo da média',
    needs_improvement: 'Necessita melhoria',
  }
  
  return (
    <span className={`px-2 py-0.5 text-xs font-medium rounded border ${colors[level] || colors.average}`}>
      {labels[level] || level}
    </span>
  )
}

// SNR badge
const SNRBadge: React.FC<{ level: string; value: number }> = ({ level, value }) => {
  const colors: Record<string, string> = {
    EXCELLENT: 'text-emerald-400',
    GOOD: 'text-green-400',
    FAIR: 'text-amber-400',
    POOR: 'text-red-400',
  }
  
  return (
    <span className={`text-xs font-mono ${colors[level] || 'text-slate-400'}`}>
      SNR: {value.toFixed(1)}
    </span>
  )
}

// KPI Card
const KPICard: React.FC<{
  title: string
  value: string | number
  subtitle?: string
  icon: React.ReactNode
  highlight?: boolean
}> = ({ title, value, subtitle, icon, highlight }) => {
  return (
    <div className={`
      p-4 rounded-lg border transition-all
      ${highlight 
        ? 'bg-gradient-to-br from-cyan-500/20 to-blue-600/20 border-cyan-500/30' 
        : 'bg-slate-800/50 border-slate-700/50'}
    `}>
      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs text-slate-400 uppercase tracking-wider">{title}</p>
          <p className="text-2xl font-bold text-white mt-1">{value}</p>
          {subtitle && <p className="text-xs text-slate-500 mt-0.5">{subtitle}</p>}
        </div>
        <div className={`p-2 rounded-lg ${highlight ? 'bg-cyan-500/20' : 'bg-slate-700/50'}`}>
          {icon}
        </div>
      </div>
    </div>
  )
}

// Mini sparkline chart
const Sparkline: React.FC<{ data: number[]; color?: string }> = ({ data, color = '#22d3ee' }) => {
  if (!data || data.length < 2) return null
  
  const max = Math.max(...data)
  const min = Math.min(...data)
  const range = max - min || 1
  
  const points = data.map((v, i) => {
    const x = (i / (data.length - 1)) * 100
    const y = 100 - ((v - min) / range) * 100
    return `${x},${y}`
  }).join(' ')
  
  return (
    <svg className="w-24 h-8" viewBox="0 0 100 100" preserveAspectRatio="none">
      <polyline
        fill="none"
        stroke={color}
        strokeWidth="3"
        points={points}
      />
    </svg>
  )
}

// Worker row component
const WorkerRow: React.FC<{
  worker: WorkerPerformance
  expanded: boolean
  onToggle: () => void
  forecast?: WorkerForecast
}> = ({ worker, expanded, onToggle, forecast }) => {
  const m = worker.metrics
  
  return (
    <motion.div
      layout
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="border border-slate-700/50 rounded-lg overflow-hidden mb-2"
    >
      {/* Header row */}
      <div
        onClick={onToggle}
        className={`
          p-4 cursor-pointer transition-colors
          ${expanded ? 'bg-slate-700/30' : 'bg-slate-800/30 hover:bg-slate-700/20'}
        `}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="w-10 h-10 rounded-full bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center text-white font-bold">
              {worker.worker_id.slice(-2)}
            </div>
            <div>
              <h3 className="font-semibold text-white">
                {m.worker_name || worker.worker_id}
              </h3>
              <p className="text-xs text-slate-400">
                {m.total_operations} operações • {m.total_time_hours.toFixed(1)}h
              </p>
            </div>
          </div>
          
          <div className="flex items-center gap-6">
            <div className="text-right hidden md:block">
              <p className="text-sm font-medium text-white">{m.productivity.toFixed(1)}</p>
              <p className="text-xs text-slate-400">unid/hora</p>
            </div>
            
            <div className="text-right hidden md:block">
              <p className="text-sm font-medium text-white">{(m.efficiency * 100).toFixed(0)}%</p>
              <p className="text-xs text-slate-400">eficiência</p>
            </div>
            
            <div className="hidden sm:block">
              <Sparkline 
                data={worker.productivity_history} 
                color={m.performance_level === 'excellent' ? '#10b981' : 
                       m.performance_level === 'needs_improvement' ? '#ef4444' : '#22d3ee'}
              />
            </div>
            
            <PerformanceBadge level={m.performance_level} />
            
            {expanded ? (
              <ChevronUp className="w-5 h-5 text-slate-400" />
            ) : (
              <ChevronDown className="w-5 h-5 text-slate-400" />
            )}
          </div>
        </div>
      </div>
      
      {/* Expanded content */}
      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="border-t border-slate-700/50"
          >
            <div className="p-4 bg-slate-900/50 space-y-4">
              {/* Metrics grid */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div>
                  <p className="text-xs text-slate-500">Produtividade</p>
                  <p className="text-lg font-semibold text-white">{m.productivity.toFixed(2)}</p>
                  <p className="text-xs text-slate-400">unidades/hora</p>
                </div>
                <div>
                  <p className="text-xs text-slate-500">Eficiência Relativa</p>
                  <p className={`text-lg font-semibold ${m.efficiency >= 1 ? 'text-emerald-400' : 'text-amber-400'}`}>
                    {(m.efficiency * 100).toFixed(0)}%
                  </p>
                  <p className="text-xs text-slate-400">vs média</p>
                </div>
                <div>
                  <p className="text-xs text-slate-500">Saturação</p>
                  <p className={`text-lg font-semibold ${
                    m.saturation_level === 'optimal' ? 'text-emerald-400' :
                    m.saturation_level === 'overloaded' ? 'text-red-400' : 'text-amber-400'
                  }`}>
                    {(m.saturation * 100).toFixed(0)}%
                  </p>
                  <p className="text-xs text-slate-400">{m.saturation_level}</p>
                </div>
                <div>
                  <p className="text-xs text-slate-500">Skill Score</p>
                  <p className="text-lg font-semibold text-cyan-400">
                    {(m.skill_score * 100).toFixed(0)}%
                  </p>
                  <p className="text-xs text-slate-400">qualificação</p>
                </div>
              </div>
              
              {/* SNR and Learning Curve */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-slate-800/50 rounded p-3">
                  <h4 className="text-sm font-medium text-slate-300 mb-2 flex items-center gap-2">
                    <Activity className="w-4 h-4" />
                    Consistência (SNR)
                  </h4>
                  <div className="flex items-center justify-between">
                    <SNRBadge level={m.snr_level} value={m.snr_performance} />
                    <span className="text-sm text-slate-400">
                      Confiança: {(m.consistency_score * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
                
                {m.learning_curve && (
                  <div className="bg-slate-800/50 rounded p-3">
                    <h4 className="text-sm font-medium text-slate-300 mb-2 flex items-center gap-2">
                      <LineChart className="w-4 h-4" />
                      Curva de Aprendizagem
                    </h4>
                    <div className="text-xs space-y-1">
                      <p>
                        <span className="text-slate-500">Máximo:</span>{' '}
                        <span className="text-white">{m.learning_curve.a_asymptote.toFixed(1)} unid/h</span>
                      </p>
                      <p>
                        <span className="text-slate-500">R²:</span>{' '}
                        <span className="text-white">{(m.learning_curve.r_squared * 100).toFixed(0)}%</span>
                      </p>
                      {m.learning_curve.time_to_90pct_days && (
                        <p>
                          <span className="text-slate-500">Tempo até 90%:</span>{' '}
                          <span className="text-cyan-400">{m.learning_curve.time_to_90pct_days.toFixed(0)} dias</span>
                        </p>
                      )}
                    </div>
                  </div>
                )}
              </div>
              
              {/* Forecast */}
              {forecast && (
                <div className="bg-slate-800/50 rounded p-3">
                  <h4 className="text-sm font-medium text-slate-300 mb-2 flex items-center gap-2">
                    <TrendingUp className="w-4 h-4" />
                    Previsão ({forecast.model_type})
                  </h4>
                  <div className="flex items-center gap-4 text-sm">
                    <span className="text-slate-400">
                      Média esperada: <span className="text-white font-medium">{forecast.forecast_mean.toFixed(1)}</span>
                    </span>
                    <span className={`${
                      forecast.forecast_trend === 'increasing' ? 'text-emerald-400' :
                      forecast.forecast_trend === 'decreasing' ? 'text-red-400' : 'text-slate-400'
                    }`}>
                      {forecast.forecast_trend === 'increasing' ? '↗ A subir' :
                       forecast.forecast_trend === 'decreasing' ? '↘ A descer' : '→ Estável'}
                    </span>
                    <span className="text-slate-400">
                      MAPE: {forecast.mape.toFixed(1)}%
                    </span>
                  </div>
                </div>
              )}
              
              {/* Recommendations */}
              {worker.recommendations.length > 0 && (
                <div className="bg-amber-900/20 border border-amber-700/30 rounded p-3">
                  <h4 className="text-sm font-medium text-amber-300 mb-2 flex items-center gap-2">
                    <AlertTriangle className="w-4 h-4" />
                    Recomendações
                  </h4>
                  <ul className="text-sm text-slate-300 space-y-1">
                    {worker.recommendations.map((rec, i) => (
                      <li key={i} className="flex items-start gap-2">
                        <span className="text-amber-400">•</span>
                        {rec}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  )
}

// ═══════════════════════════════════════════════════════════════════════════════════════════
// MAIN COMPONENT
// ═══════════════════════════════════════════════════════════════════════════════════════════

const WorkforcePerformance: React.FC = () => {
  const queryClient = useQueryClient()
  const [expandedWorker, setExpandedWorker] = useState<string | null>(null)
  const [horizonDays, setHorizonDays] = useState(14)
  const [modelType, setModelType] = useState('ARIMA')
  
  // Fetch performance data
  const { data: perfData, isLoading, error } = useQuery({
    queryKey: ['workforce-performance'],
    queryFn: fetchPerformance,
    refetchInterval: 60000,
  })
  
  // Fetch forecast data
  const { data: forecastData, refetch: refetchForecast } = useQuery({
    queryKey: ['workforce-forecast', horizonDays, modelType],
    queryFn: () => fetchForecast(horizonDays, modelType),
    enabled: !!perfData,
  })
  
  // Assignment mutation
  const assignMutation = useMutation({
    mutationFn: () => fetchAssignment(true),
    onSuccess: (data) => {
      toast.success(`Alocação otimizada: ${data.assignment_plan?.assigned_operations} operações atribuídas`)
    },
    onError: () => {
      toast.error('Erro ao otimizar alocação')
    },
  })
  
  // Create forecast map
  const forecastMap = useMemo(() => {
    const map = new Map<string, WorkerForecast>()
    forecastData?.forecasts?.forEach(f => map.set(f.worker_id, f))
    return map
  }, [forecastData])
  
  // Sort workers by productivity
  const sortedWorkers = useMemo(() => {
    if (!perfData?.workers) return []
    return [...perfData.workers].sort(
      (a, b) => b.metrics.productivity - a.metrics.productivity
    )
  }, [perfData])
  
  const global = perfData?.global_metrics
  const dist = perfData?.performance_distribution
  
  if (error) {
    return (
      <div className="p-8 text-center">
        <AlertTriangle className="w-12 h-12 text-red-400 mx-auto mb-4" />
        <h2 className="text-xl font-semibold text-white mb-2">Erro ao carregar dados</h2>
        <p className="text-slate-400">Verifique se a API está a funcionar corretamente.</p>
      </div>
    )
  }
  
  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white flex items-center gap-3">
            <Users className="w-7 h-7 text-cyan-400" />
            Performance de Colaboradores
          </h1>
          <p className="text-slate-400 mt-1">
            Análise de produtividade, eficiência e competências • {sortedWorkers.length} colaboradores
          </p>
        </div>
        
        <div className="flex items-center gap-3">
          {/* Forecast controls */}
          <select
            value={modelType}
            onChange={(e) => setModelType(e.target.value)}
            className="px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white text-sm"
          >
            <option value="ARIMA">ARIMA</option>
            <option value="LEARNING_CURVE">Curva Aprendizagem</option>
          </select>
          
          <select
            value={horizonDays}
            onChange={(e) => setHorizonDays(Number(e.target.value))}
            className="px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white text-sm"
          >
            <option value={7}>7 dias</option>
            <option value={14}>14 dias</option>
            <option value={30}>30 dias</option>
          </select>
          
          <button
            onClick={() => refetchForecast()}
            className="flex items-center gap-2 px-3 py-2 bg-slate-700 hover:bg-slate-600 text-white rounded-lg transition-colors"
          >
            <LineChart className="w-4 h-4" />
            Prever
          </button>
          
          <button
            onClick={() => assignMutation.mutate()}
            disabled={assignMutation.isPending}
            className="flex items-center gap-2 px-4 py-2 bg-cyan-600 hover:bg-cyan-500 text-white rounded-lg transition-colors disabled:opacity-50"
          >
            <Zap className={`w-4 h-4 ${assignMutation.isPending ? 'animate-spin' : ''}`} />
            Otimizar Alocação
          </button>
        </div>
      </div>
      
      {/* Global KPIs */}
      {global && (
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
          <KPICard
            title="Colaboradores"
            value={perfData.total_workers}
            icon={<Users className="w-5 h-5 text-cyan-400" />}
          />
          <KPICard
            title="Produtividade Média"
            value={(global.avg_productivity ?? 0).toFixed(1)}
            subtitle="unidades/hora"
            icon={<BarChart3 className="w-5 h-5 text-blue-400" />}
          />
          <KPICard
            title="Eficiência Média"
            value={`${((global.avg_efficiency ?? 1) * 100).toFixed(0)}%`}
            icon={<Target className="w-5 h-5 text-emerald-400" />}
            highlight={(global.avg_efficiency ?? 1) >= 1}
          />
          <KPICard
            title="Skill Médio"
            value={`${((global.avg_skill_score ?? 0) * 100).toFixed(0)}%`}
            icon={<Award className="w-5 h-5 text-amber-400" />}
          />
          <KPICard
            title="SNR Médio"
            value={(global.avg_snr ?? 1).toFixed(1)}
            subtitle="consistência"
            icon={<Activity className="w-5 h-5 text-purple-400" />}
          />
          <KPICard
            title="Excelentes"
            value={dist?.excellent || 0}
            subtitle="top performers"
            icon={<TrendingUp className="w-5 h-5 text-emerald-400" />}
            highlight={(dist?.excellent || 0) > 0}
          />
        </div>
      )}
      
      {/* Performance distribution */}
      {dist && (
        <div className="bg-slate-800/50 border border-slate-700/50 rounded-lg p-4">
          <h3 className="text-sm font-medium text-slate-300 mb-3">Distribuição de Performance</h3>
          <div className="flex items-center gap-2">
            {Object.entries(dist).map(([level, count]) => {
              const total = Object.values(dist).reduce((a, b) => a + b, 0)
              const pct = (count / total) * 100
              const colors: Record<string, string> = {
                excellent: 'bg-emerald-500',
                good: 'bg-green-500',
                average: 'bg-slate-500',
                below_average: 'bg-amber-500',
                needs_improvement: 'bg-red-500',
              }
              return (
                <div key={level} className="flex-1">
                  <div className="h-2 rounded-full overflow-hidden bg-slate-700">
                    <div
                      className={`h-full ${colors[level] || 'bg-slate-500'}`}
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                  <p className="text-xs text-slate-400 mt-1 text-center">{count}</p>
                </div>
              )
            })}
          </div>
          <div className="flex justify-between text-xs text-slate-500 mt-2">
            <span>Necessita melhoria</span>
            <span>Excelente</span>
          </div>
        </div>
      )}
      
      {/* Workers list */}
      <div className="space-y-2">
        {isLoading ? (
          <div className="text-center py-12">
            <RefreshCw className="w-8 h-8 text-cyan-400 animate-spin mx-auto mb-4" />
            <p className="text-slate-400">A carregar dados de performance...</p>
          </div>
        ) : sortedWorkers.length === 0 ? (
          <div className="text-center py-12 bg-slate-800/30 rounded-lg border border-slate-700/50">
            <Users className="w-12 h-12 text-slate-500 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-white mb-2">Nenhum colaborador encontrado</h3>
            <p className="text-slate-400">
              Verifique se existem dados de operações com colaboradores.
            </p>
          </div>
        ) : (
          sortedWorkers.map((worker) => (
            <WorkerRow
              key={worker.worker_id}
              worker={worker}
              expanded={expandedWorker === worker.worker_id}
              onToggle={() => setExpandedWorker(
                expandedWorker === worker.worker_id ? null : worker.worker_id
              )}
              forecast={forecastMap.get(worker.worker_id)}
            />
          ))
        )}
      </div>
    </div>
  )
}

export default WorkforcePerformance

