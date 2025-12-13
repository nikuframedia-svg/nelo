/**
 * ════════════════════════════════════════════════════════════════════════════════════════════════════
 * ZDM SIMULATOR - Zero Disruption Manufacturing
 * ════════════════════════════════════════════════════════════════════════════════════════════════════
 *
 * Dashboard de simulação de resiliência e análise de cenários de falha.
 * 
 * Features:
 * - Resilience Score global
 * - Simulação de cenários de falha
 * - Estratégias de recuperação
 * - Análise de riscos
 */

import React, { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Shield,
  AlertTriangle,
  Zap,
  RefreshCw,
  CheckCircle,
  XCircle,
  TrendingUp,
  Activity,
  Target,
  ChevronRight,
  Play,
  Settings,
  BarChart3,
} from 'lucide-react'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'

// ═══════════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════════

interface FailureScenario {
  scenario_id: string
  failure_type: string
  machine_id: string
  start_time: string
  duration_hours: number
  severity: number
  probability: number
  triggered_by_rul: boolean
  description: string
}

interface RecoveryPlan {
  scenario_id: string
  primary_action: {
    strategy: string
    description: string
    effectiveness_pct: number
    cost_eur: number
    recovery_time_hours: number
  }
  total_cost_eur: number
  expected_effectiveness_pct: number
  risk_level: string
  rationale: string
  warnings: string[]
}

interface SimulationResult {
  resilience_report: {
    overall_resilience_score: number
    scenarios_simulated: number
    recovery_stats: {
      full_recovery: number
      partial_recovery: number
      failed_recovery: number
      success_rate_pct: number
    }
    averages: {
      recovery_time_hours: number
      throughput_loss_pct: number
      otd_impact_pct: number
    }
    critical_machines: string[]
    recommendations: string[]
  }
  recovery_plans: RecoveryPlan[]
  summary: {
    resilience_score: number
    resilience_grade: string
    scenarios_simulated: number
    full_recovery_rate: number
    critical_machines: string[]
    avg_recovery_time_hours: number
    top_risks: Array<{
      machine: string
      type: string
      probability: number
      severity: number
    }>
  }
}

interface DashboardData {
  resilience_score: number
  resilience_grade: string
  scenarios_preview: {
    total: number
    by_type: Record<string, { count: number; avg_severity: number; avg_probability: number }>
    top_risks: FailureScenario[]
  }
  recovery_success_rate: number
  critical_machines: string[]
  recommendations: string[]
  kpis: {
    operations_at_risk: number
    high_probability_failures: number
    rul_triggered_scenarios: number
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// API CALLS
// ═══════════════════════════════════════════════════════════════════════════════

async function fetchDashboard(): Promise<DashboardData> {
  const res = await fetch(`${API_BASE_URL}/zdm/dashboard`)
  if (!res.ok) throw new Error('Failed to fetch dashboard')
  return res.json()
}

async function runSimulation(config: {
  n_scenarios: number
  enable_rerouting: boolean
  enable_overtime: boolean
}): Promise<SimulationResult> {
  const res = await fetch(`${API_BASE_URL}/zdm/simulate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      ...config,
      enable_priority_shuffle: true,
      use_rul_data: true,
    }),
  })
  if (!res.ok) throw new Error('Failed to run simulation')
  return res.json()
}

// ═══════════════════════════════════════════════════════════════════════════════
// COMPONENTS
// ═══════════════════════════════════════════════════════════════════════════════

const ResilienceGauge: React.FC<{ score: number; grade: string }> = ({ score, grade }) => {
  const color = score >= 70 ? '#10b981' : score >= 50 ? '#f59e0b' : '#ef4444'
  const radius = 70
  const stroke = 10
  const circumference = 2 * Math.PI * radius
  const offset = circumference - (score / 100) * circumference

  return (
    <div className="relative inline-flex items-center justify-center">
      <svg width={radius * 2 + stroke * 2} height={radius * 2 + stroke * 2} className="-rotate-90">
        <circle
          cx={radius + stroke}
          cy={radius + stroke}
          r={radius}
          fill="none"
          stroke="currentColor"
          strokeWidth={stroke}
          className="text-slate-700"
        />
        <circle
          cx={radius + stroke}
          cy={radius + stroke}
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth={stroke}
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          className="transition-all duration-1000"
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-4xl font-bold text-white">{grade}</span>
        <span className="text-lg text-slate-400">{score.toFixed(0)}%</span>
      </div>
    </div>
  )
}

const FailureTypeCard: React.FC<{
  type: string
  data: { count: number; avg_severity: number; avg_probability: number }
}> = ({ type, data }) => {
  const typeConfig: Record<string, { icon: React.ReactNode; color: string; label: string }> = {
    sudden: { icon: <Zap className="w-4 h-4" />, color: 'red', label: 'Falha Súbita' },
    gradual: { icon: <TrendingUp className="w-4 h-4" />, color: 'amber', label: 'Degradação' },
    quality: { icon: <Target className="w-4 h-4" />, color: 'purple', label: 'Qualidade' },
    material: { icon: <Activity className="w-4 h-4" />, color: 'blue', label: 'Material' },
    operator: { icon: <Activity className="w-4 h-4" />, color: 'cyan', label: 'Operador' },
  }

  const config = typeConfig[type] || { icon: <AlertTriangle className="w-4 h-4" />, color: 'slate', label: type }

  return (
    <div className={`p-3 rounded-lg bg-${config.color}-500/10 border border-${config.color}-500/30`}>
      <div className="flex items-center gap-2 mb-2">
        <span className={`text-${config.color}-400`}>{config.icon}</span>
        <span className="text-sm font-medium text-white">{config.label}</span>
        <span className="ml-auto text-xs bg-slate-700 px-2 py-0.5 rounded text-slate-300">{data.count}</span>
      </div>
      <div className="grid grid-cols-2 gap-2 text-xs">
        <div>
          <span className="text-slate-500">Severidade</span>
          <p className="text-slate-300">{(data.avg_severity * 100).toFixed(0)}%</p>
        </div>
        <div>
          <span className="text-slate-500">Probabilidade</span>
          <p className="text-slate-300">{(data.avg_probability * 100).toFixed(0)}%</p>
        </div>
      </div>
    </div>
  )
}

const ScenarioCard: React.FC<{ scenario: FailureScenario }> = ({ scenario }) => (
  <div className="p-4 rounded-lg bg-slate-800/50 border border-slate-700/50 hover:border-slate-600/50 transition-colors">
    <div className="flex items-center justify-between mb-2">
      <span className="text-sm font-medium text-white">{scenario.machine_id}</span>
      <span className={`text-xs px-2 py-0.5 rounded ${
        scenario.severity > 0.7 ? 'bg-red-500/20 text-red-400' :
        scenario.severity > 0.5 ? 'bg-amber-500/20 text-amber-400' :
        'bg-slate-600/50 text-slate-300'
      }`}>
        Sev: {(scenario.severity * 100).toFixed(0)}%
      </span>
    </div>
    <p className="text-xs text-slate-400 mb-2">{scenario.description}</p>
    <div className="flex items-center gap-4 text-xs">
      <span className="text-slate-500">
        Tipo: <span className="text-slate-300">{scenario.failure_type}</span>
      </span>
      <span className="text-slate-500">
        Duração: <span className="text-slate-300">{scenario.duration_hours.toFixed(1)}h</span>
      </span>
      {scenario.triggered_by_rul && (
        <span className="text-orange-400">🔧 RUL</span>
      )}
    </div>
  </div>
)

const RecoveryPlanCard: React.FC<{ plan: RecoveryPlan }> = ({ plan }) => (
  <div className="p-4 rounded-lg bg-slate-800/30 border border-slate-700/50">
    <div className="flex items-center justify-between mb-3">
      <span className="text-sm font-medium text-white">{plan.scenario_id}</span>
      <span className={`text-xs px-2 py-1 rounded font-medium ${
        plan.risk_level === 'LOW' ? 'bg-emerald-500/20 text-emerald-400' :
        plan.risk_level === 'MEDIUM' ? 'bg-amber-500/20 text-amber-400' :
        'bg-red-500/20 text-red-400'
      }`}>
        Risco: {plan.risk_level}
      </span>
    </div>
    
    <div className="mb-3">
      <p className="text-xs text-slate-500 mb-1">Estratégia Principal</p>
      <p className="text-sm text-cyan-400">{plan.primary_action.strategy.replace('_', ' ').toUpperCase()}</p>
      <p className="text-xs text-slate-400 mt-1">{plan.primary_action.description}</p>
    </div>
    
    <div className="grid grid-cols-3 gap-2 text-xs">
      <div>
        <span className="text-slate-500">Efetividade</span>
        <p className="text-emerald-400 font-medium">{plan.expected_effectiveness_pct.toFixed(0)}%</p>
      </div>
      <div>
        <span className="text-slate-500">Custo</span>
        <p className="text-amber-400 font-medium">€{plan.total_cost_eur.toFixed(0)}</p>
      </div>
      <div>
        <span className="text-slate-500">Tempo</span>
        <p className="text-slate-300">{plan.primary_action.recovery_time_hours.toFixed(1)}h</p>
      </div>
    </div>
    
    {plan.warnings.length > 0 && (
      <div className="mt-3 pt-3 border-t border-slate-700/50">
        {plan.warnings.map((w, i) => (
          <p key={i} className="text-xs text-amber-400 flex items-center gap-1">
            <AlertTriangle className="w-3 h-3" />
            {w}
          </p>
        ))}
      </div>
    )}
  </div>
)

const SimulationResultsModal: React.FC<{
  result: SimulationResult | null
  onClose: () => void
}> = ({ result, onClose }) => {
  if (!result) return null

  const { resilience_report, recovery_plans, summary } = result

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 bg-black/60 flex items-center justify-center z-50 p-4"
      onClick={onClose}
    >
      <motion.div
        initial={{ scale: 0.9, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        exit={{ scale: 0.9, opacity: 0 }}
        className="bg-slate-900 rounded-2xl p-6 max-w-4xl w-full max-h-[90vh] overflow-y-auto border border-slate-700"
        onClick={e => e.stopPropagation()}
      >
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-xl font-bold text-white">Resultados da Simulação</h2>
          <span className={`text-2xl font-bold ${
            summary.resilience_grade === 'A' ? 'text-emerald-400' :
            summary.resilience_grade === 'B' ? 'text-cyan-400' :
            summary.resilience_grade === 'C' ? 'text-amber-400' :
            'text-red-400'
          }`}>
            Grade {summary.resilience_grade}
          </span>
        </div>

        {/* Summary KPIs */}
        <div className="grid grid-cols-4 gap-4 mb-6">
          <div className="bg-slate-800/50 rounded-xl p-4 text-center">
            <p className="text-3xl font-bold text-white">{summary.resilience_score.toFixed(0)}%</p>
            <p className="text-xs text-slate-400">Resilience Score</p>
          </div>
          <div className="bg-slate-800/50 rounded-xl p-4 text-center">
            <p className="text-3xl font-bold text-emerald-400">{summary.full_recovery_rate.toFixed(0)}%</p>
            <p className="text-xs text-slate-400">Taxa de Recuperação</p>
          </div>
          <div className="bg-slate-800/50 rounded-xl p-4 text-center">
            <p className="text-3xl font-bold text-amber-400">{summary.avg_recovery_time_hours.toFixed(1)}h</p>
            <p className="text-xs text-slate-400">Tempo Médio Recuperação</p>
          </div>
          <div className="bg-slate-800/50 rounded-xl p-4 text-center">
            <p className="text-3xl font-bold text-cyan-400">{summary.scenarios_simulated}</p>
            <p className="text-xs text-slate-400">Cenários Simulados</p>
          </div>
        </div>

        {/* Recovery Stats */}
        <div className="mb-6">
          <h3 className="text-sm font-medium text-white mb-3">Estatísticas de Recuperação</h3>
          <div className="flex gap-4">
            <div className="flex items-center gap-2">
              <CheckCircle className="w-4 h-4 text-emerald-400" />
              <span className="text-sm text-slate-300">{resilience_report.recovery_stats.full_recovery} Total</span>
            </div>
            <div className="flex items-center gap-2">
              <AlertTriangle className="w-4 h-4 text-amber-400" />
              <span className="text-sm text-slate-300">{resilience_report.recovery_stats.partial_recovery} Parcial</span>
            </div>
            <div className="flex items-center gap-2">
              <XCircle className="w-4 h-4 text-red-400" />
              <span className="text-sm text-slate-300">{resilience_report.recovery_stats.failed_recovery} Falha</span>
            </div>
          </div>
        </div>

        {/* Top Risks */}
        {summary.top_risks.length > 0 && (
          <div className="mb-6">
            <h3 className="text-sm font-medium text-white mb-3">Riscos Principais</h3>
            <div className="space-y-2">
              {summary.top_risks.map((risk, i) => (
                <div key={i} className="flex items-center justify-between p-2 bg-slate-800/50 rounded">
                  <span className="text-sm text-slate-300">{risk.machine} - {risk.type}</span>
                  <div className="flex gap-4 text-xs">
                    <span className="text-slate-500">Prob: <span className="text-slate-300">{(risk.probability * 100).toFixed(0)}%</span></span>
                    <span className="text-slate-500">Sev: <span className="text-slate-300">{(risk.severity * 100).toFixed(0)}%</span></span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Recovery Plans */}
        <div className="mb-6">
          <h3 className="text-sm font-medium text-white mb-3">Planos de Recuperação</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {recovery_plans.slice(0, 4).map((plan, i) => (
              <RecoveryPlanCard key={i} plan={plan} />
            ))}
          </div>
        </div>

        {/* Recommendations */}
        <div className="mb-6">
          <h3 className="text-sm font-medium text-white mb-3">Recomendações</h3>
          <div className="space-y-2">
            {resilience_report.recommendations.map((rec, i) => (
              <div key={i} className="flex items-start gap-2 p-3 bg-slate-800/30 rounded-lg">
                <ChevronRight className="w-4 h-4 text-cyan-400 mt-0.5" />
                <p className="text-sm text-slate-300">{rec}</p>
              </div>
            ))}
          </div>
        </div>

        <button
          onClick={onClose}
          className="w-full py-3 bg-slate-700 hover:bg-slate-600 text-white rounded-lg transition-colors"
        >
          Fechar
        </button>
      </motion.div>
    </motion.div>
  )
}

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN COMPONENT
// ═══════════════════════════════════════════════════════════════════════════════

const ZDMSimulator: React.FC = () => {
  const queryClient = useQueryClient()
  const [showConfig, setShowConfig] = useState(false)
  const [simConfig, setSimConfig] = useState({
    n_scenarios: 10,
    enable_rerouting: true,
    enable_overtime: true,
  })
  const [simulationResult, setSimulationResult] = useState<SimulationResult | null>(null)

  const { data, isLoading, refetch } = useQuery({
    queryKey: ['zdm-dashboard'],
    queryFn: fetchDashboard,
    refetchInterval: 60000,
  })

  const simulateMutation = useMutation({
    mutationFn: () => runSimulation(simConfig),
    onSuccess: (result) => {
      setSimulationResult(result)
      queryClient.invalidateQueries({ queryKey: ['zdm-dashboard'] })
    },
  })

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <RefreshCw className="w-8 h-8 text-cyan-400 animate-spin" />
      </div>
    )
  }

  if (!data) {
    return (
      <div className="text-center py-12">
        <AlertTriangle className="w-12 h-12 text-amber-400 mx-auto mb-4" />
        <p className="text-slate-400">Módulo ZDM não disponível</p>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <header className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-gradient-to-br from-emerald-500/20 to-cyan-500/20 rounded-xl border border-emerald-500/30">
            <Shield className="w-6 h-6 text-emerald-400" />
          </div>
          <div>
            <p className="text-xs uppercase tracking-[0.4em] text-slate-500">Zero Disruption Manufacturing</p>
            <h2 className="text-2xl font-semibold text-white">Simulador de Resiliência</h2>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={() => setShowConfig(!showConfig)}
            className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-white transition-colors"
          >
            <Settings className="w-5 h-5" />
          </button>
          <button
            onClick={() => refetch()}
            className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-white transition-colors"
          >
            <RefreshCw className="w-5 h-5" />
          </button>
          <button
            onClick={() => simulateMutation.mutate()}
            disabled={simulateMutation.isPending}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white font-medium transition-colors disabled:opacity-50"
          >
            {simulateMutation.isPending ? (
              <RefreshCw className="w-4 h-4 animate-spin" />
            ) : (
              <Play className="w-4 h-4" />
            )}
            Simular
          </button>
        </div>
      </header>

      {/* Config Panel */}
      <AnimatePresence>
        {showConfig && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="overflow-hidden"
          >
            <div className="p-4 rounded-xl bg-slate-800/50 border border-slate-700/50">
              <h3 className="text-sm font-medium text-white mb-3">Configuração da Simulação</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div>
                  <label className="text-xs text-slate-400 block mb-1">Número de Cenários</label>
                  <input
                    type="number"
                    min={1}
                    max={50}
                    value={simConfig.n_scenarios}
                    onChange={e => setSimConfig(c => ({ ...c, n_scenarios: parseInt(e.target.value) || 10 }))}
                    className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-white text-sm"
                  />
                </div>
                <div className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    id="rerouting"
                    checked={simConfig.enable_rerouting}
                    onChange={e => setSimConfig(c => ({ ...c, enable_rerouting: e.target.checked }))}
                    className="rounded bg-slate-700 border-slate-600"
                  />
                  <label htmlFor="rerouting" className="text-sm text-slate-300">Permitir Reencaminhamento</label>
                </div>
                <div className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    id="overtime"
                    checked={simConfig.enable_overtime}
                    onChange={e => setSimConfig(c => ({ ...c, enable_overtime: e.target.checked }))}
                    className="rounded bg-slate-700 border-slate-600"
                  />
                  <label htmlFor="overtime" className="text-sm text-slate-300">Permitir Horas Extra</label>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Main Dashboard */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Resilience Score */}
        <div className="flex flex-col items-center justify-center p-6 rounded-2xl border border-slate-700/50 bg-gradient-to-br from-slate-800/50 to-slate-900/50">
          <ResilienceGauge score={data.resilience_score} grade={data.resilience_grade} />
          <p className="mt-4 text-lg font-medium text-white">Resilience Score</p>
          <p className="text-sm text-slate-400">
            Taxa de recuperação: {data.recovery_success_rate.toFixed(0)}%
          </p>
        </div>

        {/* KPIs */}
        <div className="lg:col-span-2 grid grid-cols-3 gap-4">
          <div className="p-4 rounded-xl bg-slate-800/50 border border-slate-700/50">
            <div className="flex items-center gap-2 mb-2">
              <AlertTriangle className="w-5 h-5 text-red-400" />
              <span className="text-xs text-slate-400">Operações em Risco</span>
            </div>
            <p className="text-3xl font-bold text-white">{data.kpis.operations_at_risk}</p>
          </div>
          <div className="p-4 rounded-xl bg-slate-800/50 border border-slate-700/50">
            <div className="flex items-center gap-2 mb-2">
              <TrendingUp className="w-5 h-5 text-amber-400" />
              <span className="text-xs text-slate-400">Alta Probabilidade</span>
            </div>
            <p className="text-3xl font-bold text-white">{data.kpis.high_probability_failures}</p>
          </div>
          <div className="p-4 rounded-xl bg-slate-800/50 border border-slate-700/50">
            <div className="flex items-center gap-2 mb-2">
              <Activity className="w-5 h-5 text-orange-400" />
              <span className="text-xs text-slate-400">Baseados em RUL</span>
            </div>
            <p className="text-3xl font-bold text-white">{data.kpis.rul_triggered_scenarios}</p>
          </div>
        </div>
      </div>

      {/* Failure Types */}
      <div className="space-y-3">
        <h3 className="text-sm font-medium text-white flex items-center gap-2">
          <BarChart3 className="w-4 h-4 text-cyan-400" />
          Cenários por Tipo de Falha
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
          {Object.entries(data.scenarios_preview.by_type).map(([type, typeData]) => (
            <FailureTypeCard key={type} type={type} data={typeData} />
          ))}
        </div>
      </div>

      {/* Top Risks */}
      <div className="space-y-3">
        <h3 className="text-sm font-medium text-white flex items-center gap-2">
          <AlertTriangle className="w-4 h-4 text-red-400" />
          Cenários de Risco Prioritários
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {data.scenarios_preview.top_risks.slice(0, 6).map((scenario, i) => (
            <ScenarioCard key={i} scenario={scenario} />
          ))}
        </div>
      </div>

      {/* Recommendations */}
      {data.recommendations.length > 0 && (
        <div className="p-4 rounded-xl bg-slate-800/30 border border-slate-700/50">
          <h3 className="text-sm font-medium text-white mb-3 flex items-center gap-2">
            <CheckCircle className="w-4 h-4 text-emerald-400" />
            Recomendações
          </h3>
          {data.recommendations.map((rec, i) => (
            <p key={i} className="text-sm text-slate-300">{rec}</p>
          ))}
        </div>
      )}

      {/* Critical Machines */}
      {data.critical_machines.length > 0 && (
        <div className="p-4 rounded-xl bg-red-500/10 border border-red-500/30">
          <h3 className="text-sm font-medium text-red-400 mb-2">Máquinas Críticas</h3>
          <div className="flex flex-wrap gap-2">
            {data.critical_machines.map((machine, i) => (
              <span key={i} className="px-3 py-1 bg-red-500/20 text-red-300 rounded-full text-sm">
                {machine}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Simulation Results Modal */}
      <AnimatePresence>
        {simulationResult && (
          <SimulationResultsModal
            result={simulationResult}
            onClose={() => setSimulationResult(null)}
          />
        )}
      </AnimatePresence>
    </div>
  )
}

export default ZDMSimulator



