/**
 * ════════════════════════════════════════════════════════════════════════════════════════════════════
 * CAUSAL ANALYSIS - Causal Context Models Dashboard
 * ════════════════════════════════════════════════════════════════════════════════════════════════════
 *
 * Dashboard de análise causal para compreender trade-offs e relações entre decisões e outcomes.
 * 
 * Features:
 * - Visualização do grafo causal
 * - Estimação de efeitos causais
 * - Análise de trade-offs
 * - Insights e recomendações
 * - Perguntas em linguagem natural
 */

import React, { useState } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import { motion } from 'framer-motion'
import {
  GitBranch,
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  Lightbulb,
  Target,
  MessageSquare,
  Send,
  RefreshCw,
  ChevronRight,
  Zap,
  Scale,
  BarChart3,
  Info,
} from 'lucide-react'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'

// ═══════════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════════

interface CausalEffect {
  treatment: string
  outcome: string
  estimate: number
  std_error: number
  ci_lower: number
  ci_upper: number
  direction: string
  magnitude: string
  significance: string
  explanation: string
}

interface CausalInsight {
  type: string
  title: string
  description: string
  priority: string
  treatments: string[]
  outcomes: string[]
  confidence: number
  suggested_actions: string[]
}

interface DashboardData {
  metrics: {
    structure: { n_variables: number; n_relations: number; n_treatments: number; n_outcomes: number }
    complexity: { overall_complexity: number; nonlinearity_score: number; interaction_score: number }
  }
  insights: CausalInsight[]
  summary: {
    complexity_score: number
    n_tradeoffs: number
    n_leverage_points: number
    n_risks: number
    high_priority_insights: number
  }
}

interface ExplainResult {
  success: boolean
  question: string
  treatment?: string
  outcome?: string
  effect?: CausalEffect
  explanation?: string
  error?: string
}

// ═══════════════════════════════════════════════════════════════════════════════
// API CALLS
// ═══════════════════════════════════════════════════════════════════════════════

async function fetchDashboard(): Promise<DashboardData> {
  const res = await fetch(`${API_BASE_URL}/causal/dashboard`)
  if (!res.ok) throw new Error('Failed to fetch dashboard')
  return res.json()
}

async function fetchVariables(): Promise<any> {
  const res = await fetch(`${API_BASE_URL}/causal/variables`)
  if (!res.ok) throw new Error('Failed to fetch variables')
  return res.json()
}

async function fetchTradeoffs(treatment: string): Promise<any> {
  const res = await fetch(`${API_BASE_URL}/causal/tradeoffs/${treatment}`)
  if (!res.ok) throw new Error('Failed to fetch tradeoffs')
  return res.json()
}

async function explainQuestion(question: string): Promise<ExplainResult> {
  const res = await fetch(`${API_BASE_URL}/causal/explain`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question }),
  })
  if (!res.ok) throw new Error('Failed to explain')
  return res.json()
}

// ═══════════════════════════════════════════════════════════════════════════════
// COMPONENTS
// ═══════════════════════════════════════════════════════════════════════════════

const ComplexityGauge: React.FC<{ score: number }> = ({ score }) => {
  const color = score < 40 ? '#10b981' : score < 70 ? '#f59e0b' : '#ef4444'
  const label = score < 40 ? 'Baixa' : score < 70 ? 'Moderada' : 'Alta'

  return (
    <div className="text-center">
      <div className="relative w-24 h-24 mx-auto mb-2">
        <svg className="w-24 h-24 -rotate-90">
          <circle cx="48" cy="48" r="40" fill="none" strokeWidth="8" className="stroke-slate-700" />
          <circle
            cx="48" cy="48" r="40" fill="none" strokeWidth="8"
            stroke={color}
            strokeLinecap="round"
            strokeDasharray={`${score * 2.51} 251`}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-xl font-bold text-white">{score.toFixed(0)}</span>
        </div>
      </div>
      <p className="text-sm text-slate-400">Complexidade</p>
      <p className="text-xs font-medium" style={{ color }}>{label}</p>
    </div>
  )
}

const InsightCard: React.FC<{ insight: CausalInsight }> = ({ insight }) => {
  const typeConfig: Record<string, { icon: React.ReactNode; color: string }> = {
    tradeoff: { icon: <Scale className="w-4 h-4" />, color: 'amber' },
    leverage: { icon: <Zap className="w-4 h-4" />, color: 'cyan' },
    risk: { icon: <AlertTriangle className="w-4 h-4" />, color: 'red' },
    opportunity: { icon: <TrendingUp className="w-4 h-4" />, color: 'emerald' },
    interaction: { icon: <GitBranch className="w-4 h-4" />, color: 'purple' },
  }

  const config = typeConfig[insight.type] || { icon: <Info className="w-4 h-4" />, color: 'slate' }

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className={`p-4 rounded-xl border bg-${config.color}-500/5 border-${config.color}-500/30`}
    >
      <div className="flex items-start gap-3">
        <div className={`p-2 rounded-lg bg-${config.color}-500/20 text-${config.color}-400`}>
          {config.icon}
        </div>
        <div className="flex-1">
          <div className="flex items-center justify-between mb-1">
            <h4 className="text-sm font-medium text-white">{insight.title}</h4>
            <span className={`text-xs px-2 py-0.5 rounded ${
              insight.priority === 'high' ? 'bg-red-500/20 text-red-400' :
              insight.priority === 'medium' ? 'bg-amber-500/20 text-amber-400' :
              'bg-slate-600/50 text-slate-400'
            }`}>
              {insight.priority}
            </span>
          </div>
          <p className="text-xs text-slate-400 mb-2">{insight.description}</p>
          {insight.suggested_actions.length > 0 && (
            <div className="space-y-1">
              {insight.suggested_actions.slice(0, 2).map((action, i) => (
                <div key={i} className="flex items-center gap-1 text-xs text-slate-500">
                  <ChevronRight className="w-3 h-3" />
                  {action}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </motion.div>
  )
}

const TradeoffVisualizer: React.FC<{ treatment: string }> = ({ treatment }) => {
  const { data, isLoading } = useQuery({
    queryKey: ['tradeoff', treatment],
    queryFn: () => fetchTradeoffs(treatment),
  })

  if (isLoading) {
    return <div className="animate-pulse bg-slate-800 rounded-lg h-32" />
  }

  if (!data) return null

  return (
    <div className="p-4 rounded-xl bg-slate-800/30 border border-slate-700/50">
      <h4 className="text-sm font-medium text-white mb-3">{data.treatment_description}</h4>
      
      <div className="grid grid-cols-2 gap-4">
        <div>
          <p className="text-xs text-emerald-400 mb-2 flex items-center gap-1">
            <TrendingUp className="w-3 h-3" /> Efeitos Positivos
          </p>
          {data.positive_effects?.slice(0, 3).map((e: any, i: number) => (
            <div key={i} className="text-xs text-slate-300 mb-1">
              {e.outcome}: +{e.estimate.toFixed(2)}
            </div>
          ))}
        </div>
        <div>
          <p className="text-xs text-red-400 mb-2 flex items-center gap-1">
            <TrendingDown className="w-3 h-3" /> Efeitos Negativos
          </p>
          {data.negative_effects?.slice(0, 3).map((e: any, i: number) => (
            <div key={i} className="text-xs text-slate-300 mb-1">
              {e.outcome}: {e.estimate.toFixed(2)}
            </div>
          ))}
        </div>
      </div>
      
      <div className="mt-3 pt-3 border-t border-slate-700/50">
        <p className="text-xs text-slate-500">{data.recommendation}</p>
      </div>
    </div>
  )
}

const QuestionInterface: React.FC = () => {
  const [question, setQuestion] = useState('')
  const [result, setResult] = useState<ExplainResult | null>(null)

  const mutation = useMutation({
    mutationFn: explainQuestion,
    onSuccess: setResult,
  })

  const exampleQuestions = [
    "Se eu reduzir setups, o que acontece ao custo energético?",
    "Qual o impacto de aumentar turnos noturnos no stress?",
    "Adiar manutenção afeta a probabilidade de falhas?",
    "Aumentar carga das máquinas melhora o makespan?",
  ]

  return (
    <div className="space-y-4">
      <div className="relative">
        <textarea
          value={question}
          onChange={e => setQuestion(e.target.value)}
          placeholder="Faça uma pergunta sobre relações causais..."
          className="w-full h-24 p-4 pr-12 bg-slate-900 border border-slate-700 rounded-xl text-white text-sm resize-none focus:border-cyan-500 focus:outline-none"
        />
        <button
          onClick={() => question && mutation.mutate(question)}
          disabled={!question || mutation.isPending}
          className="absolute bottom-4 right-4 p-2 bg-cyan-600 hover:bg-cyan-500 rounded-lg text-white disabled:opacity-50 transition-colors"
        >
          {mutation.isPending ? (
            <RefreshCw className="w-4 h-4 animate-spin" />
          ) : (
            <Send className="w-4 h-4" />
          )}
        </button>
      </div>

      <div className="flex flex-wrap gap-2">
        {exampleQuestions.map((q, i) => (
          <button
            key={i}
            onClick={() => setQuestion(q)}
            className="text-xs px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-white rounded-full transition-colors"
          >
            {q}
          </button>
        ))}
      </div>

      {result && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="p-4 rounded-xl bg-slate-800/50 border border-slate-700/50"
        >
          {result.success ? (
            <>
              <div className="flex items-center gap-2 mb-3">
                <MessageSquare className="w-4 h-4 text-cyan-400" />
                <span className="text-sm font-medium text-white">Resposta</span>
              </div>
              <div className="prose prose-invert prose-sm max-w-none">
                <pre className="whitespace-pre-wrap text-xs text-slate-300 bg-slate-900/50 p-3 rounded-lg">
                  {result.explanation}
                </pre>
              </div>
              {result.effect && (
                <div className="mt-4 grid grid-cols-3 gap-4 text-center">
                  <div>
                    <p className="text-lg font-bold text-white">{result.effect.estimate.toFixed(3)}</p>
                    <p className="text-xs text-slate-500">Efeito</p>
                  </div>
                  <div>
                    <p className="text-lg font-bold text-cyan-400">{result.effect.significance}</p>
                    <p className="text-xs text-slate-500">Significância</p>
                  </div>
                  <div>
                    <p className="text-lg font-bold text-amber-400">{result.effect.magnitude}</p>
                    <p className="text-xs text-slate-500">Magnitude</p>
                  </div>
                </div>
              )}
            </>
          ) : (
            <div className="text-red-400 text-sm">{result.error}</div>
          )}
        </motion.div>
      )}
    </div>
  )
}

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN COMPONENT
// ═══════════════════════════════════════════════════════════════════════════════

const CausalAnalysis: React.FC = () => {
  const [selectedTreatment, setSelectedTreatment] = useState<string | null>(null)

  const { data: dashboard, isLoading } = useQuery({
    queryKey: ['causal-dashboard'],
    queryFn: fetchDashboard,
  })

  const { data: variables } = useQuery({
    queryKey: ['causal-variables'],
    queryFn: fetchVariables,
  })

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <RefreshCw className="w-8 h-8 text-cyan-400 animate-spin" />
      </div>
    )
  }

  if (!dashboard) {
    return (
      <div className="text-center py-12">
        <AlertTriangle className="w-12 h-12 text-amber-400 mx-auto mb-4" />
        <p className="text-slate-400">Módulo CCM não disponível</p>
      </div>
    )
  }

  const { metrics, insights, summary } = dashboard

  return (
    <div className="space-y-6">
      {/* Header */}
      <header className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-gradient-to-br from-purple-500/20 to-cyan-500/20 rounded-xl border border-purple-500/30">
            <GitBranch className="w-6 h-6 text-purple-400" />
          </div>
          <div>
            <p className="text-xs uppercase tracking-[0.4em] text-slate-500">Causal Context Models</p>
            <h2 className="text-2xl font-semibold text-white">Análise Causal</h2>
          </div>
        </div>
      </header>

      {/* Summary KPIs */}
      <div className="grid grid-cols-2 lg:grid-cols-5 gap-4">
        <div className="p-4 rounded-xl bg-slate-800/50 border border-slate-700/50 text-center">
          <ComplexityGauge score={summary.complexity_score} />
        </div>
        <div className="p-4 rounded-xl bg-slate-800/50 border border-slate-700/50">
          <Scale className="w-5 h-5 text-amber-400 mb-2" />
          <p className="text-2xl font-bold text-white">{summary.n_tradeoffs}</p>
          <p className="text-xs text-slate-400">Trade-offs</p>
        </div>
        <div className="p-4 rounded-xl bg-slate-800/50 border border-slate-700/50">
          <Zap className="w-5 h-5 text-cyan-400 mb-2" />
          <p className="text-2xl font-bold text-white">{summary.n_leverage_points}</p>
          <p className="text-xs text-slate-400">Pontos de Alavancagem</p>
        </div>
        <div className="p-4 rounded-xl bg-slate-800/50 border border-slate-700/50">
          <AlertTriangle className="w-5 h-5 text-red-400 mb-2" />
          <p className="text-2xl font-bold text-white">{summary.n_risks}</p>
          <p className="text-xs text-slate-400">Riscos</p>
        </div>
        <div className="p-4 rounded-xl bg-slate-800/50 border border-slate-700/50">
          <Lightbulb className="w-5 h-5 text-emerald-400 mb-2" />
          <p className="text-2xl font-bold text-white">{summary.high_priority_insights}</p>
          <p className="text-xs text-slate-400">Prioridade Alta</p>
        </div>
      </div>

      {/* Question Interface */}
      <div className="p-6 rounded-2xl bg-slate-800/30 border border-slate-700/50">
        <h3 className="text-sm font-medium text-white mb-4 flex items-center gap-2">
          <MessageSquare className="w-4 h-4 text-cyan-400" />
          Perguntas Causais
        </h3>
        <QuestionInterface />
      </div>

      {/* Trade-off Analysis */}
      {variables?.treatments && (
        <div className="space-y-4">
          <h3 className="text-sm font-medium text-white flex items-center gap-2">
            <Scale className="w-4 h-4 text-amber-400" />
            Análise de Trade-offs
          </h3>
          <div className="flex flex-wrap gap-2 mb-4">
            {variables.treatments.map((t: any) => (
              <button
                key={t.name}
                onClick={() => setSelectedTreatment(t.name === selectedTreatment ? null : t.name)}
                className={`text-xs px-3 py-1.5 rounded-full transition-colors ${
                  selectedTreatment === t.name
                    ? 'bg-amber-500/20 text-amber-400 border border-amber-500/50'
                    : 'bg-slate-800 text-slate-400 hover:text-white border border-slate-700'
                }`}
              >
                {t.description.length > 30 ? t.description.substring(0, 30) + '...' : t.description}
              </button>
            ))}
          </div>
          {selectedTreatment && <TradeoffVisualizer treatment={selectedTreatment} />}
        </div>
      )}

      {/* Insights */}
      <div className="space-y-4">
        <h3 className="text-sm font-medium text-white flex items-center gap-2">
          <Lightbulb className="w-4 h-4 text-emerald-400" />
          Insights Causais ({insights.length})
        </h3>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {insights.slice(0, 6).map((insight, i) => (
            <InsightCard key={i} insight={insight} />
          ))}
        </div>
      </div>

      {/* Graph Stats */}
      <div className="p-4 rounded-xl bg-slate-800/30 border border-slate-700/50">
        <h3 className="text-sm font-medium text-white mb-3 flex items-center gap-2">
          <BarChart3 className="w-4 h-4 text-cyan-400" />
          Estrutura do Grafo Causal
        </h3>
        <div className="grid grid-cols-4 gap-4 text-center">
          <div>
            <p className="text-2xl font-bold text-white">{metrics.structure.n_variables}</p>
            <p className="text-xs text-slate-500">Variáveis</p>
          </div>
          <div>
            <p className="text-2xl font-bold text-white">{metrics.structure.n_relations}</p>
            <p className="text-xs text-slate-500">Relações</p>
          </div>
          <div>
            <p className="text-2xl font-bold text-cyan-400">{metrics.structure.n_treatments}</p>
            <p className="text-xs text-slate-500">Decisões</p>
          </div>
          <div>
            <p className="text-2xl font-bold text-emerald-400">{metrics.structure.n_outcomes}</p>
            <p className="text-xs text-slate-500">Outcomes</p>
          </div>
        </div>
      </div>
    </div>
  )
}

export default CausalAnalysis



