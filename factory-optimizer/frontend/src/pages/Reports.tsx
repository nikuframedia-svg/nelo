/**
 * ════════════════════════════════════════════════════════════════════════════════════════════
 * PRODPLAN 4.0 — REPORTS & COMPARISON PAGE
 * ════════════════════════════════════════════════════════════════════════════════════════════
 *
 * Interface para:
 * - Relatórios comparativos executivos
 * - Explicações técnicas dos algoritmos
 * - Resumos de cenários em linguagem natural
 */

import React, { useState } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import { motion, AnimatePresence } from 'framer-motion'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import {
  FileText,
  BarChart3,
  GitCompare,
  BookOpen,
  Lightbulb,
  Download,
  RefreshCw,
  ChevronRight,
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  CheckCircle,
  Info,
  Settings,
  Zap,
} from 'lucide-react'
import { toast } from 'sonner'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000'

// ═══════════════════════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════════════════════

interface ComparisonMetrics {
  scenario_name: string
  makespan_hours: number
  lead_time_avg_days: number
  throughput_units_per_week: number
  orders_late: number
  otd_pct: number
  total_setup_hours: number
  avg_utilization_pct: number
  bottleneck_machine: string | null
  bottleneck_utilization: number
  machine_metrics: Record<string, { utilization_pct: number; processing_hours: number }>
}

interface MetricDelta {
  baseline: number
  scenario: number
  absolute: number
  percent: number
  is_improvement: boolean
  significance: string
}

interface ComparisonResult {
  baseline: ComparisonMetrics
  scenario: ComparisonMetrics
  deltas: Record<string, MetricDelta>
  strengths: string[]
  weaknesses: string[]
  recommendations: string[]
  overall_improvement: boolean
  improvement_score: number
}

interface ReportResult {
  comparison: ComparisonResult
  report: {
    title: string
    summary: string
    key_findings: string[]
    strengths: string[]
    weaknesses: string[]
    recommendations: string[]
    conclusion: string
  }
  markdown: string
}

interface TechnicalReport {
  report: {
    title: string
    algorithm_name: string
    algorithm_description: string
    objective_function: string
    constraints: string[]
    parameters: Record<string, string>
    examples: string[]
  }
  markdown: string
}

interface AlgorithmInfo {
  id: string
  name: string
  description: string
}

// ═══════════════════════════════════════════════════════════════════════════════════════════
// API FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════════════════════

async function fetchCurrentMetrics(): Promise<ComparisonMetrics> {
  const res = await fetch(`${API_BASE_URL}/reports/current-metrics`)
  if (!res.ok) throw new Error('Failed to fetch metrics')
  return res.json()
}

async function fetchAlgorithms(): Promise<{ algorithms: AlgorithmInfo[] }> {
  const res = await fetch(`${API_BASE_URL}/reports/algorithms`)
  if (!res.ok) throw new Error('Failed to fetch algorithms')
  return res.json()
}

async function generateComparison(config: {
  scenario_name: string
  scenario_description: string
  context?: string
}): Promise<ReportResult> {
  const res = await fetch(`${API_BASE_URL}/reports/compare`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  })
  if (!res.ok) throw new Error('Failed to generate comparison')
  return res.json()
}

async function generateTechnicalExplanation(config: {
  algorithm: string
  machine_id?: string
  include_examples: boolean
}): Promise<TechnicalReport> {
  const res = await fetch(`${API_BASE_URL}/reports/technical-explanation`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  })
  if (!res.ok) throw new Error('Failed to generate explanation')
  return res.json()
}

async function compareWhatIf(scenario: string): Promise<ReportResult> {
  const res = await fetch(
    `${API_BASE_URL}/reports/compare-whatif?scenario_description=${encodeURIComponent(scenario)}`,
    { method: 'POST' }
  )
  if (!res.ok) throw new Error('Failed to compare what-if')
  return res.json()
}

// ═══════════════════════════════════════════════════════════════════════════════════════════
// COMPONENTS
// ═══════════════════════════════════════════════════════════════════════════════════════════

const MetricCard: React.FC<{
  label: string
  value: string | number
  delta?: MetricDelta
  icon?: React.ReactNode
}> = ({ label, value, delta, icon }) => (
  <div className="p-4 bg-slate-800/50 border border-slate-700/50 rounded-lg">
    <div className="flex items-center justify-between mb-2">
      <span className="text-slate-400 text-sm">{label}</span>
      {icon && <span className="text-slate-500">{icon}</span>}
    </div>
    <div className="text-2xl font-bold text-white">{value}</div>
    {delta && (
      <div className={`text-sm mt-1 flex items-center gap-1 ${
        delta.is_improvement ? 'text-emerald-400' : 'text-red-400'
      }`}>
        {delta.is_improvement ? <TrendingDown className="w-4 h-4" /> : <TrendingUp className="w-4 h-4" />}
        {delta.percent > 0 ? '+' : ''}{delta.percent.toFixed(1)}%
      </div>
    )}
  </div>
)

const InsightCard: React.FC<{
  type: 'strength' | 'weakness' | 'recommendation'
  items: string[]
}> = ({ type, items }) => {
  const config = {
    strength: {
      icon: <CheckCircle className="w-5 h-5 text-emerald-400" />,
      title: 'Pontos Fortes',
      bgColor: 'bg-emerald-900/20',
      borderColor: 'border-emerald-700/30',
    },
    weakness: {
      icon: <AlertTriangle className="w-5 h-5 text-amber-400" />,
      title: 'Pontos de Atenção',
      bgColor: 'bg-amber-900/20',
      borderColor: 'border-amber-700/30',
    },
    recommendation: {
      icon: <Lightbulb className="w-5 h-5 text-cyan-400" />,
      title: 'Recomendações',
      bgColor: 'bg-cyan-900/20',
      borderColor: 'border-cyan-700/30',
    },
  }[type]

  if (items.length === 0) return null

  return (
    <div className={`p-4 rounded-lg border ${config.bgColor} ${config.borderColor}`}>
      <div className="flex items-center gap-2 mb-3">
        {config.icon}
        <h4 className="font-medium text-white">{config.title}</h4>
      </div>
      <ul className="space-y-2">
        {items.map((item, i) => (
          <li key={i} className="text-sm text-slate-300 flex items-start gap-2">
            <ChevronRight className="w-4 h-4 mt-0.5 flex-shrink-0 text-slate-500" />
            {item}
          </li>
        ))}
      </ul>
    </div>
  )
}

const MarkdownViewer: React.FC<{ content: string }> = ({ content }) => (
  <div className="prose prose-invert prose-sm max-w-none">
    <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
  </div>
)

// ═══════════════════════════════════════════════════════════════════════════════════════════
// MAIN PAGE
// ═══════════════════════════════════════════════════════════════════════════════════════════

const Reports: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'compare' | 'technical' | 'summary'>('compare')
  const [scenarioDescription, setScenarioDescription] = useState('')
  const [selectedAlgorithm, setSelectedAlgorithm] = useState('dispatching')
  const [reportResult, setReportResult] = useState<ReportResult | null>(null)
  const [technicalResult, setTechnicalResult] = useState<TechnicalReport | null>(null)
  const [showMarkdown, setShowMarkdown] = useState(false)

  // Queries
  const { data: metrics, isLoading: metricsLoading } = useQuery({
    queryKey: ['currentMetrics'],
    queryFn: fetchCurrentMetrics,
  })

  const { data: algorithmsData } = useQuery({
    queryKey: ['algorithms'],
    queryFn: fetchAlgorithms,
  })

  // Mutations
  const compareMutation = useMutation({
    mutationFn: compareWhatIf,
    onSuccess: (data) => {
      setReportResult(data)
      toast.success('Relatório comparativo gerado!')
    },
    onError: () => toast.error('Erro ao gerar comparação'),
  })

  const technicalMutation = useMutation({
    mutationFn: generateTechnicalExplanation,
    onSuccess: (data) => {
      setTechnicalResult(data)
      toast.success('Explicação técnica gerada!')
    },
    onError: () => toast.error('Erro ao gerar explicação'),
  })

  const handleCompare = () => {
    if (!scenarioDescription.trim()) {
      toast.error('Descreva o cenário a comparar')
      return
    }
    compareMutation.mutate(scenarioDescription)
  }

  const handleTechnical = () => {
    technicalMutation.mutate({
      algorithm: selectedAlgorithm,
      include_examples: true,
    })
  }

  const downloadMarkdown = (content: string, filename: string) => {
    const blob = new Blob([content], { type: 'text/markdown' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = filename
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white">📊 Relatórios & Comparações</h1>
          <p className="text-slate-400 mt-1">
            Análise comparativa de cenários e documentação técnica
          </p>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex gap-2 p-1 bg-slate-800/50 rounded-lg w-fit">
        {[
          { id: 'compare', label: 'Comparar Cenários', icon: <GitCompare className="w-4 h-4" /> },
          { id: 'technical', label: 'Explicação Técnica', icon: <BookOpen className="w-4 h-4" /> },
          { id: 'summary', label: 'Métricas Atuais', icon: <BarChart3 className="w-4 h-4" /> },
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id as typeof activeTab)}
            className={`px-4 py-2 rounded-md text-sm font-medium transition-all flex items-center gap-2 ${
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
        {activeTab === 'compare' && (
          <motion.div
            key="compare"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="space-y-6"
          >
            {/* Input */}
            <div className="p-6 bg-slate-800/30 border border-slate-700/50 rounded-xl">
              <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                <GitCompare className="w-5 h-5 text-cyan-400" />
                Comparação de Cenários
              </h2>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm text-slate-400 mb-2">
                    Descreva o cenário a comparar com o plano atual:
                  </label>
                  <textarea
                    value={scenarioDescription}
                    onChange={(e) => setScenarioDescription(e.target.value)}
                    placeholder="Ex: Adicionar máquina M-200 com capacidade 100 un/hora para aliviar gargalo na M-101..."
                    className="w-full h-32 px-4 py-3 bg-slate-900 border border-slate-700 rounded-lg text-white placeholder-slate-500 resize-none"
                  />
                </div>
                
                <div className="flex gap-3">
                  <button
                    onClick={handleCompare}
                    disabled={compareMutation.isPending}
                    className="flex items-center gap-2 px-6 py-3 bg-cyan-600 hover:bg-cyan-500 disabled:bg-slate-700 text-white rounded-lg font-medium transition-colors"
                  >
                    {compareMutation.isPending ? (
                      <RefreshCw className="w-5 h-5 animate-spin" />
                    ) : (
                      <FileText className="w-5 h-5" />
                    )}
                    Gerar Relatório Comparativo
                  </button>
                </div>
              </div>
            </div>

            {/* Results */}
            {reportResult && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="space-y-6"
              >
                {/* Toggle view */}
                <div className="flex items-center justify-between">
                  <h2 className="text-xl font-semibold text-white">{reportResult.report.title}</h2>
                  <div className="flex gap-2">
                    <button
                      onClick={() => setShowMarkdown(!showMarkdown)}
                      className="px-3 py-1.5 text-sm bg-slate-700 hover:bg-slate-600 text-white rounded transition-colors"
                    >
                      {showMarkdown ? 'Ver Resumo' : 'Ver Markdown'}
                    </button>
                    <button
                      onClick={() => downloadMarkdown(reportResult.markdown, 'relatorio-comparativo.md')}
                      className="flex items-center gap-1 px-3 py-1.5 text-sm bg-slate-700 hover:bg-slate-600 text-white rounded transition-colors"
                    >
                      <Download className="w-4 h-4" />
                      Download
                    </button>
                  </div>
                </div>

                {showMarkdown ? (
                  <div className="p-6 bg-slate-800/50 border border-slate-700/50 rounded-xl overflow-auto max-h-[600px]">
                    <MarkdownViewer content={reportResult.markdown} />
                  </div>
                ) : (
                  <>
                    {/* Summary */}
                    <div className="p-6 bg-slate-800/30 border border-slate-700/50 rounded-xl">
                      <h3 className="text-lg font-medium text-white mb-3">Resumo Executivo</h3>
                      <p className="text-slate-300">{reportResult.report.summary}</p>
                    </div>

                    {/* Score */}
                    <div className={`p-4 rounded-lg border ${
                      reportResult.comparison.overall_improvement
                        ? 'bg-emerald-900/20 border-emerald-700/30'
                        : 'bg-amber-900/20 border-amber-700/30'
                    }`}>
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                          {reportResult.comparison.overall_improvement ? (
                            <CheckCircle className="w-8 h-8 text-emerald-400" />
                          ) : (
                            <AlertTriangle className="w-8 h-8 text-amber-400" />
                          )}
                          <div>
                            <p className="font-medium text-white">
                              {reportResult.comparison.overall_improvement ? 'Melhoria Global' : 'Sem Melhoria Significativa'}
                            </p>
                            <p className="text-sm text-slate-400">
                              Score de melhoria: {reportResult.comparison.improvement_score.toFixed(1)} pontos
                            </p>
                          </div>
                        </div>
                        <div className="text-4xl font-bold text-white">
                          {reportResult.comparison.improvement_score > 0 ? '+' : ''}
                          {reportResult.comparison.improvement_score.toFixed(0)}
                        </div>
                      </div>
                    </div>

                    {/* Metrics comparison */}
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      {Object.entries(reportResult.comparison.deltas).slice(0, 8).map(([key, delta]) => (
                        <MetricCard
                          key={key}
                          label={key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
                          value={delta.scenario.toFixed(1)}
                          delta={delta}
                        />
                      ))}
                    </div>

                    {/* Insights */}
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      <InsightCard type="strength" items={reportResult.comparison.strengths} />
                      <InsightCard type="weakness" items={reportResult.comparison.weaknesses} />
                      <InsightCard type="recommendation" items={reportResult.comparison.recommendations} />
                    </div>

                    {/* Conclusion */}
                    <div className="p-6 bg-slate-800/30 border border-slate-700/50 rounded-xl">
                      <h3 className="text-lg font-medium text-white mb-3">Conclusão</h3>
                      <p className="text-slate-300">{reportResult.report.conclusion}</p>
                    </div>
                  </>
                )}
              </motion.div>
            )}
          </motion.div>
        )}

        {activeTab === 'technical' && (
          <motion.div
            key="technical"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="space-y-6"
          >
            {/* Input */}
            <div className="p-6 bg-slate-800/30 border border-slate-700/50 rounded-xl">
              <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                <BookOpen className="w-5 h-5 text-cyan-400" />
                Explicação Técnica dos Algoritmos
              </h2>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm text-slate-400 mb-2">
                    Selecione o algoritmo:
                  </label>
                  <select
                    value={selectedAlgorithm}
                    onChange={(e) => setSelectedAlgorithm(e.target.value)}
                    className="w-full px-4 py-3 bg-slate-900 border border-slate-700 rounded-lg text-white"
                  >
                    {algorithmsData?.algorithms.map((alg) => (
                      <option key={alg.id} value={alg.id}>{alg.name}</option>
                    )) ?? (
                      <>
                        <option value="dispatching">Regras de Dispatching</option>
                        <option value="flow_shop">Flow Shop Scheduling</option>
                        <option value="setup_optimization">Otimização de Setups</option>
                      </>
                    )}
                  </select>
                </div>
                
                <button
                  onClick={handleTechnical}
                  disabled={technicalMutation.isPending}
                  className="flex items-center gap-2 px-6 py-3 bg-cyan-600 hover:bg-cyan-500 disabled:bg-slate-700 text-white rounded-lg font-medium transition-colors"
                >
                  {technicalMutation.isPending ? (
                    <RefreshCw className="w-5 h-5 animate-spin" />
                  ) : (
                    <Settings className="w-5 h-5" />
                  )}
                  Gerar Explicação
                </button>
              </div>
            </div>

            {/* Technical Report */}
            {technicalResult && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="p-6 bg-slate-800/30 border border-slate-700/50 rounded-xl"
              >
                <div className="flex items-center justify-between mb-6">
                  <h2 className="text-xl font-semibold text-white">{technicalResult.report.title}</h2>
                  <button
                    onClick={() => downloadMarkdown(technicalResult.markdown, `explicacao-${selectedAlgorithm}.md`)}
                    className="flex items-center gap-1 px-3 py-1.5 text-sm bg-slate-700 hover:bg-slate-600 text-white rounded transition-colors"
                  >
                    <Download className="w-4 h-4" />
                    Download
                  </button>
                </div>

                <div className="prose prose-invert max-w-none">
                  <MarkdownViewer content={technicalResult.markdown} />
                </div>
              </motion.div>
            )}
          </motion.div>
        )}

        {activeTab === 'summary' && (
          <motion.div
            key="summary"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="space-y-6"
          >
            <div className="p-6 bg-slate-800/30 border border-slate-700/50 rounded-xl">
              <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                <BarChart3 className="w-5 h-5 text-cyan-400" />
                Métricas do Plano Atual
              </h2>

              {metricsLoading ? (
                <div className="text-center py-8 text-slate-500">
                  <RefreshCw className="w-8 h-8 mx-auto animate-spin mb-2" />
                  A carregar métricas...
                </div>
              ) : metrics ? (
                <div className="space-y-6">
                  {/* Main metrics */}
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <MetricCard
                      label="Makespan"
                      value={`${metrics.makespan_hours.toFixed(0)}h`}
                      icon={<Zap className="w-4 h-4" />}
                    />
                    <MetricCard
                      label="Lead Time Médio"
                      value={`${metrics.lead_time_avg_days.toFixed(1)}d`}
                      icon={<BarChart3 className="w-4 h-4" />}
                    />
                    <MetricCard
                      label="Throughput/Semana"
                      value={`${Math.round(metrics.throughput_units_per_week)}`}
                      icon={<TrendingUp className="w-4 h-4" />}
                    />
                    <MetricCard
                      label="OTD"
                      value={`${metrics.otd_pct.toFixed(0)}%`}
                      icon={<CheckCircle className="w-4 h-4" />}
                    />
                  </div>

                  {/* Secondary metrics */}
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <MetricCard
                      label="Ordens Atrasadas"
                      value={metrics.orders_late}
                      icon={<AlertTriangle className="w-4 h-4" />}
                    />
                    <MetricCard
                      label="Setup Total"
                      value={`${metrics.total_setup_hours.toFixed(1)}h`}
                      icon={<Settings className="w-4 h-4" />}
                    />
                    <MetricCard
                      label="Utilização Média"
                      value={`${metrics.avg_utilization_pct.toFixed(0)}%`}
                      icon={<BarChart3 className="w-4 h-4" />}
                    />
                    <MetricCard
                      label="Gargalo"
                      value={metrics.bottleneck_machine ?? '-'}
                      icon={<Info className="w-4 h-4" />}
                    />
                  </div>

                  {/* Machine utilization */}
                  {metrics.machine_metrics && Object.keys(metrics.machine_metrics).length > 0 && (
                    <div>
                      <h3 className="text-md font-medium text-white mb-3">Utilização por Máquina</h3>
                      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-2">
                        {Object.entries(metrics.machine_metrics).map(([machineId, m]) => (
                          <div
                            key={machineId}
                            className={`p-3 rounded-lg border ${
                              m.utilization_pct > 85
                                ? 'bg-red-900/20 border-red-700/30'
                                : m.utilization_pct > 60
                                ? 'bg-amber-900/20 border-amber-700/30'
                                : 'bg-slate-800/50 border-slate-700/50'
                            }`}
                          >
                            <p className="text-xs text-slate-400">{machineId}</p>
                            <p className="text-lg font-bold text-white">{m.utilization_pct.toFixed(0)}%</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div className="text-center py-8 text-slate-500">
                  Nenhuma métrica disponível
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

export default Reports



