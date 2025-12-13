/**
 * R&D Dashboard - SIFIDE Research Programme
 * 
 * Work Packages:
 * - WP1: Routing Intelligence
 * - WP2: Suggestion Evaluation
 * - WP3: Inventory & Capacity Optimization
 * - WP4: Learning-Based Scheduler (Multi-Armed Bandit)
 */

import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { motion } from 'framer-motion'
import toast from 'react-hot-toast'
import { API_BASE_URL } from '../config/api'

// -------------------------
// Types
// -------------------------

type WorkPackage = 'overview' | 'wp1' | 'wp2' | 'wp3' | 'wp4' | 'reports'

type RDStatus = {
  available: boolean
  version: string
  work_packages: string[]
  experiments_summary: {
    total_experiments: number
    by_work_package: Record<string, { total: number; by_status: Record<string, number> }>
  }
}

type RDExperiment = {
  id: number
  wp: string
  name: string
  status: string
  created_at: string
  conclusion?: string
  kpis?: Record<string, any>
  summary?: Record<string, any>
}

type WP1PolicyResult = {
  policy: string
  makespan_hours: number
  tardiness_hours: number
  setup_hours: number
  otd_rate: number
  num_late_orders: number
  total_operations: number
  solve_time_sec: number
  vs_baseline_makespan_pct?: number | null
  vs_baseline_tardiness_pct?: number | null
}

type WP1Result = {
  experiment_id: number
  name: string
  status: string
  policies_tested: string[]
  baseline_policy: string
  results: WP1PolicyResult[]
  best_policy?: string
  improvement_vs_baseline_pct?: number
  conclusion?: string
  total_time_sec: number
}

type WP2SuggestionRecord = {
  suggestion_id: number
  suggestion_type: string
  origin: string
  title: string
  description: string
  created_at: string
  label: string
  delta_otd_pct?: number | null
  delta_tardiness_pct?: number | null
  evaluated: boolean
}

type WP2EvaluationResult = {
  experiment_id: number
  suggestion_id: number
  label: string
  pre_kpis: Record<string, number>
  post_kpis: Record<string, number>
  delta_otd_pct: number
  delta_tardiness_pct: number
  delta_makespan_pct: number
  conclusion: string
}

type WP2BatchResult = {
  experiment_id: number
  name: string
  total_suggestions: number
  evaluated_count: number
  beneficial_count: number
  neutral_count: number
  harmful_count: number
  overall_precision: number
  overall_recall: number
  f1_score: number
  conclusion: string
}

type WP3PolicyResult = {
  policy_name: string
  inventory_kpis: {
    avg_stock_qty: number
    avg_stock_value_eur: number
    stockout_days: number
    stockout_events: number
    service_level_pct: number
    inventory_turns: number
    days_of_supply: number
  }
  scheduling_kpis: {
    avg_otd_rate: number
    total_tardiness_hours: number
  }
  total_cost_eur: number
  vs_baseline_cost_pct?: number | null
  vs_baseline_service_pct?: number | null
}

type WP3Result = {
  experiment_id: number
  name: string
  status: string
  policies_tested: string[]
  results: WP3PolicyResult[]
  best_policy?: string
  recommendation?: string
  total_time_sec: number
}

type WP4PolicyStats = {
  policy: string
  num_pulls: number
  total_reward: number
  avg_reward: number
  std_reward: number
  ucb_value: number
}

type WP4EpisodeResult = {
  episode_num: number
  policy_selected: string
  reward: number
  regret: number
  kpis: Record<string, number>
  baseline_reward: number
}

type WP4Result = {
  experiment_id: number
  name: string
  status: string
  num_episodes: number
  policies: string[]
  baseline_policy: string
  episodes: WP4EpisodeResult[]
  policy_stats: WP4PolicyStats[]
  best_policy: string
  avg_reward: number
  avg_regret: number
  cumulative_regret: number
  conclusion: string
  total_time_sec: number
}

// SIFIDE Report Types
type WP1Summary = {
  num_experiments: number
  policies_tested: string[]
  baseline_policy: string
  avg_makespan_improvement_pct: number
  avg_tardiness_improvement_pct: number
  avg_otd_improvement_pct: number
  best_policy: string
  best_makespan_improvement_pct: number
  experiments_list: Array<{
    id: number
    date: string
    policies: string[]
    best_policy: string
    makespan_improvement_pct: number
  }>
}

type WP2Summary = {
  num_suggestions_evaluated: number
  num_beneficial: number
  num_neutral: number
  num_harmful: number
  pct_beneficial: number
  pct_neutral: number
  pct_harmful: number
  suggestion_types: string[]
  avg_otd_delta: number
  avg_tardiness_delta: number
  experiments_list: Array<{
    id: number
    date: string
    suggestion_type: string
    label: string
    delta_otd: number
  }>
}

type WP3Summary = {
  num_experiments: number
  policies_tested: string[]
  avg_total_cost: number
  avg_stockouts: number
  avg_otd: number
  best_policy: string
  best_cost: number
  policy_comparison: Array<{
    policy: string
    avg_cost: number
    avg_stockouts: number
    avg_otd_pct: number
    num_runs: number
  }>
  experiments_list: Array<{
    id: number
    date: string
    policy: string
    total_cost: number
    stockouts: number
    otd_pct: number
  }>
}

type WP4Summary = {
  num_episodes: number
  policies_in_bandit: string[]
  avg_reward: number
  avg_regret: number
  best_observed_policy: string
  convergence_episode: number | null
  experiments_list: Array<{
    id: number
    date: string
    selected_policy: string
    reward: number
    regret: number
  }>
}

type HypothesisSummary = {
  wp: string
  hypothesis: string
  result: string
  evidence: string
}

type RDReportSummary = {
  period: { start: string; end: string }
  generated_at: string
  total_experiments: number
  wp1: WP1Summary
  wp2: WP2Summary
  wp3: WP3Summary
  wp4: WP4Summary
  hypotheses_summary: HypothesisSummary[]
  is_demo?: boolean
}

// -------------------------
// API Functions
// -------------------------

async function fetchRDStatus(): Promise<RDStatus> {
  const res = await fetch(`${API_BASE_URL}/rd/status`)
  if (!res.ok) return { available: false, version: '0.0.0', work_packages: [], experiments_summary: { total_experiments: 0, by_work_package: {} } }
  return res.json()
}

async function fetchExperiments(wp?: string): Promise<RDExperiment[]> {
  const url = wp ? `${API_BASE_URL}/rd/experiments?wp=${wp}` : `${API_BASE_URL}/rd/experiments`
  const res = await fetch(url)
  if (!res.ok) return []
  return res.json()
}

async function runWP1Experiment(request: { name: string; policies: string[]; baseline_policy: string }): Promise<WP1Result> {
  const res = await fetch(`${API_BASE_URL}/rd/wp1/run`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  })
  if (!res.ok) throw new Error('Erro ao executar WP1')
  return res.json()
}

async function fetchWP2Suggestions(): Promise<WP2SuggestionRecord[]> {
  const res = await fetch(`${API_BASE_URL}/rd/wp2/suggestions?time_window_days=30`)
  if (!res.ok) return []
  return res.json()
}

async function runWP2BatchEvaluation(request: { name: string; time_window_days: number }): Promise<WP2BatchResult> {
  const res = await fetch(`${API_BASE_URL}/rd/wp2/evaluate-batch`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  })
  if (!res.ok) throw new Error('Erro ao executar WP2')
  return res.json()
}

async function runWP3Comparison(request: { 
  name: string; 
  policies: Array<{ name: string; rop_multiplier: number; safety_stock_multiplier: number; target_service_level: number }>;
  baseline_name: string;
  horizon: { start: string; end: string };
}): Promise<WP3Result> {
  const res = await fetch(`${API_BASE_URL}/rd/wp3/compare`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  })
  if (!res.ok) throw new Error('Erro ao executar WP3')
  return res.json()
}

async function runWP4Episode(request: { 
  name: string; 
  policies: string[];
  baseline_policy: string;
  num_episodes: number;
  bandit_type: string;
  epsilon: number;
  reward_type: string;
}): Promise<WP4Result> {
  const res = await fetch(`${API_BASE_URL}/rd/wp4/run-episode`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  })
  if (!res.ok) throw new Error('Erro ao executar WP4')
  return res.json()
}

// SIFIDE Report API Functions
async function fetchReportSummary(start: string, end: string): Promise<RDReportSummary> {
  const res = await fetch(`${API_BASE_URL}/rd/report/summary?start=${start}&end=${end}`)
  if (!res.ok) throw new Error('Erro ao obter resumo R&D')
  return res.json()
}

async function fetchAvailableYears(): Promise<{ years: number[] }> {
  const res = await fetch(`${API_BASE_URL}/rd/report/years`)
  if (!res.ok) return { years: [new Date().getFullYear()] }
  return res.json()
}

async function exportReport(start: string, end: string, format: 'json' | 'pdf'): Promise<void> {
  const res = await fetch(`${API_BASE_URL}/rd/report/export?start=${start}&end=${end}&format=${format}`)
  if (!res.ok) throw new Error('Erro ao exportar relatório')
  
  const blob = await res.blob()
  const filename = `rd_report_${start}_${end}.${format}`
  
  // Download file
  const url = window.URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  window.URL.revokeObjectURL(url)
  document.body.removeChild(a)
}

// -------------------------
// Components
// -------------------------

const TabButton: React.FC<{
  active: boolean
  onClick: () => void
  children: React.ReactNode
  color?: string
}> = ({ active, onClick, children }) => (
  <button
    onClick={onClick}
    className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
      active
        ? 'bg-nikufra text-background'
        : 'bg-background border border-border text-text-muted hover:border-nikufra/50'
    }`}
  >
    {children}
  </button>
)

const SectionCard: React.FC<{ 
  title: string
  subtitle?: string
  children: React.ReactNode
  color?: string
}> = ({ title, subtitle, children, color }) => (
  <motion.div
    initial={{ opacity: 0, y: 10 }}
    animate={{ opacity: 1, y: 0 }}
    className={`rounded-2xl border bg-surface p-6 ${color ? `border-${color}/40` : 'border-border'}`}
  >
    <div className="mb-4">
      <h3 className="text-lg font-bold text-text-primary">{title}</h3>
      {subtitle && <p className="text-xs text-text-muted">{subtitle}</p>}
    </div>
    {children}
  </motion.div>
)

const MetricCard: React.FC<{
  label: string
  value: string | number
  color?: 'default' | 'success' | 'warning' | 'danger'
}> = ({ label, value, color = 'default' }) => {
  const colors = {
    default: 'border-border bg-background',
    success: 'border-green-500/40 bg-green-500/10',
    warning: 'border-amber-500/40 bg-amber-500/10',
    danger: 'border-red-500/40 bg-red-500/10',
  }
  return (
    <div className={`rounded-xl border p-4 ${colors[color]}`}>
      <p className="text-xs text-text-muted">{label}</p>
      <p className="text-xl font-bold text-text-primary">{value}</p>
    </div>
  )
}

const ExperimentCard: React.FC<{ experiment: RDExperiment }> = ({ experiment }) => {
  const statusColors: Record<string, string> = {
    finished: 'bg-green-500/20 text-green-400',
    running: 'bg-amber-500/20 text-amber-400',
    failed: 'bg-red-500/20 text-red-400',
    created: 'bg-blue-500/20 text-blue-400',
  }
  
  return (
    <div className="rounded-lg border border-border/50 bg-background p-3">
      <div className="flex items-center justify-between mb-2">
        <span className="font-medium text-text-primary text-sm">{experiment.name}</span>
        <span className={`text-xs px-2 py-0.5 rounded ${statusColors[experiment.status] || 'bg-gray-500/20 text-gray-400'}`}>
          {experiment.status}
        </span>
      </div>
      <p className="text-xs text-text-muted">{experiment.wp}</p>
      {experiment.conclusion && (
        <p className="text-xs text-text-muted mt-2 italic">"{experiment.conclusion}"</p>
      )}
    </div>
  )
}

// -------------------------
// Work Package Components
// -------------------------

const WP1Section: React.FC = () => {
  const [policies, setPolicies] = useState(['FIFO', 'SPT', 'EDD'])
  const [baseline, setBaseline] = useState('FIFO')
  const [result, setResult] = useState<WP1Result | null>(null)
  const queryClient = useQueryClient()
  
  const mutation = useMutation({
    mutationFn: () => runWP1Experiment({ 
      name: `WP1_Routing_${Date.now()}`, 
      policies,
      baseline_policy: baseline
    }),
    onSuccess: (data) => {
      setResult(data)
      queryClient.invalidateQueries({ queryKey: ['rd-experiments-all'] })
      toast.success('WP1 Experiment completed!')
    },
    onError: () => toast.error('Error running WP1 experiment'),
  })
  
  const allPolicies = ['FIFO', 'SPT', 'EDD', 'CR', 'WSPT', 'MILP', 'CPSAT']
  
  return (
    <div className="space-y-6">
      <SectionCard title="WP1 - Routing Intelligence" subtitle="Compare dispatching rules and optimization engines">
        <div className="space-y-4">
          <div>
            <p className="text-sm text-text-muted mb-2">Select policies to compare:</p>
            <div className="flex flex-wrap gap-2">
              {allPolicies.map((p) => (
                <label
                  key={p}
                  className={`cursor-pointer px-3 py-1.5 rounded-lg text-sm transition ${
                    policies.includes(p)
                      ? 'bg-nikufra text-background'
                      : 'bg-background border border-border text-text-muted hover:border-nikufra/50'
                  }`}
                >
                  <input
                    type="checkbox"
                    checked={policies.includes(p)}
                    onChange={(e) => {
                      if (e.target.checked) {
                        setPolicies([...policies, p])
                      } else {
                        setPolicies(policies.filter((x) => x !== p))
                      }
                    }}
                    className="hidden"
                  />
                  {p}
                </label>
              ))}
            </div>
          </div>
          
          <div>
            <label className="text-sm text-text-muted">Baseline Policy</label>
            <select
              value={baseline}
              onChange={(e) => setBaseline(e.target.value)}
              className="w-full mt-1 rounded-lg border border-border bg-background p-2 text-sm text-text-primary"
            >
              {policies.map((p) => (
                <option key={p} value={p}>{p}</option>
              ))}
            </select>
          </div>
          
          <button
            onClick={() => mutation.mutate()}
            disabled={mutation.isPending || policies.length < 2}
            className="w-full rounded-xl bg-nikufra px-4 py-3 font-semibold text-background transition hover:bg-nikufra-hover disabled:opacity-50"
          >
            {mutation.isPending ? '🔄 Running...' : '🔬 Run WP1 Experiment'}
          </button>
        </div>
        
        {result && (
          <div className="mt-6 pt-4 border-t border-border">
            <p className="text-sm font-semibold text-text-primary mb-3">Results:</p>
            <div className="grid grid-cols-3 gap-3 mb-4">
              <MetricCard label="Best Policy" value={result.best_policy || '-'} color="success" />
              <MetricCard label="Improvement" value={`${result.improvement_vs_baseline_pct?.toFixed(1) || 0}%`} color="success" />
              <MetricCard label="Time (sec)" value={result.total_time_sec.toFixed(1)} />
            </div>
            
            {/* Results Table */}
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-border text-text-muted">
                    <th className="text-left py-2">Policy</th>
                    <th className="text-right py-2">Makespan (h)</th>
                    <th className="text-right py-2">Tardiness (h)</th>
                    <th className="text-right py-2">OTD</th>
                    <th className="text-right py-2">vs Baseline</th>
                  </tr>
                </thead>
                <tbody>
                  {result.results.map((r) => (
                    <tr key={r.policy} className={`border-b border-border/50 ${r.policy === result.best_policy ? 'bg-green-500/10' : ''}`}>
                      <td className="py-2 font-medium text-text-primary">{r.policy}</td>
                      <td className="text-right py-2 text-text-muted">{r.makespan_hours.toFixed(1)}</td>
                      <td className="text-right py-2 text-text-muted">{r.tardiness_hours.toFixed(1)}</td>
                      <td className="text-right py-2 text-text-muted">{(r.otd_rate * 100).toFixed(1)}%</td>
                      <td className={`text-right py-2 ${(r.vs_baseline_makespan_pct || 0) < 0 ? 'text-green-400' : 'text-red-400'}`}>
                        {r.vs_baseline_makespan_pct != null ? `${r.vs_baseline_makespan_pct.toFixed(1)}%` : '-'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            
            {result.conclusion && (
              <p className="text-sm text-text-muted italic mt-4">"{result.conclusion}"</p>
            )}
          </div>
        )}
      </SectionCard>
      
      <div className="p-4 rounded-xl border border-nikufra/30 bg-nikufra/5">
        <p className="text-xs text-nikufra font-semibold">💡 Research Hypothesis H1.1</p>
        <p className="text-sm text-text-muted mt-1">
          Dynamic routing with intelligent heuristics can reduce makespan by ≥8% compared to FIFO baseline.
        </p>
      </div>
    </div>
  )
}

const WP2Section: React.FC = () => {
  const [windowDays, setWindowDays] = useState(30)
  const [batchResult, setBatchResult] = useState<WP2BatchResult | null>(null)
  const queryClient = useQueryClient()
  
  const { data: suggestions } = useQuery({
    queryKey: ['wp2-suggestions'],
    queryFn: fetchWP2Suggestions,
    staleTime: 30_000,
  })
  
  const mutation = useMutation({
    mutationFn: () => runWP2BatchEvaluation({ 
      name: `WP2_Batch_${Date.now()}`, 
      time_window_days: windowDays 
    }),
    onSuccess: (data) => {
      setBatchResult(data)
      queryClient.invalidateQueries({ queryKey: ['wp2-suggestions'] })
      queryClient.invalidateQueries({ queryKey: ['rd-experiments-all'] })
      toast.success('WP2 Evaluation completed!')
    },
    onError: () => toast.error('Error running WP2 evaluation'),
  })
  
  const labelColors: Record<string, string> = {
    BENEFICIAL: 'bg-green-500/20 text-green-400',
    NEUTRAL: 'bg-gray-500/20 text-gray-400',
    HARMFUL: 'bg-red-500/20 text-red-400',
    UNKNOWN: 'bg-blue-500/20 text-blue-400',
  }
  
  return (
    <div className="space-y-6">
      <SectionCard title="WP2 - Suggestion Evaluation" subtitle="Evaluate quality and impact of AI-generated suggestions">
        <div className="space-y-4">
          <div>
            <label className="text-sm text-text-muted">Time Window (days)</label>
            <select
              value={windowDays}
              onChange={(e) => setWindowDays(Number(e.target.value))}
              className="w-full mt-1 rounded-lg border border-border bg-background p-2 text-sm text-text-primary"
            >
              <option value={7}>Last 7 days</option>
              <option value={30}>Last 30 days</option>
              <option value={90}>Last 90 days</option>
            </select>
          </div>
          
          <button
            onClick={() => mutation.mutate()}
            disabled={mutation.isPending}
            className="w-full rounded-xl bg-purple-500 px-4 py-3 font-semibold text-background transition hover:bg-purple-600 disabled:opacity-50"
          >
            {mutation.isPending ? '🔄 Evaluating...' : '📊 Evaluate All Suggestions'}
          </button>
        </div>
        
        {batchResult && (
          <div className="mt-6 pt-4 border-t border-border">
            <div className="grid grid-cols-4 gap-3 mb-4">
              <MetricCard label="Total" value={batchResult.total_suggestions} />
              <MetricCard label="Beneficial" value={batchResult.beneficial_count} color="success" />
              <MetricCard label="Neutral" value={batchResult.neutral_count} />
              <MetricCard label="Harmful" value={batchResult.harmful_count} color="danger" />
            </div>
            <div className="grid grid-cols-3 gap-3">
              <MetricCard label="Precision" value={`${(batchResult.overall_precision * 100).toFixed(1)}%`} />
              <MetricCard label="Recall" value={`${(batchResult.overall_recall * 100).toFixed(1)}%`} />
              <MetricCard label="F1 Score" value={batchResult.f1_score.toFixed(2)} color="success" />
            </div>
            {batchResult.conclusion && (
              <p className="text-sm text-text-muted italic mt-4">"{batchResult.conclusion}"</p>
            )}
          </div>
        )}
      </SectionCard>
      
      {/* Suggestions Table */}
      <SectionCard title="Suggestion Log" subtitle="Recent AI-generated suggestions">
        {suggestions && suggestions.length > 0 ? (
          <div className="overflow-x-auto max-h-64 overflow-y-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-border text-text-muted sticky top-0 bg-surface">
                  <th className="text-left py-2">ID</th>
                  <th className="text-left py-2">Type</th>
                  <th className="text-left py-2">Title</th>
                  <th className="text-center py-2">Label</th>
                  <th className="text-right py-2">Δ OTD</th>
                </tr>
              </thead>
              <tbody>
                {suggestions.map((s) => (
                  <tr key={s.suggestion_id} className="border-b border-border/50">
                    <td className="py-2 text-text-muted">#{s.suggestion_id}</td>
                    <td className="py-2 text-text-muted">{s.suggestion_type}</td>
                    <td className="py-2 font-medium text-text-primary truncate max-w-[200px]">{s.title}</td>
                    <td className="py-2 text-center">
                      <span className={`px-2 py-0.5 rounded text-xs ${labelColors[s.label]}`}>
                        {s.label}
                      </span>
                    </td>
                    <td className={`text-right py-2 ${(s.delta_otd_pct || 0) > 0 ? 'text-green-400' : (s.delta_otd_pct || 0) < 0 ? 'text-red-400' : 'text-text-muted'}`}>
                      {s.delta_otd_pct != null ? `${s.delta_otd_pct.toFixed(1)}%` : '-'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="text-sm text-text-muted">No suggestions found. Run batch evaluation to generate demo data.</p>
        )}
      </SectionCard>
      
      <div className="p-4 rounded-xl border border-purple-500/30 bg-purple-500/5">
        <p className="text-xs text-purple-400 font-semibold">💡 Research Hypothesis H2.1</p>
        <p className="text-sm text-text-muted mt-1">
          AI-generated suggestions achieve ≥70% precision and provide actionable insights for production optimization.
        </p>
      </div>
    </div>
  )
}

const WP3Section: React.FC = () => {
  const [result, setResult] = useState<WP3Result | null>(null)
  const queryClient = useQueryClient()
  
  const policies = [
    { name: "Conservative", rop_multiplier: 1.3, safety_stock_multiplier: 1.5, target_service_level: 0.98 },
    { name: "Baseline", rop_multiplier: 1.0, safety_stock_multiplier: 1.0, target_service_level: 0.95 },
    { name: "Lean", rop_multiplier: 0.8, safety_stock_multiplier: 0.7, target_service_level: 0.90 },
  ]
  
  const today = new Date()
  const endDate = new Date(today)
  endDate.setDate(endDate.getDate() + 30)
  
  const mutation = useMutation({
    mutationFn: () => runWP3Comparison({ 
      name: `WP3_Comparison_${Date.now()}`,
      policies,
      baseline_name: "Baseline",
      horizon: {
        start: today.toISOString().split('T')[0],
        end: endDate.toISOString().split('T')[0],
      }
    }),
    onSuccess: (data) => {
      setResult(data)
      queryClient.invalidateQueries({ queryKey: ['rd-experiments-all'] })
      toast.success('WP3 Experiment completed!')
    },
    onError: () => toast.error('Error running WP3 experiment'),
  })
  
  return (
    <div className="space-y-6">
      <SectionCard title="WP3 - Inventory & Capacity" subtitle="Compare inventory policies and optimize stock levels">
        <div className="space-y-4">
          {/* Policy Preview */}
          <div className="grid grid-cols-3 gap-3">
            {policies.map((p) => (
              <div key={p.name} className="p-3 rounded-lg border border-border bg-background">
                <p className="font-semibold text-text-primary text-sm">{p.name}</p>
                <p className="text-xs text-text-muted">ROP: {p.rop_multiplier}x</p>
                <p className="text-xs text-text-muted">SS: {p.safety_stock_multiplier}x</p>
                <p className="text-xs text-text-muted">SL: {(p.target_service_level * 100).toFixed(0)}%</p>
              </div>
            ))}
          </div>
          
          <button
            onClick={() => mutation.mutate()}
            disabled={mutation.isPending}
            className="w-full rounded-xl bg-amber-500 px-4 py-3 font-semibold text-background transition hover:bg-amber-600 disabled:opacity-50"
          >
            {mutation.isPending ? '🔄 Running...' : '📦 Run WP3 Comparison'}
          </button>
        </div>
        
        {result && (
          <div className="mt-6 pt-4 border-t border-border">
            <p className="text-sm font-semibold text-text-primary mb-3">Results:</p>
            <div className="grid grid-cols-2 gap-3 mb-4">
              <MetricCard label="Best Policy" value={result.best_policy || '-'} color="success" />
              <MetricCard label="Time (sec)" value={result.total_time_sec.toFixed(1)} />
            </div>
            
            {/* Results Table */}
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-border text-text-muted">
                    <th className="text-left py-2">Policy</th>
                    <th className="text-right py-2">Service Level</th>
                    <th className="text-right py-2">Stockouts</th>
                    <th className="text-right py-2">Cost (€)</th>
                    <th className="text-right py-2">OTD</th>
                    <th className="text-right py-2">vs Baseline</th>
                  </tr>
                </thead>
                <tbody>
                  {result.results.map((r) => (
                    <tr key={r.policy_name} className={`border-b border-border/50 ${r.policy_name === result.best_policy ? 'bg-green-500/10' : ''}`}>
                      <td className="py-2 font-medium text-text-primary">{r.policy_name}</td>
                      <td className="text-right py-2 text-text-muted">{r.inventory_kpis.service_level_pct.toFixed(1)}%</td>
                      <td className="text-right py-2 text-text-muted">{r.inventory_kpis.stockout_days}</td>
                      <td className="text-right py-2 text-text-muted">€{r.total_cost_eur.toLocaleString()}</td>
                      <td className="text-right py-2 text-text-muted">{(r.scheduling_kpis.avg_otd_rate * 100).toFixed(1)}%</td>
                      <td className={`text-right py-2 ${(r.vs_baseline_cost_pct || 0) < 0 ? 'text-green-400' : 'text-red-400'}`}>
                        {r.vs_baseline_cost_pct != null ? `${r.vs_baseline_cost_pct.toFixed(1)}%` : '-'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            
            {result.recommendation && (
              <p className="text-sm text-text-muted italic mt-4">"{result.recommendation}"</p>
            )}
          </div>
        )}
      </SectionCard>
      
      <div className="p-4 rounded-xl border border-amber-500/30 bg-amber-500/5">
        <p className="text-xs text-amber-400 font-semibold">💡 Research Hypothesis H3.1</p>
        <p className="text-sm text-text-muted mt-1">
          Dynamic ROP with capacity-aware optimization reduces stockouts by ≥20% while maintaining service levels.
        </p>
      </div>
    </div>
  )
}

const WP4Section: React.FC = () => {
  const [policies, setPolicies] = useState(['FIFO', 'SPT', 'EDD'])
  const [baseline, setBaseline] = useState('FIFO')
  const [banditType, setBanditType] = useState('epsilon_greedy')
  const [numEpisodes, setNumEpisodes] = useState(10)
  const [epsilon, setEpsilon] = useState(0.1)
  const [result, setResult] = useState<WP4Result | null>(null)
  const queryClient = useQueryClient()
  
  const mutation = useMutation({
    mutationFn: () => runWP4Episode({ 
      name: `WP4_Bandit_${Date.now()}`,
      policies,
      baseline_policy: baseline,
      num_episodes: numEpisodes,
      bandit_type: banditType,
      epsilon,
      reward_type: 'combined',
    }),
    onSuccess: (data) => {
      setResult(data)
      queryClient.invalidateQueries({ queryKey: ['rd-experiments-all'] })
      toast.success('WP4 Experiment completed!')
    },
    onError: () => toast.error('Error running WP4 experiment'),
  })
  
  const allPolicies = ['FIFO', 'SPT', 'EDD', 'CR', 'WSPT']
  
  return (
    <div className="space-y-6">
      <SectionCard title="WP4 - Learning Scheduler" subtitle="Multi-Armed Bandit for dynamic policy selection">
        <div className="space-y-4">
          <div>
            <p className="text-sm text-text-muted mb-2">Select policies for bandit:</p>
            <div className="flex flex-wrap gap-2">
              {allPolicies.map((p) => (
                <label
                  key={p}
                  className={`cursor-pointer px-3 py-1.5 rounded-lg text-sm transition ${
                    policies.includes(p)
                      ? 'bg-green-500 text-background'
                      : 'bg-background border border-border text-text-muted hover:border-green-500/50'
                  }`}
                >
                  <input
                    type="checkbox"
                    checked={policies.includes(p)}
                    onChange={(e) => {
                      if (e.target.checked) {
                        setPolicies([...policies, p])
                      } else {
                        setPolicies(policies.filter((x) => x !== p))
                      }
                    }}
                    className="hidden"
                  />
                  {p}
                </label>
              ))}
            </div>
          </div>
          
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="text-sm text-text-muted">Bandit Type</label>
              <select
                value={banditType}
                onChange={(e) => setBanditType(e.target.value)}
                className="w-full mt-1 rounded-lg border border-border bg-background p-2 text-sm text-text-primary"
              >
                <option value="epsilon_greedy">Epsilon-Greedy</option>
                <option value="ucb1">UCB1</option>
                <option value="thompson_sampling">Thompson Sampling</option>
              </select>
            </div>
            <div>
              <label className="text-sm text-text-muted">Episodes</label>
              <select
                value={numEpisodes}
                onChange={(e) => setNumEpisodes(Number(e.target.value))}
                className="w-full mt-1 rounded-lg border border-border bg-background p-2 text-sm text-text-primary"
              >
                <option value={5}>5 (quick)</option>
                <option value={10}>10 (demo)</option>
                <option value={20}>20 (standard)</option>
                <option value={50}>50 (extensive)</option>
              </select>
            </div>
          </div>
          
          {banditType === 'epsilon_greedy' && (
            <div>
              <label className="text-sm text-text-muted">Epsilon (exploration rate): {epsilon}</label>
              <input
                type="range"
                min="0"
                max="0.5"
                step="0.05"
                value={epsilon}
                onChange={(e) => setEpsilon(Number(e.target.value))}
                className="w-full mt-1"
              />
            </div>
          )}
          
          <div>
            <label className="text-sm text-text-muted">Baseline Policy</label>
            <select
              value={baseline}
              onChange={(e) => setBaseline(e.target.value)}
              className="w-full mt-1 rounded-lg border border-border bg-background p-2 text-sm text-text-primary"
            >
              {policies.map((p) => (
                <option key={p} value={p}>{p}</option>
              ))}
            </select>
          </div>
          
          <button
            onClick={() => mutation.mutate()}
            disabled={mutation.isPending || policies.length < 2}
            className="w-full rounded-xl bg-green-500 px-4 py-3 font-semibold text-background transition hover:bg-green-600 disabled:opacity-50"
          >
            {mutation.isPending ? '🔄 Training...' : '🧠 Run WP4 Experiment'}
          </button>
        </div>
        
        {result && (
          <div className="mt-6 pt-4 border-t border-border">
            <p className="text-sm font-semibold text-text-primary mb-3">Results:</p>
            <div className="grid grid-cols-4 gap-3 mb-4">
              <MetricCard label="Best Policy" value={result.best_policy} color="success" />
              <MetricCard label="Avg Reward" value={result.avg_reward.toFixed(4)} />
              <MetricCard label="Avg Regret" value={result.avg_regret.toFixed(4)} />
              <MetricCard label="Cumulative Regret" value={result.cumulative_regret.toFixed(2)} color="warning" />
            </div>
            
            {/* Policy Stats */}
            <p className="text-xs font-semibold text-text-muted mb-2">Policy Statistics:</p>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-border text-text-muted">
                    <th className="text-left py-2">Policy</th>
                    <th className="text-right py-2">Pulls</th>
                    <th className="text-right py-2">Avg Reward</th>
                    <th className="text-right py-2">Std</th>
                    <th className="text-right py-2">UCB</th>
                  </tr>
                </thead>
                <tbody>
                  {result.policy_stats.map((s) => (
                    <tr key={s.policy} className={`border-b border-border/50 ${s.policy === result.best_policy ? 'bg-green-500/10' : ''}`}>
                      <td className="py-2 font-medium text-text-primary">{s.policy}</td>
                      <td className="text-right py-2 text-text-muted">{s.num_pulls}</td>
                      <td className="text-right py-2 text-text-muted">{s.avg_reward.toFixed(4)}</td>
                      <td className="text-right py-2 text-text-muted">{s.std_reward.toFixed(4)}</td>
                      <td className="text-right py-2 text-text-muted">{s.ucb_value < 1000 ? s.ucb_value.toFixed(4) : '∞'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            
            {/* Episode Chart (simplified) */}
            <p className="text-xs font-semibold text-text-muted mt-4 mb-2">Episode Rewards:</p>
            <div className="flex items-end gap-1 h-20">
              {result.episodes.map((ep) => (
                <div
                  key={ep.episode_num}
                  className={`flex-1 rounded-t ${ep.reward > 0 ? 'bg-green-500/60' : 'bg-red-500/60'}`}
                  style={{ height: `${Math.max(10, Math.min(100, (ep.reward + 1) * 50))}%` }}
                  title={`Episode ${ep.episode_num}: ${ep.policy_selected} (r=${ep.reward.toFixed(3)})`}
                />
              ))}
            </div>
            
            {result.conclusion && (
              <p className="text-sm text-text-muted italic mt-4">"{result.conclusion}"</p>
            )}
          </div>
        )}
      </SectionCard>
      
      <div className="p-4 rounded-xl border border-green-500/30 bg-green-500/5">
        <p className="text-xs text-green-400 font-semibold">💡 Research Hypothesis H4.1</p>
        <p className="text-sm text-text-muted mt-1">
          Multi-Armed Bandit policies can adaptively learn the best dispatching rule and minimize regret over time.
        </p>
      </div>
    </div>
  )
}

// -------------------------
// Reports Section (Contract 11 - SIFIDE)
// -------------------------

const ReportsSection: React.FC = () => {
  const currentYear = new Date().getFullYear()
  const [selectedYear, setSelectedYear] = useState(currentYear)
  const [report, setReport] = useState<RDReportSummary | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  
  const { data: yearsData } = useQuery({
    queryKey: ['rd-report-years'],
    queryFn: fetchAvailableYears,
    staleTime: 60_000,
  })
  
  const loadReport = async () => {
    setIsLoading(true)
    try {
      const start = `${selectedYear}-01-01`
      const end = `${selectedYear}-12-31`
      const data = await fetchReportSummary(start, end)
      setReport(data)
      toast.success('Relatório carregado!')
    } catch (e) {
      toast.error('Erro ao carregar relatório')
    } finally {
      setIsLoading(false)
    }
  }
  
  const handleExport = async (format: 'json' | 'pdf') => {
    try {
      const start = `${selectedYear}-01-01`
      const end = `${selectedYear}-12-31`
      await exportReport(start, end, format)
      toast.success(`Relatório ${format.toUpperCase()} exportado!`)
    } catch (e) {
      toast.error('Erro ao exportar relatório')
    }
  }
  
  const years = yearsData?.years || [currentYear, currentYear - 1]
  
  const resultColors = {
    SUPPORTED: 'bg-green-500/20 text-green-400 border-green-500/40',
    PARTIAL: 'bg-amber-500/20 text-amber-400 border-amber-500/40',
    INSUFFICIENT_DATA: 'bg-gray-500/20 text-gray-400 border-gray-500/40',
  }
  
  return (
    <div className="space-y-6">
      {/* Controls */}
      <SectionCard title="Relatório SIFIDE" subtitle="Exportar documentação de I&D para dossiê fiscal">
        <div className="space-y-4">
          <div className="flex items-end gap-4">
            <div className="flex-1">
              <label className="text-sm text-text-muted">Ano Fiscal</label>
              <select
                value={selectedYear}
                onChange={(e) => setSelectedYear(Number(e.target.value))}
                className="w-full mt-1 rounded-lg border border-border bg-background p-2 text-sm text-text-primary"
              >
                {years.map((y) => (
                  <option key={y} value={y}>{y}</option>
                ))}
              </select>
            </div>
            <button
              onClick={loadReport}
              disabled={isLoading}
              className="rounded-xl bg-nikufra px-6 py-2.5 font-semibold text-background transition hover:bg-nikufra-hover disabled:opacity-50"
            >
              {isLoading ? '🔄 A carregar...' : '📊 Carregar Resumo'}
            </button>
          </div>
          
          {report && (
            <div className="flex gap-2 pt-4 border-t border-border">
              <button
                onClick={() => handleExport('json')}
                className="flex-1 rounded-xl border border-blue-500/50 bg-blue-500/10 px-4 py-3 text-sm font-semibold text-blue-400 transition hover:bg-blue-500/20"
              >
                📥 Exportar JSON
              </button>
              <button
                onClick={() => handleExport('pdf')}
                className="flex-1 rounded-xl border border-red-500/50 bg-red-500/10 px-4 py-3 text-sm font-semibold text-red-400 transition hover:bg-red-500/20"
              >
                📄 Exportar PDF
              </button>
            </div>
          )}
        </div>
      </SectionCard>
      
      {report && (
        <>
          {/* Demo Warning */}
          {report.is_demo && (
            <div className="rounded-xl border border-amber-500/40 bg-amber-500/10 p-4">
              <p className="text-sm text-amber-400">
                ⚠️ <strong>Dados de demonstração.</strong> Execute experiências nas tabs WP1-WP4 para gerar dados reais.
              </p>
            </div>
          )}
          
          {/* Summary Header */}
          <div className="grid grid-cols-4 gap-4">
            <MetricCard label="Total Experiências" value={report.total_experiments} color="success" />
            <MetricCard label="Período" value={`${selectedYear}`} />
            <MetricCard label="Gerado em" value={report.generated_at.slice(0, 10)} />
            <MetricCard label="Work Packages" value="4" />
          </div>
          
          {/* Hypotheses Summary */}
          <SectionCard title="Resumo das Hipóteses" subtitle="Estado das hipóteses de investigação">
            <div className="space-y-3">
              {report.hypotheses_summary.map((h) => (
                <div key={h.wp} className="rounded-lg border border-border/50 bg-background p-3">
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-semibold text-text-primary text-sm">{h.wp}: {h.hypothesis}</span>
                    <span className={`px-2 py-0.5 rounded text-xs border ${resultColors[h.result as keyof typeof resultColors] || resultColors.PARTIAL}`}>
                      {h.result}
                    </span>
                  </div>
                  <p className="text-xs text-text-muted">{h.evidence}</p>
                </div>
              ))}
            </div>
          </SectionCard>
          
          {/* WP1 Summary */}
          <SectionCard title="WP1 - Routing Dinâmico" subtitle={`${report.wp1.num_experiments} experiências`} color="nikufra">
            <div className="grid grid-cols-3 gap-3 mb-4">
              <MetricCard label="Melhoria Makespan" value={`${report.wp1.avg_makespan_improvement_pct.toFixed(1)}%`} color="success" />
              <MetricCard label="Melhor Política" value={report.wp1.best_policy} />
              <MetricCard label="Máx. Melhoria" value={`${report.wp1.best_makespan_improvement_pct.toFixed(1)}%`} />
            </div>
            <p className="text-xs text-text-muted">
              Políticas testadas: {report.wp1.policies_tested.join(', ')}
            </p>
          </SectionCard>
          
          {/* WP2 Summary */}
          <SectionCard title="WP2 - Avaliação de Sugestões" subtitle={`${report.wp2.num_suggestions_evaluated} sugestões avaliadas`} color="purple-500">
            <div className="grid grid-cols-4 gap-3 mb-4">
              <MetricCard label="Benéficas" value={`${report.wp2.pct_beneficial.toFixed(0)}%`} color="success" />
              <MetricCard label="Neutras" value={`${report.wp2.pct_neutral.toFixed(0)}%`} />
              <MetricCard label="Prejudiciais" value={`${report.wp2.pct_harmful.toFixed(0)}%`} color="danger" />
              <MetricCard label="Δ OTD Médio" value={`${report.wp2.avg_otd_delta.toFixed(1)}%`} />
            </div>
            {/* Simple Bar Chart */}
            <div className="flex h-8 rounded-lg overflow-hidden">
              <div 
                className="bg-green-500 flex items-center justify-center text-xs text-white font-semibold"
                style={{ width: `${report.wp2.pct_beneficial}%` }}
              >
                {report.wp2.pct_beneficial > 10 && `${report.wp2.pct_beneficial.toFixed(0)}%`}
              </div>
              <div 
                className="bg-gray-500 flex items-center justify-center text-xs text-white font-semibold"
                style={{ width: `${report.wp2.pct_neutral}%` }}
              >
                {report.wp2.pct_neutral > 10 && `${report.wp2.pct_neutral.toFixed(0)}%`}
              </div>
              <div 
                className="bg-red-500 flex items-center justify-center text-xs text-white font-semibold"
                style={{ width: `${report.wp2.pct_harmful}%` }}
              >
                {report.wp2.pct_harmful > 10 && `${report.wp2.pct_harmful.toFixed(0)}%`}
              </div>
            </div>
          </SectionCard>
          
          {/* WP3 Summary */}
          <SectionCard title="WP3 - Inventário + Capacidade" subtitle={`${report.wp3.num_experiments} cenários simulados`} color="amber-500">
            <div className="grid grid-cols-3 gap-3 mb-4">
              <MetricCard label="Melhor Política" value={report.wp3.best_policy} color="success" />
              <MetricCard label="Custo Médio" value={`€${report.wp3.avg_total_cost.toLocaleString()}`} />
              <MetricCard label="OTD Médio" value={`${report.wp3.avg_otd.toFixed(1)}%`} />
            </div>
            {report.wp3.policy_comparison.length > 0 && (
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-border text-text-muted">
                      <th className="text-left py-2">Política</th>
                      <th className="text-right py-2">Custo Médio</th>
                      <th className="text-right py-2">Stockouts</th>
                      <th className="text-right py-2">OTD</th>
                      <th className="text-right py-2">Runs</th>
                    </tr>
                  </thead>
                  <tbody>
                    {report.wp3.policy_comparison.map((p) => (
                      <tr key={p.policy} className={`border-b border-border/50 ${p.policy === report.wp3.best_policy ? 'bg-green-500/10' : ''}`}>
                        <td className="py-2 font-medium text-text-primary">{p.policy}</td>
                        <td className="text-right py-2 text-text-muted">€{p.avg_cost.toLocaleString()}</td>
                        <td className="text-right py-2 text-text-muted">{p.avg_stockouts.toFixed(1)}</td>
                        <td className="text-right py-2 text-text-muted">{p.avg_otd_pct.toFixed(1)}%</td>
                        <td className="text-right py-2 text-text-muted">{p.num_runs}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </SectionCard>
          
          {/* WP4 Summary */}
          <SectionCard title="WP4 - Learning Scheduler" subtitle={`${report.wp4.num_episodes} episódios`} color="green-500">
            <div className="grid grid-cols-4 gap-3 mb-4">
              <MetricCard label="Reward Médio" value={report.wp4.avg_reward.toFixed(3)} />
              <MetricCard label="Regret Médio" value={report.wp4.avg_regret.toFixed(3)} />
              <MetricCard label="Melhor Política" value={report.wp4.best_observed_policy} color="success" />
              <MetricCard label="Convergência" value={report.wp4.convergence_episode ? `Ep. ${report.wp4.convergence_episode}` : 'N/A'} />
            </div>
            <p className="text-xs text-text-muted">
              Políticas no bandit: {report.wp4.policies_in_bandit.join(', ')}
            </p>
          </SectionCard>
          
          {/* SIFIDE Info */}
          <div className="p-4 rounded-xl border border-nikufra/30 bg-nikufra/5">
            <p className="text-xs text-nikufra font-semibold">📋 Documentação SIFIDE</p>
            <p className="text-sm text-text-muted mt-1">
              Este relatório pode ser anexado ao dossiê de candidatura SIFIDE como evidência de:
              atividades experimentais sistemáticas, incerteza técnica nas soluções implementadas,
              e evolução incremental das funcionalidades de otimização.
            </p>
          </div>
        </>
      )}
    </div>
  )
}

const OverviewSection: React.FC<{ status: RDStatus | undefined }> = ({ status }) => {
  const { data: experiments } = useQuery({
    queryKey: ['rd-experiments-all'],
    queryFn: () => fetchExperiments(),
    staleTime: 30_000,
  })
  
  return (
    <div className="space-y-6">
      {/* Status Card */}
      <div className={`rounded-xl border p-4 ${status?.available ? 'border-green-500/40 bg-green-500/10' : 'border-red-500/40 bg-red-500/10'}`}>
        <div className="flex items-center justify-between">
          <span className="font-semibold text-text-primary">
            {status?.available ? '✅ R&D Module Active' : '❌ R&D Module Unavailable'}
          </span>
          <span className="text-xs text-text-muted">v{status?.version}</span>
        </div>
      </div>
      
      {/* Research Questions */}
      <div className="grid gap-4 md:grid-cols-2">
        <div className="rounded-xl border border-nikufra/40 bg-nikufra/10 p-4">
          <p className="text-xs font-semibold text-nikufra">WP1 - Routing</p>
          <p className="text-sm text-text-primary mt-1">Dynamic routing + ML surpassing fixed APS?</p>
        </div>
        <div className="rounded-xl border border-purple-500/40 bg-purple-500/10 p-4">
          <p className="text-xs font-semibold text-purple-400">WP2 - Suggestions</p>
          <p className="text-sm text-text-primary mt-1">AI suggestions credible and useful?</p>
        </div>
        <div className="rounded-xl border border-amber-500/40 bg-amber-500/10 p-4">
          <p className="text-xs font-semibold text-amber-400">WP3 - Inventory</p>
          <p className="text-sm text-text-primary mt-1">Joint inventory + capacity optimization?</p>
        </div>
        <div className="rounded-xl border border-green-500/40 bg-green-500/10 p-4">
          <p className="text-xs font-semibold text-green-400">WP4 - Learning</p>
          <p className="text-sm text-text-primary mt-1">Bandit scheduler competitive with heuristics?</p>
        </div>
      </div>
      
      {/* Summary Stats */}
      <div className="grid grid-cols-4 gap-4">
        <MetricCard 
          label="Total Experiments" 
          value={status?.experiments_summary?.total_experiments || 0} 
        />
        <MetricCard 
          label="WP1" 
          value={status?.experiments_summary?.by_work_package?.WP1_ROUTING?.total || 0} 
        />
        <MetricCard 
          label="WP2" 
          value={status?.experiments_summary?.by_work_package?.WP2_SUGGESTIONS?.total || 0} 
        />
        <MetricCard 
          label="WP3+WP4" 
          value={(status?.experiments_summary?.by_work_package?.WP3_INVENTORY_CAPACITY?.total || 0) + (status?.experiments_summary?.by_work_package?.WP4_LEARNING_SCHEDULER?.total || 0)} 
        />
      </div>
      
      {/* Recent Experiments */}
      <SectionCard title="Recent Experiments" subtitle="Latest R&D experiment results">
        {experiments && experiments.length > 0 ? (
          <div className="space-y-2 max-h-64 overflow-auto">
            {experiments.slice(0, 10).map((exp) => (
              <ExperimentCard key={exp.id} experiment={exp} />
            ))}
          </div>
        ) : (
          <p className="text-sm text-text-muted">No experiments yet. Run experiments in WP1-WP4 tabs.</p>
        )}
      </SectionCard>
    </div>
  )
}

// -------------------------
// Main Component
// -------------------------

export const Research = () => {
  const [activeTab, setActiveTab] = useState<WorkPackage>('overview')
  
  const { data: status } = useQuery({
    queryKey: ['rd-status'],
    queryFn: fetchRDStatus,
    staleTime: 60_000,
  })

  return (
    <div className="space-y-8">
      {/* Header */}
      <header className="space-y-2">
        <p className="text-xs uppercase tracking-[0.4em] text-text-muted">R&D Laboratory</p>
        <h2 className="text-2xl font-semibold text-text-primary">
          ProdPlan 4.0 — Research & Development
        </h2>
        <p className="max-w-3xl text-sm text-text-muted">
          Experimental module for testing research hypotheses under the SIFIDE programme.
          Run experiments, compare strategies, and analyze results across work packages.
        </p>
      </header>

      {/* Tab Navigation */}
      <div className="flex flex-wrap gap-2">
        <TabButton active={activeTab === 'overview'} onClick={() => setActiveTab('overview')}>
          📊 Overview
        </TabButton>
        <TabButton active={activeTab === 'wp1'} onClick={() => setActiveTab('wp1')}>
          🛤️ WP1 Routing
        </TabButton>
        <TabButton active={activeTab === 'wp2'} onClick={() => setActiveTab('wp2')}>
          💡 WP2 Suggestions
        </TabButton>
        <TabButton active={activeTab === 'wp3'} onClick={() => setActiveTab('wp3')}>
          📦 WP3 Inventory
        </TabButton>
        <TabButton active={activeTab === 'wp4'} onClick={() => setActiveTab('wp4')}>
          🧠 WP4 Learning
        </TabButton>
        <TabButton active={activeTab === 'reports'} onClick={() => setActiveTab('reports')}>
          📋 Relatórios
        </TabButton>
      </div>

      {/* Content */}
      <motion.div
        key={activeTab}
        initial={{ opacity: 0, x: 10 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.2 }}
      >
        {activeTab === 'overview' && <OverviewSection status={status} />}
        {activeTab === 'wp1' && <WP1Section />}
        {activeTab === 'wp2' && <WP2Section />}
        {activeTab === 'wp3' && <WP3Section />}
        {activeTab === 'wp4' && <WP4Section />}
        {activeTab === 'reports' && <ReportsSection />}
      </motion.div>
    </div>
  )
}

export default Research
