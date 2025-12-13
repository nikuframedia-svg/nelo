/**
 * ════════════════════════════════════════════════════════════════════════════════════════════════════
 * DIGITAL TWIN PRODUCT - XAI-DT for Product Conformance
 * ════════════════════════════════════════════════════════════════════════════════════════════════════
 *
 * Contract 6 Implementation: Product Digital Twin with XAI
 *
 * Features:
 * - Scan vs CAD conformance analysis
 * - XAI explanations for deviations
 * - Golden runs and process optimization
 * - Conformance history tracking
 */

import React, { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Box,
  Scan,
  AlertTriangle,
  CheckCircle,
  TrendingDown,
  RefreshCw,
  ChevronRight,
  Target,
  Award,
  BarChart3,
  FileSearch,
  Settings2,
  Sparkles,
  X,
} from 'lucide-react'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'

// ═══════════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════════

interface ConformanceSnapshot {
  id: number
  scan_id: string
  max_dev: number
  mean_dev: number
  rms_dev: number
  scalar_error_score: number
  conformity_status: 'IN_TOLERANCE' | 'OUT_OF_TOLERANCE' | 'CRITICAL'
  explanation: {
    probable_causes: Array<{
      parameter: string
      impact: number
      action: string
      confidence: number
    }>
    recommendations: string[]
    dominant_modes: number[]
  } | null
  machine_id: string | null
  created_at: string
}

interface ConformanceSummary {
  revision_id: number
  total_scans: number
  in_tolerance: number
  out_of_tolerance: number
  critical: number
  avg_error_score: number
  compliance_rate: number
}

interface GoldenRun {
  id: number
  revision_id: number
  operation_id: number | null
  machine_id: string | null
  process_params: Record<string, number>
  kpis: Record<string, number> | null
  score: number
}

interface ParamSuggestion {
  revision_id: number
  operation_id: number | null
  machine_id: string | null
  suggested_params: Record<string, number>
  based_on_golden_runs: number
  expected_quality_score: number
}

interface AnalyzeScanResponse {
  snapshot_id: number
  scan_id: string
  max_dev: number
  mean_dev: number
  rms_dev: number
  scalar_error_score: number
  conformity_status: string
  probable_causes: Array<{
    parameter: string
    impact: number
    action: string
    confidence: number
  }>
  recommendations: string[]
  dominant_modes: number[]
}

interface ProductItem {
  id: number
  item_number: string
  description: string | null
  active_revision_id: number | null
  active_revision_label: string
}

// ═══════════════════════════════════════════════════════════════════════════════
// API CALLS
// ═══════════════════════════════════════════════════════════════════════════════

async function fetchProducts(): Promise<ProductItem[]> {
  const res = await fetch(`${API_BASE_URL}/duplios/items?limit=50`)
  if (!res.ok) throw new Error('Failed to fetch products')
  const data = await res.json()
  return data.map((item: any) => ({
    id: item.id,
    item_number: item.item_number,
    description: item.description,
    active_revision_id: item.active_revision?.id || null,
    active_revision_label: item.active_revision?.revision_label || 'N/A',
  }))
}

async function fetchConformanceSummary(revisionId: number): Promise<ConformanceSummary> {
  const res = await fetch(`${API_BASE_URL}/digital-twin/product/${revisionId}/conformance/summary`)
  if (!res.ok) {
    // Return empty summary if API not available
    return {
      revision_id: revisionId,
      total_scans: 0,
      in_tolerance: 0,
      out_of_tolerance: 0,
      critical: 0,
      avg_error_score: 0,
      compliance_rate: 0,
    }
  }
  return res.json()
}

async function fetchConformanceHistory(revisionId: number): Promise<ConformanceSnapshot[]> {
  const res = await fetch(`${API_BASE_URL}/digital-twin/product/${revisionId}/conformance?limit=20`)
  if (!res.ok) return []
  return res.json()
}

async function fetchGoldenRuns(revisionId: number): Promise<GoldenRun[]> {
  const res = await fetch(`${API_BASE_URL}/digital-twin/product/${revisionId}/golden-runs?limit=10`)
  if (!res.ok) return []
  return res.json()
}

async function fetchParamSuggestion(revisionId: number): Promise<ParamSuggestion> {
  const res = await fetch(`${API_BASE_URL}/digital-twin/product/${revisionId}/suggest-params`)
  if (!res.ok) throw new Error('Failed to fetch suggestion')
  return res.json()
}

async function analyzeScan(
  revisionId: number,
  scanId: string,
  processParams: Record<string, number>
): Promise<AnalyzeScanResponse> {
  const res = await fetch(`${API_BASE_URL}/digital-twin/product/${revisionId}/analyze-scan`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      scan_id: scanId,
      process_params: processParams,
      machine_id: 'CNC-01',
    }),
  })
  if (!res.ok) throw new Error('Failed to analyze scan')
  return res.json()
}

async function computeGoldenRuns(revisionId: number): Promise<any> {
  const res = await fetch(`${API_BASE_URL}/digital-twin/product/${revisionId}/golden-runs/compute`, {
    method: 'POST',
  })
  if (!res.ok) throw new Error('Failed to compute golden runs')
  return res.json()
}

// ═══════════════════════════════════════════════════════════════════════════════
// COMPONENTS
// ═══════════════════════════════════════════════════════════════════════════════

const StatusBadge: React.FC<{ status: string }> = ({ status }) => {
  const config: Record<string, { bg: string; text: string; icon: typeof CheckCircle }> = {
    IN_TOLERANCE: { bg: 'bg-emerald-500/20', text: 'text-emerald-400', icon: CheckCircle },
    OUT_OF_TOLERANCE: { bg: 'bg-amber-500/20', text: 'text-amber-400', icon: AlertTriangle },
    CRITICAL: { bg: 'bg-red-500/20', text: 'text-red-400', icon: AlertTriangle },
  }
  const c = config[status] || { bg: 'bg-slate-500/20', text: 'text-slate-400', icon: CheckCircle }
  const Icon = c.icon

  const label = status === 'IN_TOLERANCE' ? 'Conforme' : status === 'OUT_OF_TOLERANCE' ? 'Fora Tol.' : 'Crítico'

  return (
    <span className={`inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium ${c.bg} ${c.text}`}>
      <Icon className="w-3 h-3" />
      {label}
    </span>
  )
}

const ScoreGauge: React.FC<{ score: number; label: string }> = ({ score, label }) => {
  // score is error score (0-100), lower is better
  const quality = Math.max(0, 100 - score)
  const color = quality >= 80 ? '#10b981' : quality >= 60 ? '#f59e0b' : quality >= 40 ? '#f97316' : '#ef4444'

  return (
    <div className="text-center">
      <div className="relative w-24 h-24 mx-auto">
        <svg viewBox="0 0 36 36" className="w-24 h-24 transform -rotate-90">
          <path
            d="M18 2.0845
              a 15.9155 15.9155 0 0 1 0 31.831
              a 15.9155 15.9155 0 0 1 0 -31.831"
            fill="none"
            stroke="#1e293b"
            strokeWidth="3"
          />
          <path
            d="M18 2.0845
              a 15.9155 15.9155 0 0 1 0 31.831
              a 15.9155 15.9155 0 0 1 0 -31.831"
            fill="none"
            stroke={color}
            strokeWidth="3"
            strokeDasharray={`${quality}, 100`}
            strokeLinecap="round"
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-2xl font-bold text-white">{quality.toFixed(0)}%</span>
        </div>
      </div>
      <p className="text-sm text-slate-400 mt-2">{label}</p>
    </div>
  )
}

const ProductCard: React.FC<{
  product: ProductItem
  isSelected: boolean
  onClick: () => void
}> = ({ product, isSelected, onClick }) => (
  <motion.div
    whileHover={{ scale: 1.01 }}
    onClick={onClick}
    className={`cursor-pointer p-4 rounded-xl border transition-all ${
      isSelected
        ? 'border-cyan-500 bg-cyan-500/10'
        : 'border-slate-700/50 bg-slate-800/30 hover:border-cyan-500/50'
    }`}
  >
    <div className="flex items-center gap-3">
      <Box className={`w-5 h-5 ${isSelected ? 'text-cyan-400' : 'text-slate-400'}`} />
      <div className="flex-1 min-w-0">
        <p className="font-semibold text-white truncate">{product.item_number}</p>
        <p className="text-xs text-slate-400 truncate">{product.description || 'Sem descrição'}</p>
      </div>
      <div className="text-right">
        <span className="text-xs text-cyan-400">Rev. {product.active_revision_label}</span>
      </div>
    </div>
  </motion.div>
)

const MetricCard: React.FC<{
  icon: React.ReactNode
  label: string
  value: string | number
  color?: string
}> = ({ icon, label, value, color = 'cyan' }) => (
  <div className="p-4 rounded-xl bg-slate-800/50 border border-slate-700/50">
    <div className={`p-2 rounded-lg bg-${color}-500/20 w-fit mb-2`}>{icon}</div>
    <p className="text-2xl font-bold text-white">{value}</p>
    <p className="text-xs text-slate-400">{label}</p>
  </div>
)

const HistoryItem: React.FC<{ snapshot: ConformanceSnapshot }> = ({ snapshot }) => {
  const [expanded, setExpanded] = useState(false)

  return (
    <div className="border border-slate-700/50 rounded-lg overflow-hidden">
      <div
        className="flex items-center gap-3 p-3 cursor-pointer hover:bg-slate-800/50 transition"
        onClick={() => setExpanded(!expanded)}
      >
        <StatusBadge status={snapshot.conformity_status} />
        <div className="flex-1">
          <p className="text-sm text-white">{snapshot.scan_id}</p>
          <p className="text-xs text-slate-500">
            {new Date(snapshot.created_at).toLocaleString('pt-PT')}
          </p>
        </div>
        <div className="text-right">
          <p className="text-sm font-medium text-white">{(100 - snapshot.scalar_error_score).toFixed(0)}%</p>
          <p className="text-xs text-slate-500">Qualidade</p>
        </div>
        <ChevronRight
          className={`w-4 h-4 text-slate-400 transition-transform ${expanded ? 'rotate-90' : ''}`}
        />
      </div>

      <AnimatePresence>
        {expanded && snapshot.explanation && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="border-t border-slate-700/50 bg-slate-900/50"
          >
            <div className="p-3 space-y-3">
              <div className="grid grid-cols-3 gap-2 text-xs">
                <div>
                  <p className="text-slate-500">Max Dev</p>
                  <p className="text-white">{snapshot.max_dev.toFixed(3)} mm</p>
                </div>
                <div>
                  <p className="text-slate-500">Mean Dev</p>
                  <p className="text-white">{snapshot.mean_dev.toFixed(3)} mm</p>
                </div>
                <div>
                  <p className="text-slate-500">RMS Dev</p>
                  <p className="text-white">{snapshot.rms_dev.toFixed(3)} mm</p>
                </div>
              </div>

              {snapshot.explanation.probable_causes.length > 0 && (
                <div>
                  <p className="text-xs text-slate-500 mb-2">Causas Prováveis</p>
                  {snapshot.explanation.probable_causes.map((cause, i) => (
                    <div key={i} className="flex items-center gap-2 text-xs mb-1">
                      <div
                        className="w-2 h-2 rounded-full"
                        style={{
                          backgroundColor: cause.impact > 0.6 ? '#ef4444' : cause.impact > 0.3 ? '#f59e0b' : '#10b981',
                        }}
                      />
                      <span className="text-white">{cause.parameter}</span>
                      <span className="text-slate-500">({(cause.impact * 100).toFixed(0)}%)</span>
                    </div>
                  ))}
                </div>
              )}

              {snapshot.explanation.recommendations.length > 0 && (
                <div>
                  <p className="text-xs text-slate-500 mb-1">Recomendações</p>
                  {snapshot.explanation.recommendations.map((rec, i) => (
                    <p key={i} className="text-xs text-cyan-400">• {rec}</p>
                  ))}
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

const AnalyzeScanModal: React.FC<{
  revisionId: number
  onClose: () => void
  onSuccess: () => void
}> = ({ revisionId, onClose, onSuccess }) => {
  const [scanId, setScanId] = useState(`SCAN-${Date.now()}`)
  const [params, setParams] = useState({
    feed_rate: 100,
    spindle_speed: 3000,
    depth_of_cut: 0.5,
    coolant_flow: 10,
  })
  const [result, setResult] = useState<AnalyzeScanResponse | null>(null)

  const mutation = useMutation({
    mutationFn: () => analyzeScan(revisionId, scanId, params),
    onSuccess: (data) => {
      setResult(data)
      onSuccess()
    },
  })

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4"
      onClick={onClose}
    >
      <motion.div
        initial={{ scale: 0.9, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        exit={{ scale: 0.9, opacity: 0 }}
        className="bg-slate-800 rounded-2xl p-6 max-w-lg w-full max-h-[90vh] overflow-y-auto"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <Scan className="w-6 h-6 text-cyan-400" />
            <h2 className="text-xl font-bold text-white">Analisar Scan</h2>
          </div>
          <button onClick={onClose} className="p-2 hover:bg-slate-700 rounded-lg transition">
            <X className="w-5 h-5 text-slate-400" />
          </button>
        </div>

        {!result ? (
          <div className="space-y-4">
            <div>
              <label className="text-sm text-slate-400 block mb-1">ID do Scan</label>
              <input
                type="text"
                value={scanId}
                onChange={(e) => setScanId(e.target.value)}
                className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-white focus:border-cyan-500 focus:outline-none"
              />
            </div>

            <div>
              <label className="text-sm text-slate-400 block mb-2">Parâmetros de Processo</label>
              <div className="grid grid-cols-2 gap-3">
                {Object.entries(params).map(([key, value]) => (
                  <div key={key}>
                    <label className="text-xs text-slate-500 block mb-1">
                      {key.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())}
                    </label>
                    <input
                      type="number"
                      value={value}
                      onChange={(e) => setParams((p) => ({ ...p, [key]: parseFloat(e.target.value) || 0 }))}
                      className="w-full px-2 py-1.5 bg-slate-900 border border-slate-700 rounded text-white text-sm focus:border-cyan-500 focus:outline-none"
                    />
                  </div>
                ))}
              </div>
            </div>

            <button
              onClick={() => mutation.mutate()}
              disabled={mutation.isPending}
              className="w-full py-3 bg-cyan-600 hover:bg-cyan-500 text-white font-semibold rounded-lg transition disabled:opacity-50"
            >
              {mutation.isPending ? (
                <span className="flex items-center justify-center gap-2">
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  A analisar...
                </span>
              ) : (
                <span className="flex items-center justify-center gap-2">
                  <Scan className="w-4 h-4" />
                  Executar Análise
                </span>
              )}
            </button>
          </div>
        ) : (
          <div className="space-y-4">
            <div className="flex items-center justify-center">
              <ScoreGauge score={result.scalar_error_score} label="Qualidade" />
            </div>

            <div className="flex justify-center">
              <StatusBadge status={result.conformity_status} />
            </div>

            <div className="grid grid-cols-3 gap-3 text-center">
              <div className="bg-slate-900/50 rounded-lg p-3">
                <p className="text-lg font-bold text-white">{result.max_dev.toFixed(3)}</p>
                <p className="text-xs text-slate-500">Max (mm)</p>
              </div>
              <div className="bg-slate-900/50 rounded-lg p-3">
                <p className="text-lg font-bold text-white">{result.mean_dev.toFixed(3)}</p>
                <p className="text-xs text-slate-500">Média (mm)</p>
              </div>
              <div className="bg-slate-900/50 rounded-lg p-3">
                <p className="text-lg font-bold text-white">{result.rms_dev.toFixed(3)}</p>
                <p className="text-xs text-slate-500">RMS (mm)</p>
              </div>
            </div>

            {result.probable_causes.length > 0 && (
              <div className="bg-slate-900/50 rounded-lg p-4">
                <p className="text-sm font-medium text-white mb-2">🔍 Causas Prováveis (XAI)</p>
                {result.probable_causes.map((cause, i) => (
                  <div key={i} className="flex items-start gap-2 mb-2">
                    <TrendingDown className="w-4 h-4 text-amber-400 mt-0.5" />
                    <div>
                      <p className="text-sm text-white">{cause.parameter}</p>
                      <p className="text-xs text-slate-400">{cause.action}</p>
                    </div>
                    <span className="ml-auto text-xs text-amber-400">
                      {(cause.impact * 100).toFixed(0)}%
                    </span>
                  </div>
                ))}
              </div>
            )}

            {result.recommendations.length > 0 && (
              <div className="bg-cyan-500/10 border border-cyan-500/30 rounded-lg p-4">
                <p className="text-sm font-medium text-cyan-400 mb-2">💡 Recomendações</p>
                {result.recommendations.map((rec, i) => (
                  <p key={i} className="text-sm text-slate-300">• {rec}</p>
                ))}
              </div>
            )}

            <button
              onClick={onClose}
              className="w-full py-3 bg-slate-700 hover:bg-slate-600 text-white rounded-lg transition"
            >
              Fechar
            </button>
          </div>
        )}
      </motion.div>
    </motion.div>
  )
}

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN COMPONENT
// ═══════════════════════════════════════════════════════════════════════════════

const DigitalTwinProduct: React.FC = () => {
  const queryClient = useQueryClient()
  const [selectedProduct, setSelectedProduct] = useState<ProductItem | null>(null)
  const [showAnalyzeModal, setShowAnalyzeModal] = useState(false)

  const { data: products, isLoading: loadingProducts } = useQuery({
    queryKey: ['dt-products'],
    queryFn: fetchProducts,
  })

  const revisionId = selectedProduct?.active_revision_id

  const { data: summary, isLoading: loadingSummary } = useQuery({
    queryKey: ['conformance-summary', revisionId],
    queryFn: () => fetchConformanceSummary(revisionId!),
    enabled: !!revisionId,
  })

  const { data: history } = useQuery({
    queryKey: ['conformance-history', revisionId],
    queryFn: () => fetchConformanceHistory(revisionId!),
    enabled: !!revisionId,
  })

  const { data: goldenRuns } = useQuery({
    queryKey: ['golden-runs', revisionId],
    queryFn: () => fetchGoldenRuns(revisionId!),
    enabled: !!revisionId,
  })

  const { data: suggestion } = useQuery({
    queryKey: ['param-suggestion', revisionId],
    queryFn: () => fetchParamSuggestion(revisionId!),
    enabled: !!revisionId,
  })

  const computeMutation = useMutation({
    mutationFn: () => computeGoldenRuns(revisionId!),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['golden-runs', revisionId] })
    },
  })

  const refreshData = () => {
    queryClient.invalidateQueries({ queryKey: ['conformance-summary', revisionId] })
    queryClient.invalidateQueries({ queryKey: ['conformance-history', revisionId] })
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <header className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-gradient-to-br from-purple-500/20 to-pink-500/20 rounded-xl border border-purple-500/30">
            <Target className="w-6 h-6 text-purple-400" />
          </div>
          <div>
            <p className="text-xs uppercase tracking-[0.4em] text-slate-500">Digital Twin</p>
            <h2 className="text-2xl font-semibold text-white">Conformidade de Produto</h2>
          </div>
        </div>
        {selectedProduct && revisionId && (
          <button
            onClick={() => setShowAnalyzeModal(true)}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-purple-600 hover:bg-purple-500 text-white font-medium transition"
          >
            <Scan className="w-4 h-4" />
            Analisar Scan
          </button>
        )}
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Product List - Left Panel */}
        <div className="lg:col-span-3 space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-medium text-white flex items-center gap-2">
              <Box className="w-4 h-4 text-purple-400" />
              Produtos
            </h3>
            <span className="text-xs text-slate-500">{products?.length || 0} items</span>
          </div>

          <div className="space-y-2 max-h-[calc(100vh-300px)] overflow-y-auto pr-2">
            {loadingProducts ? (
              <div className="flex items-center justify-center py-8">
                <RefreshCw className="w-6 h-6 text-purple-400 animate-spin" />
              </div>
            ) : products?.length === 0 ? (
              <p className="text-sm text-slate-400 text-center py-8">Nenhum produto encontrado</p>
            ) : (
              products?.map((product) => (
                <ProductCard
                  key={product.id}
                  product={product}
                  isSelected={selectedProduct?.id === product.id}
                  onClick={() => setSelectedProduct(product)}
                />
              ))
            )}
          </div>
        </div>

        {/* Main Content */}
        <div className="lg:col-span-9">
          {!selectedProduct ? (
            <div className="flex flex-col items-center justify-center h-64 text-center">
              <FileSearch className="w-12 h-12 text-slate-600 mb-4" />
              <p className="text-slate-400">Selecione um produto para ver a conformidade</p>
            </div>
          ) : !revisionId ? (
            <div className="flex flex-col items-center justify-center h-64 text-center">
              <AlertTriangle className="w-12 h-12 text-amber-400 mb-4" />
              <p className="text-slate-400">Produto sem revisão ativa</p>
            </div>
          ) : (
            <div className="space-y-6">
              {/* Summary Metrics */}
              {loadingSummary ? (
                <div className="flex items-center justify-center py-12">
                  <RefreshCw className="w-6 h-6 text-purple-400 animate-spin" />
                </div>
              ) : summary && (
                <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                  <div className="md:col-span-1">
                    <ScoreGauge score={summary.avg_error_score} label="Qualidade Média" />
                  </div>
                  <MetricCard
                    icon={<Scan className="w-5 h-5 text-cyan-400" />}
                    label="Total Scans"
                    value={summary.total_scans}
                  />
                  <MetricCard
                    icon={<CheckCircle className="w-5 h-5 text-emerald-400" />}
                    label="Conformes"
                    value={summary.in_tolerance}
                    color="emerald"
                  />
                  <MetricCard
                    icon={<AlertTriangle className="w-5 h-5 text-amber-400" />}
                    label="Fora Tolerância"
                    value={summary.out_of_tolerance}
                    color="amber"
                  />
                  <MetricCard
                    icon={<AlertTriangle className="w-5 h-5 text-red-400" />}
                    label="Críticos"
                    value={summary.critical}
                    color="red"
                  />
                </div>
              )}

              {/* Two Column Layout */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Conformance History */}
                <div className="space-y-3">
                  <h3 className="text-sm font-medium text-white flex items-center gap-2">
                    <BarChart3 className="w-4 h-4 text-purple-400" />
                    Histórico de Conformidade
                  </h3>
                  
                  <div className="space-y-2 max-h-96 overflow-y-auto pr-2">
                    {history?.length === 0 ? (
                      <p className="text-sm text-slate-400 text-center py-8">
                        Nenhum scan registado
                      </p>
                    ) : (
                      history?.map((snapshot) => (
                        <HistoryItem key={snapshot.id} snapshot={snapshot} />
                      ))
                    )}
                  </div>
                </div>

                {/* Golden Runs & Suggestions */}
                <div className="space-y-6">
                  {/* Golden Runs */}
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <h3 className="text-sm font-medium text-white flex items-center gap-2">
                        <Award className="w-4 h-4 text-amber-400" />
                        Golden Runs
                      </h3>
                      <button
                        onClick={() => computeMutation.mutate()}
                        disabled={computeMutation.isPending}
                        className="text-xs text-cyan-400 hover:text-cyan-300 transition"
                      >
                        {computeMutation.isPending ? 'A calcular...' : 'Recalcular'}
                      </button>
                    </div>

                    <div className="space-y-2">
                      {goldenRuns?.length === 0 ? (
                        <p className="text-sm text-slate-400 text-center py-4">
                          Sem golden runs registados
                        </p>
                      ) : (
                        goldenRuns?.slice(0, 3).map((run) => (
                          <div
                            key={run.id}
                            className="p-3 rounded-lg bg-amber-500/10 border border-amber-500/30"
                          >
                            <div className="flex items-center justify-between mb-2">
                              <span className="text-sm font-medium text-white">
                                {run.machine_id || 'Máquina N/A'}
                              </span>
                              <span className="text-sm font-bold text-amber-400">
                                {run.score.toFixed(0)}%
                              </span>
                            </div>
                            <div className="flex flex-wrap gap-2">
                              {Object.entries(run.process_params).slice(0, 4).map(([key, value]) => (
                                <span
                                  key={key}
                                  className="text-xs bg-slate-800 px-2 py-1 rounded text-slate-300"
                                >
                                  {key}: {typeof value === 'number' ? value.toFixed(1) : value}
                                </span>
                              ))}
                            </div>
                          </div>
                        ))
                      )}
                    </div>
                  </div>

                  {/* Parameter Suggestion */}
                  {suggestion && suggestion.based_on_golden_runs > 0 && (
                    <div className="space-y-3">
                      <h3 className="text-sm font-medium text-white flex items-center gap-2">
                        <Sparkles className="w-4 h-4 text-cyan-400" />
                        Parâmetros Sugeridos
                      </h3>

                      <div className="p-4 rounded-lg bg-cyan-500/10 border border-cyan-500/30">
                        <div className="flex items-center justify-between mb-3">
                          <span className="text-sm text-slate-300">
                            Baseado em {suggestion.based_on_golden_runs} golden runs
                          </span>
                          <span className="text-sm font-bold text-cyan-400">
                            {suggestion.expected_quality_score.toFixed(0)}% esperado
                          </span>
                        </div>

                        <div className="grid grid-cols-2 gap-2">
                          {Object.entries(suggestion.suggested_params).map(([key, value]) => (
                            <div key={key} className="flex justify-between text-sm">
                              <span className="text-slate-400">
                                {key.replace(/_/g, ' ')}
                              </span>
                              <span className="text-white font-medium">
                                {typeof value === 'number' ? value.toFixed(1) : value}
                              </span>
                            </div>
                          ))}
                        </div>

                        <button className="w-full mt-3 py-2 bg-cyan-600 hover:bg-cyan-500 text-white text-sm font-medium rounded-lg transition flex items-center justify-center gap-2">
                          <Settings2 className="w-4 h-4" />
                          Aplicar ao Processo
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Analyze Modal */}
      <AnimatePresence>
        {showAnalyzeModal && revisionId && (
          <AnalyzeScanModal
            revisionId={revisionId}
            onClose={() => setShowAnalyzeModal(false)}
            onSuccess={refreshData}
          />
        )}
      </AnimatePresence>
    </div>
  )
}

export default DigitalTwinProduct



