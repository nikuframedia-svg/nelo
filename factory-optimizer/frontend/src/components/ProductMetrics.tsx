/**
 * ════════════════════════════════════════════════════════════════════════════════════════════
 * PRODPLAN 4.0 — PRODUCT METRICS COMPONENT
 * ════════════════════════════════════════════════════════════════════════════════════════════
 *
 * Métricas industriais por tipo de produto.
 * - Classificação automática (vidro_duplo, vidro_triplo, etc.)
 * - KPIs por tipo: tempo, setup, lead time, SNR
 * - Estimativa automática de datas de entrega
 */

import React, { useState, useMemo } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Package,
  Clock,
  TrendingUp,
  Activity,
  Calendar,
  ChevronDown,
  ChevronUp,
  AlertTriangle,
  CheckCircle,
  Layers,
  Gauge,
  RefreshCw,
} from 'lucide-react'
import { toast } from 'sonner'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000'

// ═══════════════════════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════════════════════

interface TypeKPIs {
  product_type: string
  num_products: number
  num_orders: number
  total_units: number
  avg_processing_time_min: number
  std_processing_time_min: number
  avg_lead_time_hours: number
  percentile_90_lead_time_hours: number
  snr_between_products: number
  snr_level: string
  sigma_levels: Record<string, number>
}

interface GlobalKPIs {
  total_products: number
  total_orders: number
  fastest_type: string
  slowest_type: string
  most_stable_type: string
  global_avg_lead_time_hours: number
  global_avg_processing_time_min: number
  type_distribution: Record<string, number>
}

interface TypeKPIsResponse {
  type_kpis: Record<string, TypeKPIs>
  global_kpis: GlobalKPIs
  available_types: string[]
}

interface DeliveryEstimate {
  order_id: string
  article_id: string
  estimated_duration_hours: number
  estimated_delivery_date: string | null
  breakdown: {
    processing_hours: number
    setup_hours: number
    queue_hours: number
    buffer_hours: number
  }
  confidence_score: number
  confidence_level: string
  snr_estimate: number
  snr_level: string
}

// ═══════════════════════════════════════════════════════════════════════════════════════════
// API FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════════════════════

async function fetchTypeKPIs(): Promise<TypeKPIsResponse> {
  const res = await fetch(`${API_BASE_URL}/product/type-kpis`)
  if (!res.ok) throw new Error('Erro ao carregar KPIs de produto')
  return res.json()
}

async function fetchDeliveryEstimate(
  orderId: string,
  articleId: string,
  qty: number
): Promise<{ estimate: DeliveryEstimate }> {
  const res = await fetch(`${API_BASE_URL}/product/delivery-estimate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      order_id: orderId,
      article_id: articleId,
      qty,
      method: 'deterministic',
      buffer_strategy: 'moderate',
    }),
  })
  if (!res.ok) throw new Error('Erro ao estimar entrega')
  return res.json()
}

// ═══════════════════════════════════════════════════════════════════════════════════════════
// HELPER COMPONENTS
// ═══════════════════════════════════════════════════════════════════════════════════════════

const ProductTypeBadge: React.FC<{ type: string }> = ({ type }) => {
  const colors: Record<string, string> = {
    vidro_duplo: 'bg-blue-500/20 text-blue-300 border-blue-500/30',
    vidro_triplo: 'bg-purple-500/20 text-purple-300 border-purple-500/30',
    vidro_laminado: 'bg-amber-500/20 text-amber-300 border-amber-500/30',
    vidro_temperado: 'bg-red-500/20 text-red-300 border-red-500/30',
    vidro_simples: 'bg-slate-500/20 text-slate-300 border-slate-500/30',
    outro: 'bg-slate-600/20 text-slate-400 border-slate-600/30',
  }

  const labels: Record<string, string> = {
    vidro_duplo: 'Vidro Duplo',
    vidro_triplo: 'Vidro Triplo',
    vidro_laminado: 'Vidro Laminado',
    vidro_temperado: 'Vidro Temperado',
    vidro_simples: 'Vidro Simples',
    outro: 'Outro',
  }

  return (
    <span className={`px-2 py-1 text-xs font-medium rounded border ${colors[type] || colors.outro}`}>
      {labels[type] || type}
    </span>
  )
}

const SNRIndicator: React.FC<{ snr: number; level: string }> = ({ snr, level }) => {
  const colors: Record<string, string> = {
    EXCELLENT: 'text-emerald-400',
    GOOD: 'text-green-400',
    FAIR: 'text-amber-400',
    POOR: 'text-red-400',
  }

  return (
    <div className="flex items-center gap-2">
      <Activity className={`w-4 h-4 ${colors[level] || 'text-slate-400'}`} />
      <span className={`text-sm font-mono ${colors[level] || 'text-slate-400'}`}>
        {snr.toFixed(1)}
      </span>
      <span className="text-xs text-slate-500">({level})</span>
    </div>
  )
}

const ConfidenceBadge: React.FC<{ score: number; level: string }> = ({ score, level }) => {
  const colors: Record<string, string> = {
    HIGH: 'bg-emerald-500/20 text-emerald-300',
    MEDIUM: 'bg-amber-500/20 text-amber-300',
    LOW: 'bg-red-500/20 text-red-300',
    VERY_LOW: 'bg-red-700/20 text-red-400',
  }

  return (
    <span className={`px-2 py-0.5 text-xs font-medium rounded ${colors[level] || 'bg-slate-500/20 text-slate-300'}`}>
      {(score * 100).toFixed(0)}% ({level})
    </span>
  )
}

// ═══════════════════════════════════════════════════════════════════════════════════════════
// TYPE KPI CARD
// ═══════════════════════════════════════════════════════════════════════════════════════════

const TypeKPICard: React.FC<{
  type: string
  kpis: TypeKPIs
  isExpanded: boolean
  onToggle: () => void
  globalAvgTime: number
  globalAvgLead: number
}> = ({ type, kpis, isExpanded, onToggle, globalAvgTime, globalAvgLead }) => {
  const timeRatio = kpis.avg_processing_time_min / globalAvgTime
  const leadRatio = kpis.avg_lead_time_hours / globalAvgLead

  return (
    <motion.div
      layout
      className="border border-slate-700/50 rounded-lg overflow-hidden"
    >
      <div
        onClick={onToggle}
        className={`p-4 cursor-pointer transition-colors ${
          isExpanded ? 'bg-slate-700/30' : 'bg-slate-800/30 hover:bg-slate-700/20'
        }`}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <ProductTypeBadge type={type} />
            <span className="text-sm text-slate-400">
              {kpis.num_products} artigos • {kpis.num_orders} encomendas
            </span>
          </div>

          <div className="flex items-center gap-6">
            <div className="text-right hidden md:block">
              <p className="text-sm font-medium text-white">
                {kpis.avg_processing_time_min.toFixed(0)} min
              </p>
              <p className={`text-xs ${timeRatio <= 1 ? 'text-emerald-400' : 'text-amber-400'}`}>
                {timeRatio <= 1 ? '✓ abaixo média' : `${((timeRatio - 1) * 100).toFixed(0)}% acima`}
              </p>
            </div>

            <div className="text-right hidden md:block">
              <p className="text-sm font-medium text-white">
                {kpis.avg_lead_time_hours.toFixed(1)}h
              </p>
              <p className="text-xs text-slate-400">lead time</p>
            </div>

            <SNRIndicator snr={kpis.snr_between_products} level={kpis.snr_level} />

            {isExpanded ? (
              <ChevronUp className="w-5 h-5 text-slate-400" />
            ) : (
              <ChevronDown className="w-5 h-5 text-slate-400" />
            )}
          </div>
        </div>
      </div>

      <AnimatePresence>
        {isExpanded && (
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
                  <p className="text-xs text-slate-500">Tempo Médio</p>
                  <p className="text-lg font-semibold text-white">
                    {kpis.avg_processing_time_min.toFixed(1)} min
                  </p>
                  <p className="text-xs text-slate-400">σ = {kpis.std_processing_time_min.toFixed(1)}</p>
                </div>
                <div>
                  <p className="text-xs text-slate-500">Lead Time Médio</p>
                  <p className="text-lg font-semibold text-white">
                    {kpis.avg_lead_time_hours.toFixed(2)}h
                  </p>
                  <p className="text-xs text-slate-400">
                    P90: {kpis.percentile_90_lead_time_hours.toFixed(2)}h
                  </p>
                </div>
                <div>
                  <p className="text-xs text-slate-500">Total Unidades</p>
                  <p className="text-lg font-semibold text-cyan-400">
                    {kpis.total_units.toLocaleString()}
                  </p>
                </div>
                <div>
                  <p className="text-xs text-slate-500">Estabilidade</p>
                  <SNRIndicator snr={kpis.snr_between_products} level={kpis.snr_level} />
                </div>
              </div>

              {/* Sigma levels */}
              {Object.keys(kpis.sigma_levels).length > 0 && (
                <div className="bg-slate-800/50 rounded p-3">
                  <h4 className="text-sm font-medium text-slate-300 mb-2">
                    Níveis Sigma (Tempo de Processamento)
                  </h4>
                  <div className="flex flex-wrap gap-2">
                    {Object.entries(kpis.sigma_levels)
                      .filter(([k]) => !k.startsWith('-'))
                      .sort()
                      .map(([level, value]) => (
                        <span
                          key={level}
                          className="px-2 py-1 bg-slate-700/50 rounded text-xs font-mono"
                        >
                          {level}: {value.toFixed(1)} min
                        </span>
                      ))}
                  </div>
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
// DELIVERY ESTIMATOR
// ═══════════════════════════════════════════════════════════════════════════════════════════

const DeliveryEstimator: React.FC = () => {
  const [orderId, setOrderId] = useState('')
  const [articleId, setArticleId] = useState('')
  const [qty, setQty] = useState(1)
  const [estimate, setEstimate] = useState<DeliveryEstimate | null>(null)

  const estimateMutation = useMutation({
    mutationFn: () => fetchDeliveryEstimate(orderId || 'NEW', articleId, qty),
    onSuccess: (data) => {
      setEstimate(data.estimate)
      toast.success('Estimativa calculada')
    },
    onError: () => {
      toast.error('Erro ao calcular estimativa')
    },
  })

  return (
    <div className="bg-slate-800/50 border border-slate-700/50 rounded-lg p-4">
      <h3 className="text-sm font-medium text-slate-300 mb-3 flex items-center gap-2">
        <Calendar className="w-4 h-4" />
        Estimar Data de Entrega
      </h3>

      <div className="flex flex-wrap gap-3 items-end">
        <div>
          <label className="text-xs text-slate-500">Artigo</label>
          <input
            type="text"
            value={articleId}
            onChange={(e) => setArticleId(e.target.value)}
            placeholder="ART-XXX"
            className="block w-32 px-2 py-1.5 bg-slate-900 border border-slate-700 rounded text-sm text-white"
          />
        </div>
        <div>
          <label className="text-xs text-slate-500">Quantidade</label>
          <input
            type="number"
            value={qty}
            onChange={(e) => setQty(Number(e.target.value))}
            min={1}
            className="block w-20 px-2 py-1.5 bg-slate-900 border border-slate-700 rounded text-sm text-white"
          />
        </div>
        <button
          onClick={() => estimateMutation.mutate()}
          disabled={!articleId || estimateMutation.isPending}
          className="px-4 py-1.5 bg-cyan-600 hover:bg-cyan-500 text-white text-sm rounded transition-colors disabled:opacity-50"
        >
          {estimateMutation.isPending ? 'A calcular...' : 'Estimar'}
        </button>

        {estimate && (
          <div className="flex-1 flex items-center gap-4 px-4">
            <div>
              <p className="text-xs text-slate-500">Duração Prevista</p>
              <p className="text-lg font-semibold text-white">
                {estimate.estimated_duration_hours.toFixed(1)}h
              </p>
            </div>
            <div>
              <p className="text-xs text-slate-500">Data Entrega</p>
              <p className="text-sm font-medium text-cyan-400">
                {estimate.estimated_delivery_date?.split(' ')[0] || 'N/A'}
              </p>
            </div>
            <ConfidenceBadge score={estimate.confidence_score} level={estimate.confidence_level} />
          </div>
        )}
      </div>
    </div>
  )
}

// ═══════════════════════════════════════════════════════════════════════════════════════════
// MAIN COMPONENT
// ═══════════════════════════════════════════════════════════════════════════════════════════

const ProductMetrics: React.FC = () => {
  const [expandedType, setExpandedType] = useState<string | null>(null)

  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ['product-type-kpis'],
    queryFn: fetchTypeKPIs,
    refetchInterval: 60000,
  })

  const sortedTypes = useMemo(() => {
    if (!data?.type_kpis) return []
    return Object.entries(data.type_kpis)
      .filter(([, kpis]) => kpis.num_products > 0)
      .sort((a, b) => b[1].num_products - a[1].num_products)
  }, [data])

  const globalAvgTime = data?.global_kpis?.global_avg_processing_time_min || 1
  const globalAvgLead = data?.global_kpis?.global_avg_lead_time_hours || 1

  if (error) {
    return (
      <div className="p-4 bg-red-900/20 border border-red-700/30 rounded-lg">
        <p className="text-red-300 flex items-center gap-2">
          <AlertTriangle className="w-4 h-4" />
          Erro ao carregar métricas de produto
        </p>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-white flex items-center gap-2">
          <Package className="w-5 h-5 text-cyan-400" />
          Métricas por Tipo de Produto
        </h2>
        <button
          onClick={() => refetch()}
          className="p-2 hover:bg-slate-700/50 rounded transition-colors"
        >
          <RefreshCw className={`w-4 h-4 text-slate-400 ${isLoading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {/* Global KPIs */}
      {data?.global_kpis && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <div className="p-3 bg-slate-800/50 border border-slate-700/50 rounded-lg">
            <p className="text-xs text-slate-400">Total Produtos</p>
            <p className="text-xl font-bold text-white">{data.global_kpis.total_products}</p>
          </div>
          <div className="p-3 bg-slate-800/50 border border-slate-700/50 rounded-lg">
            <p className="text-xs text-slate-400">Tempo Médio</p>
            <p className="text-xl font-bold text-white">
              {(data.global_kpis.global_avg_processing_time_min ?? 0).toFixed(0)} min
            </p>
          </div>
          <div className="p-3 bg-slate-800/50 border border-slate-700/50 rounded-lg">
            <p className="text-xs text-slate-400">Mais Rápido</p>
            <ProductTypeBadge type={data.global_kpis.fastest_type} />
          </div>
          <div className="p-3 bg-slate-800/50 border border-slate-700/50 rounded-lg">
            <p className="text-xs text-slate-400">Mais Estável</p>
            <ProductTypeBadge type={data.global_kpis.most_stable_type} />
          </div>
        </div>
      )}

      {/* Type KPIs */}
      <div className="space-y-2">
        {isLoading ? (
          <div className="text-center py-8">
            <RefreshCw className="w-6 h-6 text-cyan-400 animate-spin mx-auto mb-2" />
            <p className="text-slate-400 text-sm">A carregar métricas...</p>
          </div>
        ) : sortedTypes.length === 0 ? (
          <div className="text-center py-8 bg-slate-800/30 rounded-lg">
            <Layers className="w-10 h-10 text-slate-500 mx-auto mb-2" />
            <p className="text-slate-400">Nenhum tipo de produto classificado</p>
          </div>
        ) : (
          sortedTypes.map(([type, kpis]) => (
            <TypeKPICard
              key={type}
              type={type}
              kpis={kpis}
              isExpanded={expandedType === type}
              onToggle={() => setExpandedType(expandedType === type ? null : type)}
              globalAvgTime={globalAvgTime}
              globalAvgLead={globalAvgLead}
            />
          ))
        )}
      </div>

      {/* Delivery Estimator */}
      <DeliveryEstimator />
    </div>
  )
}

export default ProductMetrics

