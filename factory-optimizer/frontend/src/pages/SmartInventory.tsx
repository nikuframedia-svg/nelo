/**
 * SmartInventory Page - Digital Twin de Inventário ULTRA-AVANÇADO
 * 
 * Funcionalidades:
 * - Stock em tempo real (multi-armazém) via IoT/RFID/Vision
 * - Digital Twin de inventário completo
 * - Forecast de demanda avançado (ARIMA/Prophet + SNR)
 * - ROP dinâmico e risco 30 dias por SKU
 * - Matriz ABC/XYZ para classificação
 * - Sugestões inteligentes (comprar, transferir, reduzir)
 * - Integração com sinais externos (preços, notícias, macro)
 */

import { useState, useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import { motion, AnimatePresence } from 'framer-motion'
import { API_BASE_URL } from '../config/api'
import { 
  Package, 
  TrendingUp, 
  AlertTriangle, 
  ShoppingCart, 
  ArrowRightLeft,
  BarChart3,
  Activity,
  RefreshCw,
  Download,
  Filter,
  Warehouse,
  TrendingDown,
  Clock,
  Target,
  Zap,
  Brain,
  LineChart,
  Settings,
  Layers,
  Search,
  Check,
  X,
  Edit3,
  ChevronDown,
  ChevronRight,
  Clipboard,
} from 'lucide-react'

// Lazy load new panels
import { lazy, Suspense } from 'react'
const MRPForecastPanel = lazy(() => import('../components/MRPForecastPanel'))
const WorkInstructionsAdmin = lazy(() => import('../components/WorkInstructionsAdmin'))

// ═══════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════

type StockItem = {
  sku: string
  warehouse_id: string
  quantity_on_hand: number
  quantity_available: number
  quantity_committed: number
  quantity_in_transit: number
  quantity_effective: number
  last_updated?: string
}

/* Types for reference (inferred by useQuery)
type ForecastData = {
  sku: string
  forecast: Record<string, number>
  lower_ci: Record<string, number>
  upper_ci: Record<string, number>
  model_used: string
  snr: number
  snr_class: string
  confidence_score: number
  metrics?: { MAPE?: number; RMSE?: number; MAE?: number }
}

type ROPData = {
  sku: string
  rop: number
  safety_stock: number
  reorder_quantity: number
  risk_30d: number
  coverage_days: number
  confidence: number
  current_stock?: number
  days_until_rop?: number
  explanation: string
}
*/

type InventorySuggestion = {
  suggestion_type: string
  priority: string
  sku: string
  warehouse_id?: string
  target_warehouse_id?: string
  title: string
  description: string
  quantity: number
  risk_level: number
  explanation?: string
}

type ABCXYZMatrix = {
  A: { X: number; Y: number; Z: number }
  B: { X: number; Y: number; Z: number }
  C: { X: number; Y: number; Z: number }
}

type InventorySKU = {
  sku: string
  classe: string
  xyz: string
  stock_atual: number
  ads_180: number
  cobertura_dias: number
  risco_30d: number
  rop: number
  acao: string
}

// ═══════════════════════════════════════════════════════════════════
// API FUNCTIONS
// ═══════════════════════════════════════════════════════════════════

async function fetchStock(warehouseId?: string) {
  const url = warehouseId 
    ? `${API_BASE_URL}/inventory/stock?warehouse_id=${warehouseId}`
    : `${API_BASE_URL}/inventory/stock`
  const res = await fetch(url)
  if (!res.ok) throw new Error('Erro ao carregar stock')
  return res.json()
}

async function fetchForecast(sku: string) {
  const res = await fetch(`${API_BASE_URL}/inventory/forecast/${sku}?horizon_days=90`)
  if (!res.ok) throw new Error('Erro ao carregar forecast')
  return res.json()
}

async function fetchROP(sku: string) {
  const res = await fetch(`${API_BASE_URL}/inventory/rop/${sku}`)
  if (!res.ok) throw new Error('Erro ao carregar ROP')
  return res.json()
}

async function fetchSuggestions() {
  const res = await fetch(`${API_BASE_URL}/inventory/suggestions`)
  if (!res.ok) throw new Error('Erro ao carregar sugestões')
  return res.json()
}

async function fetchInventoryOld(classe?: string, search?: string) {
  const params = new URLSearchParams()
  if (classe) params.append('classe', classe)
  if (search) params.append('search', search)
  const res = await fetch(`${API_BASE_URL}/inventory/?${params}`)
  if (!res.ok) throw new Error('Erro ao carregar inventário')
  return res.json()
}

// ═══════════════════════════════════════════════════════════════════
// COMPONENTS
// ═══════════════════════════════════════════════════════════════════

const MetricCard: React.FC<{
  title: string
  value: string | number
  subtitle?: string
  icon: React.ReactNode
  variant?: 'default' | 'success' | 'warning' | 'danger' | 'info'
  trend?: number
}> = ({ title, value, subtitle, icon, variant = 'default', trend }) => {
  const colors = {
    default: 'border-border bg-surface',
    success: 'border-green-500/40 bg-green-500/10',
    warning: 'border-amber-500/40 bg-amber-500/10',
    danger: 'border-red-500/40 bg-red-500/10',
    info: 'border-cyan-500/40 bg-cyan-500/10',
  }
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className={`rounded-xl border p-4 ${colors[variant]}`}
    >
      <div className="flex items-center justify-between mb-2">
        <p className="text-xs text-text-muted uppercase tracking-wider">{title}</p>
        {icon}
      </div>
      <p className="text-2xl font-bold text-text-primary">{value}</p>
      <div className="flex items-center justify-between mt-1">
        {subtitle && <p className="text-xs text-text-muted">{subtitle}</p>}
        {trend !== undefined && (
          <div className={`flex items-center text-xs ${trend >= 0 ? 'text-green-400' : 'text-red-400'}`}>
            {trend >= 0 ? <TrendingUp className="w-3 h-3 mr-1" /> : <TrendingDown className="w-3 h-3 mr-1" />}
            {Math.abs(trend).toFixed(1)}%
          </div>
        )}
      </div>
    </motion.div>
  )
}

const SuggestionCard: React.FC<{ 
  suggestion: InventorySuggestion
  onSelectSKU?: () => void
}> = ({ suggestion, onSelectSKU }) => {
  const priorityColors = {
    HIGH: 'border-red-500/40 bg-gradient-to-br from-red-500/10 to-red-900/10',
    MEDIUM: 'border-amber-500/40 bg-gradient-to-br from-amber-500/10 to-amber-900/10',
    LOW: 'border-border bg-surface',
  }
  
  const icons = {
    BUY: <ShoppingCart className="w-5 h-5 text-cyan-400" />,
    TRANSFER: <ArrowRightLeft className="w-5 h-5 text-purple-400" />,
    REDUCE: <TrendingDown className="w-5 h-5 text-amber-400" />,
    ALERT: <AlertTriangle className="w-5 h-5 text-red-400" />,
    RUPTURE_RISK: <AlertTriangle className="w-5 h-5 text-red-400" />,
    EXCESS_STOCK: <Package className="w-5 h-5 text-amber-400" />,
    PRICE_ALERT: <Zap className="w-5 h-5 text-yellow-400" />,
  }
  
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      whileHover={{ scale: 1.01 }}
      className={`rounded-xl border p-4 transition-shadow hover:shadow-lg ${
        priorityColors[suggestion.priority as keyof typeof priorityColors] || priorityColors.LOW
      }`}
    >
      <div className="flex items-start justify-between mb-2">
        <div className="flex items-center gap-2">
          {icons[suggestion.suggestion_type as keyof typeof icons] || <Package className="w-5 h-5" />}
          <span className="font-semibold text-text-primary">{suggestion.title}</span>
        </div>
        <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${
          suggestion.priority === 'HIGH' ? 'bg-red-500/20 text-red-400' :
          suggestion.priority === 'MEDIUM' ? 'bg-amber-500/20 text-amber-400' :
          'bg-border text-text-muted'
        }`}>
          {suggestion.priority}
        </span>
      </div>
      
      <p className="text-sm text-text-muted mb-3">{suggestion.description}</p>
      
      <div className="flex items-center gap-4 text-xs">
        {suggestion.quantity > 0 && (
          <div className="flex items-center gap-1 text-text-primary">
            <Package className="w-3 h-3" />
            <span className="font-semibold">{suggestion.quantity.toFixed(0)} un.</span>
          </div>
        )}
        {suggestion.risk_level > 0 && (
          <div className="flex-1">
            <div className="flex items-center justify-between mb-1">
              <span className="text-text-muted">Risco</span>
              <span className={`font-semibold ${
                suggestion.risk_level > 50 ? 'text-red-400' :
                suggestion.risk_level > 20 ? 'text-amber-400' :
                'text-green-400'
              }`}>
                {suggestion.risk_level.toFixed(1)}%
              </span>
            </div>
            <div className="h-1.5 bg-background rounded-full overflow-hidden">
              <div 
                className={`h-full transition-all ${
                  suggestion.risk_level > 50 ? 'bg-red-500' :
                  suggestion.risk_level > 20 ? 'bg-amber-500' :
                  'bg-green-500'
                }`}
                style={{ width: `${Math.min(100, suggestion.risk_level)}%` }}
              />
            </div>
          </div>
        )}
      </div>
      
      {onSelectSKU && (
        <button
          onClick={(e) => {
            e.stopPropagation()
            onSelectSKU()
          }}
          className="mt-3 w-full rounded-lg bg-nikufra/10 hover:bg-nikufra/20 text-nikufra px-3 py-2 text-xs font-medium transition flex items-center justify-center gap-2"
        >
          <Target className="w-3 h-3" />
          Ver detalhes de {suggestion.sku}
        </button>
      )}
    </motion.div>
  )
}

const ABCXYZMatrixComponent: React.FC<{
  matrix?: ABCXYZMatrix
  activeClass: string
  onCellClick: (abc: 'A' | 'B' | 'C', xyz: 'X' | 'Y' | 'Z') => void
}> = ({ matrix, activeClass, onCellClick }) => {
  if (!matrix) return null
  
  const cellColors = {
    AX: 'from-green-500/30 to-green-600/10',
    AY: 'from-green-400/20 to-amber-500/10',
    AZ: 'from-amber-500/20 to-amber-600/10',
    BX: 'from-green-400/20 to-green-500/10',
    BY: 'from-amber-400/20 to-amber-500/10',
    BZ: 'from-amber-500/20 to-red-500/10',
    CX: 'from-amber-400/20 to-amber-500/10',
    CY: 'from-amber-500/20 to-red-400/10',
    CZ: 'from-red-500/30 to-red-600/10',
  }
  
  return (
    <div className="rounded-xl border border-border bg-surface p-4">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h4 className="text-sm font-semibold text-text-primary">Matriz ABC/XYZ</h4>
          <p className="text-xs text-text-muted mt-1">Clica para filtrar por classe</p>
        </div>
        <div className="text-xs text-text-muted">
          <span className="text-green-400">■</span> Estável &nbsp;
          <span className="text-amber-400">■</span> Médio &nbsp;
          <span className="text-red-400">■</span> Volátil
        </div>
      </div>
      
      <div className="overflow-x-auto">
        <table className="w-full text-center text-sm">
          <thead>
            <tr>
              <th className="py-2 px-3 text-left text-text-muted text-xs">ABC / XYZ</th>
              <th className="py-2 px-3 text-text-muted text-xs">X (Estável)</th>
              <th className="py-2 px-3 text-text-muted text-xs">Y (Médio)</th>
              <th className="py-2 px-3 text-text-muted text-xs">Z (Volátil)</th>
            </tr>
          </thead>
          <tbody>
            {(['A', 'B', 'C'] as const).map((abc) => (
              <tr key={abc}>
                <td className="py-2 px-3 text-left font-semibold text-text-primary">
                  {abc} ({abc === 'A' ? 'Alto Valor' : abc === 'B' ? 'Médio Valor' : 'Baixo Valor'})
                </td>
                {(['X', 'Y', 'Z'] as const).map((xyz) => {
                  const cellKey = `${abc}${xyz}` as keyof typeof cellColors
                  const isActive = activeClass === cellKey
                  const count = matrix[abc]?.[xyz] ?? 0
                  return (
                    <td key={xyz} className="py-2 px-3">
                      <button
                        onClick={() => onCellClick(abc, xyz)}
                        className={`w-full rounded-lg px-4 py-3 font-bold transition-all ${
                          isActive
                            ? 'bg-nikufra text-white shadow-lg shadow-nikufra/30 scale-105'
                            : `bg-gradient-to-br ${cellColors[cellKey]} text-text-primary hover:scale-105 border border-border/50`
                        }`}
                      >
                        {count}
                      </button>
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

// ═══════════════════════════════════════════════════════════════════
// MAIN COMPONENT
// ═══════════════════════════════════════════════════════════════════

export const SmartInventory: React.FC = () => {
  const [selectedSKU, setSelectedSKU] = useState<string | null>(null)
  const [selectedWarehouse, setSelectedWarehouse] = useState<string | null>(null)
  // Consolidated tabs (Contrato 18): reduced from 10 to 6
  const [activeTab, setActiveTab] = useState<'realtime' | 'matrix' | 'forecast' | 'mrp-complete' | 'bom-explosion' | 'operational-data'>('realtime')
  const [selectedOrders, setSelectedOrders] = useState<string[]>([])
  const [classFilter, setClassFilter] = useState<string>('')
  const [searchTerm, setSearchTerm] = useState('')

  // ═══════════════════════════════════════════════════════════════
  // QUERIES
  // ═══════════════════════════════════════════════════════════════

  const { data: stockData, isLoading: loadingStock, refetch: refetchStock } = useQuery({
    queryKey: ['inventory-stock', selectedWarehouse],
    queryFn: () => fetchStock(selectedWarehouse || undefined),
    staleTime: 30_000,
    refetchInterval: 60_000, // Auto-refresh every minute
  })

  const { data: forecastData, isLoading: loadingForecast } = useQuery({
    queryKey: ['inventory-forecast', selectedSKU],
    queryFn: () => fetchForecast(selectedSKU!),
    enabled: !!selectedSKU,
    staleTime: 60_000,
  })

  const { data: ropData, isLoading: loadingROP } = useQuery({
    queryKey: ['inventory-rop', selectedSKU],
    queryFn: () => fetchROP(selectedSKU!),
    enabled: !!selectedSKU,
    staleTime: 60_000,
  })

  const { data: suggestionsData, isLoading: loadingSuggestions } = useQuery({
    queryKey: ['inventory-suggestions'],
    queryFn: fetchSuggestions,
    staleTime: 30_000,
  })

  const { data: inventoryOldData } = useQuery({
    queryKey: ['inventory-old', classFilter, searchTerm],
    queryFn: () => fetchInventoryOld(classFilter || undefined, searchTerm || undefined),
    staleTime: 60_000,
  })

  // ═══════════════════════════════════════════════════════════════
  // DERIVED DATA
  // ═══════════════════════════════════════════════════════════════

  const stockItems: StockItem[] = stockData?.stock || []
  const suggestions: InventorySuggestion[] = suggestionsData?.suggestions || []
  const matrix: ABCXYZMatrix | undefined = inventoryOldData?.matrix
  const skuList: InventorySKU[] = useMemo(() => {
    if (!inventoryOldData?.skus) return []
    return inventoryOldData.skus.map((raw: any) => ({
      sku: String(raw.sku || raw.SKU || ''),
      classe: String(raw.classe || raw.class || ''),
      xyz: String(raw.xyz || raw.XYZ || ''),
      stock_atual: Number(raw.stock_atual || raw.stockAtual || 0),
      ads_180: Number(raw.ads_180 || raw.ads || 0),
      cobertura_dias: Number(raw.cobertura_dias || raw.cobertura || 0),
      risco_30d: Number(raw.risco_30d || raw.risco || 0) * (Number(raw.risco_30d || raw.risco || 0) <= 1 ? 100 : 1),
      rop: Number(raw.rop || 0),
      acao: String(raw.acao || 'Monitorizar'),
    }))
  }, [inventoryOldData?.skus])

  // KPIs
  const totalSKUs = stockData?.total_skus || skuList.length || 0
  const totalStock = stockItems.reduce((sum, item) => sum + item.quantity_available, 0)
  const highRiskItems = suggestions.filter(s => s.priority === 'HIGH').length
  const avgCoverage = skuList.length > 0 
    ? skuList.reduce((sum, s) => sum + s.cobertura_dias, 0) / skuList.length 
    : stockItems.length > 0 ? 30 : 0

  // Filter stock items
  const filteredStockItems = useMemo(() => {
    let items = stockItems
    if (searchTerm) {
      items = items.filter(item => 
        item.sku.toLowerCase().includes(searchTerm.toLowerCase())
      )
    }
    return items
  }, [stockItems, searchTerm])

  // ═══════════════════════════════════════════════════════════════
  // HANDLERS
  // ═══════════════════════════════════════════════════════════════

  const handleMatrixClick = (abc: 'A' | 'B' | 'C', xyz: 'X' | 'Y' | 'Z') => {
    const key = `${abc}${xyz}`
    setClassFilter(current => current === key ? '' : key)
  }

  // ═══════════════════════════════════════════════════════════════
  // RENDER
  // ═══════════════════════════════════════════════════════════════

  return (
    <div className="space-y-6">
      {/* Header */}
      <header className="flex flex-col md:flex-row md:items-start md:justify-between gap-4">
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <Brain className="w-6 h-6 text-nikufra" />
            <p className="text-xs uppercase tracking-[0.4em] text-text-muted">SmartInventory</p>
          </div>
          <h2 className="text-2xl font-semibold text-text-primary">
            Digital Twin de Inventário
          </h2>
          <p className="max-w-3xl text-sm text-text-muted">
            Stock em tempo real, matriz ABC/XYZ, forecasting avançado (ARIMA + SNR), ROP dinâmico e sugestões inteligentes.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => refetchStock()}
            className="flex items-center gap-2 px-3 py-2 rounded-lg border border-border bg-surface hover:bg-background text-text-muted hover:text-text-primary transition"
          >
            <RefreshCw className="w-4 h-4" />
            Atualizar
          </button>
          <button className="flex items-center gap-2 px-3 py-2 rounded-lg border border-border bg-surface hover:bg-background text-text-muted hover:text-text-primary transition">
            <Download className="w-4 h-4" />
            Exportar
          </button>
        </div>
      </header>

      {/* KPIs */}
      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3">
        <MetricCard
          title="Total SKUs"
          value={totalSKUs}
          icon={<Package className="w-5 h-5 text-cyan-400" />}
        />
        <MetricCard
          title="Stock Disponível"
          value={totalStock.toLocaleString('pt-PT')}
          subtitle="unidades"
          icon={<BarChart3 className="w-5 h-5 text-blue-400" />}
        />
        <MetricCard
          title="Cobertura Média"
          value={`${avgCoverage.toFixed(0)}d`}
          subtitle="dias de stock"
          icon={<Clock className="w-5 h-5 text-green-400" />}
        />
        <MetricCard
          title="Alertas Críticos"
          value={highRiskItems}
          subtitle="requerem ação"
          icon={<AlertTriangle className="w-5 h-5 text-red-400" />}
          variant={highRiskItems > 0 ? 'danger' : 'default'}
        />
        <MetricCard
          title="Sugestões"
          value={suggestions.length}
          subtitle="ações pendentes"
          icon={<Activity className="w-5 h-5 text-amber-400" />}
          variant={suggestions.length > 5 ? 'warning' : 'default'}
        />
        <MetricCard
          title="Armazéns"
          value={stockData?.warehouses?.length || 0}
          subtitle="ativos"
          icon={<Warehouse className="w-5 h-5 text-purple-400" />}
        />
      </div>

      {/* Tabs - Consolidated (Contrato 18) */}
      <div className="flex items-center gap-2 p-1 rounded-lg bg-surface border border-border w-fit overflow-x-auto">
        {[
          { id: 'realtime', label: 'Stock & Distribuição', icon: <Activity className="w-4 h-4" /> },
          { id: 'matrix', label: 'Matriz ABC/XYZ', icon: <BarChart3 className="w-4 h-4" /> },
          { id: 'forecast', label: 'Forecast & ROP', icon: <LineChart className="w-4 h-4" /> },
          { id: 'mrp-complete', label: 'MRP Completo', icon: <Brain className="w-4 h-4" /> },
          { id: 'bom-explosion', label: 'BOM & Estrutura', icon: <Layers className="w-4 h-4" /> },
          { id: 'operational-data', label: 'Dados Operacionais', icon: <Activity className="w-4 h-4" /> },
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id as typeof activeTab)}
            className={`flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition ${
              activeTab === tab.id
                ? 'bg-nikufra text-white'
                : 'text-text-muted hover:text-text-primary'
            }`}
          >
            {tab.icon}
            {tab.label}
          </button>
        ))}
      </div>

      <AnimatePresence mode="wait">
        {/* TAB: Stock Real-Time */}
        {activeTab === 'realtime' && (
          <motion.div
            key="realtime"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="grid gap-6 lg:grid-cols-3"
          >
            {/* Stock por Armazém */}
            {stockData?.warehouses && stockData.warehouses.length > 1 && (
              <div className="lg:col-span-3 rounded-2xl border border-border bg-surface p-6">
                <h3 className="text-lg font-bold text-text-primary mb-4 flex items-center gap-2">
                  <Warehouse className="w-5 h-5 text-nikufra" />
                  Distribuição por Armazém
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {stockData.warehouses.map((wh: string) => {
                    const whStock = stockItems.filter(item => item.warehouse_id === wh)
                    const whTotal = whStock.reduce((sum, item) => sum + item.quantity_available, 0)
                    const whSKUs = new Set(whStock.map(item => item.sku)).size
                    const isSelected = selectedWarehouse === wh
                    return (
                      <button
                        key={wh}
                        onClick={() => setSelectedWarehouse(isSelected ? null : wh)}
                        className={`rounded-xl border p-4 text-left transition-all ${
                          isSelected
                            ? 'border-nikufra bg-nikufra/10 shadow-lg shadow-nikufra/20'
                            : 'border-border bg-background hover:border-nikufra/50'
                        }`}
                      >
                        <div className="flex items-center justify-between mb-2">
                          <span className="font-semibold text-text-primary">{wh}</span>
                          <span className="text-xs text-text-muted bg-surface px-2 py-1 rounded">{whSKUs} SKUs</span>
                        </div>
                        <p className="text-3xl font-bold text-nikufra">{whTotal.toLocaleString('pt-PT')}</p>
                        <p className="text-xs text-text-muted mt-1">unidades disponíveis</p>
                      </button>
                    )
                  })}
                </div>
              </div>
            )}

            {/* Stock Table */}
            <div className="lg:col-span-2 rounded-2xl border border-border bg-surface p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-bold text-text-primary flex items-center gap-2">
                  <Package className="w-5 h-5 text-nikufra" />
                  Stock em Tempo Real
                </h3>
                <div className="flex items-center gap-2">
                  <div className="relative">
                    <Filter className="w-4 h-4 text-text-muted absolute left-3 top-1/2 -translate-y-1/2" />
                    <input
                      type="text"
                      placeholder="Filtrar SKU..."
                      value={searchTerm}
                      onChange={(e) => setSearchTerm(e.target.value)}
                      className="pl-9 pr-3 py-2 rounded-lg border border-border bg-background text-sm text-text-primary w-40"
                    />
                  </div>
                  {stockData?.warehouses && stockData.warehouses.length > 0 && (
                    <select
                      value={selectedWarehouse || ''}
                      onChange={(e) => setSelectedWarehouse(e.target.value || null)}
                      className="rounded-lg border border-border bg-background px-3 py-2 text-sm text-text-primary"
                    >
                      <option value="">Todos Armazéns</option>
                      {stockData.warehouses.map((wh: string) => (
                        <option key={wh} value={wh}>{wh}</option>
                      ))}
                    </select>
                  )}
                </div>
              </div>
              
              {loadingStock ? (
                <div className="text-center py-12 text-text-muted">
                  <RefreshCw className="w-8 h-8 mx-auto mb-2 animate-spin" />
                  A carregar stock...
                </div>
              ) : filteredStockItems.length === 0 ? (
                <div className="text-center py-12 text-text-muted">
                  <Package className="w-12 h-12 mx-auto mb-2 opacity-50" />
                  Sem dados de stock
                </div>
              ) : (
                <div className="overflow-x-auto max-h-[500px] overflow-y-auto">
                  <table className="w-full text-sm">
                    <thead className="sticky top-0 bg-surface">
                      <tr className="border-b border-border">
                        <th className="text-left py-3 px-2 text-text-muted font-medium">SKU</th>
                        <th className="text-left py-3 px-2 text-text-muted font-medium">Armazém</th>
                        <th className="text-right py-3 px-2 text-text-muted font-medium">On Hand</th>
                        <th className="text-right py-3 px-2 text-text-muted font-medium">Disponível</th>
                        <th className="text-right py-3 px-2 text-text-muted font-medium">Comprometido</th>
                        <th className="text-right py-3 px-2 text-text-muted font-medium">Em Trânsito</th>
                        <th className="text-center py-3 px-2 text-text-muted font-medium">Status</th>
                      </tr>
                    </thead>
                    <tbody>
                      {filteredStockItems.slice(0, 50).map((item, idx) => {
                        const isSelected = selectedSKU === item.sku
                        const isLowStock = item.quantity_available < 50
                        return (
                          <tr
                            key={`${item.sku}-${item.warehouse_id}-${idx}`}
                            className={`border-b border-border/30 hover:bg-background/50 cursor-pointer transition ${
                              isSelected ? 'bg-nikufra/10' : ''
                            }`}
                            onClick={() => setSelectedSKU(item.sku)}
                          >
                            <td className="py-3 px-2 text-text-primary font-medium">{item.sku}</td>
                            <td className="py-3 px-2 text-text-muted">{item.warehouse_id}</td>
                            <td className="py-3 px-2 text-right text-text-primary">{item.quantity_on_hand.toLocaleString('pt-PT')}</td>
                            <td className={`py-3 px-2 text-right font-medium ${isLowStock ? 'text-amber-400' : 'text-text-primary'}`}>
                              {item.quantity_available.toLocaleString('pt-PT')}
                            </td>
                            <td className="py-3 px-2 text-right text-text-muted">{item.quantity_committed.toLocaleString('pt-PT')}</td>
                            <td className="py-3 px-2 text-right text-cyan-400">{item.quantity_in_transit.toLocaleString('pt-PT')}</td>
                            <td className="py-3 px-2 text-center">
                              {isLowStock ? (
                                <span className="text-xs px-2 py-1 rounded-full bg-amber-500/20 text-amber-400">Baixo</span>
                              ) : (
                                <span className="text-xs px-2 py-1 rounded-full bg-green-500/20 text-green-400">OK</span>
                              )}
                            </td>
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                  {filteredStockItems.length > 50 && (
                    <p className="text-xs text-text-muted text-center py-2">
                      Mostrando 50 de {filteredStockItems.length} itens
                    </p>
                  )}
                </div>
              )}
            </div>

            {/* Sugestões */}
            <div className="rounded-2xl border border-border bg-surface p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-bold text-text-primary flex items-center gap-2">
                  <Brain className="w-5 h-5 text-nikufra" />
                  Sugestões IA
                </h3>
                {suggestions.length > 0 && (
                  <div className="flex gap-1">
                    <span className="text-xs px-2 py-1 rounded-full bg-red-500/20 text-red-400">
                      {suggestions.filter(s => s.priority === 'HIGH').length} Alta
                    </span>
                  </div>
                )}
              </div>
              
              {loadingSuggestions ? (
                <div className="text-center py-8 text-text-muted">A carregar...</div>
              ) : suggestions.length === 0 ? (
                <div className="text-center py-8 text-text-muted">
                  <Activity className="w-10 h-10 mx-auto mb-2 opacity-50" />
                  Nenhuma sugestão no momento
                </div>
              ) : (
                <div className="space-y-3 max-h-[500px] overflow-y-auto">
                  {suggestions.slice(0, 10).map((suggestion, idx) => (
                    <SuggestionCard
                      key={idx}
                      suggestion={suggestion}
                      onSelectSKU={() => {
                        setSelectedSKU(suggestion.sku)
                        setActiveTab('forecast')
                      }}
                    />
                  ))}
                </div>
              )}
            </div>
          </motion.div>
        )}

        {/* TAB: Matriz ABC/XYZ */}
        {activeTab === 'matrix' && (
          <motion.div
            key="matrix"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="space-y-6"
          >
            <ABCXYZMatrixComponent
              matrix={matrix}
              activeClass={classFilter}
              onCellClick={handleMatrixClick}
            />
            
            {/* SKU Table with ABC/XYZ data */}
            <div className="rounded-2xl border border-border bg-surface p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-bold text-text-primary">
                  SKUs por Classe {classFilter && <span className="text-nikufra">({classFilter})</span>}
                </h3>
                <input
                  type="text"
                  placeholder="Pesquisar SKU..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="px-3 py-2 rounded-lg border border-border bg-background text-sm text-text-primary"
                />
              </div>
              
              {skuList.length === 0 ? (
                <div className="text-center py-8 text-text-muted">
                  Carregue dados Excel para ver a classificação ABC/XYZ
                </div>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-border">
                        <th className="text-left py-3 px-2 text-text-muted">SKU</th>
                        <th className="text-center py-3 px-2 text-text-muted">Classe</th>
                        <th className="text-center py-3 px-2 text-text-muted">XYZ</th>
                        <th className="text-right py-3 px-2 text-text-muted">Stock Atual</th>
                        <th className="text-right py-3 px-2 text-text-muted">ADS-180</th>
                        <th className="text-right py-3 px-2 text-text-muted">Cobertura</th>
                        <th className="text-center py-3 px-2 text-text-muted">Risco 30d</th>
                        <th className="text-right py-3 px-2 text-text-muted">ROP</th>
                        <th className="text-center py-3 px-2 text-text-muted">Ação</th>
                      </tr>
                    </thead>
                    <tbody>
                      {skuList
                        .filter(s => !classFilter || `${s.classe}${s.xyz}` === classFilter)
                        .filter(s => !searchTerm || s.sku.toLowerCase().includes(searchTerm.toLowerCase()))
                        .slice(0, 50)
                        .map((sku, idx) => (
                          <tr
                            key={idx}
                            className="border-b border-border/30 hover:bg-background/50 cursor-pointer"
                            onClick={() => {
                              setSelectedSKU(sku.sku)
                              setActiveTab('forecast')
                            }}
                          >
                            <td className="py-3 px-2 text-text-primary font-medium">{sku.sku}</td>
                            <td className="py-3 px-2 text-center">
                              <span className={`px-2 py-1 rounded text-xs font-medium ${
                                sku.classe === 'A' ? 'bg-green-500/20 text-green-400' :
                                sku.classe === 'B' ? 'bg-amber-500/20 text-amber-400' :
                                'bg-red-500/20 text-red-400'
                              }`}>{sku.classe}</span>
                            </td>
                            <td className="py-3 px-2 text-center">
                              <span className={`px-2 py-1 rounded text-xs font-medium ${
                                sku.xyz === 'X' ? 'bg-cyan-500/20 text-cyan-400' :
                                sku.xyz === 'Y' ? 'bg-purple-500/20 text-purple-400' :
                                'bg-orange-500/20 text-orange-400'
                              }`}>{sku.xyz}</span>
                            </td>
                            <td className="py-3 px-2 text-right text-text-primary">{sku.stock_atual.toLocaleString('pt-PT')}</td>
                            <td className="py-3 px-2 text-right text-text-muted">{sku.ads_180.toFixed(1)}</td>
                            <td className="py-3 px-2 text-right text-text-primary">{sku.cobertura_dias.toFixed(0)}d</td>
                            <td className="py-3 px-2 text-center">
                              <span className={`px-2 py-1 rounded text-xs font-semibold ${
                                sku.risco_30d > 50 ? 'bg-red-500/20 text-red-400' :
                                sku.risco_30d > 25 ? 'bg-amber-500/20 text-amber-400' :
                                'bg-green-500/20 text-green-400'
                              }`}>{sku.risco_30d.toFixed(1)}%</span>
                            </td>
                            <td className="py-3 px-2 text-right text-text-primary">{sku.rop.toLocaleString('pt-PT')}</td>
                            <td className="py-3 px-2 text-center">
                              <span className={`px-2 py-1 rounded text-xs font-medium ${
                                sku.acao === 'Comprar agora' ? 'bg-red-500/20 text-red-400' :
                                sku.acao === 'Excesso' ? 'bg-amber-500/20 text-amber-400' :
                                'bg-nikufra/20 text-nikufra'
                              }`}>{sku.acao}</span>
                            </td>
                          </tr>
                        ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </motion.div>
        )}

        {/* TAB: Forecast & ROP */}
        {activeTab === 'forecast' && (
          <motion.div
            key="forecast"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="grid gap-6 lg:grid-cols-2"
          >
            {/* SKU Selection */}
            <div className="lg:col-span-2 rounded-xl border border-border bg-surface p-4">
              <div className="flex items-center gap-4">
                <span className="text-sm text-text-muted">SKU Selecionado:</span>
                <select
                  value={selectedSKU || ''}
                  onChange={(e) => setSelectedSKU(e.target.value || null)}
                  className="flex-1 rounded-lg border border-border bg-background px-3 py-2 text-text-primary"
                >
                  <option value="">Selecionar SKU...</option>
                  {stockItems.map(item => item.sku).filter((v, i, a) => a.indexOf(v) === i).map(sku => (
                    <option key={sku} value={sku}>{sku}</option>
                  ))}
                </select>
              </div>
            </div>

            {!selectedSKU ? (
              <div className="lg:col-span-2 rounded-2xl border border-border bg-surface p-12 text-center">
                <LineChart className="w-16 h-16 mx-auto mb-4 text-text-muted/50" />
                <h3 className="text-lg font-semibold text-text-primary mb-2">Seleciona um SKU</h3>
                <p className="text-sm text-text-muted">
                  Escolhe um SKU da tabela de stock ou do dropdown acima para ver forecast e ROP
                </p>
              </div>
            ) : (
              <>
                {/* Forecast Panel */}
                <div className="rounded-2xl border border-border bg-surface p-6">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-lg font-bold text-text-primary flex items-center gap-2">
                      <LineChart className="w-5 h-5 text-nikufra" />
                      Forecast de Demanda
                    </h3>
                    {forecastData && (
                      <span className={`text-xs px-2 py-1 rounded-full ${
                        forecastData.snr_class === 'HIGH' ? 'bg-green-500/20 text-green-400' :
                        forecastData.snr_class === 'MEDIUM' ? 'bg-amber-500/20 text-amber-400' :
                        'bg-red-500/20 text-red-400'
                      }`}>
                        {forecastData.model_used} • SNR: {forecastData.snr.toFixed(2)}
                      </span>
                    )}
                  </div>
                  
                  {loadingForecast ? (
                    <div className="text-center py-8 text-text-muted">A carregar forecast...</div>
                  ) : forecastData ? (
                    <div className="space-y-4">
                      <div className="grid grid-cols-2 gap-4">
                        <div className="rounded-lg border border-border bg-background p-4">
                          <p className="text-xs text-text-muted mb-1">Média Prevista (90d)</p>
                          <p className="text-2xl font-bold text-text-primary">
                            {((Object.values(forecastData.forecast) as number[]).reduce((a, b) => a + b, 0) / 
                              Math.max(1, Object.keys(forecastData.forecast).length)).toFixed(1)}
                          </p>
                        </div>
                        <div className="rounded-lg border border-border bg-background p-4">
                          <p className="text-xs text-text-muted mb-1">Confiança</p>
                          <p className="text-2xl font-bold text-nikufra">
                            {(forecastData.confidence_score * 100).toFixed(0)}%
                          </p>
                        </div>
                      </div>
                      
                      {/* SNR Explanation */}
                      <div className={`rounded-lg p-4 ${
                        forecastData.snr_class === 'HIGH' ? 'bg-green-500/10 border border-green-500/30' :
                        forecastData.snr_class === 'MEDIUM' ? 'bg-amber-500/10 border border-amber-500/30' :
                        'bg-red-500/10 border border-red-500/30'
                      }`}>
                        <p className="text-sm">
                          <span className="font-semibold">Signal-to-Noise Ratio (SNR):</span>{' '}
                          {forecastData.snr_class === 'HIGH' 
                            ? 'Alta confiança - dados com padrão claro'
                            : forecastData.snr_class === 'MEDIUM'
                            ? 'Média confiança - algum ruído nos dados'
                            : 'Baixa confiança - dados muito voláteis'}
                        </p>
                      </div>
                      
                      {forecastData.metrics && (
                        <div className="grid grid-cols-3 gap-2 text-center">
                          {forecastData.metrics.MAPE !== undefined && (
                            <div className="rounded-lg border border-border bg-background p-2">
                              <p className="text-xs text-text-muted">MAPE</p>
                              <p className="text-lg font-bold text-text-primary">{forecastData.metrics.MAPE.toFixed(1)}%</p>
                            </div>
                          )}
                          {forecastData.metrics.RMSE !== undefined && (
                            <div className="rounded-lg border border-border bg-background p-2">
                              <p className="text-xs text-text-muted">RMSE</p>
                              <p className="text-lg font-bold text-text-primary">{forecastData.metrics.RMSE.toFixed(1)}</p>
                            </div>
                          )}
                          {forecastData.metrics.MAE !== undefined && (
                            <div className="rounded-lg border border-border bg-background p-2">
                              <p className="text-xs text-text-muted">MAE</p>
                              <p className="text-lg font-bold text-text-primary">{forecastData.metrics.MAE.toFixed(1)}</p>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="text-center py-8 text-text-muted">Sem dados de forecast</div>
                  )}
                </div>

                {/* ROP Panel */}
                <div className="rounded-2xl border border-border bg-surface p-6">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-lg font-bold text-text-primary flex items-center gap-2">
                      <Target className="w-5 h-5 text-nikufra" />
                      Reorder Point (ROP)
                    </h3>
                    {ropData && (
                      <span className="text-3xl font-bold text-nikufra">{ropData.rop.toFixed(0)}</span>
                    )}
                  </div>
                  
                  {loadingROP ? (
                    <div className="text-center py-8 text-text-muted">A carregar ROP...</div>
                  ) : ropData ? (
                    <div className="space-y-4">
                      {/* Stock vs ROP bar */}
                      {ropData.current_stock !== undefined && (
                        <div className="space-y-2">
                          <div className="flex items-center justify-between text-sm">
                            <span className="text-text-muted">Stock Atual vs ROP</span>
                            <span className={`font-bold ${
                              ropData.current_stock < ropData.rop ? 'text-red-400' :
                              ropData.current_stock < ropData.rop * 1.2 ? 'text-amber-400' :
                              'text-green-400'
                            }`}>
                              {ropData.current_stock.toFixed(0)} / {ropData.rop.toFixed(0)}
                            </span>
                          </div>
                          <div className="h-4 bg-background rounded-full overflow-hidden relative">
                            <div 
                              className={`h-full transition-all ${
                                ropData.current_stock < ropData.rop ? 'bg-red-500' :
                                ropData.current_stock < ropData.rop * 1.2 ? 'bg-amber-500' :
                                'bg-green-500'
                              }`}
                              style={{ width: `${Math.min(100, (ropData.current_stock / (ropData.rop * 1.5)) * 100)}%` }}
                            />
                            <div 
                              className="absolute top-0 bottom-0 w-1 bg-nikufra"
                              style={{ left: `${(ropData.rop / (ropData.rop * 1.5)) * 100}%` }}
                            />
                          </div>
                        </div>
                      )}
                      
                      <div className="grid grid-cols-2 gap-4">
                        <div className="rounded-lg border border-border bg-background p-4">
                          <p className="text-xs text-text-muted mb-1">Safety Stock</p>
                          <p className="text-2xl font-bold text-text-primary">{ropData.safety_stock.toFixed(0)}</p>
                        </div>
                        <div className="rounded-lg border border-border bg-background p-4">
                          <p className="text-xs text-text-muted mb-1">Cobertura</p>
                          <p className="text-2xl font-bold text-text-primary">{ropData.coverage_days.toFixed(0)}d</p>
                        </div>
                      </div>
                      
                      {/* Risk 30d */}
                      <div className="space-y-2">
                        <div className="flex items-center justify-between text-sm">
                          <span className="text-text-muted">Risco de Ruptura (30 dias)</span>
                          <span className={`font-bold ${
                            ropData.risk_30d > 50 ? 'text-red-400' :
                            ropData.risk_30d > 20 ? 'text-amber-400' :
                            'text-green-400'
                          }`}>
                            {ropData.risk_30d.toFixed(1)}%
                          </span>
                        </div>
                        <div className="h-3 bg-background rounded-full overflow-hidden">
                          <div 
                            className={`h-full ${
                              ropData.risk_30d > 50 ? 'bg-red-500' :
                              ropData.risk_30d > 20 ? 'bg-amber-500' :
                              'bg-green-500'
                            }`}
                            style={{ width: `${Math.min(100, ropData.risk_30d)}%` }}
                          />
                        </div>
                      </div>
                      
                      {/* Alerts */}
                      {ropData.current_stock !== undefined && ropData.current_stock < ropData.rop && (
                        <div className="rounded-lg bg-red-500/10 border border-red-500/40 p-4">
                          <div className="flex items-start gap-3">
                            <AlertTriangle className="w-5 h-5 text-red-400 mt-0.5" />
                            <div>
                              <p className="font-semibold text-red-400">Stock Abaixo do ROP!</p>
                              <p className="text-sm text-text-muted mt-1">
                                Recomendado encomendar {ropData.reorder_quantity.toFixed(0)} unidades
                              </p>
                            </div>
                          </div>
                        </div>
                      )}
                      
                      {ropData.days_until_rop !== undefined && ropData.days_until_rop > 0 && ropData.days_until_rop < 30 && (
                        <div className="rounded-lg bg-amber-500/10 border border-amber-500/40 p-4">
                          <div className="flex items-start gap-3">
                            <Clock className="w-5 h-5 text-amber-400 mt-0.5" />
                            <div>
                              <p className="font-semibold text-amber-400">{ropData.days_until_rop.toFixed(0)} dias até ROP</p>
                              <p className="text-sm text-text-muted mt-1">
                                Considerar antecipar encomenda
                              </p>
                            </div>
                          </div>
                        </div>
                      )}
                      
                      {/* Explanation */}
                      {ropData.explanation && (
                        <div className="rounded-lg bg-surface border border-border p-4">
                          <p className="text-xs text-text-muted">{ropData.explanation}</p>
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="text-center py-8 text-text-muted">Sem dados de ROP</div>
                  )}
                </div>
              </>
            )}
          </motion.div>
        )}

        {/* MRP Encomendas - Now integrated into MRP Completo */}

        {/* TAB: MRP Completo - Unified MRP Hub (Contrato 18) */}
        {activeTab === 'mrp-complete' && (
          <MRPUnifiedPanel 
            selectedOrders={selectedOrders}
            onSelectOrders={setSelectedOrders}
          />
        )}

        {/* TAB: BOM Explosion */}
        {activeTab === 'bom-explosion' && (
          <BOMExplosionPanel />
        )}

        {/* TAB: Dados Operacionais */}
        {activeTab === 'operational-data' && (
          <OperationalDataPanel />
        )}
      </AnimatePresence>
    </div>
  )
}

// ═══════════════════════════════════════════════════════════════════
// MRP ORDERS PANEL COMPONENT
// ═══════════════════════════════════════════════════════════════════

type MRPOrdersStatus = {
  order_id: string
  product_id: string
  quantity: number
  due_date: string
  material_status: 'OK' | 'SHORTAGE' | 'PARTIAL'
  purchase_suggestions: number
}

type PurchaseSuggestionMRP = {
  component_id: string
  quantity: number
  due_date: string
  source_orders: string[]
  lead_time_days: number
}

type InternalOrderSuggestionMRP = {
  item_id: string
  quantity: number
  due_date: string
  source_orders: string[]
}

// ═══════════════════════════════════════════════════════════════════
// MRP UNIFIED PANEL - Consolidated MRP Hub (Contrato 18)
// ═══════════════════════════════════════════════════════════════════

type MRPSubTab = 'engine' | 'orders' | 'parameters' | 'forecast'

const MRPUnifiedPanel: React.FC<{
  selectedOrders: string[]
  onSelectOrders: (orders: string[]) => void
}> = ({ selectedOrders, onSelectOrders }) => {
  const [subTab, setSubTab] = useState<MRPSubTab>('engine')
  
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      className="space-y-4"
    >
      {/* Sub-tabs for MRP sections */}
      <div className="flex items-center gap-2 p-1 rounded-lg bg-slate-800/50 border border-slate-700/50 w-fit">
        {[
          { id: 'engine' as MRPSubTab, label: 'Motor MRP', icon: <Brain className="w-4 h-4" /> },
          { id: 'orders' as MRPSubTab, label: 'Encomendas', icon: <ShoppingCart className="w-4 h-4" /> },
          { id: 'parameters' as MRPSubTab, label: 'Parâmetros', icon: <Settings className="w-4 h-4" /> },
          { id: 'forecast' as MRPSubTab, label: 'Forecast IA', icon: <TrendingUp className="w-4 h-4" /> },
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => setSubTab(tab.id)}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-md text-sm font-medium transition ${
              subTab === tab.id
                ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/30'
                : 'text-slate-400 hover:text-white hover:bg-slate-700/50'
            }`}
          >
            {tab.icon}
            {tab.label}
          </button>
        ))}
      </div>

      {/* Sub-tab content */}
      <AnimatePresence mode="wait">
        {subTab === 'engine' && (
          <motion.div
            key="engine"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <MRPCompletePanel />
          </motion.div>
        )}
        {subTab === 'orders' && (
          <motion.div
            key="orders"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <MRPOrdersPanel selectedOrders={selectedOrders} onSelectOrders={onSelectOrders} />
          </motion.div>
        )}
        {subTab === 'parameters' && (
          <motion.div
            key="parameters"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <MRPParametersPanel />
          </motion.div>
        )}
        {subTab === 'forecast' && (
          <motion.div
            key="forecast"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <Suspense fallback={<div className="flex items-center justify-center py-24"><RefreshCw className="w-8 h-8 text-cyan-400 animate-spin" /></div>}>
              <MRPForecastPanel />
            </Suspense>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  )
}

const MRPOrdersPanel: React.FC<{
  selectedOrders: string[]
  onSelectOrders: (orders: string[]) => void
}> = ({ selectedOrders, onSelectOrders }) => {
  const [isRunningMRP, setIsRunningMRP] = useState(false)
  const [mrpResult, setMrpResult] = useState<{
    purchase_suggestions: PurchaseSuggestionMRP[]
    internal_order_suggestions: InternalOrderSuggestionMRP[]
    shortages: Array<{ component_id: string; shortage_qty: number }>
    warnings: string[]
  } | null>(null)

  // Fetch orders status
  const { data: ordersStatus, isLoading: loadingOrders, refetch: refetchOrders } = useQuery({
    queryKey: ['mrp-orders-status'],
    queryFn: async () => {
      const res = await fetch(`${API_BASE_URL}/inventory/mrp/orders-status`)
      if (!res.ok) throw new Error('Erro ao carregar status de encomendas')
      return res.json()
    },
    staleTime: 30_000,
  })

  const orders: MRPOrdersStatus[] = ordersStatus?.orders || []
  const summary = ordersStatus?.summary || {}

  // Run MRP for selected orders
  const runMRP = async () => {
    if (selectedOrders.length === 0) return
    
    setIsRunningMRP(true)
    try {
      const res = await fetch(`${API_BASE_URL}/inventory/mrp/run-from-orders`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          order_ids: selectedOrders,
          horizon: null, // Use default
        }),
      })
      if (!res.ok) throw new Error('Erro ao executar MRP')
      const result = await res.json()
      setMrpResult(result)
    } catch (error) {
      console.error('MRP error:', error)
    } finally {
      setIsRunningMRP(false)
    }
  }

  const toggleOrderSelection = (orderId: string) => {
    if (selectedOrders.includes(orderId)) {
      onSelectOrders(selectedOrders.filter(id => id !== orderId))
    } else {
      onSelectOrders([...selectedOrders, orderId])
    }
  }

  const selectAllOrders = () => {
    if (selectedOrders.length === orders.length) {
      onSelectOrders([])
    } else {
      onSelectOrders(orders.map(o => o.order_id))
    }
  }

  const statusColors = {
    OK: 'bg-green-500/10 text-green-400 border-green-500/30',
    PARTIAL: 'bg-amber-500/10 text-amber-400 border-amber-500/30',
    SHORTAGE: 'bg-red-500/10 text-red-400 border-red-500/30',
  }

  return (
    <motion.div
      key="mrp"
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      className="space-y-6"
    >
      {/* Header */}
      <div className="rounded-2xl border border-border bg-surface p-6">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-lg font-bold text-text-primary flex items-center gap-2">
              <ShoppingCart className="w-5 h-5 text-nikufra" />
              MRP a partir de Encomendas
            </h3>
            <p className="text-sm text-text-muted mt-1">
              Explode BOM, verifica stock e gera sugestões de compra/produção
            </p>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={() => refetchOrders()}
              className="rounded-lg border border-border p-2 text-text-muted hover:text-text-primary transition"
            >
              <RefreshCw className="w-4 h-4" />
            </button>
            <button
              onClick={runMRP}
              disabled={selectedOrders.length === 0 || isRunningMRP}
              className="rounded-lg bg-nikufra px-4 py-2 text-sm font-semibold text-background transition hover:bg-nikufra-hover disabled:opacity-50"
            >
              {isRunningMRP ? 'A executar...' : `🔬 Executar MRP (${selectedOrders.length})`}
            </button>
          </div>
        </div>

        {/* Summary Stats */}
        <div className="grid grid-cols-4 gap-4">
          <div className="rounded-lg border border-border bg-background p-3">
            <p className="text-xs text-text-muted">Total Encomendas</p>
            <p className="text-xl font-bold text-text-primary">{orders.length}</p>
          </div>
          <div className="rounded-lg border border-green-500/30 bg-green-500/5 p-3">
            <p className="text-xs text-green-400">OK</p>
            <p className="text-xl font-bold text-green-400">
              {orders.filter(o => o.material_status === 'OK').length}
            </p>
          </div>
          <div className="rounded-lg border border-amber-500/30 bg-amber-500/5 p-3">
            <p className="text-xs text-amber-400">Parcial</p>
            <p className="text-xl font-bold text-amber-400">
              {orders.filter(o => o.material_status === 'PARTIAL').length}
            </p>
          </div>
          <div className="rounded-lg border border-red-500/30 bg-red-500/5 p-3">
            <p className="text-xs text-red-400">Falta</p>
            <p className="text-xl font-bold text-red-400">
              {orders.filter(o => o.material_status === 'SHORTAGE').length}
            </p>
          </div>
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        {/* Orders Table */}
        <div className="rounded-2xl border border-border bg-surface p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-md font-semibold text-text-primary">Encomendas Abertas</h3>
            <button
              onClick={selectAllOrders}
              className="text-xs text-nikufra hover:underline"
            >
              {selectedOrders.length === orders.length ? 'Desselecionar Tudo' : 'Selecionar Tudo'}
            </button>
          </div>

          {loadingOrders ? (
            <div className="text-center py-8 text-text-muted">A carregar...</div>
          ) : orders.length === 0 ? (
            <div className="text-center py-8 text-text-muted">Sem encomendas abertas</div>
          ) : (
            <div className="space-y-2 max-h-96 overflow-auto">
              {orders.map((order) => (
                <button
                  key={order.order_id}
                  onClick={() => toggleOrderSelection(order.order_id)}
                  className={`w-full rounded-lg border p-3 text-left transition ${
                    selectedOrders.includes(order.order_id)
                      ? 'border-nikufra bg-nikufra/10'
                      : 'border-border bg-background hover:border-nikufra/50'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div>
                      <span className="font-medium text-text-primary">{order.order_id}</span>
                      <span className="ml-2 text-xs text-text-muted">{order.product_id}</span>
                    </div>
                    <span className={`text-xs px-2 py-1 rounded border ${statusColors[order.material_status]}`}>
                      {order.material_status}
                    </span>
                  </div>
                  <div className="flex items-center justify-between mt-2 text-xs text-text-muted">
                    <span>{order.quantity} un</span>
                    <span>Due: {new Date(order.due_date).toLocaleDateString('pt-PT')}</span>
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>

        {/* MRP Results */}
        <div className="rounded-2xl border border-border bg-surface p-6">
          <h3 className="text-md font-semibold text-text-primary mb-4">Resultado MRP</h3>

          {!mrpResult ? (
            <div className="text-center py-12 text-text-muted">
              <ShoppingCart className="w-12 h-12 mx-auto mb-3 opacity-30" />
              <p>Seleciona encomendas e clica em "Executar MRP"</p>
              <p className="text-xs mt-1">para ver sugestões de compra e produção</p>
            </div>
          ) : (
            <div className="space-y-4">
              {/* Purchase Suggestions */}
              {mrpResult.purchase_suggestions.length > 0 && (
                <div>
                  <h4 className="text-sm font-medium text-text-muted mb-2 flex items-center gap-2">
                    <ShoppingCart className="w-4 h-4" /> Sugestões de Compra
                  </h4>
                  <div className="space-y-2">
                    {mrpResult.purchase_suggestions.map((s, i) => (
                      <div key={i} className="rounded-lg border border-amber-500/30 bg-amber-500/5 p-3">
                        <div className="flex items-center justify-between">
                          <span className="font-medium text-text-primary">{s.component_id}</span>
                          <span className="text-amber-400 font-semibold">{s.quantity.toFixed(0)} un</span>
                        </div>
                        <div className="text-xs text-text-muted mt-1">
                          Due: {new Date(s.due_date).toLocaleDateString('pt-PT')} • Lead: {s.lead_time_days}d
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Internal Orders */}
              {mrpResult.internal_order_suggestions.length > 0 && (
                <div>
                  <h4 className="text-sm font-medium text-text-muted mb-2 flex items-center gap-2">
                    <Zap className="w-4 h-4" /> Ordens de Produção
                  </h4>
                  <div className="space-y-2">
                    {mrpResult.internal_order_suggestions.map((s, i) => (
                      <div key={i} className="rounded-lg border border-blue-500/30 bg-blue-500/5 p-3">
                        <div className="flex items-center justify-between">
                          <span className="font-medium text-text-primary">{s.item_id}</span>
                          <span className="text-blue-400 font-semibold">{s.quantity.toFixed(0)} un</span>
                        </div>
                        <div className="text-xs text-text-muted mt-1">
                          Due: {new Date(s.due_date).toLocaleDateString('pt-PT')}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Shortages */}
              {mrpResult.shortages.length > 0 && (
                <div>
                  <h4 className="text-sm font-medium text-red-400 mb-2 flex items-center gap-2">
                    <AlertTriangle className="w-4 h-4" /> Faltas Identificadas
                  </h4>
                  <div className="space-y-2">
                    {mrpResult.shortages.map((s, i) => (
                      <div key={i} className="rounded-lg border border-red-500/30 bg-red-500/5 p-3">
                        <span className="font-medium text-red-400">{s.component_id}</span>
                        <span className="ml-2 text-text-muted">-{s.shortage_qty.toFixed(0)} un</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Warnings */}
              {mrpResult.warnings.length > 0 && (
                <div className="rounded-lg border border-amber-500/30 bg-amber-500/5 p-3">
                  <h4 className="text-sm font-medium text-amber-400 mb-1">⚠️ Avisos</h4>
                  <ul className="text-xs text-text-muted space-y-1">
                    {mrpResult.warnings.map((w, i) => (
                      <li key={i}>• {w}</li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Empty state */}
              {mrpResult.purchase_suggestions.length === 0 && 
               mrpResult.internal_order_suggestions.length === 0 && 
               mrpResult.shortages.length === 0 && (
                <div className="text-center py-8 text-green-400">
                  ✅ Todos os materiais disponíveis!
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </motion.div>
  )
}

// ═══════════════════════════════════════════════════════════════════
// MRP COMPLETE PANEL COMPONENT
// ═══════════════════════════════════════════════════════════════════

type MRPStatus = {
  service: string
  version: string
  status: string
  config: {
    horizon_days: number
    period_days: number
    enable_forecast: boolean
    enable_capacity_check: boolean
  }
  runs_stored: number
  item_sources: string[]
  order_sources: string[]
}

type MRPRunSummary = {
  run_id: string
  run_timestamp: string
  items_processed: number
  demands_processed: number
  purchase_orders_count: number
  manufacture_orders_count: number
  shortage_alerts_count: number
  capacity_alerts_count: number
}

type MRPRunDetail = {
  run_id: string
  run_timestamp: string
  config: any
  items_processed: number
  demands_processed: number
  purchase_orders: Array<{
    order_id: string
    item_id: number
    sku: string
    quantity: number
    due_date: string
    suggested_start_date: string
    lead_time_days: number
    source_orders: string[]
  }>
  manufacture_orders: Array<{
    order_id: string
    item_id: number
    sku: string
    quantity: number
    due_date: string
    suggested_start_date: string
    source_orders: string[]
  }>
  shortage_alerts: Array<{
    item_id: number
    sku: string
    shortage_qty: number
    period_start: string
    period_end: string
  }>
  capacity_alerts: Array<{
    work_center: string
    period_start: string
    period_end: string
    required_capacity: number
    available_capacity: number
  }>
}

const MRPCompletePanel: React.FC = () => {
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null)
  const [isRunningMRP, setIsRunningMRP] = useState(false)
  const [isRunningDemo, setIsRunningDemo] = useState(false)
  const [horizonDays, setHorizonDays] = useState(90)
  const [periodDays, setPeriodDays] = useState(7)
  const [showResetConfirm, setShowResetConfirm] = useState(false)

  // Fetch MRP status
  const { data: mrpStatus, isLoading: loadingStatus, refetch: refetchStatus } = useQuery({
    queryKey: ['mrp-status'],
    queryFn: async () => {
      const res = await fetch(`${API_BASE_URL}/mrp/status`)
      if (!res.ok) throw new Error('Erro ao carregar status MRP')
      return res.json() as Promise<MRPStatus>
    },
    staleTime: 30_000,
  })

  // Fetch MRP runs
  const { data: mrpRuns, isLoading: loadingRuns, refetch: refetchRuns } = useQuery({
    queryKey: ['mrp-runs'],
    queryFn: async () => {
      const res = await fetch(`${API_BASE_URL}/mrp/runs?limit=50`)
      if (!res.ok) throw new Error('Erro ao carregar runs MRP')
      const data = await res.json()
      return (data.runs || []) as MRPRunSummary[]
    },
    staleTime: 30_000,
  })

  // Fetch selected run details
  const { data: runDetails, isLoading: loadingDetails } = useQuery({
    queryKey: ['mrp-run-details', selectedRunId],
    queryFn: async () => {
      if (!selectedRunId) return null
      const res = await fetch(`${API_BASE_URL}/mrp/runs/${selectedRunId}`)
      if (!res.ok) throw new Error('Erro ao carregar detalhes do run')
      return res.json() as Promise<MRPRunDetail>
    },
    enabled: !!selectedRunId,
    staleTime: 60_000,
  })

  // Run MRP
  const runMRP = async () => {
    setIsRunningMRP(true)
    try {
      const res = await fetch(`${API_BASE_URL}/mrp/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          horizon_days: horizonDays,
          period_days: periodDays,
          load_from_pdm: true,
        }),
      })
      if (!res.ok) {
        const error = await res.json()
        throw new Error(error.detail || 'Erro ao executar MRP')
      }
      const result = await res.json()
      setSelectedRunId(result.run_id)
      refetchRuns()
      refetchStatus()
    } catch (error: any) {
      console.error('MRP error:', error)
      alert(`Erro ao executar MRP: ${error.message}`)
    } finally {
      setIsRunningMRP(false)
    }
  }

  // Run Demo MRP
  const runDemoMRP = async () => {
    setIsRunningDemo(true)
    try {
      const res = await fetch(`${API_BASE_URL}/mrp/demo`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          num_products: 3,
          num_orders: 5,
          horizon_days: horizonDays,
        }),
      })
      if (!res.ok) {
        const error = await res.json()
        throw new Error(error.detail || 'Erro ao executar demo MRP')
      }
      const result = await res.json()
      setSelectedRunId(result.run_id)
      refetchRuns()
      refetchStatus()
    } catch (error: any) {
      console.error('Demo MRP error:', error)
      alert(`Erro ao executar demo MRP: ${error.message}`)
    } finally {
      setIsRunningDemo(false)
    }
  }

  return (
    <motion.div
      key="mrp-complete"
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      className="space-y-6"
    >
      {/* Header with Status */}
      <div className="rounded-2xl border border-border bg-surface p-6">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-lg font-bold text-text-primary flex items-center gap-2">
              <Brain className="w-5 h-5 text-nikufra" />
              MRP Completo - Material Requirements Planning
            </h3>
            <p className="text-sm text-text-muted mt-1">
              Sistema avançado de planeamento de necessidades de material com BOM multi-nível, lot sizing e verificação de capacidade
            </p>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={() => {
                refetchStatus()
                refetchRuns()
              }}
              className="rounded-lg border border-border p-2 text-text-muted hover:text-text-primary transition"
            >
              <RefreshCw className="w-4 h-4" />
            </button>
            <button
              onClick={runDemoMRP}
              disabled={isRunningDemo || isRunningMRP}
              className="rounded-lg bg-purple-500/10 hover:bg-purple-500/20 text-purple-400 px-4 py-2 text-sm font-semibold transition disabled:opacity-50"
            >
              {isRunningDemo ? 'A executar...' : '🎯 Demo MRP'}
            </button>
            <button
              onClick={runMRP}
              disabled={isRunningMRP || isRunningDemo}
              className="rounded-lg bg-nikufra px-4 py-2 text-sm font-semibold text-background transition hover:bg-nikufra-hover disabled:opacity-50"
            >
              {isRunningMRP ? 'A executar...' : '🚀 Executar MRP'}
            </button>
          </div>
        </div>

        {/* Status Cards */}
        {loadingStatus ? (
          <div className="text-center py-4 text-text-muted">A carregar status...</div>
        ) : mrpStatus ? (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="rounded-lg border border-border bg-background p-3">
              <p className="text-xs text-text-muted">Versão</p>
              <p className="text-lg font-bold text-text-primary">{mrpStatus.version}</p>
            </div>
            <div className="rounded-lg border border-border bg-background p-3">
              <p className="text-xs text-text-muted">Runs Armazenados</p>
              <p className="text-lg font-bold text-nikufra">{mrpStatus.runs_stored}</p>
            </div>
            <div className="rounded-lg border border-border bg-background p-3">
              <p className="text-xs text-text-muted">Horizonte</p>
              <p className="text-lg font-bold text-text-primary">{mrpStatus.config.horizon_days}d</p>
            </div>
            <div className="rounded-lg border border-border bg-background p-3">
              <p className="text-xs text-text-muted">Período</p>
              <p className="text-lg font-bold text-text-primary">{mrpStatus.config.period_days}d</p>
            </div>
          </div>
        ) : null}

        {/* Configuration */}
        <div className="mt-4 pt-4 border-t border-border">
          <div className="flex items-center gap-4">
            <label className="text-sm text-text-muted">Horizonte (dias):</label>
            <input
              type="number"
              value={horizonDays}
              onChange={(e) => setHorizonDays(Number(e.target.value))}
              min={7}
              max={365}
              className="w-20 rounded-lg border border-border bg-background px-3 py-2 text-sm text-text-primary"
            />
            <label className="text-sm text-text-muted">Período (dias):</label>
            <input
              type="number"
              value={periodDays}
              onChange={(e) => setPeriodDays(Number(e.target.value))}
              min={1}
              max={30}
              className="w-20 rounded-lg border border-border bg-background px-3 py-2 text-sm text-text-primary"
            />
          </div>
        </div>

        {/* Reset Confirmation Modal */}
        {showResetConfirm && (
          <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              className="rounded-2xl border border-border bg-surface p-6 max-w-md w-full mx-4"
            >
              <h4 className="text-lg font-bold text-text-primary mb-2">Confirmar Reset MRP</h4>
              <p className="text-sm text-text-muted mb-4">
                Tem a certeza que deseja resetar o serviço MRP? Esta ação irá:
              </p>
              <ul className="text-sm text-text-muted mb-6 space-y-1 list-disc list-inside">
                <li>Limpar todos os runs históricos</li>
                <li>Resetar o serviço MRP</li>
                <li>Esta ação não pode ser desfeita</li>
              </ul>
              <div className="flex items-center justify-end gap-3">
                <button
                  onClick={() => setShowResetConfirm(false)}
                  className="px-4 py-2 rounded-lg border border-border text-text-muted hover:text-text-primary transition"
                >
                  Cancelar
                </button>
                <button
                  onClick={async () => {
                    try {
                      const res = await fetch(`${API_BASE_URL}/mrp/reset`, { method: 'DELETE' })
                      if (!res.ok) throw new Error('Erro ao resetar MRP')
                      setShowResetConfirm(false)
                      refetchStatus()
                      refetchRuns()
                    } catch (error) {
                      console.error('Reset error:', error)
                      alert('Erro ao resetar MRP')
                    }
                  }}
                  className="px-4 py-2 rounded-lg bg-red-500 text-white hover:bg-red-600 transition"
                >
                  Confirmar Reset
                </button>
              </div>
            </motion.div>
          </div>
        )}
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        {/* Runs List */}
        <div className="rounded-2xl border border-border bg-surface p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-md font-semibold text-text-primary">Runs Históricos</h3>
            <span className="text-xs text-text-muted">{mrpRuns?.length || 0} runs</span>
          </div>

          {loadingRuns ? (
            <div className="text-center py-8 text-text-muted">A carregar...</div>
          ) : !mrpRuns || mrpRuns.length === 0 ? (
            <div className="text-center py-8 text-text-muted">
              <ShoppingCart className="w-12 h-12 mx-auto mb-3 opacity-30" />
              <p>Nenhum run MRP executado ainda</p>
              <p className="text-xs mt-1">Clica em "Executar MRP" ou "Demo MRP" para começar</p>
            </div>
          ) : (
            <div className="space-y-2 max-h-96 overflow-auto">
              {mrpRuns.map((run) => {
                const isSelected = selectedRunId === run.run_id
                return (
                  <button
                    key={run.run_id}
                    onClick={() => setSelectedRunId(run.run_id)}
                    className={`w-full rounded-lg border p-3 text-left transition ${
                      isSelected
                        ? 'border-nikufra bg-nikufra/10'
                        : 'border-border bg-background hover:border-nikufra/50'
                    }`}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-medium text-text-primary text-sm">{run.run_id}</span>
                      <span className="text-xs text-text-muted">
                        {new Date(run.run_timestamp).toLocaleString('pt-PT')}
                      </span>
                    </div>
                    <div className="grid grid-cols-2 gap-2 text-xs">
                      <div>
                        <span className="text-text-muted">Items: </span>
                        <span className="text-text-primary font-semibold">{run.items_processed}</span>
                      </div>
                      <div>
                        <span className="text-text-muted">Demandas: </span>
                        <span className="text-text-primary font-semibold">{run.demands_processed}</span>
                      </div>
                      <div>
                        <span className="text-text-muted">Compras: </span>
                        <span className="text-amber-400 font-semibold">{run.purchase_orders_count}</span>
                      </div>
                      <div>
                        <span className="text-text-muted">Produção: </span>
                        <span className="text-blue-400 font-semibold">{run.manufacture_orders_count}</span>
                      </div>
                    </div>
                    {(run.shortage_alerts_count > 0 || run.capacity_alerts_count > 0) && (
                      <div className="mt-2 flex gap-2">
                        {run.shortage_alerts_count > 0 && (
                          <span className="text-xs px-2 py-1 rounded-full bg-red-500/20 text-red-400">
                            {run.shortage_alerts_count} Faltas
                          </span>
                        )}
                        {run.capacity_alerts_count > 0 && (
                          <span className="text-xs px-2 py-1 rounded-full bg-amber-500/20 text-amber-400">
                            {run.capacity_alerts_count} Capacidade
                          </span>
                        )}
                      </div>
                    )}
                  </button>
                )
              })}
            </div>
          )}
        </div>

        {/* Run Details */}
        <div className="rounded-2xl border border-border bg-surface p-6">
          <h3 className="text-md font-semibold text-text-primary mb-4">Detalhes do Run</h3>

          {!selectedRunId ? (
            <div className="text-center py-12 text-text-muted">
              <Target className="w-12 h-12 mx-auto mb-3 opacity-30" />
              <p>Seleciona um run para ver detalhes</p>
            </div>
          ) : loadingDetails ? (
            <div className="text-center py-8 text-text-muted">A carregar detalhes...</div>
          ) : !runDetails ? (
            <div className="text-center py-8 text-text-muted">Erro ao carregar detalhes</div>
          ) : (
            <div className="space-y-4 max-h-[600px] overflow-y-auto">
              {/* Purchase Orders */}
              {runDetails.purchase_orders.length > 0 && (
                <div>
                  <h4 className="text-sm font-medium text-text-muted mb-2 flex items-center gap-2">
                    <ShoppingCart className="w-4 h-4" /> Ordens de Compra ({runDetails.purchase_orders.length})
                  </h4>
                  <div className="space-y-2">
                    {runDetails.purchase_orders.map((po, i) => (
                      <div key={i} className="rounded-lg border border-amber-500/30 bg-amber-500/5 p-3">
                        <div className="flex items-center justify-between">
                          <div>
                            <span className="font-medium text-text-primary">{po.sku}</span>
                            <span className="ml-2 text-xs text-text-muted">#{po.order_id}</span>
                          </div>
                          <span className="text-amber-400 font-semibold">{po.quantity.toFixed(0)} un</span>
                        </div>
                        <div className="text-xs text-text-muted mt-1">
                          Due: {new Date(po.due_date).toLocaleDateString('pt-PT')} • 
                          Start: {new Date(po.suggested_start_date).toLocaleDateString('pt-PT')} • 
                          Lead: {po.lead_time_days}d
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Manufacture Orders */}
              {runDetails.manufacture_orders.length > 0 && (
                <div>
                  <h4 className="text-sm font-medium text-text-muted mb-2 flex items-center gap-2">
                    <Zap className="w-4 h-4" /> Ordens de Produção ({runDetails.manufacture_orders.length})
                  </h4>
                  <div className="space-y-2">
                    {runDetails.manufacture_orders.map((mo, i) => (
                      <div key={i} className="rounded-lg border border-blue-500/30 bg-blue-500/5 p-3">
                        <div className="flex items-center justify-between">
                          <div>
                            <span className="font-medium text-text-primary">{mo.sku}</span>
                            <span className="ml-2 text-xs text-text-muted">#{mo.order_id}</span>
                          </div>
                          <span className="text-blue-400 font-semibold">{mo.quantity.toFixed(0)} un</span>
                        </div>
                        <div className="text-xs text-text-muted mt-1">
                          Due: {new Date(mo.due_date).toLocaleDateString('pt-PT')} • 
                          Start: {new Date(mo.suggested_start_date).toLocaleDateString('pt-PT')}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Shortage Alerts */}
              {runDetails.shortage_alerts.length > 0 && (
                <div>
                  <h4 className="text-sm font-medium text-red-400 mb-2 flex items-center gap-2">
                    <AlertTriangle className="w-4 h-4" /> Faltas Identificadas ({runDetails.shortage_alerts.length})
                  </h4>
                  <div className="space-y-2">
                    {runDetails.shortage_alerts.map((alert, i) => (
                      <div key={i} className="rounded-lg border border-red-500/30 bg-red-500/5 p-3">
                        <div className="flex items-center justify-between">
                          <span className="font-medium text-red-400">{alert.sku}</span>
                          <span className="text-red-400 font-semibold">-{alert.shortage_qty.toFixed(0)} un</span>
                        </div>
                        <div className="text-xs text-text-muted mt-1">
                          {new Date(alert.period_start).toLocaleDateString('pt-PT')} - {new Date(alert.period_end).toLocaleDateString('pt-PT')}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Capacity Alerts */}
              {runDetails.capacity_alerts.length > 0 && (
                <div>
                  <h4 className="text-sm font-medium text-amber-400 mb-2 flex items-center gap-2">
                    <AlertTriangle className="w-4 h-4" /> Alertas de Capacidade ({runDetails.capacity_alerts.length})
                  </h4>
                  <div className="space-y-2">
                    {runDetails.capacity_alerts.map((alert, i) => (
                      <div key={i} className="rounded-lg border border-amber-500/30 bg-amber-500/5 p-3">
                        <div className="flex items-center justify-between">
                          <span className="font-medium text-amber-400">{alert.work_center}</span>
                          <span className="text-amber-400 font-semibold">
                            {alert.required_capacity.toFixed(1)}h / {alert.available_capacity.toFixed(1)}h
                          </span>
                        </div>
                        <div className="text-xs text-text-muted mt-1">
                          {new Date(alert.period_start).toLocaleDateString('pt-PT')} - {new Date(alert.period_end).toLocaleDateString('pt-PT')}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Empty state */}
              {runDetails.purchase_orders.length === 0 && 
               runDetails.manufacture_orders.length === 0 && 
               runDetails.shortage_alerts.length === 0 && 
               runDetails.capacity_alerts.length === 0 && (
                <div className="text-center py-8 text-green-400">
                  ✅ Nenhuma ação necessária!
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </motion.div>
  )
}

// ═══════════════════════════════════════════════════════════════════
// OPERATIONAL DATA PANEL COMPONENT
// ═══════════════════════════════════════════════════════════════════

type ImportStats = {
  orders: { total: number; last_import: string | null; last_file: string | null }
  inventory_moves: { total: number; last_import: string | null; last_file: string | null }
  hr: { total: number; last_import: string | null; last_file: string | null }
  machines: { total: number; last_import: string | null; last_file: string | null }
}

type WIPFlow = {
  order_code: string
  product_code: string
  order_quantity: number
  current_station: string | null
  total_good: number
  total_scrap: number
  completion_percent: number
  movements_count: number
}

type ImportedOrder = {
  id: number
  external_order_code: string
  product_code: string
  quantity: number
  due_date: string | null
  line_or_center: string | null
  imported_at: string | null
  source_file: string
}

const OperationalDataPanel: React.FC = () => {
  const [selectedOrderCode, setSelectedOrderCode] = useState<string | null>(null)

  // Fetch import stats
  const { data: stats, isLoading: loadingStats, refetch: refetchStats } = useQuery({
    queryKey: ['ops-import-stats'],
    queryFn: async () => {
      const res = await fetch(`${API_BASE_URL}/ops-ingestion/stats`)
      if (!res.ok) throw new Error('Erro ao carregar estatísticas')
      return res.json() as Promise<ImportStats>
    },
    staleTime: 30_000,
  })

  // Fetch WIP flows
  const { data: wipData, isLoading: loadingWIP, refetch: refetchWIP } = useQuery({
    queryKey: ['ops-wip-flows'],
    queryFn: async () => {
      const res = await fetch(`${API_BASE_URL}/ops-ingestion/wip-flow?limit=100`)
      if (!res.ok) throw new Error('Erro ao carregar WIP flows')
      const data = await res.json()
      return (data.wip_flows || []) as WIPFlow[]
    },
    staleTime: 30_000,
    refetchInterval: 60_000, // Auto-refresh every minute
  })

  // Fetch imported orders
  const { data: ordersData, isLoading: loadingOrders } = useQuery({
    queryKey: ['ops-imported-orders'],
    queryFn: async () => {
      const res = await fetch(`${API_BASE_URL}/ops-ingestion/orders?limit=100`)
      if (!res.ok) throw new Error('Erro ao carregar ordens')
      const data = await res.json()
      return (data.orders || []) as ImportedOrder[]
    },
    staleTime: 60_000,
  })

  // Fetch detailed WIP for selected order
  const { data: orderWIPDetail, isLoading: loadingDetail } = useQuery({
    queryKey: ['ops-wip-detail', selectedOrderCode],
    queryFn: async () => {
      if (!selectedOrderCode) return null
      const res = await fetch(`${API_BASE_URL}/ops-ingestion/wip-flow/${selectedOrderCode}`)
      if (!res.ok) throw new Error('Erro ao carregar detalhes WIP')
      return res.json()
    },
    enabled: !!selectedOrderCode,
    staleTime: 30_000,
  })

  const wipFlows: WIPFlow[] = wipData || []
  const orders: ImportedOrder[] = ordersData || []

  return (
    <motion.div
      key="operational-data"
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      className="space-y-6"
    >
      {/* Header with Stats */}
      <div className="rounded-2xl border border-border bg-surface p-6">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-lg font-bold text-text-primary flex items-center gap-2">
              <Activity className="w-5 h-5 text-nikufra" />
              Dados Operacionais Importados
            </h3>
            <p className="text-sm text-text-muted mt-1">
              Visualização de ordens, WIP flow e movimentos importados do Excel
            </p>
          </div>
          <button
            onClick={() => {
              refetchStats()
              refetchWIP()
            }}
            className="rounded-lg border border-border p-2 text-text-muted hover:text-text-primary transition"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
        </div>

        {/* Stats Cards */}
        {loadingStats ? (
          <div className="text-center py-4 text-text-muted">A carregar estatísticas...</div>
        ) : stats ? (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="rounded-lg border border-border bg-background p-3">
              <p className="text-xs text-text-muted">Ordens</p>
              <p className="text-xl font-bold text-text-primary">{stats.orders.total}</p>
              {stats.orders.last_file && (
                <p className="text-xs text-text-muted mt-1 truncate">{stats.orders.last_file}</p>
              )}
            </div>
            <div className="rounded-lg border border-border bg-background p-3">
              <p className="text-xs text-text-muted">Movimentos</p>
              <p className="text-xl font-bold text-nikufra">{stats.inventory_moves.total}</p>
              {stats.inventory_moves.last_file && (
                <p className="text-xs text-text-muted mt-1 truncate">{stats.inventory_moves.last_file}</p>
              )}
            </div>
            <div className="rounded-lg border border-border bg-background p-3">
              <p className="text-xs text-text-muted">Colaboradores</p>
              <p className="text-xl font-bold text-text-primary">{stats.hr.total}</p>
              {stats.hr.last_file && (
                <p className="text-xs text-text-muted mt-1 truncate">{stats.hr.last_file}</p>
              )}
            </div>
            <div className="rounded-lg border border-border bg-background p-3">
              <p className="text-xs text-text-muted">Máquinas</p>
              <p className="text-xl font-bold text-text-primary">{stats.machines.total}</p>
              {stats.machines.last_file && (
                <p className="text-xs text-text-muted mt-1 truncate">{stats.machines.last_file}</p>
              )}
            </div>
          </div>
        ) : null}
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        {/* WIP Flows List */}
        <div className="rounded-2xl border border-border bg-surface p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-md font-semibold text-text-primary">WIP Flow - Ordens em Produção</h3>
            <span className="text-xs text-text-muted">{wipFlows.length} ordens</span>
          </div>

          {loadingWIP ? (
            <div className="text-center py-8 text-text-muted">A carregar...</div>
          ) : wipFlows.length === 0 ? (
            <div className="text-center py-8 text-text-muted">
              <Package className="w-12 h-12 mx-auto mb-3 opacity-30" />
              <p>Nenhuma ordem com movimentos WIP</p>
              <p className="text-xs mt-1">Importe movimentos de inventário para ver WIP flow</p>
            </div>
          ) : (
            <div className="space-y-2 max-h-96 overflow-auto">
              {wipFlows.map((wip) => {
                const isSelected = selectedOrderCode === wip.order_code
                return (
                  <button
                    key={wip.order_code}
                    onClick={() => setSelectedOrderCode(wip.order_code)}
                    className={`w-full rounded-lg border p-3 text-left transition ${
                      isSelected
                        ? 'border-nikufra bg-nikufra/10'
                        : 'border-border bg-background hover:border-nikufra/50'
                    }`}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <div>
                        <span className="font-medium text-text-primary">{wip.order_code}</span>
                        <span className="ml-2 text-xs text-text-muted">{wip.product_code}</span>
                      </div>
                      <span className={`text-xs px-2 py-1 rounded-full font-semibold ${
                        wip.completion_percent >= 90 ? 'bg-green-500/20 text-green-400' :
                        wip.completion_percent >= 50 ? 'bg-amber-500/20 text-amber-400' :
                        'bg-blue-500/20 text-blue-400'
                      }`}>
                        {wip.completion_percent.toFixed(0)}%
                      </span>
                    </div>
                    
                    {wip.current_station ? (
                      <div className="flex items-center gap-2 text-xs text-text-muted mb-1">
                        <Activity className="w-3 h-3" />
                        <span>Estação: <span className="font-semibold text-text-primary">{wip.current_station}</span></span>
                      </div>
                    ) : (
                      <div className="text-xs text-amber-400 mb-1">Sem movimentos registados</div>
                    )}
                    
                    <div className="flex items-center gap-4 text-xs">
                      <div>
                        <span className="text-text-muted">Boa: </span>
                        <span className="text-green-400 font-semibold">{wip.total_good.toFixed(0)}</span>
                      </div>
                      {wip.total_scrap > 0 && (
                        <div>
                          <span className="text-text-muted">Refugo: </span>
                          <span className="text-red-400 font-semibold">{wip.total_scrap.toFixed(0)}</span>
                        </div>
                      )}
                      <div>
                        <span className="text-text-muted">Total: </span>
                        <span className="text-text-primary font-semibold">{wip.order_quantity.toFixed(0)}</span>
                      </div>
                    </div>
                    
                    {/* Progress bar */}
                    <div className="mt-2 h-2 bg-background rounded-full overflow-hidden">
                      <div
                        className={`h-full transition-all ${
                          wip.completion_percent >= 90 ? 'bg-green-500' :
                          wip.completion_percent >= 50 ? 'bg-amber-500' :
                          'bg-blue-500'
                        }`}
                        style={{ width: `${Math.min(100, wip.completion_percent)}%` }}
                      />
                    </div>
                  </button>
                )
              })}
            </div>
          )}
        </div>

        {/* Order Details / WIP Detail */}
        <div className="rounded-2xl border border-border bg-surface p-6">
          <h3 className="text-md font-semibold text-text-primary mb-4">
            {selectedOrderCode ? `Detalhes: ${selectedOrderCode}` : 'Detalhes da Ordem'}
          </h3>

          {!selectedOrderCode ? (
            <div className="text-center py-12 text-text-muted">
              <Target className="w-12 h-12 mx-auto mb-3 opacity-30" />
              <p>Seleciona uma ordem para ver detalhes</p>
            </div>
          ) : loadingDetail ? (
            <div className="text-center py-8 text-text-muted">A carregar detalhes...</div>
          ) : !orderWIPDetail ? (
            <div className="text-center py-8 text-text-muted">Erro ao carregar detalhes</div>
          ) : (
            <div className="space-y-4 max-h-[600px] overflow-y-auto">
              {/* Summary */}
              <div className="rounded-lg border border-border bg-background p-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-xs text-text-muted">Produto</p>
                    <p className="text-sm font-semibold text-text-primary">{orderWIPDetail.product_code}</p>
                  </div>
                  <div>
                    <p className="text-xs text-text-muted">Quantidade Total</p>
                    <p className="text-sm font-semibold text-text-primary">{orderWIPDetail.order_quantity}</p>
                  </div>
                  <div>
                    <p className="text-xs text-text-muted">Estação Atual</p>
                    <p className="text-sm font-semibold text-nikufra">
                      {orderWIPDetail.current_station || 'N/A'}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-text-muted">Conclusão</p>
                    <p className="text-sm font-semibold text-nikufra">
                      {orderWIPDetail.completion_percent.toFixed(1)}%
                    </p>
                  </div>
                </div>
              </div>

              {/* Quantities */}
              <div className="grid grid-cols-3 gap-3">
                <div className="rounded-lg border border-green-500/30 bg-green-500/5 p-3 text-center">
                  <p className="text-xs text-green-400">Boa</p>
                  <p className="text-xl font-bold text-green-400">{orderWIPDetail.total_good.toFixed(0)}</p>
                </div>
                <div className="rounded-lg border border-red-500/30 bg-red-500/5 p-3 text-center">
                  <p className="text-xs text-red-400">Refugo</p>
                  <p className="text-xl font-bold text-red-400">{orderWIPDetail.total_scrap.toFixed(0)}</p>
                </div>
                <div className="rounded-lg border border-border bg-background p-3 text-center">
                  <p className="text-xs text-text-muted">Movimentos</p>
                  <p className="text-xl font-bold text-text-primary">{orderWIPDetail.movements_count}</p>
                </div>
              </div>

              {/* Movements Timeline */}
              {orderWIPDetail.movements && orderWIPDetail.movements.length > 0 && (
                <div>
                  <h4 className="text-sm font-medium text-text-muted mb-2 flex items-center gap-2">
                    <ArrowRightLeft className="w-4 h-4" /> Histórico de Movimentos ({orderWIPDetail.movements.length})
                  </h4>
                  <div className="space-y-2">
                    {orderWIPDetail.movements.map((move: any, i: number) => (
                      <div key={i} className="rounded-lg border border-border bg-background p-3">
                        <div className="flex items-center justify-between mb-1">
                          <div className="flex items-center gap-2">
                            <span className="text-xs text-text-muted">#{i + 1}</span>
                            <span className={`text-xs px-2 py-1 rounded ${
                              move.movement_type === 'GOOD_OUTPUT' ? 'bg-green-500/20 text-green-400' :
                              move.movement_type === 'SCRAP_OUTPUT' ? 'bg-red-500/20 text-red-400' :
                              move.movement_type === 'TRANSFER' ? 'bg-blue-500/20 text-blue-400' :
                              'bg-amber-500/20 text-amber-400'
                            }`}>
                              {move.movement_type}
                            </span>
                          </div>
                          <span className="text-xs text-text-muted">
                            {move.timestamp ? new Date(move.timestamp).toLocaleString('pt-PT') : 'N/A'}
                          </span>
                        </div>
                        <div className="flex items-center gap-2 text-xs text-text-muted">
                          {move.from_station && (
                            <>
                              <span>{move.from_station}</span>
                              <ArrowRightLeft className="w-3 h-3" />
                            </>
                          )}
                          <span className="font-semibold text-text-primary">{move.to_station || 'N/A'}</span>
                          {move.quantity_good > 0 && (
                            <span className="ml-auto text-green-400">+{move.quantity_good} boa</span>
                          )}
                          {move.quantity_scrap > 0 && (
                            <span className="text-red-400">+{move.quantity_scrap} refugo</span>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Imported Orders Table */}
      <div className="rounded-2xl border border-border bg-surface p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-md font-semibold text-text-primary">Ordens Importadas</h3>
          <span className="text-xs text-text-muted">{orders.length} ordens</span>
        </div>

        {loadingOrders ? (
          <div className="text-center py-8 text-text-muted">A carregar...</div>
        ) : orders.length === 0 ? (
          <div className="text-center py-8 text-text-muted">
            <Package className="w-12 h-12 mx-auto mb-3 opacity-30" />
            <p>Nenhuma ordem importada</p>
            <p className="text-xs mt-1">Use o botão "Carregar Dados" para importar ordens do Excel</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left py-3 px-2 text-text-muted">Código</th>
                  <th className="text-left py-3 px-2 text-text-muted">Produto</th>
                  <th className="text-right py-3 px-2 text-text-muted">Quantidade</th>
                  <th className="text-left py-3 px-2 text-text-muted">Data Entrega</th>
                  <th className="text-left py-3 px-2 text-text-muted">Linha/Centro</th>
                  <th className="text-left py-3 px-2 text-text-muted">Ficheiro</th>
                  <th className="text-left py-3 px-2 text-text-muted">Importado</th>
                </tr>
              </thead>
              <tbody>
                {orders.slice(0, 50).map((order) => (
                  <tr
                    key={order.id}
                    className="border-b border-border/30 hover:bg-background/50 cursor-pointer"
                    onClick={() => setSelectedOrderCode(order.external_order_code)}
                  >
                    <td className="py-3 px-2 text-text-primary font-medium">{order.external_order_code}</td>
                    <td className="py-3 px-2 text-text-muted">{order.product_code}</td>
                    <td className="py-3 px-2 text-right text-text-primary">{order.quantity.toLocaleString('pt-PT')}</td>
                    <td className="py-3 px-2 text-text-muted">
                      {order.due_date ? new Date(order.due_date).toLocaleDateString('pt-PT') : 'N/A'}
                    </td>
                    <td className="py-3 px-2 text-text-muted">{order.line_or_center || 'N/A'}</td>
                    <td className="py-3 px-2 text-xs text-text-muted truncate max-w-xs">{order.source_file}</td>
                    <td className="py-3 px-2 text-xs text-text-muted">
                      {order.imported_at ? new Date(order.imported_at).toLocaleDateString('pt-PT') : 'N/A'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {orders.length > 50 && (
              <p className="text-xs text-text-muted text-center py-2">
                Mostrando 50 de {orders.length} ordens
              </p>
            )}
          </div>
        )}
      </div>
    </motion.div>
  )
}

// ═══════════════════════════════════════════════════════════════════
// MRP PARAMETERS PANEL COMPONENT
// ═══════════════════════════════════════════════════════════════════

type MRPParameter = {
  sku_id: string
  name: string
  min_stock: number
  max_stock: number
  reorder_min_qty: number
  reorder_multiple: number
  scrap_rate: number
  lead_time_days: number
}

const MRPParametersPanel: React.FC = () => {
  const [editingParam, setEditingParam] = useState<string | null>(null)
  const [editedValues, setEditedValues] = useState<Partial<MRPParameter>>({})
  const [searchTerm, setSearchTerm] = useState('')

  // Fetch MRP parameters
  const { data: paramsData, isLoading: loadingParams, refetch: refetchParams } = useQuery({
    queryKey: ['mrp-parameters'],
    queryFn: async () => {
      const res = await fetch(`${API_BASE_URL}/inventory/mrp/parameters`)
      if (!res.ok) throw new Error('Erro ao carregar parâmetros MRP')
      return res.json() as Promise<{ parameters: MRPParameter[]; note?: string }>
    },
    staleTime: 60_000,
  })

  const parameters: MRPParameter[] = paramsData?.parameters || []
  const filteredParams = parameters.filter(p => 
    p.sku_id.toLowerCase().includes(searchTerm.toLowerCase()) ||
    p.name.toLowerCase().includes(searchTerm.toLowerCase())
  )

  const handleEdit = (param: MRPParameter) => {
    setEditingParam(param.sku_id)
    setEditedValues({ ...param })
  }

  const handleSave = async (skuId: string) => {
    // TODO: Implementar endpoint POST para salvar parâmetros
    // Por agora apenas simula
    console.log('Saving parameters for', skuId, editedValues)
    setEditingParam(null)
    setEditedValues({})
    // refetchParams()
  }

  const handleCancel = () => {
    setEditingParam(null)
    setEditedValues({})
  }

  return (
    <motion.div
      key="mrp-parameters"
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      className="space-y-6"
    >
      {/* Header */}
      <div className="rounded-2xl border border-border bg-surface p-6">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-lg font-bold text-text-primary flex items-center gap-2">
              <Settings className="w-5 h-5 text-nikufra" />
              Parâmetros MRP
            </h3>
            <p className="text-sm text-text-muted mt-1">
              Configure parâmetros de planeamento de necessidades de material por SKU
            </p>
          </div>
          <div className="flex items-center gap-3">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-text-muted" />
              <input
                type="text"
                placeholder="Pesquisar SKU..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-10 pr-4 py-2 rounded-lg border border-border bg-background text-text-primary placeholder-text-muted focus:outline-none focus:border-nikufra"
              />
            </div>
            <button
              onClick={() => refetchParams()}
              className="rounded-lg border border-border p-2 text-text-muted hover:text-text-primary transition"
            >
              <RefreshCw className="w-4 h-4" />
            </button>
          </div>
        </div>

        {paramsData?.note && (
          <div className="rounded-lg border border-amber-500/30 bg-amber-500/5 p-3 mb-4">
            <p className="text-xs text-amber-400">{paramsData.note}</p>
          </div>
        )}
      </div>

      {/* Parameters Table */}
      {loadingParams ? (
        <div className="text-center py-12 text-text-muted">A carregar parâmetros...</div>
      ) : filteredParams.length === 0 ? (
        <div className="text-center py-12 text-text-muted">
          <Package className="w-12 h-12 mx-auto mb-3 opacity-30" />
          <p>Nenhum parâmetro encontrado</p>
        </div>
      ) : (
        <div className="rounded-2xl border border-border bg-surface p-6">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left py-3 px-2 text-text-muted">SKU</th>
                  <th className="text-left py-3 px-2 text-text-muted">Nome</th>
                  <th className="text-right py-3 px-2 text-text-muted">Stock Mín</th>
                  <th className="text-right py-3 px-2 text-text-muted">Stock Máx</th>
                  <th className="text-right py-3 px-2 text-text-muted">MOQ</th>
                  <th className="text-right py-3 px-2 text-text-muted">Múltiplo</th>
                  <th className="text-right py-3 px-2 text-text-muted">Refugo %</th>
                  <th className="text-right py-3 px-2 text-text-muted">Lead Time (dias)</th>
                  <th className="text-center py-3 px-2 text-text-muted">Ações</th>
                </tr>
              </thead>
              <tbody>
                {filteredParams.map((param) => {
                  const isEditing = editingParam === param.sku_id
                  const values = isEditing ? { ...param, ...editedValues } : param
                  
                  return (
                    <tr key={param.sku_id} className="border-b border-border/30 hover:bg-background/50">
                      <td className="py-3 px-2 text-text-primary font-medium">{param.sku_id}</td>
                      <td className="py-3 px-2 text-text-muted">{param.name}</td>
                      <td className="py-3 px-2 text-right">
                        {isEditing ? (
                          <input
                            type="number"
                            value={values.min_stock}
                            onChange={(e) => setEditedValues({ ...editedValues, min_stock: parseFloat(e.target.value) || 0 })}
                            className="w-20 px-2 py-1 rounded border border-border bg-background text-text-primary text-right"
                            min="0"
                          />
                        ) : (
                          <span className="text-text-primary">{values.min_stock.toLocaleString('pt-PT')}</span>
                        )}
                      </td>
                      <td className="py-3 px-2 text-right">
                        {isEditing ? (
                          <input
                            type="number"
                            value={values.max_stock}
                            onChange={(e) => setEditedValues({ ...editedValues, max_stock: parseFloat(e.target.value) || 0 })}
                            className="w-20 px-2 py-1 rounded border border-border bg-background text-text-primary text-right"
                            min="0"
                          />
                        ) : (
                          <span className="text-text-primary">{values.max_stock === Infinity ? '∞' : values.max_stock.toLocaleString('pt-PT')}</span>
                        )}
                      </td>
                      <td className="py-3 px-2 text-right">
                        {isEditing ? (
                          <input
                            type="number"
                            value={values.reorder_min_qty}
                            onChange={(e) => setEditedValues({ ...editedValues, reorder_min_qty: parseFloat(e.target.value) || 0 })}
                            className="w-20 px-2 py-1 rounded border border-border bg-background text-text-primary text-right"
                            min="0"
                            step="0.01"
                          />
                        ) : (
                          <span className="text-text-primary">{values.reorder_min_qty.toLocaleString('pt-PT')}</span>
                        )}
                      </td>
                      <td className="py-3 px-2 text-right">
                        {isEditing ? (
                          <input
                            type="number"
                            value={values.reorder_multiple}
                            onChange={(e) => setEditedValues({ ...editedValues, reorder_multiple: parseFloat(e.target.value) || 1 })}
                            className="w-20 px-2 py-1 rounded border border-border bg-background text-text-primary text-right"
                            min="1"
                            step="0.01"
                          />
                        ) : (
                          <span className="text-text-primary">{values.reorder_multiple.toLocaleString('pt-PT')}</span>
                        )}
                      </td>
                      <td className="py-3 px-2 text-right">
                        {isEditing ? (
                          <input
                            type="number"
                            value={(values.scrap_rate * 100).toFixed(2)}
                            onChange={(e) => setEditedValues({ ...editedValues, scrap_rate: (parseFloat(e.target.value) || 0) / 100 })}
                            className="w-20 px-2 py-1 rounded border border-border bg-background text-text-primary text-right"
                            min="0"
                            max="100"
                            step="0.01"
                          />
                        ) : (
                          <span className="text-text-primary">{(values.scrap_rate * 100).toFixed(2)}%</span>
                        )}
                      </td>
                      <td className="py-3 px-2 text-right">
                        {isEditing ? (
                          <input
                            type="number"
                            value={values.lead_time_days}
                            onChange={(e) => setEditedValues({ ...editedValues, lead_time_days: parseFloat(e.target.value) || 0 })}
                            className="w-20 px-2 py-1 rounded border border-border bg-background text-text-primary text-right"
                            min="0"
                            step="0.1"
                          />
                        ) : (
                          <span className="text-text-primary">{values.lead_time_days.toFixed(1)}</span>
                        )}
                      </td>
                      <td className="py-3 px-2 text-center">
                        {isEditing ? (
                          <div className="flex items-center justify-center gap-2">
                            <button
                              onClick={() => handleSave(param.sku_id)}
                              className="p-1.5 rounded text-green-400 hover:bg-green-500/10 transition"
                              title="Guardar"
                            >
                              <Check className="w-4 h-4" />
                            </button>
                            <button
                              onClick={handleCancel}
                              className="p-1.5 rounded text-red-400 hover:bg-red-500/10 transition"
                              title="Cancelar"
                            >
                              <X className="w-4 h-4" />
                            </button>
                          </div>
                        ) : (
                          <button
                            onClick={() => handleEdit(param)}
                            className="p-1.5 rounded text-text-muted hover:text-nikufra hover:bg-nikufra/10 transition"
                            title="Editar"
                          >
                            <Edit3 className="w-4 h-4" />
                          </button>
                        )}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
          
          <div className="mt-4 text-xs text-text-muted text-center">
            Mostrando {filteredParams.length} de {parameters.length} parâmetros
          </div>
        </div>
      )}
    </motion.div>
  )
}

// ═══════════════════════════════════════════════════════════════════
// BOM EXPLOSION PANEL COMPONENT
// ═══════════════════════════════════════════════════════════════════

type BOMComponent = {
  component_id: string
  qty_required: number
  is_purchased: boolean
  is_manufactured: boolean
  lead_time_days: number
  level: number
}

type BOMExplosionResult = {
  product_id: string
  quantity: number
  components: BOMComponent[]
  total_components: number
}

const BOMExplosionPanel: React.FC = () => {
  const [productId, setProductId] = useState('')
  const [quantity, setQuantity] = useState(1)
  const [explosionResult, setExplosionResult] = useState<BOMExplosionResult | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [expandedLevels, setExpandedLevels] = useState<Set<number>>(new Set([0]))

  const handleExplode = async () => {
    if (!productId.trim()) return
    
    setIsLoading(true)
    try {
      const res = await fetch(`${API_BASE_URL}/inventory/mrp/bom/${productId}?quantity=${quantity}`)
      if (!res.ok) throw new Error('Erro ao explodir BOM')
      const data = await res.json() as BOMExplosionResult
      setExplosionResult(data)
      // Expand all levels by default
      const levels = new Set(data.components.map(c => c.level))
      setExpandedLevels(levels)
    } catch (error) {
      console.error('BOM explosion error:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const toggleLevel = (level: number) => {
    const newExpanded = new Set(expandedLevels)
    if (newExpanded.has(level)) {
      newExpanded.delete(level)
    } else {
      newExpanded.add(level)
    }
    setExpandedLevels(newExpanded)
  }

  const renderComponentTree = (components: BOMComponent[], level: number = 0) => {
    const levelComponents = components.filter(c => c.level === level)
    if (levelComponents.length === 0) return null

    return (
      <div className="ml-4">
        {levelComponents.map((comp, idx) => {
          const hasChildren = components.some(c => c.level === level + 1)
          const isExpanded = expandedLevels.has(level + 1)
          
          return (
            <div key={`${comp.component_id}-${idx}`} className="mb-2">
              <div className="flex items-center gap-2 p-2 rounded-lg border border-border bg-background hover:bg-surface transition">
                <div className="flex items-center gap-2 flex-1">
                  {hasChildren && (
                    <button
                      onClick={() => toggleLevel(level + 1)}
                      className="p-1 rounded hover:bg-nikufra/10 transition"
                    >
                      {isExpanded ? (
                        <ChevronDown className="w-4 h-4 text-text-muted" />
                      ) : (
                        <ChevronRight className="w-4 h-4 text-text-muted" />
                      )}
                    </button>
                  )}
                  {!hasChildren && <div className="w-6" />}
                  
                  <span className="font-medium text-text-primary">{comp.component_id}</span>
                  <span className="text-text-muted">× {comp.qty_required.toFixed(3)}</span>
                  
                  <div className="flex items-center gap-2 ml-auto">
                    {comp.is_purchased && (
                      <span className="text-xs px-2 py-1 rounded bg-blue-500/20 text-blue-400">Comprado</span>
                    )}
                    {comp.is_manufactured && (
                      <span className="text-xs px-2 py-1 rounded bg-green-500/20 text-green-400">Fabricado</span>
                    )}
                    <span className="text-xs text-text-muted">
                      Lead: {comp.lead_time_days.toFixed(1)}d
                    </span>
                    <span className="text-xs text-text-muted">
                      Nível {comp.level}
                    </span>
                  </div>
                </div>
              </div>
              
              {hasChildren && isExpanded && renderComponentTree(components, level + 1)}
            </div>
          )
        })}
      </div>
    )
  }

  return (
    <motion.div
      key="bom-explosion"
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      className="space-y-6"
    >
      {/* Header */}
      <div className="rounded-2xl border border-border bg-surface p-6">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-lg font-bold text-text-primary flex items-center gap-2">
              <Layers className="w-5 h-5 text-nikufra" />
              BOM Explosion Viewer
            </h3>
            <p className="text-sm text-text-muted mt-1">
              Visualize a estrutura hierárquica de componentes de um produto
            </p>
          </div>
        </div>

        {/* Input Form */}
        <div className="flex items-center gap-4">
          <div className="flex-1">
            <label className="block text-xs text-text-muted mb-1">Código do Produto</label>
            <input
              type="text"
              value={productId}
              onChange={(e) => setProductId(e.target.value)}
              placeholder="Ex: PROD-001"
              className="w-full px-4 py-2 rounded-lg border border-border bg-background text-text-primary placeholder-text-muted focus:outline-none focus:border-nikufra"
              onKeyPress={(e) => e.key === 'Enter' && handleExplode()}
            />
          </div>
          <div className="w-32">
            <label className="block text-xs text-text-muted mb-1">Quantidade</label>
            <input
              type="number"
              value={quantity}
              onChange={(e) => setQuantity(Math.max(0.001, parseFloat(e.target.value) || 1))}
              min="0.001"
              step="0.1"
              className="w-full px-4 py-2 rounded-lg border border-border bg-background text-text-primary focus:outline-none focus:border-nikufra"
            />
          </div>
          <div className="pt-6">
            <button
              onClick={handleExplode}
              disabled={!productId.trim() || isLoading}
              className="px-6 py-2 rounded-lg bg-nikufra text-white font-medium hover:bg-nikufra/90 transition disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
            >
              {isLoading ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  A explodir...
                </>
              ) : (
                <>
                  <Layers className="w-4 h-4" />
                  Explodir BOM
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Results */}
      {explosionResult && (
        <div className="rounded-2xl border border-border bg-surface p-6">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h4 className="text-md font-semibold text-text-primary">
                {explosionResult.product_id} - Quantidade: {explosionResult.quantity}
              </h4>
              <p className="text-sm text-text-muted">
                {explosionResult.total_components} componentes encontrados
              </p>
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={() => {
                  const allLevels = new Set(explosionResult.components.map(c => c.level))
                  setExpandedLevels(allLevels)
                }}
                className="text-xs px-3 py-1 rounded border border-border text-text-muted hover:text-text-primary transition"
              >
                Expandir Tudo
              </button>
              <button
                onClick={() => setExpandedLevels(new Set([0]))}
                className="text-xs px-3 py-1 rounded border border-border text-text-muted hover:text-text-primary transition"
              >
                Colapsar Tudo
              </button>
            </div>
          </div>

          {/* Component Tree */}
          <div className="space-y-2">
            {renderComponentTree(explosionResult.components, 0)}
          </div>

          {/* Summary */}
          <div className="mt-6 pt-6 border-t border-border">
            <div className="grid grid-cols-3 gap-4">
              <div className="rounded-lg border border-border bg-background p-3">
                <p className="text-xs text-text-muted">Total Componentes</p>
                <p className="text-xl font-bold text-text-primary">{explosionResult.total_components}</p>
              </div>
              <div className="rounded-lg border border-border bg-background p-3">
                <p className="text-xs text-text-muted">Componentes Comprados</p>
                <p className="text-xl font-bold text-blue-400">
                  {explosionResult.components.filter(c => c.is_purchased).length}
                </p>
              </div>
              <div className="rounded-lg border border-border bg-background p-3">
                <p className="text-xs text-text-muted">Componentes Fabricados</p>
                <p className="text-xl font-bold text-green-400">
                  {explosionResult.components.filter(c => c.is_manufactured).length}
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {!explosionResult && !isLoading && (
        <div className="text-center py-12 text-text-muted">
          <Layers className="w-12 h-12 mx-auto mb-3 opacity-30" />
          <p>Introduza um código de produto e quantidade para explodir a BOM</p>
        </div>
      )}
    </motion.div>
  )
}

export default SmartInventory
