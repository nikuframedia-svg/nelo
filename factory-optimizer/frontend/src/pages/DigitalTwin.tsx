/**
 * ════════════════════════════════════════════════════════════════════════════════════════════════════
 * DIGITAL TWIN - PdM-IPS (Predictive Maintenance Integrated Production Scheduling)
 * ════════════════════════════════════════════════════════════════════════════════════════════════════
 *
 * Dashboard de Digital Twin para monitorização de saúde das máquinas e previsão de RUL.
 * 
 * Features:
 * - Health Indicators (HI) por máquina
 * - Remaining Useful Life (RUL) estimation
 * - Alertas de manutenção preventiva
 * - Integração com scheduling (penalizações)
 */

import React, { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Activity,
  AlertTriangle,
  CheckCircle,
  Clock,
  Cpu,
  TrendingDown,
  Wrench,
  RefreshCw,
  ChevronRight,
  Heart,
  Zap,
  Shield,
  BarChart3,
} from 'lucide-react'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'

// ═══════════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════════

interface MachineHealth {
  machine_id: string
  current_hi: number
  health_status: 'HEALTHY' | 'DEGRADED' | 'WARNING' | 'CRITICAL'
  rul_hours: number
  rul_std_hours: number
  rul_days: number
  confidence: number
  degradation_rate: number
  history_points: number
}

interface DashboardData {
  overall_health_score: number
  machines_summary: {
    total: number
    critical: number
    warning: number
    degraded: number
    healthy: number
  }
  machines: MachineHealth[]
  penalties: Record<string, number>
  alerts: Array<{
    machine_id: string
    type: string
    message: string
    rul_hours: number
  }>
  kpis: {
    total_machines: number
    critical_count: number
    warning_count: number
    avg_rul_hours: number
    maintenance_recommended: number
  }
}

interface MachineDetail {
  machine_id: string
  current_hi: number
  health_status: string
  rul: {
    mean_hours: number
    std_hours: number
    lower_hours: number
    upper_hours: number
    days: number
  }
  degradation_rate_per_hour: number
  confidence: number
  model_used: string
  history: Array<{ timestamp: string; hi: number }>
  recommendations: Array<{
    type: string
    title: string
    description: string
  }>
}

// ═══════════════════════════════════════════════════════════════════════════════
// API CALLS
// ═══════════════════════════════════════════════════════════════════════════════

async function fetchDashboard(): Promise<DashboardData> {
  const res = await fetch(`${API_BASE_URL}/digital-twin/dashboard`)
  if (!res.ok) throw new Error('Failed to fetch dashboard')
  return res.json()
}

async function fetchMachineDetail(machineId: string): Promise<MachineDetail> {
  const res = await fetch(`${API_BASE_URL}/digital-twin/machine/${machineId}`)
  if (!res.ok) throw new Error('Failed to fetch machine detail')
  return res.json()
}

async function adjustPlan(): Promise<any> {
  const res = await fetch(`${API_BASE_URL}/digital-twin/adjust-plan`, { method: 'POST' })
  if (!res.ok) throw new Error('Failed to adjust plan')
  return res.json()
}

// ═══════════════════════════════════════════════════════════════════════════════
// COMPONENTS
// ═══════════════════════════════════════════════════════════════════════════════

const StatusBadge: React.FC<{ status: string }> = ({ status }) => {
  const config = {
    HEALTHY: { bg: 'bg-emerald-500/20', text: 'text-emerald-400', icon: CheckCircle },
    DEGRADED: { bg: 'bg-amber-500/20', text: 'text-amber-400', icon: TrendingDown },
    WARNING: { bg: 'bg-orange-500/20', text: 'text-orange-400', icon: AlertTriangle },
    CRITICAL: { bg: 'bg-red-500/20', text: 'text-red-400', icon: AlertTriangle },
  }[status] || { bg: 'bg-slate-500/20', text: 'text-slate-400', icon: Activity }

  const Icon = config.icon

  return (
    <span className={`inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium ${config.bg} ${config.text}`}>
      <Icon className="w-3 h-3" />
      {status}
    </span>
  )
}

const HealthGauge: React.FC<{ value: number; size?: 'sm' | 'lg' }> = ({ value, size = 'lg' }) => {
  const percentage = Math.round(value * 100)
  const color = percentage >= 70 ? '#10b981' : percentage >= 50 ? '#f59e0b' : percentage >= 30 ? '#f97316' : '#ef4444'
  const radius = size === 'lg' ? 60 : 30
  const stroke = size === 'lg' ? 8 : 4
  const circumference = 2 * Math.PI * radius
  const offset = circumference - (percentage / 100) * circumference

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
          className="transition-all duration-500"
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className={`${size === 'lg' ? 'text-2xl' : 'text-sm'} font-bold text-white`}>{percentage}%</span>
        {size === 'lg' && <span className="text-xs text-slate-400">Health</span>}
      </div>
    </div>
  )
}

const KPICard: React.FC<{
  icon: React.ReactNode
  label: string
  value: string | number
  subtext?: string
  trend?: 'up' | 'down' | 'neutral'
  color?: string
}> = ({ icon, label, value, subtext, color = 'cyan' }) => (
  <motion.div
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
    className="rounded-xl border border-slate-700/50 bg-slate-800/50 p-4"
  >
    <div className="flex items-start justify-between">
      <div className={`p-2 rounded-lg bg-${color}-500/20`}>
        {icon}
      </div>
    </div>
    <div className="mt-3">
      <p className="text-2xl font-bold text-white">{value}</p>
      <p className="text-xs text-slate-400">{label}</p>
      {subtext && <p className="text-xs text-slate-500 mt-1">{subtext}</p>}
    </div>
  </motion.div>
)

const MachineCard: React.FC<{
  machine: MachineHealth
  onClick: () => void
}> = ({ machine, onClick }) => {
  const hiPercentage = Math.round(machine.current_hi * 100)
  
  return (
    <motion.div
      whileHover={{ scale: 1.02 }}
      onClick={onClick}
      className="cursor-pointer rounded-xl border border-slate-700/50 bg-slate-800/30 p-4 hover:border-cyan-500/50 transition-all"
    >
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Cpu className="w-5 h-5 text-cyan-400" />
          <span className="font-semibold text-white">{machine.machine_id}</span>
        </div>
        <StatusBadge status={machine.health_status} />
      </div>
      
      <div className="grid grid-cols-2 gap-4">
        <div>
          <p className="text-xs text-slate-400 mb-1">Health Index</p>
          <div className="flex items-center gap-2">
            <div className="flex-1 h-2 bg-slate-700 rounded-full overflow-hidden">
              <div
                className={`h-full transition-all ${
                  hiPercentage >= 70 ? 'bg-emerald-500' :
                  hiPercentage >= 50 ? 'bg-amber-500' :
                  hiPercentage >= 30 ? 'bg-orange-500' : 'bg-red-500'
                }`}
                style={{ width: `${hiPercentage}%` }}
              />
            </div>
            <span className="text-sm font-medium text-white">{hiPercentage}%</span>
          </div>
        </div>
        
        <div>
          <p className="text-xs text-slate-400 mb-1">RUL</p>
          <p className="text-lg font-bold text-white">
            {machine.rul_hours < 100 ? `${machine.rul_hours.toFixed(0)}h` : `${machine.rul_days.toFixed(0)}d`}
          </p>
        </div>
      </div>
      
      <div className="flex items-center justify-between mt-3 pt-3 border-t border-slate-700/50">
        <span className="text-xs text-slate-500">Confiança: {(machine.confidence * 100).toFixed(0)}%</span>
        <ChevronRight className="w-4 h-4 text-slate-500" />
      </div>
    </motion.div>
  )
}

const AlertCard: React.FC<{ alert: DashboardData['alerts'][0] }> = ({ alert }) => (
  <div className={`flex items-center gap-3 p-3 rounded-lg ${
    alert.type === 'CRITICAL' ? 'bg-red-500/10 border border-red-500/30' : 'bg-orange-500/10 border border-orange-500/30'
  }`}>
    <AlertTriangle className={`w-5 h-5 ${alert.type === 'CRITICAL' ? 'text-red-400' : 'text-orange-400'}`} />
    <div className="flex-1">
      <p className="text-sm font-medium text-white">{alert.machine_id}</p>
      <p className="text-xs text-slate-400">{alert.message}</p>
    </div>
    <Wrench className="w-4 h-4 text-slate-500" />
  </div>
)

const MachineDetailModal: React.FC<{
  machineId: string
  onClose: () => void
}> = ({ machineId, onClose }) => {
  const { data, isLoading } = useQuery({
    queryKey: ['machine-detail', machineId],
    queryFn: () => fetchMachineDetail(machineId),
  })

  if (isLoading) {
    return (
      <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
        <div className="bg-slate-800 rounded-2xl p-8">
          <RefreshCw className="w-8 h-8 text-cyan-400 animate-spin" />
        </div>
      </div>
    )
  }

  if (!data) return null

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
        className="bg-slate-800 rounded-2xl p-6 max-w-2xl w-full max-h-[90vh] overflow-y-auto"
        onClick={e => e.stopPropagation()}
      >
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <Cpu className="w-8 h-8 text-cyan-400" />
            <div>
              <h2 className="text-xl font-bold text-white">{data.machine_id}</h2>
              <StatusBadge status={data.health_status} />
            </div>
          </div>
          <HealthGauge value={data.current_hi} size="sm" />
        </div>

        <div className="grid grid-cols-2 gap-4 mb-6">
          <div className="bg-slate-900/50 rounded-xl p-4">
            <p className="text-xs text-slate-400 mb-1">RUL Estimado</p>
            <p className="text-2xl font-bold text-white">{data.rul.mean_hours.toFixed(0)}h</p>
            <p className="text-xs text-slate-500">± {data.rul.std_hours.toFixed(0)}h ({data.rul.days.toFixed(0)} dias)</p>
          </div>
          <div className="bg-slate-900/50 rounded-xl p-4">
            <p className="text-xs text-slate-400 mb-1">Confiança</p>
            <p className="text-2xl font-bold text-cyan-400">{(data.confidence * 100).toFixed(0)}%</p>
            <p className="text-xs text-slate-500">Modelo: {data.model_used}</p>
          </div>
        </div>

        {/* Health History Chart */}
        <div className="bg-slate-900/50 rounded-xl p-4 mb-6">
          <p className="text-sm font-medium text-white mb-3">Histórico de Health Index</p>
          <div className="h-32 flex items-end gap-1">
            {data.history.slice(-30).map((point, i) => (
              <div
                key={i}
                className="flex-1 rounded-t transition-all"
                style={{
                  height: `${point.hi * 100}%`,
                  backgroundColor: point.hi >= 0.7 ? '#10b981' : point.hi >= 0.5 ? '#f59e0b' : point.hi >= 0.3 ? '#f97316' : '#ef4444',
                  opacity: 0.5 + (i / 60),
                }}
                title={`HI: ${(point.hi * 100).toFixed(0)}%`}
              />
            ))}
          </div>
          <div className="flex justify-between mt-2">
            <span className="text-xs text-slate-500">30h atrás</span>
            <span className="text-xs text-slate-500">Agora</span>
          </div>
        </div>

        {/* Recommendations */}
        <div className="space-y-3">
          <p className="text-sm font-medium text-white">Recomendações</p>
          {data.recommendations.map((rec, i) => (
            <div
              key={i}
              className={`p-3 rounded-lg ${
                rec.type === 'URGENT' ? 'bg-red-500/10 border border-red-500/30' :
                rec.type === 'WARNING' ? 'bg-orange-500/10 border border-orange-500/30' :
                rec.type === 'INFO' ? 'bg-blue-500/10 border border-blue-500/30' :
                'bg-emerald-500/10 border border-emerald-500/30'
              }`}
            >
              <p className="text-sm font-medium text-white">{rec.title}</p>
              <p className="text-xs text-slate-400 mt-1">{rec.description}</p>
            </div>
          ))}
        </div>

        <button
          onClick={onClose}
          className="w-full mt-6 py-3 bg-slate-700 hover:bg-slate-600 text-white rounded-lg transition-colors"
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

const DigitalTwin: React.FC = () => {
  const queryClient = useQueryClient()
  const [selectedMachine, setSelectedMachine] = useState<string | null>(null)

  const { data, isLoading, refetch } = useQuery({
    queryKey: ['digital-twin-dashboard'],
    queryFn: fetchDashboard,
    refetchInterval: 30000, // Refresh every 30s
  })

  const adjustMutation = useMutation({
    mutationFn: adjustPlan,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['digital-twin-dashboard'] })
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
        <p className="text-slate-400">Módulo Digital Twin não disponível</p>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <header className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-gradient-to-br from-cyan-500/20 to-blue-500/20 rounded-xl border border-cyan-500/30">
            <Heart className="w-6 h-6 text-cyan-400" />
          </div>
          <div>
            <p className="text-xs uppercase tracking-[0.4em] text-slate-500">Digital Twin</p>
            <h2 className="text-2xl font-semibold text-white">Saúde das Máquinas</h2>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={() => refetch()}
            className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-white transition-colors"
          >
            <RefreshCw className="w-5 h-5" />
          </button>
          <button
            onClick={() => adjustMutation.mutate()}
            disabled={adjustMutation.isPending}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-cyan-600 hover:bg-cyan-500 text-white font-medium transition-colors disabled:opacity-50"
          >
            {adjustMutation.isPending ? (
              <RefreshCw className="w-4 h-4 animate-spin" />
            ) : (
              <Zap className="w-4 h-4" />
            )}
            Ajustar Plano
          </button>
        </div>
      </header>

      {/* Overall Health Score */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-1 flex flex-col items-center justify-center p-6 rounded-2xl border border-slate-700/50 bg-gradient-to-br from-slate-800/50 to-slate-900/50">
          <HealthGauge value={data.overall_health_score / 100} />
          <p className="mt-4 text-lg font-medium text-white">Saúde Global da Fábrica</p>
          <p className="text-sm text-slate-400">{data.kpis.total_machines} máquinas monitorizadas</p>
        </div>

        <div className="lg:col-span-2 grid grid-cols-2 md:grid-cols-4 gap-4">
          <KPICard
            icon={<Shield className="w-5 h-5 text-emerald-400" />}
            label="Saudáveis"
            value={data.machines_summary.healthy}
            color="emerald"
          />
          <KPICard
            icon={<TrendingDown className="w-5 h-5 text-amber-400" />}
            label="Degradadas"
            value={data.machines_summary.degraded}
            color="amber"
          />
          <KPICard
            icon={<AlertTriangle className="w-5 h-5 text-orange-400" />}
            label="Em Alerta"
            value={data.machines_summary.warning}
            color="orange"
          />
          <KPICard
            icon={<AlertTriangle className="w-5 h-5 text-red-400" />}
            label="Críticas"
            value={data.machines_summary.critical}
            color="red"
          />
        </div>
      </div>

      {/* Alerts */}
      {data.alerts.length > 0 && (
        <div className="space-y-3">
          <h3 className="text-sm font-medium text-white flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-orange-400" />
            Alertas de Manutenção ({data.alerts.length})
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {data.alerts.map((alert, i) => (
              <AlertCard key={i} alert={alert} />
            ))}
          </div>
        </div>
      )}

      {/* Machines Grid */}
      <div className="space-y-3">
        <h3 className="text-sm font-medium text-white flex items-center gap-2">
          <Cpu className="w-4 h-4 text-cyan-400" />
          Máquinas Monitorizadas
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {data.machines.map(machine => (
            <MachineCard
              key={machine.machine_id}
              machine={machine}
              onClick={() => setSelectedMachine(machine.machine_id)}
            />
          ))}
        </div>
      </div>

      {/* RUL Metrics */}
      <div className="rounded-2xl border border-slate-700/50 bg-slate-800/30 p-6">
        <h3 className="text-sm font-medium text-white mb-4 flex items-center gap-2">
          <BarChart3 className="w-4 h-4 text-cyan-400" />
          Métricas de RUL
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center">
            <p className="text-3xl font-bold text-white">{data.kpis.avg_rul_hours.toFixed(0)}h</p>
            <p className="text-xs text-slate-400">RUL Médio</p>
          </div>
          <div className="text-center">
            <p className="text-3xl font-bold text-orange-400">{data.kpis.maintenance_recommended}</p>
            <p className="text-xs text-slate-400">Manutenções Recomendadas</p>
          </div>
          <div className="text-center">
            <p className="text-3xl font-bold text-cyan-400">
              {Object.values(data.penalties).filter(p => p > 1).length}
            </p>
            <p className="text-xs text-slate-400">Máquinas Penalizadas</p>
          </div>
          <div className="text-center">
            <p className="text-3xl font-bold text-emerald-400">
              {Math.round((data.machines_summary.healthy / data.machines_summary.total) * 100)}%
            </p>
            <p className="text-xs text-slate-400">Taxa de Saúde</p>
          </div>
        </div>
      </div>

      {/* Machine Detail Modal */}
      <AnimatePresence>
        {selectedMachine && (
          <MachineDetailModal
            machineId={selectedMachine}
            onClose={() => setSelectedMachine(null)}
          />
        )}
      </AnimatePresence>
    </div>
  )
}

export default DigitalTwin



