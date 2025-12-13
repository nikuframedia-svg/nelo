/**
 * KPI Dashboard Component
 * 
 * Displays real industrial KPIs fetched from /plan/kpis
 * All labels in PT-PT
 */

import { type FC } from 'react'
import { useQuery } from '@tanstack/react-query'
import { motion } from 'framer-motion'
import Skeleton from 'react-loading-skeleton'
import { apiGetPlanKPIs, type PlanKPIs, type MachineLoad } from '../services/nikufraApi'

// -------------------------
// Sub-components
// -------------------------

const KPICard: FC<{
  title: string
  value: string | number
  subtitle?: string
  icon?: string
  variant?: 'default' | 'success' | 'warning' | 'danger'
  tooltip?: string
}> = ({ title, value, subtitle, icon, variant = 'default', tooltip }) => {
  const variantStyles = {
    default: 'border-border/60',
    success: 'border-green-500/40 bg-green-500/5',
    warning: 'border-amber-500/40 bg-amber-500/5',
    danger: 'border-red-500/40 bg-red-500/5',
  }

  const valueStyles = {
    default: 'text-text-primary',
    success: 'text-green-400',
    warning: 'text-amber-400',
    danger: 'text-red-400',
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className={`rounded-xl border-2 bg-gradient-to-br from-surface to-surface/80 p-4 shadow-lg ${variantStyles[variant]}`}
      title={tooltip}
    >
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm font-semibold text-text-muted">{title}</span>
        {icon && <span className="text-lg">{icon}</span>}
      </div>
      <div className={`text-2xl font-bold ${valueStyles[variant]}`}>{value}</div>
      {subtitle && <div className="text-xs text-text-muted mt-1">{subtitle}</div>}
    </motion.div>
  )
}

const RouteDistributionBar: FC<{ distribution: Record<string, number> }> = ({
  distribution,
}) => {
  const total = Object.values(distribution).reduce((a, b) => a + b, 0)
  if (total === 0) return null

  const colors: Record<string, string> = {
    A: 'bg-blue-500',
    B: 'bg-amber-500',
    C: 'bg-green-500',
    D: 'bg-purple-500',
  }

  return (
    <div className="space-y-2">
      <div className="flex h-4 rounded-full overflow-hidden bg-background">
        {Object.entries(distribution).map(([route, count]) => {
          const pct = (count / total) * 100
          return (
            <div
              key={route}
              className={`${colors[route] || 'bg-gray-500'} transition-all`}
              style={{ width: `${pct}%` }}
              title={`Rota ${route}: ${count} ops (${pct.toFixed(1)}%)`}
            />
          )
        })}
      </div>
      <div className="flex flex-wrap gap-3 text-xs">
        {Object.entries(distribution).map(([route, count]) => {
          const pct = (count / total) * 100
          return (
            <div key={route} className="flex items-center gap-1">
              <div className={`w-2 h-2 rounded ${colors[route] || 'bg-gray-500'}`} />
              <span className="text-text-muted">
                Rota {route}: <span className="font-semibold text-text-primary">{pct.toFixed(0)}%</span>
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )
}

const MachineLoadList: FC<{ loads: MachineLoad[]; maxItems?: number }> = ({
  loads,
  maxItems = 5,
}) => {
  const topLoads = loads.slice(0, maxItems)

  return (
    <div className="space-y-2">
      {topLoads.map((load, idx) => (
        <div
          key={load.machine_id}
          className="flex items-center gap-3 rounded-lg bg-background/50 px-3 py-2"
        >
          <span className="text-xs font-bold text-text-muted w-4">{idx + 1}</span>
          <div className="flex-1">
            <div className="flex items-center justify-between">
              <span className="text-sm font-semibold text-text-primary">
                {load.machine_id}
              </span>
              <span className="text-xs text-text-muted">
                {load.num_operations} ops
              </span>
            </div>
            <div className="flex items-center gap-2 mt-1">
              <div className="flex-1 h-1.5 rounded-full bg-border overflow-hidden">
                <div
                  className={`h-full rounded-full ${
                    load.utilization_pct > 90
                      ? 'bg-red-500'
                      : load.utilization_pct > 70
                      ? 'bg-amber-500'
                      : 'bg-nikufra'
                  }`}
                  style={{ width: `${Math.min(load.utilization_pct, 100)}%` }}
                />
              </div>
              <span className="text-xs font-semibold text-text-muted w-12 text-right">
                {load.utilization_pct.toFixed(0)}%
              </span>
            </div>
          </div>
        </div>
      ))}
    </div>
  )
}

const OverlapWarning: FC<{ total: number; byMachine: Record<string, number> }> = ({
  total,
  byMachine,
}) => {
  if (total === 0) return null

  const machines = Object.entries(byMachine)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 3)

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className="rounded-xl border-2 border-amber-500/40 bg-amber-500/10 p-4"
    >
      <div className="flex items-center gap-2 mb-2">
        <span className="text-lg">⚠️</span>
        <span className="text-sm font-semibold text-amber-400">
          {total} Sobreposições Detetadas
        </span>
      </div>
      {machines.length > 0 && (
        <div className="text-xs text-text-muted space-y-1">
          {machines.map(([machine, count]) => (
            <div key={machine}>
              {machine}: {count} overlap{count > 1 ? 's' : ''}
            </div>
          ))}
        </div>
      )}
    </motion.div>
  )
}

// -------------------------
// Main Component
// -------------------------

export const KPIDashboard: FC = () => {
  const { data: kpis, isLoading, error } = useQuery({
    queryKey: ['plan-kpis'],
    queryFn: apiGetPlanKPIs,
    staleTime: 30_000,
    refetchInterval: 60_000, // Refresh every minute
  })

  if (isLoading) {
    return (
      <div className="space-y-6">
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          {Array.from({ length: 8 }).map((_, i) => (
            <Skeleton key={i} height={100} baseColor="#121212" highlightColor="#1c1c1c" />
          ))}
        </div>
      </div>
    )
  }

  if (error || !kpis) {
    return (
      <div className="rounded-xl border border-red-500/40 bg-red-500/10 p-4 text-sm text-red-400">
        Erro ao carregar KPIs. Verifique se o plano foi gerado.
      </div>
    )
  }

  // Determine OTD variant
  const otdVariant = kpis.otd_percent >= 95 ? 'success' : kpis.otd_percent >= 85 ? 'warning' : 'danger'

  // Check for alerts
  const alerts: { icon: string; message: string; type: 'danger' | 'warning' }[] = []
  
  if (kpis.otd_percent < 90) {
    alerts.push({ icon: '📅', message: `OTD baixo: ${kpis.otd_percent.toFixed(1)}%`, type: 'danger' })
  }
  
  if (kpis.overlaps.total > 0) {
    alerts.push({ icon: '⚠️', message: `${kpis.overlaps.total} sobreposição(ões) detetada(s)`, type: 'warning' })
  }
  
  if (kpis.active_bottleneck) {
    const bottleneckLoad = kpis.machine_loads.find(m => m.machine_id === kpis.active_bottleneck?.machine_id)
    if (bottleneckLoad && bottleneckLoad.utilization_pct > 80) {
      alerts.push({ icon: '🔴', message: `Gargalo ${kpis.active_bottleneck.machine_id} com ${bottleneckLoad.utilization_pct.toFixed(0)}% de carga`, type: 'danger' })
    }
  }

  return (
    <div className="space-y-6">
      {/* Live Alerts Banner */}
      {alerts.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="rounded-xl border-2 border-danger/40 bg-gradient-to-r from-danger/10 to-warning/10 p-4"
        >
          <div className="flex items-center gap-2 mb-2">
            <span className="text-lg">🚨</span>
            <span className="text-sm font-bold text-danger">Alertas Ativos ({alerts.length})</span>
          </div>
          <div className="flex flex-wrap gap-3">
            {alerts.map((alert, i) => (
              <span
                key={i}
                className={`inline-flex items-center gap-1 rounded-lg px-3 py-1 text-xs font-medium ${
                  alert.type === 'danger' ? 'bg-danger/20 text-danger' : 'bg-warning/20 text-warning'
                }`}
              >
                {alert.icon} {alert.message}
              </span>
            ))}
          </div>
        </motion.div>
      )}

      {/* Row 1: Main KPIs */}
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <KPICard
          title="Makespan"
          value={`${kpis.makespan_hours.toFixed(1)}h`}
          subtitle="Tempo total de produção"
          icon="⏱️"
        />
        <KPICard
          title="OTD (On-Time Delivery)"
          value={`${kpis.otd_percent.toFixed(1)}%`}
          subtitle="Ordens entregues a tempo"
          icon={otdVariant === 'success' ? '✅' : otdVariant === 'warning' ? '⚠️' : '❌'}
          variant={otdVariant}
        />
        <KPICard
          title="Lead Time Médio"
          value={`${kpis.lead_time_average_h.toFixed(1)}h`}
          subtitle="Por artigo"
          icon="📊"
        />
        <KPICard
          title="Horas de Setup"
          value={`${kpis.setup_hours.toFixed(1)}h`}
          subtitle="Estimativa de trocas"
          icon="🔧"
        />
      </div>

      {/* Row 2: Operations & Bottleneck */}
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <KPICard
          title="Total Operações"
          value={kpis.total_operations}
          subtitle={`${kpis.total_orders} ordens • ${kpis.total_articles} artigos`}
          icon="⚙️"
        />
        <KPICard
          title="Máquinas Ativas"
          value={kpis.total_machines}
          subtitle="Com operações planeadas"
          icon="🏭"
        />
        {kpis.active_bottleneck && (
          <KPICard
            title="Gargalo Ativo"
            value={kpis.active_bottleneck.machine_id}
            subtitle={`${(kpis.active_bottleneck.total_minutes / 60).toFixed(1)}h de carga`}
            icon="🔴"
            variant="warning"
          />
        )}
        {kpis.overlaps.total > 0 && (
          <KPICard
            title="Sobreposições"
            value={kpis.overlaps.total}
            subtitle="Conflitos detetados"
            icon="⚠️"
            variant="warning"
          />
        )}
      </div>

      {/* Row 3: Route Distribution & Machine Loads */}
      <div className="grid gap-6 lg:grid-cols-2">
        {/* Route Distribution */}
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="rounded-xl border-2 border-border/60 bg-gradient-to-br from-surface to-surface/80 p-5"
        >
          <h4 className="text-sm font-semibold text-text-primary mb-4">
            Distribuição de Rotas
          </h4>
          <RouteDistributionBar distribution={kpis.route_distribution} />
        </motion.div>

        {/* Top Loaded Machines */}
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="rounded-xl border-2 border-border/60 bg-gradient-to-br from-surface to-surface/80 p-5"
        >
          <h4 className="text-sm font-semibold text-text-primary mb-4">
            Top 5 Máquinas Mais Carregadas
          </h4>
          <MachineLoadList loads={kpis.machine_loads} maxItems={5} />
        </motion.div>
      </div>

      {/* Row 4: Overlap Warning (if any) */}
      {kpis.overlaps.total > 0 && (
        <OverlapWarning
          total={kpis.overlaps.total}
          byMachine={kpis.overlaps.by_machine}
        />
      )}
    </div>
  )
}

export default KPIDashboard

