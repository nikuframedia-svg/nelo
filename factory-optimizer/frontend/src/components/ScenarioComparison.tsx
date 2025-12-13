import type { FC } from 'react'

type MetricsComparison = {
  baseline: number
  scenario: number
  delta: number
  delta_pct: number
}

type MachineLoadComparison = {
  machine_id: string
  baseline_load_min: number
  scenario_load_min: number
  delta_min: number
}

export type ScenarioComparisonData = {
  makespan?: MetricsComparison
  total_duration?: MetricsComparison
  bottleneck?: {
    baseline: string
    scenario: string
    changed: boolean
  }
  machine_loads?: MachineLoadComparison[]
  summary?: string
  error?: string
}

interface ScenarioComparisonProps {
  data: ScenarioComparisonData | null
  loading?: boolean
}

export const ScenarioComparison: FC<ScenarioComparisonProps> = ({ data, loading }) => {
  if (loading) {
    return (
      <CardWrapper title="Comparação de cenário">
        <p className="text-sm text-text-muted">A comparar cenários...</p>
      </CardWrapper>
    )
  }

  if (!data) {
    return (
      <CardWrapper title="Comparação de cenário">
        <p className="text-sm text-text-muted">
          Clica em &quot;Comparar cenário&quot; para ver o impacto das alterações face ao baseline.
        </p>
      </CardWrapper>
    )
  }

  return (
    <CardWrapper title="Comparação de cenário">
      {data.error && (
        <p className="mb-4 rounded-xl border border-danger/40 bg-danger/10 px-4 py-2 text-sm text-danger">
          {data.error}
        </p>
      )}

      {/* Summary */}
      {data.summary && (
        <div className="rounded-xl border border-nikufra/40 bg-nikufra/10 px-4 py-3 text-sm text-text-primary">
          {data.summary}
        </div>
      )}

      {/* Makespan Comparison */}
      {data.makespan && (
        <MetricCard
          title="Makespan"
          baseline={`${(data.makespan.baseline / 60).toFixed(1)}h`}
          scenario={`${(data.makespan.scenario / 60).toFixed(1)}h`}
          delta={data.makespan.delta_pct}
        />
      )}

      {/* Total Duration Comparison */}
      {data.total_duration && (
        <MetricCard
          title="Duração Total"
          baseline={`${(data.total_duration.baseline / 60).toFixed(1)}h`}
          scenario={`${(data.total_duration.scenario / 60).toFixed(1)}h`}
          delta={data.total_duration.delta_pct}
        />
      )}

      {/* Bottleneck Comparison */}
      {data.bottleneck && (
        <div className="rounded-xl border border-border/60 bg-background/70 p-4">
          <p className="text-xs font-semibold uppercase tracking-[0.3em] text-text-muted">Gargalo</p>
          <div className="mt-2 grid grid-cols-2 gap-4">
            <div>
              <p className="text-xs text-text-muted">Baseline</p>
              <p className="text-sm font-semibold text-text-primary">{data.bottleneck.baseline}</p>
            </div>
            <div>
              <p className="text-xs text-text-muted">Cenário</p>
              <p className="text-sm font-semibold text-text-primary">{data.bottleneck.scenario}</p>
            </div>
          </div>
          {data.bottleneck.changed && (
            <p className="mt-2 text-xs text-warning">⚠️ O gargalo mudou com este cenário</p>
          )}
        </div>
      )}

      {/* Machine Loads */}
      {data.machine_loads && data.machine_loads.length > 0 && (
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.3em] text-text-muted">Carga por Máquina</p>
          <div className="mt-2 space-y-2">
            {data.machine_loads.slice(0, 5).map((load) => (
              <div
                key={load.machine_id}
                className="flex items-center justify-between rounded-xl border border-border/60 bg-background/70 px-4 py-2 text-sm"
              >
                <span className="font-semibold text-text-primary">{load.machine_id}</span>
                <div className="flex items-center gap-4">
                  <span className="text-text-muted">{(load.baseline_load_min / 60).toFixed(1)}h</span>
                  <span className="text-text-muted">→</span>
                  <span className="text-text-primary">{(load.scenario_load_min / 60).toFixed(1)}h</span>
                  <DeltaBadge delta={(load.delta_min / load.baseline_load_min) * 100} />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </CardWrapper>
  )
}

const CardWrapper: FC<{ title: string; children: React.ReactNode }> = ({ title, children }) => (
  <div className="rounded-2xl border border-border bg-surface p-6 shadow-glow">
    <h3 className="text-sm font-semibold text-text-primary">{title}</h3>
    <div className="mt-4 space-y-4">{children}</div>
  </div>
)

const MetricCard: FC<{ title: string; baseline: string; scenario: string; delta: number }> = ({
  title,
  baseline,
  scenario,
  delta,
}) => (
  <div className="rounded-xl border border-border/60 bg-background/70 p-4">
    <p className="text-xs font-semibold uppercase tracking-[0.3em] text-text-muted">{title}</p>
    <div className="mt-2 flex items-center justify-between">
      <div className="flex items-center gap-4">
        <div>
          <p className="text-xs text-text-muted">Baseline</p>
          <p className="text-lg font-bold text-text-primary">{baseline}</p>
        </div>
        <span className="text-text-muted">→</span>
        <div>
          <p className="text-xs text-text-muted">Cenário</p>
          <p className="text-lg font-bold text-text-primary">{scenario}</p>
        </div>
      </div>
      <DeltaBadge delta={delta} />
    </div>
  </div>
)

const DeltaBadge: FC<{ delta: number }> = ({ delta }) => {
  const isPositive = delta > 0
  const isNegative = delta < 0
  const color = isNegative ? 'text-green-400 bg-green-500/20' : isPositive ? 'text-red-400 bg-red-500/20' : 'text-text-muted bg-background'

  return (
    <span className={`rounded-lg px-2 py-1 text-xs font-semibold ${color}`}>
      {isPositive ? '+' : ''}{delta.toFixed(1)}%
    </span>
  )
}



