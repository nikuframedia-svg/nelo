/**
 * Advanced Production Timeline Component
 * 
 * Features:
 * - View modes: Machine, Product, Route
 * - Time window presets and zoom slider
 * - Color-coded operations by route
 * - Filtering by selected entities
 */

import { useState, useEffect, useMemo, useRef, type FC } from 'react'
import { useQuery } from '@tanstack/react-query'
import { apiGetPlan, apiGetBottleneck, type PlanOp } from '../services/nikufraApi'
import {
  parsePlanOperations,
  computeTimeRange,
  computeTimeWindow,
  getGroupedOperations,
  getRouteColor,
  formatTimeAxisLabel,
  formatDayLabel,
  getUniqueValues,
  filterOperations,
  type ViewMode,
  type ParsedPlanOp,
  type TimeRange,
} from '../utils/timeline'

// -------------------------
// Types
// -------------------------

type TimePreset = '1day' | '3days' | '1week' | 'full'

// -------------------------
// Sub-components
// -------------------------

const ViewModeSelector: FC<{ value: ViewMode; onChange: (v: ViewMode) => void }> = ({
  value,
  onChange,
}) => (
  <div className="flex items-center gap-2">
    <label className="text-xs uppercase tracking-widest text-text-muted">Vista</label>
    <select
      value={value}
      onChange={(e) => onChange(e.target.value as ViewMode)}
      className="rounded-lg border border-border bg-surface px-3 py-1.5 text-sm text-text-primary focus:border-nikufra focus:outline-none"
    >
      <option value="MACHINE">Por Máquina</option>
      <option value="PRODUCT">Por Artigo</option>
      <option value="ROUTE">Por Rota</option>
    </select>
  </div>
)

const TimePresetButtons: FC<{ current: TimePreset; onChange: (p: TimePreset) => void }> = ({
  current,
  onChange,
}) => {
  const presets: { key: TimePreset; label: string }[] = [
    { key: '1day', label: '1 Dia' },
    { key: '3days', label: '3 Dias' },
    { key: '1week', label: '1 Semana' },
    { key: 'full', label: 'Completo' },
  ]

  return (
    <div className="flex items-center gap-1">
      {presets.map((p) => (
        <button
          key={p.key}
          onClick={() => onChange(p.key)}
          className={`rounded-lg px-3 py-1.5 text-xs font-medium transition ${
            current === p.key
              ? 'bg-nikufra text-white'
              : 'border border-border bg-surface text-text-muted hover:border-nikufra hover:text-nikufra'
          }`}
        >
          {p.label}
        </button>
      ))}
    </div>
  )
}

const MultiSelect: FC<{
  label: string
  options: string[]
  selected: string[]
  onChange: (v: string[]) => void
}> = ({ label, options, selected, onChange }) => {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  const toggleOption = (opt: string) => {
    if (selected.includes(opt)) {
      onChange(selected.filter((s) => s !== opt))
    } else {
      onChange([...selected, opt])
    }
  }

  const selectAll = () => onChange(options)
  const clearAll = () => onChange([])

  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-2 rounded-lg border border-border bg-surface px-3 py-1.5 text-sm text-text-muted hover:border-nikufra"
      >
        <span>{label}</span>
        <span className="rounded bg-nikufra/20 px-1.5 py-0.5 text-xs text-nikufra">
          {selected.length === 0 ? 'Todos' : selected.length}
        </span>
      </button>

      {open && (
        <div className="absolute left-0 top-full z-50 mt-1 max-h-60 w-48 overflow-auto rounded-xl border border-border bg-surface p-2 shadow-lg">
          <div className="mb-2 flex gap-1">
            <button
              onClick={selectAll}
              className="flex-1 rounded bg-nikufra/10 px-2 py-1 text-xs text-nikufra hover:bg-nikufra/20"
            >
              Todos
            </button>
            <button
              onClick={clearAll}
              className="flex-1 rounded bg-danger/10 px-2 py-1 text-xs text-danger hover:bg-danger/20"
            >
              Limpar
            </button>
          </div>
          {options.map((opt) => (
            <label
              key={opt}
              className="flex cursor-pointer items-center gap-2 rounded px-2 py-1 text-sm text-text-primary hover:bg-background"
            >
              <input
                type="checkbox"
                checked={selected.includes(opt)}
                onChange={() => toggleOption(opt)}
                className="accent-nikufra"
              />
              {opt}
            </label>
          ))}
        </div>
      )}
    </div>
  )
}

const ZoomSlider: FC<{
  value: number
  onChange: (v: number) => void
}> = ({ value, onChange }) => (
  <div className="flex items-center gap-2">
    <span className="text-xs text-text-muted">Zoom</span>
    <input
      type="range"
      min={10}
      max={100}
      value={value}
      onChange={(e) => onChange(Number(e.target.value))}
      className="h-1 w-24 cursor-pointer appearance-none rounded-lg bg-border accent-nikufra"
    />
    <span className="w-8 text-xs text-text-muted">{value}%</span>
  </div>
)

const OperationTooltip: FC<{ op: ParsedPlanOp; x: number; y: number }> = ({ op, x, y }) => (
  <div
    className="pointer-events-none fixed z-[100] max-w-xs rounded-lg border border-border bg-surface p-3 shadow-xl"
    style={{ left: x + 10, top: y + 10 }}
  >
    <p className="text-sm font-semibold text-text-primary">{op.order_id}</p>
    <div className="mt-1 space-y-0.5 text-xs text-text-muted">
      <p>Artigo: {op.article_id}</p>
      <p>Operação: {op.op_code}</p>
      <p>Máquina: {op.machine_id}</p>
      <p>
        Rota:{' '}
        <span style={{ color: getRouteColor(op.route_label) }} className="font-semibold">
          {op.route_label}
        </span>
      </p>
      <p>Quantidade: {op.qty}</p>
      <p>Duração: {op.duration_min.toFixed(0)} min</p>
      <p className="mt-1 border-t border-border pt-1">
        {op.startDate.toLocaleString('pt-PT')} → {op.endDate.toLocaleString('pt-PT')}
      </p>
    </div>
  </div>
)

const TimeAxis: FC<{ timeWindow: TimeRange; containerWidth: number }> = ({
  timeWindow,
  containerWidth,
}) => {
  const ticks = useMemo(() => {
    const { min, max } = timeWindow
    const duration = max.getTime() - min.getTime()
    const tickInterval = duration > 3 * 24 * 60 * 60 * 1000 ? 12 * 60 * 60 * 1000 : 6 * 60 * 60 * 1000

    const result: { x: number; label: string; isDay: boolean }[] = []
    let current = new Date(min)
    current.setMinutes(0, 0, 0)

    while (current <= max) {
      const x = ((current.getTime() - min.getTime()) / duration) * containerWidth
      const isDay = current.getHours() === 0
      result.push({
        x,
        label: isDay ? formatDayLabel(current) : formatTimeAxisLabel(current),
        isDay,
      })
      current = new Date(current.getTime() + tickInterval)
    }

    return result
  }, [timeWindow, containerWidth])

  return (
    <div className="relative h-8 border-b border-border bg-background/50">
      {ticks.map((tick, i) => (
        <div
          key={i}
          className="absolute top-0 flex h-full flex-col items-center"
          style={{ left: tick.x }}
        >
          <div className={`h-2 w-px ${tick.isDay ? 'bg-nikufra' : 'bg-border'}`} />
          <span
            className={`mt-0.5 whitespace-nowrap text-[10px] ${
              tick.isDay ? 'font-semibold text-nikufra' : 'text-text-muted'
            }`}
          >
            {tick.label}
          </span>
        </div>
      ))}
    </div>
  )
}

const OperationBar: FC<{
  op: ParsedPlanOp
  left: number
  width: number
  onHover: (op: ParsedPlanOp | null, x: number, y: number) => void
}> = ({ op, left, width, onHover }) => {
  const handleMouseEnter = (e: React.MouseEvent) => {
    onHover(op, e.clientX, e.clientY)
  }
  const handleMouseLeave = () => {
    onHover(null, 0, 0)
  }

  return (
    <div
      className="absolute top-1 h-6 cursor-pointer rounded transition-all hover:z-10 hover:scale-y-110 hover:shadow-lg"
      style={{
        left: `${left}px`,
        width: `${Math.max(width, 4)}px`,
        backgroundColor: getRouteColor(op.route_label),
      }}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      onMouseMove={(e) => onHover(op, e.clientX, e.clientY)}
    >
      {width > 40 && (
        <span className="block truncate px-1 text-[10px] font-medium text-white leading-6">
          {op.order_id}
        </span>
      )}
    </div>
  )
}

const Lane: FC<{
  label: string
  operations: ParsedPlanOp[]
  timeWindow: TimeRange
  containerWidth: number
  onHover: (op: ParsedPlanOp | null, x: number, y: number) => void
}> = ({ label, operations, timeWindow, containerWidth, onHover }) => {
  const duration = timeWindow.max.getTime() - timeWindow.min.getTime()

  return (
    <div className="flex border-b border-border/50">
      <div className="flex w-28 shrink-0 items-center border-r border-border bg-surface px-2 py-1">
        <span className="truncate text-xs font-medium text-text-primary" title={label}>
          {label}
        </span>
      </div>
      <div className="relative h-8 flex-1 bg-background/30">
        {operations.map((op, i) => {
          const left = ((op.startDate.getTime() - timeWindow.min.getTime()) / duration) * containerWidth
          const width = ((op.endDate.getTime() - op.startDate.getTime()) / duration) * containerWidth

          return <OperationBar key={`${op.order_id}-${op.op_seq}-${i}`} op={op} left={left} width={width} onHover={onHover} />
        })}
      </div>
    </div>
  )
}

const BottleneckCard: FC<{ machineId: string; totalMinutes: number }> = ({
  machineId,
  totalMinutes,
}) => (
  <div className="rounded-xl border border-warning/40 bg-warning/10 p-3">
    <p className="text-xs uppercase tracking-widest text-warning">Gargalo Atual</p>
    <p className="mt-1 text-lg font-bold text-text-primary">{machineId}</p>
    <p className="text-xs text-text-muted">{(totalMinutes / 60).toFixed(1)} horas de carga</p>
  </div>
)

const RouteLegend: FC = () => (
  <div className="flex flex-wrap items-center gap-3">
    {['A', 'B', 'C', 'D', 'E'].map((r) => (
      <div key={r} className="flex items-center gap-1">
        <div className="h-3 w-3 rounded" style={{ backgroundColor: getRouteColor(r) }} />
        <span className="text-xs text-text-muted">Rota {r}</span>
      </div>
    ))}
  </div>
)

// -------------------------
// Main Component
// -------------------------

export const PlanTimeline: FC = () => {
  const containerRef = useRef<HTMLDivElement>(null)
  const [containerWidth, setContainerWidth] = useState(800)

  // State
  const [viewMode, setViewMode] = useState<ViewMode>('MACHINE')
  const [timePreset, setTimePreset] = useState<TimePreset>('3days')
  const [zoomPercent, setZoomPercent] = useState(100)
  const [selectedFilters, setSelectedFilters] = useState<string[]>([])
  const [hoveredOp, setHoveredOp] = useState<{ op: ParsedPlanOp; x: number; y: number } | null>(null)

  // Queries
  const { data: planData, isLoading: planLoading } = useQuery({
    queryKey: ['plan-timeline'],
    queryFn: apiGetPlan,
    staleTime: 30_000,
  })

  const { data: bottleneck } = useQuery({
    queryKey: ['bottleneck-timeline'],
    queryFn: apiGetBottleneck,
    staleTime: 60_000,
  })

  // Parse operations
  const parsedOps = useMemo(() => {
    if (!planData) return []
    return parsePlanOperations(planData)
  }, [planData])

  // Compute global time range
  const globalRange = useMemo(() => computeTimeRange(parsedOps), [parsedOps])

  // Compute current time window
  const timeWindow = useMemo(() => {
    const baseWindow = computeTimeWindow(globalRange, timePreset)
    // Apply zoom
    const duration = baseWindow.max.getTime() - baseWindow.min.getTime()
    const zoomedDuration = duration * (zoomPercent / 100)
    return {
      min: baseWindow.min,
      max: new Date(baseWindow.min.getTime() + zoomedDuration),
    }
  }, [globalRange, timePreset, zoomPercent])

  // Filter options based on view mode
  const filterOptions = useMemo(() => {
    switch (viewMode) {
      case 'MACHINE':
        return getUniqueValues(parsedOps, 'machine_id')
      case 'PRODUCT':
        return getUniqueValues(parsedOps, 'article_id')
      case 'ROUTE':
        return getUniqueValues(parsedOps, 'route_label')
    }
  }, [parsedOps, viewMode])

  // Filtered and grouped operations
  const groupedOps = useMemo(() => {
    const filtered = filterOperations(parsedOps, viewMode, selectedFilters)
    return getGroupedOperations(filtered, viewMode)
  }, [parsedOps, viewMode, selectedFilters])

  // Reset filters when view mode changes
  useEffect(() => {
    setSelectedFilters([])
  }, [viewMode])

  // Measure container width
  useEffect(() => {
    const updateWidth = () => {
      if (containerRef.current) {
        setContainerWidth(containerRef.current.offsetWidth - 112) // 112 = label width
      }
    }
    updateWidth()
    window.addEventListener('resize', updateWidth)
    return () => window.removeEventListener('resize', updateWidth)
  }, [])

  const handleHover = (op: ParsedPlanOp | null, x: number, y: number) => {
    if (op) {
      setHoveredOp({ op, x, y })
    } else {
      setHoveredOp(null)
    }
  }

  const filterLabel = viewMode === 'MACHINE' ? 'Máquinas' : viewMode === 'PRODUCT' ? 'Artigos' : 'Rotas'

  if (planLoading) {
    return (
      <div className="flex h-96 items-center justify-center rounded-2xl border border-border bg-surface">
        <p className="text-text-muted">A carregar plano de produção...</p>
      </div>
    )
  }

  if (!planData || planData.length === 0) {
    return (
      <div className="flex h-96 items-center justify-center rounded-2xl border border-border bg-surface">
        <p className="text-text-muted">Nenhum plano de produção disponível.</p>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <p className="text-xs uppercase tracking-[0.4em] text-text-muted">Timeline Avançada</p>
          <h2 className="text-2xl font-semibold text-text-primary">Plano de Produção</h2>
        </div>
        {bottleneck?.machine_id && (
          <BottleneckCard machineId={bottleneck.machine_id} totalMinutes={bottleneck.total_minutes} />
        )}
      </div>

      {/* Controls */}
      <div className="flex flex-wrap items-center gap-4 rounded-xl border border-border bg-surface p-4">
        <ViewModeSelector value={viewMode} onChange={setViewMode} />
        <div className="h-6 w-px bg-border" />
        <MultiSelect
          label={filterLabel}
          options={filterOptions}
          selected={selectedFilters}
          onChange={setSelectedFilters}
        />
        <div className="h-6 w-px bg-border" />
        <TimePresetButtons current={timePreset} onChange={setTimePreset} />
        <div className="h-6 w-px bg-border" />
        <ZoomSlider value={zoomPercent} onChange={setZoomPercent} />
        <div className="ml-auto">
          <RouteLegend />
        </div>
      </div>

      {/* Stats */}
      <div className="flex gap-4 text-xs text-text-muted">
        <span>{parsedOps.length} operações</span>
        <span>•</span>
        <span>{groupedOps.size} {viewMode === 'MACHINE' ? 'máquinas' : viewMode === 'PRODUCT' ? 'artigos' : 'rotas'}</span>
        <span>•</span>
        <span>
          {timeWindow.min.toLocaleDateString('pt-PT')} – {timeWindow.max.toLocaleDateString('pt-PT')}
        </span>
      </div>

      {/* Timeline */}
      <div
        ref={containerRef}
        className="overflow-x-auto rounded-xl border border-border bg-surface"
      >
        {/* Time Axis */}
        <div className="sticky top-0 z-10 ml-28">
          <TimeAxis timeWindow={timeWindow} containerWidth={containerWidth} />
        </div>

        {/* Lanes */}
        <div className="min-w-max">
          {[...groupedOps.entries()].map(([key, ops]) => (
            <Lane
              key={key}
              label={key}
              operations={ops}
              timeWindow={timeWindow}
              containerWidth={containerWidth}
              onHover={handleHover}
            />
          ))}
        </div>
      </div>

      {/* Tooltip */}
      {hoveredOp && <OperationTooltip op={hoveredOp.op} x={hoveredOp.x} y={hoveredOp.y} />}
    </div>
  )
}

export default PlanTimeline



