/**
 * Unified Gantt Chart Component - Industrial Production Cockpit
 * 
 * Features:
 * - View modes: Por Máquina, Por Artigo, Por Rota
 * - MultiSelect filter for entities
 * - Time presets: 1 Dia, 3 Dias, 1 Semana, Completo
 * - Zoom slider
 * - Route-based color coding
 * - Bottleneck indicator with alerts
 * - Sub-row layout for overlapping operations
 * - Intelligent suggestions panel
 * - Heatmap shading based on machine load
 * - Product route side panel on bar click
 */

import { useState, useEffect, useMemo, useRef, type FC } from 'react'
import { useQuery } from '@tanstack/react-query'
import { motion, AnimatePresence } from 'framer-motion'
import {
  apiGetPlan,
  apiGetBottleneck,
  apiGetPlanSuggestions,
  type PlanOp,
  type PlanSuggestions,
} from '../services/nikufraApi'

// -------------------------
// Types
// -------------------------

type ViewMode = 'MACHINE' | 'PRODUCT' | 'ROUTE'
type TimePreset = '1day' | '3days' | '1week' | 'full'

type ParsedOp = PlanOp & {
  start: Date
  end: Date
}

type TimeWindow = {
  from: Date
  to: Date
}

// -------------------------
// Utilities
// -------------------------

function parseOperations(ops: PlanOp[]): ParsedOp[] {
  return ops.map((op) => ({
    ...op,
    start: new Date(op.start_time),
    end: new Date(op.end_time),
  }))
}

function computeGlobalRange(ops: ParsedOp[]): TimeWindow {
  if (ops.length === 0) {
    const now = new Date()
    return { from: now, to: new Date(now.getTime() + 24 * 60 * 60 * 1000) }
  }
  let minTime = ops[0].start.getTime()
  let maxTime = ops[0].end.getTime()
  for (const op of ops) {
    if (op.start.getTime() < minTime) minTime = op.start.getTime()
    if (op.end.getTime() > maxTime) maxTime = op.end.getTime()
  }
  return { from: new Date(minTime), to: new Date(maxTime) }
}

function computeTimeWindow(preset: TimePreset, globalRange: TimeWindow, zoomPercent: number): TimeWindow {
  if (preset === 'full') {
    const globalDuration = globalRange.to.getTime() - globalRange.from.getTime()
    const visibleDuration = (zoomPercent / 100) * globalDuration
    return { from: globalRange.from, to: new Date(globalRange.from.getTime() + visibleDuration) }
  }
  const days = preset === '1day' ? 1 : preset === '3days' ? 3 : 7
  return { from: globalRange.from, to: new Date(globalRange.from.getTime() + days * 24 * 60 * 60 * 1000) }
}

function computeX(date: Date, window: TimeWindow, containerWidth: number): number {
  const duration = window.to.getTime() - window.from.getTime()
  if (duration <= 0) return 0
  const ratio = (date.getTime() - window.from.getTime()) / duration
  return Math.max(0, Math.min(containerWidth, ratio * containerWidth))
}

function computeWidth(start: Date, end: Date, window: TimeWindow, containerWidth: number): number {
  const duration = window.to.getTime() - window.from.getTime()
  if (duration <= 0) return 0
  const clampedStart = Math.max(start.getTime(), window.from.getTime())
  const clampedEnd = Math.min(end.getTime(), window.to.getTime())
  if (clampedEnd <= clampedStart) return 0
  return Math.max(4, ((clampedEnd - clampedStart) / duration) * containerWidth)
}

function isVisible(op: ParsedOp, window: TimeWindow): boolean {
  return op.end.getTime() > window.from.getTime() && op.start.getTime() < window.to.getTime()
}

function getGroupKey(op: ParsedOp, mode: ViewMode): string {
  switch (mode) {
    case 'MACHINE': return op.machine_id
    case 'PRODUCT': return op.article_id
    case 'ROUTE': return `Rota ${op.route_label}`
  }
}

function groupOperations(ops: ParsedOp[], mode: ViewMode): Map<string, ParsedOp[]> {
  const groups = new Map<string, ParsedOp[]>()
  for (const op of ops) {
    const key = getGroupKey(op, mode)
    if (!groups.has(key)) groups.set(key, [])
    groups.get(key)!.push(op)
  }
  for (const [, groupOps] of groups) {
    groupOps.sort((a, b) => a.start.getTime() - b.start.getTime())
  }
  return new Map([...groups.entries()].sort((a, b) => a[0].localeCompare(b[0])))
}

function calculateSubRows(ops: ParsedOp[]): Map<ParsedOp, number> {
  const assignments = new Map<ParsedOp, number>()
  const rows: ParsedOp[][] = []
  for (const op of ops) {
    let placed = false
    for (let i = 0; i < rows.length; i++) {
      const hasCollision = rows[i].some(
        (existing) => !(op.end.getTime() <= existing.start.getTime() || op.start.getTime() >= existing.end.getTime())
      )
      if (!hasCollision) {
        rows[i].push(op)
        assignments.set(op, i)
        placed = true
        break
      }
    }
    if (!placed) {
      rows.push([op])
      assignments.set(op, rows.length - 1)
    }
  }
  return assignments
}

function getUniqueValues(ops: ParsedOp[], mode: ViewMode): string[] {
  const values = new Set<string>()
  for (const op of ops) {
    values.add(getGroupKey(op, mode))
  }
  return [...values].sort()
}

function filterOperations(ops: ParsedOp[], mode: ViewMode, selected: string[]): ParsedOp[] {
  if (selected.length === 0) return ops
  return ops.filter((op) => selected.includes(getGroupKey(op, mode)))
}

function getRouteColor(label: string): string {
  switch (label?.toUpperCase()) {
    case 'A': return '#3B82F6'
    case 'B': return '#F59E0B'
    case 'C': return '#10B981'
    case 'D': return '#8B5CF6'
    case 'E': return '#EC4899'
    default: return '#6B7280'
  }
}

function getLoadColor(utilizationPct: number): string {
  if (utilizationPct <= 50) return 'rgba(16, 185, 129, 0.1)' // Green
  if (utilizationPct <= 80) return 'rgba(245, 158, 11, 0.1)' // Yellow
  return 'rgba(239, 68, 68, 0.15)' // Red
}

function formatDate(date: Date, style: 'short' | 'full' = 'short'): string {
  const d = String(date.getDate()).padStart(2, '0')
  const m = String(date.getMonth() + 1).padStart(2, '0')
  if (style === 'full') {
    const h = String(date.getHours()).padStart(2, '0')
    const min = String(date.getMinutes()).padStart(2, '0')
    return `${d}/${m} ${h}:${min}`
  }
  return `${d}/${m}`
}

function formatDuration(minutes: number): string {
  if (minutes < 60) return `${Math.round(minutes)}min`
  const hours = minutes / 60
  if (hours < 24) return `${hours.toFixed(1)}h`
  return `${(hours / 24).toFixed(1)}d`
}

type TickMark = { x: number; label: string; isMajor: boolean }

function generateTicks(window: TimeWindow, containerWidth: number): TickMark[] {
  const ticks: TickMark[] = []
  const duration = window.to.getTime() - window.from.getTime()
  const hours = duration / (60 * 60 * 1000)
  const intervalHours = hours <= 24 ? 2 : hours <= 72 ? 6 : hours <= 168 ? 12 : 24
  const intervalMs = intervalHours * 60 * 60 * 1000

  let current = new Date(window.from)
  current.setMinutes(0, 0, 0)
  const h = current.getHours()
  current.setHours(Math.floor(h / intervalHours) * intervalHours)

  while (current <= window.to) {
    if (current >= window.from) {
      const x = computeX(current, window, containerWidth)
      const isMajor = current.getHours() === 0
      const label = isMajor || intervalHours >= 24
        ? formatDate(current, 'short')
        : `${String(current.getHours()).padStart(2, '0')}:00`
      ticks.push({ x, label, isMajor })
    }
    current = new Date(current.getTime() + intervalMs)
  }
  return ticks
}

// -------------------------
// Sub-components
// -------------------------

const ViewModeSelector: FC<{ value: ViewMode; onChange: (v: ViewMode) => void }> = ({ value, onChange }) => (
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

const TimePresetButtons: FC<{ current: TimePreset; onChange: (p: TimePreset) => void }> = ({ current, onChange }) => {
  const presets: { key: TimePreset; label: string }[] = [
    { key: '1day', label: '1 Dia' },
    { key: '3days', label: '3 Dias' },
    { key: '1week', label: '1 Semana' },
    { key: 'full', label: 'Completo' },
  ]
  return (
    <div className="flex gap-1">
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
    const handleClick = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', handleClick)
    return () => document.removeEventListener('mousedown', handleClick)
  }, [])

  const toggle = (opt: string) => {
    if (selected.includes(opt)) onChange(selected.filter((s) => s !== opt))
    else onChange([...selected, opt])
  }

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
            <button onClick={() => onChange(options)} className="flex-1 rounded bg-nikufra/10 px-2 py-1 text-xs text-nikufra hover:bg-nikufra/20">Todos</button>
            <button onClick={() => onChange([])} className="flex-1 rounded bg-danger/10 px-2 py-1 text-xs text-danger hover:bg-danger/20">Limpar</button>
          </div>
          {options.map((opt) => (
            <label key={opt} className="flex cursor-pointer items-center gap-2 rounded px-2 py-1 text-sm text-text-primary hover:bg-background">
              <input type="checkbox" checked={selected.includes(opt)} onChange={() => toggle(opt)} className="accent-nikufra" />
              {opt}
            </label>
          ))}
        </div>
      )}
    </div>
  )
}

const ZoomSlider: FC<{ value: number; onChange: (v: number) => void }> = ({ value, onChange }) => (
  <div className="flex items-center gap-2">
    <span className="text-xs text-text-muted">Zoom</span>
    <input
      type="range"
      min={10}
      max={100}
      value={value}
      onChange={(e) => onChange(Number(e.target.value))}
      className="h-1.5 w-24 cursor-pointer appearance-none rounded bg-border accent-nikufra"
    />
    <span className="w-8 text-xs text-text-muted">{value}%</span>
  </div>
)

const RouteLegend: FC = () => (
  <div className="flex flex-wrap items-center gap-3">
    {['A', 'B', 'C', 'D'].map((r) => (
      <div key={r} className="flex items-center gap-1.5">
        <div className="h-3 w-3 rounded" style={{ backgroundColor: getRouteColor(r) }} />
        <span className="text-xs text-text-muted">Rota {r}</span>
      </div>
    ))}
  </div>
)

const BottleneckCard: FC<{ machineId: string; totalMinutes: number; hasAlerts?: boolean }> = ({ machineId, totalMinutes, hasAlerts }) => (
  <div className={`rounded-xl border p-3 ${hasAlerts ? 'border-danger/60 bg-danger/10' : 'border-warning/40 bg-warning/10'}`}>
    <div className="flex items-center gap-2">
      <p className={`text-xs uppercase tracking-widest ${hasAlerts ? 'text-danger' : 'text-warning'}`}>
        {hasAlerts && '⚠️ '} Gargalo Atual
      </p>
    </div>
    <p className="mt-1 text-lg font-bold text-text-primary">{machineId}</p>
    <p className="text-xs text-text-muted">{(totalMinutes / 60).toFixed(1)} horas de carga</p>
  </div>
)

const TimeAxis: FC<{ ticks: TickMark[] }> = ({ ticks }) => (
  <div className="relative h-8 border-b border-border bg-background/30">
    {ticks.map((tick, i) => (
      <div key={i} className="absolute top-0 h-full" style={{ left: tick.x }}>
        <div className={`h-2 w-px ${tick.isMajor ? 'bg-nikufra' : 'bg-border'}`} />
        <span
          className={`absolute top-2 whitespace-nowrap text-[10px] ${tick.isMajor ? 'font-semibold text-nikufra' : 'text-text-muted'}`}
          style={{ transform: 'translateX(-50%)' }}
        >
          {tick.label}
        </span>
      </div>
    ))}
  </div>
)

const OperationBar: FC<{
  op: ParsedOp
  x: number
  width: number
  subRow: number
  rowHeight: number
  onHover: (op: ParsedOp | null, x: number, y: number) => void
  onClick: (op: ParsedOp) => void
}> = ({ op, x, width, subRow, rowHeight, onHover, onClick }) => {
  const color = getRouteColor(op.route_label)
  const barHeight = rowHeight - 8
  const top = subRow * rowHeight + 4

  return (
    <motion.div
      initial={{ opacity: 0, scaleX: 0.8 }}
      animate={{ opacity: 1, scaleX: 1 }}
      className="absolute flex cursor-pointer items-center gap-1 overflow-hidden rounded-md px-1.5 transition-transform hover:z-20 hover:scale-y-105"
      style={{ left: x, width: Math.max(width, 4), top, height: barHeight, backgroundColor: color }}
      onMouseEnter={(e) => onHover(op, e.clientX, e.clientY)}
      onMouseMove={(e) => onHover(op, e.clientX, e.clientY)}
      onMouseLeave={() => onHover(null, 0, 0)}
      onClick={() => onClick(op)}
    >
      <span className="truncate text-[10px] font-bold text-white">{op.order_id}</span>
      {width > 60 && <span className="truncate text-[9px] text-white/80">{op.op_code}</span>}
      {width > 100 && (
        <span className="ml-auto shrink-0 rounded bg-white/20 px-1 py-0.5 text-[8px] text-white">{op.route_label}</span>
      )}
    </motion.div>
  )
}

const Tooltip: FC<{ op: ParsedOp; x: number; y: number }> = ({ op, x, y }) => (
  <motion.div
    initial={{ opacity: 0, scale: 0.95 }}
    animate={{ opacity: 1, scale: 1 }}
    exit={{ opacity: 0, scale: 0.95 }}
    className="pointer-events-none fixed z-50 max-w-xs rounded-lg border border-nikufra/50 bg-surface p-3 shadow-xl"
    style={{ left: x + 12, top: y + 12 }}
  >
    <div className="space-y-2">
      <div className="border-b border-border pb-1 text-sm font-bold text-text-primary">
        {op.order_id} • {op.op_code}
      </div>
      <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
        <div><span className="text-text-muted">Máquina:</span> <span className="font-medium text-text-primary">{op.machine_id}</span></div>
        <div><span className="text-text-muted">Artigo:</span> <span className="font-medium text-text-primary">{op.article_id}</span></div>
        <div><span className="text-text-muted">Rota:</span> <span className="font-bold" style={{ color: getRouteColor(op.route_label) }}>{op.route_label}</span></div>
        <div><span className="text-text-muted">Quantidade:</span> <span className="font-medium text-text-primary">{op.qty}</span></div>
        <div><span className="text-text-muted">Duração:</span> <span className="font-medium text-nikufra">{formatDuration(op.duration_min || (op.end.getTime() - op.start.getTime()) / 60000)}</span></div>
        <div className="col-span-2 border-t border-border/50 pt-1">
          <span className="text-text-muted">Início:</span> <span className="text-text-primary">{formatDate(op.start, 'full')}</span>
        </div>
        <div className="col-span-2">
          <span className="text-text-muted">Fim:</span> <span className="text-text-primary">{formatDate(op.end, 'full')}</span>
        </div>
      </div>
    </div>
  </motion.div>
)

// Product Route Side Panel
const ProductRoutePanel: FC<{
  article: string
  operations: ParsedOp[]
  onClose: () => void
  onViewArticle: () => void
}> = ({ article, operations, onClose, onViewArticle }) => {
  // Sort by op_seq or start time
  const sortedOps = [...operations].sort((a, b) => {
    if (a.op_seq !== b.op_seq) return a.op_seq - b.op_seq
    return a.start.getTime() - b.start.getTime()
  })

  // Find gaps between operations
  const findGap = (prevEnd: Date, nextStart: Date): number => {
    return (nextStart.getTime() - prevEnd.getTime()) / 60000 // minutes
  }

  return (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: 20 }}
      className="fixed right-4 top-24 z-40 w-80 max-h-[80vh] overflow-auto rounded-xl border border-nikufra/40 bg-surface p-4 shadow-2xl"
    >
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-bold text-text-primary">Percurso {article}</h3>
        <button onClick={onClose} className="text-text-muted hover:text-text-primary">✕</button>
      </div>
      
      <div className="space-y-3">
        {sortedOps.map((op, idx) => {
          const prevOp = idx > 0 ? sortedOps[idx - 1] : null
          const gap = prevOp ? findGap(prevOp.end, op.start) : 0
          const hasDelay = gap > 60 // > 1 hour

          return (
            <div key={`${op.op_code}-${idx}`}>
              {hasDelay && (
                <div className="mb-2 rounded-lg bg-warning/10 border border-warning/40 px-3 py-2 text-xs text-warning">
                  ⏳ Espera de {formatDuration(gap)} antes desta operação
                </div>
              )}
              <div className="rounded-lg border border-border bg-background/50 p-3">
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-text-primary">{op.op_code}</span>
                  <span className="rounded px-2 py-0.5 text-xs" style={{ backgroundColor: getRouteColor(op.route_label) + '33', color: getRouteColor(op.route_label) }}>
                    Rota {op.route_label}
                  </span>
                </div>
                <div className="mt-2 grid grid-cols-2 gap-2 text-xs text-text-muted">
                  <div>Máquina: <span className="text-text-primary">{op.machine_id}</span></div>
                  <div>Duração: <span className="text-nikufra">{formatDuration(op.duration_min)}</span></div>
                  <div className="col-span-2">
                    {formatDate(op.start, 'full')} → {formatDate(op.end, 'full')}
                  </div>
                </div>
              </div>
            </div>
          )
        })}
      </div>

      <button
        onClick={onViewArticle}
        className="mt-4 w-full rounded-lg bg-nikufra/20 border border-nikufra px-4 py-2 text-sm font-semibold text-nikufra hover:bg-nikufra hover:text-white transition"
      >
        Ver percurso de {article} no Gantt
      </button>
    </motion.div>
  )
}

// Suggestions Panel
const SuggestionsPanel: FC<{ suggestions: PlanSuggestions | undefined }> = ({ suggestions }) => {
  if (!suggestions || suggestions.summary.total_suggestions === 0) return null

  const allSuggestions = [
    ...suggestions.overload_suggestions.map(s => ({ ...s, priority: s.expected_gain_h > 2 ? 'high' : 'medium' })),
    ...suggestions.idle_gaps.map(s => ({ ...s, priority: s.gap_min > 120 ? 'high' : 'low' })),
    ...suggestions.product_risks.map(s => ({ ...s, priority: s.wait_min > 120 ? 'high' : 'medium' })),
  ].slice(0, 5) // Show top 5

  return (
    <div className="rounded-xl border border-nikufra/40 bg-gradient-to-br from-nikufra/10 to-nikufra/5 p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-bold text-text-primary flex items-center gap-2">
          💡 Sugestões Inteligentes
          <span className="rounded-full bg-nikufra/20 px-2 py-0.5 text-xs text-nikufra">
            {suggestions.summary.total_suggestions}
          </span>
          {suggestions.summary.high_priority > 0 && (
            <span className="rounded-full bg-danger/20 px-2 py-0.5 text-xs text-danger">
              {suggestions.summary.high_priority} urgentes
            </span>
          )}
        </h3>
      </div>
      <div className="space-y-2">
        {allSuggestions.map((s, i) => (
          <motion.div
            key={i}
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: i * 0.05 }}
            className={`rounded-lg border px-3 py-2 text-sm ${
              s.priority === 'high'
                ? 'border-danger/40 bg-danger/10 text-danger'
                : s.priority === 'medium'
                ? 'border-warning/40 bg-warning/10 text-warning'
                : 'border-nikufra/40 bg-nikufra/10 text-nikufra'
            }`}
          >
            {s.formatted_pt}
          </motion.div>
        ))}
      </div>
    </div>
  )
}

// -------------------------
// Main Component
// -------------------------

export const UnifiedGantt: FC = () => {
  const containerRef = useRef<HTMLDivElement>(null)
  const [containerWidth, setContainerWidth] = useState(800)

  // State
  const [viewMode, setViewMode] = useState<ViewMode>('MACHINE')
  const [timePreset, setTimePreset] = useState<TimePreset>('full')
  const [zoomPercent, setZoomPercent] = useState(100)
  const [selectedFilters, setSelectedFilters] = useState<string[]>([])
  const [hoveredOp, setHoveredOp] = useState<{ op: ParsedOp; x: number; y: number } | null>(null)
  const [selectedOp, setSelectedOp] = useState<ParsedOp | null>(null)

  // Queries
  const { data: planData, isLoading } = useQuery({
    queryKey: ['unified-gantt-plan'],
    queryFn: apiGetPlan,
    staleTime: 30_000,
  })

  const { data: bottleneck } = useQuery({
    queryKey: ['unified-gantt-bottleneck'],
    queryFn: apiGetBottleneck,
    staleTime: 60_000,
  })

  const { data: suggestions } = useQuery({
    queryKey: ['unified-gantt-suggestions'],
    queryFn: apiGetPlanSuggestions,
    staleTime: 60_000,
  })

  // Parse operations
  const parsedOps = useMemo(() => (planData ? parseOperations(planData) : []), [planData])

  // Global time range
  const globalRange = useMemo(() => computeGlobalRange(parsedOps), [parsedOps])

  // Current time window
  const timeWindow = useMemo(
    () => computeTimeWindow(timePreset, globalRange, zoomPercent),
    [timePreset, globalRange, zoomPercent]
  )

  // Filter options
  const filterOptions = useMemo(() => getUniqueValues(parsedOps, viewMode), [parsedOps, viewMode])

  // Filtered and grouped operations
  const groupedOps = useMemo(() => {
    const filtered = filterOperations(parsedOps, viewMode, selectedFilters)
    return groupOperations(filtered, viewMode)
  }, [parsedOps, viewMode, selectedFilters])

  // Ticks for time axis
  const ticks = useMemo(() => generateTicks(timeWindow, containerWidth), [timeWindow, containerWidth])

  // Machine loads for heatmap (only in MACHINE view)
  const machineLoads = useMemo(() => {
    return suggestions?.machine_loads || {}
  }, [suggestions])

  // Operations for selected article (for side panel)
  const selectedArticleOps = useMemo(() => {
    if (!selectedOp) return []
    return parsedOps.filter(op => op.article_id === selectedOp.article_id)
  }, [selectedOp, parsedOps])

  // Check if there are alerts
  const hasAlerts = useMemo(() => {
    if (!suggestions) return false
    return suggestions.summary.high_priority > 0
  }, [suggestions])

  // Reset filters when view mode changes
  useEffect(() => {
    setSelectedFilters([])
  }, [viewMode])

  // Measure container
  useEffect(() => {
    const updateWidth = () => {
      if (containerRef.current) {
        setContainerWidth(containerRef.current.offsetWidth - 140)
      }
    }
    updateWidth()
    window.addEventListener('resize', updateWidth)
    return () => window.removeEventListener('resize', updateWidth)
  }, [])

  const handlePresetChange = (preset: TimePreset) => {
    setTimePreset(preset)
    if (preset === 'full') setZoomPercent(100)
  }

  const handleZoomChange = (value: number) => {
    setZoomPercent(value)
    setTimePreset('full')
  }

  const handleHover = (op: ParsedOp | null, x: number, y: number) => {
    if (op) setHoveredOp({ op, x, y })
    else setHoveredOp(null)
  }

  const handleBarClick = (op: ParsedOp) => {
    setSelectedOp(op)
  }

  const handleViewArticle = () => {
    if (selectedOp) {
      setViewMode('PRODUCT')
      setSelectedFilters([selectedOp.article_id])
      setSelectedOp(null)
    }
  }

  const filterLabel = viewMode === 'MACHINE' ? 'Máquinas' : viewMode === 'PRODUCT' ? 'Artigos' : 'Rotas'
  const ROW_HEIGHT = 28
  const LABEL_WIDTH = 140

  if (isLoading) {
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
          <h2 className="text-xl font-bold text-text-primary">Plano de Produção</h2>
          <p className="text-xs text-text-muted">
            {formatDate(timeWindow.from, 'full')} → {formatDate(timeWindow.to, 'full')}
          </p>
        </div>
        {bottleneck?.machine_id && (
          <BottleneckCard 
            machineId={bottleneck.machine_id} 
            totalMinutes={bottleneck.total_minutes}
            hasAlerts={hasAlerts}
          />
        )}
      </div>

      {/* Suggestions Panel */}
      <SuggestionsPanel suggestions={suggestions} />

      {/* Controls */}
      <div className="flex flex-wrap items-center gap-4 rounded-xl border border-border bg-surface p-4">
        <ViewModeSelector value={viewMode} onChange={setViewMode} />
        <div className="h-6 w-px bg-border" />
        <MultiSelect label={filterLabel} options={filterOptions} selected={selectedFilters} onChange={setSelectedFilters} />
        <div className="h-6 w-px bg-border" />
        <TimePresetButtons current={timePreset} onChange={handlePresetChange} />
        <div className="h-6 w-px bg-border" />
        <ZoomSlider value={zoomPercent} onChange={handleZoomChange} />
        <div className="ml-auto">
          <RouteLegend />
        </div>
      </div>

      {/* Stats */}
      <div className="flex gap-4 text-xs text-text-muted">
        <span>{parsedOps.length} operações</span>
        <span>•</span>
        <span>{groupedOps.size} {viewMode === 'MACHINE' ? 'máquinas' : viewMode === 'PRODUCT' ? 'artigos' : 'rotas'}</span>
        {viewMode === 'MACHINE' && (
          <>
            <span>•</span>
            <span className="flex items-center gap-1">
              <span className="inline-block h-2 w-2 rounded bg-green-500/50" /> ≤50%
              <span className="ml-1 inline-block h-2 w-2 rounded bg-yellow-500/50" /> 50-80%
              <span className="ml-1 inline-block h-2 w-2 rounded bg-red-500/50" /> &gt;80%
            </span>
          </>
        )}
      </div>

      {/* Gantt Chart */}
      <div ref={containerRef} className="overflow-hidden rounded-xl border border-border bg-surface">
        {/* Time Axis Header */}
        <div className="flex">
          <div
            className="shrink-0 border-b border-r border-border bg-background/50 px-3 py-2 text-xs font-semibold text-text-primary"
            style={{ width: LABEL_WIDTH }}
          >
            {viewMode === 'MACHINE' ? 'Máquina' : viewMode === 'PRODUCT' ? 'Artigo' : 'Rota'}
          </div>
          <div className="flex-1">
            <TimeAxis ticks={ticks} />
          </div>
        </div>

        {/* Rows */}
        <div className="max-h-[500px] overflow-y-auto">
          {[...groupedOps.entries()].map(([groupKey, ops], groupIdx) => {
            const visibleOps = ops.filter((op) => isVisible(op, timeWindow))
            const subRowMap = calculateSubRows(visibleOps)
            const maxSubRow = Math.max(...Array.from(subRowMap.values()), 0)
            const rowCount = maxSubRow + 1
            const rowHeight = rowCount * ROW_HEIGHT + 8

            // Get load color for heatmap (only in MACHINE view)
            const loadPct = viewMode === 'MACHINE' && machineLoads[groupKey]
              ? machineLoads[groupKey].utilization_pct
              : 0
            const heatmapBg = viewMode === 'MACHINE' ? getLoadColor(loadPct) : undefined

            return (
              <div
                key={groupKey}
                className={`flex border-b border-border/50 ${groupIdx % 2 === 0 && !heatmapBg ? 'bg-background/10' : ''}`}
                style={{ minHeight: rowHeight, backgroundColor: heatmapBg }}
              >
                {/* Row Label */}
                <div className="shrink-0 border-r border-border/50 px-3 py-2" style={{ width: LABEL_WIDTH }}>
                  <div className={`text-sm font-semibold ${visibleOps.length > 0 ? 'text-text-primary' : 'text-text-muted/40'}`}>
                    {groupKey}
                  </div>
                  <div className="flex items-center gap-2">
                    {ops.length > 0 && <span className="text-[10px] text-text-muted">{ops.length} op{ops.length > 1 ? 's' : ''}</span>}
                    {viewMode === 'MACHINE' && loadPct > 0 && (
                      <span className={`text-[10px] font-semibold ${loadPct > 80 ? 'text-danger' : loadPct > 50 ? 'text-warning' : 'text-nikufra'}`}>
                        {loadPct.toFixed(0)}%
                      </span>
                    )}
                  </div>
                </div>

                {/* Operations Area */}
                <div className="relative flex-1" style={{ minHeight: rowHeight }}>
                  {/* Grid lines */}
                  {ticks.map((tick, i) => (
                    <div key={i} className="absolute bottom-0 top-0 w-px bg-border/20" style={{ left: tick.x }} />
                  ))}

                  {/* Operation bars */}
                  {visibleOps.map((op, opIdx) => {
                    const x = computeX(op.start, timeWindow, containerWidth)
                    const width = computeWidth(op.start, op.end, timeWindow, containerWidth)
                    const subRow = subRowMap.get(op) || 0
                    return (
                      <OperationBar
                        key={`${op.order_id}-${op.op_code}-${opIdx}`}
                        op={op}
                        x={x}
                        width={width}
                        subRow={subRow}
                        rowHeight={ROW_HEIGHT}
                        onHover={handleHover}
                        onClick={handleBarClick}
                      />
                    )
                  })}
                </div>
              </div>
            )
          })}
        </div>
      </div>

      {/* Tooltip */}
      <AnimatePresence>
        {hoveredOp && <Tooltip op={hoveredOp.op} x={hoveredOp.x} y={hoveredOp.y} />}
      </AnimatePresence>

      {/* Product Route Side Panel */}
      <AnimatePresence>
        {selectedOp && (
          <ProductRoutePanel
            article={selectedOp.article_id}
            operations={selectedArticleOps}
            onClose={() => setSelectedOp(null)}
            onViewArticle={handleViewArticle}
          />
        )}
      </AnimatePresence>
    </div>
  )
}

export default UnifiedGantt
