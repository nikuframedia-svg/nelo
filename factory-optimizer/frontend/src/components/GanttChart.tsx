/**
 * GanttChart Component - Production Timeline Visualization
 * 
 * Features:
 * - Proper time-to-pixel mapping using timeScale utilities
 * - View modes: Por Máquina / Por Produto
 * - Time window controls: 1 dia, 3 dias, 1 semana, Completo
 * - Zoom slider for fine-grained control
 * - Clean time axis with readable tick labels
 * - Color-coded by route (A=blue, B=orange, C=green)
 */

import { useState, useMemo, useRef, useEffect, type FC } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Operation } from '../types'
import {
  type TimeWindow,
  type TimePreset,
  type ViewMode,
  type ParsedOperation,
  type TickMark,
  computeGlobalTimeRange,
  createTimeWindow,
  createTimeWindowFromZoom,
  computeX,
  computeWidth,
  isOperationVisible,
  generateTimeAxisTicks,
  groupOperations,
  calculateSubRows,
  getRouteColor,
  formatDate,
  formatDuration,
} from '../utils/timeScale'

// -------------------------
// Types
// -------------------------

interface GanttChartProps {
  operations: Operation[]
  title: string
  startDate: Date
  endDate: Date
  isBaseline?: boolean
  allMachines?: string[]
}

// Convert Operation to ParsedOperation
// Supports both old format (ordem, artigo, rota) and new format (order_id, article_id, route_label)
function convertOperations(ops: Operation[]): ParsedOperation[] {
  return ops.map(op => {
    // New API format fields (preferred)
    const order_id = (op as any).order_id || op.ordem || ''
    const article_id = (op as any).article_id || op.artigo || ''
    const route_label = (op as any).route_label || op.rota || 'A'
    const op_code = (op as any).op_code || op.operacao || ''
    const machine_id = (op as any).machine_id || op.recurso || ''
    const qty = (op as any).qty || 0
    const duration_min = (op as any).duration_min || 0
    
    return {
      order_id,
      article_id,
      route_id: route_label,
      route_label,
      op_seq: (op as any).op_seq || 0,
      op_code,
      machine_id,
      qty,
      start: new Date(op.start_time),
      end: new Date(op.end_time),
      duration_min,
    }
  })
}

// -------------------------
// Sub-components
// -------------------------

const ViewModeToggle: FC<{
  mode: ViewMode
  onChange: (mode: ViewMode) => void
}> = ({ mode, onChange }) => (
  <div className="flex rounded-lg border border-border bg-surface p-1">
    <button
      onClick={() => onChange('MACHINE')}
      className={`rounded-md px-3 py-1.5 text-xs font-medium transition ${
        mode === 'MACHINE'
          ? 'bg-nikufra text-white'
          : 'text-text-muted hover:text-text-primary'
      }`}
    >
      Por Máquina
    </button>
    <button
      onClick={() => onChange('PRODUCT')}
      className={`rounded-md px-3 py-1.5 text-xs font-medium transition ${
        mode === 'PRODUCT'
          ? 'bg-nikufra text-white'
          : 'text-text-muted hover:text-text-primary'
      }`}
    >
      Por Produto
    </button>
  </div>
)

const TimePresetButtons: FC<{
  current: TimePreset
  onChange: (preset: TimePreset) => void
}> = ({ current, onChange }) => {
  const presets: { key: TimePreset; label: string }[] = [
    { key: '1day', label: '1 Dia' },
    { key: '3days', label: '3 Dias' },
    { key: '1week', label: '1 Semana' },
    { key: 'full', label: 'Completo' },
  ]

  return (
    <div className="flex gap-1">
      {presets.map(p => (
        <button
          key={p.key}
          onClick={() => onChange(p.key)}
          className={`rounded-md px-3 py-1.5 text-xs font-medium transition ${
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

const ZoomSlider: FC<{
  value: number
  onChange: (value: number) => void
}> = ({ value, onChange }) => (
  <div className="flex items-center gap-2">
    <span className="text-xs text-text-muted">Zoom</span>
    <input
      type="range"
      min={10}
      max={100}
      value={value}
      onChange={e => onChange(Number(e.target.value))}
      className="h-1.5 w-24 cursor-pointer appearance-none rounded bg-border accent-nikufra"
    />
    <span className="w-8 text-xs text-text-muted">{value}%</span>
  </div>
)

const RouteLegend: FC = () => (
  <div className="flex gap-3">
    {['A', 'B', 'C'].map(r => (
      <div key={r} className="flex items-center gap-1.5">
        <div
          className="h-3 w-3 rounded"
          style={{ backgroundColor: getRouteColor(r) }}
        />
        <span className="text-xs text-text-muted">Rota {r}</span>
      </div>
    ))}
  </div>
)

const TimeAxis: FC<{
  ticks: TickMark[]
  containerWidth: number
}> = ({ ticks }) => (
  <div className="relative h-8 border-b border-border bg-background/30">
    {ticks.map((tick, i) => (
      <div
        key={i}
        className="absolute top-0 h-full"
        style={{ left: tick.x }}
      >
        <div
          className={`h-2 w-px ${tick.isMajor ? 'bg-nikufra' : 'bg-border'}`}
        />
        <span
          className={`absolute top-2 whitespace-nowrap text-[10px] ${
            tick.isMajor ? 'font-semibold text-nikufra' : 'text-text-muted'
          }`}
          style={{ transform: 'translateX(-50%)' }}
        >
          {tick.label}
        </span>
      </div>
    ))}
  </div>
)

const OperationBar: FC<{
  op: ParsedOperation
  x: number
  width: number
  subRow: number
  rowHeight: number
  onHover: (op: ParsedOperation | null, x: number, y: number) => void
}> = ({ op, x, width, subRow, rowHeight, onHover }) => {
  const color = getRouteColor(op.route_label)
  const barHeight = rowHeight - 8
  const top = subRow * rowHeight + 4

  return (
    <motion.div
      initial={{ opacity: 0, scaleX: 0.8 }}
      animate={{ opacity: 1, scaleX: 1 }}
      className="absolute flex cursor-pointer items-center gap-1 overflow-hidden rounded-md px-1.5 transition-transform hover:z-20 hover:scale-y-105"
      style={{
        left: x,
        width: Math.max(width, 4),
        top,
        height: barHeight,
        backgroundColor: color,
      }}
      onMouseEnter={e => onHover(op, e.clientX, e.clientY)}
      onMouseMove={e => onHover(op, e.clientX, e.clientY)}
      onMouseLeave={() => onHover(null, 0, 0)}
    >
      <span className="truncate text-[10px] font-bold text-white">
        {op.order_id}
      </span>
      {width > 60 && (
        <span className="truncate text-[9px] text-white/80">{op.op_code}</span>
      )}
      {width > 100 && (
        <span className="ml-auto shrink-0 rounded bg-white/20 px-1 py-0.5 text-[8px] text-white">
          {op.route_label}
        </span>
      )}
    </motion.div>
  )
}

const Tooltip: FC<{
  op: ParsedOperation
  x: number
  y: number
}> = ({ op, x, y }) => (
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
        <div>
          <span className="text-text-muted">Máquina:</span>
          <span className="ml-1 font-medium text-text-primary">
            {op.machine_id}
          </span>
        </div>
        <div>
          <span className="text-text-muted">Artigo:</span>
          <span className="ml-1 font-medium text-text-primary">
            {op.article_id}
          </span>
        </div>
        <div>
          <span className="text-text-muted">Rota:</span>
          <span
            className="ml-1 font-bold"
            style={{ color: getRouteColor(op.route_label) }}
          >
            {op.route_label}
          </span>
        </div>
        <div>
          <span className="text-text-muted">Duração:</span>
          <span className="ml-1 font-medium text-nikufra">
            {formatDuration(op.duration_min || 
              (op.end.getTime() - op.start.getTime()) / 60000)}
          </span>
        </div>
        <div className="col-span-2 border-t border-border/50 pt-1">
          <span className="text-text-muted">Início:</span>
          <span className="ml-1 text-text-primary">
            {formatDate(op.start, 'full')}
          </span>
        </div>
        <div className="col-span-2">
          <span className="text-text-muted">Fim:</span>
          <span className="ml-1 text-text-primary">
            {formatDate(op.end, 'full')}
          </span>
        </div>
      </div>
    </div>
  </motion.div>
)

// -------------------------
// Main Component
// -------------------------

export const GanttChart: FC<GanttChartProps> = ({
  operations,
  title,
  isBaseline = false,
}) => {
  const containerRef = useRef<HTMLDivElement>(null)
  const [containerWidth, setContainerWidth] = useState(800)

  // Convert and parse operations
  const parsedOps = useMemo(() => convertOperations(operations), [operations])

  // Compute global time range from data
  const globalRange = useMemo(
    () => computeGlobalTimeRange(parsedOps),
    [parsedOps]
  )

  // State
  const [viewMode, setViewMode] = useState<ViewMode>('MACHINE')
  const [timePreset, setTimePreset] = useState<TimePreset>('full')
  const [zoomPercent, setZoomPercent] = useState(100)
  const [hoveredOp, setHoveredOp] = useState<{
    op: ParsedOperation
    x: number
    y: number
  } | null>(null)

  // Compute time window based on preset/zoom
  const timeWindow = useMemo<TimeWindow>(() => {
    if (timePreset === 'full') {
      return createTimeWindowFromZoom(zoomPercent, globalRange)
    }
    return createTimeWindow(timePreset, globalRange)
  }, [timePreset, zoomPercent, globalRange])

  // Group operations by view mode
  const groupedOps = useMemo(
    () => groupOperations(parsedOps, viewMode),
    [parsedOps, viewMode]
  )

  // Generate time axis ticks
  const ticks = useMemo(
    () => generateTimeAxisTicks(timeWindow, containerWidth),
    [timeWindow, containerWidth]
  )

  // Measure container width
  useEffect(() => {
    const updateWidth = () => {
      if (containerRef.current) {
        const rect = containerRef.current.getBoundingClientRect()
        setContainerWidth(rect.width - 140) // Subtract label column width
      }
    }
    updateWidth()
    window.addEventListener('resize', updateWidth)
    return () => window.removeEventListener('resize', updateWidth)
  }, [])

  // Handle preset change
  const handlePresetChange = (preset: TimePreset) => {
    setTimePreset(preset)
    if (preset === 'full') {
      setZoomPercent(100)
    }
  }

  // Handle zoom change
  const handleZoomChange = (value: number) => {
    setZoomPercent(value)
    setTimePreset('full') // Switch to full mode when using zoom
  }

  const handleHover = (op: ParsedOperation | null, x: number, y: number) => {
    if (op) {
      setHoveredOp({ op, x, y })
    } else {
      setHoveredOp(null)
    }
  }

  const ROW_HEIGHT = 28
  const LABEL_WIDTH = 140

  if (parsedOps.length === 0) {
    return (
      <div className="flex h-64 items-center justify-center rounded-xl border border-border bg-surface">
        <p className="text-text-muted">Sem operações para mostrar</p>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h3 className="text-lg font-bold text-text-primary">{title}</h3>
          <p className="text-xs text-text-muted">
            {formatDate(timeWindow.from, 'full')} →{' '}
            {formatDate(timeWindow.to, 'full')}
          </p>
        </div>
        <RouteLegend />
      </div>

      {/* Controls */}
      <div className="flex flex-wrap items-center gap-4 rounded-lg border border-border bg-surface p-3">
        <ViewModeToggle mode={viewMode} onChange={setViewMode} />
        <div className="h-5 w-px bg-border" />
        <TimePresetButtons current={timePreset} onChange={handlePresetChange} />
        <div className="h-5 w-px bg-border" />
        <ZoomSlider value={zoomPercent} onChange={handleZoomChange} />
      </div>

      {/* Stats */}
      <div className="flex gap-4 text-xs text-text-muted">
        <span>{parsedOps.length} operações</span>
        <span>•</span>
        <span>
          {groupedOps.size} {viewMode === 'MACHINE' ? 'máquinas' : 'artigos'}
        </span>
      </div>

      {/* Gantt Chart */}
      <div
        ref={containerRef}
        className="overflow-hidden rounded-xl border border-border bg-surface"
      >
        {/* Time Axis Header */}
        <div className="flex">
          <div
            className="shrink-0 border-b border-r border-border bg-background/50 px-3 py-2 text-xs font-semibold text-text-primary"
            style={{ width: LABEL_WIDTH }}
          >
            {viewMode === 'MACHINE' ? 'Máquina' : 'Artigo'}
          </div>
          <div className="flex-1">
            <TimeAxis ticks={ticks} containerWidth={containerWidth} />
          </div>
        </div>

        {/* Rows */}
        <div className="max-h-[500px] overflow-y-auto">
          {[...groupedOps.entries()].map(([groupKey, ops], groupIdx) => {
            // Filter visible operations
            const visibleOps = ops.filter(op =>
              isOperationVisible(op, timeWindow)
            )
            
            // Calculate sub-rows for non-overlapping bars
            const subRowMap = calculateSubRows(visibleOps)
            const maxSubRow = Math.max(...Array.from(subRowMap.values()), 0)
            const rowCount = maxSubRow + 1
            const rowHeight = rowCount * ROW_HEIGHT + 8

            return (
              <div
                key={groupKey}
                className={`flex border-b border-border/50 ${
                  groupIdx % 2 === 0 ? 'bg-background/10' : ''
                }`}
                style={{ minHeight: rowHeight }}
              >
                {/* Row Label */}
                <div
                  className="shrink-0 border-r border-border/50 px-3 py-2"
                  style={{ width: LABEL_WIDTH }}
                >
                  <div
                    className={`text-sm font-semibold ${
                      visibleOps.length > 0
                        ? 'text-text-primary'
                        : 'text-text-muted/40'
                    }`}
                  >
                    {groupKey}
                  </div>
                  {ops.length > 0 && (
                    <div className="text-[10px] text-text-muted">
                      {ops.length} op{ops.length > 1 ? 's' : ''}
                    </div>
                  )}
                </div>

                {/* Operations Area */}
                <div className="relative flex-1" style={{ minHeight: rowHeight }}>
                  {/* Grid lines */}
                  {ticks.map((tick, i) => (
                    <div
                      key={i}
                      className="absolute bottom-0 top-0 w-px bg-border/20"
                      style={{ left: tick.x }}
                    />
                  ))}

                  {/* Operation bars */}
                  {visibleOps.map((op, opIdx) => {
                    const x = computeX(op.start, timeWindow, containerWidth)
                    const width = computeWidth(
                      op.start,
                      op.end,
                      timeWindow,
                      containerWidth
                    )
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
        {hoveredOp && (
          <Tooltip op={hoveredOp.op} x={hoveredOp.x} y={hoveredOp.y} />
        )}
      </AnimatePresence>
    </div>
  )
}
