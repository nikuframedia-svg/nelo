/**
 * Time Scale Utilities for Gantt/Timeline visualization
 * 
 * Provides precise time-to-pixel mapping for timeline components.
 */

// -------------------------
// Types
// -------------------------

export type TimeWindow = {
  from: Date
  to: Date
}

export type ParsedOperation = {
  order_id: string
  article_id: string
  route_id: string
  route_label: string
  op_seq: number
  op_code: string
  machine_id: string
  qty: number
  start: Date
  end: Date
  duration_min: number
}

// Raw plan operation from API
export type RawPlanOp = {
  order_id: string
  article_id: string
  route_id: string
  route_label: string
  op_seq: number
  op_code: string
  machine_id: string
  qty: number
  start_time: string
  end_time: string
  duration_min: number
}

// -------------------------
// Parsing Functions
// -------------------------

/**
 * Parse a single ISO date string to Date object
 */
export function parseDate(isoString: string): Date {
  const d = new Date(isoString)
  if (isNaN(d.getTime())) {
    console.warn(`Invalid date string: ${isoString}`)
    return new Date()
  }
  return d
}

/**
 * Parse raw plan operations into typed operations with Date objects
 */
export function parsePlanOperations(rawOps: RawPlanOp[]): ParsedOperation[] {
  return rawOps.map(op => ({
    order_id: op.order_id,
    article_id: op.article_id,
    route_id: op.route_id,
    route_label: op.route_label || 'A',
    op_seq: op.op_seq,
    op_code: op.op_code,
    machine_id: op.machine_id,
    qty: op.qty,
    start: parseDate(op.start_time),
    end: parseDate(op.end_time),
    duration_min: op.duration_min,
  }))
}

/**
 * Find global min/max times from parsed operations
 */
export function computeGlobalTimeRange(ops: ParsedOperation[]): TimeWindow {
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

  return {
    from: new Date(minTime),
    to: new Date(maxTime),
  }
}

// -------------------------
// Coordinate Mapping
// -------------------------

/**
 * Map a date to X pixel position within container
 * Clamps to [0, containerWidth]
 */
export function computeX(
  date: Date,
  window: TimeWindow,
  containerWidth: number
): number {
  const windowStart = window.from.getTime()
  const windowEnd = window.to.getTime()
  const windowDuration = windowEnd - windowStart

  if (windowDuration <= 0) return 0

  const dateTime = date.getTime()
  const ratio = (dateTime - windowStart) / windowDuration
  const x = ratio * containerWidth

  // Clamp to container bounds
  return Math.max(0, Math.min(containerWidth, x))
}

/**
 * Compute width in pixels for an operation bar
 * Handles partial visibility (clamps to window bounds)
 */
export function computeWidth(
  start: Date,
  end: Date,
  window: TimeWindow,
  containerWidth: number
): number {
  const windowStart = window.from.getTime()
  const windowEnd = window.to.getTime()
  const windowDuration = windowEnd - windowStart

  if (windowDuration <= 0) return 0

  // Clamp operation times to window
  const clampedStart = Math.max(start.getTime(), windowStart)
  const clampedEnd = Math.min(end.getTime(), windowEnd)

  if (clampedEnd <= clampedStart) return 0

  const visibleDuration = clampedEnd - clampedStart
  const width = (visibleDuration / windowDuration) * containerWidth

  // Minimum width of 4px for visibility
  return Math.max(4, width)
}

/**
 * Check if an operation is visible in the current time window
 */
export function isOperationVisible(
  op: ParsedOperation,
  window: TimeWindow
): boolean {
  const opStart = op.start.getTime()
  const opEnd = op.end.getTime()
  const windowStart = window.from.getTime()
  const windowEnd = window.to.getTime()

  // Visible if any part of the operation overlaps with the window
  return opEnd > windowStart && opStart < windowEnd
}

// -------------------------
// Time Window Presets
// -------------------------

export type TimePreset = '1day' | '3days' | '1week' | 'full'

/**
 * Create a time window from a preset, starting from globalMin
 */
export function createTimeWindow(
  preset: TimePreset,
  globalRange: TimeWindow
): TimeWindow {
  const from = globalRange.from

  switch (preset) {
    case '1day':
      return {
        from,
        to: new Date(from.getTime() + 24 * 60 * 60 * 1000),
      }
    case '3days':
      return {
        from,
        to: new Date(from.getTime() + 3 * 24 * 60 * 60 * 1000),
      }
    case '1week':
      return {
        from,
        to: new Date(from.getTime() + 7 * 24 * 60 * 60 * 1000),
      }
    case 'full':
    default:
      return { ...globalRange }
  }
}

/**
 * Create a time window from zoom slider value (0-100)
 */
export function createTimeWindowFromZoom(
  zoomPercent: number,
  globalRange: TimeWindow
): TimeWindow {
  const globalDuration = globalRange.to.getTime() - globalRange.from.getTime()
  
  // zoomPercent: 100 = full view, 10 = 10% of total duration
  const percent = Math.max(10, Math.min(100, zoomPercent))
  const visibleDuration = (percent / 100) * globalDuration

  return {
    from: globalRange.from,
    to: new Date(globalRange.from.getTime() + visibleDuration),
  }
}

// -------------------------
// Time Axis Ticks
// -------------------------

export type TickMark = {
  date: Date
  x: number
  label: string
  isMajor: boolean // True for day boundaries
}

/**
 * Generate tick marks for the time axis
 * Automatically adjusts interval based on window duration
 */
export function generateTimeAxisTicks(
  window: TimeWindow,
  containerWidth: number
): TickMark[] {
  const ticks: TickMark[] = []
  const windowDuration = window.to.getTime() - window.from.getTime()
  const windowHours = windowDuration / (60 * 60 * 1000)

  // Determine tick interval based on window size
  let intervalHours: number
  if (windowHours <= 24) {
    intervalHours = 2 // Every 2 hours for 1 day view
  } else if (windowHours <= 72) {
    intervalHours = 6 // Every 6 hours for 3 days
  } else if (windowHours <= 168) {
    intervalHours = 12 // Every 12 hours for 1 week
  } else {
    intervalHours = 24 // Every day for longer views
  }

  const intervalMs = intervalHours * 60 * 60 * 1000

  // Start from the beginning of the window, rounded to interval
  let current = new Date(window.from)
  current.setMinutes(0, 0, 0)
  
  // Round to nearest interval
  const hours = current.getHours()
  const roundedHours = Math.floor(hours / intervalHours) * intervalHours
  current.setHours(roundedHours)

  while (current <= window.to) {
    if (current >= window.from) {
      const x = computeX(current, window, containerWidth)
      const isMajor = current.getHours() === 0 // Midnight = major tick
      
      // Format label
      let label: string
      if (isMajor || intervalHours >= 24) {
        label = formatDate(current, 'short') // "24/11"
      } else {
        label = formatTime(current) // "14:00"
      }

      ticks.push({
        date: new Date(current),
        x,
        label,
        isMajor,
      })
    }

    current = new Date(current.getTime() + intervalMs)
  }

  return ticks
}

// -------------------------
// Formatting Helpers
// -------------------------

/**
 * Format date as "dd/MM" (pt-PT style)
 */
export function formatDate(date: Date, style: 'short' | 'full' = 'short'): string {
  const day = String(date.getDate()).padStart(2, '0')
  const month = String(date.getMonth() + 1).padStart(2, '0')
  
  if (style === 'full') {
    const hours = String(date.getHours()).padStart(2, '0')
    const mins = String(date.getMinutes()).padStart(2, '0')
    return `${day}/${month} ${hours}:${mins}`
  }
  
  return `${day}/${month}`
}

/**
 * Format time as "HH:mm"
 */
export function formatTime(date: Date): string {
  const hours = String(date.getHours()).padStart(2, '0')
  const mins = String(date.getMinutes()).padStart(2, '0')
  return `${hours}:${mins}`
}

/**
 * Format duration in human readable form
 */
export function formatDuration(minutes: number): string {
  if (minutes < 60) {
    return `${Math.round(minutes)}min`
  }
  const hours = minutes / 60
  if (hours < 24) {
    return `${hours.toFixed(1)}h`
  }
  const days = hours / 24
  return `${days.toFixed(1)}d`
}

// -------------------------
// Grouping Utilities
// -------------------------

export type ViewMode = 'MACHINE' | 'PRODUCT'

/**
 * Group operations by a key (machine_id or article_id)
 */
export function groupOperations(
  ops: ParsedOperation[],
  mode: ViewMode
): Map<string, ParsedOperation[]> {
  const groups = new Map<string, ParsedOperation[]>()
  const key = mode === 'MACHINE' ? 'machine_id' : 'article_id'

  for (const op of ops) {
    const groupKey = op[key]
    if (!groups.has(groupKey)) {
      groups.set(groupKey, [])
    }
    groups.get(groupKey)!.push(op)
  }

  // Sort operations within each group by start time
  for (const [, groupOps] of groups) {
    groupOps.sort((a, b) => a.start.getTime() - b.start.getTime())
  }

  // Sort groups by key name
  return new Map([...groups.entries()].sort((a, b) => a[0].localeCompare(b[0])))
}

/**
 * Calculate sub-rows for operations within a group to avoid overlap
 */
export function calculateSubRows(ops: ParsedOperation[]): Map<ParsedOperation, number> {
  const rowAssignments = new Map<ParsedOperation, number>()
  const rows: ParsedOperation[][] = []

  for (const op of ops) {
    let placed = false
    
    for (let rowIdx = 0; rowIdx < rows.length; rowIdx++) {
      const hasCollision = rows[rowIdx].some(existing => {
        // Check if time ranges overlap
        return !(op.end.getTime() <= existing.start.getTime() || 
                 op.start.getTime() >= existing.end.getTime())
      })
      
      if (!hasCollision) {
        rows[rowIdx].push(op)
        rowAssignments.set(op, rowIdx)
        placed = true
        break
      }
    }
    
    if (!placed) {
      rows.push([op])
      rowAssignments.set(op, rows.length - 1)
    }
  }

  return rowAssignments
}

// -------------------------
// Color Utilities
// -------------------------

/**
 * Get color for a route label
 */
export function getRouteColor(routeLabel: string): string {
  switch (routeLabel?.toUpperCase()) {
    case 'A': return '#3B82F6' // Blue
    case 'B': return '#F59E0B' // Amber/Orange
    case 'C': return '#10B981' // Green
    case 'D': return '#8B5CF6' // Purple
    case 'E': return '#EC4899' // Pink
    default: return '#6B7280' // Gray
  }
}



