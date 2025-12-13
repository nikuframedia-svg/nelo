/**
 * Timeline utilities for production planning visualization.
 * Handles parsing, normalization, and grouping of plan operations.
 */

import type { PlanOp } from '../services/nikufraApi'

// -------------------------
// Types
// -------------------------

export type ParsedPlanOp = PlanOp & {
  startDate: Date
  endDate: Date
}

export type TimeRange = {
  min: Date
  max: Date
}

export type ViewMode = 'MACHINE' | 'PRODUCT' | 'ROUTE'

// -------------------------
// Parsing & Normalization
// -------------------------

/**
 * Parse plan operations, converting ISO strings to Date objects.
 */
export function parsePlanOperations(ops: PlanOp[]): ParsedPlanOp[] {
  return ops.map((op) => ({
    ...op,
    startDate: new Date(op.start_time),
    endDate: new Date(op.end_time),
  }))
}

/**
 * Compute global min and max time from all operations.
 */
export function computeTimeRange(ops: ParsedPlanOp[]): TimeRange {
  if (ops.length === 0) {
    const now = new Date()
    return { min: now, max: new Date(now.getTime() + 24 * 60 * 60 * 1000) }
  }

  let min = ops[0].startDate
  let max = ops[0].endDate

  for (const op of ops) {
    if (op.startDate < min) min = op.startDate
    if (op.endDate > max) max = op.endDate
  }

  return { min, max }
}

/**
 * Compute time window based on preset or custom range.
 */
export function computeTimeWindow(
  globalRange: TimeRange,
  preset: '1day' | '3days' | '1week' | 'full' | 'custom',
  customDays?: number
): TimeRange {
  const { min, max } = globalRange

  switch (preset) {
    case '1day':
      return { min, max: new Date(min.getTime() + 1 * 24 * 60 * 60 * 1000) }
    case '3days':
      return { min, max: new Date(min.getTime() + 3 * 24 * 60 * 60 * 1000) }
    case '1week':
      return { min, max: new Date(min.getTime() + 7 * 24 * 60 * 60 * 1000) }
    case 'custom':
      if (customDays) {
        return { min, max: new Date(min.getTime() + customDays * 24 * 60 * 60 * 1000) }
      }
      return globalRange
    case 'full':
    default:
      return globalRange
  }
}

// -------------------------
// Grouping Functions
// -------------------------

/**
 * Group operations by machine_id.
 */
export function groupByMachine(ops: ParsedPlanOp[]): Map<string, ParsedPlanOp[]> {
  const map = new Map<string, ParsedPlanOp[]>()
  for (const op of ops) {
    const key = op.machine_id
    if (!map.has(key)) map.set(key, [])
    map.get(key)!.push(op)
  }
  // Sort keys alphabetically
  return new Map([...map.entries()].sort((a, b) => a[0].localeCompare(b[0])))
}

/**
 * Group operations by article_id.
 */
export function groupByProduct(ops: ParsedPlanOp[]): Map<string, ParsedPlanOp[]> {
  const map = new Map<string, ParsedPlanOp[]>()
  for (const op of ops) {
    const key = op.article_id
    if (!map.has(key)) map.set(key, [])
    map.get(key)!.push(op)
  }
  return new Map([...map.entries()].sort((a, b) => a[0].localeCompare(b[0])))
}

/**
 * Group operations by route_label.
 */
export function groupByRoute(ops: ParsedPlanOp[]): Map<string, ParsedPlanOp[]> {
  const map = new Map<string, ParsedPlanOp[]>()
  for (const op of ops) {
    const key = op.route_label || 'Sem rota'
    if (!map.has(key)) map.set(key, [])
    map.get(key)!.push(op)
  }
  return new Map([...map.entries()].sort((a, b) => a[0].localeCompare(b[0])))
}

/**
 * Get grouped operations based on view mode.
 */
export function getGroupedOperations(
  ops: ParsedPlanOp[],
  viewMode: ViewMode
): Map<string, ParsedPlanOp[]> {
  switch (viewMode) {
    case 'MACHINE':
      return groupByMachine(ops)
    case 'PRODUCT':
      return groupByProduct(ops)
    case 'ROUTE':
      return groupByRoute(ops)
  }
}

// -------------------------
// Display Helpers
// -------------------------

/**
 * Get color for a route label.
 */
export function getRouteColor(routeLabel: string): string {
  switch (routeLabel?.toUpperCase()) {
    case 'A':
      return '#3B82F6' // blue
    case 'B':
      return '#F59E0B' // amber/orange
    case 'C':
      return '#10B981' // green
    case 'D':
      return '#8B5CF6' // purple
    case 'E':
      return '#EF4444' // red
    default:
      return '#6B7280' // gray
  }
}

/**
 * Format date for display in timeline axis.
 */
export function formatTimeAxisLabel(date: Date): string {
  return date.toLocaleTimeString('pt-PT', { hour: '2-digit', minute: '2-digit' })
}

/**
 * Format date for day header.
 */
export function formatDayLabel(date: Date): string {
  return date.toLocaleDateString('pt-PT', { weekday: 'short', day: 'numeric', month: 'short' })
}

/**
 * Get unique values from operations for filter options.
 */
export function getUniqueValues(ops: ParsedPlanOp[], field: keyof ParsedPlanOp): string[] {
  const values = new Set<string>()
  for (const op of ops) {
    const val = op[field]
    if (typeof val === 'string') values.add(val)
  }
  return [...values].sort()
}

/**
 * Filter operations by selected values.
 */
export function filterOperations(
  ops: ParsedPlanOp[],
  viewMode: ViewMode,
  selectedValues: string[]
): ParsedPlanOp[] {
  if (selectedValues.length === 0) return ops

  return ops.filter((op) => {
    switch (viewMode) {
      case 'MACHINE':
        return selectedValues.includes(op.machine_id)
      case 'PRODUCT':
        return selectedValues.includes(op.article_id)
      case 'ROUTE':
        return selectedValues.includes(op.route_label)
    }
  })
}



