import { API_BASE_URL } from '../config/api'
import type { ScenarioPreviewData } from '../components/ScenarioPreview'
import type { ScenarioComparisonData } from '../components/ScenarioComparison'

// -------------------------
// Types
// -------------------------

export type PlanOp = {
  order_id: string
  article_id: string
  route_id: string
  route_label: string
  op_seq: number
  op_code: string
  machine_id: string
  qty: number
  start_time: string // ISO
  end_time: string // ISO
  duration_min: number
}

export type BottleneckInfo = {
  machine_id: string
  total_minutes: number
}

export type MachineLoad = {
  machine_id: string
  load_min: number
  load_hours: number
  idle_min: number
  idle_hours: number
  utilization_pct: number
  num_operations: number
}

export type PlanKPIs = {
  makespan_hours: number
  route_distribution: Record<string, number>
  overlaps: {
    total: number
    by_machine: Record<string, number>
  }
  active_bottleneck: BottleneckInfo | null
  machine_loads: MachineLoad[]
  lead_time_average_h: number
  otd_percent: number
  setup_hours: number
  total_operations: number
  total_orders: number
  total_articles: number
  total_machines: number
  plan_start: string | null
  plan_end: string | null
}

// Suggestion Types
export type OverloadSuggestion = {
  type: 'overload_reduction'
  machine: string
  candidate_op: string
  order_id: string
  article_id: string
  duration_min: number
  alternative_machine: string | null
  expected_gain_h: number
  reason: string
  formatted_pt: string
}

export type IdleGapSuggestion = {
  type: 'idle_gap'
  machine: string
  gap_start: string
  gap_end: string
  gap_min: number
  reason: string
  formatted_pt: string
}

export type ProductRiskSuggestion = {
  type: 'product_risk'
  article: string
  bottleneck_op: string
  wait_min: number
  from_op: string
  to_op: string
  reason: string
  formatted_pt: string
}

export type PlanSuggestions = {
  overload_suggestions: OverloadSuggestion[]
  idle_gaps: IdleGapSuggestion[]
  product_risks: ProductRiskSuggestion[]
  machine_loads: Record<string, {
    total_min: number
    num_ops: number
    idle_min: number
    span_min: number
    utilization_pct: number
  }>
  summary: {
    total_suggestions: number
    high_priority: number
    overload_count: number
    idle_gap_count: number
    product_risk_count: number
  }
}

// -------------------------
// Helpers
// -------------------------

async function handleResponse<T>(response: Response, endpoint: string): Promise<T> {
  if (!response.ok) {
    throw new Error(`Erro no ${endpoint}: ${response.status}`)
  }
  return response.json()
}

// -------------------------
// API Functions
// -------------------------

export async function apiGetPlan(): Promise<PlanOp[]> {
  const res = await fetch(`${API_BASE_URL}/plan`)
  return handleResponse<PlanOp[]>(res, '/plan')
}

export async function apiGetPlanKPIs(): Promise<PlanKPIs> {
  const res = await fetch(`${API_BASE_URL}/plan/kpis`)
  return handleResponse<PlanKPIs>(res, '/plan/kpis')
}

export async function apiGetPlanSuggestions(): Promise<PlanSuggestions> {
  const res = await fetch(`${API_BASE_URL}/plan/suggestions`)
  return handleResponse<PlanSuggestions>(res, '/plan/suggestions')
}

export async function apiGetBottleneck(): Promise<BottleneckInfo> {
  const res = await fetch(`${API_BASE_URL}/bottleneck`)
  return handleResponse<BottleneckInfo>(res, '/bottleneck')
}

export async function apiDescribeScenario(scenario: string): Promise<ScenarioPreviewData> {
  const res = await fetch(`${API_BASE_URL}/what-if/describe`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ scenario }),
  })
  return handleResponse<ScenarioPreviewData>(res, '/what-if/describe')
}

export async function apiCompareScenario(scenario: string): Promise<ScenarioComparisonData> {
  const res = await fetch(`${API_BASE_URL}/what-if/compare`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ scenario }),
  })
  return handleResponse<ScenarioComparisonData>(res, '/what-if/compare')
}

// -------------------------
// Actions / Approval Queue Types
// -------------------------

export type ActionType =
  | 'SET_MACHINE_DOWN'
  | 'SET_MACHINE_UP'
  | 'CHANGE_ROUTE'
  | 'MOVE_OPERATION'
  | 'SET_VIP_ARTICLE'
  | 'CHANGE_HORIZON'
  | 'ADD_OVERTIME'
  | 'ADD_ORDER'

export type ActionStatus = 'PENDING' | 'APPROVED' | 'REJECTED' | 'APPLIED'

export type Action = {
  id: string
  type: ActionType
  payload: Record<string, unknown>
  source: string
  created_at: string
  status: ActionStatus
  approved_by: string | null
  approved_at: string | null
  applied_at: string | null
  notes: string | null
  description: string | null
  expected_impact: Record<string, unknown> | null
}

export type ActionsResponse = {
  count: number
  actions: Action[]
  pending_count: number
}

export type ProposeActionRequest = {
  type: ActionType
  payload: Record<string, unknown>
  source?: string
  description?: string
}

// -------------------------
// Actions API Functions
// -------------------------

export async function apiGetActions(status?: ActionStatus): Promise<ActionsResponse> {
  const url = status
    ? `${API_BASE_URL}/actions?status=${status}`
    : `${API_BASE_URL}/actions`
  const res = await fetch(url)
  return handleResponse<ActionsResponse>(res, '/actions')
}

export async function apiGetPendingActionsCount(): Promise<{ count: number }> {
  const res = await fetch(`${API_BASE_URL}/actions/pending/count`)
  return handleResponse<{ count: number }>(res, '/actions/pending/count')
}

export async function apiProposeAction(request: ProposeActionRequest): Promise<{ message: string; action: Action }> {
  const res = await fetch(`${API_BASE_URL}/actions/propose`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  })
  return handleResponse<{ message: string; action: Action }>(res, '/actions/propose')
}

export async function apiApproveAction(
  actionId: string,
  approvedBy: string,
  notes?: string
): Promise<{ message: string; result: unknown }> {
  const res = await fetch(`${API_BASE_URL}/actions/${actionId}/approve`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ approved_by: approvedBy, notes }),
  })
  return handleResponse<{ message: string; result: unknown }>(res, `/actions/${actionId}/approve`)
}

export async function apiRejectAction(
  actionId: string,
  rejectedBy: string,
  reason?: string
): Promise<{ message: string; action: Action }> {
  const res = await fetch(`${API_BASE_URL}/actions/${actionId}/reject`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ rejected_by: rejectedBy, reason }),
  })
  return handleResponse<{ message: string; action: Action }>(res, `/actions/${actionId}/reject`)
}

export async function apiCreateActionFromSuggestion(
  suggestion: Record<string, unknown>
): Promise<{ message: string; action: Action }> {
  const res = await fetch(`${API_BASE_URL}/actions/from-suggestion`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(suggestion),
  })
  return handleResponse<{ message: string; action: Action }>(res, '/actions/from-suggestion')
}

