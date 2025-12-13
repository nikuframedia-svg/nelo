/**
 * ════════════════════════════════════════════════════════════════════════════════════════════════════
 * SHOPFLOOR APP - Operator Interface
 * ════════════════════════════════════════════════════════════════════════════════════════════════════
 *
 * Contract 8 Implementation: Shopfloor App
 *
 * Features:
 * - Machine/Workstation selection
 * - Order queue with execution controls (Start, Pause, Complete)
 * - Work Instructions display
 * - Quality checkpoints and reporting
 * - Downtime tracking
 */

import React, { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Factory,
  Play,
  Pause,
  CheckCircle,
  AlertTriangle,
  Clock,
  Package,
  FileText,
  ChevronRight,
  RefreshCw,
  Plus,
  Minus,
  Wrench,
  X,
  CheckSquare,
  Square,
} from 'lucide-react'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'

// ═══════════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════════

interface Machine {
  id: string
  name: string
  type: string
  status: 'running' | 'idle' | 'maintenance'
}

interface ShopfloorOrder {
  id: string
  article_id: string
  article_name: string
  operation_code: string
  machine_id: string
  planned_qty: number
  good_qty: number
  scrap_qty: number
  status: 'pending' | 'in_progress' | 'paused' | 'completed' | 'cancelled'
  planned_start: string
  planned_end: string
  actual_start: string | null
  actual_end: string | null
  priority: number
  has_work_instructions: boolean
}

interface WorkInstruction {
  id: number
  title: string
  version: string
  steps: WorkInstructionStep[]
  total_estimated_time: number
  tools_required: string[]
  materials_required: string[]
  safety_equipment: string[]
}

interface WorkInstructionStep {
  step_number: number
  title: string
  description: string
  image_url: string | null
  duration_minutes: number | null
  checkpoints: QualityCheckpoint[]
  safety_warning: string | null
}

interface QualityCheckpoint {
  id: string
  label: string
  type: 'checkbox' | 'numeric' | 'text'
  required: boolean
  unit: string | null
  min_value: number | null
  max_value: number | null
}

// ═══════════════════════════════════════════════════════════════════════════════
// API CALLS
// ═══════════════════════════════════════════════════════════════════════════════

async function fetchMachines(): Promise<Machine[]> {
  const res = await fetch(`${API_BASE_URL}/shopfloor/machines`)
  if (!res.ok) return []
  return res.json()
}

async function fetchOrders(machineId: string | null): Promise<ShopfloorOrder[]> {
  const url = machineId
    ? `${API_BASE_URL}/shopfloor/orders?machine_id=${machineId}`
    : `${API_BASE_URL}/shopfloor/orders`
  const res = await fetch(url)
  if (!res.ok) return []
  return res.json()
}

async function fetchWorkInstructions(revisionId: number, operationId: number): Promise<WorkInstruction | null> {
  const res = await fetch(`${API_BASE_URL}/work-instructions/${revisionId}/${operationId}`)
  if (!res.ok) return null
  return res.json()
}

async function startOrder(orderId: string, machineId: string): Promise<any> {
  const res = await fetch(`${API_BASE_URL}/shopfloor/orders/${orderId}/start?machine_id=${machineId}`, {
    method: 'POST',
  })
  if (!res.ok) throw new Error('Failed to start order')
  return res.json()
}

async function pauseOrder(orderId: string, reason?: string): Promise<any> {
  const url = reason
    ? `${API_BASE_URL}/shopfloor/orders/${orderId}/pause?reason=${encodeURIComponent(reason)}`
    : `${API_BASE_URL}/shopfloor/orders/${orderId}/pause`
  const res = await fetch(url, { method: 'POST' })
  if (!res.ok) throw new Error('Failed to pause order')
  return res.json()
}

async function completeOrder(orderId: string, machineId: string): Promise<any> {
  const res = await fetch(`${API_BASE_URL}/shopfloor/orders/${orderId}/complete?machine_id=${machineId}`, {
    method: 'POST',
  })
  if (!res.ok) throw new Error('Failed to complete order')
  return res.json()
}

async function reportExecution(
  orderId: string,
  machineId: string,
  data: {
    good_qty: number
    scrap_qty: number
    downtime_reason?: string
    downtime_minutes?: number
    notes?: string
  }
): Promise<any> {
  const params = new URLSearchParams({
    machine_id: machineId,
    good_qty: data.good_qty.toString(),
    scrap_qty: data.scrap_qty.toString(),
  })
  if (data.downtime_reason) params.set('downtime_reason', data.downtime_reason)
  if (data.downtime_minutes) params.set('downtime_minutes', data.downtime_minutes.toString())
  if (data.notes) params.set('notes', data.notes)

  const res = await fetch(`${API_BASE_URL}/shopfloor/orders/${orderId}/report?${params}`, {
    method: 'POST',
  })
  if (!res.ok) throw new Error('Failed to report execution')
  return res.json()
}

// ═══════════════════════════════════════════════════════════════════════════════
// COMPONENTS
// ═══════════════════════════════════════════════════════════════════════════════

const StatusBadge: React.FC<{ status: string }> = ({ status }) => {
  const config: Record<string, { bg: string; text: string; icon: typeof Clock }> = {
    pending: { bg: 'bg-slate-500/20', text: 'text-slate-400', icon: Clock },
    in_progress: { bg: 'bg-cyan-500/20', text: 'text-cyan-400', icon: Play },
    paused: { bg: 'bg-amber-500/20', text: 'text-amber-400', icon: Pause },
    completed: { bg: 'bg-emerald-500/20', text: 'text-emerald-400', icon: CheckCircle },
    cancelled: { bg: 'bg-red-500/20', text: 'text-red-400', icon: X },
  }
  const c = config[status] || config.pending
  const Icon = c.icon

  const labels: Record<string, string> = {
    pending: 'Pendente',
    in_progress: 'Em Execução',
    paused: 'Pausado',
    completed: 'Concluído',
    cancelled: 'Cancelado',
  }

  return (
    <span className={`inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium ${c.bg} ${c.text}`}>
      <Icon className="w-3 h-3" />
      {labels[status] || status}
    </span>
  )
}

const MachineCard: React.FC<{
  machine: Machine
  isSelected: boolean
  onClick: () => void
}> = ({ machine, isSelected, onClick }) => {
  const statusColors: Record<string, string> = {
    running: 'bg-emerald-500',
    idle: 'bg-amber-500',
    maintenance: 'bg-red-500',
  }

  return (
    <motion.div
      whileHover={{ scale: 1.02 }}
      onClick={onClick}
      className={`cursor-pointer p-4 rounded-xl border transition-all ${
        isSelected
          ? 'border-cyan-500 bg-cyan-500/10'
          : 'border-slate-700/50 bg-slate-800/30 hover:border-cyan-500/50'
      }`}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Factory className={`w-5 h-5 ${isSelected ? 'text-cyan-400' : 'text-slate-400'}`} />
          <div>
            <p className="font-semibold text-white">{machine.name}</p>
            <p className="text-xs text-slate-400">{machine.type}</p>
          </div>
        </div>
        <div className={`w-3 h-3 rounded-full ${statusColors[machine.status] || 'bg-slate-500'}`} />
      </div>
    </motion.div>
  )
}

const OrderCard: React.FC<{
  order: ShopfloorOrder
  isSelected: boolean
  onClick: () => void
}> = ({ order, isSelected, onClick }) => {
  const progress = order.planned_qty > 0 ? ((order.good_qty + order.scrap_qty) / order.planned_qty) * 100 : 0

  return (
    <motion.div
      whileHover={{ scale: 1.01 }}
      onClick={onClick}
      className={`cursor-pointer p-4 rounded-xl border transition-all ${
        isSelected
          ? 'border-cyan-500 bg-cyan-500/10'
          : 'border-slate-700/50 bg-slate-800/30 hover:border-cyan-500/50'
      }`}
    >
      <div className="flex items-start justify-between mb-3">
        <div>
          <p className="font-semibold text-white">{order.id}</p>
          <p className="text-sm text-slate-400">{order.article_name}</p>
        </div>
        <StatusBadge status={order.status} />
      </div>

      <div className="grid grid-cols-3 gap-2 text-xs mb-3">
        <div>
          <p className="text-slate-500">Operação</p>
          <p className="text-white font-medium">{order.operation_code}</p>
        </div>
        <div>
          <p className="text-slate-500">Qtd. Plan.</p>
          <p className="text-white font-medium">{order.planned_qty}</p>
        </div>
        <div>
          <p className="text-slate-500">OK / Sucata</p>
          <p className="text-white font-medium">
            <span className="text-emerald-400">{order.good_qty}</span>
            {' / '}
            <span className="text-red-400">{order.scrap_qty}</span>
          </p>
        </div>
      </div>

      {/* Progress bar */}
      <div className="h-1.5 bg-slate-700 rounded-full overflow-hidden">
        <div
          className="h-full bg-gradient-to-r from-cyan-500 to-emerald-500 transition-all"
          style={{ width: `${Math.min(100, progress)}%` }}
        />
      </div>

      <div className="flex items-center justify-between mt-2 text-xs text-slate-500">
        <span>{progress.toFixed(0)}% completo</span>
        <span className="flex items-center gap-1">
          {order.has_work_instructions && <FileText className="w-3 h-3" />}
          Prioridade: {order.priority}
        </span>
      </div>
    </motion.div>
  )
}

const WorkInstructionView: React.FC<{
  instruction: WorkInstruction
}> = ({ instruction }) => {
  const [expandedStep, setExpandedStep] = useState<number | null>(null)

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-semibold text-white">{instruction.title}</h3>
          <p className="text-xs text-slate-400">v{instruction.version} • {instruction.total_estimated_time} min estimado</p>
        </div>
      </div>

      {/* Tools & Safety */}
      <div className="grid grid-cols-2 gap-3">
        <div className="p-3 rounded-lg bg-slate-800/50 border border-slate-700/50">
          <p className="text-xs text-slate-500 mb-1">🔧 Ferramentas</p>
          <div className="flex flex-wrap gap-1">
            {instruction.tools_required.map((t, i) => (
              <span key={i} className="px-2 py-0.5 bg-slate-700 rounded text-xs text-slate-300">{t}</span>
            ))}
          </div>
        </div>
        <div className="p-3 rounded-lg bg-slate-800/50 border border-slate-700/50">
          <p className="text-xs text-slate-500 mb-1">🛡️ EPI</p>
          <div className="flex flex-wrap gap-1">
            {instruction.safety_equipment.map((s, i) => (
              <span key={i} className="px-2 py-0.5 bg-amber-500/20 rounded text-xs text-amber-400">{s}</span>
            ))}
          </div>
        </div>
      </div>

      {/* Steps */}
      <div className="space-y-2">
        {instruction.steps.map((step) => (
          <div
            key={step.step_number}
            className="rounded-lg border border-slate-700/50 overflow-hidden"
          >
            <div
              className="flex items-center gap-3 p-3 cursor-pointer hover:bg-slate-800/50 transition"
              onClick={() => setExpandedStep(expandedStep === step.step_number ? null : step.step_number)}
            >
              <div className="w-8 h-8 rounded-full bg-cyan-500/20 flex items-center justify-center text-cyan-400 font-bold">
                {step.step_number}
              </div>
              <div className="flex-1">
                <p className="font-medium text-white">{step.title}</p>
                {step.duration_minutes && (
                  <p className="text-xs text-slate-500">{step.duration_minutes} min</p>
                )}
              </div>
              <ChevronRight
                className={`w-4 h-4 text-slate-400 transition-transform ${
                  expandedStep === step.step_number ? 'rotate-90' : ''
                }`}
              />
            </div>

            <AnimatePresence>
              {expandedStep === step.step_number && (
                <motion.div
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: 'auto', opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  className="border-t border-slate-700/50"
                >
                  <div className="p-4 space-y-3 bg-slate-900/50">
                    <p className="text-sm text-slate-300">{step.description}</p>

                    {step.safety_warning && (
                      <div className="flex items-start gap-2 p-2 rounded bg-amber-500/10 border border-amber-500/30">
                        <AlertTriangle className="w-4 h-4 text-amber-400 mt-0.5" />
                        <p className="text-sm text-amber-400">{step.safety_warning}</p>
                      </div>
                    )}

                    {step.checkpoints.length > 0 && (
                      <div className="space-y-2">
                        <p className="text-xs text-slate-500 uppercase">Verificações</p>
                        {step.checkpoints.map((cp) => (
                          <div key={cp.id} className="flex items-center gap-3 p-2 rounded bg-slate-800/50">
                            {cp.type === 'checkbox' ? (
                              <Square className="w-5 h-5 text-cyan-400" />
                            ) : (
                              <div className="w-5 h-5 flex items-center justify-center text-cyan-400 text-xs">
                                {cp.type === 'numeric' ? '#' : 'T'}
                              </div>
                            )}
                            <span className="text-sm text-white">{cp.label}</span>
                            {cp.unit && <span className="text-xs text-slate-500">({cp.unit})</span>}
                            {cp.required && <span className="text-xs text-red-400">*</span>}
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        ))}
      </div>
    </div>
  )
}

const ReportForm: React.FC<{
  order: ShopfloorOrder
  machineId: string
  onSubmit: (data: any) => void
  isPending: boolean
}> = ({ order, machineId, onSubmit, isPending }) => {
  const [goodQty, setGoodQty] = useState(0)
  const [scrapQty, setScrapQty] = useState(0)
  const [downtimeReason, setDowntimeReason] = useState('')
  const [downtimeMinutes, setDowntimeMinutes] = useState(0)
  const [notes, setNotes] = useState('')

  const downtimeReasons = [
    { value: '', label: 'Sem paragem' },
    { value: 'machine_failure', label: 'Avaria máquina' },
    { value: 'material_shortage', label: 'Falta de material' },
    { value: 'tool_change', label: 'Troca ferramenta' },
    { value: 'quality_issue', label: 'Problema qualidade' },
    { value: 'maintenance', label: 'Manutenção' },
    { value: 'setup', label: 'Setup' },
    { value: 'other', label: 'Outro' },
  ]

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    onSubmit({
      good_qty: goodQty,
      scrap_qty: scrapQty,
      downtime_reason: downtimeReason || undefined,
      downtime_minutes: downtimeMinutes || undefined,
      notes: notes || undefined,
    })
  }

  const remaining = order.planned_qty - order.good_qty - order.scrap_qty

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <h4 className="text-sm font-semibold text-white">Reportar Produção</h4>

      {/* Quantities */}
      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="text-xs text-slate-500 block mb-1">Peças OK</label>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => setGoodQty(Math.max(0, goodQty - 1))}
              className="p-2 rounded bg-slate-700 hover:bg-slate-600 text-white"
            >
              <Minus className="w-4 h-4" />
            </button>
            <input
              type="number"
              value={goodQty}
              onChange={(e) => setGoodQty(Number(e.target.value))}
              className="flex-1 px-3 py-2 bg-slate-800 border border-slate-700 rounded text-center text-white"
            />
            <button
              type="button"
              onClick={() => setGoodQty(Math.min(remaining, goodQty + 1))}
              className="p-2 rounded bg-emerald-600 hover:bg-emerald-500 text-white"
            >
              <Plus className="w-4 h-4" />
            </button>
          </div>
        </div>

        <div>
          <label className="text-xs text-slate-500 block mb-1">Sucata</label>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => setScrapQty(Math.max(0, scrapQty - 1))}
              className="p-2 rounded bg-slate-700 hover:bg-slate-600 text-white"
            >
              <Minus className="w-4 h-4" />
            </button>
            <input
              type="number"
              value={scrapQty}
              onChange={(e) => setScrapQty(Number(e.target.value))}
              className="flex-1 px-3 py-2 bg-slate-800 border border-slate-700 rounded text-center text-white"
            />
            <button
              type="button"
              onClick={() => setScrapQty(scrapQty + 1)}
              className="p-2 rounded bg-red-600 hover:bg-red-500 text-white"
            >
              <Plus className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>

      <p className="text-xs text-slate-500">Restam: {remaining} peças</p>

      {/* Downtime */}
      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="text-xs text-slate-500 block mb-1">Razão Paragem</label>
          <select
            value={downtimeReason}
            onChange={(e) => setDowntimeReason(e.target.value)}
            className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded text-white text-sm"
          >
            {downtimeReasons.map((r) => (
              <option key={r.value} value={r.value}>{r.label}</option>
            ))}
          </select>
        </div>

        <div>
          <label className="text-xs text-slate-500 block mb-1">Tempo Paragem (min)</label>
          <input
            type="number"
            value={downtimeMinutes}
            onChange={(e) => setDowntimeMinutes(Number(e.target.value))}
            disabled={!downtimeReason}
            className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded text-white text-sm disabled:opacity-50"
          />
        </div>
      </div>

      {/* Notes */}
      <div>
        <label className="text-xs text-slate-500 block mb-1">Observações</label>
        <textarea
          value={notes}
          onChange={(e) => setNotes(e.target.value)}
          className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded text-white text-sm h-20 resize-none"
          placeholder="Notas adicionais..."
        />
      </div>

      <button
        type="submit"
        disabled={isPending || (goodQty === 0 && scrapQty === 0)}
        className="w-full py-3 bg-cyan-600 hover:bg-cyan-500 text-white font-semibold rounded-lg transition disabled:opacity-50"
      >
        {isPending ? (
          <span className="flex items-center justify-center gap-2">
            <RefreshCw className="w-4 h-4 animate-spin" />
            A submeter...
          </span>
        ) : (
          'Submeter Registo'
        )}
      </button>
    </form>
  )
}

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN COMPONENT
// ═══════════════════════════════════════════════════════════════════════════════

const Shopfloor: React.FC = () => {
  const queryClient = useQueryClient()
  const [selectedMachine, setSelectedMachine] = useState<string | null>(null)
  const [selectedOrder, setSelectedOrder] = useState<ShopfloorOrder | null>(null)
  const [showInstructions, setShowInstructions] = useState(false)

  // Queries
  const { data: machines, isLoading: loadingMachines } = useQuery({
    queryKey: ['shopfloor-machines'],
    queryFn: fetchMachines,
  })

  const { data: orders, isLoading: loadingOrders } = useQuery({
    queryKey: ['shopfloor-orders', selectedMachine],
    queryFn: () => fetchOrders(selectedMachine),
    enabled: true,
  })

  const { data: workInstructions } = useQuery({
    queryKey: ['work-instructions', selectedOrder?.id],
    queryFn: () => fetchWorkInstructions(1, 1), // Demo: revision_id=1, operation_id=1
    enabled: !!selectedOrder && showInstructions,
  })

  // Mutations
  const startMutation = useMutation({
    mutationFn: () => startOrder(selectedOrder!.id, selectedMachine!),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['shopfloor-orders'] })
    },
  })

  const pauseMutation = useMutation({
    mutationFn: () => pauseOrder(selectedOrder!.id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['shopfloor-orders'] })
    },
  })

  const completeMutation = useMutation({
    mutationFn: () => completeOrder(selectedOrder!.id, selectedMachine!),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['shopfloor-orders'] })
      setSelectedOrder(null)
    },
  })

  const reportMutation = useMutation({
    mutationFn: (data: any) => reportExecution(selectedOrder!.id, selectedMachine!, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['shopfloor-orders'] })
    },
  })

  return (
    <div className="space-y-6">
      {/* Header */}
      <header className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-gradient-to-br from-orange-500/20 to-red-500/20 rounded-xl border border-orange-500/30">
            <Factory className="w-6 h-6 text-orange-400" />
          </div>
          <div>
            <p className="text-xs uppercase tracking-[0.4em] text-slate-500">Prodplan</p>
            <h2 className="text-2xl font-semibold text-white">Shopfloor</h2>
          </div>
        </div>
        {selectedMachine && (
          <div className="flex items-center gap-2 px-3 py-2 bg-cyan-500/10 border border-cyan-500/30 rounded-lg">
            <Factory className="w-4 h-4 text-cyan-400" />
            <span className="text-sm text-cyan-400">{selectedMachine}</span>
          </div>
        )}
      </header>

      <div className="grid grid-cols-12 gap-6">
        {/* Machines - Left Panel */}
        <div className="col-span-3 space-y-4">
          <h3 className="text-sm font-medium text-white flex items-center gap-2">
            <Factory className="w-4 h-4 text-orange-400" />
            Máquinas
          </h3>
          
          {loadingMachines ? (
            <div className="flex justify-center py-8">
              <RefreshCw className="w-6 h-6 text-orange-400 animate-spin" />
            </div>
          ) : (
            <div className="space-y-2">
              {machines?.map((machine) => (
                <MachineCard
                  key={machine.id}
                  machine={machine}
                  isSelected={selectedMachine === machine.id}
                  onClick={() => {
                    setSelectedMachine(machine.id)
                    setSelectedOrder(null)
                  }}
                />
              ))}
            </div>
          )}
        </div>

        {/* Orders - Middle Panel */}
        <div className="col-span-4 space-y-4">
          <h3 className="text-sm font-medium text-white flex items-center gap-2">
            <Package className="w-4 h-4 text-cyan-400" />
            Ordens
          </h3>

          {loadingOrders ? (
            <div className="flex justify-center py-8">
              <RefreshCw className="w-6 h-6 text-cyan-400 animate-spin" />
            </div>
          ) : orders?.length === 0 ? (
            <p className="text-sm text-slate-400 text-center py-8">
              {selectedMachine ? 'Nenhuma ordem para esta máquina' : 'Selecione uma máquina'}
            </p>
          ) : (
            <div className="space-y-2 max-h-[600px] overflow-y-auto pr-2">
              {orders?.map((order) => (
                <OrderCard
                  key={order.id}
                  order={order}
                  isSelected={selectedOrder?.id === order.id}
                  onClick={() => setSelectedOrder(order)}
                />
              ))}
            </div>
          )}
        </div>

        {/* Detail - Right Panel */}
        <div className="col-span-5 space-y-4">
          {!selectedOrder ? (
            <div className="flex flex-col items-center justify-center h-[600px] text-center">
              <FileText className="w-12 h-12 text-slate-600 mb-4" />
              <p className="text-slate-400">Selecione uma ordem para ver detalhes</p>
            </div>
          ) : (
            <>
              {/* Order Detail Header */}
              <div className="rounded-xl border border-slate-700/50 bg-slate-800/30 p-4">
                <div className="flex items-center justify-between mb-4">
                  <div>
                    <h3 className="text-lg font-semibold text-white">{selectedOrder.id}</h3>
                    <p className="text-sm text-slate-400">
                      {selectedOrder.article_name} • {selectedOrder.operation_code}
                    </p>
                  </div>
                  <StatusBadge status={selectedOrder.status} />
                </div>

                {/* Action Buttons */}
                <div className="flex gap-2">
                  {selectedOrder.status === 'pending' && (
                    <button
                      onClick={() => startMutation.mutate()}
                      disabled={!selectedMachine || startMutation.isPending}
                      className="flex-1 flex items-center justify-center gap-2 py-2 bg-emerald-600 hover:bg-emerald-500 text-white font-medium rounded-lg transition disabled:opacity-50"
                    >
                      <Play className="w-4 h-4" />
                      Iniciar
                    </button>
                  )}

                  {selectedOrder.status === 'in_progress' && (
                    <>
                      <button
                        onClick={() => pauseMutation.mutate()}
                        disabled={pauseMutation.isPending}
                        className="flex-1 flex items-center justify-center gap-2 py-2 bg-amber-600 hover:bg-amber-500 text-white font-medium rounded-lg transition disabled:opacity-50"
                      >
                        <Pause className="w-4 h-4" />
                        Pausar
                      </button>
                      <button
                        onClick={() => completeMutation.mutate()}
                        disabled={completeMutation.isPending}
                        className="flex-1 flex items-center justify-center gap-2 py-2 bg-cyan-600 hover:bg-cyan-500 text-white font-medium rounded-lg transition disabled:opacity-50"
                      >
                        <CheckCircle className="w-4 h-4" />
                        Terminar
                      </button>
                    </>
                  )}

                  {selectedOrder.status === 'paused' && (
                    <button
                      onClick={() => startMutation.mutate()}
                      disabled={!selectedMachine || startMutation.isPending}
                      className="flex-1 flex items-center justify-center gap-2 py-2 bg-emerald-600 hover:bg-emerald-500 text-white font-medium rounded-lg transition disabled:opacity-50"
                    >
                      <Play className="w-4 h-4" />
                      Retomar
                    </button>
                  )}
                </div>
              </div>

              {/* Tabs */}
              <div className="flex gap-1 p-1 bg-slate-800/50 rounded-lg w-fit">
                <button
                  onClick={() => setShowInstructions(false)}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition ${
                    !showInstructions ? 'bg-slate-700 text-white' : 'text-slate-400 hover:text-white'
                  }`}
                >
                  📊 Registo
                </button>
                <button
                  onClick={() => setShowInstructions(true)}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition ${
                    showInstructions ? 'bg-slate-700 text-white' : 'text-slate-400 hover:text-white'
                  }`}
                >
                  📋 Instruções
                </button>
              </div>

              {/* Content */}
              <div className="rounded-xl border border-slate-700/50 bg-slate-800/30 p-4 max-h-[400px] overflow-y-auto">
                {showInstructions ? (
                  workInstructions ? (
                    <WorkInstructionView instruction={workInstructions} />
                  ) : (
                    <p className="text-sm text-slate-400 text-center py-8">
                      Sem instruções de trabalho disponíveis
                    </p>
                  )
                ) : (
                  <ReportForm
                    order={selectedOrder}
                    machineId={selectedMachine || ''}
                    onSubmit={(data) => reportMutation.mutate(data)}
                    isPending={reportMutation.isPending}
                  />
                )}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}

export default Shopfloor



