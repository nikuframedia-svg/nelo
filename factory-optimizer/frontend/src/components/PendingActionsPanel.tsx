/**
 * Pending Actions Panel - Industry 5.0 Human-Centric Approval
 * 
 * Displays proposed actions and allows human approval/rejection.
 * 
 * Key principle: O sistema PROPÕE, o humano DECIDE.
 * 
 * All labels in Portuguese (Portugal).
 */

import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { motion, AnimatePresence } from 'framer-motion'
import toast from 'react-hot-toast'
import {
  apiGetActions,
  apiApproveAction,
  apiRejectAction,
  type Action,
  type ActionStatus,
} from '../services/nikufraApi'

// -------------------------
// Helpers
// -------------------------

const ACTION_TYPE_LABELS: Record<string, string> = {
  SET_MACHINE_DOWN: '🔴 Parar Máquina',
  SET_MACHINE_UP: '🟢 Reativar Máquina',
  CHANGE_ROUTE: '🔀 Alterar Rota',
  MOVE_OPERATION: '➡️ Mover Operação',
  SET_VIP_ARTICLE: '⭐ Definir VIP',
  CHANGE_HORIZON: '📅 Alterar Horizonte',
  ADD_OVERTIME: '⏰ Horas Extra',
  ADD_ORDER: '📦 Nova Ordem',
}

const SOURCE_LABELS: Record<string, string> = {
  suggestion_engine: '🤖 Sugestão Automática',
  chat_command: '💬 Comando de Chat',
  what_if: '🔮 Cenário What-If',
  manual: '👤 Manual',
}

const STATUS_LABELS: Record<ActionStatus, string> = {
  PENDING: 'Pendente',
  APPROVED: 'Aprovada',
  REJECTED: 'Rejeitada',
  APPLIED: 'Aplicada',
}

const STATUS_COLORS: Record<ActionStatus, string> = {
  PENDING: 'bg-amber-500/20 text-amber-400 border-amber-500/40',
  APPROVED: 'bg-blue-500/20 text-blue-400 border-blue-500/40',
  REJECTED: 'bg-red-500/20 text-red-400 border-red-500/40',
  APPLIED: 'bg-green-500/20 text-green-400 border-green-500/40',
}

function formatDate(isoString: string): string {
  const date = new Date(isoString)
  return date.toLocaleString('pt-PT', {
    day: '2-digit',
    month: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  })
}

// -------------------------
// Action Card Component
// -------------------------

interface ActionCardProps {
  action: Action
  onApprove: (id: string, notes?: string) => void
  onReject: (id: string, reason?: string) => void
  isApproving: boolean
  isRejecting: boolean
}

const ActionCard: React.FC<ActionCardProps> = ({
  action,
  onApprove,
  onReject,
  isApproving,
  isRejecting,
}) => {
  const [showDetails, setShowDetails] = useState(false)
  const [rejectReason, setRejectReason] = useState('')
  const [showRejectInput, setShowRejectInput] = useState(false)

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      className="rounded-xl border border-border bg-surface p-4"
    >
      {/* Header */}
      <div className="flex items-start justify-between gap-4">
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-1">
            <span className="text-sm font-semibold text-text-primary">
              {ACTION_TYPE_LABELS[action.type] || action.type}
            </span>
            <span className={`rounded-full border px-2 py-0.5 text-xs ${STATUS_COLORS[action.status]}`}>
              {STATUS_LABELS[action.status]}
            </span>
          </div>
          <p className="text-sm text-text-primary">{action.description || 'Sem descrição'}</p>
          <div className="mt-2 flex flex-wrap gap-3 text-xs text-text-muted">
            <span>{SOURCE_LABELS[action.source] || action.source}</span>
            <span>•</span>
            <span>{formatDate(action.created_at)}</span>
            <span>•</span>
            <span>ID: {action.id}</span>
          </div>
        </div>

        {/* Actions */}
        {action.status === 'PENDING' && (
          <div className="flex gap-2">
            <button
              onClick={() => onApprove(action.id)}
              disabled={isApproving || isRejecting}
              className="rounded-lg bg-green-500/20 border border-green-500/40 px-3 py-1.5 text-sm font-medium text-green-400 hover:bg-green-500/30 disabled:opacity-50 transition"
            >
              {isApproving ? '...' : '✓ Aprovar'}
            </button>
            <button
              onClick={() => setShowRejectInput(!showRejectInput)}
              disabled={isApproving || isRejecting}
              className="rounded-lg bg-red-500/20 border border-red-500/40 px-3 py-1.5 text-sm font-medium text-red-400 hover:bg-red-500/30 disabled:opacity-50 transition"
            >
              ✕ Rejeitar
            </button>
          </div>
        )}
      </div>

      {/* Reject Input */}
      {showRejectInput && action.status === 'PENDING' && (
        <motion.div
          initial={{ height: 0, opacity: 0 }}
          animate={{ height: 'auto', opacity: 1 }}
          className="mt-3 pt-3 border-t border-border/50"
        >
          <input
            type="text"
            value={rejectReason}
            onChange={(e) => setRejectReason(e.target.value)}
            placeholder="Motivo da rejeição (opcional)..."
            className="w-full rounded-lg border border-border bg-background px-3 py-2 text-sm text-text-primary placeholder:text-text-muted focus:border-nikufra focus:outline-none"
          />
          <div className="mt-2 flex gap-2">
            <button
              onClick={() => {
                onReject(action.id, rejectReason || undefined)
                setShowRejectInput(false)
              }}
              disabled={isRejecting}
              className="rounded-lg bg-red-500 px-3 py-1.5 text-sm font-medium text-white hover:bg-red-600 disabled:opacity-50"
            >
              Confirmar Rejeição
            </button>
            <button
              onClick={() => setShowRejectInput(false)}
              className="rounded-lg border border-border px-3 py-1.5 text-sm text-text-muted hover:border-nikufra"
            >
              Cancelar
            </button>
          </div>
        </motion.div>
      )}

      {/* Details Toggle */}
      <button
        onClick={() => setShowDetails(!showDetails)}
        className="mt-3 text-xs text-nikufra hover:underline"
      >
        {showDetails ? '▲ Esconder detalhes' : '▼ Ver detalhes'}
      </button>

      {/* Expanded Details */}
      {showDetails && (
        <motion.div
          initial={{ height: 0, opacity: 0 }}
          animate={{ height: 'auto', opacity: 1 }}
          className="mt-3 rounded-lg bg-background/50 p-3 text-xs"
        >
          <p className="font-semibold text-text-muted mb-2">Payload:</p>
          <pre className="text-text-primary overflow-auto max-h-32">
            {JSON.stringify(action.payload, null, 2)}
          </pre>
          {action.expected_impact && (
            <>
              <p className="font-semibold text-text-muted mt-3 mb-2">Impacto Esperado:</p>
              <pre className="text-text-primary overflow-auto max-h-32">
                {JSON.stringify(action.expected_impact, null, 2)}
              </pre>
            </>
          )}
          {action.notes && (
            <p className="mt-3 text-text-muted">
              <span className="font-semibold">Notas:</span> {action.notes}
            </p>
          )}
          {action.approved_by && (
            <p className="mt-2 text-text-muted">
              <span className="font-semibold">Processada por:</span> {action.approved_by} em{' '}
              {action.approved_at ? formatDate(action.approved_at) : '—'}
            </p>
          )}
        </motion.div>
      )}
    </motion.div>
  )
}

// -------------------------
// Main Panel Component
// -------------------------

interface PendingActionsPanelProps {
  showHistory?: boolean
}

export const PendingActionsPanel: React.FC<PendingActionsPanelProps> = ({
  showHistory = false,
}) => {
  const [filter, setFilter] = useState<ActionStatus | 'ALL'>('PENDING')
  const [userName, setUserName] = useState('Operador')
  const queryClient = useQueryClient()

  // Query actions
  const { data, isLoading, error } = useQuery({
    queryKey: ['actions', filter === 'ALL' ? undefined : filter],
    queryFn: () => apiGetActions(filter === 'ALL' ? undefined : filter),
    staleTime: 10_000,
    refetchInterval: 30_000,
  })

  // Approve mutation
  const approveMutation = useMutation({
    mutationFn: ({ id, notes }: { id: string; notes?: string }) =>
      apiApproveAction(id, userName, notes),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['actions'] })
      queryClient.invalidateQueries({ queryKey: ['unified-gantt-plan'] })
      queryClient.invalidateQueries({ queryKey: ['plan-kpis'] })
      toast.success('Ação aprovada e aplicada ao plano!')
    },
    onError: () => {
      toast.error('Erro ao aprovar ação')
    },
  })

  // Reject mutation
  const rejectMutation = useMutation({
    mutationFn: ({ id, reason }: { id: string; reason?: string }) =>
      apiRejectAction(id, userName, reason),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['actions'] })
      toast.success('Ação rejeitada')
    },
    onError: () => {
      toast.error('Erro ao rejeitar ação')
    },
  })

  const handleApprove = (id: string, notes?: string) => {
    approveMutation.mutate({ id, notes })
  }

  const handleReject = (id: string, reason?: string) => {
    rejectMutation.mutate({ id, reason })
  }

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h3 className="text-lg font-bold text-text-primary flex items-center gap-2">
            📋 Ações Pendentes
            {data?.pending_count && data.pending_count > 0 && (
              <span className="rounded-full bg-amber-500/20 px-2 py-0.5 text-sm text-amber-400">
                {data.pending_count}
              </span>
            )}
          </h3>
          <p className="text-xs text-text-muted mt-1">
            O sistema propõe • O humano decide • Industry 5.0
          </p>
        </div>

        {/* User name input */}
        <div className="flex items-center gap-2">
          <span className="text-xs text-text-muted">Utilizador:</span>
          <input
            type="text"
            value={userName}
            onChange={(e) => setUserName(e.target.value)}
            className="rounded-lg border border-border bg-background px-2 py-1 text-sm text-text-primary w-32 focus:border-nikufra focus:outline-none"
          />
        </div>
      </div>

      {/* Filter tabs */}
      {showHistory && (
        <div className="flex gap-2">
          {(['PENDING', 'APPLIED', 'REJECTED', 'ALL'] as const).map((status) => (
            <button
              key={status}
              onClick={() => setFilter(status)}
              className={`rounded-lg px-3 py-1.5 text-xs font-medium transition ${
                filter === status
                  ? 'bg-nikufra text-background'
                  : 'border border-border text-text-muted hover:border-nikufra hover:text-nikufra'
              }`}
            >
              {status === 'ALL' ? 'Todas' : STATUS_LABELS[status]}
            </button>
          ))}
        </div>
      )}

      {/* Loading state */}
      {isLoading && (
        <div className="flex items-center justify-center py-8 text-text-muted">
          A carregar ações...
        </div>
      )}

      {/* Error state */}
      {error && (
        <div className="rounded-xl border border-red-500/40 bg-red-500/10 p-4 text-sm text-red-400">
          Erro ao carregar ações. Tente novamente.
        </div>
      )}

      {/* Empty state */}
      {data && data.actions.length === 0 && (
        <div className="rounded-xl border border-border bg-surface p-8 text-center">
          <p className="text-lg">✅</p>
          <p className="text-sm text-text-muted mt-2">
            {filter === 'PENDING'
              ? 'Não existem ações pendentes de aprovação.'
              : 'Não existem ações nesta categoria.'}
          </p>
        </div>
      )}

      {/* Actions list */}
      <AnimatePresence mode="popLayout">
        {data?.actions.map((action) => (
          <ActionCard
            key={action.id}
            action={action}
            onApprove={handleApprove}
            onReject={handleReject}
            isApproving={approveMutation.isPending && approveMutation.variables?.id === action.id}
            isRejecting={rejectMutation.isPending && rejectMutation.variables?.id === action.id}
          />
        ))}
      </AnimatePresence>

      {/* Info box */}
      <div className="rounded-xl border border-nikufra/40 bg-nikufra/10 p-4 text-sm">
        <p className="font-semibold text-nikufra mb-2">ℹ️ Como funciona</p>
        <ul className="text-text-muted space-y-1 text-xs">
          <li>• O sistema analisa o plano e propõe ações de melhoria</li>
          <li>• Cada ação aguarda aprovação humana antes de ser executada</li>
          <li>• Ao aprovar, o plano é recalculado automaticamente</li>
          <li>• Nenhuma alteração é feita a sistemas externos (ERP/MES)</li>
        </ul>
      </div>
    </div>
  )
}

export default PendingActionsPanel



