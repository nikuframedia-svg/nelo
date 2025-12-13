import { useMemo, useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { format, addDays, startOfWeek } from 'date-fns'
import { motion, AnimatePresence } from 'framer-motion'
import { useNavigate } from 'react-router-dom'
import Skeleton from 'react-loading-skeleton'
import 'react-loading-skeleton/dist/skeleton.css'
// Legacy imports - kept for reference but not currently used
// import { KPICard } from '../components/KPICard'
// import { GanttChart } from '../components/GanttChart'
// import { PlanTimeline } from '../components/PlanTimeline'
import { InsightBanner } from '../components/InsightBanner'
import { KPIDashboard } from '../components/KPIDashboard'
import { UnifiedGantt } from '../components/UnifiedGantt'
import { PendingActionsPanel } from '../components/PendingActionsPanel'
import ProductMetrics from '../components/ProductMetrics'
import api from '../utils/api'
import { Plan, PlanResponse, PlanV2Response, PlanV2Operation } from '../types'
import { batchStorage } from '../utils/batchStorage'

// Legacy type - no longer used
// type ChartViewMode = 'gantt' | 'timeline'

const asNumber = (value: unknown, fallback = 0) => {
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : fallback
}

const asString = (value: unknown, fallback = 'N/A') => {
  if (value === null || value === undefined || value === '') return fallback
  return String(value)
}

const normalizePlan = (plan?: Plan | Record<string, unknown>): Plan | null => {
  if (!plan || typeof plan !== 'object') {
    return null
  }

  const typed = plan as Plan & Record<string, unknown>
  const kpisObj = (typed.kpis as unknown) as Record<string, unknown> | undefined

  return {
    kpis: {
      otd_pct: asNumber(kpisObj?.otd_pct ?? typed.otd_pct, 0),
      lead_time_h: asNumber(kpisObj?.lead_time_h ?? typed.lead_time_h, 0),
      gargalo_ativo: asString(kpisObj?.gargalo_ativo ?? typed.gargalo_ativo ?? 'N/A'),
      horas_setup_semana: asNumber(
        kpisObj?.horas_setup_semana ?? typed.horas_setup_semana,
        0,
      ),
    },
    operations: Array.isArray(typed.operations) ? (typed.operations as Plan['operations']) : [],
    explicacoes: Array.isArray(typed.explicacoes)
      ? (typed.explicacoes as Plan['explicacoes'])
      : Array.isArray((typed as Record<string, unknown>).explicações)
      ? (((typed as Record<string, unknown>).explicações as unknown[]) as Plan['explicacoes'])
      : [],
  }
}

// Componente para cards de decisões
const DecisionCard = ({ decision, index }: { decision: string; index: number }) => {
  const getIcon = () => {
    if (decision.toLowerCase().includes('overlap')) return '🧩'
    if (decision.toLowerCase().includes('colar') || decision.toLowerCase().includes('família')) return '🔗'
    if (decision.toLowerCase().includes('desvio') || decision.toLowerCase().includes('rota')) return '🧭'
    if (decision.toLowerCase().includes('setup')) return '⚙️'
    return '🔧'
  }

  const getColor = () => {
    if (decision.toLowerCase().includes('overlap')) return 'bg-blue-500/20 border-blue-500/40 text-blue-400'
    if (decision.toLowerCase().includes('colar') || decision.toLowerCase().includes('família')) return 'bg-green-500/20 border-green-500/40 text-green-400'
    if (decision.toLowerCase().includes('desvio') || decision.toLowerCase().includes('rota')) return 'bg-purple-500/20 border-purple-500/40 text-purple-400'
    return 'bg-nikufra/20 border-nikufra/40 text-nikufra'
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.1 }}
      className={`rounded-xl border-2 p-4 ${getColor()} transition-all hover:scale-105 hover:shadow-lg`}
    >
      <div className="flex items-start gap-3">
        <span className="text-2xl">{getIcon()}</span>
        <div className="flex-1">
          <p className="text-sm font-semibold leading-relaxed">{decision}</p>
        </div>
      </div>
    </motion.div>
  )
}

// Modal de explicação do plano
const PlanExplanationModal = ({
  isOpen,
  onClose,
  antes: _antes,
  depois: _depois,
}: {
  isOpen: boolean
  onClose: () => void
  antes: Plan | null
  depois: Plan | null
}) => {
  const [explanation, setExplanation] = useState<string>('')
  const [loading, setLoading] = useState(false)

  const fetchExplanation = async () => {
    setLoading(true)
    try {
      const response = await api.get('/insights/generate', {
        params: { mode: 'planeamento' },
      })
      setExplanation(response.data.insight || response.data.text || 'Explicação não disponível.')
    } catch (error) {
      setExplanation('Erro ao gerar explicação. Verifique se os dados foram carregados.')
    } finally {
      setLoading(false)
    }
  }

  if (!isOpen) return null

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-md"
        onClick={onClose}
      >
        <motion.div
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.9, opacity: 0 }}
          onClick={(e) => e.stopPropagation()}
          className="relative w-full max-w-3xl max-h-[80vh] overflow-y-auto rounded-2xl border-2 border-nikufra/60 bg-surface p-6 shadow-2xl"
        >
          <button
            onClick={onClose}
            className="absolute top-4 right-4 rounded-full border border-border p-2 text-text-muted transition hover:text-text-primary"
          >
            ✕
          </button>
          
          <h2 className="text-2xl font-bold text-text-primary mb-4">Explicar este plano</h2>
          
          {!explanation && (
            <button
              onClick={fetchExplanation}
              disabled={loading}
              className="w-full rounded-xl border-2 border-nikufra bg-nikufra/10 px-6 py-3 text-sm font-semibold text-nikufra transition hover:bg-nikufra hover:text-background disabled:opacity-50"
            >
              {loading ? 'A gerar explicação...' : 'Gerar explicação industrial'}
            </button>
          )}
          
          {explanation && (
            <div className="mt-4 text-sm leading-relaxed text-text-body whitespace-pre-wrap">
              {explanation}
            </div>
          )}
        </motion.div>
      </motion.div>
    </AnimatePresence>
  )
}

// Componente de Chat de Planeamento
const PlanningChat = ({ batchId, horizonHours, onCommandApplied }: { batchId?: string; horizonHours: number; onCommandApplied: () => Promise<void> }) => {
  const [message, setMessage] = useState('')
  const [isOpen, setIsOpen] = useState(false)
  const [history, setHistory] = useState<Array<{ role: 'user' | 'assistant'; content: string }>>([])
  
  // CRÍTICO: Obter queryClient para invalidar cache
  const queryClient = useQueryClient()

  const interpretMutation = useMutation({
    mutationFn: async (userMessage: string) => {
      const response = await api.post('/planning/chat/interpret', {
        message: userMessage,
        batch_id: batchId,
      })
      return response.data
    },
  })

  const applyMutation = useMutation({
    mutationFn: async (command: any) => {
      // Log do que está a ser enviado
      console.log('📤 Frontend enviando para /apply:', {
        command,
        batch_id: batchId,
        command_type: command?.command_type,
        has_machine_unavailable: !!command?.machine_unavailable,
      })
      
      try {
        const response = await api.post('/planning/chat/apply', {
          command,
          batch_id: batchId,
        })
        console.log('✅ Frontend recebeu resposta de /apply:', response.data)
        return response.data
      } catch (error: any) {
        console.error('❌ Frontend erro em /apply:', {
          status: error?.response?.status,
          statusText: error?.response?.statusText,
          data: error?.response?.data,
          message: error?.message,
        })
        throw error
      }
    },
    onSuccess: async (data) => {
      // Detectar se é recalculate_plan para mensagem específica
      const isRecalculatePlan = data?.command_type === 'recalculate_plan'
      
      // Após aplicar comando, invalidar cache e recalcular plano
      if (!isRecalculatePlan) {
        setHistory((prev) => [
          ...prev,
          {
            role: 'assistant',
            content: '✅ Comando aplicado! A recalcular plano...',
          },
        ])
      }
      
      // Forçar recálculo imediato do plano
      try {
        // Invalidar cache ANTES de recalcular (garante que não usa cache antigo)
        queryClient.invalidateQueries({ queryKey: ['plan', batchId, horizonHours] })
        
        // Chamar recalculate para forçar novo cálculo com configuração atualizada
        await api.post('/planning/v2/recalculate', null, {
          params: {
            batch_id: batchId,
            horizon_hours: horizonHours,
          },
        })
        
        // Invalidar cache novamente e refetch (garantir que pega dados frescos)
        queryClient.invalidateQueries({ queryKey: ['plan', batchId, horizonHours] })
        await queryClient.refetchQueries({ queryKey: ['plan', batchId, horizonHours] })
        
        // Também chamar callback
        onCommandApplied()
        
        setHistory((prev) => [
          ...prev,
          {
            role: 'assistant',
            content: isRecalculatePlan 
              ? '✅ Plano otimizado novamente com sucesso!'
              : '✅ Plano recalculado com sucesso!',
          },
        ])
      } catch (err: any) {
        console.error('Erro ao recalcular plano:', err)
        setHistory((prev) => [
          ...prev,
          {
            role: 'assistant',
            content: `❌ Erro ao recalcular: ${err?.response?.data?.detail || err?.message || 'Erro desconhecido'}`,
          },
        ])
      }
    },
  })

  const handleSend = async () => {
    if (!message.trim()) return

    const userMessage = message.trim()
    setMessage('')
    setHistory((prev) => [...prev, { role: 'user', content: userMessage }])

    try {
      // Interpretar comando
      const interpretation = await interpretMutation.mutateAsync(userMessage)

      // CRÍTICO: Verificar ANTES de mostrar qualquer mensagem
      // Se for "unknown" ou requer clarificação, NUNCA tentar aplicar
      if (interpretation.command?.command_type === 'unknown' || interpretation.requires_clarification) {
        setHistory((prev) => [
          ...prev,
          {
            role: 'assistant',
            content: `❓ ${interpretation.clarification_message || 'Não consegui perceber a instrução. Exemplos: "máquina 300 indisponível", "planeia só 6 horas", "GO4 VIP".'}`,
          },
        ])
        return  // NUNCA continuar se for unknown ou requer clarificação
      }

      // Mostrar comando interpretado (só se não for unknown)
      const commandType = interpretation.command.command_type
      let commandDescription = ''
      if (commandType === 'machine_unavailable') {
        const cmd = interpretation.command.machine_unavailable
        if (cmd && cmd.maquina_id) {
          if (cmd.start_time && cmd.end_time) {
            commandDescription = `Máquina ${cmd.maquina_id} marcada como indisponível de ${format(new Date(cmd.start_time), 'HH:mm')} até ${format(new Date(cmd.end_time), 'HH:mm')}`
          } else {
            commandDescription = `Máquina ${cmd.maquina_id} marcada como indisponível`
          }
        } else {
          commandDescription = 'Máquina marcada como indisponível (detalhes incompletos)'
        }
      } else if (commandType === 'machine_available') {
        const cmd = interpretation.command.machine_available
        if (cmd && cmd.maquina_id) {
          commandDescription = `Máquina ${cmd.maquina_id} marcada como disponível novamente`
        } else {
          commandDescription = 'Máquina marcada como disponível novamente (detalhes incompletos)'
        }
      } else if (commandType === 'add_manual_order') {
        const cmd = interpretation.command.manual_order
        if (cmd && cmd.artigo && cmd.quantidade) {
          commandDescription = `Adicionar ordem: ${cmd.artigo} (${cmd.quantidade} unidades, ${cmd.prioridade || 'NORMAL'})`
        } else {
          commandDescription = 'Adicionar ordem manual (detalhes incompletos)'
        }
      } else if (commandType === 'change_priority') {
        const cmd = interpretation.command.priority_change
        if (cmd && cmd.order_id && cmd.new_priority) {
          if (cmd.new_priority === 'NORMAL') {
            commandDescription = `Prioridade do ${cmd.order_id} removida (volta ao normal)`
          } else {
            commandDescription = `Prioridade do ${cmd.order_id} atualizada para ${cmd.new_priority}`
          }
        } else {
          commandDescription = 'Alterar prioridade (detalhes incompletos)'
        }
      } else if (commandType === 'change_horizon') {
        const cmd = interpretation.command.horizon_change
        if (cmd && cmd.horizon_hours) {
          commandDescription = `Alterar horizonte: ${cmd.horizon_hours}h`
        } else {
          commandDescription = 'Alterar horizonte (detalhes incompletos)'
        }
      } else if (commandType === 'recalculate_plan') {
        commandDescription = 'Recalcular plano com configuração atual'
      } else if (commandType === 'unknown') {
        // Se for unknown, não devia chegar aqui (já foi verificado antes)
        // Mas por segurança, tratar como clarificação
        console.warn('⚠️ Frontend: Comando unknown chegou à fase de descrição (não devia acontecer)')
        setHistory((prev) => [
          ...prev,
          {
            role: 'assistant',
            content: `❓ ${interpretation.clarification_message || 'Não consegui perceber a instrução. Reformula ou dá mais contexto.'}`,
          },
        ])
        return
      } else {
        // Se chegou aqui e não é um tipo conhecido, é erro
        console.error('❌ Frontend: Tipo de comando desconhecido:', commandType)
        setHistory((prev) => [
          ...prev,
          {
            role: 'assistant',
            content: `❌ Erro: Tipo de comando desconhecido: ${commandType}. Por favor, tente novamente.`,
          },
        ])
        return
      }

      // Mensagem específica para recalculate_plan
      const applyMessage = commandType === 'recalculate_plan' 
        ? '⏳ A recalcular com as configurações atuais…'
        : '⏳ A aplicar e a recalcular o plano...'
      
      setHistory((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: `📋 ${commandDescription}\n\n${applyMessage}`,
        },
      ])
      
      // Aplicar comando automaticamente SEMPRE (sem pedir confirmação)
      // O sistema já validou que o comando é válido
      try {
        // Log do comando antes de enviar
        console.log('📤 Frontend: Aplicando comando interpretado:', {
          interpretation,
          command: interpretation.command,
          command_type: interpretation.command?.command_type,
        })
        
        await applyMutation.mutateAsync(interpretation.command)
        // onSuccess do applyMutation já mostra mensagem de sucesso
      } catch (applyError: any) {
        // Log detalhado do erro
        console.error('❌ Frontend: Erro ao aplicar comando:', {
          error: applyError,
          response: applyError?.response,
          status: applyError?.response?.status,
          data: applyError?.response?.data,
          detail: applyError?.response?.data?.detail,
        })
        
        // Se falhar ao aplicar, mostrar erro detalhado
        const errorDetail = applyError?.response?.data?.detail || applyError?.message || 'Erro desconhecido'
        setHistory((prev) => [
          ...prev,
          {
            role: 'assistant',
            content: `❌ Erro ao aplicar comando: ${errorDetail}\n\nVerifique a consola do browser (F12) para mais detalhes.`,
          },
        ])
      }
    } catch (error: any) {
      setHistory((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: `❌ Erro: ${error?.response?.data?.detail || error?.message || 'Erro desconhecido'}`,
        },
      ])
    }
  }

  if (!isOpen) {
    return (
      <button
        onClick={() => setIsOpen(true)}
        className="fixed bottom-6 right-6 z-50 rounded-full bg-nikufra p-4 shadow-lg transition hover:scale-110"
        title="Chat de Planeamento"
      >
        <span className="text-2xl">💬</span>
      </button>
    )
  }

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: 20 }}
        className="fixed bottom-6 right-6 z-50 w-96 rounded-2xl border-2 border-nikufra/60 bg-surface shadow-2xl"
      >
        <div className="flex items-center justify-between border-b border-border/60 px-4 py-3">
          <h3 className="text-sm font-bold text-text-primary">💬 Chat de Planeamento</h3>
          <button
            onClick={() => setIsOpen(false)}
            className="rounded-full p-1 text-text-muted transition hover:text-text-primary"
          >
            ✕
          </button>
        </div>

        <div className="max-h-96 overflow-y-auto p-4 space-y-2">
          {history.length === 0 && (
            <div className="text-xs text-text-muted space-y-1">
              <p>Exemplos:</p>
              <p>• "Máquina 190 indisponível das 14h às 18h"</p>
              <p>• "Ordem VIP para GO6 com 200 unidades para amanhã"</p>
              <p>• "Planear só para as próximas 4 horas"</p>
            </div>
          )}
          {history.map((msg, idx) => (
            <div
              key={idx}
              className={`rounded-lg p-2 text-xs ${
                msg.role === 'user'
                  ? 'bg-nikufra/20 text-text-primary ml-auto max-w-[80%]'
                  : 'bg-background/60 text-text-muted'
              }`}
            >
              {msg.content}
            </div>
          ))}
          {(interpretMutation.isPending || applyMutation.isPending) && (
            <div className="text-xs text-text-muted">⏳ A processar...</div>
          )}
        </div>

        <div className="border-t border-border/60 p-3">
          <div className="flex gap-2">
            <input
              type="text"
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && !e.shiftKey && handleSend()}
              placeholder="Escreva uma instrução..."
              className="flex-1 rounded-lg border border-border bg-background px-3 py-2 text-sm text-text-primary outline-none focus:border-nikufra"
              disabled={interpretMutation.isPending || applyMutation.isPending}
            />
            <button
              onClick={handleSend}
              disabled={!message.trim() || interpretMutation.isPending || applyMutation.isPending}
              className="rounded-lg bg-nikufra px-4 py-2 text-sm font-semibold text-background transition hover:bg-nikufra/80 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Enviar
            </button>
          </div>
        </div>
      </motion.div>
    </AnimatePresence>
  )
}

// Advanced scheduling options types
type SchedulerEngine = 'heuristic' | 'milp' | 'cpsat'
type DispatchRule = 'FIFO' | 'SPT' | 'EDD' | 'CR' | 'WSPT'

const engineOptions: { id: SchedulerEngine; name: string; desc: string }[] = [
  { id: 'heuristic', name: 'Heurístico', desc: 'Regras de dispatching rápidas' },
  { id: 'milp', name: 'MILP', desc: 'Otimização matemática (mais lento)' },
  { id: 'cpsat', name: 'CP-SAT', desc: 'Constraint Programming' },
]

const ruleOptions: { id: DispatchRule; name: string; desc: string }[] = [
  { id: 'EDD', name: 'Earliest Due Date', desc: 'Prioriza por data de entrega' },
  { id: 'SPT', name: 'Shortest Processing', desc: 'Operações mais curtas primeiro' },
  { id: 'FIFO', name: 'First In, First Out', desc: 'Ordem de chegada' },
  { id: 'WSPT', name: 'Weighted SPT', desc: 'SPT ponderado por prioridade' },
  { id: 'CR', name: 'Critical Ratio', desc: 'Rácio crítico' },
]

export const Planning = () => {
  const navigate = useNavigate()
  const [timeRange, setTimeRange] = useState<'this_week' | 'next_week'>('this_week')
  const [cell, setCell] = useState<string>('')
  const [showExplanation, setShowExplanation] = useState(false)
  
  // Advanced scheduling options state
  const [showAdvancedOptions, setShowAdvancedOptions] = useState(false)
  const [selectedEngine, setSelectedEngine] = useState<SchedulerEngine>('heuristic')
  const [selectedRule, setSelectedRule] = useState<DispatchRule>('EDD')
  const [useDataDriven, setUseDataDriven] = useState(false)

  const startDate =
    timeRange === 'this_week'
      ? startOfWeek(new Date(), { weekStartsOn: 1 })
      : startOfWeek(addDays(new Date(), 7), { weekStartsOn: 1 })
  const endDate = addDays(startDate, 7)

  // Função para converter PlanV2 para formato antigo (compatibilidade)
  const convertPlanV2ToPlan = (planV2: PlanV2Response['baseline'] | PlanV2Response['optimized']): Plan | null => {
    if (!planV2) return null
    
    // DEBUG: Verificar rotas no JSON recebido
    const rotasNoJson = planV2.operations.map(op => op.rota).filter(Boolean)
    const rotasUnicas = [...new Set(rotasNoJson)]
    console.log('📊 [FRONTEND] Rotas recebidas do backend:', {
      total: planV2.operations.length,
      comRota: rotasNoJson.length,
      semRota: planV2.operations.length - rotasNoJson.length,
      rotasUnicas: rotasUnicas,
      sample: planV2.operations.slice(0, 5).map(op => ({
        order_id: op.order_id,
        op_id: op.op_id,
        rota: op.rota || 'MISSING'
      }))
    })
    
    const operations = planV2.operations.map((op: PlanV2Operation) => {
      // CRÍTICO: Ler rota do backend e garantir que está presente
      const rota = op.rota || 'A'  // Fallback para 'A' se não vier do backend
      
      // Log para debug se rota estiver faltando
      if (!op.rota) {
        console.warn(`⚠️ [FRONTEND] Operação ${op.order_id}/${op.op_id} sem rota do backend, usando fallback 'A'`)
      }
      
      return {
        ordem: op.order_id,
        artigo: op.artigo || op.order_id.replace('ORD-', ''),
        operacao: op.op_id,
        recurso: op.maquina_id,
        rota: rota,  // Usar rota do backend (ou fallback 'A')
        start_time: op.start_time,
        end_time: op.end_time,
        setor: op.family || '',
        overlap: 0,
        explicacao: '',
      }
    })
    
    // Log para debug com rotas
    const articles = new Set(operations.map(op => op.artigo))
    const rotas = operations.map(op => op.rota)
    const rotasCount = rotas.reduce((acc, r) => {
      acc[r] = (acc[r] || 0) + 1
      return acc
    }, {} as Record<string, number>)
    
    console.log('🔄 convertPlanV2ToPlan:', {
      totalOps: operations.length,
      articles: Array.from(articles),
      articlesCount: articles.size,
      rotas: rotasCount,
      sampleRotas: rotas.slice(0, 10)  // Primeiras 10 rotas
    })
    
    return {
      kpis: {
        otd_pct: 0,
        lead_time_h: 0,
        gargalo_ativo: 'N/A',
        horas_setup_semana: 0,
      },
      operations,
      explicacoes: [],
    }
  }

  const queryClient = useQueryClient()
  // Calcular horizonte em horas, com mínimo de 24h para agendar todas as operações
  const horizonHours = Math.max(
    Math.ceil((endDate.getTime() - startDate.getTime()) / (1000 * 60 * 60)),
    24
  )

  // Obter batch_id: primeiro do localStorage (persistência), depois do ETL status
  const { data: etlStatus } = useQuery({
    queryKey: ['etl-status'],
    queryFn: async () => {
      const response = await api.get('/etl/status')
      return response.data
    },
    staleTime: Infinity, // Cache infinito até invalidar manualmente
  })
  
  // Prioridade: localStorage > ETL status
  const batchIdFromStorage = batchStorage.get()
  const batchIdFromETL = etlStatus?.latest_batch_id || etlStatus?.batch_id
  const batchId = batchIdFromStorage || batchIdFromETL || undefined
  
  // Atualizar localStorage quando batch_id do ETL mudar
  if (batchIdFromETL && batchIdFromETL !== batchIdFromStorage) {
    batchStorage.set(batchIdFromETL)
  }

  const { data, isLoading, error } = useQuery<PlanV2Response | PlanResponse>({
    queryKey: ['plan', timeRange, cell, horizonHours, batchId],
    queryFn: async (): Promise<PlanV2Response | PlanResponse> => {
      // Tentar nova API v2 primeiro
      const response = await api.get('/planning/v2/plano', {
        params: {
          batch_id: batchId,
          horizon_hours: horizonHours,
        },
      })
      
      // Se retornar estrutura v2, usar diretamente
      if (response.data.baseline || response.data.optimized) {
        const planData = response.data as PlanV2Response
        // Log para debug
        const baselineOps = planData.baseline?.operations || []
        const optimizedOps = planData.optimized?.operations || []
        const baselineArticles = new Set(baselineOps.map(op => op.artigo || op.order_id?.replace('ORD-', '')))
        const optimizedArticles = new Set(optimizedOps.map(op => op.artigo || op.order_id?.replace('ORD-', '')))
        console.log('📥 Frontend recebeu plano:', {
          baseline: { ops: baselineOps.length, articles: Array.from(baselineArticles) },
          optimized: { ops: optimizedOps.length, articles: Array.from(optimizedArticles) },
          orders_summary: planData.orders_summary
        })
        return planData
      }
      
      // Fallback para estrutura antiga
      return response.data as PlanResponse
    },
    retry: false, // Não tentar novamente automaticamente se falhar
    staleTime: Infinity, // Cache infinito até invalidar manualmente
    gcTime: Infinity, // Nunca remover do cache
  })

  // Mutation para recalcular plano
  const recalculateMutation = useMutation({
    mutationFn: async () => {
      console.log('🔄 Iniciando recálculo do plano...', { batchId, horizonHours })
      const response = await api.post('/planning/v2/recalculate', null, {
        params: {
          batch_id: batchId,
          horizon_hours: horizonHours,
        },
      })
      console.log('✅ Recálculo concluído:', response.data)
      return response.data
    },
    onSuccess: () => {
      // Invalidar cache e refetch
      console.log('🔄 Invalidando cache e refetching...')
      queryClient.invalidateQueries({ queryKey: ['plan'] })
      // Forçar refetch imediato
      queryClient.refetchQueries({ queryKey: ['plan'] })
    },
    onError: (error) => {
      console.error('❌ Erro ao recalcular plano:', error)
    },
  })

  // Type guard for v2 response
  const isV2Response = (d: PlanV2Response | PlanResponse | undefined): d is PlanV2Response => {
    return d !== undefined && 'baseline' in d && 'optimized' in d
  }
  
  const v2Data = isV2Response(data) ? data : null
  
  const antes = useMemo(() => {
    if (v2Data) {
      return convertPlanV2ToPlan(v2Data.baseline)
    }
    return normalizePlan((data as PlanResponse)?.antes)
  }, [data, v2Data])
  
  const depois = useMemo(() => {
    if (v2Data) {
      return convertPlanV2ToPlan(v2Data.optimized)
    }
    return normalizePlan((data as PlanResponse)?.depois)
  }, [data, v2Data])

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <p className="text-xs uppercase tracking-[0.4em] text-text-muted">APS assistido por IA</p>
          <h2 className="text-2xl font-bold text-text-primary">Planeamento Herdmar</h2>
          {v2Data?.orders_summary && (
            <p className="mt-1 text-xs text-text-muted">
              {v2Data.orders_summary?.total_orders || 0} artigos processados
            </p>
          )}
          <p className="mt-2 max-w-4xl text-sm text-text-muted leading-relaxed">
            O ProdPlan 4.0 lê os dados de produção da Herdmar, escolhe rotas A/B por artigo e gera um plano encadeado que reduz filas nos recursos lentos e o tempo total de produção.
          </p>
          {error && (
            <div className="mt-3 rounded-xl border-2 border-warning/40 bg-warning/10 p-3 text-sm text-warning">
              <p className="font-semibold">Plano não encontrado</p>
              <p className="mt-1">
                {(error as any)?.response?.status === 404 || (error as Error)?.message?.includes('PLANO_NAO_ENCONTRADO')
                  ? 'Clique em "Recalcular plano" para gerar um novo plano com todos os artigos.'
                  : 'Carregue os dados do Excel e clique em "Recalcular plano".'}
              </p>
            </div>
          )}
        </div>
        <div className="flex items-center gap-3">
          {/* Sempre mostrar botão de recalcular */}
          <button
            onClick={() => recalculateMutation.mutate()}
            disabled={recalculateMutation.isPending}
            className="rounded-xl border-2 border-nikufra bg-nikufra/10 px-4 py-2 text-sm font-semibold text-nikufra transition hover:bg-nikufra hover:text-background disabled:opacity-50"
          >
            {recalculateMutation.isPending ? 'A recalcular...' : '🔄 Recalcular plano'}
          </button>
          
          {/* Advanced Options Toggle */}
          <button
            onClick={() => setShowAdvancedOptions(!showAdvancedOptions)}
            className={`rounded-xl border-2 px-3 py-2 text-sm font-medium transition ${
              showAdvancedOptions
                ? 'border-purple-500/60 bg-purple-500/10 text-purple-400'
                : 'border-border bg-surface text-text-muted hover:border-purple-500/40'
            }`}
            title="Opções avançadas de planeamento"
          >
            ⚙️ {showAdvancedOptions ? 'Fechar' : 'Opções'}
          </button>
          
          <select
            value={timeRange}
            onChange={(event) => setTimeRange(event.target.value as typeof timeRange)}
            className="h-11 rounded-2xl border-2 border-border bg-surface px-4 text-sm font-semibold text-text-primary outline-none transition hover:border-nikufra"
          >
            <option value="this_week">Esta semana</option>
            <option value="next_week">Próxima semana</option>
          </select>
          <input
            type="text"
            placeholder="Filtrar célula/linha"
            value={cell}
            onChange={(event) => setCell(event.target.value)}
            className="h-11 rounded-2xl border-2 border-border bg-surface px-4 text-sm text-text-primary outline-none transition focus:border-nikufra"
          />
          <button
            onClick={() => navigate('/whatif', { state: { mode: 'VIP' } })}
            className="rounded-2xl border-2 border-nikufra bg-nikufra/10 px-4 py-2 text-sm font-semibold text-nikufra transition hover:bg-nikufra hover:text-background"
          >
            Simular VIP
          </button>
          <button
            onClick={() => navigate('/whatif', { state: { mode: 'Avaria' } })}
            className="rounded-2xl border-2 border-border bg-surface px-4 py-2 text-sm font-semibold text-text-primary transition hover:border-warning hover:text-warning"
          >
            Avaria
          </button>
        </div>
      </div>

      {/* Advanced Scheduling Options Panel */}
      <AnimatePresence>
        {showAdvancedOptions && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="rounded-2xl border-2 border-purple-500/30 bg-purple-500/5 p-4 overflow-hidden"
          >
            <div className="flex items-center gap-2 mb-4">
              <span className="text-purple-400">⚙️</span>
              <h3 className="text-sm font-semibold text-purple-400">Opções Avançadas de Planeamento</h3>
              {useDataDriven && (
                <span className="ml-auto text-xs px-2 py-0.5 rounded-full bg-green-500/20 text-green-400 border border-green-500/30">
                  📊 Tempos históricos ativos
                </span>
              )}
            </div>
            
            <div className="grid gap-4 md:grid-cols-3">
              {/* Engine Selection */}
              <div>
                <label className="block text-xs text-text-muted mb-2">Motor de Scheduling</label>
                <select
                  value={selectedEngine}
                  onChange={(e) => setSelectedEngine(e.target.value as SchedulerEngine)}
                  className="w-full rounded-lg border border-purple-500/30 bg-background p-2 text-sm text-text-primary"
                >
                  {engineOptions.map((opt) => (
                    <option key={opt.id} value={opt.id}>
                      {opt.name}
                    </option>
                  ))}
                </select>
                <p className="text-xs text-text-muted mt-1">
                  {engineOptions.find((o) => o.id === selectedEngine)?.desc}
                </p>
              </div>
              
              {/* Dispatching Rule (only for heuristic) */}
              {selectedEngine === 'heuristic' && (
                <div>
                  <label className="block text-xs text-text-muted mb-2">Regra de Dispatching</label>
                  <select
                    value={selectedRule}
                    onChange={(e) => setSelectedRule(e.target.value as DispatchRule)}
                    className="w-full rounded-lg border border-purple-500/30 bg-background p-2 text-sm text-text-primary"
                  >
                    {ruleOptions.map((opt) => (
                      <option key={opt.id} value={opt.id}>
                        {opt.name}
                      </option>
                    ))}
                  </select>
                  <p className="text-xs text-text-muted mt-1">
                    {ruleOptions.find((o) => o.id === selectedRule)?.desc}
                  </p>
                </div>
              )}
              
              {/* Data-Driven Toggle */}
              <div>
                <label className="block text-xs text-text-muted mb-2">Durações Data-Driven</label>
                <button
                  onClick={() => setUseDataDriven(!useDataDriven)}
                  className={`w-full rounded-lg border p-2 text-sm font-medium transition ${
                    useDataDriven
                      ? 'border-green-500/50 bg-green-500/10 text-green-400'
                      : 'border-border bg-background text-text-muted hover:border-green-500/30'
                  }`}
                >
                  {useDataDriven ? '✅ Usar tempos históricos' : '📐 Usar tempos teóricos'}
                </button>
                <p className="text-xs text-text-muted mt-1">
                  {useDataDriven
                    ? 'Durações baseadas em execuções passadas'
                    : 'Durações da ficha técnica'}
                </p>
              </div>
            </div>
            
            {/* Info message */}
            <div className="mt-4 flex items-center gap-2 text-xs text-text-muted">
              <span>💡</span>
              <span>
                {selectedEngine === 'heuristic'
                  ? 'Motor heurístico: muito rápido, boa qualidade para maioria dos casos'
                  : selectedEngine === 'milp'
                  ? 'Motor MILP: pode demorar 30-60s, encontra solução ótima ou próxima'
                  : 'Motor CP-SAT: bom equilíbrio entre qualidade e velocidade'}
              </span>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <InsightBanner mode="planeamento" />

      {/* Chat de Planeamento */}
      <PlanningChat
        batchId={batchId}
        horizonHours={horizonHours}
        onCommandApplied={async () => {
          // Invalidar cache e forçar recálculo do plano
          queryClient.invalidateQueries({ queryKey: ['plan'] })
          
          // Chamar recalculate para forçar novo cálculo com configuração atualizada
          try {
            await api.post('/planning/v2/recalculate', null, {
              params: {
                batch_id: batchId,
                horizon_hours: horizonHours,
              },
            })
            // Refetch após recalcular
            await queryClient.refetchQueries({ queryKey: ['plan', timeRange, cell, horizonHours, batchId] })
          } catch (err) {
            console.error('Erro ao recalcular plano:', err)
            // Mesmo assim, tentar refetch
            await queryClient.refetchQueries({ queryKey: ['plan', timeRange, cell, horizonHours, batchId] })
          }
        }}
      />

      {/* KPIs Reais - Dashboard Completo */}
      <KPIDashboard />

      {/* Ações Pendentes - Industry 5.0 Human-Centric */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3, delay: 0.1 }}
        className="rounded-2xl border-2 border-border bg-gradient-to-br from-surface to-surface/80 p-6"
      >
        <PendingActionsPanel showHistory={false} />
      </motion.div>

      {/* Gantt Unificado com Controlos Avançados */}
      {isLoading ? (
        <Skeleton height={500} baseColor="#121212" highlightColor="#1c1c1c" />
      ) : (
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3 }}>
          <UnifiedGantt />
        </motion.div>
      )}

      {/* Métricas por Tipo de Produto + Estimativa de Entrega */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3, delay: 0.15 }}
        className="rounded-2xl border-2 border-border bg-gradient-to-br from-surface to-surface/80 p-6"
      >
        <ProductMetrics />
      </motion.div>

      {/* Painel 3: Decisões da IA */}
      {!isLoading && Boolean(depois?.explicacoes?.length) && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
          className="rounded-2xl border-2 border-border bg-gradient-to-br from-surface to-surface/80 p-6 shadow-[0_0_32px_rgba(69,255,193,0.08)]"
        >
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-bold text-text-primary">🔧 Decisões tomadas pelo motor</h3>
            <button
              onClick={() => setShowExplanation(true)}
              className="rounded-xl border-2 border-nikufra/60 bg-nikufra/10 px-4 py-2 text-sm font-semibold text-nikufra transition hover:bg-nikufra hover:text-background"
            >
              💡 Explicar este plano
            </button>
          </div>
          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
            {depois?.explicacoes?.map((exp, index) => (
              <DecisionCard key={`${exp}-${index}`} decision={exp} index={index} />
            ))}
          </div>
        </motion.div>
      )}

      {/* Modal de explicação */}
      <PlanExplanationModal
        isOpen={showExplanation}
        onClose={() => setShowExplanation(false)}
        antes={antes}
        depois={depois}
      />
    </div>
  )
}
