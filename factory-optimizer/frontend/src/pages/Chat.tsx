import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { AnimatePresence, motion } from 'framer-motion'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import {
  Send,
  MessageCircle,
  TrendingUp,
  TrendingDown,
  Minus,
  Sparkles,
  RefreshCw,
  ChevronRight,
} from 'lucide-react'
import api from '../utils/api'

// ═══════════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════════

interface KpiPayload {
  label: string
  value: string
  unit?: string
  trend?: 'up' | 'down' | 'neutral'
  color?: string
}

interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
  kpis?: KpiPayload[]
  suggestions?: string[]
  actions?: { label: string; action_type: string; action_data: string }[]
  intent?: string
  confidence?: number
}

interface ChatResponse {
  message: string
  intent: string
  confidence: number
  kpis: KpiPayload[]
  suggestions: string[]
  actions: { label: string; action_type: string; action_data: string }[]
  timestamp: string
  // Legacy support
  answer?: string
  model?: string
}

// ═══════════════════════════════════════════════════════════════════════════════
// COMPONENTS
// ═══════════════════════════════════════════════════════════════════════════════

const quickPrompts = [
  { label: '📊 Estado do plano', query: 'Qual o estado do plano de produção?' },
  { label: '📦 Alertas de stock', query: 'Há SKUs em risco de rutura?' },
  { label: '🔧 Saúde máquinas', query: 'Qual a saúde das máquinas?' },
  { label: '📋 DPPs', query: 'Estado dos passaportes digitais?' },
]

const typingDelay = 8 // ms per character

const AnimatedMarkdown = ({ content }: { content: string }) => {
  const [displayed, setDisplayed] = useState('')

  useEffect(() => {
    setDisplayed('')
    let index = 0
    const interval = window.setInterval(() => {
      index += 1
      setDisplayed(content.slice(0, index))
      if (index >= content.length) {
        window.clearInterval(interval)
      }
    }, typingDelay)
    return () => window.clearInterval(interval)
  }, [content])

  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      className="prose prose-invert prose-sm max-w-none [&>p]:mb-2 [&>p]:leading-relaxed [&>ul>li]:mb-1"
    >
      {displayed || '🧠 ...'}
    </ReactMarkdown>
  )
}

const KpiCard: React.FC<{ kpi: KpiPayload }> = ({ kpi }) => {
  const colorMap: Record<string, string> = {
    green: 'from-emerald-500/20 to-emerald-600/10 border-emerald-500/30',
    red: 'from-red-500/20 to-red-600/10 border-red-500/30',
    yellow: 'from-amber-500/20 to-amber-600/10 border-amber-500/30',
    blue: 'from-cyan-500/20 to-cyan-600/10 border-cyan-500/30',
    default: 'from-slate-500/20 to-slate-600/10 border-slate-500/30',
  }
  
  const TrendIcon = kpi.trend === 'up' ? TrendingUp : kpi.trend === 'down' ? TrendingDown : Minus
  const trendColor = kpi.trend === 'up' ? 'text-emerald-400' : kpi.trend === 'down' ? 'text-red-400' : 'text-slate-400'
  
  return (
    <div className={`rounded-lg bg-gradient-to-br ${colorMap[kpi.color || 'default']} border p-3`}>
      <p className="text-xs text-slate-400 uppercase tracking-wider">{kpi.label}</p>
      <div className="flex items-end gap-1 mt-1">
        <span className="text-xl font-bold text-white">{kpi.value}</span>
        {kpi.unit && <span className="text-sm text-slate-300">{kpi.unit}</span>}
        {kpi.trend && <TrendIcon className={`w-4 h-4 ml-1 ${trendColor}`} />}
      </div>
    </div>
  )
}

const SuggestionChip: React.FC<{ text: string; onClick: () => void }> = ({ text, onClick }) => (
  <button
    onClick={onClick}
    className="px-3 py-1.5 rounded-full text-xs bg-slate-800 hover:bg-slate-700 border border-slate-600 text-slate-300 hover:text-white transition flex items-center gap-1"
  >
    <Sparkles className="w-3 h-3" />
    {text}
  </button>
)

const ActionButton: React.FC<{
  action: { label: string; action_type: string; action_data: string }
  onNavigate: (path: string) => void
}> = ({ action, onNavigate }) => (
  <button
    onClick={() => {
      if (action.action_type === 'navigate') {
        onNavigate(action.action_data)
      }
    }}
    className="flex items-center gap-2 px-3 py-2 rounded-lg bg-cyan-600/20 hover:bg-cyan-600/30 border border-cyan-500/30 text-cyan-400 text-sm transition"
  >
    {action.label}
    <ChevronRight className="w-4 h-4" />
  </button>
)

const LoadingBubble = () => (
  <div className="flex max-w-[85%] items-center gap-3 rounded-2xl border border-border/60 bg-surface/70 px-4 py-3 text-sm text-text-muted shadow-glow">
    <RefreshCw className="w-4 h-4 animate-spin text-cyan-400" />
    <span className="animate-pulse">Analisando dados…</span>
  </div>
)

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN COMPONENT
// ═══════════════════════════════════════════════════════════════════════════════

export const Chat = () => {
  const [input, setInput] = useState('')
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const containerRef = useRef<HTMLDivElement>(null)

  const autoScroll = useCallback(() => {
    requestAnimationFrame(() => {
      if (containerRef.current) {
        containerRef.current.scrollTo({ top: containerRef.current.scrollHeight, behavior: 'smooth' })
      }
    })
  }, [])

  const chatMutation = useMutation<ChatResponse, Error, string>({
    mutationFn: async (message: string) => {
      const response = await api.post('/chat', { message })
      const data = response.data
      if (data?.detail) {
        throw new Error(data.detail)
      }
      return data
    },
    onSuccess: (data) => {
      const assistantMessage: ChatMessage = {
        role: 'assistant',
        content: data.message || data.answer || '',
        kpis: data.kpis || [],
        suggestions: data.suggestions || [],
        actions: data.actions || [],
        intent: data.intent,
        confidence: data.confidence,
      }
      setMessages((prev) => [...prev, assistantMessage])
      autoScroll()
    },
    onError: () => {
      autoScroll()
    },
  })

  const emptyState = useMemo(() => messages.length === 0 && !chatMutation.isPending, [messages, chatMutation.isPending])

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    if (!input.trim() || chatMutation.isPending) return

    const userMessage: ChatMessage = { role: 'user', content: input.trim() }
    setMessages((prev) => [...prev, userMessage])
    setInput('')
    chatMutation.mutate(input.trim())
    autoScroll()
  }

  const handleQuickPrompt = (query: string) => {
    if (chatMutation.isPending) return
    const userMessage: ChatMessage = { role: 'user', content: query }
    setMessages((prev) => [...prev, userMessage])
    setInput('')
    chatMutation.mutate(query)
    autoScroll()
  }

  const handleSuggestion = (suggestion: string) => {
    setInput(suggestion)
  }

  const handleNavigate = (path: string) => {
    window.location.href = path
  }

  useEffect(() => {
    autoScroll()
  }, [messages, autoScroll])

  return (
    <div className="space-y-6">
      {/* Header */}
      <header className="flex items-start justify-between">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-gradient-to-br from-cyan-500/20 to-emerald-500/20 rounded-xl border border-cyan-500/30">
            <MessageCircle className="w-6 h-6 text-cyan-400" />
          </div>
          <div>
            <p className="text-xs uppercase tracking-[0.4em] text-text-muted">Industrial Copilot</p>
            <h2 className="text-2xl font-semibold text-text-primary">Chat Inteligente</h2>
          </div>
        </div>
        
        {/* Quick Prompts */}
        <div className="flex items-center gap-2">
          {quickPrompts.map((prompt) => (
            <button
              key={prompt.label}
              onClick={() => handleQuickPrompt(prompt.query)}
              disabled={chatMutation.isPending}
              className="rounded-xl border border-border bg-surface/70 px-3 py-2 text-xs text-text-muted transition hover:border-nikufra hover:text-text-primary disabled:opacity-50"
            >
              {prompt.label}
            </button>
          ))}
        </div>
      </header>

      {/* Chat Area */}
      <div className="grid gap-6 lg:grid-cols-[2fr,1fr]">
        <div className="flex h-[650px] flex-col rounded-2xl border border-border bg-gradient-to-br from-[#0c1110] via-[#10181a] to-[#0c1413] shadow-glow">
          {/* Messages */}
          <div ref={containerRef} className="flex-1 space-y-4 overflow-y-auto p-6 pr-4">
            {emptyState && (
              <div className="flex h-full flex-col items-center justify-center text-center text-text-muted">
                <span className="mb-4 text-5xl">🤖</span>
                <p className="text-lg font-medium text-text-primary mb-2">Industrial Copilot</p>
                <p className="max-w-lg text-sm">
                  Olá! Sou o assistente industrial do ProdPlan 4.0. Posso ajudar com planeamento, 
                  inventário, digital twin, DPPs e muito mais.
                </p>
                <div className="flex flex-wrap gap-2 mt-6 justify-center">
                  {quickPrompts.map((p) => (
                    <button
                      key={p.label}
                      onClick={() => handleQuickPrompt(p.query)}
                      className="px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-sm text-slate-300 transition"
                    >
                      {p.label}
                    </button>
                  ))}
                </div>
              </div>
            )}

            <AnimatePresence initial={false}>
              {messages.map((message, index) => (
                <motion.div
                  key={`${message.role}-${index}-${message.content.slice(0, 20)}`}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  transition={{ duration: 0.2 }}
                  className={
                    message.role === 'user'
                      ? 'ml-auto max-w-[80%]'
                      : 'max-w-[90%]'
                  }
                >
                  {message.role === 'user' ? (
                    <div className="rounded-2xl border border-nikufra/30 bg-nikufra/10 px-5 py-4 text-sm text-text-primary shadow-[0_0_25px_rgba(69,255,193,0.12)]">
                      <p className="leading-relaxed">{message.content}</p>
                    </div>
                  ) : (
                    <div className="space-y-3">
                      {/* Message content */}
                      <div className="rounded-2xl border border-border/60 bg-black/40 px-5 py-4 text-sm text-text-body shadow-glow">
                        <AnimatedMarkdown content={message.content} />
                      </div>

                      {/* KPIs */}
                      {message.kpis && message.kpis.length > 0 && (
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-2 pl-2">
                          {message.kpis.map((kpi, i) => (
                            <KpiCard key={i} kpi={kpi} />
                          ))}
                        </div>
                      )}

                      {/* Suggestions */}
                      {message.suggestions && message.suggestions.length > 0 && (
                        <div className="flex flex-wrap gap-2 pl-2">
                          {message.suggestions.map((s, i) => (
                            <SuggestionChip key={i} text={s} onClick={() => handleSuggestion(s)} />
                          ))}
                        </div>
                      )}

                      {/* Actions */}
                      {message.actions && message.actions.length > 0 && (
                        <div className="flex flex-wrap gap-2 pl-2">
                          {message.actions.map((action, i) => (
                            <ActionButton key={i} action={action} onNavigate={handleNavigate} />
                          ))}
                        </div>
                      )}

                      {/* Intent badge */}
                      {message.intent && (
                        <div className="flex items-center gap-2 pl-2 text-xs text-slate-500">
                          <span>Intent: {message.intent}</span>
                          {message.confidence && (
                            <span className="px-2 py-0.5 rounded-full bg-slate-800">
                              {Math.round(message.confidence * 100)}% conf
                            </span>
                          )}
                        </div>
                      )}
                    </div>
                  )}
                </motion.div>
              ))}
            </AnimatePresence>

            {chatMutation.isPending && <LoadingBubble />}

            {chatMutation.isError && !chatMutation.isPending && (
              <div className="max-w-[85%] rounded-2xl border border-danger/40 bg-danger/10 px-4 py-3 text-sm text-danger">
                {chatMutation.error.message}
              </div>
            )}
          </div>

          {/* Input */}
          <div className="border-t border-border bg-black/40 p-4">
            <form onSubmit={handleSubmit} className="flex items-end gap-3">
              <textarea
                value={input}
                onChange={(event) => setInput(event.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault()
                    if (input.trim() && !chatMutation.isPending) {
                      const userMessage: ChatMessage = { role: 'user', content: input.trim() }
                      setMessages((prev) => [...prev, userMessage])
                      chatMutation.mutate(input.trim())
                      setInput('')
                      autoScroll()
                    }
                  }
                }}
                placeholder="Pergunta sobre planeamento, inventário, máquinas, DPPs..."
                className="flex-1 resize-none rounded-xl border border-border bg-black/40 px-4 py-3 text-sm text-text-primary outline-none transition focus:border-nikufra h-12 min-h-[48px] max-h-32"
                rows={1}
              />
              <button
                type="submit"
                disabled={!input.trim() || chatMutation.isPending}
                className="flex h-12 w-12 items-center justify-center rounded-xl bg-nikufra text-background shadow-[0_0_30px_rgba(69,255,193,0.25)] transition hover:bg-nikufra-hover hover:shadow-[0_0_35px_rgba(69,255,193,0.4)] disabled:cursor-not-allowed disabled:bg-border"
              >
                <Send className="w-5 h-5" />
              </button>
            </form>
          </div>
        </div>

        {/* Sidebar */}
        <aside className="flex flex-col gap-4">
          <div className="rounded-2xl border border-border bg-surface/60 p-5 shadow-glow">
            <h3 className="text-sm font-semibold text-text-primary flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-cyan-400" />
              Industrial Copilot
            </h3>
            <p className="mt-2 text-sm text-text-muted">
              Assistente inteligente que responde a questões sobre todos os módulos do ProdPlan 4.0.
            </p>
            <div className="mt-4 space-y-2">
              <p className="text-xs uppercase tracking-widest text-slate-500">Áreas Cobertas</p>
              <div className="grid grid-cols-2 gap-2 text-xs text-slate-400">
                <div className="flex items-center gap-1">📊 Planeamento</div>
                <div className="flex items-center gap-1">📦 Inventário</div>
                <div className="flex items-center gap-1">📋 Duplios</div>
                <div className="flex items-center gap-1">🔧 Digital Twin</div>
                <div className="flex items-center gap-1">🔬 R&D</div>
                <div className="flex items-center gap-1">🔗 Causal</div>
              </div>
            </div>
          </div>
          
          <div className="rounded-2xl border border-border bg-surface/60 p-5 shadow-glow">
            <h3 className="text-sm font-semibold text-text-primary">💡 Exemplos de Perguntas</h3>
            <ul className="mt-3 space-y-2 text-sm text-text-muted">
              <li className="cursor-pointer hover:text-cyan-400 transition" onClick={() => handleQuickPrompt('Qual o estado do plano atual?')}>
                • "Qual o estado do plano atual?"
              </li>
              <li className="cursor-pointer hover:text-cyan-400 transition" onClick={() => handleQuickPrompt('Há máquinas em alerta?')}>
                • "Há máquinas em alerta?"
              </li>
              <li className="cursor-pointer hover:text-cyan-400 transition" onClick={() => handleQuickPrompt('Quantos DPPs estão conformes?')}>
                • "Quantos DPPs estão conformes?"
              </li>
              <li className="cursor-pointer hover:text-cyan-400 transition" onClick={() => handleQuickPrompt('Resultados de R&D?')}>
                • "Resultados de R&D?"
              </li>
            </ul>
          </div>
        </aside>
      </div>
    </div>
  )
}
