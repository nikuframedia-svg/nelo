import { useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import { AnimatePresence, motion } from 'framer-motion'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import api from '../utils/api'
import Skeleton from 'react-loading-skeleton'
import 'react-loading-skeleton/dist/skeleton.css'
import { useNavigate } from 'react-router-dom'

interface SuggestionItem {
  id: string
  icon: string
  action: string
  explanation: string
  impact: string
  impact_level: 'baixo' | 'medio' | 'alto'
  gain: string
  reasoning_markdown: string
  data_points: Record<string, unknown>
  dados_tecnicos?: Record<string, unknown> // Legacy field
}

interface SuggestionsResponse {
  count: number
  items: SuggestionItem[]
  detail?: string
}

const impactColor = {
  alto: 'from-[#ff4d6d]/20 to-[#ff4d6d]/5 text-danger',
  medio: 'from-[#ffb347]/20 to-[#ffb347]/5 text-warning',
  baixo: 'from-[#45ffc1]/20 to-[#45ffc1]/5 text-nikufra',
}

export const Suggestions = () => {
  const navigate = useNavigate()
  const { data, isLoading, error } = useQuery<SuggestionsResponse>({
    queryKey: ['suggestions', 'resumo'],
    queryFn: async () => {
      const response = await api.get('/suggestions/', { params: { mode: 'resumo' } })
      return response.data
    },
    staleTime: 30_000,
  })

  const suggestions = useMemo(() => data?.items ?? [], [data])
  const offlineDetail = data?.detail

  return (
    <div className="space-y-8">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-xs uppercase tracking-[0.4em] text-text-muted">IA operacional</p>
          <h2 className="text-2xl font-semibold text-text-primary">Sugestões Inteligentes</h2>
          <p className="mt-2 max-w-3xl text-sm text-text-muted">
            Recomendações geradas automaticamente com base nos dados importados. Cada ação inclui racional técnico, impacto
            previsto e dados de suporte para ativar simulações What-If.
          </p>
        </div>
        <button
          onClick={() => navigate('/whatif')}
          className="rounded-2xl border border-nikufra bg-nikufra/10 px-4 py-2 text-sm font-semibold text-nikufra transition hover:bg-nikufra hover:text-background"
        >
          Abrir simulações What-If
        </button>
      </div>

      {isLoading && (
        <div className="grid gap-6 md:grid-cols-2 xl:grid-cols-3">
          {Array.from({ length: 6 }).map((_, index) => (
            <Skeleton key={index} height={280} baseColor="#121212" highlightColor="#1c1c1c" />
          ))}
        </div>
      )}

      {(error || offlineDetail) && (
        <div className="rounded-2xl border border-warning/40 bg-warning/10 p-4 text-sm text-warning">
          {offlineDetail ?? error?.message ?? 'Não foi possível carregar as sugestões.'}
        </div>
      )}

      {!isLoading && !error && !offlineDetail && suggestions.length === 0 && (
        <div className="rounded-2xl border border-border bg-surface p-8 text-center text-text-muted">
          Ainda não existem sugestões. Carregue os ficheiros de produção e stocks para ativar o motor de recomendações.
        </div>
      )}

      <AnimatePresence>
        <div className="grid gap-6 md:grid-cols-2 xl:grid-cols-3">
          {suggestions.map((suggestion) => (
            <motion.div
              key={suggestion.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3, ease: 'easeOut' }}
              className={`relative overflow-hidden rounded-2xl border border-border bg-gradient-to-br ${impactColor[suggestion.impact_level]} p-6 shadow-[0_0_32px_rgba(69,255,193,0.1)]`}
            >
              <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_right,rgba(69,255,193,0.18),transparent_55%)]" />
              <div className="relative flex h-full flex-col gap-4">
                <div className="flex items-start justify-between">
                  <span className="text-3xl drop-shadow-[0_0_15px_rgba(69,255,193,0.3)]">{suggestion.icon}</span>
                  <span className="rounded-full bg-black/40 px-3 py-1 text-xs font-semibold uppercase">
                    Impacto {suggestion.impact_level}
                  </span>
                </div>

                <div>
                  <h3 className="text-lg font-semibold text-text-primary">{suggestion.action}</h3>
                  <p className="mt-2 text-sm text-text-muted">{suggestion.explanation}</p>
                </div>

                <div className="space-y-2 text-sm text-text-primary">
                  <p className="flex items-center justify-between rounded-xl border border-border/60 bg-black/30 px-4 py-2">
                    <span className="text-text-muted">Impacto</span>
                    <span className="font-semibold text-nikufra">{suggestion.impact}</span>
                  </p>
                  <p className="flex items-center justify-between rounded-xl border border-border/60 bg-black/30 px-4 py-2">
                    <span className="text-text-muted">Ganho estimado</span>
                    <span className="font-semibold text-nikufra">{suggestion.gain}</span>
                  </p>
                </div>

                <div className="mt-auto space-y-3 text-xs text-text-muted">
                  <details className="group rounded-2xl border border-border/60 bg-black/30 p-4 transition hover:border-nikufra/60">
                    <summary className="flex cursor-pointer items-center justify-between text-sm font-semibold text-text-primary">
                      Porquê sugeri isto?
                      <span className="text-nikufra">▼</span>
                    </summary>
                    <div className="mt-3 space-y-3 text-sm leading-relaxed text-text-muted">
                      <ReactMarkdown remarkPlugins={[remarkGfm]} className="prose prose-invert prose-sm max-w-none">
                        {suggestion.reasoning_markdown}
                      </ReactMarkdown>
                    </div>
                  </details>
                  {(suggestion.dados_tecnicos || suggestion.data_points) && (
                    <details className="group rounded-2xl border border-border/60 bg-black/30 p-4 transition hover:border-nikufra/60">
                      <summary className="flex cursor-pointer items-center justify-between text-sm font-semibold text-text-primary">
                        Dados técnicos
                        <span className="text-nikufra">▼</span>
                      </summary>
                      <div className="mt-3">
                        <pre className="max-h-48 overflow-auto rounded-xl bg-background/60 p-3 text-[11px]">
{JSON.stringify(suggestion.dados_tecnicos || suggestion.data_points, null, 2)}
                        </pre>
                      </div>
                    </details>
                  )}
                </div>

                <button
                  onClick={() => navigate('/whatif', { state: { suggestionId: suggestion.id } })}
                  className="inline-flex items-center justify-center gap-2 rounded-2xl border border-nikufra bg-nikufra/10 px-4 py-2 text-sm font-semibold text-nikufra transition hover:bg-nikufra hover:text-background"
                >
                  Simular What-If
                  <span>➜</span>
                </button>
              </div>
            </motion.div>
          ))}
        </div>
      </AnimatePresence>
    </div>
  )
}
