import { useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import Skeleton from 'react-loading-skeleton'
import 'react-loading-skeleton/dist/skeleton.css'
import { KPICard } from '../components/KPICard'
import { Heatmap } from '../components/Heatmap'
import { DataTable } from '../components/DataTable'
import api from '../utils/api'
import { BottleneckResponse, Bottleneck } from '../types'

export const Bottlenecks = () => {
  const [appliedActions, setAppliedActions] = useState<Record<string, string>>({})

  const { data, isLoading, error } = useQuery<BottleneckResponse>({
    queryKey: ['bottlenecks'],
    queryFn: async () => {
      const response = await api.get('/bottlenecks/')
      return response.data
    },
    staleTime: 30_000,
  })

  const appliedCount = useMemo(() => Object.keys(appliedActions).length, [appliedActions])

  const safeData: BottleneckResponse = useMemo(
    () =>
      data ?? {
        bottlenecks: [],
        overlap_applied: { transformacao: 0, acabamentos: 0, embalagem: 0 },
        lead_time_gain: 0,
        heatmap: [],
        top_losses: [],
        demo_mode: false,
      },
    [data],
  )

  const handleAction = (row: Bottleneck, action: string) => {
    setAppliedActions((prev) => ({ ...prev, [row.recurso]: action }))
  }

  return (
    <div className="space-y-8">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <p className="text-xs uppercase tracking-[0.4em] text-text-muted">Balanceamento dinâmico</p>
          <h2 className="text-2xl font-semibold text-text-primary">Gargalos & Overlap</h2>
          <p className="mt-2 max-w-3xl text-sm text-text-muted">
            Monitoriza a utilização prevista por recurso, identifica perdas e visualiza o impacto do overlap aplicado em
            cada etapa.
          </p>
        </div>
        {appliedCount > 0 && (
          <span className="rounded-full border border-nikufra/60 bg-nikufra/10 px-4 py-1 text-xs font-semibold text-nikufra">
            ✔️ {appliedCount} ações aplicadas nesta sessão
          </span>
        )}
      </div>

      {error && (
        <div className="rounded-2xl border border-danger/40 bg-danger/10 p-4 text-sm text-danger">
          Não foi possível carregar os gargalos. {String(error)}
        </div>
      )}

      {isLoading ? (
        <div className="grid gap-4 md:grid-cols-3">
          {Array.from({ length: 3 }).map((_, index) => (
            <Skeleton key={index} height={140} baseColor="#121212" highlightColor="#1c1c1c" />
          ))}
        </div>
      ) : (
        <div className="grid gap-4 md:grid-cols-3">
          <KPICard
            title="Overlap Transformação"
            value={safeData.overlap_applied.transformacao * 100}
            format="percentage"
            tooltip="Percentagem de overlap aplicada nas operações de Transformação."
          />
          <KPICard
            title="Overlap Acabamentos"
            value={safeData.overlap_applied.acabamentos * 100}
            format="percentage"
            tooltip="Overlap aplicado na etapa de Acabamentos para reduzir filas."
          />
          <KPICard
            title="Ganho de Lead Time"
            value={safeData.lead_time_gain}
            unit="%"
            tooltip="Redução percentual do lead time face ao baseline após aplicar overlap."
          />
        </div>
      )}

      {isLoading ? (
        <Skeleton height={320} baseColor="#121212" highlightColor="#1c1c1c" />
      ) : (
        safeData.heatmap.length > 0 && (
          <div className="space-y-3">
            {safeData.demo_mode && (
              <div className="rounded-xl border-2 border-warning/40 bg-warning/10 p-3 text-sm text-warning">
                ⚠️ Modo DEMO ativo: Os recursos 27, 29 e 248 aparecem saturados para simular um cenário realista de gargalo.
              </div>
            )}
            <p className="text-xs text-text-muted">
              Vermelho = saturado (&gt;90% de carga), Verde = folgado (&lt;70%). Passa o cursor para ver o detalhe.
            </p>
            <Heatmap data={safeData.heatmap} />
          </div>
        )
      )}

      {isLoading ? (
        <Skeleton height={280} baseColor="#121212" highlightColor="#1c1c1c" />
      ) : (
        safeData.top_losses.length > 0 && (
          <div className="space-y-4">
            <h3 className="text-base font-semibold text-text-primary">Top 5 perdas e ações recomendadas</h3>
            <DataTable<Bottleneck>
              columns={[
                { key: 'recurso', header: 'Recurso' },
                {
                  key: 'utilizacao_pct',
                  header: 'Utilização %',
                  align: 'center',
                  tooltip: 'Carga prevista na janela analisada.',
                  render: (row) => `${row.utilizacao_pct.toFixed(1)}%`,
                },
                {
                  key: 'fila_horas',
                  header: 'Fila (h)',
                  align: 'center',
                  tooltip: 'Horas de fila acumuladas se nada for feito.',
                  render: (row) => `${row.fila_horas.toFixed(1)} h`,
                },
                {
                  key: 'probabilidade',
                  header: 'Prob. Gargalo',
                  align: 'center',
                  tooltip: 'Probabilidade estimada de o recurso ser gargalo.',
                  render: (row) => `${(row.probabilidade * 100).toFixed(1)}%`,
                },
                {
                  key: 'acao',
                  header: 'Plano sugerido',
                  tooltip: 'Ação automática sugerida pelo motor APS.',
                  render: (row) => (
                    <div className="space-y-1">
                      <p className="text-sm text-text-primary">{row.acao}</p>
                      {appliedActions[row.recurso] && (
                        <span className="inline-flex items-center gap-1 rounded-full bg-nikufra/15 px-2 py-[2px] text-[10px] font-semibold text-nikufra">
                          ✔️ {appliedActions[row.recurso]}
                        </span>
                      )}
                    </div>
                  ),
                },
                {
                  key: 'impacto',
                  header: 'Impacto estimado',
                  tooltip: 'Variação prevista em OTD e fila se aplicar a ação.',
                  render: (row) => (
                    <span className="text-sm text-text-muted">
                      OTD: {row.impacto_otd.toFixed(1)}pp • Horas: {row.impacto_horas.toFixed(1)}h
                    </span>
                  ),
                },
                {
                  key: 'cta',
                  header: 'One-click',
                  tooltip: 'Aplica a ação sugerida e ajusta o Gantt Depois.',
                  render: (row) => (
                    <div className="flex flex-wrap gap-2">
                      <button
                        type="button"
                        onClick={() => handleAction(row, 'Rota alternativa aplicada')}
                        className="rounded-full border border-border px-3 py-1 text-xs text-text-muted transition hover:border-nikufra hover:text-nikufra"
                      >
                        Mover para alternativa
                      </button>
                      <button
                        type="button"
                        onClick={() => handleAction(row, 'Famílias coladas')}
                        className="rounded-full border border-border px-3 py-1 text-xs text-text-muted transition hover:border-nikufra hover:text-nikufra"
                      >
                        Colar famílias
                      </button>
                      <button
                        type="button"
                        onClick={() => handleAction(row, 'Preventiva agendada')}
                        className="rounded-full border border-border px-3 py-1 text-xs text-text-muted transition hover:border-nikufra hover:text-nikufra"
                      >
                        Agendar preventiva 20 min
                      </button>
                    </div>
                  ),
                },
              ]}
              data={safeData.top_losses}
            />
          </div>
        )
      )}
    </div>
  )
}

