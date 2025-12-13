import type { FC } from 'react'

type NewMachine = {
  machine_id?: string
  description?: string
  group?: string
  speed_factor_delta?: number
}

type UpdatedTime = {
  article_id?: string
  op_code?: string
  time_factor?: number
}

type UpdatedShift = {
  machine_id?: string
  shift_id?: string
  change?: string
}

export type ScenarioPreviewData = {
  new_machines?: NewMachine[]
  updated_times?: UpdatedTime[]
  updated_shifts?: UpdatedShift[]
  error?: string
  raw_response?: string
}

interface ScenarioPreviewProps {
  data: ScenarioPreviewData | null
  loading?: boolean
}

export const ScenarioPreview: FC<ScenarioPreviewProps> = ({ data, loading }) => {
  if (loading) {
    return (
      <CardWrapper title="Pré-visualização do cenário">
        <p className="text-sm text-text-muted">A processar descrição...</p>
      </CardWrapper>
    )
  }

  if (!data) {
    return (
      <CardWrapper title="Pré-visualização do cenário">
        <p className="text-sm text-text-muted">
          Descreve o cenário para veres as alterações propostas (máquinas, tempos e turnos).
        </p>
      </CardWrapper>
    )
  }

  return (
    <CardWrapper title="Pré-visualização do cenário">
      {data.error && (
        <p className="mb-4 rounded-xl border border-warning/40 bg-warning/10 px-4 py-2 text-sm text-warning">
          {data.error}
        </p>
      )}

      <Section title="Novas máquinas" fallback="Sem novas máquinas">
        {(data.new_machines ?? []).map((machine, index) => (
          <Item key={`${machine.machine_id}-${index}`} label={machine.machine_id || 'Sem ID'}>
            <p className="text-xs text-text-muted">
              {machine.description || 'Sem descrição'} • Grupo: {machine.group || '—'} • Fator velocidade:{' '}
              {machine.speed_factor_delta ?? 1}
            </p>
          </Item>
        ))}
      </Section>

      <Section title="Tempos atualizados" fallback="Sem ajustes de tempos">
        {(data.updated_times ?? []).map((update, index) => (
          <Item key={`${update.article_id}-${update.op_code}-${index}`} label={`Artigo ${update.article_id || '—'}`}>
            <p className="text-xs text-text-muted">
              Operação {update.op_code || '—'} • Fator tempo: {update.time_factor ?? 1}
            </p>
          </Item>
        ))}
      </Section>

      <Section title="Turnos ajustados" fallback="Sem alterações a turnos">
        {(data.updated_shifts ?? []).map((shift, index) => (
          <Item key={`${shift.machine_id}-${shift.shift_id}-${index}`} label={`Máquina ${shift.machine_id || '—'}`}>
            <p className="text-xs text-text-muted">
              Turno {shift.shift_id || '—'} • Alteração: {shift.change || '—'}
            </p>
          </Item>
        ))}
      </Section>
    </CardWrapper>
  )
}

const CardWrapper: FC<{ title: string; children: React.ReactNode }> = ({ title, children }) => (
  <div className="rounded-2xl border border-border bg-surface p-6 shadow-glow">
    <h3 className="text-sm font-semibold text-text-primary">{title}</h3>
    <div className="mt-4 space-y-4">{children}</div>
  </div>
)

const Section: FC<{ title: string; fallback: string; children: React.ReactNode }> = ({
  title,
  fallback,
  children,
}) => {
  const hasChildren = Array.isArray(children) ? children.length > 0 : Boolean(children)
  return (
    <div>
      <p className="text-xs font-semibold uppercase tracking-[0.3em] text-text-muted">{title}</p>
      <div className="mt-2 space-y-2">
        {hasChildren ? children : <p className="text-sm text-text-muted">{fallback}</p>}
      </div>
    </div>
  )
}

const Item: FC<{ label: string; children: React.ReactNode }> = ({ label, children }) => (
  <div className="rounded-xl border border-border/60 bg-background/70 p-3 text-sm text-text-primary">
    <p className="font-semibold">{label}</p>
    {children}
  </div>
)
