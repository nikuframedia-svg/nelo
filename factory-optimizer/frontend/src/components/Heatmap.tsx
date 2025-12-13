import React from 'react';
import { tokens } from '../theme/tokens';

interface HeatmapProps {
  data: Array<{
    recurso: string;
    utilizacao: Array<{
      hora: number;
      utilizacao_pct: number;
    }>;
  }>;
}

const cellColor = (value: number) => {
  if (value < 70) return 'bg-nikufra/20 text-nikufra'
  if (value < 90) return 'bg-warning/20 text-warning'
  return 'bg-danger/30 text-danger'
}

const cellLabel = (value: number) => {
  if (value < 70) return 'Zona folgada — capacidade disponível'
  if (value < 90) return 'Zona de atenção — monitorizar setup e prioridades'
  return 'Zona crítica — risco de fila e incumprimento'
}

export const Heatmap = ({ data }: HeatmapProps) => {
  // Guard against undefined or empty data
  if (!data || !data.length) return null
  
  // Guard against missing utilizacao array
  const firstRow = data[0]
  const headerSlots = firstRow?.utilizacao ?? []

  return (
    <div className="rounded-2xl border border-border bg-surface shadow-glow">
      <div className="flex items-center justify-between px-6 py-5">
        <h3 className="text-base font-semibold text-text-primary">Utilização de Recursos</h3>
        <span className="text-xs text-text-muted">Verde → livre, Vermelho → saturado</span>
      </div>
      <div className="overflow-x-auto px-6 pb-6">
        <table className="w-full border-separate border-spacing-2 text-sm">
          <thead className="text-xs uppercase tracking-wide text-text-muted">
            <tr>
              <th className="w-32 text-left">Recurso</th>
              {headerSlots.map((slot) => (
                <th key={slot.hora} className="text-center font-normal">
                  {slot.hora}h
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.map((row) => (
              <tr key={row.recurso}>
                <td className="whitespace-nowrap text-left text-text-primary">{row.recurso}</td>
                {(row.utilizacao ?? []).map((slot) => (
                  <td key={slot.hora} className="text-center">
                    <span
                      title={`${row.recurso} às ${slot.hora}h — utilização ${slot.utilizacao_pct.toFixed(1)}%. ${cellLabel(slot.utilizacao_pct)}.`}
                      className={`inline-flex min-w-[52px] items-center justify-center rounded-xl px-2 py-2 text-xs font-semibold transition hover:scale-[1.03] ${cellColor(slot.utilizacao_pct)}`}
                    >
                      {slot.utilizacao_pct.toFixed(0)}%
                    </span>
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

