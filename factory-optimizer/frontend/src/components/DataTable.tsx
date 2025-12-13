import clsx from 'clsx'
import { ReactNode } from 'react'

interface Column<T> {
  key: keyof T | string
  header: string
  render?: (row: T) => ReactNode
  align?: 'left' | 'center' | 'right'
  tooltip?: string
}

interface DataTableProps<T> {
  columns: Column<T>[]
  data: T[]
  onRowClick?: (row: T) => void
}

export function DataTable<T extends Record<string, unknown>>({ columns, data, onRowClick }: DataTableProps<T>) {
  return (
    <div className="overflow-hidden rounded-2xl border border-border/60 bg-gradient-to-br from-surface to-surface/80 shadow-[0_0_32px_rgba(69,255,193,0.08)] backdrop-blur-sm">
      <div className="overflow-x-auto">
        <table className="w-full border-separate border-spacing-0 text-sm">
          <thead className="sticky top-0 z-10 bg-gradient-to-b from-surface/95 to-surface/90 backdrop-blur-md">
            <tr>
              {columns.map((col) => (
                <th
                  key={String(col.key)}
                  className={clsx(
                    'border-b border-border/80 px-5 py-4 text-xs font-bold uppercase tracking-wider text-text-primary',
                    'bg-gradient-to-b from-surface/95 to-surface/90',
                    col.align === 'center' && 'text-center',
                    col.align === 'right' && 'text-right',
                    'group relative transition-colors',
                  )}
                >
                  <span className="inline-flex items-center gap-2">
                    {col.header}
                    {col.tooltip && (
                      <span
                        title={col.tooltip}
                        className="cursor-help rounded-full border border-nikufra/30 bg-nikufra/10 px-[6px] py-[2px] text-[10px] font-semibold text-nikufra/70 transition-all hover:border-nikufra/60 hover:bg-nikufra/20 hover:text-nikufra"
                      >
                        ⓘ
                      </span>
                    )}
                  </span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.map((row, index) => (
              <tr
                key={index}
                onClick={() => onRowClick?.(row)}
                className={clsx(
                  'border-b border-border/40 transition-all duration-200',
                  index % 2 === 0 
                    ? 'bg-surface/50' 
                    : 'bg-surface/30',
                  onRowClick 
                    ? 'cursor-pointer hover:bg-nikufra/10 hover:shadow-[0_0_16px_rgba(69,255,193,0.1)]' 
                    : 'hover:bg-surface/70',
                  'group',
                )}
              >
                {columns.map((col) => (
                  <td
                    key={String(col.key)}
                    className={clsx(
                      'px-5 py-4 text-text-body transition-colors group-hover:text-text-primary',
                      col.align === 'center' && 'text-center',
                      col.align === 'right' && 'text-right',
                    )}
                  >
                    {col.render ? col.render(row) : String(row[col.key as keyof T] ?? '')}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {data.length === 0 && (
        <div className="px-6 py-12 text-center text-sm text-text-muted">
          <p className="text-text-muted/60">Nenhum dado disponível</p>
        </div>
      )}
    </div>
  )
}

