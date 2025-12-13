import { motion } from 'framer-motion'
import clsx from 'clsx'

interface KPICardProps {
  title: string
  value: string | number
  delta?: number
  unit?: string
  format?: 'number' | 'percentage' | 'text'
  tooltip?: string
}

const formatValue = (val: string | number, format: KPICardProps['format']) => {
  if (format === 'percentage' && typeof val === 'number') {
    return `${val.toFixed(1)}%`
  }
  if (format === 'number' && typeof val === 'number') {
    return val.toLocaleString('pt-PT', { minimumFractionDigits: 1, maximumFractionDigits: 1 })
  }
  return String(val)
}

export const KPICard = ({ title, value, delta, unit, format = 'number', tooltip }: KPICardProps) => {
  const formatted = formatValue(value, format)
  const showDelta = delta !== undefined && !Number.isNaN(delta)
  const deltaPositive = (delta ?? 0) >= 0

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      whileHover={{ scale: 1.02, boxShadow: '0 0 32px rgba(69,255,193,0.18)' }}
      transition={{ duration: 0.25 }}
      className="relative flex flex-col gap-3 overflow-hidden rounded-2xl border border-border bg-gradient-to-br from-[#0b1010] via-[#0f1a19] to-[#0d1412] p-6 shadow-glow"
    >
      <div className="flex items-center gap-2 text-xs uppercase tracking-[0.24em] text-text-muted">
        <span>{title}</span>
        {tooltip && (
          <span
            title={tooltip}
            className="cursor-help rounded-full border border-border/60 px-[6px] py-[2px] text-[10px] font-semibold text-text-muted/80 transition hover:border-nikufra/60 hover:text-nikufra"
          >
            ⓘ
          </span>
        )}
      </div>
      <div className="flex items-baseline gap-2 text-text-primary">
        <span className="font-mono text-3xl font-semibold drop-shadow-[0_0_18px_rgba(69,255,193,0.15)]">{formatted}</span>
        {unit && <span className="text-sm text-text-muted">{unit}</span>}
      </div>
      {showDelta && (
        <span
          className={clsx(
            'inline-flex w-fit items-center gap-1 rounded-full px-3 py-1 text-xs font-semibold transition',
            deltaPositive ? 'bg-nikufra/15 text-nikufra' : 'bg-danger/15 text-danger',
          )}
        >
          {deltaPositive ? '⬆' : '⬇'} {deltaPositive ? '+' : ''}
          {(delta ?? 0).toFixed(1)}%
        </span>
      )}
    </motion.div>
  )
}

