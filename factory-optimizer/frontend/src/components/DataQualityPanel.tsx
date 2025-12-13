/**
 * Data Quality Panel - SNR (Signal-to-Noise Ratio) Analysis
 * 
 * Displays data quality metrics for production planning:
 * - Global SNR score
 * - SNR by machine
 * - SNR by operation type
 * - Quality warnings and recommendations
 */

import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { motion } from 'framer-motion';
import {
  Activity,
  AlertTriangle,
  BarChart3,
  CheckCircle,
  ChevronDown,
  ChevronRight,
  Cpu,
  Gauge,
  Info,
  Loader2,
  RefreshCw,
  Settings,
  TrendingUp,
  XCircle,
  Zap,
} from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

// ═══════════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════════

interface SNRMachineData {
  machine_id: string;
  snr: number;
  snr_level: string;
  sample_count: number;
  mean_time: number;
  std_time: number;
}

interface SNROperationData {
  operation_type: string;
  snr: number;
  snr_level: string;
  sample_count: number;
}

interface DataQualityReport {
  global_snr: number;
  global_snr_level: string;
  quality_level: string;
  snr_by_machine: SNRMachineData[];
  snr_by_operation: SNROperationData[];
  low_snr_warnings: string[];
  recommendations: string[];
  total_operations: number;
  total_machines: number;
}

// ═══════════════════════════════════════════════════════════════════════════════
// API
// ═══════════════════════════════════════════════════════════════════════════════

async function fetchDataQuality(): Promise<DataQualityReport> {
  const res = await fetch(`${API_BASE}/plan/data_quality`);
  if (!res.ok) throw new Error('Failed to fetch data quality');
  return res.json();
}

// ═══════════════════════════════════════════════════════════════════════════════
// COMPONENTS
// ═══════════════════════════════════════════════════════════════════════════════

const SNRGauge: React.FC<{ value: number; size?: 'sm' | 'lg' }> = ({ value, size = 'lg' }) => {
  const percentage = Math.min(100, Math.max(0, value * 10)); // Scale SNR to 0-100
  const radius = size === 'lg' ? 70 : 40;
  const stroke = size === 'lg' ? 10 : 6;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (percentage / 100) * circumference;

  const getColor = (snr: number) => {
    if (snr > 10) return '#10b981'; // Excellent
    if (snr > 3) return '#22c55e';  // Good
    if (snr > 1) return '#f59e0b';  // Fair
    return '#ef4444';               // Poor
  };

  const getLevel = (snr: number) => {
    if (snr > 10) return 'EXCELENTE';
    if (snr > 3) return 'BOM';
    if (snr > 1) return 'RAZOÁVEL';
    return 'FRACO';
  };

  return (
    <div className="relative inline-flex items-center justify-center">
      <svg width={(radius + stroke) * 2} height={(radius + stroke) * 2} className="-rotate-90">
        <circle
          cx={radius + stroke}
          cy={radius + stroke}
          r={radius}
          fill="none"
          stroke="currentColor"
          strokeWidth={stroke}
          className="text-slate-700"
        />
        <motion.circle
          cx={radius + stroke}
          cy={radius + stroke}
          r={radius}
          fill="none"
          stroke={getColor(value)}
          strokeWidth={stroke}
          strokeLinecap="round"
          strokeDasharray={circumference}
          initial={{ strokeDashoffset: circumference }}
          animate={{ strokeDashoffset: offset }}
          transition={{ duration: 1, ease: 'easeOut' }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className={`${size === 'lg' ? 'text-3xl' : 'text-lg'} font-bold`} style={{ color: getColor(value) }}>
          {value.toFixed(1)}
        </span>
        <span className={`${size === 'lg' ? 'text-sm' : 'text-xs'} text-slate-400`}>SNR</span>
        {size === 'lg' && (
          <span className="text-xs font-medium mt-1" style={{ color: getColor(value) }}>
            {getLevel(value)}
          </span>
        )}
      </div>
    </div>
  );
};

const SNRBar: React.FC<{ snr: number; maxSNR?: number }> = ({ snr, maxSNR = 15 }) => {
  const percentage = Math.min(100, (snr / maxSNR) * 100);
  
  const getColor = (val: number) => {
    if (val > 10) return 'bg-emerald-500';
    if (val > 3) return 'bg-green-500';
    if (val > 1) return 'bg-amber-500';
    return 'bg-red-500';
  };

  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-2 bg-slate-700 rounded-full overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${percentage}%` }}
          transition={{ duration: 0.5 }}
          className={`h-full ${getColor(snr)}`}
        />
      </div>
      <span className="text-xs text-slate-400 w-12 text-right">{snr.toFixed(1)}</span>
    </div>
  );
};

const QualityLevelBadge: React.FC<{ level: string }> = ({ level }) => {
  const config: Record<string, { bg: string; text: string; icon: React.ReactNode }> = {
    EXCELENTE: { bg: 'bg-emerald-500/20', text: 'text-emerald-400', icon: <CheckCircle className="w-4 h-4" /> },
    EXCELLENT: { bg: 'bg-emerald-500/20', text: 'text-emerald-400', icon: <CheckCircle className="w-4 h-4" /> },
    BOM: { bg: 'bg-green-500/20', text: 'text-green-400', icon: <CheckCircle className="w-4 h-4" /> },
    GOOD: { bg: 'bg-green-500/20', text: 'text-green-400', icon: <CheckCircle className="w-4 h-4" /> },
    RAZOÁVEL: { bg: 'bg-amber-500/20', text: 'text-amber-400', icon: <AlertTriangle className="w-4 h-4" /> },
    FAIR: { bg: 'bg-amber-500/20', text: 'text-amber-400', icon: <AlertTriangle className="w-4 h-4" /> },
    FRACO: { bg: 'bg-red-500/20', text: 'text-red-400', icon: <XCircle className="w-4 h-4" /> },
    POOR: { bg: 'bg-red-500/20', text: 'text-red-400', icon: <XCircle className="w-4 h-4" /> },
  };

  const c = config[level.toUpperCase()] || config.FAIR;

  return (
    <span className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-sm font-medium ${c.bg} ${c.text}`}>
      {c.icon}
      {level}
    </span>
  );
};

const MachineCard: React.FC<{ machine: SNRMachineData }> = ({ machine }) => (
  <motion.div
    initial={{ opacity: 0, y: 10 }}
    animate={{ opacity: 1, y: 0 }}
    className="bg-slate-800/50 rounded-lg p-4 border border-slate-700/50"
  >
    <div className="flex items-center justify-between mb-3">
      <div className="flex items-center gap-2">
        <Cpu className="w-4 h-4 text-cyan-400" />
        <span className="font-medium text-white">{machine.machine_id}</span>
      </div>
      <QualityLevelBadge level={machine.snr_level} />
    </div>
    <SNRBar snr={machine.snr} />
    <div className="flex justify-between mt-2 text-xs text-slate-500">
      <span>{machine.sample_count} amostras</span>
      <span>μ={machine.mean_time?.toFixed(1) || 'N/A'}min σ={machine.std_time?.toFixed(1) || 'N/A'}</span>
    </div>
  </motion.div>
);

const OperationRow: React.FC<{ operation: SNROperationData }> = ({ operation }) => (
  <tr className="border-b border-slate-700/50 hover:bg-slate-800/30">
    <td className="py-3 px-2">
      <span className="font-medium text-white">{operation.operation_type}</span>
    </td>
    <td className="py-3 px-2 w-48">
      <SNRBar snr={operation.snr} />
    </td>
    <td className="py-3 px-2 text-center">
      <QualityLevelBadge level={operation.snr_level} />
    </td>
    <td className="py-3 px-2 text-right text-slate-400">
      {operation.sample_count}
    </td>
  </tr>
);

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN COMPONENT
// ═══════════════════════════════════════════════════════════════════════════════

export const DataQualityPanel: React.FC = () => {
  const [showMachines, setShowMachines] = useState(true);
  const [showOperations, setShowOperations] = useState(true);

  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ['data-quality'],
    queryFn: fetchDataQuality,
    staleTime: 60_000,
  });

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="w-8 h-8 text-cyan-400 animate-spin" />
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="rounded-xl border border-amber-500/30 bg-amber-500/10 p-6 text-center">
        <AlertTriangle className="w-12 h-12 text-amber-400 mx-auto mb-3" />
        <p className="text-amber-400 font-medium">Não foi possível carregar dados de qualidade</p>
        <p className="text-sm text-amber-400/70 mt-1">Verifique se existem dados de planeamento</p>
        <button
          onClick={() => refetch()}
          className="mt-4 px-4 py-2 bg-amber-500/20 hover:bg-amber-500/30 text-amber-400 rounded-lg text-sm"
        >
          Tentar novamente
        </button>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold text-white flex items-center gap-2">
            <Activity className="w-5 h-5 text-cyan-500" />
            Qualidade dos Dados (SNR)
          </h2>
          <p className="text-sm text-slate-400 mt-1">
            Signal-to-Noise Ratio mede a previsibilidade dos tempos de operação
          </p>
        </div>
        <button
          onClick={() => refetch()}
          className="flex items-center gap-2 px-3 py-2 bg-slate-800 hover:bg-slate-700 rounded-lg text-slate-300 text-sm"
        >
          <RefreshCw className="w-4 h-4" />
          Atualizar
        </button>
      </div>

      {/* Global SNR + Stats */}
      <div className="grid grid-cols-12 gap-6">
        <div className="col-span-4 flex flex-col items-center justify-center p-6 rounded-2xl border border-slate-700/50 bg-gradient-to-br from-slate-800/50 to-slate-900/50">
          <SNRGauge value={data.global_snr} />
          <p className="mt-4 text-lg font-medium text-white">SNR Global</p>
          <QualityLevelBadge level={data.quality_level} />
        </div>

        <div className="col-span-8 grid grid-cols-3 gap-4">
          <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700/50">
            <div className="flex items-center gap-2 text-slate-400 text-sm mb-2">
              <Cpu className="w-4 h-4" />
              Máquinas
            </div>
            <p className="text-2xl font-bold text-white">{data.total_machines}</p>
            <p className="text-xs text-slate-500">monitorizadas</p>
          </div>
          <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700/50">
            <div className="flex items-center gap-2 text-slate-400 text-sm mb-2">
              <Settings className="w-4 h-4" />
              Operações
            </div>
            <p className="text-2xl font-bold text-white">{data.total_operations}</p>
            <p className="text-xs text-slate-500">analisadas</p>
          </div>
          <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700/50">
            <div className="flex items-center gap-2 text-slate-400 text-sm mb-2">
              <AlertTriangle className="w-4 h-4" />
              Alertas
            </div>
            <p className="text-2xl font-bold text-amber-400">{data.low_snr_warnings?.length || 0}</p>
            <p className="text-xs text-slate-500">baixo SNR</p>
          </div>
        </div>
      </div>

      {/* SNR Explanation */}
      <div className="rounded-xl border border-cyan-500/20 bg-cyan-500/5 p-4">
        <div className="flex items-start gap-3">
          <Info className="w-5 h-5 text-cyan-400 mt-0.5" />
          <div>
            <p className="font-medium text-cyan-400">O que é SNR (Signal-to-Noise Ratio)?</p>
            <p className="text-sm text-slate-400 mt-1">
              SNR = Var(sinal) / Var(ruído). Mede a previsibilidade dos tempos de operação.
              <br />
              <strong>SNR &gt; 10:</strong> Excelente | <strong>3-10:</strong> Bom | <strong>1-3:</strong> Razoável | <strong>&lt; 1:</strong> Fraco
            </p>
          </div>
        </div>
      </div>

      {/* Warnings */}
      {data.low_snr_warnings && data.low_snr_warnings.length > 0 && (
        <div className="rounded-xl border border-amber-500/30 bg-amber-500/10 p-4">
          <h3 className="font-medium text-amber-400 mb-3 flex items-center gap-2">
            <AlertTriangle className="w-4 h-4" />
            Alertas de Baixa Qualidade
          </h3>
          <ul className="space-y-2">
            {data.low_snr_warnings.map((warning, i) => (
              <li key={i} className="flex items-start gap-2 text-sm text-amber-300/80">
                <ChevronRight className="w-4 h-4 mt-0.5 flex-shrink-0" />
                {warning}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Recommendations */}
      {data.recommendations && data.recommendations.length > 0 && (
        <div className="rounded-xl border border-emerald-500/30 bg-emerald-500/10 p-4">
          <h3 className="font-medium text-emerald-400 mb-3 flex items-center gap-2">
            <Zap className="w-4 h-4" />
            Recomendações
          </h3>
          <ul className="space-y-2">
            {data.recommendations.map((rec, i) => (
              <li key={i} className="flex items-start gap-2 text-sm text-emerald-300/80">
                <CheckCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
                {rec}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* SNR by Machine */}
      {data.snr_by_machine && data.snr_by_machine.length > 0 && (
        <div className="rounded-xl border border-slate-700/50 bg-slate-800/30 p-4">
          <button
            onClick={() => setShowMachines(!showMachines)}
            className="w-full flex items-center justify-between mb-4"
          >
            <h3 className="font-medium text-white flex items-center gap-2">
              <Cpu className="w-4 h-4 text-cyan-400" />
              SNR por Máquina ({data.snr_by_machine.length})
            </h3>
            {showMachines ? <ChevronDown className="w-5 h-5 text-slate-400" /> : <ChevronRight className="w-5 h-5 text-slate-400" />}
          </button>
          
          {showMachines && (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {data.snr_by_machine.map((machine) => (
                <MachineCard key={machine.machine_id} machine={machine} />
              ))}
            </div>
          )}
        </div>
      )}

      {/* SNR by Operation */}
      {data.snr_by_operation && data.snr_by_operation.length > 0 && (
        <div className="rounded-xl border border-slate-700/50 bg-slate-800/30 p-4">
          <button
            onClick={() => setShowOperations(!showOperations)}
            className="w-full flex items-center justify-between mb-4"
          >
            <h3 className="font-medium text-white flex items-center gap-2">
              <BarChart3 className="w-4 h-4 text-purple-400" />
              SNR por Tipo de Operação ({data.snr_by_operation.length})
            </h3>
            {showOperations ? <ChevronDown className="w-5 h-5 text-slate-400" /> : <ChevronRight className="w-5 h-5 text-slate-400" />}
          </button>
          
          {showOperations && (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-slate-700 text-slate-400">
                    <th className="text-left py-2 px-2">Operação</th>
                    <th className="text-left py-2 px-2">SNR</th>
                    <th className="text-center py-2 px-2">Nível</th>
                    <th className="text-right py-2 px-2">Amostras</th>
                  </tr>
                </thead>
                <tbody>
                  {data.snr_by_operation.map((op) => (
                    <OperationRow key={op.operation_type} operation={op} />
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default DataQualityPanel;


