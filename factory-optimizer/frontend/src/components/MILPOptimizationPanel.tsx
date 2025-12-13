/**
 * MILP Optimization Panel - Advanced Scheduling Configuration
 * 
 * Allows users to configure and run MILP optimization with:
 * - Time limits
 * - Gap tolerance
 * - Objective weights
 * - Comparison with heuristics
 */

import React, { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { motion } from 'framer-motion';
import {
  AlertTriangle,
  BarChart3,
  CheckCircle,
  Clock,
  Cpu,
  Loader2,
  Play,
  Settings,
  Sliders,
  Target,
  TrendingDown,
  Zap,
} from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

// ═══════════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════════

interface ScheduleResult {
  schedule_id: string;
  solver_used: string;
  total_tardiness: number;
  total_makespan_minutes: number;
  machine_utilization: Record<string, number>;
  job_count: number;
  solve_time_seconds: number;
}

interface ComparisonResult {
  milp?: { tardiness: number; makespan: number; error?: string };
  heuristic?: { tardiness: number; makespan: number; error?: string };
  improvement_percent?: number;
}

interface OptimizationConfig {
  time_limit: number;
  gap_tolerance: number;
  objective: 'tardiness' | 'makespan' | 'balanced';
  weights: {
    tardiness: number;
    makespan: number;
    utilization: number;
  };
}

// ═══════════════════════════════════════════════════════════════════════════════
// API
// ═══════════════════════════════════════════════════════════════════════════════

async function runOptimization(config: OptimizationConfig): Promise<{ schedule: ScheduleResult; comparison: ComparisonResult }> {
  const res = await fetch(`${API_BASE}/optimization/schedule/demo`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      num_jobs: 15,
      num_machines: 4,
      solver: 'cpsat',
      time_limit_seconds: config.time_limit,
    }),
  });
  if (!res.ok) throw new Error('Failed to run optimization');
  return res.json();
}

// ═══════════════════════════════════════════════════════════════════════════════
// COMPONENTS
// ═══════════════════════════════════════════════════════════════════════════════

const SliderInput: React.FC<{
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  unit?: string;
  onChange: (value: number) => void;
}> = ({ label, value, min, max, step = 1, unit = '', onChange }) => (
  <div className="space-y-2">
    <div className="flex justify-between text-sm">
      <span className="text-slate-400">{label}</span>
      <span className="text-white font-medium">{value}{unit}</span>
    </div>
    <input
      type="range"
      min={min}
      max={max}
      step={step}
      value={value}
      onChange={(e) => onChange(Number(e.target.value))}
      className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-cyan-500"
    />
    <div className="flex justify-between text-xs text-slate-500">
      <span>{min}{unit}</span>
      <span>{max}{unit}</span>
    </div>
  </div>
);

const ObjectiveSelector: React.FC<{
  value: string;
  onChange: (value: 'tardiness' | 'makespan' | 'balanced') => void;
}> = ({ value, onChange }) => {
  const options = [
    { id: 'tardiness', label: 'Minimizar Atrasos', icon: <Clock className="w-4 h-4" />, color: 'cyan' },
    { id: 'makespan', label: 'Minimizar Makespan', icon: <TrendingDown className="w-4 h-4" />, color: 'purple' },
    { id: 'balanced', label: 'Balanceado', icon: <Target className="w-4 h-4" />, color: 'emerald' },
  ];

  return (
    <div className="space-y-2">
      <span className="text-sm text-slate-400">Objetivo Principal</span>
      <div className="grid grid-cols-3 gap-2">
        {options.map((opt) => (
          <button
            key={opt.id}
            onClick={() => onChange(opt.id as 'tardiness' | 'makespan' | 'balanced')}
            className={`flex items-center gap-2 p-3 rounded-lg border transition-all ${
              value === opt.id
                ? `bg-${opt.color}-500/20 border-${opt.color}-500/50 text-${opt.color}-400`
                : 'bg-slate-800/50 border-slate-700 text-slate-400 hover:border-slate-600'
            }`}
          >
            {opt.icon}
            <span className="text-sm font-medium">{opt.label}</span>
          </button>
        ))}
      </div>
    </div>
  );
};

const MetricCard: React.FC<{
  label: string;
  value: string | number;
  unit?: string;
  icon: React.ReactNode;
  trend?: 'up' | 'down' | 'neutral';
  trendValue?: string;
}> = ({ label, value, unit, icon, trend, trendValue }) => (
  <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700/50">
    <div className="flex items-center gap-2 text-slate-400 text-sm mb-2">
      {icon}
      {label}
    </div>
    <div className="flex items-end gap-2">
      <p className="text-2xl font-bold text-white">
        {value}
        {unit && <span className="text-sm text-slate-400 ml-1">{unit}</span>}
      </p>
      {trendValue && (
        <span className={`text-sm ${trend === 'down' ? 'text-emerald-400' : trend === 'up' ? 'text-red-400' : 'text-slate-400'}`}>
          {trendValue}
        </span>
      )}
    </div>
  </div>
);

const ComparisonChart: React.FC<{ comparison: ComparisonResult }> = ({ comparison }) => {
  const maxTardiness = Math.max(
    comparison.milp?.tardiness || 0,
    comparison.heuristic?.tardiness || 0,
    1
  );

  return (
    <div className="space-y-4">
      <h4 className="text-sm font-medium text-slate-400">Comparação MILP vs Heurística</h4>
      
      <div className="space-y-3">
        {/* MILP */}
        <div className="flex items-center gap-3">
          <span className="w-20 text-sm text-slate-400">MILP</span>
          <div className="flex-1 h-8 bg-slate-700 rounded overflow-hidden">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${((comparison.milp?.tardiness || 0) / maxTardiness) * 100}%` }}
              className="h-full bg-gradient-to-r from-cyan-500 to-blue-500 flex items-center justify-end px-2"
            >
              <span className="text-xs text-white font-medium">
                {comparison.milp?.tardiness?.toFixed(0) || 0} min
              </span>
            </motion.div>
          </div>
        </div>
        
        {/* Heuristic */}
        <div className="flex items-center gap-3">
          <span className="w-20 text-sm text-slate-400">Heurística</span>
          <div className="flex-1 h-8 bg-slate-700 rounded overflow-hidden">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${((comparison.heuristic?.tardiness || 0) / maxTardiness) * 100}%` }}
              className="h-full bg-gradient-to-r from-purple-500 to-pink-500 flex items-center justify-end px-2"
            >
              <span className="text-xs text-white font-medium">
                {comparison.heuristic?.tardiness?.toFixed(0) || 0} min
              </span>
            </motion.div>
          </div>
        </div>
      </div>

      {comparison.improvement_percent !== undefined && (
        <div className={`text-center p-3 rounded-lg ${
          comparison.improvement_percent > 0 
            ? 'bg-emerald-500/10 border border-emerald-500/30' 
            : 'bg-slate-800/50 border border-slate-700'
        }`}>
          <span className={comparison.improvement_percent > 0 ? 'text-emerald-400' : 'text-slate-400'}>
            {comparison.improvement_percent > 0 ? '↓' : '→'} {Math.abs(comparison.improvement_percent).toFixed(1)}% 
            {comparison.improvement_percent > 0 ? ' melhoria com MILP' : ' diferença'}
          </span>
        </div>
      )}
    </div>
  );
};

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN COMPONENT
// ═══════════════════════════════════════════════════════════════════════════════

export const MILPOptimizationPanel: React.FC = () => {
  const [config, setConfig] = useState<OptimizationConfig>({
    time_limit: 30,
    gap_tolerance: 0.05,
    objective: 'balanced',
    weights: {
      tardiness: 0.5,
      makespan: 0.3,
      utilization: 0.2,
    },
  });

  const optimizationMutation = useMutation({
    mutationFn: () => runOptimization(config),
  });

  const result = optimizationMutation.data;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold text-white flex items-center gap-2">
            <Cpu className="w-5 h-5 text-cyan-500" />
            Otimização MILP (CP-SAT)
          </h2>
          <p className="text-sm text-slate-400 mt-1">
            Configuração avançada para otimização de scheduling
          </p>
        </div>
      </div>

      <div className="grid grid-cols-12 gap-6">
        {/* Configuration Panel */}
        <div className="col-span-5 space-y-6">
          <div className="rounded-xl border border-slate-700/50 bg-slate-800/30 p-6 space-y-6">
            <h3 className="font-medium text-white flex items-center gap-2">
              <Settings className="w-4 h-4 text-slate-400" />
              Configuração do Solver
            </h3>

            <SliderInput
              label="Limite de Tempo"
              value={config.time_limit}
              min={5}
              max={120}
              step={5}
              unit="s"
              onChange={(v) => setConfig({ ...config, time_limit: v })}
            />

            <SliderInput
              label="Gap Tolerance"
              value={config.gap_tolerance * 100}
              min={1}
              max={20}
              step={1}
              unit="%"
              onChange={(v) => setConfig({ ...config, gap_tolerance: v / 100 })}
            />

            <ObjectiveSelector
              value={config.objective}
              onChange={(v) => setConfig({ ...config, objective: v })}
            />

            <div className="space-y-3">
              <span className="text-sm text-slate-400">Pesos dos Objetivos</span>
              
              <SliderInput
                label="Tardiness"
                value={config.weights.tardiness * 100}
                min={0}
                max={100}
                step={5}
                unit="%"
                onChange={(v) => setConfig({
                  ...config,
                  weights: { ...config.weights, tardiness: v / 100 }
                })}
              />
              
              <SliderInput
                label="Makespan"
                value={config.weights.makespan * 100}
                min={0}
                max={100}
                step={5}
                unit="%"
                onChange={(v) => setConfig({
                  ...config,
                  weights: { ...config.weights, makespan: v / 100 }
                })}
              />
              
              <SliderInput
                label="Utilização"
                value={config.weights.utilization * 100}
                min={0}
                max={100}
                step={5}
                unit="%"
                onChange={(v) => setConfig({
                  ...config,
                  weights: { ...config.weights, utilization: v / 100 }
                })}
              />
            </div>

            <button
              onClick={() => optimizationMutation.mutate()}
              disabled={optimizationMutation.isPending}
              className="w-full flex items-center justify-center gap-2 py-3 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white font-semibold rounded-lg transition-all disabled:opacity-50"
            >
              {optimizationMutation.isPending ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  Otimizando...
                </>
              ) : (
                <>
                  <Play className="w-5 h-5" />
                  Executar Otimização
                </>
              )}
            </button>
          </div>

          {/* Info Box */}
          <div className="rounded-xl border border-cyan-500/20 bg-cyan-500/5 p-4">
            <div className="flex items-start gap-3">
              <Zap className="w-5 h-5 text-cyan-400 mt-0.5" />
              <div>
                <p className="font-medium text-cyan-400">Sobre CP-SAT</p>
                <p className="text-sm text-slate-400 mt-1">
                  O Google OR-Tools CP-SAT é um solver de constraint programming
                  que encontra soluções ótimas ou near-optimal para problemas de scheduling.
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Results Panel */}
        <div className="col-span-7 space-y-6">
          {result ? (
            <>
              {/* Metrics */}
              <div className="grid grid-cols-2 gap-4">
                <MetricCard
                  label="Tardiness Total"
                  value={result.schedule.total_tardiness.toFixed(0)}
                  unit="min"
                  icon={<Clock className="w-4 h-4" />}
                />
                <MetricCard
                  label="Makespan"
                  value={result.schedule.total_makespan_minutes.toFixed(0)}
                  unit="min"
                  icon={<BarChart3 className="w-4 h-4" />}
                />
                <MetricCard
                  label="Jobs Agendados"
                  value={result.schedule.job_count}
                  icon={<Target className="w-4 h-4" />}
                />
                <MetricCard
                  label="Tempo de Solução"
                  value={result.schedule.solve_time_seconds.toFixed(2)}
                  unit="s"
                  icon={<Zap className="w-4 h-4" />}
                />
              </div>

              {/* Comparison */}
              {result.comparison && (
                <div className="rounded-xl border border-slate-700/50 bg-slate-800/30 p-6">
                  <ComparisonChart comparison={result.comparison} />
                </div>
              )}

              {/* Machine Utilization */}
              <div className="rounded-xl border border-slate-700/50 bg-slate-800/30 p-6">
                <h4 className="text-sm font-medium text-slate-400 mb-4">Utilização por Máquina</h4>
                <div className="space-y-3">
                  {Object.entries(result.schedule.machine_utilization).map(([machine, util]) => (
                    <div key={machine} className="flex items-center gap-3">
                      <span className="w-16 text-sm text-slate-400">{machine}</span>
                      <div className="flex-1 h-4 bg-slate-700 rounded overflow-hidden">
                        <motion.div
                          initial={{ width: 0 }}
                          animate={{ width: `${util * 100}%` }}
                          className={`h-full ${
                            util > 0.8 ? 'bg-emerald-500' : util > 0.5 ? 'bg-cyan-500' : 'bg-amber-500'
                          }`}
                        />
                      </div>
                      <span className="w-12 text-sm text-slate-400 text-right">
                        {(util * 100).toFixed(0)}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Success Badge */}
              <div className="flex items-center justify-center gap-2 p-4 rounded-xl bg-emerald-500/10 border border-emerald-500/30">
                <CheckCircle className="w-5 h-5 text-emerald-400" />
                <span className="text-emerald-400 font-medium">
                  Otimização concluída com solver {result.schedule.solver_used.toUpperCase()}
                </span>
              </div>
            </>
          ) : (
            <div className="h-full flex items-center justify-center rounded-xl border border-slate-700/50 bg-slate-800/30 p-12">
              <div className="text-center">
                <Sliders className="w-16 h-16 text-slate-600 mx-auto mb-4" />
                <p className="text-lg font-medium text-slate-400">Configure e execute a otimização</p>
                <p className="text-sm text-slate-500 mt-1">
                  Ajuste os parâmetros e clique em "Executar Otimização"
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default MILPOptimizationPanel;


