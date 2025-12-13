/**
 * ═══════════════════════════════════════════════════════════════════════════════
 * Optimization Dashboard - Mathematical & ML Optimization
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * Dashboard for optimization operations:
 * - Time prediction
 * - Golden runs benchmarking
 * - Parameter optimization (Bayesian/GA)
 * - Advanced scheduling (CP-SAT)
 * - Pareto multi-objective analysis
 *
 * R&D / SIFIDE: WP4 - Learning Scheduler & Advanced Optimization
 */

import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  Activity,
  Award,
  BarChart3,
  Calendar,
  ChevronRight,
  Clock,
  Cpu,
  Factory,
  Gauge,
  Layers,
  Loader2,
  Play,
  RefreshCw,
  Settings,
  Sparkles,
  Target,
  TrendingUp,
  Zap,
} from 'lucide-react';
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  Legend,
} from 'recharts';

// ═══════════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════════

interface OptStatus {
  engines: {
    time_prediction: string;
    golden_runs: number;
    scheduler: string;
  };
  objectives: string[];
}

interface ScheduleResult {
  schedule_id: string;
  solver_used: string;
  total_tardiness: number;
  total_makespan_minutes: number;
  machine_utilization: Record<string, number>;
  scheduled_jobs: Array<{
    job_id: string;
    machine_id: string;
    start_time: string;
    end_time: string;
    tardiness_minutes: number;
  }>;
  job_count: number;
  solve_time_seconds: number;
}

interface ParameterResult {
  optimal_parameters: Record<string, number>;
  improvement_percent: number;
  iterations_used: number;
  predicted_time: number;
  predicted_defect_rate: number;
}

interface GoldenRun {
  run_id: string;
  product_id: string;
  operation_id: string;
  machine_id: string;
  cycle_time_minutes: number;
  defect_rate: number;
  oee: number;
  parameters: Record<string, number>;
  recorded_at: string;
}

interface ParetoSolution {
  solution_id: string;
  parameters: Record<string, number>;
  objectives: { time: number; defect_rate: number };
}

// ═══════════════════════════════════════════════════════════════════════════════
// API
// ═══════════════════════════════════════════════════════════════════════════════

const API_BASE = 'http://127.0.0.1:8000/optimization';

const fetchStatus = async (): Promise<OptStatus> => {
  const res = await fetch(`${API_BASE}/status`);
  if (!res.ok) throw new Error('Failed');
  return res.json();
};

const runScheduleDemo = async (): Promise<{ schedule: ScheduleResult; comparison: Record<string, any> }> => {
  const res = await fetch(`${API_BASE}/schedule/demo`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ num_jobs: 10, num_machines: 3 }),
  });
  if (!res.ok) throw new Error('Failed');
  return res.json();
};

const runParamDemo = async (): Promise<{ result: ParameterResult }> => {
  const res = await fetch(`${API_BASE}/parameters/demo`, { method: 'POST' });
  if (!res.ok) throw new Error('Failed');
  return res.json();
};

const runParetoDemo = async (): Promise<{ solutions: ParetoSolution[] }> => {
  const res = await fetch(`${API_BASE}/pareto/demo`, { method: 'POST' });
  if (!res.ok) throw new Error('Failed');
  return res.json();
};

const listGoldenRuns = async (): Promise<{ golden_runs: GoldenRun[] }> => {
  const res = await fetch(`${API_BASE}/golden-runs`);
  if (!res.ok) throw new Error('Failed');
  return res.json();
};

// ═══════════════════════════════════════════════════════════════════════════════
// COMPONENTS
// ═══════════════════════════════════════════════════════════════════════════════

const StatCard: React.FC<{ label: string; value: string | number; icon: React.ReactNode; color: string; subtext?: string }> = ({
  label, value, icon, color, subtext,
}) => (
  <div className="bg-zinc-800/50 rounded-xl p-4 border border-zinc-700/50">
    <div className="flex items-center gap-3">
      <div className={`p-2 rounded-lg ${color}`}>{icon}</div>
      <div className="flex-1">
        <p className="text-2xl font-bold text-white">{value}</p>
        <p className="text-sm text-zinc-400">{label}</p>
        {subtext && <p className="text-xs text-zinc-500">{subtext}</p>}
      </div>
    </div>
  </div>
);

const ScheduleView: React.FC<{ schedule: ScheduleResult }> = ({ schedule }) => {
  // Prepare Gantt data
  const machines = [...new Set(schedule.scheduled_jobs.map(j => j.machine_id))];
  const baseTime = new Date(schedule.scheduled_jobs[0]?.start_time || new Date());
  
  const ganttData = machines.map(machine => {
    const jobs = schedule.scheduled_jobs.filter(j => j.machine_id === machine);
    return {
      machine,
      jobs: jobs.map(j => ({
        job_id: j.job_id,
        start: (new Date(j.start_time).getTime() - baseTime.getTime()) / 60000,
        duration: (new Date(j.end_time).getTime() - new Date(j.start_time).getTime()) / 60000,
        tardiness: j.tardiness_minutes,
      })),
    };
  });

  return (
    <div className="space-y-4">
      {/* Metrics */}
      <div className="grid grid-cols-4 gap-4">
        <div className="bg-zinc-900/50 p-3 rounded-lg">
          <p className="text-xs text-zinc-500">Makespan</p>
          <p className="text-lg font-bold text-white">{schedule.total_makespan_minutes.toFixed(0)} min</p>
        </div>
        <div className="bg-zinc-900/50 p-3 rounded-lg">
          <p className="text-xs text-zinc-500">Tardiness Total</p>
          <p className={`text-lg font-bold ${schedule.total_tardiness > 0 ? 'text-red-400' : 'text-emerald-400'}`}>
            {schedule.total_tardiness.toFixed(0)} min
          </p>
        </div>
        <div className="bg-zinc-900/50 p-3 rounded-lg">
          <p className="text-xs text-zinc-500">Solver</p>
          <p className="text-lg font-bold text-blue-400">{schedule.solver_used.toUpperCase()}</p>
        </div>
        <div className="bg-zinc-900/50 p-3 rounded-lg">
          <p className="text-xs text-zinc-500">Solve Time</p>
          <p className="text-lg font-bold text-white">{schedule.solve_time_seconds.toFixed(2)}s</p>
        </div>
      </div>

      {/* Simple Gantt */}
      <div className="bg-zinc-900/50 rounded-lg p-4">
        <h4 className="text-sm font-medium text-zinc-400 mb-3">Gantt Chart</h4>
        {ganttData.map(({ machine, jobs }) => (
          <div key={machine} className="flex items-center gap-2 mb-2">
            <span className="w-16 text-xs text-zinc-500 truncate">{machine}</span>
            <div className="flex-1 h-6 bg-zinc-800 rounded relative overflow-hidden">
              {jobs.map((job, idx) => {
                const width = (job.duration / schedule.total_makespan_minutes) * 100;
                const left = (job.start / schedule.total_makespan_minutes) * 100;
                return (
                  <div
                    key={idx}
                    className={`absolute h-full rounded ${job.tardiness > 0 ? 'bg-red-500/70' : 'bg-blue-500/70'}`}
                    style={{ left: `${left}%`, width: `${Math.max(width, 2)}%` }}
                    title={`${job.job_id}: ${job.duration.toFixed(0)}min`}
                  />
                );
              })}
            </div>
          </div>
        ))}
      </div>

      {/* Utilization */}
      <div className="bg-zinc-900/50 rounded-lg p-4">
        <h4 className="text-sm font-medium text-zinc-400 mb-3">Machine Utilization</h4>
        <div className="h-32">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={Object.entries(schedule.machine_utilization).map(([m, u]) => ({ machine: m, utilization: u * 100 }))}>
              <CartesianGrid strokeDasharray="3 3" stroke="#333" />
              <XAxis dataKey="machine" tick={{ fill: '#888', fontSize: 10 }} />
              <YAxis domain={[0, 100]} tick={{ fill: '#888', fontSize: 10 }} />
              <Tooltip contentStyle={{ backgroundColor: '#1f1f1f', border: '1px solid #333' }} />
              <Bar dataKey="utilization" fill="#3b82f6" name="Utilização %" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};

const ParetoChart: React.FC<{ solutions: ParetoSolution[] }> = ({ solutions }) => {
  const data = solutions.map(s => ({
    time: s.objectives.time,
    defect: s.objectives.defect_rate * 100,
    id: s.solution_id,
  }));

  return (
    <div className="h-64">
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#333" />
          <XAxis
            dataKey="time"
            name="Time"
            tick={{ fill: '#888', fontSize: 10 }}
            label={{ value: 'Time (min)', position: 'bottom', fill: '#888', fontSize: 10 }}
          />
          <YAxis
            dataKey="defect"
            name="Defect %"
            tick={{ fill: '#888', fontSize: 10 }}
            label={{ value: 'Defect %', angle: -90, position: 'left', fill: '#888', fontSize: 10 }}
          />
          <Tooltip
            contentStyle={{ backgroundColor: '#1f1f1f', border: '1px solid #333' }}
            formatter={(value: number, name: string) => [value.toFixed(3), name]}
          />
          <Scatter data={data} fill="#8b5cf6" />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
};

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN PAGE
// ═══════════════════════════════════════════════════════════════════════════════

const OptimizationDashboard: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'scheduling' | 'parameters' | 'pareto' | 'golden'>('scheduling');
  
  const queryClient = useQueryClient();

  const { data: status } = useQuery({ queryKey: ['opt-status'], queryFn: fetchStatus });
  const { data: goldenRuns } = useQuery({ queryKey: ['golden-runs'], queryFn: listGoldenRuns });

  const scheduleMutation = useMutation({
    mutationFn: runScheduleDemo,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['opt-status'] }),
  });

  const paramMutation = useMutation({ mutationFn: runParamDemo });
  const paretoMutation = useMutation({ mutationFn: runParetoDemo });

  const tabs = [
    { id: 'scheduling' as const, label: 'Scheduling', icon: <Calendar className="w-4 h-4" /> },
    { id: 'parameters' as const, label: 'Parameters', icon: <Settings className="w-4 h-4" /> },
    { id: 'pareto' as const, label: 'Pareto', icon: <Target className="w-4 h-4" /> },
    { id: 'golden' as const, label: 'Golden Runs', icon: <Award className="w-4 h-4" /> },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-zinc-950 via-zinc-900 to-zinc-950 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-3xl font-bold text-white mb-2 flex items-center gap-3">
              <div className="p-2 bg-gradient-to-br from-violet-500/20 to-purple-500/20 rounded-xl">
                <Cpu className="w-8 h-8 text-violet-500" />
              </div>
              Otimização Matemática
            </h1>
            <p className="text-zinc-400">Scheduling avançado, otimização de parâmetros e análise Pareto</p>
          </div>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-4 gap-4 mb-6">
          <StatCard
            label="Time Predictor"
            value={status?.engines.time_prediction === 'ml_pytorch' ? 'ML' : 'Base'}
            icon={<Clock className="w-5 h-5" />}
            color="bg-blue-500/20 text-blue-400"
            subtext={status?.engines.time_prediction}
          />
          <StatCard
            label="Golden Runs"
            value={status?.engines.golden_runs || 0}
            icon={<Award className="w-5 h-5" />}
            color="bg-amber-500/20 text-amber-400"
          />
          <StatCard
            label="Scheduler"
            value="CP-SAT"
            icon={<Factory className="w-5 h-5" />}
            color="bg-emerald-500/20 text-emerald-400"
            subtext="OR-Tools"
          />
          <StatCard
            label="Multi-Objective"
            value="NSGA-II"
            icon={<Target className="w-5 h-5" />}
            color="bg-purple-500/20 text-purple-400"
            subtext="Pareto Frontier"
          />
        </div>

        {/* Tabs */}
        <div className="flex gap-2 mb-6">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-all ${
                activeTab === tab.id
                  ? 'bg-violet-500/20 text-violet-400 border border-violet-500/30'
                  : 'bg-zinc-800/50 text-zinc-400 hover:bg-zinc-700/50'
              }`}
            >
              {tab.icon}
              {tab.label}
            </button>
          ))}
        </div>

        {/* Content */}
        <div className="bg-zinc-900/50 rounded-xl border border-zinc-800 p-6">
          {/* SCHEDULING TAB */}
          {activeTab === 'scheduling' && (
            <div className="space-y-6">
              <div className="flex items-center justify-between">
                <h2 className="text-xl font-bold text-white flex items-center gap-2">
                  <Calendar className="w-5 h-5 text-blue-500" />
                  Advanced Scheduling (CP-SAT)
                </h2>
                <button
                  onClick={() => scheduleMutation.mutate()}
                  disabled={scheduleMutation.isPending}
                  className="flex items-center gap-2 px-4 py-2 bg-blue-500/20 hover:bg-blue-500/30 text-blue-400 rounded-lg"
                >
                  {scheduleMutation.isPending ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
                  Run Demo Schedule
                </button>
              </div>

              {scheduleMutation.data ? (
                <>
                  {/* Comparison */}
                  {scheduleMutation.data.comparison && (
                    <div className="bg-zinc-800/50 rounded-lg p-4">
                      <h4 className="text-sm font-medium text-zinc-400 mb-3">Method Comparison</h4>
                      <div className="grid grid-cols-2 gap-4">
                        {Object.entries(scheduleMutation.data.comparison).map(([method, metrics]: [string, any]) => (
                          <div key={method} className="bg-zinc-900/50 p-3 rounded-lg">
                            <p className="font-medium text-white">{method.toUpperCase()}</p>
                            {metrics.error ? (
                              <p className="text-xs text-red-400">{metrics.error}</p>
                            ) : (
                              <div className="text-xs text-zinc-400 mt-1">
                                <p>Tardiness: {metrics.tardiness.toFixed(0)} min</p>
                                <p>Makespan: {metrics.makespan.toFixed(0)} min</p>
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                  <ScheduleView schedule={scheduleMutation.data.schedule} />
                </>
              ) : (
                <div className="text-center py-12 text-zinc-500">
                  <Calendar className="w-12 h-12 mx-auto mb-3 opacity-30" />
                  <p>Execute o demo para ver o scheduling otimizado</p>
                </div>
              )}
            </div>
          )}

          {/* PARAMETERS TAB */}
          {activeTab === 'parameters' && (
            <div className="space-y-6">
              <div className="flex items-center justify-between">
                <h2 className="text-xl font-bold text-white flex items-center gap-2">
                  <Settings className="w-5 h-5 text-emerald-500" />
                  Parameter Optimization (Bayesian/GA)
                </h2>
                <button
                  onClick={() => paramMutation.mutate()}
                  disabled={paramMutation.isPending}
                  className="flex items-center gap-2 px-4 py-2 bg-emerald-500/20 hover:bg-emerald-500/30 text-emerald-400 rounded-lg"
                >
                  {paramMutation.isPending ? <Loader2 className="w-4 h-4 animate-spin" /> : <Zap className="w-4 h-4" />}
                  Optimize Parameters
                </button>
              </div>

              {paramMutation.data ? (
                <div className="grid grid-cols-2 gap-6">
                  <div className="bg-zinc-800/50 rounded-lg p-4">
                    <h4 className="text-sm font-medium text-zinc-400 mb-3">Optimal Parameters</h4>
                    <div className="space-y-3">
                      {Object.entries(paramMutation.data.result.optimal_parameters).map(([name, value]) => (
                        <div key={name} className="flex items-center justify-between">
                          <span className="text-zinc-300">{name}</span>
                          <span className="font-mono text-emerald-400">{value.toFixed(3)}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                  <div className="bg-zinc-800/50 rounded-lg p-4">
                    <h4 className="text-sm font-medium text-zinc-400 mb-3">Results</h4>
                    <div className="space-y-3">
                      <div className="flex items-center justify-between">
                        <span className="text-zinc-400">Improvement</span>
                        <span className={`font-bold ${paramMutation.data.result.improvement_percent > 0 ? 'text-emerald-400' : 'text-zinc-400'}`}>
                          {paramMutation.data.result.improvement_percent.toFixed(1)}%
                        </span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-zinc-400">Iterations</span>
                        <span className="text-white">{paramMutation.data.result.iterations_used}</span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-zinc-400">Predicted Time</span>
                        <span className="text-white">{paramMutation.data.result.predicted_time.toFixed(2)}</span>
                      </div>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="text-center py-12 text-zinc-500">
                  <Settings className="w-12 h-12 mx-auto mb-3 opacity-30" />
                  <p>Execute a otimização para encontrar parâmetros ótimos</p>
                </div>
              )}
            </div>
          )}

          {/* PARETO TAB */}
          {activeTab === 'pareto' && (
            <div className="space-y-6">
              <div className="flex items-center justify-between">
                <h2 className="text-xl font-bold text-white flex items-center gap-2">
                  <Target className="w-5 h-5 text-purple-500" />
                  Multi-Objective Pareto (NSGA-II)
                </h2>
                <button
                  onClick={() => paretoMutation.mutate()}
                  disabled={paretoMutation.isPending}
                  className="flex items-center gap-2 px-4 py-2 bg-purple-500/20 hover:bg-purple-500/30 text-purple-400 rounded-lg"
                >
                  {paretoMutation.isPending ? <Loader2 className="w-4 h-4 animate-spin" /> : <Sparkles className="w-4 h-4" />}
                  Generate Pareto Frontier
                </button>
              </div>

              {paretoMutation.data ? (
                <div className="space-y-4">
                  <div className="bg-zinc-800/50 rounded-lg p-4">
                    <h4 className="text-sm font-medium text-zinc-400 mb-3">
                      Pareto Frontier ({paretoMutation.data.solutions.length} solutions)
                    </h4>
                    <ParetoChart solutions={paretoMutation.data.solutions} />
                    <p className="text-xs text-zinc-500 mt-2 text-center">
                      Trade-off: Time vs Defect Rate • Cada ponto é uma solução não-dominada
                    </p>
                  </div>
                  <div className="bg-zinc-800/50 rounded-lg p-4">
                    <h4 className="text-sm font-medium text-zinc-400 mb-3">Sample Solutions</h4>
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm">
                        <thead>
                          <tr className="text-zinc-500 border-b border-zinc-700">
                            <th className="text-left py-2">Solution</th>
                            <th className="text-right py-2">Time</th>
                            <th className="text-right py-2">Defect %</th>
                            <th className="text-right py-2">Speed</th>
                            <th className="text-right py-2">Temp</th>
                          </tr>
                        </thead>
                        <tbody>
                          {paretoMutation.data.solutions.slice(0, 8).map((sol) => (
                            <tr key={sol.solution_id} className="border-b border-zinc-800">
                              <td className="py-2 font-mono text-xs text-zinc-500">{sol.solution_id}</td>
                              <td className="py-2 text-right text-white">{sol.objectives.time.toFixed(1)}</td>
                              <td className="py-2 text-right text-white">{(sol.objectives.defect_rate * 100).toFixed(2)}%</td>
                              <td className="py-2 text-right text-zinc-400">{sol.parameters.speed?.toFixed(2) || '-'}</td>
                              <td className="py-2 text-right text-zinc-400">{sol.parameters.temperature?.toFixed(0) || '-'}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="text-center py-12 text-zinc-500">
                  <Target className="w-12 h-12 mx-auto mb-3 opacity-30" />
                  <p>Gere a fronteira Pareto para análise multi-objetivo</p>
                </div>
              )}
            </div>
          )}

          {/* GOLDEN RUNS TAB */}
          {activeTab === 'golden' && (
            <div className="space-y-6">
              <div className="flex items-center justify-between">
                <h2 className="text-xl font-bold text-white flex items-center gap-2">
                  <Award className="w-5 h-5 text-amber-500" />
                  Golden Runs (Best Performance Records)
                </h2>
              </div>

              {goldenRuns && goldenRuns.golden_runs.length > 0 ? (
                <div className="space-y-3">
                  {goldenRuns.golden_runs.map((gr) => (
                    <div key={gr.run_id} className="bg-zinc-800/50 rounded-lg p-4 border border-amber-500/20">
                      <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center gap-3">
                          <Award className="w-5 h-5 text-amber-500" />
                          <div>
                            <p className="font-medium text-white">{gr.product_id} • {gr.operation_id}</p>
                            <p className="text-xs text-zinc-500">{gr.machine_id}</p>
                          </div>
                        </div>
                        <span className="text-xs text-zinc-500">
                          {new Date(gr.recorded_at).toLocaleDateString('pt-PT')}
                        </span>
                      </div>
                      <div className="grid grid-cols-4 gap-4">
                        <div>
                          <p className="text-xs text-zinc-500">Cycle Time</p>
                          <p className="font-bold text-emerald-400">{gr.cycle_time_minutes.toFixed(1)} min</p>
                        </div>
                        <div>
                          <p className="text-xs text-zinc-500">Defect Rate</p>
                          <p className="font-bold text-emerald-400">{(gr.defect_rate * 100).toFixed(2)}%</p>
                        </div>
                        <div>
                          <p className="text-xs text-zinc-500">OEE</p>
                          <p className="font-bold text-emerald-400">{(gr.oee * 100).toFixed(1)}%</p>
                        </div>
                        <div>
                          <p className="text-xs text-zinc-500">Parameters</p>
                          <p className="text-xs text-zinc-400">
                            {Object.entries(gr.parameters).map(([k, v]) => `${k}=${v}`).join(', ')}
                          </p>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-12 text-zinc-500">
                  <Award className="w-12 h-12 mx-auto mb-3 opacity-30" />
                  <p>Nenhum golden run registado</p>
                  <p className="text-sm mt-1">Golden runs são registados automaticamente quando há um novo recorde de performance</p>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="mt-8 text-center text-xs text-zinc-600">
          <p>Otimização matemática com CP-SAT, Bayesian Optimization, GA e NSGA-II</p>
          <p className="mt-1">R&D / SIFIDE: WP4 - Learning Scheduler & Advanced Optimization</p>
        </div>
      </div>
    </div>
  );
};

export default OptimizationDashboard;



