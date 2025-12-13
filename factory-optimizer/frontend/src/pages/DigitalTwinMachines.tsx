/**
 * ═══════════════════════════════════════════════════════════════════════════════
 * SHI-DT (Smart Health Index Digital Twin) - Machine Health Dashboard
 * ═══════════════════════════════════════════════════════════════════════════════
 * 
 * Dashboard for real-time machine health monitoring and RUL estimation.
 * 
 * Features:
 * - Health Index overview for all machines
 * - RUL (Remaining Useful Life) estimation
 * - Sensor contribution analysis
 * - Active alerts management
 * - Operational profile detection
 * 
 * R&D / SIFIDE: WP1 - Digital Twin for predictive maintenance
 */

import React, { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  Activity,
  AlertTriangle,
  CheckCircle,
  Clock,
  Cpu,
  Gauge,
  Heart,
  RefreshCw,
  Settings,
  ThermometerSun,
  TrendingDown,
  TrendingUp,
  Vibrate,
  Zap,
  Info,
  ChevronRight,
  Play,
} from 'lucide-react';

// ═══════════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════════

interface RULEstimate {
  timestamp: string;
  rul_hours: number;
  rul_lower: number;
  rul_upper: number;
  confidence: number;
  degradation_rate_per_hour: number;
  method: string;
  data_points: number;
}

interface SensorContribution {
  sensor: string;
  contribution_pct: number;
  current: number;
  baseline: number;
  deviation: number;
  trend: 'increasing' | 'stable' | 'decreasing';
}

interface HealthAlert {
  alert_id: string;
  machine_id: string;
  timestamp: string;
  severity: 'info' | 'warning' | 'critical' | 'emergency';
  title: string;
  message: string;
  hi_current: number;
  rul?: RULEstimate;
  contributing_sensors: SensorContribution[];
  recommended_actions: string[];
  acknowledged: boolean;
}

interface MachineStatus {
  machine_id: string;
  timestamp: string;
  health_index: number;
  health_index_std: number;
  status: 'HEALTHY' | 'WARNING' | 'CRITICAL';
  rul: RULEstimate | null;
  profile: string;
  degradation: {
    trend: 'stable' | 'degrading' | 'improving';
    rate_per_hour: number;
  };
  top_contributors: SensorContribution[];
  active_alerts: HealthAlert[];
  last_sensor_update: string;
  model_version: string;
}

interface MachineListItem {
  machine_id: string;
  health_index: number;
  status: 'HEALTHY' | 'WARNING' | 'CRITICAL';
  profile: string;
  rul_hours: number | null;
  alerts_count: number;
}

interface MetricsSummary {
  healthy_count: number;
  warning_count: number;
  critical_count: number;
  average_health_index: number;
  total_alerts: number;
}

// ═══════════════════════════════════════════════════════════════════════════════
// API HOOKS
// ═══════════════════════════════════════════════════════════════════════════════

const API_BASE = 'http://127.0.0.1:8000/shi-dt';

const fetchMachines = async (): Promise<{ count: number; machines: MachineListItem[] }> => {
  const res = await fetch(`${API_BASE}/machines`);
  if (!res.ok) throw new Error('Failed to fetch machines');
  return res.json();
};

const fetchMachineStatus = async (machineId: string): Promise<MachineStatus> => {
  const res = await fetch(`${API_BASE}/machines/${machineId}/status`);
  if (!res.ok) throw new Error('Failed to fetch machine status');
  return res.json();
};

const fetchMetrics = async (): Promise<{ summary: MetricsSummary; machines: Record<string, any> }> => {
  const res = await fetch(`${API_BASE}/metrics`);
  if (!res.ok) throw new Error('Failed to fetch metrics');
  return res.json();
};

const generateDemoData = async (numMachines: number, readingsPerMachine: number) => {
  const res = await fetch(
    `${API_BASE}/demo/generate-data?num_machines=${numMachines}&readings_per_machine=${readingsPerMachine}`,
    { method: 'POST' }
  );
  if (!res.ok) throw new Error('Failed to generate demo data');
  return res.json();
};

// ═══════════════════════════════════════════════════════════════════════════════
// COMPONENTS
// ═══════════════════════════════════════════════════════════════════════════════

const StatusBadge: React.FC<{ status: string }> = ({ status }) => {
  const colors = {
    HEALTHY: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30',
    WARNING: 'bg-amber-500/20 text-amber-400 border-amber-500/30',
    CRITICAL: 'bg-red-500/20 text-red-400 border-red-500/30',
  };

  const icons = {
    HEALTHY: <CheckCircle className="w-3.5 h-3.5" />,
    WARNING: <AlertTriangle className="w-3.5 h-3.5" />,
    CRITICAL: <Zap className="w-3.5 h-3.5" />,
  };

  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2.5 py-1 text-xs font-medium rounded-full border ${
        colors[status as keyof typeof colors] || colors.HEALTHY
      }`}
    >
      {icons[status as keyof typeof icons]}
      {status}
    </span>
  );
};

const HealthGauge: React.FC<{ value: number; size?: 'sm' | 'md' | 'lg' }> = ({
  value,
  size = 'md',
}) => {
  const sizes = {
    sm: { width: 80, stroke: 6, fontSize: 'text-lg' },
    md: { width: 120, stroke: 8, fontSize: 'text-2xl' },
    lg: { width: 160, stroke: 10, fontSize: 'text-3xl' },
  };

  const { width, stroke, fontSize } = sizes[size];
  const radius = (width - stroke) / 2;
  const circumference = radius * Math.PI * 2;
  const progress = Math.max(0, Math.min(100, value));
  const offset = circumference - (progress / 100) * circumference;

  const getColor = (val: number) => {
    if (val >= 80) return '#10b981'; // emerald
    if (val >= 50) return '#f59e0b'; // amber
    return '#ef4444'; // red
  };

  return (
    <div className="relative" style={{ width, height: width }}>
      <svg width={width} height={width} className="transform -rotate-90">
        {/* Background circle */}
        <circle
          cx={width / 2}
          cy={width / 2}
          r={radius}
          strokeWidth={stroke}
          fill="none"
          className="stroke-zinc-800"
        />
        {/* Progress circle */}
        <motion.circle
          cx={width / 2}
          cy={width / 2}
          r={radius}
          strokeWidth={stroke}
          fill="none"
          stroke={getColor(progress)}
          strokeLinecap="round"
          strokeDasharray={circumference}
          initial={{ strokeDashoffset: circumference }}
          animate={{ strokeDashoffset: offset }}
          transition={{ duration: 1, ease: 'easeOut' }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className={`font-bold ${fontSize}`} style={{ color: getColor(progress) }}>
          {progress.toFixed(0)}%
        </span>
        <span className="text-xs text-zinc-500">Health</span>
      </div>
    </div>
  );
};

const TrendIndicator: React.FC<{ trend: string; rate: number }> = ({ trend, rate }) => {
  const config = {
    degrading: { icon: TrendingDown, color: 'text-red-400', label: 'Degrading' },
    stable: { icon: Activity, color: 'text-zinc-400', label: 'Stable' },
    improving: { icon: TrendingUp, color: 'text-emerald-400', label: 'Improving' },
  };

  const { icon: Icon, color, label } = config[trend as keyof typeof config] || config.stable;

  return (
    <div className={`flex items-center gap-1.5 ${color}`}>
      <Icon className="w-4 h-4" />
      <span className="text-sm">{label}</span>
      {rate !== 0 && (
        <span className="text-xs text-zinc-500">({rate > 0 ? '+' : ''}{rate.toFixed(3)}/h)</span>
      )}
    </div>
  );
};

const SensorCard: React.FC<{ sensor: SensorContribution }> = ({ sensor }) => {
  const sensorIcons: Record<string, React.ReactNode> = {
    vibration: <Vibrate className="w-4 h-4" />,
    temp: <ThermometerSun className="w-4 h-4" />,
    current: <Zap className="w-4 h-4" />,
    acoustic: <Activity className="w-4 h-4" />,
    pressure: <Gauge className="w-4 h-4" />,
  };

  const getIcon = () => {
    for (const [key, icon] of Object.entries(sensorIcons)) {
      if (sensor.sensor.toLowerCase().includes(key)) return icon;
    }
    return <Cpu className="w-4 h-4" />;
  };

  const getTrendIcon = () => {
    if (sensor.trend === 'increasing') return <TrendingUp className="w-3 h-3 text-red-400" />;
    if (sensor.trend === 'decreasing') return <TrendingDown className="w-3 h-3 text-emerald-400" />;
    return null;
  };

  return (
    <div className="bg-zinc-800/50 rounded-lg p-3 border border-zinc-700/50">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2 text-zinc-300">
          {getIcon()}
          <span className="text-sm font-medium capitalize">
            {sensor.sensor.replace(/_/g, ' ')}
          </span>
        </div>
        <div className="flex items-center gap-1">
          {getTrendIcon()}
          <span className="text-xs text-zinc-500">{sensor.contribution_pct.toFixed(1)}%</span>
        </div>
      </div>
      <div className="flex justify-between text-xs">
        <span className="text-zinc-500">Current: {sensor.current.toFixed(2)}</span>
        <span className="text-zinc-500">Baseline: {sensor.baseline.toFixed(2)}</span>
      </div>
      {/* Progress bar for contribution */}
      <div className="mt-2 h-1.5 bg-zinc-700 rounded-full overflow-hidden">
        <motion.div
          className="h-full bg-gradient-to-r from-amber-500 to-red-500"
          initial={{ width: 0 }}
          animate={{ width: `${sensor.contribution_pct}%` }}
          transition={{ duration: 0.5 }}
        />
      </div>
    </div>
  );
};

const MachineCard: React.FC<{
  machine: MachineListItem;
  isSelected: boolean;
  onClick: () => void;
}> = ({ machine, isSelected, onClick }) => {
  const statusColors = {
    HEALTHY: 'border-emerald-500/30 hover:border-emerald-500/50',
    WARNING: 'border-amber-500/30 hover:border-amber-500/50',
    CRITICAL: 'border-red-500/30 hover:border-red-500/50 animate-pulse',
  };

  return (
    <motion.div
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
      onClick={onClick}
      className={`cursor-pointer bg-zinc-900/70 rounded-xl p-4 border transition-all ${
        statusColors[machine.status]
      } ${isSelected ? 'ring-2 ring-cyan-500/50' : ''}`}
    >
      <div className="flex items-center justify-between mb-3">
        <span className="font-mono text-sm text-zinc-300">{machine.machine_id}</span>
        <StatusBadge status={machine.status} />
      </div>
      <div className="flex items-center justify-between">
        <HealthGauge value={machine.health_index} size="sm" />
        <div className="text-right">
          <div className="text-xs text-zinc-500 mb-1">RUL</div>
          <div className="font-medium text-zinc-300">
            {machine.rul_hours !== null ? `${machine.rul_hours.toFixed(0)}h` : '—'}
          </div>
          {machine.alerts_count > 0 && (
            <div className="flex items-center gap-1 text-amber-400 text-xs mt-1">
              <AlertTriangle className="w-3 h-3" />
              {machine.alerts_count}
            </div>
          )}
        </div>
      </div>
    </motion.div>
  );
};

const MachineDetail: React.FC<{ machineId: string }> = ({ machineId }) => {
  const { data: status, isLoading, error } = useQuery({
    queryKey: ['machine-status', machineId],
    queryFn: () => fetchMachineStatus(machineId),
    refetchInterval: 5000,
  });

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <RefreshCw className="w-8 h-8 text-cyan-500 animate-spin" />
      </div>
    );
  }

  if (error || !status) {
    return (
      <div className="flex items-center justify-center h-full text-red-400">
        <AlertTriangle className="w-6 h-6 mr-2" />
        Failed to load machine data
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h2 className="text-2xl font-bold text-white mb-1">{status.machine_id}</h2>
          <div className="flex items-center gap-3">
            <StatusBadge status={status.status} />
            <span className="text-sm text-zinc-400">Profile: {status.profile}</span>
          </div>
        </div>
        <HealthGauge value={status.health_index} size="lg" />
      </div>

      {/* Health & RUL Stats */}
      <div className="grid grid-cols-2 gap-4">
        <div className="bg-zinc-800/50 rounded-xl p-4 border border-zinc-700/50">
          <div className="flex items-center gap-2 text-zinc-400 mb-2">
            <Heart className="w-4 h-4" />
            <span className="text-sm">Health Index</span>
          </div>
          <div className="text-3xl font-bold text-white mb-1">
            {status.health_index.toFixed(1)}%
            <span className="text-lg text-zinc-500 ml-1">±{status.health_index_std.toFixed(1)}</span>
          </div>
          <TrendIndicator
            trend={status.degradation.trend}
            rate={status.degradation.rate_per_hour}
          />
        </div>

        <div className="bg-zinc-800/50 rounded-xl p-4 border border-zinc-700/50">
          <div className="flex items-center gap-2 text-zinc-400 mb-2">
            <Clock className="w-4 h-4" />
            <span className="text-sm">Remaining Useful Life</span>
          </div>
          {status.rul ? (
            <>
              <div className="text-3xl font-bold text-white mb-1">
                {status.rul.rul_hours.toFixed(0)}h
              </div>
              <div className="text-sm text-zinc-500">
                95% CI: {status.rul.rul_lower.toFixed(0)}-{status.rul.rul_upper.toFixed(0)}h
                <span className="ml-2">({(status.rul.confidence * 100).toFixed(0)}% conf.)</span>
              </div>
            </>
          ) : (
            <div className="text-zinc-500">Insufficient data</div>
          )}
        </div>
      </div>

      {/* Top Contributors */}
      <div>
        <h3 className="text-lg font-semibold text-white mb-3 flex items-center gap-2">
          <Activity className="w-5 h-5 text-cyan-500" />
          Top Degradation Contributors
        </h3>
        <div className="grid grid-cols-2 gap-3">
          {status.top_contributors.map((sensor) => (
            <SensorCard key={sensor.sensor} sensor={sensor} />
          ))}
        </div>
      </div>

      {/* Active Alerts */}
      {status.active_alerts.length > 0 && (
        <div>
          <h3 className="text-lg font-semibold text-white mb-3 flex items-center gap-2">
            <AlertTriangle className="w-5 h-5 text-amber-500" />
            Active Alerts ({status.active_alerts.length})
          </h3>
          <div className="space-y-2">
            {status.active_alerts.map((alert) => (
              <div
                key={alert.alert_id}
                className={`rounded-lg p-3 border ${
                  alert.severity === 'critical'
                    ? 'bg-red-500/10 border-red-500/30'
                    : 'bg-amber-500/10 border-amber-500/30'
                }`}
              >
                <div className="flex items-start justify-between">
                  <div>
                    <div className="font-medium text-white">{alert.title}</div>
                    <div className="text-sm text-zinc-400">{alert.message}</div>
                  </div>
                  <span
                    className={`text-xs px-2 py-0.5 rounded ${
                      alert.severity === 'critical' ? 'bg-red-500/20 text-red-400' : 'bg-amber-500/20 text-amber-400'
                    }`}
                  >
                    {alert.severity.toUpperCase()}
                  </span>
                </div>
                {alert.recommended_actions.length > 0 && (
                  <div className="mt-2 pt-2 border-t border-zinc-700/50">
                    <div className="text-xs text-zinc-500 mb-1">Recommended Actions:</div>
                    <ul className="text-sm text-zinc-300 list-disc list-inside">
                      {alert.recommended_actions.map((action, i) => (
                        <li key={i}>{action}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Model Info */}
      <div className="text-xs text-zinc-600 flex items-center justify-between pt-4 border-t border-zinc-800">
        <span>Model: SHI-DT v{status.model_version}</span>
        <span>Last update: {new Date(status.last_sensor_update).toLocaleTimeString()}</span>
      </div>
    </div>
  );
};

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN PAGE
// ═══════════════════════════════════════════════════════════════════════════════

const DigitalTwinMachines: React.FC = () => {
  const [selectedMachine, setSelectedMachine] = useState<string | null>(null);
  const queryClient = useQueryClient();

  const { data: machinesData, isLoading: machinesLoading } = useQuery({
    queryKey: ['shi-dt-machines'],
    queryFn: fetchMachines,
    refetchInterval: 10000,
  });

  const { data: metricsData } = useQuery({
    queryKey: ['shi-dt-metrics'],
    queryFn: fetchMetrics,
    refetchInterval: 10000,
  });

  const demoMutation = useMutation({
    mutationFn: () => generateDemoData(5, 20),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['shi-dt-machines'] });
      queryClient.invalidateQueries({ queryKey: ['shi-dt-metrics'] });
    },
  });

  const machines = machinesData?.machines || [];
  const summary = metricsData?.summary;

  // Auto-select first machine
  useEffect(() => {
    if (machines.length > 0 && !selectedMachine) {
      setSelectedMachine(machines[0].machine_id);
    }
  }, [machines, selectedMachine]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-zinc-950 via-zinc-900 to-zinc-950 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-3xl font-bold text-white mb-2 flex items-center gap-3">
              <div className="p-2 bg-gradient-to-br from-cyan-500/20 to-blue-500/20 rounded-xl">
                <Heart className="w-8 h-8 text-cyan-500" />
              </div>
              SHI-DT: Machine Health Monitor
            </h1>
            <p className="text-zinc-400">
              Real-time health monitoring with CVAE-based anomaly detection and RUL prediction
            </p>
          </div>
          <button
            onClick={() => demoMutation.mutate()}
            disabled={demoMutation.isPending}
            className="flex items-center gap-2 px-4 py-2 bg-cyan-500/20 hover:bg-cyan-500/30 text-cyan-400 rounded-lg transition-colors border border-cyan-500/30"
          >
            {demoMutation.isPending ? (
              <RefreshCw className="w-4 h-4 animate-spin" />
            ) : (
              <Play className="w-4 h-4" />
            )}
            Generate Demo Data
          </button>
        </div>

        {/* Summary Cards */}
        {summary && (
          <div className="grid grid-cols-4 gap-4 mb-8">
            <div className="bg-zinc-900/70 rounded-xl p-4 border border-zinc-800">
              <div className="flex items-center gap-2 text-zinc-400 text-sm mb-1">
                <Gauge className="w-4 h-4" />
                Avg. Health
              </div>
              <div className="text-2xl font-bold text-white">
                {summary.average_health_index.toFixed(1)}%
              </div>
            </div>
            <div className="bg-emerald-500/10 rounded-xl p-4 border border-emerald-500/20">
              <div className="flex items-center gap-2 text-emerald-400 text-sm mb-1">
                <CheckCircle className="w-4 h-4" />
                Healthy
              </div>
              <div className="text-2xl font-bold text-emerald-400">{summary.healthy_count}</div>
            </div>
            <div className="bg-amber-500/10 rounded-xl p-4 border border-amber-500/20">
              <div className="flex items-center gap-2 text-amber-400 text-sm mb-1">
                <AlertTriangle className="w-4 h-4" />
                Warning
              </div>
              <div className="text-2xl font-bold text-amber-400">{summary.warning_count}</div>
            </div>
            <div className="bg-red-500/10 rounded-xl p-4 border border-red-500/20">
              <div className="flex items-center gap-2 text-red-400 text-sm mb-1">
                <Zap className="w-4 h-4" />
                Critical
              </div>
              <div className="text-2xl font-bold text-red-400">{summary.critical_count}</div>
            </div>
          </div>
        )}

        {/* Main Content */}
        <div className="grid grid-cols-12 gap-6">
          {/* Machine List */}
          <div className="col-span-4">
            <div className="bg-zinc-900/50 rounded-2xl p-4 border border-zinc-800">
              <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                <Cpu className="w-5 h-5 text-cyan-500" />
                Machines ({machines.length})
              </h3>
              {machinesLoading ? (
                <div className="flex items-center justify-center py-8">
                  <RefreshCw className="w-6 h-6 text-cyan-500 animate-spin" />
                </div>
              ) : machines.length === 0 ? (
                <div className="text-center py-8 text-zinc-500">
                  <Info className="w-8 h-8 mx-auto mb-2 opacity-50" />
                  <p>No machines monitored yet</p>
                  <p className="text-sm mt-1">Click "Generate Demo Data" to start</p>
                </div>
              ) : (
                <div className="space-y-3 max-h-[600px] overflow-y-auto pr-2">
                  <AnimatePresence>
                    {machines.map((machine) => (
                      <MachineCard
                        key={machine.machine_id}
                        machine={machine}
                        isSelected={selectedMachine === machine.machine_id}
                        onClick={() => setSelectedMachine(machine.machine_id)}
                      />
                    ))}
                  </AnimatePresence>
                </div>
              )}
            </div>
          </div>

          {/* Machine Detail */}
          <div className="col-span-8">
            <div className="bg-zinc-900/50 rounded-2xl p-6 border border-zinc-800 min-h-[600px]">
              {selectedMachine ? (
                <MachineDetail machineId={selectedMachine} />
              ) : (
                <div className="flex items-center justify-center h-full text-zinc-500">
                  <div className="text-center">
                    <ChevronRight className="w-12 h-12 mx-auto mb-2 opacity-30" />
                    <p>Select a machine to view details</p>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Footer Info */}
        <div className="mt-8 text-center text-xs text-zinc-600">
          <p>
            SHI-DT uses Conditional Variational Autoencoders (CVAE) for health indicator extraction
            and exponential degradation models for RUL estimation.
          </p>
          <p className="mt-1">
            R&D / SIFIDE: WP1 - Digital Twin for Predictive Maintenance
          </p>
        </div>
      </div>
    </div>
  );
};

export default DigitalTwinMachines;



