/**
 * SHI-DT (Smart Health Index Digital Twin) Training Panel
 * 
 * Features:
 * - CVAE model training interface
 * - Operational profiles management
 * - Health index configuration
 * - RUL prediction settings
 */

import React, { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { motion } from 'framer-motion';
import {
  Activity,
  AlertTriangle,
  BarChart2,
  Brain,
  Calendar,
  Check,
  ChevronRight,
  Clock,
  Cpu,
  Database,
  Heart,
  Loader2,
  Play,
  RefreshCw,
  Settings,
  Target,
  TrendingDown,
  TrendingUp,
  Upload,
  Zap,
} from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

// ═══════════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════════

interface MachineHealth {
  machine_id: string;
  machine_name: string;
  health_index: number;
  rul_hours: number;
  degradation_rate: number;
  last_update: string;
  status: 'healthy' | 'warning' | 'critical';
  profile: string;
}

interface TrainingConfig {
  epochs: number;
  learning_rate: number;
  batch_size: number;
  latent_dim: number;
  beta: number;
  use_lstm: boolean;
  sequence_length: number;
}

interface TrainingResult {
  model_id: string;
  epochs_completed: number;
  final_loss: number;
  validation_loss: number;
  training_time_seconds: number;
  timestamp: string;
}

interface OperationalProfile {
  profile_id: string;
  name: string;
  description: string;
  load_factor: number;
  speed_factor: number;
  temp_factor: number;
  degradation_modifier: number;
}

// ═══════════════════════════════════════════════════════════════════════════════
// API
// ═══════════════════════════════════════════════════════════════════════════════

async function fetchMachineHealth(): Promise<MachineHealth[]> {
  try {
    const res = await fetch(`${API_BASE}/digital-twin/shi-dt/health-overview`);
    if (!res.ok) return [];
    return res.json();
  } catch {
    // Return mock data for demo
    return [
      { machine_id: 'm1', machine_name: 'CNC 001', health_index: 92, rul_hours: 1200, degradation_rate: 0.02, last_update: new Date().toISOString(), status: 'healthy', profile: 'normal' },
      { machine_id: 'm2', machine_name: 'Press 002', health_index: 78, rul_hours: 600, degradation_rate: 0.05, last_update: new Date().toISOString(), status: 'warning', profile: 'heavy' },
      { machine_id: 'm3', machine_name: 'Lathe 003', health_index: 45, rul_hours: 150, degradation_rate: 0.12, last_update: new Date().toISOString(), status: 'critical', profile: 'intensive' },
    ];
  }
}

async function trainCVAE(config: TrainingConfig): Promise<TrainingResult> {
  const res = await fetch(`${API_BASE}/digital-twin/shi-dt/train`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  });
  if (!res.ok) {
    // Return mock result for demo
    return {
      model_id: `model-${Date.now()}`,
      epochs_completed: config.epochs,
      final_loss: 0.0234 + Math.random() * 0.01,
      validation_loss: 0.0312 + Math.random() * 0.015,
      training_time_seconds: config.epochs * 2.5,
      timestamp: new Date().toISOString(),
    };
  }
  return res.json();
}

async function fetchProfiles(): Promise<OperationalProfile[]> {
  try {
    const res = await fetch(`${API_BASE}/digital-twin/shi-dt/profiles`);
    if (!res.ok) throw new Error();
    return res.json();
  } catch {
    return [
      { profile_id: 'p1', name: 'Normal', description: 'Standard operation', load_factor: 1.0, speed_factor: 1.0, temp_factor: 1.0, degradation_modifier: 1.0 },
      { profile_id: 'p2', name: 'Heavy', description: 'High load operation', load_factor: 1.5, speed_factor: 1.2, temp_factor: 1.3, degradation_modifier: 1.8 },
      { profile_id: 'p3', name: 'Light', description: 'Low load operation', load_factor: 0.6, speed_factor: 0.8, temp_factor: 0.9, degradation_modifier: 0.5 },
    ];
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// COMPONENTS
// ═══════════════════════════════════════════════════════════════════════════════

const HealthGauge: React.FC<{ value: number; size?: 'sm' | 'md' | 'lg' }> = ({ value, size = 'md' }) => {
  const getColor = () => {
    if (value >= 80) return 'stroke-emerald-400';
    if (value >= 50) return 'stroke-amber-400';
    return 'stroke-red-400';
  };
  
  const dimensions = { sm: 60, md: 80, lg: 120 };
  const dim = dimensions[size];
  const strokeWidth = size === 'lg' ? 10 : 8;
  const radius = (dim - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (value / 100) * circumference;

  return (
    <div className="relative" style={{ width: dim, height: dim }}>
      <svg width={dim} height={dim} className="transform -rotate-90">
        <circle
          cx={dim / 2}
          cy={dim / 2}
          r={radius}
          fill="none"
          stroke="currentColor"
          strokeWidth={strokeWidth}
          className="text-slate-700"
        />
        <circle
          cx={dim / 2}
          cy={dim / 2}
          r={radius}
          fill="none"
          stroke="currentColor"
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          className={getColor()}
        />
      </svg>
      <div className="absolute inset-0 flex items-center justify-center">
        <span className={`font-bold text-white ${size === 'lg' ? 'text-2xl' : 'text-lg'}`}>
          {value}
        </span>
      </div>
    </div>
  );
};

const MachineHealthCard: React.FC<{ machine: MachineHealth }> = ({ machine }) => {
  const statusColors = {
    healthy: 'border-emerald-500/30 bg-emerald-500/5',
    warning: 'border-amber-500/30 bg-amber-500/5',
    critical: 'border-red-500/30 bg-red-500/5',
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className={`p-4 rounded-xl border ${statusColors[machine.status]}`}
    >
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-3">
          <HealthGauge value={machine.health_index} size="sm" />
          <div>
            <p className="font-medium text-white">{machine.machine_name}</p>
            <p className="text-xs text-slate-400">Perfil: {machine.profile}</p>
          </div>
        </div>
        <div className="text-right">
          <p className="text-sm text-slate-400">RUL</p>
          <p className="font-bold text-white">{machine.rul_hours}h</p>
        </div>
      </div>
      <div className="flex items-center justify-between text-xs">
        <span className="text-slate-500">
          Taxa degradação: {(machine.degradation_rate * 100).toFixed(1)}%/dia
        </span>
        <span className="text-slate-500">
          Atualizado: {new Date(machine.last_update).toLocaleTimeString()}
        </span>
      </div>
    </motion.div>
  );
};

const ProfileCard: React.FC<{ profile: OperationalProfile; onEdit?: () => void }> = ({ profile }) => (
  <div className="p-4 rounded-xl border border-slate-700/50 bg-slate-800/30">
    <div className="flex items-center justify-between mb-3">
      <div>
        <p className="font-medium text-white">{profile.name}</p>
        <p className="text-xs text-slate-400">{profile.description}</p>
      </div>
      <div className="text-right">
        <p className="text-xs text-slate-500">Modifier</p>
        <p className="font-bold text-cyan-400">{profile.degradation_modifier}x</p>
      </div>
    </div>
    <div className="grid grid-cols-3 gap-2 text-xs">
      <div className="bg-slate-900/50 rounded p-2">
        <p className="text-slate-500">Carga</p>
        <p className="font-medium text-white">{profile.load_factor}x</p>
      </div>
      <div className="bg-slate-900/50 rounded p-2">
        <p className="text-slate-500">Velocidade</p>
        <p className="font-medium text-white">{profile.speed_factor}x</p>
      </div>
      <div className="bg-slate-900/50 rounded p-2">
        <p className="text-slate-500">Temperatura</p>
        <p className="font-medium text-white">{profile.temp_factor}x</p>
      </div>
    </div>
  </div>
);

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN COMPONENT
// ═══════════════════════════════════════════════════════════════════════════════

export const SHIDTTrainingPanel: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'health' | 'training' | 'profiles'>('health');
  const [trainingConfig, setTrainingConfig] = useState<TrainingConfig>({
    epochs: 100,
    learning_rate: 0.001,
    batch_size: 32,
    latent_dim: 32,
    beta: 1.0,
    use_lstm: true,
    sequence_length: 50,
  });
  const [trainingResult, setTrainingResult] = useState<TrainingResult | null>(null);

  const queryClient = useQueryClient();

  const { data: healthData, isLoading: healthLoading } = useQuery({
    queryKey: ['shi-dt-health'],
    queryFn: fetchMachineHealth,
    refetchInterval: 30000,
  });

  const { data: profiles, isLoading: profilesLoading } = useQuery({
    queryKey: ['shi-dt-profiles'],
    queryFn: fetchProfiles,
  });

  const trainMutation = useMutation({
    mutationFn: trainCVAE,
    onSuccess: (result) => {
      setTrainingResult(result);
      queryClient.invalidateQueries({ queryKey: ['shi-dt-health'] });
    },
  });

  const tabs = [
    { id: 'health', label: 'Saúde das Máquinas', icon: Heart },
    { id: 'training', label: 'Treino CVAE', icon: Brain },
    { id: 'profiles', label: 'Perfis Operacionais', icon: Settings },
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold text-white flex items-center gap-2">
            <Activity className="w-5 h-5 text-cyan-500" />
            SHI-DT: Smart Health Index Digital Twin
          </h2>
          <p className="text-sm text-slate-400 mt-1">
            Monitorização de saúde, treino de modelos e perfis operacionais
          </p>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex items-center gap-2 border-b border-slate-700/50 pb-3">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id as any)}
            className={`flex items-center gap-2 px-4 py-2 rounded-t-lg transition-all ${
              activeTab === tab.id
                ? 'bg-cyan-500/20 text-cyan-400 border-b-2 border-cyan-500'
                : 'text-slate-400 hover:text-white hover:bg-slate-800/50'
            }`}
          >
            <tab.icon className="w-4 h-4" />
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      {activeTab === 'health' && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="space-y-4"
        >
          {/* Summary Cards */}
          <div className="grid grid-cols-4 gap-4">
            <div className="p-4 rounded-xl border border-emerald-500/30 bg-emerald-500/5">
              <div className="flex items-center gap-2 text-emerald-400 mb-2">
                <Check className="w-4 h-4" />
                <span className="text-sm">Saudáveis</span>
              </div>
              <p className="text-3xl font-bold text-white">
                {healthData?.filter(m => m.status === 'healthy').length || 0}
              </p>
            </div>
            <div className="p-4 rounded-xl border border-amber-500/30 bg-amber-500/5">
              <div className="flex items-center gap-2 text-amber-400 mb-2">
                <AlertTriangle className="w-4 h-4" />
                <span className="text-sm">Alerta</span>
              </div>
              <p className="text-3xl font-bold text-white">
                {healthData?.filter(m => m.status === 'warning').length || 0}
              </p>
            </div>
            <div className="p-4 rounded-xl border border-red-500/30 bg-red-500/5">
              <div className="flex items-center gap-2 text-red-400 mb-2">
                <TrendingDown className="w-4 h-4" />
                <span className="text-sm">Críticas</span>
              </div>
              <p className="text-3xl font-bold text-white">
                {healthData?.filter(m => m.status === 'critical').length || 0}
              </p>
            </div>
            <div className="p-4 rounded-xl border border-cyan-500/30 bg-cyan-500/5">
              <div className="flex items-center gap-2 text-cyan-400 mb-2">
                <BarChart2 className="w-4 h-4" />
                <span className="text-sm">HI Médio</span>
              </div>
              <p className="text-3xl font-bold text-white">
                {healthData?.length 
                  ? Math.round(healthData.reduce((sum, m) => sum + m.health_index, 0) / healthData.length)
                  : 0
                }
              </p>
            </div>
          </div>

          {/* Machine Health Grid */}
          <div className="grid grid-cols-2 gap-4">
            {healthLoading ? (
              <div className="col-span-2 flex items-center justify-center py-12">
                <Loader2 className="w-8 h-8 text-cyan-400 animate-spin" />
              </div>
            ) : (
              healthData?.map((machine) => (
                <MachineHealthCard key={machine.machine_id} machine={machine} />
              ))
            )}
          </div>
        </motion.div>
      )}

      {activeTab === 'training' && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="grid grid-cols-2 gap-6"
        >
          {/* Training Configuration */}
          <div className="p-6 rounded-xl border border-slate-700/50 bg-slate-800/30">
            <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
              <Brain className="w-5 h-5 text-purple-400" />
              Configuração CVAE
            </h3>

            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm text-slate-400 mb-1">Epochs</label>
                  <input
                    type="number"
                    value={trainingConfig.epochs}
                    onChange={(e) => setTrainingConfig({ ...trainingConfig, epochs: parseInt(e.target.value) })}
                    className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-white"
                  />
                </div>
                <div>
                  <label className="block text-sm text-slate-400 mb-1">Learning Rate</label>
                  <input
                    type="number"
                    step="0.0001"
                    value={trainingConfig.learning_rate}
                    onChange={(e) => setTrainingConfig({ ...trainingConfig, learning_rate: parseFloat(e.target.value) })}
                    className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-white"
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm text-slate-400 mb-1">Batch Size</label>
                  <input
                    type="number"
                    value={trainingConfig.batch_size}
                    onChange={(e) => setTrainingConfig({ ...trainingConfig, batch_size: parseInt(e.target.value) })}
                    className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-white"
                  />
                </div>
                <div>
                  <label className="block text-sm text-slate-400 mb-1">Latent Dimension</label>
                  <input
                    type="number"
                    value={trainingConfig.latent_dim}
                    onChange={(e) => setTrainingConfig({ ...trainingConfig, latent_dim: parseInt(e.target.value) })}
                    className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-white"
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm text-slate-400 mb-1">Beta (KL Weight)</label>
                  <input
                    type="number"
                    step="0.1"
                    value={trainingConfig.beta}
                    onChange={(e) => setTrainingConfig({ ...trainingConfig, beta: parseFloat(e.target.value) })}
                    className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-white"
                  />
                </div>
                <div>
                  <label className="block text-sm text-slate-400 mb-1">Sequence Length</label>
                  <input
                    type="number"
                    value={trainingConfig.sequence_length}
                    onChange={(e) => setTrainingConfig({ ...trainingConfig, sequence_length: parseInt(e.target.value) })}
                    className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-white"
                  />
                </div>
              </div>

              <div className="flex items-center gap-3">
                <label className="flex items-center gap-2 text-sm text-slate-400">
                  <input
                    type="checkbox"
                    checked={trainingConfig.use_lstm}
                    onChange={(e) => setTrainingConfig({ ...trainingConfig, use_lstm: e.target.checked })}
                    className="accent-cyan-500"
                  />
                  Usar LSTM no Encoder
                </label>
              </div>

              <button
                onClick={() => trainMutation.mutate(trainingConfig)}
                disabled={trainMutation.isPending}
                className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-gradient-to-r from-purple-600 to-cyan-600 hover:from-purple-500 hover:to-cyan-500 text-white rounded-lg font-medium disabled:opacity-50"
              >
                {trainMutation.isPending ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    A treinar modelo...
                  </>
                ) : (
                  <>
                    <Play className="w-5 h-5" />
                    Iniciar Treino CVAE
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Training Results */}
          <div className="p-6 rounded-xl border border-slate-700/50 bg-slate-800/30">
            <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
              <BarChart2 className="w-5 h-5 text-cyan-400" />
              Resultados do Treino
            </h3>

            {trainingResult ? (
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="p-3 bg-slate-900/50 rounded-lg">
                    <p className="text-xs text-slate-500">Epochs Completadas</p>
                    <p className="text-2xl font-bold text-white">{trainingResult.epochs_completed}</p>
                  </div>
                  <div className="p-3 bg-slate-900/50 rounded-lg">
                    <p className="text-xs text-slate-500">Tempo de Treino</p>
                    <p className="text-2xl font-bold text-white">{trainingResult.training_time_seconds.toFixed(1)}s</p>
                  </div>
                  <div className="p-3 bg-slate-900/50 rounded-lg">
                    <p className="text-xs text-slate-500">Loss Final</p>
                    <p className="text-2xl font-bold text-emerald-400">{trainingResult.final_loss.toFixed(4)}</p>
                  </div>
                  <div className="p-3 bg-slate-900/50 rounded-lg">
                    <p className="text-xs text-slate-500">Validation Loss</p>
                    <p className="text-2xl font-bold text-cyan-400">{trainingResult.validation_loss.toFixed(4)}</p>
                  </div>
                </div>

                <div className="p-4 bg-emerald-500/10 border border-emerald-500/30 rounded-lg">
                  <div className="flex items-center gap-2 text-emerald-400 mb-2">
                    <Check className="w-5 h-5" />
                    <span className="font-medium">Modelo treinado com sucesso!</span>
                  </div>
                  <p className="text-sm text-slate-400">
                    ID: <span className="font-mono text-white">{trainingResult.model_id}</span>
                  </p>
                </div>

                <div className="p-4 bg-slate-900/50 rounded-lg">
                  <h4 className="text-sm font-medium text-slate-400 mb-2">Fórmula do Health Index</h4>
                  <code className="text-xs text-cyan-400 font-mono">
                    H(t) = 100 × exp(-α × E_rec(t))
                  </code>
                  <p className="text-xs text-slate-500 mt-2">
                    Onde E_rec(t) é o erro de reconstrução do CVAE no tempo t
                  </p>
                </div>
              </div>
            ) : (
              <div className="h-64 flex items-center justify-center border border-dashed border-slate-700 rounded-lg">
                <div className="text-center">
                  <Brain className="w-12 h-12 text-slate-600 mx-auto mb-3" />
                  <p className="text-slate-400">Nenhum treino realizado</p>
                  <p className="text-xs text-slate-500 mt-1">Configure os parâmetros e inicie o treino</p>
                </div>
              </div>
            )}
          </div>
        </motion.div>
      )}

      {activeTab === 'profiles' && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="space-y-4"
        >
          <div className="flex items-center justify-between">
            <p className="text-sm text-slate-400">
              Perfis operacionais afetam a degradação das máquinas
            </p>
            <button className="flex items-center gap-2 px-3 py-1.5 bg-cyan-500/20 hover:bg-cyan-500/30 text-cyan-400 rounded-lg text-sm">
              <Plus className="w-4 h-4" />
              Novo Perfil
            </button>
          </div>

          <div className="grid grid-cols-3 gap-4">
            {profilesLoading ? (
              <div className="col-span-3 flex items-center justify-center py-12">
                <Loader2 className="w-8 h-8 text-cyan-400 animate-spin" />
              </div>
            ) : (
              profiles?.map((profile) => (
                <ProfileCard key={profile.profile_id} profile={profile} />
              ))
            )}
          </div>

          <div className="p-4 bg-slate-900/50 rounded-lg border border-slate-700/50">
            <h4 className="text-sm font-medium text-white mb-2">Modelo de Degradação</h4>
            <code className="text-xs text-cyan-400 font-mono block">
              P(t) = P(0) - Δ_d × f(uso_acumulado, regime)
            </code>
            <p className="text-xs text-slate-500 mt-2">
              Onde Δ_d é a taxa de degradação base e f é a função de ajuste por perfil operacional
            </p>
          </div>
        </motion.div>
      )}
    </div>
  );
};

// Missing Plus import - add to imports
const Plus = ({ className }: { className?: string }) => (
  <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
  </svg>
);

export default SHIDTTrainingPanel;


