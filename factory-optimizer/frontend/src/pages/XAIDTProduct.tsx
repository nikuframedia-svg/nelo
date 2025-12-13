/**
 * ═══════════════════════════════════════════════════════════════════════════════
 * XAI-DT Product (Explainable Digital Twin do Produto) - Quality Analysis Dashboard
 * ═══════════════════════════════════════════════════════════════════════════════
 * 
 * Dashboard for geometric quality analysis comparing CAD vs Scanned parts.
 * 
 * Features:
 * - Deviation field visualization with heatmap
 * - Pattern identification (offset, scale, random, etc.)
 * - Root Cause Analysis with confidence scores
 * - Corrective action recommendations
 * - Analysis history
 * 
 * R&D / SIFIDE: WP1 - Digital Twin & Explainability
 */

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  Box,
  Check,
  ChevronRight,
  Crosshair,
  FileSearch,
  Gauge,
  Grid3X3,
  Layers,
  Lightbulb,
  Play,
  RefreshCw,
  Ruler,
  Search,
  Settings,
  Target,
  TrendingDown,
  TrendingUp,
  Wrench,
  XCircle,
  Zap,
} from 'lucide-react';

// ═══════════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════════

interface DeviationField {
  n_points: number;
  tolerance: number;
  mean_deviation: number;
  max_deviation: number;
  rms_deviation: number;
  pct_out_of_tolerance: number;
  deviation_score: number;
  alignment_rmse: number;
}

interface Pattern {
  pattern: string;
  confidence: number;
  parameters: Record<string, number>;
  affected_region: string;
  evidence: string[];
}

interface RootCause {
  category: string;
  description: string;
  confidence: number;
  evidence: string[];
  patterns_linked: string[];
}

interface CorrectiveAction {
  action: string;
  priority: 'high' | 'medium' | 'low';
  root_cause: string;
  expected_impact: string;
}

interface AnalysisResult {
  analysis_id: string;
  timestamp: string;
  cad_name: string;
  scan_name: string;
  deviation_field: DeviationField;
  pca_result: Record<string, any>;
  regional_analysis: Record<string, any>;
  identified_patterns: Pattern[];
  root_causes: RootCause[];
  corrective_actions: CorrectiveAction[];
  overall_quality: string;
  summary_text: string;
}

interface AnalysisListItem {
  analysis_id: string;
  timestamp: string;
  cad_name: string;
  scan_name: string;
  deviation_score: number;
  overall_quality: string;
}

// ═══════════════════════════════════════════════════════════════════════════════
// API HOOKS
// ═══════════════════════════════════════════════════════════════════════════════

const API_BASE = 'http://127.0.0.1:8000/xai-dt';

const fetchAnalyses = async (): Promise<{ total: number; analyses: AnalysisListItem[] }> => {
  const res = await fetch(`${API_BASE}/analyses`);
  if (!res.ok) throw new Error('Failed to fetch analyses');
  return res.json();
};

const fetchAnalysis = async (id: string): Promise<AnalysisResult> => {
  const res = await fetch(`${API_BASE}/analyses/${id}`);
  if (!res.ok) throw new Error('Failed to fetch analysis');
  return res.json();
};

const runDemoAnalysis = async (params: {
  n_points: number;
  deviation_type: string;
  deviation_magnitude: number;
  tolerance: number;
}): Promise<AnalysisResult> => {
  const res = await fetch(`${API_BASE}/demo`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!res.ok) throw new Error('Failed to run analysis');
  return res.json();
};

// ═══════════════════════════════════════════════════════════════════════════════
// COMPONENTS
// ═══════════════════════════════════════════════════════════════════════════════

const QualityBadge: React.FC<{ quality: string }> = ({ quality }) => {
  const colors: Record<string, string> = {
    excellent: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30',
    good: 'bg-green-500/20 text-green-400 border-green-500/30',
    acceptable: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30',
    poor: 'bg-orange-500/20 text-orange-400 border-orange-500/30',
    reject: 'bg-red-500/20 text-red-400 border-red-500/30',
  };

  const labels: Record<string, string> = {
    excellent: 'Excelente',
    good: 'Bom',
    acceptable: 'Aceitável',
    poor: 'Fraco',
    reject: 'Rejeitado',
  };

  return (
    <span
      className={`inline-flex items-center gap-1.5 px-3 py-1 text-sm font-medium rounded-full border ${
        colors[quality] || colors.reject
      }`}
    >
      {quality === 'excellent' || quality === 'good' ? (
        <Check className="w-4 h-4" />
      ) : quality === 'reject' ? (
        <XCircle className="w-4 h-4" />
      ) : (
        <AlertTriangle className="w-4 h-4" />
      )}
      {labels[quality] || quality}
    </span>
  );
};

const DeviationScoreGauge: React.FC<{ score: number }> = ({ score }) => {
  const radius = 60;
  const circumference = radius * Math.PI * 2;
  const progress = Math.max(0, Math.min(100, score));
  const offset = circumference - (progress / 100) * circumference;

  const getColor = (val: number) => {
    if (val >= 85) return '#10b981';
    if (val >= 70) return '#22c55e';
    if (val >= 50) return '#eab308';
    if (val >= 30) return '#f97316';
    return '#ef4444';
  };

  return (
    <div className="relative w-40 h-40">
      <svg width={160} height={160} className="transform -rotate-90">
        <circle
          cx={80}
          cy={80}
          r={radius}
          strokeWidth={12}
          fill="none"
          className="stroke-zinc-800"
        />
        <motion.circle
          cx={80}
          cy={80}
          r={radius}
          strokeWidth={12}
          fill="none"
          stroke={getColor(progress)}
          strokeLinecap="round"
          strokeDasharray={circumference}
          initial={{ strokeDashoffset: circumference }}
          animate={{ strokeDashoffset: offset }}
          transition={{ duration: 1.5, ease: 'easeOut' }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-4xl font-bold" style={{ color: getColor(progress) }}>
          {progress.toFixed(0)}
        </span>
        <span className="text-sm text-zinc-500">Score</span>
      </div>
    </div>
  );
};

const MetricCard: React.FC<{
  label: string;
  value: string | number;
  unit?: string;
  icon: React.ReactNode;
  status?: 'good' | 'warning' | 'bad';
}> = ({ label, value, unit, icon, status }) => {
  const statusColors = {
    good: 'text-emerald-400',
    warning: 'text-amber-400',
    bad: 'text-red-400',
  };

  return (
    <div className="bg-zinc-800/50 rounded-xl p-4 border border-zinc-700/50">
      <div className="flex items-center gap-2 text-zinc-400 text-sm mb-2">
        {icon}
        {label}
      </div>
      <div className={`text-2xl font-bold ${status ? statusColors[status] : 'text-white'}`}>
        {value}
        {unit && <span className="text-base text-zinc-500 ml-1">{unit}</span>}
      </div>
    </div>
  );
};

const PatternCard: React.FC<{ pattern: Pattern }> = ({ pattern }) => {
  const patternLabels: Record<string, string> = {
    uniform_offset: 'Deslocamento Uniforme',
    uniform_scale: 'Escala Uniforme',
    directional_trend: 'Tendência Direcional',
    local_hotspot: 'Ponto Quente Local',
    periodic: 'Padrão Periódico',
    random: 'Variabilidade Aleatória',
    warping: 'Deformação',
    taper: 'Afilamento',
    twist: 'Torção',
  };

  const patternIcons: Record<string, React.ReactNode> = {
    uniform_offset: <ArrowRight className="w-5 h-5" />,
    uniform_scale: <Layers className="w-5 h-5" />,
    directional_trend: <TrendingUp className="w-5 h-5" />,
    local_hotspot: <Target className="w-5 h-5" />,
    periodic: <Activity className="w-5 h-5" />,
    random: <Grid3X3 className="w-5 h-5" />,
    warping: <TrendingDown className="w-5 h-5" />,
  };

  return (
    <div className="bg-zinc-800/50 rounded-xl p-4 border border-zinc-700/50">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <div className="p-2 bg-cyan-500/20 rounded-lg text-cyan-400">
            {patternIcons[pattern.pattern] || <Box className="w-5 h-5" />}
          </div>
          <span className="font-medium text-white">
            {patternLabels[pattern.pattern] || pattern.pattern}
          </span>
        </div>
        <div className="flex items-center gap-1">
          <div
            className="h-2 w-16 bg-zinc-700 rounded-full overflow-hidden"
          >
            <motion.div
              className="h-full bg-gradient-to-r from-cyan-500 to-blue-500"
              initial={{ width: 0 }}
              animate={{ width: `${pattern.confidence * 100}%` }}
              transition={{ duration: 0.5 }}
            />
          </div>
          <span className="text-sm text-zinc-400 ml-1">
            {(pattern.confidence * 100).toFixed(0)}%
          </span>
        </div>
      </div>
      <div className="space-y-1">
        {pattern.evidence.slice(0, 3).map((e, i) => (
          <p key={i} className="text-sm text-zinc-400 flex items-start gap-2">
            <ChevronRight className="w-4 h-4 mt-0.5 text-zinc-600" />
            {e}
          </p>
        ))}
      </div>
    </div>
  );
};

const RootCauseCard: React.FC<{ cause: RootCause }> = ({ cause }) => {
  const categoryLabels: Record<string, string> = {
    fixturing: 'Fixação',
    calibration: 'Calibração',
    tool_wear: 'Desgaste Ferramenta',
    thermal: 'Térmico',
    material: 'Material',
    vibration: 'Vibração',
    programming: 'Programação',
    machine: 'Máquina',
  };

  const categoryColors: Record<string, string> = {
    fixturing: 'bg-blue-500/20 text-blue-400',
    calibration: 'bg-purple-500/20 text-purple-400',
    tool_wear: 'bg-orange-500/20 text-orange-400',
    thermal: 'bg-red-500/20 text-red-400',
    material: 'bg-amber-500/20 text-amber-400',
    vibration: 'bg-pink-500/20 text-pink-400',
    programming: 'bg-cyan-500/20 text-cyan-400',
    machine: 'bg-green-500/20 text-green-400',
  };

  return (
    <div className="bg-zinc-800/50 rounded-xl p-4 border border-zinc-700/50">
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-2">
          <span
            className={`px-2 py-1 text-xs font-medium rounded ${
              categoryColors[cause.category] || 'bg-zinc-500/20 text-zinc-400'
            }`}
          >
            {categoryLabels[cause.category] || cause.category}
          </span>
          <span className="text-sm text-zinc-400">
            {(cause.confidence * 100).toFixed(0)}% confiança
          </span>
        </div>
        <Search className="w-4 h-4 text-zinc-500" />
      </div>
      <p className="text-white font-medium mb-2">{cause.description}</p>
      <div className="text-xs text-zinc-500">
        Padrões: {cause.patterns_linked.join(', ')}
      </div>
    </div>
  );
};

const ActionCard: React.FC<{ action: CorrectiveAction }> = ({ action }) => {
  const priorityColors: Record<string, string> = {
    high: 'bg-red-500/20 text-red-400 border-red-500/30',
    medium: 'bg-amber-500/20 text-amber-400 border-amber-500/30',
    low: 'bg-green-500/20 text-green-400 border-green-500/30',
  };

  const priorityLabels: Record<string, string> = {
    high: 'Alta',
    medium: 'Média',
    low: 'Baixa',
  };

  return (
    <div className="bg-zinc-800/50 rounded-xl p-4 border border-zinc-700/50">
      <div className="flex items-start gap-3">
        <div className="p-2 bg-emerald-500/20 rounded-lg text-emerald-400">
          <Wrench className="w-5 h-5" />
        </div>
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-1">
            <span className="font-medium text-white">{action.action}</span>
            <span
              className={`px-2 py-0.5 text-xs font-medium rounded border ${
                priorityColors[action.priority]
              }`}
            >
              {priorityLabels[action.priority]}
            </span>
          </div>
          <p className="text-sm text-zinc-400">{action.expected_impact}</p>
        </div>
      </div>
    </div>
  );
};

const AnalysisDetail: React.FC<{ analysis: AnalysisResult }> = ({ analysis }) => {
  const df = analysis.deviation_field;
  
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h2 className="text-2xl font-bold text-white mb-2">
            Análise: {analysis.analysis_id}
          </h2>
          <div className="flex items-center gap-4 text-sm text-zinc-400">
            <span>CAD: {analysis.cad_name}</span>
            <span>Scan: {analysis.scan_name}</span>
            <span>{new Date(analysis.timestamp).toLocaleString()}</span>
          </div>
        </div>
        <QualityBadge quality={analysis.overall_quality} />
      </div>

      {/* Score + Metrics */}
      <div className="grid grid-cols-12 gap-6">
        <div className="col-span-3 flex justify-center">
          <DeviationScoreGauge score={df.deviation_score} />
        </div>
        <div className="col-span-9 grid grid-cols-3 gap-4">
          <MetricCard
            label="Desvio Médio"
            value={df.mean_deviation.toFixed(3)}
            unit="mm"
            icon={<Ruler className="w-4 h-4" />}
            status={df.mean_deviation > df.tolerance ? 'bad' : 'good'}
          />
          <MetricCard
            label="Desvio Máximo"
            value={df.max_deviation.toFixed(3)}
            unit="mm"
            icon={<TrendingUp className="w-4 h-4" />}
            status={df.max_deviation > df.tolerance * 2 ? 'bad' : df.max_deviation > df.tolerance ? 'warning' : 'good'}
          />
          <MetricCard
            label="Fora de Tolerância"
            value={df.pct_out_of_tolerance.toFixed(1)}
            unit="%"
            icon={<AlertTriangle className="w-4 h-4" />}
            status={df.pct_out_of_tolerance > 10 ? 'bad' : df.pct_out_of_tolerance > 5 ? 'warning' : 'good'}
          />
          <MetricCard
            label="Pontos Analisados"
            value={df.n_points.toLocaleString()}
            icon={<Grid3X3 className="w-4 h-4" />}
          />
          <MetricCard
            label="Tolerância"
            value={df.tolerance.toFixed(2)}
            unit="mm"
            icon={<Target className="w-4 h-4" />}
          />
          <MetricCard
            label="RMSE Alinhamento"
            value={df.alignment_rmse.toFixed(4)}
            unit="mm"
            icon={<Crosshair className="w-4 h-4" />}
          />
        </div>
      </div>

      {/* Patterns */}
      {analysis.identified_patterns.length > 0 && (
        <div>
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <FileSearch className="w-5 h-5 text-cyan-500" />
            Padrões Identificados
          </h3>
          <div className="grid grid-cols-2 gap-4">
            {analysis.identified_patterns.map((pattern, i) => (
              <PatternCard key={i} pattern={pattern} />
            ))}
          </div>
        </div>
      )}

      {/* Root Causes */}
      {analysis.root_causes.length > 0 && (
        <div>
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <Search className="w-5 h-5 text-amber-500" />
            Causas Prováveis (RCA)
          </h3>
          <div className="grid grid-cols-2 gap-4">
            {analysis.root_causes.map((cause, i) => (
              <RootCauseCard key={i} cause={cause} />
            ))}
          </div>
        </div>
      )}

      {/* Corrective Actions */}
      {analysis.corrective_actions.length > 0 && (
        <div>
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <Lightbulb className="w-5 h-5 text-emerald-500" />
            Ações Corretivas Recomendadas
          </h3>
          <div className="space-y-3">
            {analysis.corrective_actions.map((action, i) => (
              <ActionCard key={i} action={action} />
            ))}
          </div>
        </div>
      )}

      {/* Summary */}
      <div className="bg-zinc-800/50 rounded-xl p-4 border border-zinc-700/50">
        <h3 className="text-sm font-medium text-zinc-400 mb-2">Resumo</h3>
        <pre className="text-sm text-zinc-300 whitespace-pre-wrap font-mono">
          {analysis.summary_text}
        </pre>
      </div>
    </div>
  );
};

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN PAGE
// ═══════════════════════════════════════════════════════════════════════════════

const XAIDTProduct: React.FC = () => {
  const [selectedAnalysis, setSelectedAnalysis] = useState<string | null>(null);
  const [demoParams, setDemoParams] = useState({
    n_points: 500,
    deviation_type: 'random',
    deviation_magnitude: 0.8,
    tolerance: 0.5,
  });
  
  const queryClient = useQueryClient();

  const { data: analysesData, isLoading: analysesLoading } = useQuery({
    queryKey: ['xai-dt-analyses'],
    queryFn: fetchAnalyses,
  });

  const { data: analysisDetail, isLoading: detailLoading } = useQuery({
    queryKey: ['xai-dt-analysis', selectedAnalysis],
    queryFn: () => (selectedAnalysis ? fetchAnalysis(selectedAnalysis) : Promise.resolve(null)),
    enabled: !!selectedAnalysis,
  });

  const demoMutation = useMutation({
    mutationFn: runDemoAnalysis,
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['xai-dt-analyses'] });
      setSelectedAnalysis(data.analysis_id);
    },
  });

  const analyses = analysesData?.analyses || [];

  return (
    <div className="min-h-screen bg-gradient-to-br from-zinc-950 via-zinc-900 to-zinc-950 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-3xl font-bold text-white mb-2 flex items-center gap-3">
              <div className="p-2 bg-gradient-to-br from-purple-500/20 to-pink-500/20 rounded-xl">
                <Box className="w-8 h-8 text-purple-500" />
              </div>
              XAI-DT: Qualidade Geométrica do Produto
            </h1>
            <p className="text-zinc-400">
              Análise explicável CAD vs Scan com ICP, PCA e Root Cause Analysis
            </p>
          </div>
        </div>

        {/* Demo Controls */}
        <div className="bg-zinc-900/50 rounded-xl p-4 border border-zinc-800 mb-6">
          <h3 className="text-sm font-medium text-zinc-400 mb-4 flex items-center gap-2">
            <Settings className="w-4 h-4" />
            Análise Demo
          </h3>
          <div className="flex flex-wrap items-end gap-4">
            <div>
              <label className="block text-xs text-zinc-500 mb-1">Pontos</label>
              <input
                type="number"
                value={demoParams.n_points}
                onChange={(e) => setDemoParams({ ...demoParams, n_points: parseInt(e.target.value) })}
                className="w-24 px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-white text-sm"
              />
            </div>
            <div>
              <label className="block text-xs text-zinc-500 mb-1">Tipo Desvio</label>
              <select
                value={demoParams.deviation_type}
                onChange={(e) => setDemoParams({ ...demoParams, deviation_type: e.target.value })}
                className="w-32 px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-white text-sm"
              >
                <option value="offset">Offset</option>
                <option value="scale">Escala</option>
                <option value="random">Aleatório</option>
                <option value="local">Local</option>
              </select>
            </div>
            <div>
              <label className="block text-xs text-zinc-500 mb-1">Magnitude (mm)</label>
              <input
                type="number"
                step="0.1"
                value={demoParams.deviation_magnitude}
                onChange={(e) => setDemoParams({ ...demoParams, deviation_magnitude: parseFloat(e.target.value) })}
                className="w-24 px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-white text-sm"
              />
            </div>
            <div>
              <label className="block text-xs text-zinc-500 mb-1">Tolerância (mm)</label>
              <input
                type="number"
                step="0.1"
                value={demoParams.tolerance}
                onChange={(e) => setDemoParams({ ...demoParams, tolerance: parseFloat(e.target.value) })}
                className="w-24 px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-white text-sm"
              />
            </div>
            <button
              onClick={() => demoMutation.mutate(demoParams)}
              disabled={demoMutation.isPending}
              className="flex items-center gap-2 px-4 py-2 bg-purple-500/20 hover:bg-purple-500/30 text-purple-400 rounded-lg transition-colors border border-purple-500/30"
            >
              {demoMutation.isPending ? (
                <RefreshCw className="w-4 h-4 animate-spin" />
              ) : (
                <Play className="w-4 h-4" />
              )}
              Executar Análise
            </button>
          </div>
        </div>

        {/* Main Content */}
        <div className="grid grid-cols-12 gap-6">
          {/* Analysis List */}
          <div className="col-span-3">
            <div className="bg-zinc-900/50 rounded-xl p-4 border border-zinc-800">
              <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                <FileSearch className="w-5 h-5 text-purple-500" />
                Análises ({analyses.length})
              </h3>
              {analysesLoading ? (
                <div className="flex items-center justify-center py-8">
                  <RefreshCw className="w-6 h-6 text-purple-500 animate-spin" />
                </div>
              ) : analyses.length === 0 ? (
                <div className="text-center py-8 text-zinc-500">
                  <Box className="w-8 h-8 mx-auto mb-2 opacity-50" />
                  <p>Nenhuma análise</p>
                  <p className="text-sm">Execute uma análise demo</p>
                </div>
              ) : (
                <div className="space-y-2 max-h-[500px] overflow-y-auto">
                  <AnimatePresence>
                    {analyses.map((a) => (
                      <motion.div
                        key={a.analysis_id}
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        onClick={() => setSelectedAnalysis(a.analysis_id)}
                        className={`cursor-pointer p-3 rounded-lg border transition-all ${
                          selectedAnalysis === a.analysis_id
                            ? 'bg-purple-500/10 border-purple-500/30'
                            : 'bg-zinc-800/50 border-zinc-700/50 hover:border-zinc-600'
                        }`}
                      >
                        <div className="flex items-center justify-between mb-1">
                          <span className="text-xs font-mono text-zinc-400">
                            {a.analysis_id}
                          </span>
                          <QualityBadge quality={a.overall_quality} />
                        </div>
                        <div className="flex items-center justify-between">
                          <span className="text-sm text-zinc-300">{a.scan_name}</span>
                          <span className="text-sm font-medium text-white">
                            {a.deviation_score.toFixed(0)}%
                          </span>
                        </div>
                      </motion.div>
                    ))}
                  </AnimatePresence>
                </div>
              )}
            </div>
          </div>

          {/* Analysis Detail */}
          <div className="col-span-9">
            <div className="bg-zinc-900/50 rounded-xl p-6 border border-zinc-800 min-h-[600px]">
              {detailLoading ? (
                <div className="flex items-center justify-center h-full">
                  <RefreshCw className="w-8 h-8 text-purple-500 animate-spin" />
                </div>
              ) : analysisDetail ? (
                <AnalysisDetail analysis={analysisDetail} />
              ) : (
                <div className="flex items-center justify-center h-full text-zinc-500">
                  <div className="text-center">
                    <ChevronRight className="w-12 h-12 mx-auto mb-2 opacity-30" />
                    <p>Selecione uma análise ou execute uma nova</p>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="mt-8 text-center text-xs text-zinc-600">
          <p>
            XAI-DT utiliza alinhamento ICP, análise PCA de desvios e Root Cause Analysis
            baseado em regras para identificar causas prováveis de desvios geométricos.
          </p>
          <p className="mt-1">R&D / SIFIDE: WP1 - Digital Twin & Explainability</p>
        </div>
      </div>
    </div>
  );
};

export default XAIDTProduct;



