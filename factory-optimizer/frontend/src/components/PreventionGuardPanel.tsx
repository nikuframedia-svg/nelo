/**
 * Prevention Guard Panel - Rules Editor & Risk Prediction
 * 
 * Features:
 * - Rule toggle on/off
 * - Custom rules editor
 * - Training data upload
 * - Risk predictions visualization
 */

import React, { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { motion, AnimatePresence } from 'framer-motion';
import {
  AlertTriangle,
  Brain,
  Check,
  ChevronDown,
  ChevronRight,
  Edit3,
  Eye,
  FileUp,
  Filter,
  Loader2,
  Plus,
  RefreshCw,
  Save,
  Settings,
  Shield,
  ShieldCheck,
  ToggleLeft,
  ToggleRight,
  Trash2,
  Upload,
  X,
  Zap,
} from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

// ═══════════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════════

interface ValidationRule {
  rule_id: string;
  name: string;
  description: string;
  category: 'pdm' | 'shopfloor' | 'predictive';
  severity: 'warning' | 'error' | 'critical';
  enabled: boolean;
  conditions: string;
  action: string;
  created_at: string;
}

interface RiskPrediction {
  order_id: string;
  product_name: string;
  machine_id: string;
  risk_score: number;
  risk_level: 'low' | 'medium' | 'high' | 'critical';
  factors: { name: string; contribution: number }[];
  recommendations: string[];
  timestamp: string;
}

interface TrainingData {
  id: string;
  filename: string;
  records_count: number;
  defect_rate: number;
  uploaded_at: string;
  status: 'pending' | 'processed' | 'error';
}

// ═══════════════════════════════════════════════════════════════════════════════
// API
// ═══════════════════════════════════════════════════════════════════════════════

async function fetchRules(): Promise<ValidationRule[]> {
  try {
    const res = await fetch(`${API_BASE}/prevention-guard/rules`);
    if (!res.ok) throw new Error();
    return res.json();
  } catch {
    // Mock data for demo
    return [
      { rule_id: 'r1', name: 'BOM Duplicates', description: 'Detecta componentes duplicados na BOM', category: 'pdm', severity: 'error', enabled: true, conditions: 'bom.has_duplicates()', action: 'block_release', created_at: new Date().toISOString() },
      { rule_id: 'r2', name: 'Obsolete Components', description: 'Bloqueia uso de componentes obsoletos', category: 'pdm', severity: 'critical', enabled: true, conditions: 'component.status == "obsolete"', action: 'block_release', created_at: new Date().toISOString() },
      { rule_id: 'r3', name: 'Material Verification', description: 'Verifica material via barcode/RFID', category: 'shopfloor', severity: 'error', enabled: true, conditions: 'material.scanned != order.required', action: 'block_start', created_at: new Date().toISOString() },
      { rule_id: 'r4', name: 'Tool Calibration', description: 'Verifica calibração de ferramentas', category: 'shopfloor', severity: 'warning', enabled: false, conditions: 'tool.calibration_date > 30_days', action: 'alert', created_at: new Date().toISOString() },
      { rule_id: 'r5', name: 'High Defect Risk', description: 'ML prediz alto risco de defeito', category: 'predictive', severity: 'warning', enabled: true, conditions: 'predict_defect(context) > 0.3', action: 'require_inspection', created_at: new Date().toISOString() },
    ];
  }
}

async function toggleRule(ruleId: string, enabled: boolean): Promise<ValidationRule> {
  const res = await fetch(`${API_BASE}/prevention-guard/rules/${ruleId}/toggle`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ enabled }),
  });
  if (!res.ok) throw new Error('Failed to toggle rule');
  return res.json();
}

async function fetchRiskPredictions(): Promise<RiskPrediction[]> {
  try {
    const res = await fetch(`${API_BASE}/prevention-guard/risk-predictions`);
    if (!res.ok) throw new Error();
    return res.json();
  } catch {
    return [
      { order_id: 'OP-001', product_name: 'Widget A', machine_id: 'CNC-01', risk_score: 0.72, risk_level: 'high', factors: [{ name: 'Turno noturno', contribution: 0.25 }, { name: 'Máquina próxima de manutenção', contribution: 0.3 }, { name: 'Novo operador', contribution: 0.17 }], recommendations: ['Solicitar dupla verificação', 'Programar inspeção extra'], timestamp: new Date().toISOString() },
      { order_id: 'OP-002', product_name: 'Gadget B', machine_id: 'PRESS-02', risk_score: 0.15, risk_level: 'low', factors: [{ name: 'Operador experiente', contribution: -0.2 }, { name: 'Máquina recém calibrada', contribution: -0.15 }], recommendations: [], timestamp: new Date().toISOString() },
      { order_id: 'OP-003', product_name: 'Part C', machine_id: 'LATHE-03', risk_score: 0.45, risk_level: 'medium', factors: [{ name: 'Material de novo fornecedor', contribution: 0.2 }, { name: 'Primeiro lote do dia', contribution: 0.15 }], recommendations: ['Monitorar primeiras peças'], timestamp: new Date().toISOString() },
    ];
  }
}

async function fetchTrainingData(): Promise<TrainingData[]> {
  try {
    const res = await fetch(`${API_BASE}/prevention-guard/training-data`);
    if (!res.ok) throw new Error();
    return res.json();
  } catch {
    return [
      { id: 'td1', filename: 'defects_2024_Q3.csv', records_count: 15420, defect_rate: 0.032, uploaded_at: '2024-10-15T10:30:00Z', status: 'processed' },
      { id: 'td2', filename: 'quality_logs_september.xlsx', records_count: 8750, defect_rate: 0.028, uploaded_at: '2024-10-01T14:15:00Z', status: 'processed' },
    ];
  }
}

async function uploadTrainingData(file: File): Promise<TrainingData> {
  const formData = new FormData();
  formData.append('file', file);
  const res = await fetch(`${API_BASE}/prevention-guard/training-data/upload`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) throw new Error('Upload failed');
  return res.json();
}

// ═══════════════════════════════════════════════════════════════════════════════
// COMPONENTS
// ═══════════════════════════════════════════════════════════════════════════════

const SeverityBadge: React.FC<{ severity: string }> = ({ severity }) => {
  const config: Record<string, { bg: string; text: string }> = {
    warning: { bg: 'bg-amber-500/20', text: 'text-amber-400' },
    error: { bg: 'bg-red-500/20', text: 'text-red-400' },
    critical: { bg: 'bg-red-600/30', text: 'text-red-300' },
  };
  const c = config[severity] || config.warning;
  
  return (
    <span className={`px-2 py-0.5 rounded text-xs font-medium ${c.bg} ${c.text}`}>
      {severity.toUpperCase()}
    </span>
  );
};

const CategoryBadge: React.FC<{ category: string }> = ({ category }) => {
  const config: Record<string, { bg: string; text: string; label: string }> = {
    pdm: { bg: 'bg-purple-500/20', text: 'text-purple-400', label: 'PDM Guard' },
    shopfloor: { bg: 'bg-cyan-500/20', text: 'text-cyan-400', label: 'Shopfloor' },
    predictive: { bg: 'bg-emerald-500/20', text: 'text-emerald-400', label: 'Predictive' },
  };
  const c = config[category] || config.pdm;
  
  return (
    <span className={`px-2 py-0.5 rounded text-xs font-medium ${c.bg} ${c.text}`}>
      {c.label}
    </span>
  );
};

const RiskGauge: React.FC<{ score: number }> = ({ score }) => {
  const getColor = () => {
    if (score < 0.2) return 'from-emerald-500 to-emerald-400';
    if (score < 0.4) return 'from-amber-500 to-amber-400';
    if (score < 0.7) return 'from-orange-500 to-orange-400';
    return 'from-red-500 to-red-400';
  };
  
  return (
    <div className="relative w-16 h-3 bg-slate-700 rounded-full overflow-hidden">
      <div 
        className={`absolute inset-y-0 left-0 rounded-full bg-gradient-to-r ${getColor()}`}
        style={{ width: `${score * 100}%` }}
      />
    </div>
  );
};

const RuleCard: React.FC<{
  rule: ValidationRule;
  onToggle: (enabled: boolean) => void;
  onEdit?: () => void;
}> = ({ rule, onToggle, onEdit }) => (
  <motion.div
    layout
    initial={{ opacity: 0, y: 10 }}
    animate={{ opacity: 1, y: 0 }}
    className={`p-4 rounded-xl border transition-all ${
      rule.enabled 
        ? 'bg-slate-800/50 border-slate-700/50' 
        : 'bg-slate-900/30 border-slate-800/30 opacity-60'
    }`}
  >
    <div className="flex items-start justify-between mb-3">
      <div className="flex-1">
        <div className="flex items-center gap-2 mb-1">
          <span className="font-medium text-white">{rule.name}</span>
          <SeverityBadge severity={rule.severity} />
          <CategoryBadge category={rule.category} />
        </div>
        <p className="text-sm text-slate-400">{rule.description}</p>
      </div>
      <button
        onClick={() => onToggle(!rule.enabled)}
        className={`p-1 rounded-lg transition-colors ${
          rule.enabled 
            ? 'text-emerald-400 hover:bg-emerald-500/20' 
            : 'text-slate-500 hover:bg-slate-700'
        }`}
      >
        {rule.enabled ? <ToggleRight className="w-8 h-8" /> : <ToggleLeft className="w-8 h-8" />}
      </button>
    </div>
    
    <div className="flex items-center gap-4 text-xs">
      <div className="flex-1">
        <span className="text-slate-500">Condição:</span>
        <code className="ml-1 text-cyan-400 font-mono">{rule.conditions}</code>
      </div>
      <div>
        <span className="text-slate-500">Ação:</span>
        <span className="ml-1 text-amber-400">{rule.action}</span>
      </div>
    </div>
  </motion.div>
);

const RiskCard: React.FC<{ prediction: RiskPrediction }> = ({ prediction }) => {
  const [expanded, setExpanded] = useState(false);
  
  const levelColors = {
    low: 'border-emerald-500/30 bg-emerald-500/5',
    medium: 'border-amber-500/30 bg-amber-500/5',
    high: 'border-orange-500/30 bg-orange-500/5',
    critical: 'border-red-500/30 bg-red-500/5',
  };

  return (
    <motion.div
      layout
      className={`p-4 rounded-xl border ${levelColors[prediction.risk_level]}`}
    >
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-3">
          <div className="text-center">
            <p className="text-2xl font-bold text-white">{Math.round(prediction.risk_score * 100)}%</p>
            <p className="text-xs text-slate-500">Risco</p>
          </div>
          <div>
            <p className="font-medium text-white">{prediction.order_id}</p>
            <p className="text-sm text-slate-400">{prediction.product_name}</p>
          </div>
        </div>
        <button 
          onClick={() => setExpanded(!expanded)}
          className="p-2 text-slate-400 hover:bg-slate-800 rounded-lg"
        >
          {expanded ? <ChevronDown className="w-5 h-5" /> : <ChevronRight className="w-5 h-5" />}
        </button>
      </div>
      
      <RiskGauge score={prediction.risk_score} />
      
      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="mt-4 pt-4 border-t border-slate-700/50 overflow-hidden"
          >
            <div className="mb-3">
              <h4 className="text-xs text-slate-500 mb-2">FATORES CONTRIBUINTES</h4>
              <div className="space-y-2">
                {prediction.factors.map((factor, idx) => (
                  <div key={idx} className="flex items-center justify-between text-sm">
                    <span className="text-slate-300">{factor.name}</span>
                    <span className={factor.contribution > 0 ? 'text-red-400' : 'text-emerald-400'}>
                      {factor.contribution > 0 ? '+' : ''}{(factor.contribution * 100).toFixed(0)}%
                    </span>
                  </div>
                ))}
              </div>
            </div>
            
            {prediction.recommendations.length > 0 && (
              <div>
                <h4 className="text-xs text-slate-500 mb-2">RECOMENDAÇÕES</h4>
                <ul className="space-y-1">
                  {prediction.recommendations.map((rec, idx) => (
                    <li key={idx} className="flex items-center gap-2 text-sm text-amber-400">
                      <AlertTriangle className="w-3 h-3" />
                      {rec}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
};

const CreateRuleModal: React.FC<{
  isOpen: boolean;
  onClose: () => void;
  onCreate: (rule: Partial<ValidationRule>) => void;
}> = ({ isOpen, onClose, onCreate }) => {
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [category, setCategory] = useState<'pdm' | 'shopfloor' | 'predictive'>('pdm');
  const [severity, setSeverity] = useState<'warning' | 'error' | 'critical'>('warning');
  const [conditions, setConditions] = useState('');
  const [action, setAction] = useState('alert');

  if (!isOpen) return null;

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4"
      onClick={onClose}
    >
      <motion.div
        initial={{ scale: 0.95 }}
        animate={{ scale: 1 }}
        className="bg-slate-900 rounded-2xl p-6 max-w-lg w-full"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-xl font-bold text-white flex items-center gap-2">
            <Plus className="w-5 h-5 text-cyan-400" />
            Nova Regra de Validação
          </h2>
          <button onClick={onClose} className="p-2 text-slate-400 hover:bg-slate-800 rounded-lg">
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="space-y-4">
          <div>
            <label className="block text-sm text-slate-400 mb-1">Nome</label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white"
              placeholder="Nome da regra..."
            />
          </div>

          <div>
            <label className="block text-sm text-slate-400 mb-1">Descrição</label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white resize-none"
              rows={2}
              placeholder="Descrição..."
            />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm text-slate-400 mb-1">Categoria</label>
              <select
                value={category}
                onChange={(e) => setCategory(e.target.value as any)}
                className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white"
              >
                <option value="pdm">PDM Guard</option>
                <option value="shopfloor">Shopfloor Guard</option>
                <option value="predictive">Predictive Guard</option>
              </select>
            </div>
            <div>
              <label className="block text-sm text-slate-400 mb-1">Severidade</label>
              <select
                value={severity}
                onChange={(e) => setSeverity(e.target.value as any)}
                className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white"
              >
                <option value="warning">Warning</option>
                <option value="error">Error</option>
                <option value="critical">Critical</option>
              </select>
            </div>
          </div>

          <div>
            <label className="block text-sm text-slate-400 mb-1">Condição (expressão)</label>
            <input
              type="text"
              value={conditions}
              onChange={(e) => setConditions(e.target.value)}
              className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white font-mono text-sm"
              placeholder="component.quantity <= 0"
            />
          </div>

          <div>
            <label className="block text-sm text-slate-400 mb-1">Ação</label>
            <select
              value={action}
              onChange={(e) => setAction(e.target.value)}
              className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white"
            >
              <option value="alert">Alertar</option>
              <option value="block_release">Bloquear Release</option>
              <option value="block_start">Bloquear Início</option>
              <option value="require_inspection">Exigir Inspeção</option>
              <option value="require_approval">Exigir Aprovação</option>
            </select>
          </div>
        </div>

        <div className="flex gap-3 mt-6 pt-4 border-t border-slate-700">
          <button
            onClick={onClose}
            className="flex-1 py-2 bg-slate-700 hover:bg-slate-600 text-white rounded-lg"
          >
            Cancelar
          </button>
          <button
            onClick={() => {
              onCreate({ name, description, category, severity, conditions, action, enabled: true });
              onClose();
            }}
            disabled={!name || !conditions}
            className="flex-1 py-2 bg-cyan-600 hover:bg-cyan-500 text-white rounded-lg disabled:opacity-50 flex items-center justify-center gap-2"
          >
            <Save className="w-4 h-4" />
            Criar Regra
          </button>
        </div>
      </motion.div>
    </motion.div>
  );
};

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN COMPONENT
// ═══════════════════════════════════════════════════════════════════════════════

export const PreventionGuardPanel: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'rules' | 'predictions' | 'training'>('rules');
  const [categoryFilter, setCategoryFilter] = useState<string>('all');
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [uploadingFile, setUploadingFile] = useState<File | null>(null);
  
  const queryClient = useQueryClient();

  const { data: rules, isLoading: rulesLoading } = useQuery({
    queryKey: ['prevention-guard-rules'],
    queryFn: fetchRules,
  });

  const { data: predictions, isLoading: predictionsLoading } = useQuery({
    queryKey: ['prevention-guard-predictions'],
    queryFn: fetchRiskPredictions,
    refetchInterval: 30000,
  });

  const { data: trainingData, isLoading: trainingLoading } = useQuery({
    queryKey: ['prevention-guard-training'],
    queryFn: fetchTrainingData,
  });

  const toggleMutation = useMutation({
    mutationFn: ({ ruleId, enabled }: { ruleId: string; enabled: boolean }) => 
      toggleRule(ruleId, enabled),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['prevention-guard-rules'] });
    },
  });

  const uploadMutation = useMutation({
    mutationFn: uploadTrainingData,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['prevention-guard-training'] });
      setUploadingFile(null);
    },
  });

  const filteredRules = rules?.filter((r) => {
    if (categoryFilter === 'all') return true;
    return r.category === categoryFilter;
  }) || [];

  const tabs = [
    { id: 'rules', label: 'Regras de Validação', icon: Shield, count: rules?.length },
    { id: 'predictions', label: 'Previsões de Risco', icon: Brain, count: predictions?.filter(p => p.risk_level !== 'low').length },
    { id: 'training', label: 'Dados de Treino', icon: Upload, count: trainingData?.length },
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold text-white flex items-center gap-2">
            <ShieldCheck className="w-5 h-5 text-cyan-500" />
            Prevention Guard: Process & Quality Guard
          </h2>
          <p className="text-sm text-slate-400 mt-1">
            Regras de validação, previsão de risco e dados de treino
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
            {tab.count !== undefined && tab.count > 0 && (
              <span className="px-1.5 py-0.5 bg-slate-700 rounded text-xs">{tab.count}</span>
            )}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      {activeTab === 'rules' && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="space-y-4"
        >
          {/* Filters & Actions */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Filter className="w-4 h-4 text-slate-500" />
              {['all', 'pdm', 'shopfloor', 'predictive'].map((cat) => (
                <button
                  key={cat}
                  onClick={() => setCategoryFilter(cat)}
                  className={`px-3 py-1 rounded-lg text-sm ${
                    categoryFilter === cat
                      ? 'bg-cyan-500/20 text-cyan-400'
                      : 'text-slate-400 hover:bg-slate-800'
                  }`}
                >
                  {cat === 'all' ? 'Todas' : cat === 'pdm' ? 'PDM' : cat === 'shopfloor' ? 'Shopfloor' : 'Predictive'}
                </button>
              ))}
            </div>
            <button
              onClick={() => setShowCreateModal(true)}
              className="flex items-center gap-2 px-4 py-2 bg-cyan-600 hover:bg-cyan-500 text-white rounded-lg font-medium"
            >
              <Plus className="w-4 h-4" />
              Nova Regra
            </button>
          </div>

          {/* Rules List */}
          <div className="space-y-3">
            {rulesLoading ? (
              <div className="flex items-center justify-center py-12">
                <Loader2 className="w-8 h-8 text-cyan-400 animate-spin" />
              </div>
            ) : (
              filteredRules.map((rule) => (
                <RuleCard
                  key={rule.rule_id}
                  rule={rule}
                  onToggle={(enabled) => toggleMutation.mutate({ ruleId: rule.rule_id, enabled })}
                />
              ))
            )}
          </div>
        </motion.div>
      )}

      {activeTab === 'predictions' && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="space-y-4"
        >
          {/* Summary */}
          <div className="grid grid-cols-4 gap-4">
            <div className="p-4 rounded-xl border border-emerald-500/30 bg-emerald-500/5">
              <div className="flex items-center gap-2 text-emerald-400 mb-2">
                <Check className="w-4 h-4" />
                <span className="text-sm">Baixo Risco</span>
              </div>
              <p className="text-3xl font-bold text-white">
                {predictions?.filter(p => p.risk_level === 'low').length || 0}
              </p>
            </div>
            <div className="p-4 rounded-xl border border-amber-500/30 bg-amber-500/5">
              <div className="flex items-center gap-2 text-amber-400 mb-2">
                <AlertTriangle className="w-4 h-4" />
                <span className="text-sm">Médio</span>
              </div>
              <p className="text-3xl font-bold text-white">
                {predictions?.filter(p => p.risk_level === 'medium').length || 0}
              </p>
            </div>
            <div className="p-4 rounded-xl border border-orange-500/30 bg-orange-500/5">
              <div className="flex items-center gap-2 text-orange-400 mb-2">
                <Zap className="w-4 h-4" />
                <span className="text-sm">Alto</span>
              </div>
              <p className="text-3xl font-bold text-white">
                {predictions?.filter(p => p.risk_level === 'high').length || 0}
              </p>
            </div>
            <div className="p-4 rounded-xl border border-red-500/30 bg-red-500/5">
              <div className="flex items-center gap-2 text-red-400 mb-2">
                <Shield className="w-4 h-4" />
                <span className="text-sm">Crítico</span>
              </div>
              <p className="text-3xl font-bold text-white">
                {predictions?.filter(p => p.risk_level === 'critical').length || 0}
              </p>
            </div>
          </div>

          {/* Risk Cards */}
          <div className="grid grid-cols-2 gap-4">
            {predictionsLoading ? (
              <div className="col-span-2 flex items-center justify-center py-12">
                <Loader2 className="w-8 h-8 text-cyan-400 animate-spin" />
              </div>
            ) : (
              predictions?.map((pred) => (
                <RiskCard key={pred.order_id} prediction={pred} />
              ))
            )}
          </div>

          {/* Model Info */}
          <div className="p-4 bg-slate-900/50 rounded-lg border border-slate-700/50">
            <h4 className="text-sm font-medium text-white mb-2">Modelo de Previsão de Risco</h4>
            <code className="text-xs text-cyan-400 font-mono block">
              P(Defeito|X) = σ(β₀ + β₁×Machine + β₂×Operator + ... + βₙ×Interaction)
            </code>
            <p className="text-xs text-slate-500 mt-2">
              Modelo MLP com entropia cruzada, treinado com dados históricos de defeitos
            </p>
          </div>
        </motion.div>
      )}

      {activeTab === 'training' && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="space-y-4"
        >
          {/* Upload Section */}
          <div className="p-6 rounded-xl border-2 border-dashed border-slate-700 bg-slate-800/30 text-center">
            <Upload className="w-12 h-12 text-slate-500 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-white mb-2">Upload de Dados de Treino</h3>
            <p className="text-sm text-slate-400 mb-4">
              Carregue ficheiros CSV ou Excel com histórico de defeitos para treinar o modelo
            </p>
            <label className="inline-flex items-center gap-2 px-6 py-3 bg-cyan-600 hover:bg-cyan-500 text-white rounded-lg cursor-pointer font-medium">
              <FileUp className="w-5 h-5" />
              Selecionar Ficheiro
              <input
                type="file"
                accept=".csv,.xlsx,.xls"
                className="hidden"
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) {
                    setUploadingFile(file);
                    uploadMutation.mutate(file);
                  }
                }}
              />
            </label>
            {uploadMutation.isPending && (
              <p className="mt-4 text-sm text-cyan-400 flex items-center justify-center gap-2">
                <Loader2 className="w-4 h-4 animate-spin" />
                A processar {uploadingFile?.name}...
              </p>
            )}
          </div>

          {/* Training Data List */}
          <div>
            <h3 className="text-lg font-medium text-white mb-3">Ficheiros Carregados</h3>
            <div className="space-y-3">
              {trainingLoading ? (
                <div className="flex items-center justify-center py-8">
                  <Loader2 className="w-6 h-6 text-cyan-400 animate-spin" />
                </div>
              ) : trainingData?.length === 0 ? (
                <div className="text-center py-8 text-slate-400">
                  Nenhum ficheiro de treino carregado
                </div>
              ) : (
                trainingData?.map((td) => (
                  <div key={td.id} className="flex items-center justify-between p-4 rounded-xl border border-slate-700/50 bg-slate-800/30">
                    <div className="flex items-center gap-4">
                      <div className="p-2 bg-cyan-500/20 rounded-lg">
                        <FileUp className="w-5 h-5 text-cyan-400" />
                      </div>
                      <div>
                        <p className="font-medium text-white">{td.filename}</p>
                        <p className="text-sm text-slate-400">
                          {td.records_count.toLocaleString()} registos • {(td.defect_rate * 100).toFixed(1)}% defeitos
                        </p>
                      </div>
                    </div>
                    <div className="flex items-center gap-4">
                      <span className={`px-2 py-0.5 rounded text-xs ${
                        td.status === 'processed' 
                          ? 'bg-emerald-500/20 text-emerald-400' 
                          : td.status === 'pending'
                          ? 'bg-amber-500/20 text-amber-400'
                          : 'bg-red-500/20 text-red-400'
                      }`}>
                        {td.status === 'processed' ? 'Processado' : td.status === 'pending' ? 'Pendente' : 'Erro'}
                      </span>
                      <span className="text-xs text-slate-500">
                        {new Date(td.uploaded_at).toLocaleDateString()}
                      </span>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </motion.div>
      )}

      {/* Create Rule Modal */}
      <AnimatePresence>
        {showCreateModal && (
          <CreateRuleModal
            isOpen={showCreateModal}
            onClose={() => setShowCreateModal(false)}
            onCreate={(rule) => console.log('Create rule:', rule)}
          />
        )}
      </AnimatePresence>
    </div>
  );
};

export default PreventionGuardPanel;


