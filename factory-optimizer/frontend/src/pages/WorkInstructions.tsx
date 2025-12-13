/**
 * ═══════════════════════════════════════════════════════════════════════════════
 * Work Instructions - Operator Interface
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * Interface para operadores executarem instruções de trabalho:
 * - Visualização passo-a-passo
 * - Captura de valores e evidências
 * - Quality checklists
 * - Poka-yoke visual (progresso e validações)
 *
 * R&D / SIFIDE: WP1 - Digital Twin & Shopfloor
 */

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  AlertCircle,
  AlertTriangle,
  ArrowLeft,
  ArrowRight,
  Camera,
  Check,
  CheckCircle,
  ChevronRight,
  Clipboard,
  ClipboardCheck,
  Edit3,
  FileText,
  Hash,
  Image,
  Info,
  Layers,
  List,
  Loader2,
  Lock,
  Pause,
  Play,
  RefreshCw,
  RotateCcw,
  Settings,
  Square,
  Target,
  ThumbsDown,
  ThumbsUp,
  X,
  Zap,
} from 'lucide-react';

// ═══════════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════════

interface Tolerance {
  nominal: number;
  min_value: number;
  max_value: number;
  unit: string;
}

interface VisualReference {
  type: string;
  url: string;
  caption?: string;
}

interface InstructionStep {
  step_id: string;
  sequence: number;
  title: string;
  description: string;
  step_type: string;
  input_type: string;
  visual_references: VisualReference[];
  tolerance?: Tolerance;
  options: string[];
  is_critical: boolean;
  is_quality_check: boolean;
  required: boolean;
}

interface QualityCheck {
  check_id: string;
  sequence: number;
  question: string;
  check_type: string;
  tolerance?: Tolerance;
  is_critical: boolean;
  fail_action: string;
}

interface WorkInstruction {
  instruction_id: string;
  title: string;
  description: string;
  version: number;
  steps: InstructionStep[];
  quality_checks: QualityCheck[];
  status: string;
  model_3d_url?: string;
  estimated_time_minutes: number;
}

interface StepExecution {
  step_id: string;
  status: string;
  completed_at?: string;
  input_value?: any;
  within_tolerance?: boolean;
}

interface Execution {
  execution_id: string;
  instruction_id: string;
  order_id: string;
  status: string;
  current_step_index: number;
  operator_id: string;
  operator_name: string;
  started_at?: string;
  completed_at?: string;
  step_executions: StepExecution[];
  total_steps: number;
  completed_steps: number;
  nok_count: number;
  progress_percent: number;
  instruction?: WorkInstruction;
  current_step?: InstructionStep;
}

// ═══════════════════════════════════════════════════════════════════════════════
// API
// ═══════════════════════════════════════════════════════════════════════════════

const API_BASE = 'http://127.0.0.1:8000/work-instructions';

const fetchStatus = async () => {
  const res = await fetch(`${API_BASE}/status`);
  if (!res.ok) throw new Error('Failed to fetch status');
  return res.json();
};

const createDemoInstruction = async (): Promise<WorkInstruction> => {
  const res = await fetch(`${API_BASE}/demo`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ product_name: 'Motor Assembly', num_steps: 8 }),
  });
  if (!res.ok) throw new Error('Failed to create demo');
  return res.json();
};

const startExecution = async (instructionId: string): Promise<Execution> => {
  const res = await fetch(`${API_BASE}/${instructionId}/execute`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      order_id: `OP-${Date.now()}`,
      operator_id: 'OP001',
      operator_name: 'Operador Demo',
    }),
  });
  if (!res.ok) throw new Error('Failed to start execution');
  return res.json();
};

const completeStep = async (executionId: string, stepId: string, inputValue: any): Promise<Execution> => {
  const res = await fetch(`${API_BASE}/executions/${executionId}/steps/${stepId}/complete`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ input_value: inputValue }),
  });
  if (!res.ok) {
    const error = await res.json();
    throw new Error(error.detail || 'Failed to complete step');
  }
  return res.json();
};

const recordQualityCheck = async (executionId: string, checkId: string, result: string): Promise<any> => {
  const res = await fetch(`${API_BASE}/executions/${executionId}/quality-checks`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ check_id: checkId, result }),
  });
  if (!res.ok) throw new Error('Failed to record check');
  return res.json();
};

const getExecution = async (executionId: string): Promise<Execution> => {
  const res = await fetch(`${API_BASE}/executions/${executionId}`);
  if (!res.ok) throw new Error('Failed to get execution');
  return res.json();
};

// ═══════════════════════════════════════════════════════════════════════════════
// COMPONENTS
// ═══════════════════════════════════════════════════════════════════════════════

const StepProgressBar: React.FC<{ current: number; total: number; executions: StepExecution[] }> = ({
  current, total, executions,
}) => (
  <div className="flex items-center gap-1">
    {Array.from({ length: total }).map((_, i) => {
      const exec = executions[i];
      const status = exec?.status || 'pending';
      
      return (
        <div
          key={i}
          className={`flex-1 h-2 rounded-full transition-all ${
            status === 'completed' ? 'bg-emerald-500' :
            status === 'current' ? 'bg-blue-500 animate-pulse' :
            status === 'skipped' ? 'bg-zinc-600' :
            'bg-zinc-700'
          }`}
        />
      );
    })}
  </div>
);

const ToleranceIndicator: React.FC<{ tolerance: Tolerance; value?: number }> = ({ tolerance, value }) => {
  const isWithin = value !== undefined && value >= tolerance.min_value && value <= tolerance.max_value;
  const percentage = value !== undefined
    ? ((value - tolerance.min_value) / (tolerance.max_value - tolerance.min_value)) * 100
    : 50;
  
  return (
    <div className="bg-zinc-800 rounded-lg p-4">
      <div className="flex items-center justify-between text-sm text-zinc-400 mb-2">
        <span>{tolerance.min_value} {tolerance.unit}</span>
        <span className="font-medium text-zinc-300">{tolerance.nominal} {tolerance.unit}</span>
        <span>{tolerance.max_value} {tolerance.unit}</span>
      </div>
      <div className="relative h-3 bg-zinc-700 rounded-full overflow-hidden">
        <div className="absolute inset-y-0 left-[25%] right-[25%] bg-emerald-500/30" />
        {value !== undefined && (
          <motion.div
            initial={{ left: '50%' }}
            animate={{ left: `${Math.min(100, Math.max(0, percentage))}%` }}
            className={`absolute top-0 bottom-0 w-1 -translate-x-1/2 ${
              isWithin ? 'bg-emerald-500' : 'bg-red-500'
            }`}
          />
        )}
      </div>
      {value !== undefined && (
        <div className={`text-center mt-2 text-sm font-medium ${isWithin ? 'text-emerald-400' : 'text-red-400'}`}>
          {value} {tolerance.unit} - {isWithin ? '✓ Dentro da tolerância' : '✗ Fora da tolerância'}
        </div>
      )}
    </div>
  );
};

const StepCard: React.FC<{
  step: InstructionStep;
  execution?: StepExecution;
  isActive: boolean;
  onComplete: (value: any) => void;
  isCompleting: boolean;
  error?: string;
}> = ({ step, execution, isActive, onComplete, isCompleting, error }) => {
  const [inputValue, setInputValue] = useState<any>('');
  const [showError, setShowError] = useState(false);

  useEffect(() => {
    if (error) {
      setShowError(true);
      setTimeout(() => setShowError(false), 5000);
    }
  }, [error]);

  const handleComplete = () => {
    let value = inputValue;
    
    if (step.input_type === 'boolean') {
      value = true;
    } else if (step.input_type === 'numeric') {
      value = parseFloat(inputValue);
      if (isNaN(value)) {
        setShowError(true);
        return;
      }
    }
    
    onComplete(value);
  };

  const getStepIcon = () => {
    switch (step.step_type) {
      case 'measurement': return <Target className="w-6 h-6" />;
      case 'checklist': return <ClipboardCheck className="w-6 h-6" />;
      case 'photo': return <Camera className="w-6 h-6" />;
      case 'confirmation': return <Check className="w-6 h-6" />;
      default: return <FileText className="w-6 h-6" />;
    }
  };

  const isCompleted = execution?.status === 'completed';

  return (
    <motion.div
      layout
      className={`rounded-2xl border-2 transition-all ${
        isActive
          ? 'bg-zinc-800/80 border-blue-500 shadow-lg shadow-blue-500/20'
          : isCompleted
          ? 'bg-zinc-800/50 border-emerald-500/50'
          : 'bg-zinc-900/50 border-zinc-700/50 opacity-50'
      }`}
    >
      {/* Header */}
      <div className={`p-4 border-b ${isActive ? 'border-blue-500/30' : 'border-zinc-700/50'}`}>
        <div className="flex items-center gap-4">
          <div className={`p-3 rounded-xl ${
            isCompleted ? 'bg-emerald-500/20 text-emerald-400' :
            isActive ? 'bg-blue-500/20 text-blue-400' :
            'bg-zinc-700/50 text-zinc-500'
          }`}>
            {isCompleted ? <CheckCircle className="w-6 h-6" /> : getStepIcon()}
          </div>
          <div className="flex-1">
            <div className="flex items-center gap-2">
              <span className="text-sm font-medium text-zinc-500">Passo {step.sequence}</span>
              {step.is_critical && (
                <span className="px-2 py-0.5 bg-red-500/20 text-red-400 text-xs font-medium rounded">
                  CRÍTICO
                </span>
              )}
              {step.is_quality_check && (
                <span className="px-2 py-0.5 bg-amber-500/20 text-amber-400 text-xs font-medium rounded">
                  QUALIDADE
                </span>
              )}
            </div>
            <h3 className="text-lg font-semibold text-white">{step.title}</h3>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="p-4 space-y-4">
        <p className="text-zinc-300">{step.description}</p>

        {/* Visual References */}
        {step.visual_references.length > 0 && (
          <div className="flex gap-2 overflow-x-auto pb-2">
            {step.visual_references.map((ref, idx) => (
              <div
                key={idx}
                className="flex-shrink-0 w-32 h-24 bg-zinc-700 rounded-lg overflow-hidden flex items-center justify-center"
              >
                <Image className="w-8 h-8 text-zinc-500" />
              </div>
            ))}
          </div>
        )}

        {/* Tolerance */}
        {step.tolerance && (
          <ToleranceIndicator
            tolerance={step.tolerance}
            value={execution?.input_value !== undefined ? parseFloat(execution.input_value) : undefined}
          />
        )}

        {/* Input Area - Only for active step */}
        {isActive && !isCompleted && (
          <div className="space-y-3">
            {step.input_type === 'numeric' && (
              <div>
                <label className="block text-sm text-zinc-400 mb-1">Valor medido:</label>
                <input
                  type="number"
                  step="0.1"
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  className="w-full px-4 py-3 bg-zinc-900 border border-zinc-600 rounded-lg text-white text-lg focus:border-blue-500 focus:outline-none"
                  placeholder={step.tolerance ? `${step.tolerance.nominal} ${step.tolerance.unit}` : 'Inserir valor'}
                  autoFocus
                />
              </div>
            )}

            {step.input_type === 'text' && (
              <div>
                <label className="block text-sm text-zinc-400 mb-1">Inserir:</label>
                <input
                  type="text"
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  className="w-full px-4 py-3 bg-zinc-900 border border-zinc-600 rounded-lg text-white focus:border-blue-500 focus:outline-none"
                  placeholder="Inserir texto"
                  autoFocus
                />
              </div>
            )}

            {step.input_type === 'select' && (
              <div>
                <label className="block text-sm text-zinc-400 mb-1">Selecionar:</label>
                <select
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  className="w-full px-4 py-3 bg-zinc-900 border border-zinc-600 rounded-lg text-white focus:border-blue-500 focus:outline-none"
                >
                  <option value="">Escolher...</option>
                  {step.options.map((opt) => (
                    <option key={opt} value={opt}>{opt}</option>
                  ))}
                </select>
              </div>
            )}

            {step.input_type === 'photo' && (
              <button className="w-full py-4 bg-zinc-700 hover:bg-zinc-600 rounded-lg flex items-center justify-center gap-2 text-zinc-300">
                <Camera className="w-5 h-5" />
                Tirar Foto
              </button>
            )}

            {/* Error Message */}
            <AnimatePresence>
              {(showError || error) && (
                <motion.div
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                  className="p-3 bg-red-500/20 border border-red-500/50 rounded-lg flex items-center gap-2 text-red-400"
                >
                  <AlertCircle className="w-5 h-5" />
                  {error || 'Valor inválido'}
                </motion.div>
              )}
            </AnimatePresence>

            {/* Complete Button */}
            <button
              onClick={handleComplete}
              disabled={isCompleting || (step.required && step.input_type !== 'boolean' && step.input_type !== 'none' && !inputValue)}
              className="w-full py-4 bg-emerald-500 hover:bg-emerald-600 disabled:bg-zinc-700 disabled:text-zinc-500 rounded-lg font-semibold text-white flex items-center justify-center gap-2 transition-all"
            >
              {isCompleting ? (
                <Loader2 className="w-5 h-5 animate-spin" />
              ) : (
                <>
                  <CheckCircle className="w-5 h-5" />
                  {step.input_type === 'boolean' ? 'Confirmar' : 'Concluir Passo'}
                </>
              )}
            </button>
          </div>
        )}

        {/* Completed State */}
        {isCompleted && execution && (
          <div className="p-3 bg-emerald-500/10 border border-emerald-500/30 rounded-lg">
            <div className="flex items-center gap-2 text-emerald-400">
              <CheckCircle className="w-5 h-5" />
              <span className="font-medium">Concluído</span>
              {execution.input_value !== undefined && (
                <span className="ml-auto text-zinc-400">
                  Valor: <span className="text-white font-mono">{String(execution.input_value)}</span>
                </span>
              )}
            </div>
          </div>
        )}
      </div>
    </motion.div>
  );
};

const QualityCheckPanel: React.FC<{
  checks: QualityCheck[];
  executionId: string;
  onCheckRecorded: () => void;
}> = ({ checks, executionId, onCheckRecorded }) => {
  const [completedChecks, setCompletedChecks] = useState<Record<string, string>>({});
  const [recording, setRecording] = useState<string | null>(null);

  const handleCheck = async (checkId: string, result: string) => {
    setRecording(checkId);
    try {
      await recordQualityCheck(executionId, checkId, result);
      setCompletedChecks((prev) => ({ ...prev, [checkId]: result }));
      onCheckRecorded();
    } catch (e) {
      console.error(e);
    }
    setRecording(null);
  };

  return (
    <div className="bg-zinc-800/50 rounded-xl border border-zinc-700/50 p-4">
      <h3 className="font-semibold text-white mb-4 flex items-center gap-2">
        <ClipboardCheck className="w-5 h-5 text-amber-500" />
        Checklist de Qualidade
      </h3>
      <div className="space-y-3">
        {checks.map((check) => {
          const result = completedChecks[check.check_id];
          const isRecording = recording === check.check_id;
          
          return (
            <div
              key={check.check_id}
              className={`p-3 rounded-lg border ${
                result === 'ok' ? 'bg-emerald-500/10 border-emerald-500/30' :
                result === 'nok' ? 'bg-red-500/10 border-red-500/30' :
                'bg-zinc-900/50 border-zinc-700/50'
              }`}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  {check.is_critical && (
                    <AlertTriangle className="w-4 h-4 text-amber-500" />
                  )}
                  <span className="text-zinc-300">{check.question}</span>
                </div>
                {result ? (
                  <span className={`px-3 py-1 rounded text-sm font-medium ${
                    result === 'ok' ? 'bg-emerald-500/20 text-emerald-400' : 'bg-red-500/20 text-red-400'
                  }`}>
                    {result.toUpperCase()}
                  </span>
                ) : (
                  <div className="flex gap-2">
                    <button
                      onClick={() => handleCheck(check.check_id, 'ok')}
                      disabled={isRecording}
                      className="p-2 bg-emerald-500/20 hover:bg-emerald-500/30 rounded-lg text-emerald-400"
                    >
                      <ThumbsUp className="w-4 h-4" />
                    </button>
                    <button
                      onClick={() => handleCheck(check.check_id, 'nok')}
                      disabled={isRecording}
                      className="p-2 bg-red-500/20 hover:bg-red-500/30 rounded-lg text-red-400"
                    >
                      <ThumbsDown className="w-4 h-4" />
                    </button>
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN PAGE
// ═══════════════════════════════════════════════════════════════════════════════

const WorkInstructions: React.FC = () => {
  const [execution, setExecution] = useState<Execution | null>(null);
  const [stepError, setStepError] = useState<string>('');
  const queryClient = useQueryClient();

  const { data: status } = useQuery({ queryKey: ['wi-status'], queryFn: fetchStatus });

  const createDemoMutation = useMutation({
    mutationFn: async () => {
      const instruction = await createDemoInstruction();
      return startExecution(instruction.instruction_id);
    },
    onSuccess: (data) => setExecution(data),
  });

  const completeStepMutation = useMutation({
    mutationFn: async ({ stepId, value }: { stepId: string; value: any }) => {
      if (!execution) throw new Error('No execution');
      return completeStep(execution.execution_id, stepId, value);
    },
    onSuccess: (data) => {
      setExecution(data);
      setStepError('');
    },
    onError: (err: Error) => setStepError(err.message),
  });

  const handleRefresh = async () => {
    if (execution) {
      const updated = await getExecution(execution.execution_id);
      setExecution(updated);
    }
  };

  const instruction = execution?.instruction;
  const currentStep = execution?.current_step;

  return (
    <div className="min-h-screen bg-gradient-to-br from-zinc-950 via-zinc-900 to-zinc-950">
      {/* Header */}
      <div className="sticky top-0 z-10 bg-zinc-900/90 backdrop-blur border-b border-zinc-800 px-6 py-4">
        <div className="max-w-4xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="p-2 bg-gradient-to-br from-blue-500/20 to-cyan-500/20 rounded-xl">
              <Clipboard className="w-6 h-6 text-blue-500" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-white">Work Instructions</h1>
              <p className="text-sm text-zinc-400">
                {instruction ? instruction.title : 'Instruções de Trabalho Digitais'}
              </p>
            </div>
          </div>
          {execution && (
            <div className="flex items-center gap-4">
              <div className="text-right">
                <p className="text-sm text-zinc-500">Progresso</p>
                <p className="text-xl font-bold text-white">{execution.progress_percent}%</p>
              </div>
              <div className={`px-3 py-1 rounded-full text-sm font-medium ${
                execution.status === 'completed' ? 'bg-emerald-500/20 text-emerald-400' :
                execution.status === 'paused' ? 'bg-amber-500/20 text-amber-400' :
                'bg-blue-500/20 text-blue-400'
              }`}>
                {execution.status === 'completed' ? 'Concluído' :
                 execution.status === 'paused' ? 'Pausado' :
                 'Em Progresso'}
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="max-w-4xl mx-auto p-6">
        {/* No Execution - Start Screen */}
        {!execution && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-center py-16"
          >
            <div className="w-20 h-20 mx-auto mb-6 bg-gradient-to-br from-blue-500/20 to-cyan-500/20 rounded-2xl flex items-center justify-center">
              <Clipboard className="w-10 h-10 text-blue-500" />
            </div>
            <h2 className="text-2xl font-bold text-white mb-2">Instruções de Trabalho</h2>
            <p className="text-zinc-400 mb-8 max-w-md mx-auto">
              Sistema de instruções passo-a-passo com validações poka-yoke e captura de evidências.
            </p>
            <button
              onClick={() => createDemoMutation.mutate()}
              disabled={createDemoMutation.isPending}
              className="px-6 py-3 bg-blue-500 hover:bg-blue-600 rounded-lg font-semibold text-white flex items-center gap-2 mx-auto"
            >
              {createDemoMutation.isPending ? (
                <Loader2 className="w-5 h-5 animate-spin" />
              ) : (
                <Play className="w-5 h-5" />
              )}
              Iniciar Demo
            </button>
          </motion.div>
        )}

        {/* Execution View */}
        {execution && instruction && (
          <div className="space-y-6">
            {/* Progress Bar */}
            <div className="bg-zinc-800/50 rounded-xl p-4 border border-zinc-700/50">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-zinc-400">
                  Passo {execution.completed_steps + (execution.status === 'completed' ? 0 : 1)} de {execution.total_steps}
                </span>
                <span className="text-sm text-zinc-400">
                  {execution.nok_count > 0 && (
                    <span className="text-red-400">{execution.nok_count} NOK</span>
                  )}
                </span>
              </div>
              <StepProgressBar
                current={execution.current_step_index}
                total={execution.total_steps}
                executions={execution.step_executions}
              />
            </div>

            {/* Paused Warning */}
            {execution.status === 'paused' && (
              <motion.div
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                className="p-4 bg-amber-500/20 border border-amber-500/50 rounded-xl flex items-center gap-3"
              >
                <Pause className="w-6 h-6 text-amber-500" />
                <div>
                  <p className="font-semibold text-amber-400">Execução Pausada</p>
                  <p className="text-sm text-amber-300/70">Quality check NOK detectado. Verificar com supervisor.</p>
                </div>
              </motion.div>
            )}

            {/* Completed Message */}
            {execution.status === 'completed' && (
              <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                className="p-6 bg-emerald-500/20 border border-emerald-500/50 rounded-xl text-center"
              >
                <CheckCircle className="w-12 h-12 text-emerald-500 mx-auto mb-3" />
                <h3 className="text-xl font-bold text-emerald-400 mb-1">Instrução Concluída!</h3>
                <p className="text-emerald-300/70">Todos os passos foram executados com sucesso.</p>
              </motion.div>
            )}

            {/* Current Step - Main Focus */}
            {currentStep && execution.status === 'in_progress' && (
              <StepCard
                step={currentStep}
                execution={execution.step_executions.find((s) => s.step_id === currentStep.step_id)}
                isActive={true}
                onComplete={(value) => completeStepMutation.mutate({ stepId: currentStep.step_id, value })}
                isCompleting={completeStepMutation.isPending}
                error={stepError}
              />
            )}

            {/* Completed Steps (collapsed) */}
            {execution.completed_steps > 0 && (
              <div className="space-y-2">
                <h4 className="text-sm font-medium text-zinc-500 flex items-center gap-2">
                  <Check className="w-4 h-4" />
                  Passos Concluídos ({execution.completed_steps})
                </h4>
                <div className="space-y-2">
                  {instruction.steps
                    .filter((_, idx) => execution.step_executions[idx]?.status === 'completed')
                    .map((step, idx) => (
                      <div
                        key={step.step_id}
                        className="p-3 bg-zinc-800/30 border border-zinc-700/30 rounded-lg flex items-center gap-3"
                      >
                        <CheckCircle className="w-5 h-5 text-emerald-500" />
                        <span className="text-zinc-400">{step.sequence}. {step.title}</span>
                        {execution.step_executions.find((s) => s.step_id === step.step_id)?.input_value !== undefined && (
                          <span className="ml-auto font-mono text-sm text-zinc-500">
                            {String(execution.step_executions.find((s) => s.step_id === step.step_id)?.input_value)}
                          </span>
                        )}
                      </div>
                    ))}
                </div>
              </div>
            )}

            {/* Quality Checks */}
            {instruction.quality_checks.length > 0 && execution.status !== 'not_started' && (
              <QualityCheckPanel
                checks={instruction.quality_checks}
                executionId={execution.execution_id}
                onCheckRecorded={handleRefresh}
              />
            )}
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="fixed bottom-0 left-0 right-0 bg-zinc-900/90 backdrop-blur border-t border-zinc-800 px-6 py-3">
        <div className="max-w-4xl mx-auto flex items-center justify-between text-xs text-zinc-500">
          <span>Work Instructions v1.0 • Poka-Yoke Digital</span>
          <span>R&D / SIFIDE: WP1 - Digital Twin & Shopfloor</span>
        </div>
      </div>
    </div>
  );
};

export default WorkInstructions;



