/**
 * Work Instructions Admin Panel - Authoring & Management
 * 
 * Features:
 * - List all work instructions
 * - Create new instructions
 * - Edit existing instructions
 * - Release/Archive workflow
 * - 3D model association
 */

import React, { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { motion, AnimatePresence } from 'framer-motion';
import {
  AlertTriangle,
  Archive,
  Check,
  CheckCircle,
  ChevronRight,
  Clock,
  Edit3,
  Eye,
  FileText,
  Image,
  Layers,
  Loader2,
  Play,
  Plus,
  RefreshCw,
  Save,
  Settings,
  Trash2,
  Upload,
  X,
} from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

// ═══════════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════════

interface WorkInstruction {
  instruction_id: string;
  title: string;
  description: string;
  version: number;
  status: 'draft' | 'released' | 'archived';
  steps_count: number;
  quality_checks_count: number;
  estimated_time_minutes: number;
  created_at: string;
  updated_at: string;
  model_3d_url?: string;
}

interface InstructionStep {
  step_id: string;
  sequence: number;
  title: string;
  description: string;
  step_type: string;
  input_type: string;
  is_critical: boolean;
  is_quality_check: boolean;
}

// ═══════════════════════════════════════════════════════════════════════════════
// API
// ═══════════════════════════════════════════════════════════════════════════════

async function fetchInstructions(): Promise<WorkInstruction[]> {
  const res = await fetch(`${API_BASE}/work-instructions`);
  if (!res.ok) return [];
  return res.json();
}

async function fetchInstruction(id: string): Promise<any> {
  const res = await fetch(`${API_BASE}/work-instructions/${id}`);
  if (!res.ok) throw new Error('Failed to fetch instruction');
  return res.json();
}

async function createInstruction(data: any): Promise<any> {
  const res = await fetch(`${API_BASE}/work-instructions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  if (!res.ok) throw new Error('Failed to create instruction');
  return res.json();
}

async function releaseInstruction(id: string): Promise<any> {
  const res = await fetch(`${API_BASE}/work-instructions/${id}/release`, {
    method: 'POST',
  });
  if (!res.ok) throw new Error('Failed to release instruction');
  return res.json();
}

async function createDemo(): Promise<any> {
  const res = await fetch(`${API_BASE}/work-instructions/demo`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ product_name: 'Demo Product', num_steps: 5 }),
  });
  if (!res.ok) throw new Error('Failed to create demo');
  return res.json();
}

// ═══════════════════════════════════════════════════════════════════════════════
// COMPONENTS
// ═══════════════════════════════════════════════════════════════════════════════

const StatusBadge: React.FC<{ status: string }> = ({ status }) => {
  const config: Record<string, { bg: string; text: string }> = {
    draft: { bg: 'bg-amber-500/20', text: 'text-amber-400' },
    released: { bg: 'bg-emerald-500/20', text: 'text-emerald-400' },
    archived: { bg: 'bg-slate-500/20', text: 'text-slate-400' },
  };
  const c = config[status] || config.draft;
  
  return (
    <span className={`px-2 py-0.5 rounded text-xs font-medium ${c.bg} ${c.text}`}>
      {status.toUpperCase()}
    </span>
  );
};

const InstructionCard: React.FC<{
  instruction: WorkInstruction;
  onSelect: () => void;
  onRelease: () => void;
  isSelected: boolean;
}> = ({ instruction, onSelect, onRelease, isSelected }) => (
  <motion.div
    initial={{ opacity: 0, y: 10 }}
    animate={{ opacity: 1, y: 0 }}
    onClick={onSelect}
    className={`cursor-pointer p-4 rounded-xl border transition-all ${
      isSelected
        ? 'bg-cyan-500/10 border-cyan-500/50'
        : 'bg-slate-800/50 border-slate-700/50 hover:border-slate-600'
    }`}
  >
    <div className="flex items-start justify-between mb-3">
      <div className="flex items-center gap-2">
        <FileText className="w-5 h-5 text-cyan-400" />
        <span className="font-medium text-white">{instruction.title}</span>
      </div>
      <StatusBadge status={instruction.status} />
    </div>
    
    <p className="text-sm text-slate-400 mb-3 line-clamp-2">{instruction.description}</p>
    
    <div className="flex items-center justify-between text-xs text-slate-500">
      <div className="flex items-center gap-3">
        <span>{instruction.steps_count} passos</span>
        <span>{instruction.quality_checks_count} checks</span>
        <span>{instruction.estimated_time_minutes} min</span>
      </div>
      <span>v{instruction.version}</span>
    </div>
    
    {instruction.status === 'draft' && (
      <div className="mt-3 pt-3 border-t border-slate-700/50 flex gap-2">
        <button
          onClick={(e) => { e.stopPropagation(); onRelease(); }}
          className="flex-1 flex items-center justify-center gap-1 px-3 py-1.5 bg-emerald-500/20 hover:bg-emerald-500/30 text-emerald-400 rounded-lg text-sm"
        >
          <Check className="w-4 h-4" />
          Publicar
        </button>
      </div>
    )}
  </motion.div>
);

const StepEditor: React.FC<{
  step: InstructionStep;
  onChange: (step: InstructionStep) => void;
  onDelete: () => void;
}> = ({ step, onChange, onDelete }) => (
  <div className="p-4 rounded-lg border border-slate-700/50 bg-slate-800/30">
    <div className="flex items-center justify-between mb-3">
      <div className="flex items-center gap-2">
        <span className="w-6 h-6 flex items-center justify-center bg-cyan-500/20 text-cyan-400 text-sm font-medium rounded">
          {step.sequence}
        </span>
        <input
          type="text"
          value={step.title}
          onChange={(e) => onChange({ ...step, title: e.target.value })}
          className="bg-transparent border-none text-white font-medium focus:outline-none"
          placeholder="Título do passo"
        />
      </div>
      <button onClick={onDelete} className="p-1 text-red-400 hover:bg-red-500/20 rounded">
        <Trash2 className="w-4 h-4" />
      </button>
    </div>
    
    <textarea
      value={step.description}
      onChange={(e) => onChange({ ...step, description: e.target.value })}
      className="w-full bg-slate-900/50 border border-slate-700 rounded-lg p-2 text-sm text-slate-300 resize-none"
      rows={2}
      placeholder="Descrição do passo..."
    />
    
    <div className="flex items-center gap-4 mt-3">
      <select
        value={step.step_type}
        onChange={(e) => onChange({ ...step, step_type: e.target.value })}
        className="bg-slate-900 border border-slate-700 rounded-lg px-2 py-1 text-sm text-white"
      >
        <option value="instruction">Instrução</option>
        <option value="measurement">Medição</option>
        <option value="checklist">Checklist</option>
        <option value="photo">Foto</option>
        <option value="confirmation">Confirmação</option>
      </select>
      
      <select
        value={step.input_type}
        onChange={(e) => onChange({ ...step, input_type: e.target.value })}
        className="bg-slate-900 border border-slate-700 rounded-lg px-2 py-1 text-sm text-white"
      >
        <option value="none">Sem input</option>
        <option value="boolean">Confirmação</option>
        <option value="numeric">Numérico</option>
        <option value="text">Texto</option>
        <option value="select">Seleção</option>
      </select>
      
      <label className="flex items-center gap-2 text-sm text-slate-400">
        <input
          type="checkbox"
          checked={step.is_critical}
          onChange={(e) => onChange({ ...step, is_critical: e.target.checked })}
          className="accent-red-500"
        />
        Crítico
      </label>
      
      <label className="flex items-center gap-2 text-sm text-slate-400">
        <input
          type="checkbox"
          checked={step.is_quality_check}
          onChange={(e) => onChange({ ...step, is_quality_check: e.target.checked })}
          className="accent-amber-500"
        />
        Quality Check
      </label>
    </div>
  </div>
);

const CreateInstructionModal: React.FC<{
  isOpen: boolean;
  onClose: () => void;
  onCreate: (data: any) => void;
}> = ({ isOpen, onClose, onCreate }) => {
  const [title, setTitle] = useState('');
  const [description, setDescription] = useState('');
  const [steps, setSteps] = useState<InstructionStep[]>([]);

  const addStep = () => {
    setSteps([...steps, {
      step_id: `step-${Date.now()}`,
      sequence: steps.length + 1,
      title: '',
      description: '',
      step_type: 'instruction',
      input_type: 'none',
      is_critical: false,
      is_quality_check: false,
    }]);
  };

  const handleCreate = () => {
    onCreate({
      title,
      description,
      steps: steps.map((s, i) => ({ ...s, sequence: i + 1 })),
    });
    onClose();
  };

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
        className="bg-slate-900 rounded-2xl p-6 max-w-3xl w-full max-h-[90vh] overflow-y-auto"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-xl font-bold text-white flex items-center gap-2">
            <Plus className="w-5 h-5 text-cyan-400" />
            Nova Instrução de Trabalho
          </h2>
          <button onClick={onClose} className="p-2 text-slate-400 hover:bg-slate-800 rounded-lg">
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="space-y-4">
          <div>
            <label className="block text-sm text-slate-400 mb-1">Título</label>
            <input
              type="text"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white focus:border-cyan-500 focus:outline-none"
              placeholder="Nome da instrução..."
            />
          </div>

          <div>
            <label className="block text-sm text-slate-400 mb-1">Descrição</label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white focus:border-cyan-500 focus:outline-none resize-none"
              rows={3}
              placeholder="Descrição da instrução..."
            />
          </div>

          <div>
            <div className="flex items-center justify-between mb-3">
              <label className="text-sm text-slate-400">Passos ({steps.length})</label>
              <button
                onClick={addStep}
                className="flex items-center gap-1 px-3 py-1 bg-cyan-500/20 hover:bg-cyan-500/30 text-cyan-400 rounded-lg text-sm"
              >
                <Plus className="w-4 h-4" />
                Adicionar Passo
              </button>
            </div>
            
            <div className="space-y-3">
              {steps.map((step, idx) => (
                <StepEditor
                  key={step.step_id}
                  step={step}
                  onChange={(updated) => {
                    const newSteps = [...steps];
                    newSteps[idx] = updated;
                    setSteps(newSteps);
                  }}
                  onDelete={() => setSteps(steps.filter((_, i) => i !== idx))}
                />
              ))}
            </div>
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
            onClick={handleCreate}
            disabled={!title || steps.length === 0}
            className="flex-1 py-2 bg-cyan-600 hover:bg-cyan-500 text-white rounded-lg disabled:opacity-50 flex items-center justify-center gap-2"
          >
            <Save className="w-4 h-4" />
            Criar Instrução
          </button>
        </div>
      </motion.div>
    </motion.div>
  );
};

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN COMPONENT
// ═══════════════════════════════════════════════════════════════════════════════

export const WorkInstructionsAdmin: React.FC = () => {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [filter, setFilter] = useState<'all' | 'draft' | 'released'>('all');
  
  const queryClient = useQueryClient();

  const { data: instructions, isLoading, refetch } = useQuery({
    queryKey: ['work-instructions'],
    queryFn: fetchInstructions,
  });

  const { data: selectedInstruction } = useQuery({
    queryKey: ['work-instruction', selectedId],
    queryFn: () => selectedId ? fetchInstruction(selectedId) : null,
    enabled: !!selectedId,
  });

  const createMutation = useMutation({
    mutationFn: createInstruction,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['work-instructions'] });
    },
  });

  const releaseMutation = useMutation({
    mutationFn: releaseInstruction,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['work-instructions'] });
    },
  });

  const demoMutation = useMutation({
    mutationFn: createDemo,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['work-instructions'] });
    },
  });

  const filteredInstructions = instructions?.filter((i) => {
    if (filter === 'all') return true;
    return i.status === filter;
  }) || [];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold text-white flex items-center gap-2">
            <Settings className="w-5 h-5 text-cyan-500" />
            Gestão de Instruções de Trabalho
          </h2>
          <p className="text-sm text-slate-400 mt-1">
            Criar, editar e publicar instruções para o chão-de-fábrica
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => demoMutation.mutate()}
            disabled={demoMutation.isPending}
            className="flex items-center gap-2 px-3 py-2 bg-slate-700 hover:bg-slate-600 text-white rounded-lg text-sm"
          >
            {demoMutation.isPending ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
            Criar Demo
          </button>
          <button
            onClick={() => setShowCreateModal(true)}
            className="flex items-center gap-2 px-4 py-2 bg-cyan-600 hover:bg-cyan-500 text-white rounded-lg font-medium"
          >
            <Plus className="w-4 h-4" />
            Nova Instrução
          </button>
        </div>
      </div>

      {/* Filters */}
      <div className="flex items-center gap-2">
        {['all', 'draft', 'released'].map((f) => (
          <button
            key={f}
            onClick={() => setFilter(f as any)}
            className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
              filter === f
                ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/30'
                : 'bg-slate-800/50 text-slate-400 hover:bg-slate-700/50'
            }`}
          >
            {f === 'all' ? 'Todas' : f === 'draft' ? 'Rascunho' : 'Publicadas'}
          </button>
        ))}
        <span className="text-sm text-slate-500 ml-2">
          {filteredInstructions.length} instrução(ões)
        </span>
      </div>

      {/* Content */}
      <div className="grid grid-cols-12 gap-6">
        {/* Instructions List */}
        <div className="col-span-5 space-y-3">
          {isLoading ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="w-8 h-8 text-cyan-400 animate-spin" />
            </div>
          ) : filteredInstructions.length === 0 ? (
            <div className="text-center py-12 rounded-xl border border-slate-700/50 bg-slate-800/30">
              <FileText className="w-12 h-12 text-slate-600 mx-auto mb-3" />
              <p className="text-slate-400">Nenhuma instrução encontrada</p>
              <button
                onClick={() => setShowCreateModal(true)}
                className="mt-4 px-4 py-2 bg-cyan-500/20 text-cyan-400 rounded-lg text-sm"
              >
                Criar primeira instrução
              </button>
            </div>
          ) : (
            <div className="space-y-3 max-h-[600px] overflow-y-auto">
              {filteredInstructions.map((instruction) => (
                <InstructionCard
                  key={instruction.instruction_id}
                  instruction={instruction}
                  isSelected={selectedId === instruction.instruction_id}
                  onSelect={() => setSelectedId(instruction.instruction_id)}
                  onRelease={() => releaseMutation.mutate(instruction.instruction_id)}
                />
              ))}
            </div>
          )}
        </div>

        {/* Detail Panel */}
        <div className="col-span-7">
          {selectedInstruction ? (
            <div className="rounded-xl border border-slate-700/50 bg-slate-800/30 p-6">
              <div className="flex items-center justify-between mb-6">
                <div>
                  <h3 className="text-xl font-bold text-white">{selectedInstruction.title}</h3>
                  <p className="text-sm text-slate-400 mt-1">{selectedInstruction.description}</p>
                </div>
                <StatusBadge status={selectedInstruction.status} />
              </div>

              <div className="grid grid-cols-3 gap-4 mb-6">
                <div className="bg-slate-900/50 rounded-lg p-3">
                  <p className="text-xs text-slate-500">Passos</p>
                  <p className="text-2xl font-bold text-white">{selectedInstruction.steps?.length || 0}</p>
                </div>
                <div className="bg-slate-900/50 rounded-lg p-3">
                  <p className="text-xs text-slate-500">Versão</p>
                  <p className="text-2xl font-bold text-white">{selectedInstruction.version}</p>
                </div>
                <div className="bg-slate-900/50 rounded-lg p-3">
                  <p className="text-xs text-slate-500">Tempo Est.</p>
                  <p className="text-2xl font-bold text-white">{selectedInstruction.estimated_time_minutes} min</p>
                </div>
              </div>

              {/* Steps Preview */}
              <div>
                <h4 className="text-sm font-medium text-slate-400 mb-3">Passos ({selectedInstruction.steps?.length || 0})</h4>
                <div className="space-y-2 max-h-80 overflow-y-auto">
                  {selectedInstruction.steps?.map((step: any) => (
                    <div key={step.step_id} className="flex items-center gap-3 p-3 bg-slate-900/50 rounded-lg">
                      <span className="w-6 h-6 flex items-center justify-center bg-cyan-500/20 text-cyan-400 text-sm font-medium rounded">
                        {step.sequence}
                      </span>
                      <div className="flex-1">
                        <p className="font-medium text-white">{step.title}</p>
                        <p className="text-sm text-slate-400 line-clamp-1">{step.description}</p>
                      </div>
                      <div className="flex items-center gap-2">
                        {step.is_critical && (
                          <span className="px-2 py-0.5 bg-red-500/20 text-red-400 text-xs rounded">Crítico</span>
                        )}
                        {step.is_quality_check && (
                          <span className="px-2 py-0.5 bg-amber-500/20 text-amber-400 text-xs rounded">QC</span>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* 3D Model */}
              {selectedInstruction.model_3d_url && (
                <div className="mt-6 p-4 bg-slate-900/50 rounded-lg border border-slate-700/50">
                  <div className="flex items-center gap-2 text-slate-400 mb-2">
                    <Layers className="w-4 h-4" />
                    <span className="text-sm">Modelo 3D Associado</span>
                  </div>
                  <p className="text-sm text-cyan-400 font-mono">{selectedInstruction.model_3d_url}</p>
                </div>
              )}
            </div>
          ) : (
            <div className="h-full flex items-center justify-center rounded-xl border border-slate-700/50 bg-slate-800/30 p-12">
              <div className="text-center">
                <ChevronRight className="w-16 h-16 text-slate-600 mx-auto mb-4" />
                <p className="text-lg font-medium text-slate-400">Selecione uma instrução</p>
                <p className="text-sm text-slate-500 mt-1">
                  Ou crie uma nova instrução de trabalho
                </p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Create Modal */}
      <AnimatePresence>
        {showCreateModal && (
          <CreateInstructionModal
            isOpen={showCreateModal}
            onClose={() => setShowCreateModal(false)}
            onCreate={(data) => createMutation.mutate(data)}
          />
        )}
      </AnimatePresence>
    </div>
  );
};

export default WorkInstructionsAdmin;


