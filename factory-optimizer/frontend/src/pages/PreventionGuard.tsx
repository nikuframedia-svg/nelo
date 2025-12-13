/**
 * ═══════════════════════════════════════════════════════════════════════════════
 * Prevention Guard Dashboard - Process & Quality Guard
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * Dashboard for error prevention system:
 * - PDM validation (BOM, Routing, Documentation)
 * - Shopfloor validation (Material, Equipment)
 * - Predictive risk assessment
 * - Exception management
 * - Event monitoring
 *
 * R&D / SIFIDE: WP4 - Zero Defect Manufacturing
 */

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  AlertTriangle,
  Check,
  CheckCircle,
  ChevronRight,
  Clock,
  FileCheck,
  FileWarning,
  Gauge,
  Loader2,
  Play,
  RefreshCw,
  Shield,
  ShieldAlert,
  ShieldCheck,
  ShieldOff,
  Target,
  ThumbsDown,
  ThumbsUp,
  XCircle,
  Zap,
} from 'lucide-react';

// ═══════════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════════

interface GuardStatus {
  statistics: {
    validations_performed: number;
    issues_detected: number;
    errors_prevented: number;
    exceptions_requested: number;
    exceptions_approved: number;
    pending_exceptions: number;
    active_rules: { pdm: number; shopfloor: number };
    predictive_model: { trained: boolean; training_samples: number };
  };
}

interface ValidationIssue {
  issue_id: string;
  rule_name: string;
  category: string;
  severity: string;
  action: string;
  message: string;
  resolved: boolean;
}

interface ValidationResult {
  passed: boolean;
  issues: ValidationIssue[];
  errors: number;
  warnings: number;
  blocked: boolean;
  requires_approval: boolean;
}

interface RiskPrediction {
  risk_level: string;
  defect_probability: number;
  risk_factors: Record<string, number>;
  recommendations: string[];
  model_version: string;
}

interface ExceptionRequest {
  exception_id: string;
  validation_issue_id: string;
  order_id: string;
  requested_by: string;
  reason: string;
  status: string;
  resolved_by?: string;
  resolution_note?: string;
  requested_at: string;
}

interface GuardEvent {
  event_id: string;
  event_type: string;
  entity_type: string;
  entity_id: string;
  message: string;
  timestamp: string;
}

interface ValidationRule {
  rule_id: string;
  name: string;
  description: string;
  category: string;
  severity: string;
  action: string;
  enabled: boolean;
}

// ═══════════════════════════════════════════════════════════════════════════════
// API
// ═══════════════════════════════════════════════════════════════════════════════

const API_BASE = 'http://127.0.0.1:8000/guard';

const fetchStatus = async (): Promise<GuardStatus> => {
  const res = await fetch(`${API_BASE}/status`);
  if (!res.ok) throw new Error('Failed');
  return res.json();
};

const fetchRules = async (): Promise<{ rules: ValidationRule[] }> => {
  const res = await fetch(`${API_BASE}/rules`);
  if (!res.ok) throw new Error('Failed');
  return res.json();
};

const fetchPendingExceptions = async (): Promise<{ exceptions: ExceptionRequest[] }> => {
  const res = await fetch(`${API_BASE}/exceptions/pending`);
  if (!res.ok) throw new Error('Failed');
  return res.json();
};

const fetchEvents = async (): Promise<{ events: GuardEvent[] }> => {
  const res = await fetch(`${API_BASE}/events?limit=50`);
  if (!res.ok) throw new Error('Failed');
  return res.json();
};

const runDemo = async (): Promise<any> => {
  const res = await fetch(`${API_BASE}/demo`, { method: 'POST' });
  if (!res.ok) throw new Error('Failed');
  return res.json();
};

const approveException = async ({ id, by }: { id: string; by: string }): Promise<any> => {
  const res = await fetch(`${API_BASE}/exceptions/${id}/approve`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ resolved_by: by, note: 'Approved via dashboard' }),
  });
  if (!res.ok) throw new Error('Failed');
  return res.json();
};

const rejectException = async ({ id, by }: { id: string; by: string }): Promise<any> => {
  const res = await fetch(`${API_BASE}/exceptions/${id}/reject`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ resolved_by: by, note: 'Rejected via dashboard' }),
  });
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

const SeverityBadge: React.FC<{ severity: string }> = ({ severity }) => {
  const colors: Record<string, string> = {
    info: 'bg-blue-500/20 text-blue-400 border-blue-500/30',
    warning: 'bg-amber-500/20 text-amber-400 border-amber-500/30',
    error: 'bg-red-500/20 text-red-400 border-red-500/30',
    critical: 'bg-red-600/30 text-red-300 border-red-500/50',
  };
  return (
    <span className={`px-2 py-0.5 rounded-full text-xs font-medium border ${colors[severity] || colors.info}`}>
      {severity.toUpperCase()}
    </span>
  );
};

const ActionBadge: React.FC<{ action: string }> = ({ action }) => {
  const colors: Record<string, string> = {
    allow: 'bg-green-500/20 text-green-400',
    warn: 'bg-amber-500/20 text-amber-400',
    block: 'bg-red-500/20 text-red-400',
    approval_required: 'bg-purple-500/20 text-purple-400',
  };
  return (
    <span className={`px-2 py-0.5 rounded text-xs font-medium ${colors[action] || colors.allow}`}>
      {action.replace('_', ' ')}
    </span>
  );
};

const RiskGauge: React.FC<{ level: string; probability: number }> = ({ level, probability }) => {
  const colors: Record<string, string> = {
    low: '#22c55e',
    medium: '#f59e0b',
    high: '#ef4444',
    critical: '#dc2626',
  };
  const color = colors[level] || colors.low;
  const pct = Math.round(probability * 100);
  
  return (
    <div className="flex items-center gap-4">
      <div className="relative w-24 h-24">
        <svg className="w-full h-full transform -rotate-90">
          <circle
            cx="48"
            cy="48"
            r="40"
            stroke="#333"
            strokeWidth="8"
            fill="none"
          />
          <circle
            cx="48"
            cy="48"
            r="40"
            stroke={color}
            strokeWidth="8"
            fill="none"
            strokeDasharray={`${pct * 2.51} 251`}
            strokeLinecap="round"
          />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="text-xl font-bold" style={{ color }}>{pct}%</span>
        </div>
      </div>
      <div>
        <p className="text-lg font-bold uppercase" style={{ color }}>{level}</p>
        <p className="text-sm text-zinc-400">Defect Risk</p>
      </div>
    </div>
  );
};

const ValidationIssueCard: React.FC<{ issue: ValidationIssue }> = ({ issue }) => (
  <div className={`p-3 rounded-lg border ${issue.resolved ? 'bg-zinc-900/30 border-zinc-700/30 opacity-60' : 'bg-zinc-800/50 border-zinc-700/50'}`}>
    <div className="flex items-start justify-between gap-2 mb-2">
      <span className="font-medium text-white text-sm">{issue.rule_name}</span>
      <div className="flex gap-2">
        <SeverityBadge severity={issue.severity} />
        <ActionBadge action={issue.action} />
      </div>
    </div>
    <p className="text-sm text-zinc-400">{issue.message}</p>
    {issue.resolved && (
      <p className="text-xs text-emerald-400 mt-1 flex items-center gap-1">
        <CheckCircle className="w-3 h-3" /> Resolved
      </p>
    )}
  </div>
);

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN PAGE
// ═══════════════════════════════════════════════════════════════════════════════

const PreventionGuard: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'overview' | 'rules' | 'exceptions' | 'events'>('overview');
  const [demoResult, setDemoResult] = useState<any>(null);
  
  const queryClient = useQueryClient();

  const { data: status, refetch: refetchStatus } = useQuery({ queryKey: ['guard-status'], queryFn: fetchStatus });
  const { data: rulesData } = useQuery({ queryKey: ['guard-rules'], queryFn: fetchRules });
  const { data: exceptionsData, refetch: refetchExceptions } = useQuery({ queryKey: ['guard-exceptions'], queryFn: fetchPendingExceptions });
  const { data: eventsData } = useQuery({ queryKey: ['guard-events'], queryFn: fetchEvents });

  const demoMutation = useMutation({
    mutationFn: runDemo,
    onSuccess: (data) => {
      setDemoResult(data);
      refetchStatus();
      queryClient.invalidateQueries({ queryKey: ['guard-events'] });
    },
  });

  const approveMutation = useMutation({
    mutationFn: approveException,
    onSuccess: () => {
      refetchExceptions();
      refetchStatus();
      queryClient.invalidateQueries({ queryKey: ['guard-events'] });
    },
  });

  const rejectMutation = useMutation({
    mutationFn: rejectException,
    onSuccess: () => {
      refetchExceptions();
      refetchStatus();
    },
  });

  const stats = status?.statistics;

  const tabs = [
    { id: 'overview' as const, label: 'Overview', icon: <Shield className="w-4 h-4" /> },
    { id: 'rules' as const, label: 'Rules', icon: <FileCheck className="w-4 h-4" /> },
    { id: 'exceptions' as const, label: 'Exceptions', icon: <ShieldAlert className="w-4 h-4" />, badge: stats?.pending_exceptions },
    { id: 'events' as const, label: 'Events', icon: <Clock className="w-4 h-4" /> },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-zinc-950 via-zinc-900 to-zinc-950 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-3xl font-bold text-white mb-2 flex items-center gap-3">
              <div className="p-2 bg-gradient-to-br from-emerald-500/20 to-teal-500/20 rounded-xl">
                <ShieldCheck className="w-8 h-8 text-emerald-500" />
              </div>
              Prevention Guard
            </h1>
            <p className="text-zinc-400">Sistema de prevenção de erros e validação de qualidade</p>
          </div>
          <button
            onClick={() => demoMutation.mutate()}
            disabled={demoMutation.isPending}
            className="flex items-center gap-2 px-4 py-2 bg-emerald-500/20 hover:bg-emerald-500/30 text-emerald-400 rounded-lg"
          >
            {demoMutation.isPending ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
            Run Demo
          </button>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-5 gap-4 mb-6">
          <StatCard
            label="Validações"
            value={stats?.validations_performed || 0}
            icon={<FileCheck className="w-5 h-5" />}
            color="bg-blue-500/20 text-blue-400"
          />
          <StatCard
            label="Issues Detectados"
            value={stats?.issues_detected || 0}
            icon={<AlertTriangle className="w-5 h-5" />}
            color="bg-amber-500/20 text-amber-400"
          />
          <StatCard
            label="Erros Prevenidos"
            value={stats?.errors_prevented || 0}
            icon={<ShieldCheck className="w-5 h-5" />}
            color="bg-emerald-500/20 text-emerald-400"
          />
          <StatCard
            label="Exceções Pendentes"
            value={stats?.pending_exceptions || 0}
            icon={<ShieldAlert className="w-5 h-5" />}
            color={stats?.pending_exceptions ? "bg-red-500/20 text-red-400" : "bg-zinc-500/20 text-zinc-400"}
          />
          <StatCard
            label="Regras Ativas"
            value={(stats?.active_rules.pdm || 0) + (stats?.active_rules.shopfloor || 0)}
            icon={<Target className="w-5 h-5" />}
            color="bg-purple-500/20 text-purple-400"
            subtext={`PDM: ${stats?.active_rules.pdm || 0} | Shopfloor: ${stats?.active_rules.shopfloor || 0}`}
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
                  ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                  : 'bg-zinc-800/50 text-zinc-400 hover:bg-zinc-700/50'
              }`}
            >
              {tab.icon}
              {tab.label}
              {tab.badge ? (
                <span className="ml-1 px-1.5 py-0.5 bg-red-500/30 text-red-400 text-xs rounded-full">
                  {tab.badge}
                </span>
              ) : null}
            </button>
          ))}
        </div>

        {/* Content */}
        <div className="bg-zinc-900/50 rounded-xl border border-zinc-800 p-6">
          {/* OVERVIEW TAB */}
          {activeTab === 'overview' && (
            <div className="space-y-6">
              <h2 className="text-xl font-bold text-white flex items-center gap-2">
                <Shield className="w-5 h-5 text-emerald-500" />
                Validation Overview
              </h2>

              {demoResult ? (
                <div className="grid grid-cols-2 gap-6">
                  {/* Product Release */}
                  <div className="bg-zinc-800/50 rounded-lg p-4">
                    <div className="flex items-center justify-between mb-4">
                      <h3 className="font-medium text-white">Product Release Validation</h3>
                      {demoResult.results.product_release.passed ? (
                        <span className="flex items-center gap-1 text-emerald-400 text-sm">
                          <CheckCircle className="w-4 h-4" /> Passed
                        </span>
                      ) : (
                        <span className="flex items-center gap-1 text-red-400 text-sm">
                          <XCircle className="w-4 h-4" /> Failed
                        </span>
                      )}
                    </div>
                    <div className="flex gap-4 mb-4">
                      <div className="text-center">
                        <p className="text-2xl font-bold text-red-400">{demoResult.results.product_release.errors}</p>
                        <p className="text-xs text-zinc-500">Errors</p>
                      </div>
                      <div className="text-center">
                        <p className="text-2xl font-bold text-amber-400">{demoResult.results.product_release.warnings}</p>
                        <p className="text-xs text-zinc-500">Warnings</p>
                      </div>
                    </div>
                    <div className="space-y-2 max-h-64 overflow-y-auto">
                      {demoResult.results.product_release.issues.map((issue: ValidationIssue) => (
                        <ValidationIssueCard key={issue.issue_id} issue={issue} />
                      ))}
                    </div>
                  </div>

                  {/* Order Start */}
                  <div className="bg-zinc-800/50 rounded-lg p-4">
                    <div className="flex items-center justify-between mb-4">
                      <h3 className="font-medium text-white">Order Start Validation</h3>
                      {demoResult.results.order_start.validation.passed ? (
                        <span className="flex items-center gap-1 text-emerald-400 text-sm">
                          <CheckCircle className="w-4 h-4" /> Passed
                        </span>
                      ) : (
                        <span className="flex items-center gap-1 text-red-400 text-sm">
                          <XCircle className="w-4 h-4" /> Failed
                        </span>
                      )}
                    </div>
                    
                    {/* Risk Prediction */}
                    <div className="mb-4 p-3 bg-zinc-900/50 rounded-lg">
                      <RiskGauge 
                        level={demoResult.results.order_start.risk.risk_level}
                        probability={demoResult.results.order_start.risk.defect_probability}
                      />
                      {demoResult.results.order_start.risk.recommendations?.length > 0 && (
                        <div className="mt-3 pt-3 border-t border-zinc-700">
                          <p className="text-xs text-zinc-500 mb-2">Recommendations:</p>
                          <ul className="space-y-1">
                            {demoResult.results.order_start.risk.recommendations.map((rec: string, i: number) => (
                              <li key={i} className="text-xs text-amber-400 flex items-start gap-1">
                                <ChevronRight className="w-3 h-3 mt-0.5 flex-shrink-0" />
                                {rec}
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>

                    <div className="space-y-2 max-h-48 overflow-y-auto">
                      {demoResult.results.order_start.validation.issues.map((issue: ValidationIssue) => (
                        <ValidationIssueCard key={issue.issue_id} issue={issue} />
                      ))}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="text-center py-12 text-zinc-500">
                  <Shield className="w-12 h-12 mx-auto mb-3 opacity-30" />
                  <p>Execute o demo para ver validações em ação</p>
                </div>
              )}
            </div>
          )}

          {/* RULES TAB */}
          {activeTab === 'rules' && (
            <div className="space-y-6">
              <h2 className="text-xl font-bold text-white flex items-center gap-2">
                <FileCheck className="w-5 h-5 text-blue-500" />
                Validation Rules ({rulesData?.rules.length || 0})
              </h2>

              <div className="grid grid-cols-2 gap-4">
                {rulesData?.rules.map((rule) => (
                  <div key={rule.rule_id} className={`p-4 rounded-lg border ${rule.enabled ? 'bg-zinc-800/50 border-zinc-700/50' : 'bg-zinc-900/30 border-zinc-800/30 opacity-50'}`}>
                    <div className="flex items-start justify-between mb-2">
                      <div>
                        <span className="font-mono text-xs text-zinc-500">{rule.rule_id}</span>
                        <p className="font-medium text-white">{rule.name}</p>
                      </div>
                      <div className="flex gap-2">
                        <SeverityBadge severity={rule.severity} />
                        <ActionBadge action={rule.action} />
                      </div>
                    </div>
                    <p className="text-sm text-zinc-400">{rule.description}</p>
                    <div className="mt-2 flex items-center gap-2">
                      <span className="px-2 py-0.5 bg-zinc-700/50 text-zinc-400 text-xs rounded">{rule.category}</span>
                      {rule.enabled ? (
                        <span className="text-xs text-emerald-400 flex items-center gap-1">
                          <Check className="w-3 h-3" /> Enabled
                        </span>
                      ) : (
                        <span className="text-xs text-zinc-500 flex items-center gap-1">
                          <ShieldOff className="w-3 h-3" /> Disabled
                        </span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* EXCEPTIONS TAB */}
          {activeTab === 'exceptions' && (
            <div className="space-y-6">
              <h2 className="text-xl font-bold text-white flex items-center gap-2">
                <ShieldAlert className="w-5 h-5 text-amber-500" />
                Pending Exceptions ({exceptionsData?.exceptions.length || 0})
              </h2>

              {exceptionsData && exceptionsData.exceptions.length > 0 ? (
                <div className="space-y-4">
                  {exceptionsData.exceptions.map((ex) => (
                    <div key={ex.exception_id} className="p-4 bg-zinc-800/50 rounded-lg border border-amber-500/20">
                      <div className="flex items-start justify-between mb-3">
                        <div>
                          <span className="font-mono text-xs text-zinc-500">{ex.exception_id}</span>
                          <p className="font-medium text-white">Order: {ex.order_id}</p>
                          <p className="text-sm text-zinc-400">Requested by: {ex.requested_by}</p>
                        </div>
                        <span className="px-2 py-0.5 bg-amber-500/20 text-amber-400 text-xs rounded">
                          {ex.status.toUpperCase()}
                        </span>
                      </div>
                      <p className="text-sm text-zinc-300 mb-3">{ex.reason}</p>
                      <div className="flex gap-2">
                        <button
                          onClick={() => approveMutation.mutate({ id: ex.exception_id, by: 'supervisor' })}
                          disabled={approveMutation.isPending}
                          className="flex items-center gap-1 px-3 py-1.5 bg-emerald-500/20 hover:bg-emerald-500/30 text-emerald-400 rounded text-sm"
                        >
                          <ThumbsUp className="w-4 h-4" /> Approve
                        </button>
                        <button
                          onClick={() => rejectMutation.mutate({ id: ex.exception_id, by: 'supervisor' })}
                          disabled={rejectMutation.isPending}
                          className="flex items-center gap-1 px-3 py-1.5 bg-red-500/20 hover:bg-red-500/30 text-red-400 rounded text-sm"
                        >
                          <ThumbsDown className="w-4 h-4" /> Reject
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-12 text-zinc-500">
                  <ShieldCheck className="w-12 h-12 mx-auto mb-3 opacity-30" />
                  <p>Nenhuma exceção pendente</p>
                </div>
              )}
            </div>
          )}

          {/* EVENTS TAB */}
          {activeTab === 'events' && (
            <div className="space-y-6">
              <div className="flex items-center justify-between">
                <h2 className="text-xl font-bold text-white flex items-center gap-2">
                  <Clock className="w-5 h-5 text-zinc-500" />
                  Event Log
                </h2>
                <button
                  onClick={() => queryClient.invalidateQueries({ queryKey: ['guard-events'] })}
                  className="flex items-center gap-1 px-3 py-1 bg-zinc-700/50 hover:bg-zinc-600/50 text-zinc-300 rounded text-sm"
                >
                  <RefreshCw className="w-3 h-3" /> Refresh
                </button>
              </div>

              {eventsData && eventsData.events.length > 0 ? (
                <div className="space-y-2">
                  {eventsData.events.map((event) => {
                    const typeColors: Record<string, string> = {
                      validation_passed: 'text-emerald-400',
                      validation_failed: 'text-red-400',
                      risk_alert: 'text-amber-400',
                      exception_requested: 'text-purple-400',
                      exception_resolved: 'text-blue-400',
                      error_prevented: 'text-emerald-400',
                    };
                    return (
                      <div key={event.event_id} className="flex items-start gap-3 p-3 bg-zinc-800/30 rounded-lg">
                        <div className={`mt-1 ${typeColors[event.event_type] || 'text-zinc-400'}`}>
                          {event.event_type === 'validation_passed' && <CheckCircle className="w-4 h-4" />}
                          {event.event_type === 'validation_failed' && <XCircle className="w-4 h-4" />}
                          {event.event_type === 'risk_alert' && <AlertTriangle className="w-4 h-4" />}
                          {event.event_type === 'exception_requested' && <FileWarning className="w-4 h-4" />}
                          {event.event_type === 'exception_resolved' && <Check className="w-4 h-4" />}
                          {event.event_type === 'error_prevented' && <ShieldCheck className="w-4 h-4" />}
                        </div>
                        <div className="flex-1">
                          <p className="text-sm text-white">{event.message}</p>
                          <div className="flex gap-2 mt-1 text-xs text-zinc-500">
                            <span>{event.entity_type}: {event.entity_id}</span>
                            <span>•</span>
                            <span>{new Date(event.timestamp).toLocaleString('pt-PT')}</span>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              ) : (
                <div className="text-center py-12 text-zinc-500">
                  <Clock className="w-12 h-12 mx-auto mb-3 opacity-30" />
                  <p>Nenhum evento registado</p>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="mt-8 text-center text-xs text-zinc-600">
          <p>Prevention Guard: PDM Validation + Shopfloor Guard + Predictive Risk Assessment</p>
          <p className="mt-1">R&D / SIFIDE: WP4 - Zero Defect Manufacturing</p>
        </div>
      </div>
    </div>
  );
};

export default PreventionGuard;



