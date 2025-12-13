/**
 * ════════════════════════════════════════════════════════════════════════════════════════════
 * MACHINES PANEL - HUB DE MÁQUINAS & MANUTENÇÃO (PRODUCTIVE CARE)
 * ════════════════════════════════════════════════════════════════════════════════════════════
 * 
 * Contrato 16 - Nova aba "Máquinas" no ProdPlan
 * 
 * Funcionalidades agregadas:
 * - Estado de saúde (SHI - Smart Health Index)
 * - RUL (Remaining Useful Life)
 * - Paragens e alarmes
 * - Manutenções planeadas e em atraso
 * - OEE/eficiência por máquina
 * - Integração com plano de produção (PdM-IPS)
 * - Drill-down para Digital Twin
 */

import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Activity,
  AlertTriangle,
  Calendar,
  Check,
  ChevronDown,
  ChevronRight,
  Clock,
  Cpu,
  ExternalLink,
  Filter,
  Gauge,
  Heart,
  Loader2,
  RefreshCw,
  Settings,
  Timer,
  TrendingDown,
  TrendingUp,
  Wrench,
  X,
  Zap,
} from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

// ═══════════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════════

interface Machine {
  machine_id: string;
  name: string;
  cell: string;
  type: string;
  status: 'healthy' | 'warning' | 'critical' | 'offline';
  health_index: number;
  rul_hours: number;
  rul_days: number;
  oee: number;
  availability: number;
  performance: number;
  quality: number;
  last_maintenance: string;
  next_maintenance: string;
  downtime_today_hours: number;
  downtime_week_hours: number;
}

interface MaintenanceEvent {
  id: string;
  machine_id: string;
  machine_name: string;
  type: 'preventive' | 'predictive' | 'corrective' | 'scheduled';
  status: 'planned' | 'in_progress' | 'completed' | 'overdue';
  scheduled_date: string;
  duration_hours: number;
  description: string;
  impact_on_plan: boolean;
}

interface StopEvent {
  id: string;
  machine_id: string;
  machine_name: string;
  start_time: string;
  end_time: string | null;
  duration_minutes: number;
  type: 'setup' | 'breakdown' | 'micro_stop' | 'planned' | 'quality';
  cause: string;
  order_affected: string | null;
}

interface MachinesOverview {
  total_machines: number;
  healthy: number;
  warning: number;
  critical: number;
  offline: number;
  avg_health_index: number;
  avg_oee: number;
  planned_maintenances_7d: number;
  overdue_maintenances: number;
  total_downtime_today: number;
}

// ═══════════════════════════════════════════════════════════════════════════════
// API
// ═══════════════════════════════════════════════════════════════════════════════

async function fetchMachinesOverview(): Promise<MachinesOverview> {
  try {
    const res = await fetch(`${API_BASE}/digital-twin/machines/overview`);
    if (!res.ok) throw new Error();
    return res.json();
  } catch {
    // Mock data
    return {
      total_machines: 24,
      healthy: 18,
      warning: 4,
      critical: 1,
      offline: 1,
      avg_health_index: 82.5,
      avg_oee: 78.3,
      planned_maintenances_7d: 6,
      overdue_maintenances: 2,
      total_downtime_today: 4.5,
    };
  }
}

async function fetchMachines(): Promise<Machine[]> {
  try {
    const res = await fetch(`${API_BASE}/digital-twin/machines`);
    if (!res.ok) throw new Error();
    return res.json();
  } catch {
    // Mock data
    return [
      { machine_id: 'm1', name: 'CNC-001', cell: 'Célula A', type: 'CNC', status: 'healthy', health_index: 94, rul_hours: 1800, rul_days: 75, oee: 85.2, availability: 92, performance: 95, quality: 97.5, last_maintenance: '2024-11-15', next_maintenance: '2025-01-15', downtime_today_hours: 0.5, downtime_week_hours: 2.5 },
      { machine_id: 'm2', name: 'CNC-002', cell: 'Célula A', type: 'CNC', status: 'healthy', health_index: 88, rul_hours: 1200, rul_days: 50, oee: 82.1, availability: 89, performance: 93, quality: 99.2, last_maintenance: '2024-11-20', next_maintenance: '2025-01-20', downtime_today_hours: 0, downtime_week_hours: 1.2 },
      { machine_id: 'm3', name: 'PRESS-001', cell: 'Célula B', type: 'Press', status: 'warning', health_index: 68, rul_hours: 450, rul_days: 19, oee: 72.5, availability: 85, performance: 88, quality: 96.8, last_maintenance: '2024-10-10', next_maintenance: '2024-12-15', downtime_today_hours: 1.5, downtime_week_hours: 8.0 },
      { machine_id: 'm4', name: 'LATHE-001', cell: 'Célula C', type: 'Lathe', status: 'critical', health_index: 42, rul_hours: 120, rul_days: 5, oee: 58.3, availability: 70, performance: 85, quality: 98.0, last_maintenance: '2024-09-01', next_maintenance: '2024-12-01', downtime_today_hours: 2.5, downtime_week_hours: 15.0 },
      { machine_id: 'm5', name: 'WELD-001', cell: 'Célula D', type: 'Welding', status: 'healthy', health_index: 91, rul_hours: 2100, rul_days: 87, oee: 88.7, availability: 94, performance: 96, quality: 98.5, last_maintenance: '2024-11-25', next_maintenance: '2025-02-25', downtime_today_hours: 0, downtime_week_hours: 0.8 },
      { machine_id: 'm6', name: 'ROBOT-001', cell: 'Célula A', type: 'Robot', status: 'offline', health_index: 0, rul_hours: 0, rul_days: 0, oee: 0, availability: 0, performance: 0, quality: 0, last_maintenance: '2024-11-30', next_maintenance: '2024-12-30', downtime_today_hours: 8, downtime_week_hours: 40 },
    ];
  }
}

async function fetchMaintenanceSchedule(): Promise<MaintenanceEvent[]> {
  try {
    const res = await fetch(`${API_BASE}/digital-twin/maintenance/schedule`);
    if (!res.ok) throw new Error();
    return res.json();
  } catch {
    return [
      { id: 'mt1', machine_id: 'm3', machine_name: 'PRESS-001', type: 'preventive', status: 'overdue', scheduled_date: '2024-12-01', duration_hours: 4, description: 'Substituição de rolamentos', impact_on_plan: true },
      { id: 'mt2', machine_id: 'm4', machine_name: 'LATHE-001', type: 'predictive', status: 'planned', scheduled_date: '2024-12-10', duration_hours: 8, description: 'Revisão geral - SHI crítico', impact_on_plan: true },
      { id: 'mt3', machine_id: 'm1', machine_name: 'CNC-001', type: 'scheduled', status: 'planned', scheduled_date: '2025-01-15', duration_hours: 2, description: 'Calibração programada', impact_on_plan: false },
      { id: 'mt4', machine_id: 'm5', machine_name: 'WELD-001', type: 'preventive', status: 'planned', scheduled_date: '2025-02-25', duration_hours: 3, description: 'Limpeza de contactos', impact_on_plan: false },
    ];
  }
}

async function fetchRecentStops(): Promise<StopEvent[]> {
  try {
    const res = await fetch(`${API_BASE}/shopfloor/stops/recent`);
    if (!res.ok) throw new Error();
    return res.json();
  } catch {
    return [
      { id: 's1', machine_id: 'm4', machine_name: 'LATHE-001', start_time: '2024-12-05T08:30:00', end_time: '2024-12-05T10:00:00', duration_minutes: 90, type: 'breakdown', cause: 'Falha no motor principal', order_affected: 'OP-2024-1234' },
      { id: 's2', machine_id: 'm3', machine_name: 'PRESS-001', start_time: '2024-12-05T14:00:00', end_time: '2024-12-05T14:45:00', duration_minutes: 45, type: 'setup', cause: 'Troca de ferramenta', order_affected: 'OP-2024-1238' },
      { id: 's3', machine_id: 'm4', machine_name: 'LATHE-001', start_time: '2024-12-05T15:30:00', end_time: null, duration_minutes: 120, type: 'breakdown', cause: 'Aquecimento excessivo', order_affected: 'OP-2024-1240' },
      { id: 's4', machine_id: 'm1', machine_name: 'CNC-001', start_time: '2024-12-05T11:00:00', end_time: '2024-12-05T11:15:00', duration_minutes: 15, type: 'micro_stop', cause: 'Ajuste de parâmetros', order_affected: null },
    ];
  }
}

// PredictiveCare APIs
interface WorkOrder {
  id: number;
  work_order_number: string;
  machine_id: string;
  title: string;
  description: string;
  priority: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL' | 'EMERGENCY';
  maintenance_type: 'PREDICTIVE' | 'PREVENTIVE' | 'CORRECTIVE' | 'IMPROVEMENT';
  status: 'OPEN' | 'PLANNED' | 'IN_PROGRESS' | 'COMPLETED' | 'CANCELLED' | 'OVERDUE';
  source: string;
  suggested_start?: string;
  scheduled_start?: string;
  shi_at_creation?: number;
  rul_at_creation?: number;
  risk_at_creation?: number;
  created_at: string;
}

interface SpareNeed {
  sku_id: string;
  component_name: string;
  machine_id: string;
  expected_replacements: number;
  recommended_date: string;
  confidence: number;
  reason: string;
  criticality: string;
  current_stock?: number;
  reorder_needed: boolean;
}

interface MaintenanceKPIs {
  total_work_orders: number;
  open_work_orders: number;
  overdue_work_orders: number;
  completed_last_30d: number;
  mttr_hours?: number;
  failures_prevented: number;
  predictive_maintenance_rate: number;
}

async function fetchWorkOrders(): Promise<WorkOrder[]> {
  try {
    const res = await fetch(`${API_BASE}/maintenance/workorders?limit=20`);
    if (!res.ok) throw new Error();
    const data = await res.json();
    return data.items || [];
  } catch {
    return [
      { id: 1, work_order_number: 'WO-20241211-001', machine_id: 'm4', title: 'PredictiveCare: Critical risk: 65%', description: 'Auto-generated by PredictiveCare', priority: 'CRITICAL', maintenance_type: 'PREDICTIVE', status: 'OPEN', source: 'PREDICTIVECARE', shi_at_creation: 42, rul_at_creation: 120, risk_at_creation: 0.65, created_at: '2024-12-11T10:00:00Z' },
      { id: 2, work_order_number: 'WO-20241210-002', machine_id: 'm3', title: 'PredictiveCare: High risk: 35%', description: 'Auto-generated by PredictiveCare', priority: 'HIGH', maintenance_type: 'PREDICTIVE', status: 'PLANNED', source: 'PREDICTIVECARE', shi_at_creation: 68, rul_at_creation: 450, risk_at_creation: 0.35, created_at: '2024-12-10T08:00:00Z', scheduled_start: '2024-12-15T06:00:00Z' },
      { id: 3, work_order_number: 'WO-20241205-003', machine_id: 'm1', title: 'Preventive: Calibration', description: 'Scheduled calibration', priority: 'MEDIUM', maintenance_type: 'PREVENTIVE', status: 'COMPLETED', source: 'MANUAL', created_at: '2024-12-05T09:00:00Z' },
    ];
  }
}

async function fetchSpareNeeds(): Promise<SpareNeed[]> {
  try {
    const res = await fetch(`${API_BASE}/smart-inventory/spares/forecast?horizon_days=30`);
    if (!res.ok) throw new Error();
    const data = await res.json();
    return data.needs || [];
  } catch {
    return [
      { sku_id: 'SPARE-BRG-001', component_name: 'Main Spindle Bearing', machine_id: 'm4', expected_replacements: 0.85, recommended_date: '2024-12-18T00:00:00Z', confidence: 0.72, reason: 'Due in 168 hours (accelerated by machine degradation)', criticality: 'HIGH', current_stock: 2, reorder_needed: false },
      { sku_id: 'SPARE-BLT-001', component_name: 'Drive Belt', machine_id: 'm3', expected_replacements: 0.6, recommended_date: '2024-12-25T00:00:00Z', confidence: 0.65, reason: 'Due in ~14 days', criticality: 'MEDIUM', current_stock: 3, reorder_needed: false },
      { sku_id: 'SPARE-FLT-001', component_name: 'Hydraulic Filter', machine_id: 'm3', expected_replacements: 0.9, recommended_date: '2024-12-14T00:00:00Z', confidence: 0.85, reason: 'Due in 72 hours', criticality: 'MEDIUM', current_stock: 1, reorder_needed: true },
    ];
  }
}

async function fetchMaintenanceKPIs(): Promise<MaintenanceKPIs> {
  try {
    const res = await fetch(`${API_BASE}/maintenance/kpis`);
    if (!res.ok) throw new Error();
    return res.json();
  } catch {
    return {
      total_work_orders: 12,
      open_work_orders: 3,
      overdue_work_orders: 1,
      completed_last_30d: 8,
      mttr_hours: 3.5,
      failures_prevented: 5,
      predictive_maintenance_rate: 0.58,
    };
  }
}

async function triggerPredictiveCareEvaluation(): Promise<any> {
  const res = await fetch(`${API_BASE}/maintenance/predictivecare/evaluate`, { method: 'POST' });
  if (!res.ok) throw new Error('Failed to evaluate');
  return res.json();
}

// ═══════════════════════════════════════════════════════════════════════════════
// COMPONENTS
// ═══════════════════════════════════════════════════════════════════════════════

const KpiCard: React.FC<{
  title: string;
  value: string | number;
  subtitle?: string;
  icon: React.ReactNode;
  color: 'cyan' | 'emerald' | 'amber' | 'red' | 'purple';
  trend?: { value: number; direction: 'up' | 'down' };
}> = ({ title, value, subtitle, icon, color, trend }) => {
  const colors = {
    cyan: 'from-cyan-500/20 to-blue-500/20 border-cyan-500/30 text-cyan-400',
    emerald: 'from-emerald-500/20 to-green-500/20 border-emerald-500/30 text-emerald-400',
    amber: 'from-amber-500/20 to-orange-500/20 border-amber-500/30 text-amber-400',
    red: 'from-red-500/20 to-rose-500/20 border-red-500/30 text-red-400',
    purple: 'from-purple-500/20 to-pink-500/20 border-purple-500/30 text-purple-400',
  };
  
  return (
    <div className={`p-4 rounded-xl bg-gradient-to-br ${colors[color]} border`}>
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm text-slate-400">{title}</span>
        {icon}
      </div>
      <div className="flex items-end justify-between">
        <div>
          <p className="text-2xl font-bold text-white">{value}</p>
          {subtitle && <p className="text-xs text-slate-500">{subtitle}</p>}
        </div>
        {trend && (
          <div className={`flex items-center gap-1 text-xs ${trend.direction === 'up' ? 'text-emerald-400' : 'text-red-400'}`}>
            {trend.direction === 'up' ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />}
            {trend.value}%
          </div>
        )}
      </div>
    </div>
  );
};

const HealthBadge: React.FC<{ status: string; size?: 'sm' | 'md' }> = ({ status, size = 'md' }) => {
  const config: Record<string, { bg: string; text: string; label: string }> = {
    healthy: { bg: 'bg-emerald-500/20', text: 'text-emerald-400', label: 'Saudável' },
    warning: { bg: 'bg-amber-500/20', text: 'text-amber-400', label: 'Alerta' },
    critical: { bg: 'bg-red-500/20', text: 'text-red-400', label: 'Crítico' },
    offline: { bg: 'bg-slate-500/20', text: 'text-slate-400', label: 'Offline' },
  };
  const c = config[status] || config.healthy;
  const sizeClass = size === 'sm' ? 'px-2 py-0.5 text-xs' : 'px-3 py-1 text-sm';
  
  return (
    <span className={`rounded-full font-medium ${c.bg} ${c.text} ${sizeClass}`}>
      {c.label}
    </span>
  );
};

const HealthGauge: React.FC<{ value: number; size?: number }> = ({ value, size = 48 }) => {
  const getColor = () => {
    if (value >= 80) return 'stroke-emerald-400';
    if (value >= 50) return 'stroke-amber-400';
    return 'stroke-red-400';
  };
  
  const radius = (size - 6) / 2;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (value / 100) * circumference;

  return (
    <div className="relative" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="transform -rotate-90">
        <circle cx={size/2} cy={size/2} r={radius} fill="none" stroke="currentColor" strokeWidth={6} className="text-slate-700" />
        <circle cx={size/2} cy={size/2} r={radius} fill="none" strokeWidth={6} strokeLinecap="round" strokeDasharray={circumference} strokeDashoffset={offset} className={getColor()} />
      </svg>
      <div className="absolute inset-0 flex items-center justify-center">
        <span className="text-xs font-bold text-white">{value}</span>
      </div>
    </div>
  );
};

const MachineRow: React.FC<{ 
  machine: Machine; 
  onViewDetails: () => void;
  onOpenDigitalTwin: () => void;
}> = ({ machine, onViewDetails, onOpenDigitalTwin }) => (
  <motion.tr
    initial={{ opacity: 0 }}
    animate={{ opacity: 1 }}
    className="border-b border-slate-700/30 hover:bg-slate-800/30 transition-colors"
  >
    <td className="px-4 py-3">
      <div className="flex items-center gap-3">
        <HealthGauge value={machine.health_index} size={40} />
        <div>
          <p className="font-medium text-white">{machine.name}</p>
          <p className="text-xs text-slate-500">{machine.cell} • {machine.type}</p>
        </div>
      </div>
    </td>
    <td className="px-4 py-3">
      <HealthBadge status={machine.status} size="sm" />
    </td>
    <td className="px-4 py-3 text-right">
      <span className={`font-medium ${machine.rul_days <= 7 ? 'text-red-400' : machine.rul_days <= 30 ? 'text-amber-400' : 'text-white'}`}>
        {machine.rul_days} dias
      </span>
      <p className="text-xs text-slate-500">{machine.rul_hours}h</p>
    </td>
    <td className="px-4 py-3 text-right">
      <span className={`font-medium ${machine.oee >= 80 ? 'text-emerald-400' : machine.oee >= 60 ? 'text-amber-400' : 'text-red-400'}`}>
        {machine.oee.toFixed(1)}%
      </span>
    </td>
    <td className="px-4 py-3 text-right text-slate-400 text-sm">
      {machine.next_maintenance}
    </td>
    <td className="px-4 py-3 text-right">
      <span className={machine.downtime_today_hours > 2 ? 'text-red-400' : 'text-slate-400'}>
        {machine.downtime_today_hours.toFixed(1)}h
      </span>
    </td>
    <td className="px-4 py-3">
      <div className="flex items-center gap-2 justify-end">
        <button
          onClick={onViewDetails}
          className="p-1.5 text-slate-400 hover:text-white hover:bg-slate-700 rounded-lg transition"
          title="Ver detalhes"
        >
          <ChevronRight className="w-4 h-4" />
        </button>
        <button
          onClick={onOpenDigitalTwin}
          className="p-1.5 text-cyan-400 hover:text-cyan-300 hover:bg-cyan-500/20 rounded-lg transition"
          title="Abrir em Digital Twin"
        >
          <ExternalLink className="w-4 h-4" />
        </button>
      </div>
    </td>
  </motion.tr>
);

const MaintenanceCard: React.FC<{ event: MaintenanceEvent }> = ({ event }) => {
  const typeColors = {
    preventive: 'bg-blue-500/20 text-blue-400',
    predictive: 'bg-purple-500/20 text-purple-400',
    corrective: 'bg-red-500/20 text-red-400',
    scheduled: 'bg-slate-500/20 text-slate-400',
  };
  
  const statusColors = {
    planned: 'border-slate-600',
    in_progress: 'border-cyan-500',
    completed: 'border-emerald-500',
    overdue: 'border-red-500',
  };
  
  return (
    <div className={`p-3 rounded-lg bg-slate-800/50 border-l-4 ${statusColors[event.status]}`}>
      <div className="flex items-start justify-between mb-2">
        <div>
          <p className="font-medium text-white">{event.machine_name}</p>
          <p className="text-xs text-slate-400">{event.description}</p>
        </div>
        <span className={`px-2 py-0.5 rounded text-xs font-medium ${typeColors[event.type]}`}>
          {event.type}
        </span>
      </div>
      <div className="flex items-center justify-between text-xs">
        <span className="text-slate-500">
          <Calendar className="w-3 h-3 inline mr-1" />
          {event.scheduled_date}
        </span>
        <span className="text-slate-500">
          <Timer className="w-3 h-3 inline mr-1" />
          {event.duration_hours}h
        </span>
        {event.impact_on_plan && (
          <span className="text-amber-400">
            <AlertTriangle className="w-3 h-3 inline mr-1" />
            Impacto no plano
          </span>
        )}
      </div>
    </div>
  );
};

const StopRow: React.FC<{ stop: StopEvent }> = ({ stop }) => {
  const typeColors = {
    setup: 'bg-blue-500/20 text-blue-400',
    breakdown: 'bg-red-500/20 text-red-400',
    micro_stop: 'bg-amber-500/20 text-amber-400',
    planned: 'bg-slate-500/20 text-slate-400',
    quality: 'bg-purple-500/20 text-purple-400',
  };
  
  const typeLabels = {
    setup: 'Setup',
    breakdown: 'Avaria',
    micro_stop: 'Microparagem',
    planned: 'Planeada',
    quality: 'Qualidade',
  };
  
  return (
    <tr className="border-b border-slate-700/30">
      <td className="px-3 py-2 text-sm text-white">{stop.machine_name}</td>
      <td className="px-3 py-2">
        <span className={`px-2 py-0.5 rounded text-xs font-medium ${typeColors[stop.type]}`}>
          {typeLabels[stop.type]}
        </span>
      </td>
      <td className="px-3 py-2 text-sm text-slate-400">{stop.cause}</td>
      <td className="px-3 py-2 text-sm text-right">
        <span className={stop.duration_minutes > 60 ? 'text-red-400' : 'text-slate-400'}>
          {stop.duration_minutes} min
        </span>
      </td>
      <td className="px-3 py-2 text-sm text-slate-500">
        {stop.order_affected || '—'}
      </td>
      <td className="px-3 py-2 text-sm text-right">
        {stop.end_time ? (
          <span className="text-emerald-400">Resolvida</span>
        ) : (
          <span className="text-red-400 animate-pulse">Em curso</span>
        )}
      </td>
    </tr>
  );
};

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN COMPONENT
// ═══════════════════════════════════════════════════════════════════════════════

export const MachinesPanel: React.FC = () => {
  const [activeSubTab, setActiveSubTab] = useState<'map' | 'maintenance' | 'stops' | 'workorders' | 'spares'>('map');
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [cellFilter, setCellFilter] = useState<string>('all');
  const [selectedMachine, setSelectedMachine] = useState<Machine | null>(null);

  const { data: overview, isLoading: overviewLoading, refetch: refetchOverview } = useQuery({
    queryKey: ['machines-overview'],
    queryFn: fetchMachinesOverview,
    refetchInterval: 30000,
  });

  const { data: machines, isLoading: machinesLoading, refetch: refetchMachines } = useQuery({
    queryKey: ['machines-list'],
    queryFn: fetchMachines,
    refetchInterval: 30000,
  });

  const { data: maintenanceSchedule } = useQuery({
    queryKey: ['maintenance-schedule'],
    queryFn: fetchMaintenanceSchedule,
  });

  const { data: recentStops } = useQuery({
    queryKey: ['recent-stops'],
    queryFn: fetchRecentStops,
    refetchInterval: 60000,
  });

  // PredictiveCare queries
  const { data: workOrders, refetch: refetchWorkOrders } = useQuery({
    queryKey: ['work-orders'],
    queryFn: fetchWorkOrders,
    refetchInterval: 60000,
  });

  const { data: spareNeeds } = useQuery({
    queryKey: ['spare-needs'],
    queryFn: fetchSpareNeeds,
    refetchInterval: 300000, // 5 minutes
  });

  const { data: maintenanceKPIs } = useQuery({
    queryKey: ['maintenance-kpis'],
    queryFn: fetchMaintenanceKPIs,
    refetchInterval: 60000,
  });

  const [evaluating, setEvaluating] = useState(false);
  
  const handleEvaluate = async () => {
    setEvaluating(true);
    try {
      await triggerPredictiveCareEvaluation();
      refetchWorkOrders();
    } catch (e) {
      console.error('Evaluation failed:', e);
    } finally {
      setEvaluating(false);
    }
  };

  // Ensure arrays are always valid (fallback for API errors)
  const machinesArray = Array.isArray(machines) ? machines : [];
  const maintenanceArray = Array.isArray(maintenanceSchedule) ? maintenanceSchedule : [];
  const stopsArray = Array.isArray(recentStops) ? recentStops : [];
  const workOrdersArray = Array.isArray(workOrders) ? workOrders : [];
  const spareNeedsArray = Array.isArray(spareNeeds) ? spareNeeds : [];
  
  const filteredMachines = machinesArray.filter(m => {
    if (statusFilter !== 'all' && m.status !== statusFilter) return false;
    if (cellFilter !== 'all' && m.cell !== cellFilter) return false;
    return true;
  });

  const uniqueCells = [...new Set(machinesArray.map(m => m.cell))];
  
  const overdueMaintenances = maintenanceArray.filter(m => m.status === 'overdue');
  const plannedMaintenances = maintenanceArray.filter(m => m.status === 'planned');
  
  const openWorkOrders = workOrdersArray.filter(wo => wo.status === 'OPEN' || wo.status === 'PLANNED');
  const criticalWorkOrders = workOrdersArray.filter(wo => wo.priority === 'CRITICAL' || wo.priority === 'EMERGENCY');
  const urgentSpares = spareNeedsArray.filter(s => s.reorder_needed || s.criticality === 'CRITICAL');

  const handleRefresh = () => {
    refetchOverview();
    refetchMachines();
  };

  const openDigitalTwin = (machineId: string) => {
    window.open(`/digital-twin/machines?machine=${machineId}`, '_blank');
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold text-white flex items-center gap-2">
            <Cpu className="w-5 h-5 text-purple-400" />
            Máquinas & Manutenção
          </h2>
          <p className="text-sm text-slate-400 mt-1">
            Estado, saúde, paragens e manutenção dos recursos produtivos
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={handleRefresh}
            className="flex items-center gap-2 px-4 py-2 bg-slate-700 hover:bg-slate-600 text-white rounded-lg text-sm"
          >
            <RefreshCw className="w-4 h-4" />
            Atualizar
          </button>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-5 gap-4">
        <KpiCard
          title="Máquinas Ativas"
          value={overview?.total_machines || 0}
          subtitle={`${overview?.offline || 0} offline`}
          icon={<Cpu className="w-5 h-5" />}
          color="cyan"
        />
        <KpiCard
          title="SHI Médio"
          value={`${overview?.avg_health_index?.toFixed(1) || 0}%`}
          subtitle={`${overview?.critical || 0} críticas`}
          icon={<Heart className="w-5 h-5" />}
          color={overview?.avg_health_index && overview.avg_health_index >= 80 ? 'emerald' : overview?.avg_health_index && overview.avg_health_index >= 60 ? 'amber' : 'red'}
        />
        <KpiCard
          title="OEE Médio"
          value={`${overview?.avg_oee?.toFixed(1) || 0}%`}
          icon={<Gauge className="w-5 h-5" />}
          color={overview?.avg_oee && overview.avg_oee >= 80 ? 'emerald' : 'amber'}
        />
        <KpiCard
          title="Manutenções 7d"
          value={overview?.planned_maintenances_7d || 0}
          subtitle={`${overview?.overdue_maintenances || 0} em atraso`}
          icon={<Wrench className="w-5 h-5" />}
          color={overview?.overdue_maintenances && overview.overdue_maintenances > 0 ? 'red' : 'purple'}
        />
        <KpiCard
          title="Paragens Hoje"
          value={`${overview?.total_downtime_today?.toFixed(1) || 0}h`}
          icon={<Clock className="w-5 h-5" />}
          color={overview?.total_downtime_today && overview.total_downtime_today > 4 ? 'red' : 'amber'}
        />
      </div>

      {/* Toolbar */}
      <div className="flex items-center justify-between p-3 bg-slate-800/50 rounded-xl border border-slate-700/50">
        <div className="flex items-center gap-2">
          <Filter className="w-4 h-4 text-slate-500" />
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="bg-slate-900 border border-slate-700 rounded-lg px-3 py-1.5 text-sm text-white"
          >
            <option value="all">Todos os estados</option>
            <option value="healthy">Saudáveis</option>
            <option value="warning">Alerta</option>
            <option value="critical">Críticas</option>
            <option value="offline">Offline</option>
          </select>
          <select
            value={cellFilter}
            onChange={(e) => setCellFilter(e.target.value)}
            className="bg-slate-900 border border-slate-700 rounded-lg px-3 py-1.5 text-sm text-white"
          >
            <option value="all">Todas as células</option>
            {uniqueCells.map(cell => (
              <option key={cell} value={cell}>{cell}</option>
            ))}
          </select>
        </div>
        <div className="flex items-center gap-2">
          {[
            { id: 'map', label: 'Mapa' },
            { id: 'maintenance', label: 'Agenda' },
            { id: 'stops', label: 'Paragens' },
            { id: 'workorders', label: 'Ordens de Trabalho' },
            { id: 'spares', label: 'Peças Sobressalentes' },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveSubTab(tab.id as any)}
              className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
                activeSubTab === tab.id
                  ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/30'
                  : 'text-slate-400 hover:text-white hover:bg-slate-700/50'
              }`}
            >
              {tab.label}
              {tab.id === 'workorders' && openWorkOrders.length > 0 && (
                <span className="ml-1.5 px-1.5 py-0.5 bg-red-500/20 text-red-400 text-xs rounded-full">
                  {openWorkOrders.length}
                </span>
              )}
              {tab.id === 'spares' && urgentSpares.length > 0 && (
                <span className="ml-1.5 px-1.5 py-0.5 bg-amber-500/20 text-amber-400 text-xs rounded-full">
                  {urgentSpares.length}
                </span>
              )}
            </button>
          ))}
        </div>
      </div>

      {/* Main Content */}
      <AnimatePresence mode="wait">
        {activeSubTab === 'map' && (
          <motion.div
            key="map"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="rounded-xl border border-slate-700/50 bg-slate-800/30 overflow-hidden"
          >
            {machinesLoading ? (
              <div className="flex items-center justify-center py-16">
                <Loader2 className="w-8 h-8 text-cyan-400 animate-spin" />
              </div>
            ) : (
              <table className="w-full">
                <thead className="bg-slate-900/50">
                  <tr className="text-left text-xs text-slate-500 uppercase tracking-wider">
                    <th className="px-4 py-3">Máquina</th>
                    <th className="px-4 py-3">Estado</th>
                    <th className="px-4 py-3 text-right">RUL</th>
                    <th className="px-4 py-3 text-right">OEE</th>
                    <th className="px-4 py-3 text-right">Próx. Manutenção</th>
                    <th className="px-4 py-3 text-right">Paragem Hoje</th>
                    <th className="px-4 py-3 text-right">Ações</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredMachines?.map((machine) => (
                    <MachineRow
                      key={machine.machine_id}
                      machine={machine}
                      onViewDetails={() => setSelectedMachine(machine)}
                      onOpenDigitalTwin={() => openDigitalTwin(machine.machine_id)}
                    />
                  ))}
                </tbody>
              </table>
            )}
          </motion.div>
        )}

        {activeSubTab === 'maintenance' && (
          <motion.div
            key="maintenance"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="grid grid-cols-2 gap-6"
          >
            {/* Overdue */}
            <div className="rounded-xl border border-red-500/30 bg-red-500/5 p-4">
              <h3 className="text-lg font-semibold text-red-400 mb-4 flex items-center gap-2">
                <AlertTriangle className="w-5 h-5" />
                Em Atraso ({overdueMaintenances.length})
              </h3>
              <div className="space-y-3">
                {overdueMaintenances.map(event => (
                  <MaintenanceCard key={event.id} event={event} />
                ))}
                {overdueMaintenances.length === 0 && (
                  <p className="text-slate-500 text-sm text-center py-4">Sem manutenções em atraso</p>
                )}
              </div>
            </div>
            
            {/* Planned */}
            <div className="rounded-xl border border-slate-700/50 bg-slate-800/30 p-4">
              <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                <Calendar className="w-5 h-5 text-cyan-400" />
                Planeadas ({plannedMaintenances.length})
              </h3>
              <div className="space-y-3 max-h-80 overflow-y-auto">
                {plannedMaintenances.map(event => (
                  <MaintenanceCard key={event.id} event={event} />
                ))}
              </div>
            </div>
          </motion.div>
        )}

        {activeSubTab === 'stops' && (
          <motion.div
            key="stops"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="rounded-xl border border-slate-700/50 bg-slate-800/30 overflow-hidden"
          >
            <table className="w-full">
              <thead className="bg-slate-900/50">
                <tr className="text-left text-xs text-slate-500 uppercase tracking-wider">
                  <th className="px-3 py-3">Máquina</th>
                  <th className="px-3 py-3">Tipo</th>
                  <th className="px-3 py-3">Causa</th>
                  <th className="px-3 py-3 text-right">Duração</th>
                  <th className="px-3 py-3">Ordem Afetada</th>
                  <th className="px-3 py-3 text-right">Estado</th>
                </tr>
              </thead>
              <tbody>
                {stopsArray.map((stop) => (
                  <StopRow key={stop.id} stop={stop} />
                ))}
                {stopsArray.length === 0 && (
                  <tr>
                    <td colSpan={6} className="px-3 py-8 text-center text-slate-500">
                      Sem paragens recentes
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </motion.div>
        )}

        {/* Work Orders Tab - PredictiveCare */}
        {activeSubTab === 'workorders' && (
          <motion.div
            key="workorders"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="space-y-4"
          >
            {/* PredictiveCare KPIs */}
            <div className="grid grid-cols-4 gap-4">
              <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700/50">
                <p className="text-xs text-slate-500 mb-1">Ordens Abertas</p>
                <p className="text-2xl font-bold text-white">{maintenanceKPIs?.open_work_orders || 0}</p>
                <p className="text-xs text-amber-400 mt-1">{maintenanceKPIs?.overdue_work_orders || 0} em atraso</p>
              </div>
              <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700/50">
                <p className="text-xs text-slate-500 mb-1">Falhas Evitadas</p>
                <p className="text-2xl font-bold text-emerald-400">{maintenanceKPIs?.failures_prevented || 0}</p>
                <p className="text-xs text-slate-500 mt-1">últimos 30 dias</p>
              </div>
              <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700/50">
                <p className="text-xs text-slate-500 mb-1">MTTR Médio</p>
                <p className="text-2xl font-bold text-cyan-400">{maintenanceKPIs?.mttr_hours?.toFixed(1) || '-'}h</p>
                <p className="text-xs text-slate-500 mt-1">tempo de reparação</p>
              </div>
              <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700/50">
                <p className="text-xs text-slate-500 mb-1">Taxa Preditiva</p>
                <p className="text-2xl font-bold text-purple-400">{((maintenanceKPIs?.predictive_maintenance_rate || 0) * 100).toFixed(0)}%</p>
                <p className="text-xs text-slate-500 mt-1">manutenções preditivas</p>
              </div>
            </div>

            {/* Evaluate Button */}
            <div className="flex justify-between items-center">
              <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                <Wrench className="w-5 h-5 text-cyan-400" />
                Ordens de Trabalho
              </h3>
              <button
                onClick={handleEvaluate}
                disabled={evaluating}
                className="flex items-center gap-2 px-4 py-2 bg-cyan-500/20 hover:bg-cyan-500/30 text-cyan-400 rounded-lg text-sm border border-cyan-500/30"
              >
                {evaluating ? <Loader2 className="w-4 h-4 animate-spin" /> : <Zap className="w-4 h-4" />}
                {evaluating ? 'Avaliando...' : 'Avaliar PredictiveCare'}
              </button>
            </div>

            {/* Work Orders List */}
            <div className="rounded-xl border border-slate-700/50 overflow-hidden">
              <table className="w-full">
                <thead className="bg-slate-900/50">
                  <tr className="text-left text-xs text-slate-500 uppercase tracking-wider">
                    <th className="px-3 py-3">Ordem</th>
                    <th className="px-3 py-3">Máquina</th>
                    <th className="px-3 py-3">Título</th>
                    <th className="px-3 py-3">Prioridade</th>
                    <th className="px-3 py-3">Tipo</th>
                    <th className="px-3 py-3">Estado</th>
                    <th className="px-3 py-3">SHI/Risco</th>
                    <th className="px-3 py-3">Criado</th>
                  </tr>
                </thead>
                <tbody>
                  {workOrdersArray.map((wo) => (
                    <tr key={wo.id} className="border-t border-slate-700/30 hover:bg-slate-800/30">
                      <td className="px-3 py-3 text-sm font-mono text-white">{wo.work_order_number}</td>
                      <td className="px-3 py-3 text-sm text-slate-300">{wo.machine_id}</td>
                      <td className="px-3 py-3 text-sm text-white max-w-xs truncate">{wo.title}</td>
                      <td className="px-3 py-3">
                        <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${
                          wo.priority === 'EMERGENCY' ? 'bg-red-500/20 text-red-400' :
                          wo.priority === 'CRITICAL' ? 'bg-orange-500/20 text-orange-400' :
                          wo.priority === 'HIGH' ? 'bg-amber-500/20 text-amber-400' :
                          wo.priority === 'MEDIUM' ? 'bg-cyan-500/20 text-cyan-400' :
                          'bg-slate-500/20 text-slate-400'
                        }`}>
                          {wo.priority}
                        </span>
                      </td>
                      <td className="px-3 py-3 text-sm text-slate-400">{wo.maintenance_type}</td>
                      <td className="px-3 py-3">
                        <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${
                          wo.status === 'OPEN' ? 'bg-blue-500/20 text-blue-400' :
                          wo.status === 'PLANNED' ? 'bg-cyan-500/20 text-cyan-400' :
                          wo.status === 'IN_PROGRESS' ? 'bg-purple-500/20 text-purple-400' :
                          wo.status === 'COMPLETED' ? 'bg-emerald-500/20 text-emerald-400' :
                          wo.status === 'OVERDUE' ? 'bg-red-500/20 text-red-400' :
                          'bg-slate-500/20 text-slate-400'
                        }`}>
                          {wo.status}
                        </span>
                      </td>
                      <td className="px-3 py-3 text-sm">
                        {wo.shi_at_creation != null && (
                          <span className="text-slate-400">
                            SHI: {wo.shi_at_creation.toFixed(0)}% • Risk: {((wo.risk_at_creation || 0) * 100).toFixed(0)}%
                          </span>
                        )}
                      </td>
                      <td className="px-3 py-3 text-sm text-slate-500">
                        {new Date(wo.created_at).toLocaleDateString('pt-PT')}
                      </td>
                    </tr>
                  ))}
                  {workOrdersArray.length === 0 && (
                    <tr>
                      <td colSpan={8} className="px-3 py-8 text-center text-slate-500">
                        Sem ordens de trabalho
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </motion.div>
        )}

        {/* Spare Parts Tab */}
        {activeSubTab === 'spares' && (
          <motion.div
            key="spares"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="space-y-4"
          >
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                <Settings className="w-5 h-5 text-amber-400" />
                Previsão de Peças Sobressalentes (30 dias)
              </h3>
              <div className="text-sm text-slate-400">
                {urgentSpares.length} peças requerem atenção
              </div>
            </div>

            <div className="rounded-xl border border-slate-700/50 overflow-hidden">
              <table className="w-full">
                <thead className="bg-slate-900/50">
                  <tr className="text-left text-xs text-slate-500 uppercase tracking-wider">
                    <th className="px-3 py-3">SKU</th>
                    <th className="px-3 py-3">Componente</th>
                    <th className="px-3 py-3">Máquina</th>
                    <th className="px-3 py-3">Criticidade</th>
                    <th className="px-3 py-3">Data Prevista</th>
                    <th className="px-3 py-3">Prob.</th>
                    <th className="px-3 py-3">Stock</th>
                    <th className="px-3 py-3">Razão</th>
                  </tr>
                </thead>
                <tbody>
                  {spareNeedsArray.map((spare, idx) => (
                    <tr key={idx} className={`border-t border-slate-700/30 hover:bg-slate-800/30 ${spare.reorder_needed ? 'bg-red-500/5' : ''}`}>
                      <td className="px-3 py-3 text-sm font-mono text-white">{spare.sku_id}</td>
                      <td className="px-3 py-3 text-sm text-slate-300">{spare.component_name}</td>
                      <td className="px-3 py-3 text-sm text-slate-400">{spare.machine_id}</td>
                      <td className="px-3 py-3">
                        <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${
                          spare.criticality === 'CRITICAL' ? 'bg-red-500/20 text-red-400' :
                          spare.criticality === 'HIGH' ? 'bg-amber-500/20 text-amber-400' :
                          'bg-slate-500/20 text-slate-400'
                        }`}>
                          {spare.criticality}
                        </span>
                      </td>
                      <td className="px-3 py-3 text-sm text-slate-300">
                        {new Date(spare.recommended_date).toLocaleDateString('pt-PT')}
                      </td>
                      <td className="px-3 py-3 text-sm">
                        <span className={`font-medium ${
                          spare.expected_replacements >= 0.8 ? 'text-red-400' :
                          spare.expected_replacements >= 0.5 ? 'text-amber-400' :
                          'text-slate-400'
                        }`}>
                          {(spare.expected_replacements * 100).toFixed(0)}%
                        </span>
                      </td>
                      <td className="px-3 py-3 text-sm">
                        <span className={`font-medium ${spare.reorder_needed ? 'text-red-400' : 'text-emerald-400'}`}>
                          {spare.current_stock || 0}
                          {spare.reorder_needed && (
                            <span className="ml-1 text-xs text-red-500">⚠️</span>
                          )}
                        </span>
                      </td>
                      <td className="px-3 py-3 text-sm text-slate-500 max-w-xs truncate" title={spare.reason}>
                        {spare.reason}
                      </td>
                    </tr>
                  ))}
                  {spareNeedsArray.length === 0 && (
                    <tr>
                      <td colSpan={8} className="px-3 py-8 text-center text-slate-500">
                        Sem previsões de substituição nos próximos 30 dias
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Machine Detail Modal */}
      <AnimatePresence>
        {selectedMachine && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4"
            onClick={() => setSelectedMachine(null)}
          >
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              className="bg-slate-900 rounded-2xl p-6 max-w-2xl w-full border border-slate-700"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-4">
                  <HealthGauge value={selectedMachine.health_index} size={64} />
                  <div>
                    <h3 className="text-xl font-bold text-white">{selectedMachine.name}</h3>
                    <p className="text-sm text-slate-400">{selectedMachine.cell} • {selectedMachine.type}</p>
                  </div>
                </div>
                <button
                  onClick={() => setSelectedMachine(null)}
                  className="p-2 text-slate-400 hover:text-white hover:bg-slate-800 rounded-lg"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>

              <div className="grid grid-cols-4 gap-4 mb-6">
                <div className="bg-slate-800/50 rounded-lg p-3">
                  <p className="text-xs text-slate-500">SHI</p>
                  <p className="text-xl font-bold text-white">{selectedMachine.health_index}%</p>
                </div>
                <div className="bg-slate-800/50 rounded-lg p-3">
                  <p className="text-xs text-slate-500">RUL</p>
                  <p className="text-xl font-bold text-white">{selectedMachine.rul_days}d</p>
                </div>
                <div className="bg-slate-800/50 rounded-lg p-3">
                  <p className="text-xs text-slate-500">OEE</p>
                  <p className="text-xl font-bold text-white">{selectedMachine.oee.toFixed(1)}%</p>
                </div>
                <div className="bg-slate-800/50 rounded-lg p-3">
                  <p className="text-xs text-slate-500">Paragens (sem.)</p>
                  <p className="text-xl font-bold text-white">{selectedMachine.downtime_week_hours}h</p>
                </div>
              </div>

              <div className="grid grid-cols-3 gap-4 mb-6">
                <div className="bg-slate-800/50 rounded-lg p-3">
                  <p className="text-xs text-slate-500">Disponibilidade</p>
                  <p className="text-lg font-bold text-emerald-400">{selectedMachine.availability}%</p>
                </div>
                <div className="bg-slate-800/50 rounded-lg p-3">
                  <p className="text-xs text-slate-500">Performance</p>
                  <p className="text-lg font-bold text-cyan-400">{selectedMachine.performance}%</p>
                </div>
                <div className="bg-slate-800/50 rounded-lg p-3">
                  <p className="text-xs text-slate-500">Qualidade</p>
                  <p className="text-lg font-bold text-purple-400">{selectedMachine.quality}%</p>
                </div>
              </div>

              <div className="flex gap-3">
                <button
                  onClick={() => {
                    setSelectedMachine(null);
                    openDigitalTwin(selectedMachine.machine_id);
                  }}
                  className="flex-1 flex items-center justify-center gap-2 py-2 bg-cyan-600 hover:bg-cyan-500 text-white rounded-lg font-medium"
                >
                  <ExternalLink className="w-4 h-4" />
                  Abrir em Digital Twin
                </button>
                <button
                  className="flex-1 flex items-center justify-center gap-2 py-2 bg-purple-600 hover:bg-purple-500 text-white rounded-lg font-medium"
                >
                  <Wrench className="w-4 h-4" />
                  Agendar Manutenção
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default MachinesPanel;

