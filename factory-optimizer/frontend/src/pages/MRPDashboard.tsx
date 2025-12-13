/**
 * ═══════════════════════════════════════════════════════════════════════════════
 * MRP Dashboard - Material Requirements Planning
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * Dashboard for MRP planning and analysis:
 * - Run MRP calculations
 * - View planned orders (purchase + manufacture)
 * - Analyze item plans
 * - Monitor alerts and shortages
 *
 * R&D / SIFIDE: WP3 - Inventory & Capacity Optimization
 */

import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  AlertTriangle,
  ArrowRight,
  BarChart3,
  Box,
  Calendar,
  Check,
  ChevronDown,
  ChevronRight,
  Clock,
  Factory,
  FileText,
  Layers,
  Package,
  Play,
  RefreshCw,
  Settings,
  ShoppingCart,
  TrendingDown,
  TrendingUp,
  Truck,
  Zap,
} from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  LineChart,
  Line,
  Legend,
} from 'recharts';

// ═══════════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════════

interface PlannedOrder {
  order_id: string;
  item_id: number;
  sku: string;
  order_type: string;
  status: string;
  quantity: number;
  start_date: string;
  due_date: string;
  lead_time_days: number;
  gross_requirement: number;
  net_requirement: number;
}

interface PeriodBucket {
  period_start: string;
  gross_requirements: number;
  scheduled_receipts: number;
  projected_on_hand: number;
  net_requirements: number;
  planned_receipts: number;
}

interface ItemPlan {
  item_id: number;
  sku: string;
  name: string;
  periods: PeriodBucket[];
  planned_orders: PlannedOrder[];
  total_gross_requirement: number;
  total_net_requirement: number;
  total_planned_quantity: number;
  coverage_days: number;
  stockout_risk: boolean;
}

interface MRPRunResult {
  run_id: string;
  run_timestamp: string;
  horizon_start: string;
  horizon_end: string;
  items_processed: number;
  demands_processed: number;
  bom_levels_exploded: number;
  item_plans: Record<string, ItemPlan>;
  purchase_orders: PlannedOrder[];
  manufacture_orders: PlannedOrder[];
  shortage_alerts: Array<{ sku: string; name: string; stockout_periods: string[] }>;
  capacity_alerts: Array<{ work_center: string; overload_hours: number }>;
  warnings: string[];
  summary: {
    total_purchase_orders: number;
    total_manufacture_orders: number;
    total_shortage_alerts: number;
    total_capacity_alerts: number;
  };
}

interface MRPStatus {
  config: {
    horizon_days: number;
    period_days: number;
  };
  runs_stored: number;
}

// ═══════════════════════════════════════════════════════════════════════════════
// API
// ═══════════════════════════════════════════════════════════════════════════════

const API_BASE = 'http://127.0.0.1:8000/mrp';

const fetchStatus = async (): Promise<MRPStatus> => {
  const res = await fetch(`${API_BASE}/status`);
  if (!res.ok) throw new Error('Failed to fetch status');
  return res.json();
};

const runDemoMRP = async (): Promise<MRPRunResult> => {
  const res = await fetch(`${API_BASE}/demo`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ num_products: 3, num_orders: 8, horizon_days: 60 }),
  });
  if (!res.ok) throw new Error('Failed to run MRP');
  return res.json();
};

const listRuns = async (): Promise<{ runs: Array<{ run_id: string; run_timestamp: string; items_processed: number }> }> => {
  const res = await fetch(`${API_BASE}/runs`);
  if (!res.ok) throw new Error('Failed to list runs');
  return res.json();
};

const getRunDetails = async (runId: string): Promise<MRPRunResult> => {
  const res = await fetch(`${API_BASE}/runs/${runId}`);
  if (!res.ok) throw new Error('Failed to get run');
  return res.json();
};

// ═══════════════════════════════════════════════════════════════════════════════
// COMPONENTS
// ═══════════════════════════════════════════════════════════════════════════════

const StatCard: React.FC<{ label: string; value: number | string; icon: React.ReactNode; color: string }> = ({
  label, value, icon, color,
}) => (
  <div className="bg-zinc-800/50 rounded-xl p-4 border border-zinc-700/50">
    <div className="flex items-center gap-3">
      <div className={`p-2 rounded-lg ${color}`}>{icon}</div>
      <div>
        <p className="text-2xl font-bold text-white">{value}</p>
        <p className="text-sm text-zinc-400">{label}</p>
      </div>
    </div>
  </div>
);

const OrderTypeBadge: React.FC<{ type: string }> = ({ type }) => {
  const isPurchase = type === 'purchase';
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 text-xs font-medium rounded ${
      isPurchase ? 'bg-blue-500/20 text-blue-400' : 'bg-purple-500/20 text-purple-400'
    }`}>
      {isPurchase ? <Truck className="w-3 h-3" /> : <Factory className="w-3 h-3" />}
      {isPurchase ? 'Compra' : 'Produção'}
    </span>
  );
};

const PlannedOrdersTable: React.FC<{ orders: PlannedOrder[]; title: string; icon: React.ReactNode }> = ({
  orders, title, icon,
}) => (
  <div className="bg-zinc-800/50 rounded-xl p-4 border border-zinc-700/50">
    <h3 className="font-semibold text-white mb-4 flex items-center gap-2">
      {icon}
      {title} ({orders.length})
    </h3>
    {orders.length === 0 ? (
      <p className="text-sm text-zinc-500 text-center py-4">Nenhuma ordem planeada</p>
    ) : (
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-zinc-500 border-b border-zinc-700">
              <th className="text-left py-2 px-2">Ordem</th>
              <th className="text-left py-2 px-2">SKU</th>
              <th className="text-right py-2 px-2">Qtd</th>
              <th className="text-left py-2 px-2">Início</th>
              <th className="text-left py-2 px-2">Entrega</th>
              <th className="text-right py-2 px-2">Lead Time</th>
            </tr>
          </thead>
          <tbody>
            {orders.slice(0, 10).map((order) => (
              <tr key={order.order_id} className="border-b border-zinc-800 hover:bg-zinc-800/50">
                <td className="py-2 px-2 font-mono text-xs text-zinc-400">{order.order_id}</td>
                <td className="py-2 px-2">
                  <span className="font-medium text-white">{order.sku}</span>
                </td>
                <td className="py-2 px-2 text-right font-medium text-white">{order.quantity.toLocaleString()}</td>
                <td className="py-2 px-2 text-zinc-400">{order.start_date.slice(0, 10)}</td>
                <td className="py-2 px-2 text-zinc-400">{order.due_date.slice(0, 10)}</td>
                <td className="py-2 px-2 text-right text-zinc-500">{order.lead_time_days}d</td>
              </tr>
            ))}
          </tbody>
        </table>
        {orders.length > 10 && (
          <p className="text-xs text-zinc-500 text-center py-2">
            + {orders.length - 10} mais ordens
          </p>
        )}
      </div>
    )}
  </div>
);

const ItemPlanCard: React.FC<{ plan: ItemPlan; expanded: boolean; onToggle: () => void }> = ({
  plan, expanded, onToggle,
}) => {
  // Prepare chart data
  const chartData = plan.periods.map((p) => ({
    period: new Date(p.period_start).toLocaleDateString('pt-PT', { month: 'short', day: 'numeric' }),
    gross: p.gross_requirements,
    poh: p.projected_on_hand,
    planned: p.planned_receipts,
  }));

  return (
    <div className="bg-zinc-800/50 rounded-xl border border-zinc-700/50 overflow-hidden">
      <div
        onClick={onToggle}
        className="cursor-pointer p-4 flex items-center justify-between hover:bg-zinc-800/80"
      >
        <div className="flex items-center gap-4">
          <div className="p-2 bg-purple-500/20 rounded-lg">
            <Package className="w-5 h-5 text-purple-400" />
          </div>
          <div>
            <h4 className="font-medium text-white">{plan.sku}</h4>
            <p className="text-sm text-zinc-400">{plan.name}</p>
          </div>
        </div>
        <div className="flex items-center gap-6">
          <div className="text-right">
            <p className="text-sm text-zinc-500">Necessidade</p>
            <p className="font-medium text-white">{plan.total_gross_requirement.toLocaleString()}</p>
          </div>
          <div className="text-right">
            <p className="text-sm text-zinc-500">Planeado</p>
            <p className="font-medium text-emerald-400">{plan.total_planned_quantity.toLocaleString()}</p>
          </div>
          <div className="text-right">
            <p className="text-sm text-zinc-500">Cobertura</p>
            <p className={`font-medium ${plan.coverage_days < 14 ? 'text-amber-400' : 'text-white'}`}>
              {plan.coverage_days.toFixed(0)}d
            </p>
          </div>
          {plan.stockout_risk && (
            <AlertTriangle className="w-5 h-5 text-red-400" />
          )}
          {expanded ? (
            <ChevronDown className="w-5 h-5 text-zinc-500" />
          ) : (
            <ChevronRight className="w-5 h-5 text-zinc-500" />
          )}
        </div>
      </div>
      
      {expanded && (
        <div className="p-4 border-t border-zinc-700">
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                <XAxis dataKey="period" tick={{ fill: '#888', fontSize: 10 }} />
                <YAxis tick={{ fill: '#888', fontSize: 10 }} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1f1f1f', border: '1px solid #333' }}
                  labelStyle={{ color: '#fff' }}
                />
                <Legend />
                <Bar dataKey="gross" name="Necessidade" fill="#ef4444" />
                <Bar dataKey="planned" name="Planeado" fill="#22c55e" />
                <Bar dataKey="poh" name="Stock Proj." fill="#8b5cf6" />
              </BarChart>
            </ResponsiveContainer>
          </div>
          
          {plan.planned_orders.length > 0 && (
            <div className="mt-4">
              <h5 className="text-sm font-medium text-zinc-400 mb-2">Ordens Planeadas</h5>
              <div className="space-y-1">
                {plan.planned_orders.map((order) => (
                  <div key={order.order_id} className="flex items-center justify-between text-sm p-2 bg-zinc-900/50 rounded">
                    <div className="flex items-center gap-2">
                      <OrderTypeBadge type={order.order_type} />
                      <span className="font-mono text-xs text-zinc-500">{order.order_id}</span>
                    </div>
                    <div className="flex items-center gap-4">
                      <span className="font-medium text-white">{order.quantity.toLocaleString()} un</span>
                      <span className="text-zinc-400">{order.due_date.slice(0, 10)}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN PAGE
// ═══════════════════════════════════════════════════════════════════════════════

const MRPDashboard: React.FC = () => {
  const [selectedRun, setSelectedRun] = useState<string | null>(null);
  const [expandedItems, setExpandedItems] = useState<Set<string>>(new Set());
  
  const queryClient = useQueryClient();

  const { data: status } = useQuery({ queryKey: ['mrp-status'], queryFn: fetchStatus });
  const { data: runsData } = useQuery({ queryKey: ['mrp-runs'], queryFn: listRuns });
  const { data: runDetails, isLoading: detailsLoading } = useQuery({
    queryKey: ['mrp-run', selectedRun],
    queryFn: () => (selectedRun ? getRunDetails(selectedRun) : Promise.resolve(null)),
    enabled: !!selectedRun,
  });

  const runMRPMutation = useMutation({
    mutationFn: runDemoMRP,
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['mrp-runs'] });
      setSelectedRun(data.run_id);
    },
  });

  const toggleItem = (sku: string) => {
    setExpandedItems((prev) => {
      const next = new Set(prev);
      if (next.has(sku)) {
        next.delete(sku);
      } else {
        next.add(sku);
      }
      return next;
    });
  };

  const runs = runsData?.runs || [];

  return (
    <div className="min-h-screen bg-gradient-to-br from-zinc-950 via-zinc-900 to-zinc-950 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-3xl font-bold text-white mb-2 flex items-center gap-3">
              <div className="p-2 bg-gradient-to-br from-emerald-500/20 to-cyan-500/20 rounded-xl">
                <Layers className="w-8 h-8 text-emerald-500" />
              </div>
              MRP: Material Requirements Planning
            </h1>
            <p className="text-zinc-400">Planeamento de necessidades com explosão de BOM e lot sizing</p>
          </div>
          <button
            onClick={() => runMRPMutation.mutate()}
            disabled={runMRPMutation.isPending}
            className="flex items-center gap-2 px-4 py-2 bg-emerald-500/20 hover:bg-emerald-500/30 text-emerald-400 rounded-lg border border-emerald-500/30"
          >
            {runMRPMutation.isPending ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
            Executar MRP Demo
          </button>
        </div>

        {/* Stats */}
        {runDetails && (
          <div className="grid grid-cols-6 gap-4 mb-6">
            <StatCard
              label="Items"
              value={runDetails.items_processed}
              icon={<Package className="w-5 h-5" />}
              color="bg-purple-500/20 text-purple-400"
            />
            <StatCard
              label="Demandas"
              value={runDetails.demands_processed}
              icon={<FileText className="w-5 h-5" />}
              color="bg-blue-500/20 text-blue-400"
            />
            <StatCard
              label="Níveis BOM"
              value={runDetails.bom_levels_exploded}
              icon={<Layers className="w-5 h-5" />}
              color="bg-cyan-500/20 text-cyan-400"
            />
            <StatCard
              label="Ordens Compra"
              value={runDetails.summary.total_purchase_orders}
              icon={<Truck className="w-5 h-5" />}
              color="bg-emerald-500/20 text-emerald-400"
            />
            <StatCard
              label="Ordens Produção"
              value={runDetails.summary.total_manufacture_orders}
              icon={<Factory className="w-5 h-5" />}
              color="bg-amber-500/20 text-amber-400"
            />
            <StatCard
              label="Alertas"
              value={runDetails.summary.total_shortage_alerts + runDetails.summary.total_capacity_alerts}
              icon={<AlertTriangle className="w-5 h-5" />}
              color={runDetails.summary.total_shortage_alerts > 0 ? "bg-red-500/20 text-red-400" : "bg-zinc-500/20 text-zinc-400"}
            />
          </div>
        )}

        {/* Main Content */}
        <div className="grid grid-cols-12 gap-6">
          {/* Run History */}
          <div className="col-span-3">
            <div className="bg-zinc-900/50 rounded-xl p-4 border border-zinc-800">
              <h3 className="font-semibold text-white mb-4 flex items-center gap-2">
                <Clock className="w-5 h-5 text-emerald-500" />
                Execuções MRP ({runs.length})
              </h3>
              {runs.length === 0 ? (
                <div className="text-center py-8 text-zinc-500">
                  <BarChart3 className="w-8 h-8 mx-auto mb-2 opacity-30" />
                  <p>Nenhuma execução</p>
                  <p className="text-sm">Execute MRP Demo</p>
                </div>
              ) : (
                <div className="space-y-2 max-h-[400px] overflow-y-auto">
                  {runs.map((run) => (
                    <div
                      key={run.run_id}
                      onClick={() => setSelectedRun(run.run_id)}
                      className={`cursor-pointer p-3 rounded-lg border transition-all ${
                        selectedRun === run.run_id
                          ? 'bg-emerald-500/10 border-emerald-500/30'
                          : 'bg-zinc-800/50 border-zinc-700/50 hover:border-zinc-600'
                      }`}
                    >
                      <span className="font-mono text-xs text-zinc-400">{run.run_id}</span>
                      <div className="flex items-center justify-between mt-1">
                        <span className="text-sm text-zinc-300">
                          {new Date(run.run_timestamp).toLocaleString('pt-PT', { dateStyle: 'short', timeStyle: 'short' })}
                        </span>
                        <span className="text-sm text-emerald-400">{run.items_processed} items</span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Detail Panel */}
          <div className="col-span-9 space-y-6">
            {detailsLoading ? (
              <div className="bg-zinc-900/50 rounded-xl p-8 border border-zinc-800 flex justify-center">
                <RefreshCw className="w-8 h-8 text-emerald-500 animate-spin" />
              </div>
            ) : runDetails ? (
              <>
                {/* Planned Orders */}
                <div className="grid grid-cols-2 gap-6">
                  <PlannedOrdersTable
                    orders={runDetails.purchase_orders}
                    title="Ordens de Compra"
                    icon={<Truck className="w-5 h-5 text-blue-500" />}
                  />
                  <PlannedOrdersTable
                    orders={runDetails.manufacture_orders}
                    title="Ordens de Produção"
                    icon={<Factory className="w-5 h-5 text-purple-500" />}
                  />
                </div>

                {/* Item Plans */}
                <div className="bg-zinc-900/50 rounded-xl p-4 border border-zinc-800">
                  <h3 className="font-semibold text-white mb-4 flex items-center gap-2">
                    <Package className="w-5 h-5 text-purple-500" />
                    Planos por Item
                  </h3>
                  <div className="space-y-3">
                    {Object.values(runDetails.item_plans).map((plan) => (
                      <ItemPlanCard
                        key={plan.sku}
                        plan={plan}
                        expanded={expandedItems.has(plan.sku)}
                        onToggle={() => toggleItem(plan.sku)}
                      />
                    ))}
                  </div>
                </div>

                {/* Alerts */}
                {(runDetails.shortage_alerts.length > 0 || runDetails.capacity_alerts.length > 0) && (
                  <div className="bg-red-500/10 rounded-xl p-4 border border-red-500/30">
                    <h3 className="font-semibold text-red-400 mb-4 flex items-center gap-2">
                      <AlertTriangle className="w-5 h-5" />
                      Alertas
                    </h3>
                    <div className="space-y-2">
                      {runDetails.shortage_alerts.map((alert, idx) => (
                        <div key={idx} className="flex items-center gap-2 text-sm text-red-300">
                          <span className="font-medium">{alert.sku}</span>
                          <span className="text-red-400/70">- Risco de stockout em {alert.stockout_periods.length} períodos</span>
                        </div>
                      ))}
                      {runDetails.capacity_alerts.map((alert, idx) => (
                        <div key={idx} className="flex items-center gap-2 text-sm text-amber-300">
                          <span className="font-medium">{alert.work_center}</span>
                          <span className="text-amber-400/70">- Sobrecarga de {alert.overload_hours}h</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </>
            ) : (
              <div className="bg-zinc-900/50 rounded-xl p-12 border border-zinc-800 text-center text-zinc-500">
                <ChevronRight className="w-12 h-12 mx-auto mb-2 opacity-30" />
                <p>Selecione uma execução ou execute um novo MRP</p>
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="mt-8 text-center text-xs text-zinc-600">
          <p>MRP explode BOM multi-nível, calcula necessidades líquidas e gera ordens planeadas com MOQ, múltiplos e scrap.</p>
          <p className="mt-1">R&D / SIFIDE: WP3 - Inventory & Capacity Optimization</p>
        </div>
      </div>
    </div>
  );
};

export default MRPDashboard;



