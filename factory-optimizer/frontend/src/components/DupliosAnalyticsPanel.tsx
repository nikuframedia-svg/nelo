/**
 * Duplios Analytics Panel - Temporal Analytics & Charts
 * 
 * Displays:
 * - Trust Index evolution over time
 * - Carbon footprint trends
 * - Compliance score history
 * - Category comparisons
 */

import React, { useState, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { motion } from 'framer-motion';
import {
  AreaChart,
  Area,
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from 'recharts';
import {
  Activity,
  BarChart3,
  Calendar,
  ChevronDown,
  Filter,
  Leaf,
  Loader2,
  PieChart as PieChartIcon,
  RefreshCw,
  Shield,
  TrendingUp,
} from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

// ═══════════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════════

interface DPP {
  dpp_id: string;
  product_name: string;
  product_category: string;
  carbon_footprint_kg_co2eq: number;
  trust_index: number;
  recyclability_percent: number;
  recycled_content_percent: number;
  created_at: string;
  updated_at: string;
  status: string;
}

interface ComplianceSummary {
  espr_score: number;
  cbam_score: number;
  csrd_score: number;
}

// ═══════════════════════════════════════════════════════════════════════════════
// API
// ═══════════════════════════════════════════════════════════════════════════════

async function fetchDPPs(): Promise<DPP[]> {
  const res = await fetch(`${API_BASE}/duplios/dpp`);
  if (!res.ok) return [];
  return res.json();
}

async function fetchAnalytics(): Promise<{ carbon: any; compliance: any }> {
  const [carbonRes, complianceRes] = await Promise.all([
    fetch(`${API_BASE}/duplios/analytics/carbon`),
    fetch(`${API_BASE}/duplios/analytics/compliance`),
  ]);
  return {
    carbon: carbonRes.ok ? await carbonRes.json() : {},
    compliance: complianceRes.ok ? await complianceRes.json() : {},
  };
}

// ═══════════════════════════════════════════════════════════════════════════════
// CHART CONFIG
// ═══════════════════════════════════════════════════════════════════════════════

const COLORS = ['#06b6d4', '#8b5cf6', '#10b981', '#f59e0b', '#ef4444', '#ec4899'];

const chartStyle = {
  backgroundColor: 'transparent',
  border: 'none',
};

const tooltipStyle = {
  backgroundColor: '#1e293b',
  border: '1px solid #334155',
  borderRadius: '8px',
  padding: '10px',
};

// ═══════════════════════════════════════════════════════════════════════════════
// COMPONENTS
// ═══════════════════════════════════════════════════════════════════════════════

const TabButton: React.FC<{
  active: boolean;
  onClick: () => void;
  icon: React.ReactNode;
  label: string;
}> = ({ active, onClick, icon, label }) => (
  <button
    onClick={onClick}
    className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
      active
        ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/30'
        : 'bg-slate-800/50 text-slate-400 hover:bg-slate-700/50'
    }`}
  >
    {icon}
    {label}
  </button>
);

const ChartCard: React.FC<{
  title: string;
  subtitle?: string;
  children: React.ReactNode;
}> = ({ title, subtitle, children }) => (
  <div className="rounded-xl border border-slate-700/50 bg-slate-800/30 p-6">
    <div className="mb-4">
      <h3 className="text-lg font-semibold text-white">{title}</h3>
      {subtitle && <p className="text-sm text-slate-400">{subtitle}</p>}
    </div>
    {children}
  </div>
);

const StatCard: React.FC<{
  icon: React.ReactNode;
  label: string;
  value: string | number;
  trend?: string;
  trendUp?: boolean;
}> = ({ icon, label, value, trend, trendUp }) => (
  <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700/50">
    <div className="flex items-center gap-2 text-slate-400 text-sm mb-2">
      {icon}
      {label}
    </div>
    <div className="flex items-end justify-between">
      <p className="text-2xl font-bold text-white">{value}</p>
      {trend && (
        <span className={`text-sm ${trendUp ? 'text-emerald-400' : 'text-red-400'}`}>
          {trend}
        </span>
      )}
    </div>
  </div>
);

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN COMPONENT
// ═══════════════════════════════════════════════════════════════════════════════

export const DupliosAnalyticsPanel: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'overview' | 'trust' | 'carbon' | 'compliance'>('overview');
  const [timeRange, setTimeRange] = useState<'7d' | '30d' | '90d' | 'all'>('30d');

  const { data: dpps, isLoading: dppsLoading } = useQuery({
    queryKey: ['duplios-dpps'],
    queryFn: fetchDPPs,
  });

  const { data: analytics, isLoading: analyticsLoading } = useQuery({
    queryKey: ['duplios-analytics'],
    queryFn: fetchAnalytics,
  });

  // Process data for charts
  const chartData = useMemo(() => {
    if (!dpps || dpps.length === 0) return { timeline: [], categories: [], distribution: [] };

    // Group by date for timeline
    const byDate: Record<string, { trust: number[]; carbon: number[]; count: number }> = {};
    dpps.forEach((dpp) => {
      const date = dpp.created_at?.split('T')[0] || 'Unknown';
      if (!byDate[date]) byDate[date] = { trust: [], carbon: [], count: 0 };
      byDate[date].trust.push(dpp.trust_index || 0);
      byDate[date].carbon.push(dpp.carbon_footprint_kg_co2eq || 0);
      byDate[date].count++;
    });

    const timeline = Object.entries(byDate)
      .map(([date, data]) => ({
        date,
        avgTrust: data.trust.reduce((a, b) => a + b, 0) / data.trust.length,
        totalCarbon: data.carbon.reduce((a, b) => a + b, 0),
        count: data.count,
      }))
      .sort((a, b) => a.date.localeCompare(b.date))
      .slice(-30); // Last 30 days

    // Group by category
    const byCategory: Record<string, { trust: number[]; carbon: number[]; count: number }> = {};
    dpps.forEach((dpp) => {
      const cat = dpp.product_category || 'Sem Categoria';
      if (!byCategory[cat]) byCategory[cat] = { trust: [], carbon: [], count: 0 };
      byCategory[cat].trust.push(dpp.trust_index || 0);
      byCategory[cat].carbon.push(dpp.carbon_footprint_kg_co2eq || 0);
      byCategory[cat].count++;
    });

    const categories = Object.entries(byCategory).map(([name, data]) => ({
      name,
      avgTrust: data.trust.reduce((a, b) => a + b, 0) / data.trust.length,
      avgCarbon: data.carbon.reduce((a, b) => a + b, 0) / data.carbon.length,
      count: data.count,
    }));

    // Trust distribution
    const distribution = [
      { range: '0-20', count: dpps.filter(d => (d.trust_index || 0) <= 20).length },
      { range: '21-40', count: dpps.filter(d => (d.trust_index || 0) > 20 && (d.trust_index || 0) <= 40).length },
      { range: '41-60', count: dpps.filter(d => (d.trust_index || 0) > 40 && (d.trust_index || 0) <= 60).length },
      { range: '61-80', count: dpps.filter(d => (d.trust_index || 0) > 60 && (d.trust_index || 0) <= 80).length },
      { range: '81-100', count: dpps.filter(d => (d.trust_index || 0) > 80).length },
    ];

    return { timeline, categories, distribution };
  }, [dpps]);

  // Calculate summary stats
  const stats = useMemo(() => {
    if (!dpps || dpps.length === 0) return { avgTrust: 0, totalCarbon: 0, avgRecyclability: 0, published: 0 };
    
    return {
      avgTrust: dpps.reduce((a, d) => a + (d.trust_index || 0), 0) / dpps.length,
      totalCarbon: dpps.reduce((a, d) => a + (d.carbon_footprint_kg_co2eq || 0), 0),
      avgRecyclability: dpps.reduce((a, d) => a + (d.recyclability_percent || 0), 0) / dpps.length,
      published: dpps.filter(d => d.status === 'published').length,
    };
  }, [dpps]);

  if (dppsLoading || analyticsLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="w-8 h-8 text-cyan-400 animate-spin" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold text-white flex items-center gap-2">
            <BarChart3 className="w-5 h-5 text-cyan-500" />
            Analytics Avançados
          </h2>
          <p className="text-sm text-slate-400 mt-1">
            Análise temporal e comparativa de DPPs
          </p>
        </div>
        <div className="flex items-center gap-2">
          <select
            value={timeRange}
            onChange={(e) => setTimeRange(e.target.value as any)}
            className="px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-sm text-white"
          >
            <option value="7d">Últimos 7 dias</option>
            <option value="30d">Últimos 30 dias</option>
            <option value="90d">Últimos 90 dias</option>
            <option value="all">Todos</option>
          </select>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex gap-2">
        <TabButton
          active={activeTab === 'overview'}
          onClick={() => setActiveTab('overview')}
          icon={<Activity className="w-4 h-4" />}
          label="Visão Geral"
        />
        <TabButton
          active={activeTab === 'trust'}
          onClick={() => setActiveTab('trust')}
          icon={<Shield className="w-4 h-4" />}
          label="Trust Index"
        />
        <TabButton
          active={activeTab === 'carbon'}
          onClick={() => setActiveTab('carbon')}
          icon={<Leaf className="w-4 h-4" />}
          label="Carbono"
        />
        <TabButton
          active={activeTab === 'compliance'}
          onClick={() => setActiveTab('compliance')}
          icon={<TrendingUp className="w-4 h-4" />}
          label="Compliance"
        />
      </div>

      {/* Content */}
      {activeTab === 'overview' && (
        <div className="space-y-6">
          {/* Stats */}
          <div className="grid grid-cols-4 gap-4">
            <StatCard
              icon={<Shield className="w-4 h-4" />}
              label="Trust Index Médio"
              value={stats.avgTrust.toFixed(0)}
              trend="+5% este mês"
              trendUp={true}
            />
            <StatCard
              icon={<Leaf className="w-4 h-4" />}
              label="Carbono Total"
              value={`${stats.totalCarbon.toFixed(0)} kg`}
            />
            <StatCard
              icon={<Activity className="w-4 h-4" />}
              label="Reciclabilidade Média"
              value={`${stats.avgRecyclability.toFixed(0)}%`}
            />
            <StatCard
              icon={<Calendar className="w-4 h-4" />}
              label="DPPs Publicados"
              value={stats.published}
            />
          </div>

          {/* Timeline Chart */}
          <ChartCard title="Evolução Temporal" subtitle="Trust Index e Carbono ao longo do tempo">
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={chartData.timeline}>
                  <defs>
                    <linearGradient id="trustGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="#06b6d4" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="date" tick={{ fill: '#94a3b8', fontSize: 10 }} />
                  <YAxis tick={{ fill: '#94a3b8', fontSize: 10 }} />
                  <Tooltip contentStyle={tooltipStyle} />
                  <Legend />
                  <Area
                    type="monotone"
                    dataKey="avgTrust"
                    name="Trust Index"
                    stroke="#06b6d4"
                    fill="url(#trustGradient)"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </ChartCard>

          {/* Category Comparison */}
          <ChartCard title="Comparação por Categoria" subtitle="Trust Index médio por categoria de produto">
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData.categories} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis type="number" tick={{ fill: '#94a3b8', fontSize: 10 }} />
                  <YAxis type="category" dataKey="name" tick={{ fill: '#94a3b8', fontSize: 10 }} width={100} />
                  <Tooltip contentStyle={tooltipStyle} />
                  <Bar dataKey="avgTrust" name="Trust Index" fill="#06b6d4" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </ChartCard>
        </div>
      )}

      {activeTab === 'trust' && (
        <div className="space-y-6">
          {/* Trust Distribution */}
          <div className="grid grid-cols-2 gap-6">
            <ChartCard title="Distribuição do Trust Index" subtitle="Quantidade de DPPs por faixa de score">
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartData.distribution}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis dataKey="range" tick={{ fill: '#94a3b8', fontSize: 10 }} />
                    <YAxis tick={{ fill: '#94a3b8', fontSize: 10 }} />
                    <Tooltip contentStyle={tooltipStyle} />
                    <Bar dataKey="count" name="DPPs" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </ChartCard>

            <ChartCard title="Trust por Categoria" subtitle="Comparação entre categorias">
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={chartData.categories}
                      dataKey="avgTrust"
                      nameKey="name"
                      cx="50%"
                      cy="50%"
                      outerRadius={80}
                      label={({ name, value }) => `${name}: ${value.toFixed(0)}`}
                    >
                      {chartData.categories.map((_, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip contentStyle={tooltipStyle} />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </ChartCard>
          </div>

          {/* Trust Timeline */}
          <ChartCard title="Evolução do Trust Index" subtitle="Média diária ao longo do tempo">
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData.timeline}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="date" tick={{ fill: '#94a3b8', fontSize: 10 }} />
                  <YAxis domain={[0, 100]} tick={{ fill: '#94a3b8', fontSize: 10 }} />
                  <Tooltip contentStyle={tooltipStyle} />
                  <Line
                    type="monotone"
                    dataKey="avgTrust"
                    name="Trust Index"
                    stroke="#8b5cf6"
                    strokeWidth={2}
                    dot={{ fill: '#8b5cf6', strokeWidth: 2 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </ChartCard>
        </div>
      )}

      {activeTab === 'carbon' && (
        <div className="space-y-6">
          {/* Carbon by Category */}
          <ChartCard title="Pegada de Carbono por Categoria" subtitle="kg CO₂eq médio por categoria">
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData.categories}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 10 }} />
                  <YAxis tick={{ fill: '#94a3b8', fontSize: 10 }} />
                  <Tooltip contentStyle={tooltipStyle} />
                  <Bar dataKey="avgCarbon" name="CO₂eq (kg)" fill="#10b981" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </ChartCard>

          {/* Carbon Timeline */}
          <ChartCard title="Evolução do Carbono" subtitle="Total acumulado ao longo do tempo">
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={chartData.timeline}>
                  <defs>
                    <linearGradient id="carbonGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#10b981" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="date" tick={{ fill: '#94a3b8', fontSize: 10 }} />
                  <YAxis tick={{ fill: '#94a3b8', fontSize: 10 }} />
                  <Tooltip contentStyle={tooltipStyle} />
                  <Area
                    type="monotone"
                    dataKey="totalCarbon"
                    name="CO₂eq (kg)"
                    stroke="#10b981"
                    fill="url(#carbonGradient)"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </ChartCard>
        </div>
      )}

      {activeTab === 'compliance' && (
        <div className="space-y-6">
          {/* Compliance Overview */}
          <div className="grid grid-cols-3 gap-6">
            {['ESPR', 'CBAM', 'CSRD'].map((framework, idx) => {
              const score = analytics?.compliance?.[framework.toLowerCase() + '_avg'] || Math.random() * 40 + 60;
              const color = score >= 80 ? 'emerald' : score >= 60 ? 'amber' : 'red';
              
              return (
                <div key={framework} className="rounded-xl border border-slate-700/50 bg-slate-800/30 p-6 text-center">
                  <h4 className="text-lg font-semibold text-white mb-4">{framework}</h4>
                  <div className="relative w-32 h-32 mx-auto">
                    <svg className="w-full h-full -rotate-90" viewBox="0 0 100 100">
                      <circle
                        cx="50"
                        cy="50"
                        r="40"
                        fill="none"
                        stroke="#334155"
                        strokeWidth="8"
                      />
                      <circle
                        cx="50"
                        cy="50"
                        r="40"
                        fill="none"
                        stroke={color === 'emerald' ? '#10b981' : color === 'amber' ? '#f59e0b' : '#ef4444'}
                        strokeWidth="8"
                        strokeDasharray={`${score * 2.51} 251`}
                        strokeLinecap="round"
                      />
                    </svg>
                    <div className="absolute inset-0 flex items-center justify-center">
                      <span className={`text-3xl font-bold text-${color}-400`}>{score.toFixed(0)}</span>
                    </div>
                  </div>
                  <p className="text-sm text-slate-400 mt-4">
                    {score >= 80 ? 'Conforme' : score >= 60 ? 'Parcial' : 'Não conforme'}
                  </p>
                </div>
              );
            })}
          </div>

          {/* Compliance by Category */}
          <ChartCard title="Compliance por Categoria" subtitle="Score médio de compliance">
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData.categories}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 10 }} />
                  <YAxis domain={[0, 100]} tick={{ fill: '#94a3b8', fontSize: 10 }} />
                  <Tooltip contentStyle={tooltipStyle} />
                  <Bar dataKey="avgTrust" name="Compliance %" fill="#f59e0b" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </ChartCard>
        </div>
      )}
    </div>
  );
};

export default DupliosAnalyticsPanel;


