/**
 * ═══════════════════════════════════════════════════════════════════════════════
 * PDM Dashboard - Product Data Management
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * Dashboard for managing product data:
 * - Items with revisions
 * - Bill of Materials (BOM)
 * - Manufacturing Routing
 * - Engineering Change Requests (ECR)
 *
 * Features:
 * - Item browser with search/filter
 * - Revision workflow (Draft → Released → Obsolete)
 * - BOM explosion view
 * - Release validation
 * - ECR management
 *
 * R&D / SIFIDE: WP1 - PLM/PDM Core
 */

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  AlertCircle,
  Archive,
  Box,
  Check,
  ChevronRight,
  ClipboardList,
  Clock,
  Edit3,
  FileText,
  Folder,
  FolderOpen,
  GitBranch,
  Layers,
  List,
  Package,
  Play,
  Plus,
  RefreshCw,
  Search,
  Settings,
  Shield,
  Tag,
  Trash2,
  Upload,
  XCircle,
  Zap,
} from 'lucide-react';

// ═══════════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════════

interface Item {
  id: number;
  sku: string;
  name: string;
  type: string;
  unit: string;
  family?: string;
  weight_kg?: number;
  current_revision?: string;
}

interface Revision {
  id: number;
  item_id: number;
  code: string;
  status: string;
  effective_from?: string;
  effective_to?: string;
  notes?: string;
  created_at: string;
}

interface BomLine {
  id: number;
  parent_revision_id: number;
  component_revision_id: number;
  component_sku: string;
  component_name: string;
  component_revision_code: string;
  qty_per_unit: number;
  scrap_rate: number;
  position?: string;
  unit: string;
}

interface RoutingOperation {
  id: number;
  revision_id: number;
  sequence: number;
  op_code: string;
  name?: string;
  machine_group?: string;
  setup_time: number;
  cycle_time: number;
  tool_id?: string;
  is_critical: boolean;
  requires_inspection: boolean;
}

interface ValidationResult {
  valid: boolean;
  errors_count: number;
  warnings_count: number;
  issues: Array<{
    code: string;
    message: string;
    severity: string;
    field?: string;
  }>;
}

interface ECR {
  id: number;
  item_id: number;
  title: string;
  description: string;
  reason?: string;
  priority: string;
  status: string;
  requested_by?: string;
  requested_at?: string;
}

interface PDMStatus {
  statistics: {
    total_items: number;
    total_revisions: number;
    released_revisions: number;
    draft_revisions: number;
    open_ecrs: number;
  };
}

// ═══════════════════════════════════════════════════════════════════════════════
// API HOOKS
// ═══════════════════════════════════════════════════════════════════════════════

const API_BASE = 'http://127.0.0.1:8000/pdm';

const fetchStatus = async (): Promise<PDMStatus> => {
  const res = await fetch(`${API_BASE}/status`);
  if (!res.ok) throw new Error('Failed to fetch status');
  return res.json();
};

const fetchItems = async (search?: string): Promise<Item[]> => {
  const url = search ? `${API_BASE}/items?search=${encodeURIComponent(search)}` : `${API_BASE}/items`;
  const res = await fetch(url);
  if (!res.ok) throw new Error('Failed to fetch items');
  return res.json();
};

const fetchRevisions = async (itemId: number): Promise<Revision[]> => {
  const res = await fetch(`${API_BASE}/items/${itemId}/revisions`);
  if (!res.ok) throw new Error('Failed to fetch revisions');
  return res.json();
};

const fetchBom = async (revisionId: number): Promise<BomLine[]> => {
  const res = await fetch(`${API_BASE}/revisions/${revisionId}/bom`);
  if (!res.ok) throw new Error('Failed to fetch BOM');
  return res.json();
};

const fetchRouting = async (revisionId: number): Promise<RoutingOperation[]> => {
  const res = await fetch(`${API_BASE}/revisions/${revisionId}/routing`);
  if (!res.ok) throw new Error('Failed to fetch routing');
  return res.json();
};

const validateRevision = async (revisionId: number): Promise<ValidationResult> => {
  const res = await fetch(`${API_BASE}/revisions/${revisionId}/validate`);
  if (!res.ok) throw new Error('Failed to validate');
  return res.json();
};

const releaseRevision = async (revisionId: number, force: boolean = false): Promise<any> => {
  const res = await fetch(`${API_BASE}/revisions/${revisionId}/release`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ released_by: 'user', force }),
  });
  if (!res.ok) throw new Error('Failed to release');
  return res.json();
};

const fetchEcrs = async (): Promise<ECR[]> => {
  const res = await fetch(`${API_BASE}/ecr`);
  if (!res.ok) throw new Error('Failed to fetch ECRs');
  return res.json();
};

const seedDemoData = async (): Promise<any> => {
  const res = await fetch(`${API_BASE}/demo/seed`, { method: 'POST' });
  if (!res.ok) throw new Error('Failed to seed');
  return res.json();
};

// ═══════════════════════════════════════════════════════════════════════════════
// COMPONENTS
// ═══════════════════════════════════════════════════════════════════════════════

const StatusBadge: React.FC<{ status: string }> = ({ status }) => {
  const colors: Record<string, string> = {
    DRAFT: 'bg-amber-500/20 text-amber-400 border-amber-500/30',
    RELEASED: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30',
    OBSOLETE: 'bg-zinc-500/20 text-zinc-400 border-zinc-500/30',
    OPEN: 'bg-blue-500/20 text-blue-400 border-blue-500/30',
    CLOSED: 'bg-zinc-500/20 text-zinc-400 border-zinc-500/30',
  };

  const icons: Record<string, React.ReactNode> = {
    DRAFT: <Edit3 className="w-3 h-3" />,
    RELEASED: <Check className="w-3 h-3" />,
    OBSOLETE: <Archive className="w-3 h-3" />,
  };

  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 text-xs font-medium rounded border ${colors[status] || colors.DRAFT}`}>
      {icons[status]}
      {status}
    </span>
  );
};

const TypeBadge: React.FC<{ type: string }> = ({ type }) => {
  const colors: Record<string, string> = {
    FINISHED: 'bg-purple-500/20 text-purple-400',
    SEMI_FINISHED: 'bg-blue-500/20 text-blue-400',
    RAW_MATERIAL: 'bg-orange-500/20 text-orange-400',
    TOOLING: 'bg-cyan-500/20 text-cyan-400',
    PACKAGING: 'bg-green-500/20 text-green-400',
  };

  const labels: Record<string, string> = {
    FINISHED: 'Produto',
    SEMI_FINISHED: 'Semi-Acabado',
    RAW_MATERIAL: 'Matéria-Prima',
    TOOLING: 'Ferramenta',
    PACKAGING: 'Embalagem',
  };

  return (
    <span className={`px-2 py-0.5 text-xs font-medium rounded ${colors[type] || 'bg-zinc-500/20 text-zinc-400'}`}>
      {labels[type] || type}
    </span>
  );
};

const StatCard: React.FC<{ label: string; value: number; icon: React.ReactNode; color: string }> = ({
  label,
  value,
  icon,
  color,
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

const ItemList: React.FC<{
  items: Item[];
  selectedId?: number;
  onSelect: (item: Item) => void;
}> = ({ items, selectedId, onSelect }) => (
  <div className="space-y-1 max-h-[500px] overflow-y-auto">
    {items.map((item) => (
      <motion.div
        key={item.id}
        whileHover={{ scale: 1.01 }}
        onClick={() => onSelect(item)}
        className={`cursor-pointer p-3 rounded-lg border transition-all ${
          selectedId === item.id
            ? 'bg-purple-500/10 border-purple-500/30'
            : 'bg-zinc-800/50 border-zinc-700/50 hover:border-zinc-600'
        }`}
      >
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-2">
            <Package className="w-4 h-4 text-zinc-500" />
            <span className="font-mono text-sm text-purple-400">{item.sku}</span>
          </div>
          <TypeBadge type={item.type} />
        </div>
        <p className="text-sm text-white mt-1 truncate">{item.name}</p>
        <div className="flex items-center gap-2 mt-2 text-xs text-zinc-500">
          <span>{item.unit}</span>
          {item.current_revision && (
            <>
              <span>•</span>
              <span className="text-emerald-400">Rev {item.current_revision}</span>
            </>
          )}
        </div>
      </motion.div>
    ))}
  </div>
);

const RevisionPanel: React.FC<{
  item: Item;
  revisions: Revision[];
  selectedRevision?: Revision;
  onSelectRevision: (rev: Revision) => void;
  onRelease: (revId: number) => void;
}> = ({ item, revisions, selectedRevision, onSelectRevision, onRelease }) => (
  <div className="bg-zinc-800/50 rounded-xl p-4 border border-zinc-700/50">
    <h3 className="font-semibold text-white mb-4 flex items-center gap-2">
      <GitBranch className="w-5 h-5 text-purple-500" />
      Revisões de {item.sku}
    </h3>
    <div className="space-y-2">
      {revisions.map((rev) => (
        <div
          key={rev.id}
          onClick={() => onSelectRevision(rev)}
          className={`cursor-pointer p-3 rounded-lg border transition-all ${
            selectedRevision?.id === rev.id
              ? 'bg-purple-500/10 border-purple-500/30'
              : 'bg-zinc-900/50 border-zinc-700/50 hover:border-zinc-600'
          }`}
        >
          <div className="flex items-center justify-between">
            <span className="font-mono text-lg text-white">Rev {rev.code}</span>
            <StatusBadge status={rev.status} />
          </div>
          <p className="text-xs text-zinc-500 mt-1">
            {rev.effective_from
              ? `Efetivo desde ${new Date(rev.effective_from).toLocaleDateString()}`
              : 'Não liberado'}
          </p>
          {rev.status === 'DRAFT' && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                onRelease(rev.id);
              }}
              className="mt-2 text-xs px-2 py-1 bg-emerald-500/20 text-emerald-400 rounded hover:bg-emerald-500/30"
            >
              Liberar Revisão
            </button>
          )}
        </div>
      ))}
    </div>
  </div>
);

const BomPanel: React.FC<{ bom: BomLine[]; loading: boolean }> = ({ bom, loading }) => (
  <div className="bg-zinc-800/50 rounded-xl p-4 border border-zinc-700/50">
    <h3 className="font-semibold text-white mb-4 flex items-center gap-2">
      <Layers className="w-5 h-5 text-blue-500" />
      Lista de Materiais (BOM)
    </h3>
    {loading ? (
      <div className="flex justify-center py-4">
        <RefreshCw className="w-5 h-5 text-zinc-500 animate-spin" />
      </div>
    ) : bom.length === 0 ? (
      <p className="text-sm text-zinc-500 text-center py-4">Nenhum componente na BOM</p>
    ) : (
      <div className="space-y-2">
        {bom.map((line) => (
          <div key={line.id} className="flex items-center gap-3 p-2 bg-zinc-900/50 rounded-lg">
            <Box className="w-4 h-4 text-blue-400" />
            <div className="flex-1">
              <div className="flex items-center gap-2">
                <span className="font-mono text-sm text-white">{line.component_sku}</span>
                <span className="text-xs text-zinc-500">Rev {line.component_revision_code}</span>
              </div>
              <p className="text-xs text-zinc-400">{line.component_name}</p>
            </div>
            <div className="text-right">
              <p className="text-sm font-medium text-white">
                {line.qty_per_unit} {line.unit}
              </p>
              {line.scrap_rate > 0 && (
                <p className="text-xs text-amber-400">+{(line.scrap_rate * 100).toFixed(0)}% scrap</p>
              )}
            </div>
          </div>
        ))}
      </div>
    )}
  </div>
);

const RoutingPanel: React.FC<{ routing: RoutingOperation[]; loading: boolean }> = ({ routing, loading }) => (
  <div className="bg-zinc-800/50 rounded-xl p-4 border border-zinc-700/50">
    <h3 className="font-semibold text-white mb-4 flex items-center gap-2">
      <ClipboardList className="w-5 h-5 text-cyan-500" />
      Roteiro de Fabricação
    </h3>
    {loading ? (
      <div className="flex justify-center py-4">
        <RefreshCw className="w-5 h-5 text-zinc-500 animate-spin" />
      </div>
    ) : routing.length === 0 ? (
      <p className="text-sm text-zinc-500 text-center py-4">Nenhuma operação no roteiro</p>
    ) : (
      <div className="space-y-2">
        {routing.map((op, idx) => (
          <div key={op.id} className="flex items-center gap-3 p-2 bg-zinc-900/50 rounded-lg">
            <div className="w-8 h-8 rounded-full bg-cyan-500/20 flex items-center justify-center text-cyan-400 font-mono text-sm">
              {op.sequence}
            </div>
            <div className="flex-1">
              <div className="flex items-center gap-2">
                <span className="font-medium text-white">{op.op_code}</span>
                {op.is_critical && (
                  <span className="px-1 py-0.5 text-xs bg-red-500/20 text-red-400 rounded">Crítica</span>
                )}
                {op.requires_inspection && (
                  <span className="px-1 py-0.5 text-xs bg-blue-500/20 text-blue-400 rounded">Inspeção</span>
                )}
              </div>
              <p className="text-xs text-zinc-400">{op.name || op.machine_group}</p>
            </div>
            <div className="text-right text-xs text-zinc-500">
              <p>Setup: {op.setup_time}min</p>
              <p>Ciclo: {op.cycle_time}min</p>
            </div>
          </div>
        ))}
      </div>
    )}
  </div>
);

const ValidationPanel: React.FC<{ validation: ValidationResult | null; loading: boolean }> = ({
  validation,
  loading,
}) => {
  if (loading) {
    return (
      <div className="bg-zinc-800/50 rounded-xl p-4 border border-zinc-700/50 flex justify-center">
        <RefreshCw className="w-5 h-5 text-zinc-500 animate-spin" />
      </div>
    );
  }

  if (!validation) return null;

  return (
    <div
      className={`rounded-xl p-4 border ${
        validation.valid
          ? 'bg-emerald-500/10 border-emerald-500/30'
          : 'bg-red-500/10 border-red-500/30'
      }`}
    >
      <div className="flex items-center gap-2 mb-3">
        {validation.valid ? (
          <>
            <Check className="w-5 h-5 text-emerald-400" />
            <span className="font-medium text-emerald-400">Pronto para Liberação</span>
          </>
        ) : (
          <>
            <XCircle className="w-5 h-5 text-red-400" />
            <span className="font-medium text-red-400">
              {validation.errors_count} erro(s), {validation.warnings_count} aviso(s)
            </span>
          </>
        )}
      </div>
      {validation.issues.length > 0 && (
        <div className="space-y-1">
          {validation.issues.map((issue, idx) => (
            <div
              key={idx}
              className={`text-sm p-2 rounded ${
                issue.severity === 'error' ? 'bg-red-500/10 text-red-300' : 'bg-amber-500/10 text-amber-300'
              }`}
            >
              <span className="font-mono text-xs opacity-70">[{issue.code}]</span> {issue.message}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

const ECRList: React.FC<{ ecrs: ECR[] }> = ({ ecrs }) => (
  <div className="space-y-2">
    {ecrs.map((ecr) => (
      <div key={ecr.id} className="p-3 bg-zinc-800/50 rounded-lg border border-zinc-700/50">
        <div className="flex items-start justify-between">
          <div>
            <span className="font-mono text-xs text-zinc-500">ECR-{ecr.id}</span>
            <h4 className="font-medium text-white">{ecr.title}</h4>
          </div>
          <div className="flex items-center gap-2">
            <span
              className={`px-2 py-0.5 text-xs rounded ${
                ecr.priority === 'HIGH' || ecr.priority === 'CRITICAL'
                  ? 'bg-red-500/20 text-red-400'
                  : ecr.priority === 'MEDIUM'
                  ? 'bg-amber-500/20 text-amber-400'
                  : 'bg-green-500/20 text-green-400'
              }`}
            >
              {ecr.priority}
            </span>
            <StatusBadge status={ecr.status} />
          </div>
        </div>
        <p className="text-sm text-zinc-400 mt-1">{ecr.description}</p>
      </div>
    ))}
  </div>
);

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN PAGE
// ═══════════════════════════════════════════════════════════════════════════════

const PDMDashboard: React.FC = () => {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedItem, setSelectedItem] = useState<Item | null>(null);
  const [selectedRevision, setSelectedRevision] = useState<Revision | null>(null);
  const [activeTab, setActiveTab] = useState<'bom' | 'routing' | 'ecr'>('bom');

  const queryClient = useQueryClient();

  // Queries
  const { data: status } = useQuery({ queryKey: ['pdm-status'], queryFn: fetchStatus });
  const { data: items = [], isLoading: itemsLoading } = useQuery({
    queryKey: ['pdm-items', searchTerm],
    queryFn: () => fetchItems(searchTerm || undefined),
  });
  const { data: revisions = [] } = useQuery({
    queryKey: ['pdm-revisions', selectedItem?.id],
    queryFn: () => (selectedItem ? fetchRevisions(selectedItem.id) : Promise.resolve([])),
    enabled: !!selectedItem,
  });
  const { data: bom = [], isLoading: bomLoading } = useQuery({
    queryKey: ['pdm-bom', selectedRevision?.id],
    queryFn: () => (selectedRevision ? fetchBom(selectedRevision.id) : Promise.resolve([])),
    enabled: !!selectedRevision,
  });
  const { data: routing = [], isLoading: routingLoading } = useQuery({
    queryKey: ['pdm-routing', selectedRevision?.id],
    queryFn: () => (selectedRevision ? fetchRouting(selectedRevision.id) : Promise.resolve([])),
    enabled: !!selectedRevision,
  });
  const { data: validation, isLoading: validationLoading } = useQuery({
    queryKey: ['pdm-validation', selectedRevision?.id],
    queryFn: () => (selectedRevision ? validateRevision(selectedRevision.id) : Promise.resolve(null)),
    enabled: !!selectedRevision && selectedRevision.status === 'DRAFT',
  });
  const { data: ecrs = [] } = useQuery({
    queryKey: ['pdm-ecrs'],
    queryFn: fetchEcrs,
    enabled: activeTab === 'ecr',
  });

  // Mutations
  const releaseMutation = useMutation({
    mutationFn: (revisionId: number) => releaseRevision(revisionId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['pdm-revisions'] });
      queryClient.invalidateQueries({ queryKey: ['pdm-items'] });
      queryClient.invalidateQueries({ queryKey: ['pdm-status'] });
    },
  });

  const seedMutation = useMutation({
    mutationFn: seedDemoData,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['pdm-items'] });
      queryClient.invalidateQueries({ queryKey: ['pdm-status'] });
    },
  });

  const stats = status?.statistics;

  return (
    <div className="min-h-screen bg-gradient-to-br from-zinc-950 via-zinc-900 to-zinc-950 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-3xl font-bold text-white mb-2 flex items-center gap-3">
              <div className="p-2 bg-gradient-to-br from-purple-500/20 to-blue-500/20 rounded-xl">
                <FolderOpen className="w-8 h-8 text-purple-500" />
              </div>
              PDM: Product Data Management
            </h1>
            <p className="text-zinc-400">Gestão de Items, Revisões, BOM e Roteiros</p>
          </div>
          <button
            onClick={() => seedMutation.mutate()}
            disabled={seedMutation.isPending}
            className="flex items-center gap-2 px-4 py-2 bg-purple-500/20 hover:bg-purple-500/30 text-purple-400 rounded-lg border border-purple-500/30"
          >
            {seedMutation.isPending ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Plus className="w-4 h-4" />}
            Seed Demo Data
          </button>
        </div>

        {/* Stats */}
        {stats && (
          <div className="grid grid-cols-5 gap-4 mb-6">
            <StatCard label="Items" value={stats.total_items} icon={<Package className="w-5 h-5" />} color="bg-purple-500/20 text-purple-400" />
            <StatCard label="Revisões" value={stats.total_revisions} icon={<GitBranch className="w-5 h-5" />} color="bg-blue-500/20 text-blue-400" />
            <StatCard label="Liberadas" value={stats.released_revisions} icon={<Check className="w-5 h-5" />} color="bg-emerald-500/20 text-emerald-400" />
            <StatCard label="Rascunhos" value={stats.draft_revisions} icon={<Edit3 className="w-5 h-5" />} color="bg-amber-500/20 text-amber-400" />
            <StatCard label="ECRs Abertas" value={stats.open_ecrs} icon={<AlertCircle className="w-5 h-5" />} color="bg-red-500/20 text-red-400" />
          </div>
        )}

        {/* Main Content */}
        <div className="grid grid-cols-12 gap-6">
          {/* Item List */}
          <div className="col-span-3">
            <div className="bg-zinc-900/50 rounded-xl p-4 border border-zinc-800">
              <div className="relative mb-4">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-zinc-500" />
                <input
                  type="text"
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  placeholder="Pesquisar items..."
                  className="w-full pl-10 pr-4 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-white text-sm focus:outline-none focus:border-purple-500"
                />
              </div>
              {itemsLoading ? (
                <div className="flex justify-center py-8">
                  <RefreshCw className="w-6 h-6 text-purple-500 animate-spin" />
                </div>
              ) : (
                <ItemList items={items} selectedId={selectedItem?.id} onSelect={(item) => {
                  setSelectedItem(item);
                  setSelectedRevision(null);
                }} />
              )}
            </div>
          </div>

          {/* Revisions */}
          <div className="col-span-3">
            {selectedItem ? (
              <RevisionPanel
                item={selectedItem}
                revisions={revisions}
                selectedRevision={selectedRevision || undefined}
                onSelectRevision={setSelectedRevision}
                onRelease={(revId) => releaseMutation.mutate(revId)}
              />
            ) : (
              <div className="bg-zinc-900/50 rounded-xl p-8 border border-zinc-800 flex flex-col items-center justify-center text-zinc-500">
                <ChevronRight className="w-8 h-8 opacity-30 mb-2" />
                <p>Selecione um item</p>
              </div>
            )}
          </div>

          {/* Detail Panel */}
          <div className="col-span-6">
            <div className="bg-zinc-900/50 rounded-xl border border-zinc-800">
              {/* Tabs */}
              <div className="flex border-b border-zinc-800">
                {(['bom', 'routing', 'ecr'] as const).map((tab) => (
                  <button
                    key={tab}
                    onClick={() => setActiveTab(tab)}
                    className={`flex-1 py-3 text-sm font-medium transition-colors ${
                      activeTab === tab
                        ? 'text-purple-400 border-b-2 border-purple-500'
                        : 'text-zinc-500 hover:text-zinc-300'
                    }`}
                  >
                    {tab === 'bom' && 'BOM'}
                    {tab === 'routing' && 'Roteiro'}
                    {tab === 'ecr' && 'ECRs'}
                  </button>
                ))}
              </div>

              {/* Tab Content */}
              <div className="p-4 space-y-4">
                {activeTab === 'bom' && selectedRevision && (
                  <>
                    {selectedRevision.status === 'DRAFT' && (
                      <ValidationPanel validation={validation || null} loading={validationLoading} />
                    )}
                    <BomPanel bom={bom} loading={bomLoading} />
                  </>
                )}
                {activeTab === 'routing' && selectedRevision && (
                  <RoutingPanel routing={routing} loading={routingLoading} />
                )}
                {activeTab === 'ecr' && <ECRList ecrs={ecrs} />}
                {!selectedRevision && activeTab !== 'ecr' && (
                  <div className="flex flex-col items-center justify-center py-12 text-zinc-500">
                    <Layers className="w-8 h-8 opacity-30 mb-2" />
                    <p>Selecione uma revisão para ver detalhes</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="mt-8 text-center text-xs text-zinc-600">
          <p>PDM gerencia dados mestres de produto com controle de revisões, validação de BOM e workflow de liberação.</p>
          <p className="mt-1">R&D / SIFIDE: WP1 - PLM/PDM Core</p>
        </div>
      </div>
    </div>
  );
};

export default PDMDashboard;



