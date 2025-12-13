import { useState, useEffect } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { DPPBuilder } from '../components/Duplios/DPPBuilder'
import { DPPList } from '../components/Duplios/DPPList'
import { DPPViewer } from '../components/Duplios/DPPViewer'
import { apiListDPPs } from '../services/dupliosApi'
import { API_BASE_URL } from '../config/api'
import {
  FileText,
  Layers,
  Leaf,
  Shield,
  Fingerprint,
  RefreshCw,
  Plus,
  ChevronRight,
  AlertTriangle,
  CheckCircle,
  X,
  History,
  GitBranch,
  BarChart3,
} from 'lucide-react'
import { lazy, Suspense } from 'react'

const DupliosAnalyticsPanel = lazy(() => import('../components/DupliosAnalyticsPanel'))

// ═══════════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════════

interface PDMItem {
  id: number
  item_number: string
  item_type: string
  description: string | null
  active_revision?: {
    id: number
    revision_label: string
    status: string
  }
  created_at: string
}

interface PDMRevision {
  id: number
  item_id: number
  revision_label: string
  status: string
  description: string | null
  created_at: string
  released_at: string | null
}

interface DigitalIdentity {
  id: number
  revision_id: number
  identity_type: string
  identity_value: string
  status: string
  created_at: string
  verified_at: string | null
}

interface LCAResult {
  carbon_kg_co2eq: number
  water_m3: number
  energy_mj: number
  recycled_content_pct: number
  recyclability_pct: number
}

// ═══════════════════════════════════════════════════════════════════════════════
// API CALLS
// ═══════════════════════════════════════════════════════════════════════════════

async function fetchPDMItems(): Promise<PDMItem[]> {
  const res = await fetch(`${API_BASE_URL}/duplios/items?limit=100`)
  if (!res.ok) return []
  return res.json()
}

async function fetchRevisions(itemId: number): Promise<PDMRevision[]> {
  const res = await fetch(`${API_BASE_URL}/duplios/items/${itemId}/revisions`)
  if (!res.ok) return []
  return res.json()
}

async function fetchIdentities(revisionId: number): Promise<DigitalIdentity[]> {
  const res = await fetch(`${API_BASE_URL}/duplios/identity/${revisionId}/list`)
  if (!res.ok) return []
  return res.json()
}

async function recalculateLCA(revisionId: number): Promise<LCAResult> {
  const res = await fetch(`${API_BASE_URL}/duplios/revisions/${revisionId}/lca/recalculate`, {
    method: 'POST',
  })
  if (!res.ok) throw new Error('Failed to recalculate LCA')
  return res.json()
}

async function createRevision(itemId: number, description: string): Promise<any> {
  const res = await fetch(`${API_BASE_URL}/duplios/items/${itemId}/revisions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ description }),
  })
  if (!res.ok) throw new Error('Failed to create revision')
  return res.json()
}

async function releaseRevision(revisionId: number): Promise<any> {
  const res = await fetch(`${API_BASE_URL}/duplios/revisions/${revisionId}/release`, {
    method: 'POST',
  })
  if (!res.ok) throw new Error('Failed to release revision')
  return res.json()
}

async function ingestIdentity(revisionId: number, type: string, value: string): Promise<any> {
  const res = await fetch(`${API_BASE_URL}/duplios/identity/ingest`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ revision_id: revisionId, identity_type: type, identity_value: value }),
  })
  if (!res.ok) throw new Error('Failed to ingest identity')
  return res.json()
}

async function verifyIdentity(revisionId: number, type: string, value: string): Promise<any> {
  const res = await fetch(`${API_BASE_URL}/duplios/identity/verify`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ revision_id: revisionId, identity_type: type, identity_value: value }),
  })
  if (!res.ok) throw new Error('Failed to verify identity')
  return res.json()
}

// ═══════════════════════════════════════════════════════════════════════════════
// COMPONENTS
// ═══════════════════════════════════════════════════════════════════════════════

const KPICard = ({ title, value, subtitle, icon, color }: {
  title: string
  value: string | number
  subtitle?: string
  icon: string
  color: string
}) => (
  <div className={`bg-gradient-to-br ${color} rounded-2xl p-4 border border-white/10`}>
    <div className="flex items-center justify-between">
      <div>
        <p className="text-xs text-white/70 uppercase tracking-wider">{title}</p>
        <p className="text-2xl font-bold text-white mt-1">{value}</p>
        {subtitle && <p className="text-xs text-white/50 mt-1">{subtitle}</p>}
      </div>
      <span className="text-3xl">{icon}</span>
    </div>
  </div>
)

const ComplianceBar = ({ label, count, total, color }: {
  label: string
  count: number
  total: number
  color: string
}) => {
  const pct = total > 0 ? Math.round((count / total) * 100) : 0
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-sm">
        <span className="text-text-muted">{label}</span>
        <span className="text-text-primary font-medium">{count}/{total} ({pct}%)</span>
      </div>
      <div className="h-2 bg-border/30 rounded-full overflow-hidden">
        <div className={`h-full ${color} transition-all`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  )
}

const StatusBadge: React.FC<{ status: string }> = ({ status }) => {
  const config: Record<string, { bg: string; text: string }> = {
    DRAFT: { bg: 'bg-slate-500/20', text: 'text-slate-400' },
    RELEASED: { bg: 'bg-emerald-500/20', text: 'text-emerald-400' },
    OBSOLETE: { bg: 'bg-red-500/20', text: 'text-red-400' },
    VERIFIED: { bg: 'bg-emerald-500/20', text: 'text-emerald-400' },
    PENDING: { bg: 'bg-amber-500/20', text: 'text-amber-400' },
    INVALID: { bg: 'bg-red-500/20', text: 'text-red-400' },
  }
  const c = config[status] || config.DRAFT
  return (
    <span className={`px-2 py-0.5 rounded text-xs font-medium ${c.bg} ${c.text}`}>
      {status}
    </span>
  )
}

// PDM Tab Component
const PDMTab: React.FC = () => {
  const queryClient = useQueryClient()
  const [selectedItem, setSelectedItem] = useState<PDMItem | null>(null)
  const [showNewRevision, setShowNewRevision] = useState(false)
  const [newRevisionDesc, setNewRevisionDesc] = useState('')

  const { data: items, isLoading } = useQuery({
    queryKey: ['pdm-items'],
    queryFn: fetchPDMItems,
  })

  const { data: revisions } = useQuery({
    queryKey: ['pdm-revisions', selectedItem?.id],
    queryFn: () => fetchRevisions(selectedItem!.id),
    enabled: !!selectedItem,
  })

  const createMutation = useMutation({
    mutationFn: () => createRevision(selectedItem!.id, newRevisionDesc),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['pdm-revisions', selectedItem?.id] })
      setShowNewRevision(false)
      setNewRevisionDesc('')
    },
  })

  const releaseMutation = useMutation({
    mutationFn: (revisionId: number) => releaseRevision(revisionId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['pdm-revisions', selectedItem?.id] })
      queryClient.invalidateQueries({ queryKey: ['pdm-items'] })
    },
  })

  return (
    <div className="grid lg:grid-cols-2 gap-6">
      {/* Items List */}
      <div className="rounded-2xl border border-border bg-surface p-4">
        <h3 className="text-lg font-semibold text-text-primary mb-4 flex items-center gap-2">
          <Layers className="w-5 h-5 text-cyan-400" />
          Itens PDM
        </h3>
        
        {isLoading ? (
          <div className="flex justify-center py-8">
            <RefreshCw className="w-6 h-6 text-cyan-400 animate-spin" />
          </div>
        ) : (
          <div className="space-y-2 max-h-96 overflow-y-auto">
            {items?.map((item) => (
              <div
                key={item.id}
                onClick={() => setSelectedItem(item)}
                className={`p-3 rounded-lg cursor-pointer transition-all ${
                  selectedItem?.id === item.id
                    ? 'bg-cyan-500/10 border border-cyan-500/30'
                    : 'bg-background hover:bg-slate-800/50 border border-transparent'
                }`}
              >
                <div className="flex items-center justify-between">
                  <div>
                    <p className="font-medium text-text-primary">{item.item_number}</p>
                    <p className="text-xs text-text-muted">{item.description || item.item_type}</p>
                  </div>
                  <div className="text-right">
                    {item.active_revision && (
                      <>
                        <p className="text-xs text-cyan-400">Rev. {item.active_revision.revision_label}</p>
                        <StatusBadge status={item.active_revision.status} />
                      </>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Revisions Detail */}
      <div className="rounded-2xl border border-border bg-surface p-4">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-text-primary flex items-center gap-2">
            <History className="w-5 h-5 text-purple-400" />
            Revisões
          </h3>
          {selectedItem && (
            <button
              onClick={() => setShowNewRevision(true)}
              className="flex items-center gap-1 px-3 py-1.5 bg-purple-600 hover:bg-purple-500 text-white text-sm rounded-lg transition"
            >
              <Plus className="w-4 h-4" />
              Nova Revisão
            </button>
          )}
        </div>

        {!selectedItem ? (
          <p className="text-center text-text-muted py-8">Selecione um item para ver as revisões</p>
        ) : (
          <div className="space-y-3">
            {/* New Revision Form */}
            {showNewRevision && (
              <div className="p-3 rounded-lg bg-purple-500/10 border border-purple-500/30 space-y-3">
                <input
                  type="text"
                  placeholder="Descrição da nova revisão..."
                  value={newRevisionDesc}
                  onChange={(e) => setNewRevisionDesc(e.target.value)}
                  className="w-full px-3 py-2 bg-background border border-border rounded-lg text-text-primary text-sm focus:border-purple-500 focus:outline-none"
                />
                <div className="flex gap-2">
                  <button
                    onClick={() => createMutation.mutate()}
                    disabled={createMutation.isPending}
                    className="flex-1 py-2 bg-purple-600 hover:bg-purple-500 text-white text-sm font-medium rounded-lg transition disabled:opacity-50"
                  >
                    {createMutation.isPending ? 'A criar...' : 'Criar Revisão'}
                  </button>
                  <button
                    onClick={() => setShowNewRevision(false)}
                    className="px-3 py-2 bg-slate-700 hover:bg-slate-600 text-white text-sm rounded-lg transition"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>
              </div>
            )}

            {/* Revisions List */}
            <div className="space-y-2 max-h-72 overflow-y-auto">
              {revisions?.map((rev) => (
                <div key={rev.id} className="p-3 rounded-lg bg-background border border-border/50">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <GitBranch className="w-4 h-4 text-purple-400" />
                      <span className="font-medium text-text-primary">Rev. {rev.revision_label}</span>
                      <StatusBadge status={rev.status} />
                    </div>
                    {rev.status === 'DRAFT' && (
                      <button
                        onClick={() => releaseMutation.mutate(rev.id)}
                        disabled={releaseMutation.isPending}
                        className="text-xs text-emerald-400 hover:text-emerald-300 transition"
                      >
                        Publicar
                      </button>
                    )}
                  </div>
                  {rev.description && (
                    <p className="text-xs text-text-muted mt-1">{rev.description}</p>
                  )}
                  <p className="text-xs text-text-muted mt-1">
                    Criada: {new Date(rev.created_at).toLocaleString('pt-PT')}
                  </p>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

// Identity Tab Component
const IdentityTab: React.FC = () => {
  const queryClient = useQueryClient()
  const [selectedItem, setSelectedItem] = useState<PDMItem | null>(null)
  const [newIdentity, setNewIdentity] = useState({ type: 'SERIAL', value: '' })
  const [verifyInput, setVerifyInput] = useState({ type: 'SERIAL', value: '' })

  const { data: items } = useQuery({
    queryKey: ['pdm-items'],
    queryFn: fetchPDMItems,
  })

  const revisionId = selectedItem?.active_revision?.id

  const { data: identities, isLoading } = useQuery({
    queryKey: ['identities', revisionId],
    queryFn: () => fetchIdentities(revisionId!),
    enabled: !!revisionId,
  })

  const ingestMutation = useMutation({
    mutationFn: () => ingestIdentity(revisionId!, newIdentity.type, newIdentity.value),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['identities', revisionId] })
      setNewIdentity({ type: 'SERIAL', value: '' })
    },
  })

  const verifyMutation = useMutation({
    mutationFn: () => verifyIdentity(revisionId!, verifyInput.type, verifyInput.value),
  })

  const identityTypes = ['SERIAL', 'BATCH', 'RFID', 'QR_CODE', 'GTIN', 'EPC']

  return (
    <div className="grid lg:grid-cols-3 gap-6">
      {/* Products List */}
      <div className="rounded-2xl border border-border bg-surface p-4">
        <h3 className="text-lg font-semibold text-text-primary mb-4 flex items-center gap-2">
          <Layers className="w-5 h-5 text-cyan-400" />
          Produtos
        </h3>
        <div className="space-y-2 max-h-[500px] overflow-y-auto">
          {items?.filter(i => i.active_revision).map((item) => (
            <div
              key={item.id}
              onClick={() => setSelectedItem(item)}
              className={`p-3 rounded-lg cursor-pointer transition-all ${
                selectedItem?.id === item.id
                  ? 'bg-cyan-500/10 border border-cyan-500/30'
                  : 'bg-background hover:bg-slate-800/50 border border-transparent'
              }`}
            >
              <p className="font-medium text-text-primary">{item.item_number}</p>
              <p className="text-xs text-text-muted">Rev. {item.active_revision?.revision_label}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Identities & Actions */}
      <div className="lg:col-span-2 space-y-6">
        {/* Add Identity */}
        <div className="rounded-2xl border border-border bg-surface p-4">
          <h3 className="text-lg font-semibold text-text-primary mb-4 flex items-center gap-2">
            <Plus className="w-5 h-5 text-emerald-400" />
            Registar Identidade
          </h3>
          
          {!selectedItem ? (
            <p className="text-text-muted text-sm">Selecione um produto primeiro</p>
          ) : (
            <div className="grid grid-cols-3 gap-3">
              <select
                value={newIdentity.type}
                onChange={(e) => setNewIdentity(p => ({ ...p, type: e.target.value }))}
                className="px-3 py-2 bg-background border border-border rounded-lg text-text-primary text-sm"
              >
                {identityTypes.map(t => (
                  <option key={t} value={t}>{t}</option>
                ))}
              </select>
              <input
                type="text"
                placeholder="Valor da identidade..."
                value={newIdentity.value}
                onChange={(e) => setNewIdentity(p => ({ ...p, value: e.target.value }))}
                className="px-3 py-2 bg-background border border-border rounded-lg text-text-primary text-sm"
              />
              <button
                onClick={() => ingestMutation.mutate()}
                disabled={!newIdentity.value || ingestMutation.isPending}
                className="py-2 bg-emerald-600 hover:bg-emerald-500 text-white text-sm font-medium rounded-lg transition disabled:opacity-50"
              >
                {ingestMutation.isPending ? 'A registar...' : 'Registar'}
              </button>
            </div>
          )}
        </div>

        {/* Verify Identity */}
        <div className="rounded-2xl border border-border bg-surface p-4">
          <h3 className="text-lg font-semibold text-text-primary mb-4 flex items-center gap-2">
            <Shield className="w-5 h-5 text-amber-400" />
            Verificar Identidade
          </h3>
          
          {!selectedItem ? (
            <p className="text-text-muted text-sm">Selecione um produto primeiro</p>
          ) : (
            <div className="space-y-3">
              <div className="grid grid-cols-3 gap-3">
                <select
                  value={verifyInput.type}
                  onChange={(e) => setVerifyInput(p => ({ ...p, type: e.target.value }))}
                  className="px-3 py-2 bg-background border border-border rounded-lg text-text-primary text-sm"
                >
                  {identityTypes.map(t => (
                    <option key={t} value={t}>{t}</option>
                  ))}
                </select>
                <input
                  type="text"
                  placeholder="Valor a verificar..."
                  value={verifyInput.value}
                  onChange={(e) => setVerifyInput(p => ({ ...p, value: e.target.value }))}
                  className="px-3 py-2 bg-background border border-border rounded-lg text-text-primary text-sm"
                />
                <button
                  onClick={() => verifyMutation.mutate()}
                  disabled={!verifyInput.value || verifyMutation.isPending}
                  className="py-2 bg-amber-600 hover:bg-amber-500 text-white text-sm font-medium rounded-lg transition disabled:opacity-50"
                >
                  {verifyMutation.isPending ? 'A verificar...' : 'Verificar'}
                </button>
              </div>

              {verifyMutation.data && (
                <div className={`p-3 rounded-lg ${
                  verifyMutation.data.match ? 'bg-emerald-500/10 border border-emerald-500/30' : 'bg-red-500/10 border border-red-500/30'
                }`}>
                  <div className="flex items-center gap-2">
                    {verifyMutation.data.match ? (
                      <CheckCircle className="w-5 h-5 text-emerald-400" />
                    ) : (
                      <AlertTriangle className="w-5 h-5 text-red-400" />
                    )}
                    <span className={verifyMutation.data.match ? 'text-emerald-400' : 'text-red-400'}>
                      {verifyMutation.data.match ? 'Identidade Válida ✓' : 'Identidade Não Encontrada'}
                    </span>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Registered Identities */}
        <div className="rounded-2xl border border-border bg-surface p-4">
          <h3 className="text-lg font-semibold text-text-primary mb-4 flex items-center gap-2">
            <Fingerprint className="w-5 h-5 text-purple-400" />
            Identidades Registadas
          </h3>
          
          {!revisionId ? (
            <p className="text-text-muted text-sm">Selecione um produto para ver as identidades</p>
          ) : isLoading ? (
            <div className="flex justify-center py-4">
              <RefreshCw className="w-5 h-5 text-purple-400 animate-spin" />
            </div>
          ) : identities?.length === 0 ? (
            <p className="text-text-muted text-sm text-center py-4">Nenhuma identidade registada</p>
          ) : (
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {identities?.map((id) => (
                <div key={id.id} className="flex items-center justify-between p-3 rounded-lg bg-background border border-border/50">
                  <div>
                    <p className="text-sm text-text-primary font-mono">{id.identity_value}</p>
                    <p className="text-xs text-text-muted">{id.identity_type}</p>
                  </div>
                  <StatusBadge status={id.status} />
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

// Impact Tab Component (with LCA recalculate)
const ImpactTab: React.FC<{ products: any[]; stats: any }> = ({ products, stats }) => {
  const queryClient = useQueryClient()
  const [selectedRevisionId, setSelectedRevisionId] = useState<number | null>(null)

  const { data: items } = useQuery({
    queryKey: ['pdm-items'],
    queryFn: fetchPDMItems,
  })

  const lcaMutation = useMutation({
    mutationFn: (revisionId: number) => recalculateLCA(revisionId),
    onSuccess: (data) => {
      // Show result
      alert(`LCA Recalculado!\nCarbono: ${data.carbon_kg_co2eq.toFixed(2)} kg CO₂e\nÁgua: ${data.water_m3.toFixed(2)} m³\nEnergia: ${data.energy_mj.toFixed(2)} MJ`)
    },
  })

  return (
    <div className="space-y-6">
      <div className="grid md:grid-cols-2 gap-6">
        <div className="rounded-2xl border border-border bg-surface p-6 space-y-4">
          <h3 className="text-lg font-semibold text-text-primary flex items-center gap-2">
            🌍 Impacto Ambiental Total
          </h3>
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-gradient-to-br from-green-500/10 to-green-600/5 rounded-xl p-4 border border-green-500/20">
              <p className="text-xs text-green-400 uppercase tracking-wider">Carbono Total</p>
              <p className="text-2xl font-bold text-text-primary">{stats.totalCarbon.toFixed(0)} <span className="text-sm font-normal">kg CO₂e</span></p>
            </div>
            <div className="bg-gradient-to-br from-blue-500/10 to-blue-600/5 rounded-xl p-4 border border-blue-500/20">
              <p className="text-xs text-blue-400 uppercase tracking-wider">Produtos</p>
              <p className="text-2xl font-bold text-text-primary">{stats.total}</p>
            </div>
          </div>
          <div className="space-y-3 pt-4 border-t border-border/50">
            <div className="flex justify-between items-center">
              <span className="text-sm text-text-muted">Média por produto</span>
              <span className="text-sm text-text-primary font-medium">
                {stats.total > 0 ? (stats.totalCarbon / stats.total).toFixed(1) : 0} kg CO₂e
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-text-muted">Reciclabilidade média</span>
              <span className="text-sm text-text-primary font-medium">{stats.avgRecyclability.toFixed(0)}%</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-text-muted">Conteúdo reciclado médio</span>
              <span className="text-sm text-text-primary font-medium">{stats.avgRecycledContent.toFixed(0)}%</span>
            </div>
          </div>
        </div>

        <div className="rounded-2xl border border-border bg-surface p-6 space-y-4">
          <h3 className="text-lg font-semibold text-text-primary flex items-center gap-2">
            <Leaf className="w-5 h-5 text-emerald-400" />
            Recalcular LCA
          </h3>
          <p className="text-sm text-text-muted">Selecione um produto para recalcular o impacto ambiental com base na BOM atual.</p>
          
          <select
            value={selectedRevisionId || ''}
            onChange={(e) => setSelectedRevisionId(Number(e.target.value) || null)}
            className="w-full px-3 py-2 bg-background border border-border rounded-lg text-text-primary"
          >
            <option value="">Selecionar produto...</option>
            {items?.filter(i => i.active_revision).map((item) => (
              <option key={item.id} value={item.active_revision?.id}>
                {item.item_number} (Rev. {item.active_revision?.revision_label})
              </option>
            ))}
          </select>

          <button
            onClick={() => selectedRevisionId && lcaMutation.mutate(selectedRevisionId)}
            disabled={!selectedRevisionId || lcaMutation.isPending}
            className="w-full py-2 bg-emerald-600 hover:bg-emerald-500 text-white font-medium rounded-lg transition disabled:opacity-50"
          >
            {lcaMutation.isPending ? 'A calcular...' : '🔄 Recalcular Impacto'}
          </button>
        </div>
      </div>

      <div className="rounded-2xl border border-border bg-surface p-6 space-y-4">
        <h3 className="text-lg font-semibold text-text-primary flex items-center gap-2">
          📊 Top Produtos por Carbono
        </h3>
        <div className="space-y-3">
          {products
            .sort((a, b) => (b.carbon_footprint_kg_co2eq || 0) - (a.carbon_footprint_kg_co2eq || 0))
            .slice(0, 5)
            .map((p, idx) => (
              <div key={p.dpp_id} className="flex items-center gap-3">
                <span className="text-lg font-bold text-text-muted w-6">{idx + 1}</span>
                <div className="flex-1">
                  <p className="text-sm text-text-primary font-medium truncate">{p.product_name}</p>
                  <p className="text-xs text-text-muted">{p.product_category}</p>
                </div>
                <span className="text-sm font-bold text-green-400">{p.carbon_footprint_kg_co2eq?.toFixed(0)} kg</span>
              </div>
            ))}
        </div>
      </div>
    </div>
  )
}

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN COMPONENT
// ═══════════════════════════════════════════════════════════════════════════════

type TabId = 'overview' | 'pdm' | 'impact' | 'compliance' | 'identity' | 'analytics'

const DupliosPage = () => {
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [refreshKey, setRefreshKey] = useState(0)
  const [showBuilder, setShowBuilder] = useState(false)
  const [activeTab, setActiveTab] = useState<TabId>('overview')
  const [stats, setStats] = useState({
    total: 0,
    published: 0,
    totalCarbon: 0,
    avgTrust: 0,
    avgRecyclability: 0,
    avgRecycledContent: 0,
    categories: 0,
    espr: 0,
    cbam: 0,
    csrd: 0,
  })
  const [products, setProducts] = useState<any[]>([])

  useEffect(() => {
    apiListDPPs().then((items: any[]) => {
      setProducts(items)
      const published = items.filter((i) => i.status === 'published').length
      const totalCarbon = items.reduce((sum, i) => sum + (i.carbon_footprint_kg_co2eq || 0), 0)
      const avgTrust = items.length > 0 ? items.reduce((sum, i) => sum + (i.trust_index || 0), 0) / items.length : 0
      const avgRecyclability = items.length > 0 ? items.reduce((sum, i) => sum + (i.recyclability_percent || 0), 0) / items.length : 0
      const avgRecycledContent = items.length > 0 ? items.reduce((sum, i) => sum + (i.recycled_content_percent || 0), 0) / items.length : 0
      const categories = new Set(items.map((i) => i.product_category).filter(Boolean)).size
      
      const espr = items.filter((i) => i.compliance?.espr_compliant).length
      const cbam = items.filter((i) => i.compliance?.cbam_compliant).length
      const csrd = items.filter((i) => i.compliance?.csrd_compliant).length
      
      setStats({ total: items.length, published, totalCarbon, avgTrust, avgRecyclability, avgRecycledContent, categories, espr, cbam, csrd })
    }).catch(() => {})
  }, [refreshKey])

  const exportCSV = () => window.open(`${API_BASE_URL}/duplios/export/csv`, '_blank')
  const exportJSON = () => window.open(`${API_BASE_URL}/duplios/export/json`, '_blank')

  const tabs: { id: TabId; label: string; icon: React.ReactNode }[] = [
    { id: 'overview', label: 'Visão Geral', icon: <FileText className="w-4 h-4" /> },
    { id: 'pdm', label: 'PDM', icon: <Layers className="w-4 h-4" /> },
    { id: 'impact', label: 'Impacto', icon: <Leaf className="w-4 h-4" /> },
    { id: 'compliance', label: 'Compliance', icon: <Shield className="w-4 h-4" /> },
    { id: 'identity', label: 'Identidade', icon: <Fingerprint className="w-4 h-4" /> },
    { id: 'analytics', label: 'Analytics', icon: <BarChart3 className="w-4 h-4" /> },
  ]

  return (
    <div className="space-y-6">
      {/* Header */}
      <header className="flex items-start justify-between">
        <div>
          <p className="text-xs uppercase tracking-[0.3em] text-text-muted">DUPLIOS</p>
          <h1 className="text-2xl font-semibold text-text-primary">Passaportes Digitais de Produto</h1>
          <p className="text-sm text-text-muted max-w-2xl mt-1">
            PDM Lite + DPP + LCA + Identidade Digital. Compliant com ESPR, CBAM e CSRD.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button onClick={exportCSV} className="flex items-center gap-2 px-3 py-2 bg-surface border border-border hover:border-nikufra/50 rounded-lg text-text-primary text-sm transition-colors">
            📊 CSV
          </button>
          <button onClick={exportJSON} className="flex items-center gap-2 px-3 py-2 bg-surface border border-border hover:border-nikufra/50 rounded-lg text-text-primary text-sm transition-colors">
            📋 JSON
          </button>
          <button
            onClick={() => setShowBuilder(!showBuilder)}
            className="flex items-center gap-2 px-4 py-2 bg-nikufra hover:bg-nikufra-light rounded-lg text-white font-medium transition-colors"
          >
            {showBuilder ? '✕ Fechar' : '+ Novo DPP'}
          </button>
        </div>
      </header>

      {/* Tabs */}
      <div className="flex gap-1 p-1 bg-surface rounded-xl w-fit">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              activeTab === tab.id
                ? 'bg-nikufra text-white'
                : 'text-text-muted hover:text-text-primary hover:bg-slate-700/30'
            }`}
          >
            {tab.icon}
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      {activeTab === 'overview' && (
        <>
          {/* KPIs */}
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
            <KPICard title="Total DPPs" value={stats.total} subtitle={`${stats.published} publicados`} icon="📋" color="from-blue-600/80 to-blue-800/80" />
            <KPICard title="Categorias" value={stats.categories} subtitle="tipos de produto" icon="📦" color="from-purple-600/80 to-purple-800/80" />
            <KPICard title="Pegada Carbónica" value={`${stats.totalCarbon.toFixed(0)}`} subtitle="kg CO₂e total" icon="🌍" color="from-emerald-600/80 to-emerald-800/80" />
            <KPICard title="Trust Index" value={stats.avgTrust.toFixed(0)} subtitle="média global" icon="🛡️" color="from-amber-600/80 to-amber-800/80" />
            <KPICard title="Reciclabilidade" value={`${stats.avgRecyclability.toFixed(0)}%`} subtitle="média" icon="♻️" color="from-green-600/80 to-green-800/80" />
            <KPICard title="Reciclado" value={`${stats.avgRecycledContent.toFixed(0)}%`} subtitle="conteúdo médio" icon="🔄" color="from-cyan-600/80 to-cyan-800/80" />
          </div>

          {/* Builder */}
          {showBuilder && (
            <DPPBuilder onCreated={() => {
              setRefreshKey((prev) => prev + 1)
              setShowBuilder(false)
            }} />
          )}

          {/* Main Content */}
          <div className="grid lg:grid-cols-5 gap-6">
            <div className="lg:col-span-2">
              <DPPList key={refreshKey} onSelect={setSelectedId} selectedId={selectedId} />
            </div>
            <div className="lg:col-span-3">
              <DPPViewer dppId={selectedId} />
            </div>
          </div>
        </>
      )}

      {activeTab === 'pdm' && <PDMTab />}
      
      {activeTab === 'impact' && <ImpactTab products={products} stats={stats} />}

      {activeTab === 'compliance' && (
        <div className="grid md:grid-cols-3 gap-6">
          <div className="rounded-2xl border border-border bg-surface p-6 space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold text-text-primary">ESPR</h3>
              <span className="text-2xl font-bold text-green-400">{Math.round((stats.espr / stats.total) * 100) || 0}%</span>
            </div>
            <p className="text-xs text-text-muted">Ecodesign for Sustainable Products Regulation</p>
            <ComplianceBar label="Conformes" count={stats.espr} total={stats.total} color="bg-green-500" />
            <div className="pt-3 border-t border-border/50">
              <p className="text-xs text-text-muted">Requisitos: GTIN, materiais, reciclabilidade, instruções fim de vida</p>
            </div>
          </div>
          
          <div className="rounded-2xl border border-border bg-surface p-6 space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold text-text-primary">CBAM</h3>
              <span className="text-2xl font-bold text-blue-400">{Math.round((stats.cbam / stats.total) * 100) || 0}%</span>
            </div>
            <p className="text-xs text-text-muted">Carbon Border Adjustment Mechanism</p>
            <ComplianceBar label="Conformes" count={stats.cbam} total={stats.total} color="bg-blue-500" />
            <div className="pt-3 border-t border-border/50">
              <p className="text-xs text-text-muted">Requisitos: pegada carbónica, origem, dados energia/transporte</p>
            </div>
          </div>
          
          <div className="rounded-2xl border border-border bg-surface p-6 space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold text-text-primary">CSRD</h3>
              <span className="text-2xl font-bold text-purple-400">{Math.round((stats.csrd / stats.total) * 100) || 0}%</span>
            </div>
            <p className="text-xs text-text-muted">Corporate Sustainability Reporting Directive</p>
            <ComplianceBar label="Conformes" count={stats.csrd} total={stats.total} color="bg-purple-500" />
            <div className="pt-3 border-t border-border/50">
              <p className="text-xs text-text-muted">Requisitos: certificações, auditorias de terceiros</p>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'identity' && <IdentityTab />}

      {/* TAB: Analytics */}
      {activeTab === 'analytics' && (
        <Suspense fallback={
          <div className="flex items-center justify-center py-24">
            <RefreshCw className="w-8 h-8 text-primary animate-spin" />
          </div>
        }>
          <DupliosAnalyticsPanel />
        </Suspense>
      )}

      {/* Info Footer */}
      <div className="rounded-2xl border border-border bg-surface/50 p-6">
        <div className="grid md:grid-cols-5 gap-6">
          <div>
            <h4 className="font-semibold text-text-primary flex items-center gap-2">📱 QR Codes</h4>
            <p className="text-sm text-text-muted mt-2">Cada DPP gera um QR code único.</p>
          </div>
          <div>
            <h4 className="font-semibold text-text-primary flex items-center gap-2">🛡️ Trust Index</h4>
            <p className="text-sm text-text-muted mt-2">Score 0-100 baseado em completude.</p>
          </div>
          <div>
            <h4 className="font-semibold text-text-primary flex items-center gap-2">✅ Compliance</h4>
            <p className="text-sm text-text-muted mt-2">Avaliação ESPR, CBAM e CSRD.</p>
          </div>
          <div>
            <h4 className="font-semibold text-text-primary flex items-center gap-2">🏭 PDM Lite</h4>
            <p className="text-sm text-text-muted mt-2">Gestão de items e revisões.</p>
          </div>
          <div>
            <h4 className="font-semibold text-text-primary flex items-center gap-2">🔐 Identidade</h4>
            <p className="text-sm text-text-muted mt-2">Rastreabilidade total.</p>
          </div>
        </div>
      </div>
    </div>
  )
}

export default DupliosPage
