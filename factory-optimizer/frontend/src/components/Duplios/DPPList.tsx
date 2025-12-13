import { useEffect, useState } from 'react'
import { apiListDPPs, apiGetComplianceSummary } from '../../services/dupliosApi'
import { API_BASE_URL } from '../../config/api'

interface DPPSummary {
  dpp_id: string
  gtin: string
  product_name: string
  product_category: string
  manufacturer_name: string
  status: string
  trust_index: number
  carbon_footprint_kg_co2eq: number
  recyclability_percent: number
  data_completeness_percent: number
  created_at: string
}

const CategoryBadge = ({ category }: { category: string }) => {
  const colors: Record<string, string> = {
    'Equipamento Industrial': 'bg-blue-500/20 text-blue-400',
    'Energia Renovável': 'bg-green-500/20 text-green-400',
    'Armazenamento Energia': 'bg-purple-500/20 text-purple-400',
    'Transmissão Mecânica': 'bg-amber-500/20 text-amber-400',
    'Têxtil Industrial': 'bg-pink-500/20 text-pink-400',
    'Automação Industrial': 'bg-cyan-500/20 text-cyan-400',
    'Eletrónica Industrial': 'bg-indigo-500/20 text-indigo-400',
    'Mobiliário': 'bg-orange-500/20 text-orange-400',
  }
  return (
    <span className={`px-2 py-0.5 rounded text-xs ${colors[category] || 'bg-gray-500/20 text-gray-400'}`}>
      {category}
    </span>
  )
}

const TrustMini = ({ value }: { value: number }) => {
  const color = value >= 80 ? 'text-green-400' : value >= 60 ? 'text-amber-400' : 'text-red-400'
  return (
    <div className={`text-center ${color}`}>
      <span className="text-lg font-bold">{value?.toFixed(0)}</span>
      <p className="text-[8px] uppercase tracking-wider opacity-70">Trust</p>
    </div>
  )
}

export const DPPList = ({ onSelect, selectedId }: { onSelect: (id: string) => void; selectedId?: string | null }) => {
  const [items, setItems] = useState<DPPSummary[]>([])
  const [filter, setFilter] = useState('')
  const [categoryFilter, setCategoryFilter] = useState<string>('')
  const [sortBy, setSortBy] = useState<'name' | 'trust' | 'carbon' | 'date'>('name')
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc')
  const [trustFilter, setTrustFilter] = useState<number | null>(null)
  const [complianceFilter, setComplianceFilter] = useState<'none' | 'espr_low' | 'cbam_low' | 'csrd_low'>('none')
  const [loading, setLoading] = useState(true)
  const [complianceScores, setComplianceScores] = useState<Record<string, {espr: number, cbam: number | null, csrd: number}>>({})

  useEffect(() => {
    setLoading(true)
    apiListDPPs()
      .then(setItems)
      .catch(() => setItems([]))
      .finally(() => setLoading(false))
  }, [])

  // Fetch compliance scores for all items
  useEffect(() => {
    if (items.length === 0) return
    const scores: Record<string, {espr: number, cbam: number | null, csrd: number}> = {}
    Promise.all(
      items.map(item =>
        apiGetComplianceSummary(item.dpp_id)
          .then(data => {
            if (data) scores[item.dpp_id] = {espr: data.espr_score, cbam: data.cbam_score, csrd: data.csrd_score}
          })
          .catch(() => {})
      )
    ).then(() => setComplianceScores(scores))
  }, [items])

  const categories = [...new Set(items.map((i) => i.product_category).filter(Boolean))]

  const filtered = items
    .filter((item) => {
      const matchesText =
        item.product_name.toLowerCase().includes(filter.toLowerCase()) ||
        item.gtin.includes(filter) ||
        item.manufacturer_name?.toLowerCase().includes(filter.toLowerCase())
      const matchesCategory = !categoryFilter || item.product_category === categoryFilter
      const matchesTrust = trustFilter === null || item.trust_index >= trustFilter
      const matchesCompliance = (() => {
        if (complianceFilter === 'none') return true
        const scores = complianceScores[item.dpp_id]
        if (!scores) return true
        if (complianceFilter === 'espr_low') return scores.espr < 80
        if (complianceFilter === 'cbam_low' && scores.cbam !== null) return scores.cbam < 80
        if (complianceFilter === 'csrd_low') return scores.csrd < 80
        return true
      })()
      return matchesText && matchesCategory && matchesTrust && matchesCompliance
    })
    .sort((a, b) => {
      let comparison = 0
      switch (sortBy) {
        case 'trust':
          comparison = a.trust_index - b.trust_index
          break
        case 'carbon':
          comparison = (a.carbon_footprint_kg_co2eq || 0) - (b.carbon_footprint_kg_co2eq || 0)
          break
        case 'date':
          comparison = new Date(a.created_at).getTime() - new Date(b.created_at).getTime()
          break
        case 'name':
        default:
          comparison = a.product_name.localeCompare(b.product_name)
          break
      }
      return sortOrder === 'asc' ? comparison : -comparison
    })

  // Aggregated KPIs
  const totalCarbon = items.reduce((sum, i) => sum + (i.carbon_footprint_kg_co2eq || 0), 0)
  const avgTrust = items.length > 0 ? items.reduce((sum, i) => sum + (i.trust_index || 0), 0) / items.length : 0
  const avgRecyclability = items.length > 0 ? items.reduce((sum, i) => sum + (i.recyclability_percent || 0), 0) / items.length : 0
  const publishedCount = items.filter((i) => i.status === 'published').length

  return (
    <div className="rounded-2xl border border-border bg-surface p-6 space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-text-primary">Passaportes Digitais ({items.length})</h3>
        <span className="text-xs text-text-muted">{publishedCount} publicados</span>
      </div>

      {/* Aggregated Stats */}
      <div className="grid grid-cols-4 gap-2">
        <div className="bg-background/50 rounded-lg p-2 text-center">
          <p className="text-lg font-bold text-text-primary">{items.length}</p>
          <p className="text-[10px] text-text-muted uppercase">DPPs</p>
        </div>
        <div className="bg-background/50 rounded-lg p-2 text-center">
          <p className="text-lg font-bold text-blue-400">{totalCarbon.toFixed(0)}</p>
          <p className="text-[10px] text-text-muted uppercase">kg CO₂e Total</p>
        </div>
        <div className="bg-background/50 rounded-lg p-2 text-center">
          <p className="text-lg font-bold text-green-400">{avgRecyclability.toFixed(0)}%</p>
          <p className="text-[10px] text-text-muted uppercase">Reciclável Médio</p>
        </div>
        <div className="bg-background/50 rounded-lg p-2 text-center">
          <p className="text-lg font-bold text-amber-400">{avgTrust.toFixed(0)}</p>
          <p className="text-[10px] text-text-muted uppercase">Trust Médio</p>
        </div>
      </div>

      {/* Filters */}
      <div className="space-y-2">
        <div className="flex gap-2">
          <input
            type="text"
            placeholder="Pesquisar produto, GTIN ou fabricante..."
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            className="flex-1 bg-background border border-border rounded-lg px-3 py-2 text-sm text-text-primary placeholder:text-text-muted focus:outline-none focus:border-nikufra"
          />
          <select
            value={categoryFilter}
            onChange={(e) => setCategoryFilter(e.target.value)}
            className="bg-background border border-border rounded-lg px-3 py-2 text-sm text-text-primary focus:outline-none focus:border-nikufra"
          >
            <option value="">Todas categorias</option>
            {categories.map((cat) => (
              <option key={cat} value={cat}>{cat}</option>
            ))}
          </select>
        </div>
        <div className="flex gap-2">
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value as any)}
            className="bg-background border border-border rounded-lg px-3 py-2 text-sm text-text-primary focus:outline-none focus:border-nikufra"
          >
            <option value="name">Ordenar por Nome</option>
            <option value="trust">Ordenar por Trust Index</option>
            <option value="carbon">Ordenar por Carbono</option>
            <option value="date">Ordenar por Data</option>
          </select>
          <button
            onClick={() => setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')}
            className="bg-background border border-border rounded-lg px-3 py-2 text-sm text-text-primary hover:bg-background/80 transition"
          >
            {sortOrder === 'asc' ? '↑' : '↓'}
          </button>
          <select
            value={trustFilter || ''}
            onChange={(e) => setTrustFilter(e.target.value ? Number(e.target.value) : null)}
            className="bg-background border border-border rounded-lg px-3 py-2 text-sm text-text-primary focus:outline-none focus:border-nikufra"
          >
            <option value="">Todos Trust Index</option>
            <option value="80">Trust ≥ 80</option>
            <option value="60">Trust ≥ 60</option>
            <option value="40">Trust ≥ 40</option>
          </select>
          <select
            value={complianceFilter}
            onChange={(e) => setComplianceFilter(e.target.value as any)}
            className="bg-background border border-border rounded-lg px-3 py-2 text-sm text-text-primary focus:outline-none focus:border-nikufra"
          >
            <option value="none">Todos Compliance</option>
            <option value="espr_low">ESPR &lt; 80</option>
            <option value="cbam_low">CBAM &lt; 80</option>
            <option value="csrd_low">CSRD &lt; 80</option>
          </select>
        </div>
      </div>

      {/* List */}
      {loading ? (
        <div className="flex justify-center py-8">
          <div className="animate-spin w-6 h-6 border-2 border-nikufra border-t-transparent rounded-full" />
        </div>
      ) : filtered.length === 0 ? (
        <p className="text-sm text-text-muted text-center py-8">Nenhum DPP encontrado.</p>
      ) : (
        <div className="space-y-2 max-h-[400px] overflow-y-auto pr-1">
          {filtered.map((item) => (
            <div
              key={item.dpp_id}
              onClick={() => onSelect(item.dpp_id)}
              className={`border rounded-xl p-3 cursor-pointer transition-all hover:border-nikufra/50 ${
                selectedId === item.dpp_id ? 'border-nikufra bg-nikufra/5' : 'border-border/50 bg-background/30'
              }`}
            >
              <div className="flex items-center gap-3">
                <TrustMini value={item.trust_index} />
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <p className="font-semibold text-text-primary truncate">{item.product_name}</p>
                    <span className={`px-1.5 py-0.5 rounded text-[10px] ${
                      item.status === 'published' ? 'bg-green-500/20 text-green-400' : 'bg-amber-500/20 text-amber-400'
                    }`}>
                      {item.status === 'published' ? '✓' : '●'}
                    </span>
                  </div>
                  <div className="flex items-center gap-2 text-xs text-text-muted">
                    <span className="font-mono">{item.gtin}</span>
                    <span>•</span>
                    <span>{item.manufacturer_name}</span>
                  </div>
                  <div className="flex items-center gap-2 mt-1">
                    <CategoryBadge category={item.product_category} />
                    <span className="text-xs text-text-muted">{item.carbon_footprint_kg_co2eq?.toFixed(0)} kg CO₂e</span>
                    <span className="text-xs text-green-400">{item.recyclability_percent?.toFixed(0)}% recicl.</span>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
