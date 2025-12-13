import { useEffect, useState } from 'react'
import { apiGetDPP, apiGapFillLite, apiGetComplianceRadar, apiGetTrustIndex, apiRecalculateTrustIndex } from '../../services/dupliosApi'
import { DPPQRCodeCard } from './DPPQRCodeCard'
import { API_BASE_URL } from '../../config/api'
import { RefreshCw, AlertCircle, CheckCircle, Shield, RotateCcw, X } from 'lucide-react'

const ScoreBadge = ({ value, label, max = 10 }: { value: number; label: string; max?: number }) => {
  const pct = (value / max) * 100
  const color = pct >= 80 ? 'bg-green-500' : pct >= 60 ? 'bg-amber-500' : 'bg-red-500'
  return (
    <div className="flex items-center gap-2">
      <div className="w-20 h-2 bg-border/50 rounded-full overflow-hidden">
        <div className={`h-full ${color}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-xs text-text-muted">{label}: {value}/{max}</span>
    </div>
  )
}

const TrustBadge = ({ value }: { value: number }) => {
  const color = value >= 80 ? 'from-green-500 to-emerald-600' : value >= 60 ? 'from-amber-500 to-orange-600' : 'from-red-500 to-rose-600'
  return (
    <div className={`relative w-20 h-20 rounded-full bg-gradient-to-br ${color} flex items-center justify-center shadow-lg`}>
      <div className="absolute inset-1 rounded-full bg-surface flex items-center justify-center">
        <div className="text-center">
          <span className="text-xl font-bold text-text-primary">{value?.toFixed(0)}</span>
          <p className="text-[8px] text-text-muted uppercase tracking-wider">Trust</p>
        </div>
      </div>
    </div>
  )
}

const Section = ({ title, children, icon }: { title: string; children: React.ReactNode; icon?: string }) => (
  <div className="border border-border/50 rounded-xl p-4 space-y-3">
    <h4 className="font-semibold text-text-primary flex items-center gap-2">
      {icon && <span>{icon}</span>}
      {title}
    </h4>
    {children}
  </div>
)

interface TrustIndexResult {
  dpp_id: string
  overall_trust_index: number
  field_scores: Record<string, number>
  field_metas: Record<string, {
    field_key: string
    base_class: string
    field_score: number
    recency_days: number
    third_party_verified: boolean
    last_updated: string | null
  }>
  key_messages: string[]
  calculated_at: string
}

export const DPPViewer = ({ dppId }: { dppId: string | null }) => {
  const [data, setData] = useState<any | null>(null)
  const [trustIndex, setTrustIndex] = useState<TrustIndexResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [loadingTrust, setLoadingTrust] = useState(false)
  const [gapFilling, setGapFilling] = useState(false)
  const [gapFillResult, setGapFillResult] = useState<any | null>(null)
  const [compliance, setCompliance] = useState<any | null>(null)
  const [loadingCompliance, setLoadingCompliance] = useState(false)
  const [recalculating, setRecalculating] = useState(false)
  const [recalculateResult, setRecalculateResult] = useState<{ success: boolean; message: string; before?: any; after?: any } | null>(null)

  useEffect(() => {
    if (!dppId) return
    setLoading(true)
    apiGetDPP(dppId)
      .then(setData)
      .catch(() => setData(null))
      .finally(() => setLoading(false))
  }, [dppId])

  // Fetch Trust Index breakdown
  useEffect(() => {
    if (!dppId) return
    setLoadingTrust(true)
    apiGetTrustIndex(dppId)
      .then(setTrustIndex)
      .catch(() => setTrustIndex(null))
      .finally(() => setLoadingTrust(false))
  }, [dppId])

  // Fetch Compliance Radar
  useEffect(() => {
    if (!dppId) return
    setLoadingCompliance(true)
    apiGetComplianceRadar(dppId)
      .then(setCompliance)
      .catch(() => setCompliance(null))
      .finally(() => setLoadingCompliance(false))
  }, [dppId])

  if (!dppId) {
    return (
      <div className="rounded-2xl border border-border bg-surface p-8 flex flex-col items-center justify-center min-h-[400px]">
        <div className="text-6xl mb-4">📋</div>
        <p className="text-text-muted">Seleciona um DPP da lista para ver detalhes</p>
      </div>
    )
  }

  if (loading) {
    return (
      <div className="rounded-2xl border border-border bg-surface p-8 flex items-center justify-center min-h-[400px]">
        <div className="animate-spin w-8 h-8 border-2 border-nikufra border-t-transparent rounded-full" />
      </div>
    )
  }

  if (!data) {
    return (
      <div className="rounded-2xl border border-border bg-surface p-8 text-center min-h-[400px]">
        <p className="text-red-400">Erro ao carregar DPP</p>
      </div>
    )
  }

  return (
    <div className="rounded-2xl border border-border bg-surface p-6 space-y-6 max-h-[85vh] overflow-y-auto">
      {/* Header */}
      <div className="flex items-start justify-between gap-4">
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-1">
            <span className={`px-2 py-0.5 rounded text-xs font-medium ${
              data.status === 'published' ? 'bg-green-500/20 text-green-400' : 'bg-amber-500/20 text-amber-400'
            }`}>
              {data.status === 'published' ? '✓ Publicado' : '● Rascunho'}
            </span>
            <span className="text-xs text-text-muted">{data.product_category}</span>
          </div>
          <h2 className="text-xl font-bold text-text-primary">{data.product_name}</h2>
          <p className="text-sm text-text-muted mt-1">GTIN: {data.gtin}</p>
        </div>
        <div className="flex items-center gap-4">
          <TrustBadge value={trustIndex?.overall_trust_index || data.trust_index || 0} />
          <DPPQRCodeCard dppId={data.dpp_id} />
          <button
            onClick={async () => {
              if (!dppId) return
              setRecalculating(true)
              setRecalculateResult(null)
              const before = {
                trust_index: trustIndex?.overall_trust_index || data.trust_index || 0,
                carbon: data.carbon_footprint_kg_co2eq || 0,
                water: data.water_consumption_m3 || 0,
              }
              try {
                // Recalculate Trust Index
                await apiRecalculateTrustIndex(dppId)
                // Reload all data
                const [updatedData, updatedTrust, updatedCompliance] = await Promise.all([
                  apiGetDPP(dppId),
                  apiGetTrustIndex(dppId),
                  apiGetComplianceRadar(dppId).catch(() => null),
                ])
                setData(updatedData)
                setTrustIndex(updatedTrust)
                if (updatedCompliance) setCompliance(updatedCompliance)
                
                const after = {
                  trust_index: updatedTrust?.overall_trust_index || updatedData.trust_index || 0,
                  carbon: updatedData.carbon_footprint_kg_co2eq || 0,
                  water: updatedData.water_consumption_m3 || 0,
                }
                
                setRecalculateResult({
                  success: true,
                  message: 'Métricas recalculadas com sucesso',
                  before,
                  after,
                })
              } catch (error) {
                console.error('Recalculate failed:', error)
                setRecalculateResult({
                  success: false,
                  message: 'Erro ao recalcular métricas',
                })
              } finally {
                setRecalculating(false)
              }
            }}
            disabled={recalculating}
            className="flex items-center gap-2 px-4 py-2 bg-nikufra hover:bg-nikufra/90 text-white rounded-lg transition disabled:opacity-50 disabled:cursor-not-allowed"
            title="Recalcular todas as métricas (Trust Index, Compliance, Gap Filling)"
          >
            {recalculating ? (
              <>
                <RefreshCw className="w-4 h-4 animate-spin" />
                <span>A recalcular...</span>
              </>
            ) : (
              <>
                <RotateCcw className="w-4 h-4" />
                <span>Recalcular Métricas</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Recalculate Result */}
      {recalculateResult && (
        <div className={`rounded-lg p-4 border ${
          recalculateResult.success
            ? 'bg-green-500/10 border-green-500/30'
            : 'bg-red-500/10 border-red-500/30'
        }`}>
          <div className="flex items-start gap-3">
            {recalculateResult.success ? (
              <CheckCircle className="w-5 h-5 text-green-400 mt-0.5" />
            ) : (
              <AlertCircle className="w-5 h-5 text-red-400 mt-0.5" />
            )}
            <div className="flex-1">
              <p className={`text-sm font-medium ${
                recalculateResult.success ? 'text-green-400' : 'text-red-400'
              }`}>
                {recalculateResult.message}
              </p>
              {recalculateResult.success && recalculateResult.before && recalculateResult.after && (
                <div className="mt-3 space-y-2 text-xs">
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <p className="text-text-muted mb-1">Antes:</p>
                      <p className="text-text-primary">Trust: {recalculateResult.before.trust_index.toFixed(0)}</p>
                      <p className="text-text-primary">Carbono: {recalculateResult.before.carbon.toFixed(1)} kg</p>
                      <p className="text-text-primary">Água: {recalculateResult.before.water.toFixed(1)} m³</p>
                    </div>
                    <div>
                      <p className="text-text-muted mb-1">Depois:</p>
                      <p className="text-text-primary">Trust: {recalculateResult.after.trust_index.toFixed(0)}</p>
                      <p className="text-text-primary">Carbono: {recalculateResult.after.carbon.toFixed(1)} kg</p>
                      <p className="text-text-primary">Água: {recalculateResult.after.water.toFixed(1)} m³</p>
                    </div>
                  </div>
                  {Math.abs(recalculateResult.after.trust_index - recalculateResult.before.trust_index) > 0.1 && (
                    <p className={`text-xs mt-2 ${
                      recalculateResult.after.trust_index > recalculateResult.before.trust_index
                        ? 'text-green-400'
                        : 'text-amber-400'
                    }`}>
                      {recalculateResult.after.trust_index > recalculateResult.before.trust_index ? '↑' : '↓'} 
                      Trust Index: {recalculateResult.before.trust_index.toFixed(0)} → {recalculateResult.after.trust_index.toFixed(0)}
                    </p>
                  )}
                </div>
              )}
            </div>
            <button
              onClick={() => setRecalculateResult(null)}
              className="text-text-muted hover:text-text-primary transition"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>
      )}

      {/* Gap Filling Lite */}
      {data && (
        <Section title="Estimativas Automáticas" icon="🔬">
          <div className="space-y-3">
            {/* Check if environmental fields are missing */}
            {(!data.carbon_footprint_kg_co2eq || data.carbon_footprint_kg_co2eq === 0 ||
              !data.water_consumption_m3 || data.water_consumption_m3 === 0 ||
              !data.recyclability_percent || data.recyclability_percent === 0) ? (
              <div className="bg-amber-500/10 border border-amber-500/30 rounded-lg p-4">
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1">
                    <h5 className="font-semibold text-amber-400 mb-2">Campos ambientais em falta</h5>
                    <p className="text-sm text-text-muted mb-3">
                      Alguns campos ambientais não estão preenchidos. Pode usar estimativas científicas baseadas em fatores médios por material e país.
                    </p>
                    <p className="text-xs text-text-muted">
                      Precisão típica: ±30%. Baseado em fatores médios por material e país.
                    </p>
                  </div>
                  <button
                    onClick={async () => {
                      if (!dppId) return
                      setGapFilling(true)
                      try {
                        const result = await apiGapFillLite(dppId, false)
                        setGapFillResult(result)
                        // Reload DPP data
                        const updated = await apiGetDPP(dppId)
                        setData(updated)
                        // Reload trust index
                        apiGetTrustIndex(dppId).then(setTrustIndex).catch(() => {})
                      } catch (error) {
                        console.error('Gap filling failed:', error)
                        setGapFillResult({ success: false, message: 'Erro ao preencher campos' })
                      } finally {
                        setGapFilling(false)
                      }
                    }}
                    disabled={gapFilling}
                    className="flex items-center gap-2 px-4 py-2 bg-amber-600 hover:bg-amber-500 text-white rounded-lg transition disabled:opacity-50"
                  >
                    {gapFilling ? (
                      <>
                        <RefreshCw className="w-4 h-4 animate-spin" />
                        <span>A preencher...</span>
                      </>
                    ) : (
                      <>
                        <CheckCircle className="w-4 h-4" />
                        <span>Preencher automaticamente</span>
                      </>
                    )}
                  </button>
                </div>
              </div>
            ) : (
              <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-4">
                <div className="flex items-center gap-2">
                  <CheckCircle className="w-5 h-5 text-green-400" />
                  <p className="text-sm text-text-primary">
                    Todos os campos ambientais estão preenchidos.
                  </p>
                </div>
              </div>
            )}
            
            {/* Gap Fill Result */}
            {gapFillResult && (
              <div className={`rounded-lg p-3 ${
                gapFillResult.success 
                  ? 'bg-green-500/10 border border-green-500/30' 
                  : 'bg-red-500/10 border border-red-500/30'
              }`}>
                <div className="flex items-start gap-2">
                  {gapFillResult.success ? (
                    <CheckCircle className="w-5 h-5 text-green-400 mt-0.5" />
                  ) : (
                    <AlertCircle className="w-5 h-5 text-red-400 mt-0.5" />
                  )}
                  <div className="flex-1">
                    <p className={`text-sm font-medium ${
                      gapFillResult.success ? 'text-green-400' : 'text-red-400'
                    }`}>
                      {gapFillResult.message}
                    </p>
                    {gapFillResult.success && gapFillResult.filled_fields?.length > 0 && (
                      <div className="mt-2 space-y-1">
                        <p className="text-xs text-text-muted">Campos preenchidos:</p>
                        <ul className="text-xs text-text-muted list-disc list-inside">
                          {gapFillResult.filled_fields.map((field: string) => (
                            <li key={field}>
                              {field === 'carbon_kg_co2eq' ? 'Carbono' :
                               field === 'water_m3' ? 'Água' :
                               field === 'energy_kwh' ? 'Energia' :
                               field === 'recyclability_pct' ? 'Reciclabilidade' : field}
                              {gapFillResult.values?.[field] && (
                                <span className="ml-2 text-text-primary">
                                  ({gapFillResult.values[field].toFixed(2)})
                                </span>
                              )}
                            </li>
                          ))}
                        </ul>
                        <p className="text-xs text-amber-400 mt-2">
                          ⚠️ Estes valores são estimados (precisão típica ±30%)
                        </p>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}
          </div>
        </Section>
      )}

      {/* Trust Index Breakdown */}
      {trustIndex && (
        <Section title="Trust Index - Breakdown" icon="🛡️">
          <div className="space-y-3">
            {/* Key Messages */}
            {trustIndex.key_messages && trustIndex.key_messages.length > 0 && (
              <div className="bg-background/50 rounded-lg p-3 space-y-1">
                {trustIndex.key_messages.map((msg, idx) => (
                  <p key={idx} className="text-sm text-text-muted">{msg}</p>
                ))}
              </div>
            )}
            
            {/* Field Breakdown Table */}
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border/50">
                    <th className="text-left py-2 text-text-muted font-medium">Campo</th>
                    <th className="text-center py-2 text-text-muted font-medium">Score</th>
                    <th className="text-center py-2 text-text-muted font-medium">Tipo</th>
                    <th className="text-right py-2 text-text-muted font-medium">Última Atualização</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(trustIndex.field_metas || {}).map(([fieldKey, meta]) => {
                    const fieldNames: Record<string, string> = {
                      'carbon_footprint_kg_co2eq': 'Carbono',
                      'water_m3': 'Água',
                      'energy_kwh': 'Energia',
                      'recycled_content_pct': 'Conteúdo Reciclado',
                      'recyclability_pct': 'Reciclabilidade',
                    }
                    const fieldName = fieldNames[fieldKey] || fieldKey
                    const typeNames: Record<string, string> = {
                      'MEDIDO': 'Medido',
                      'REPORTADO': 'Reportado',
                      'ESTIMADO': 'Estimado',
                      'DESCONHECIDO': 'Desconhecido',
                    }
                    const typeName = typeNames[meta.base_class] || meta.base_class
                    const lastUpdated = meta.last_updated 
                      ? new Date(meta.last_updated).toLocaleDateString('pt-PT')
                      : '—'
                    
                    return (
                      <tr key={fieldKey} className="border-b border-border/30">
                        <td className="py-2 text-text-primary">{fieldName}</td>
                        <td className="text-center py-2">
                          <span className={`font-semibold ${
                            meta.field_score >= 80 ? 'text-green-400' :
                            meta.field_score >= 60 ? 'text-amber-400' : 'text-red-400'
                          }`}>
                            {meta.field_score?.toFixed(0) || '—'}
                          </span>
                        </td>
                        <td className="text-center py-2">
                          <span className={`px-2 py-0.5 rounded text-xs ${
                            meta.base_class === 'MEDIDO' ? 'bg-green-500/20 text-green-400' :
                            meta.base_class === 'REPORTADO' ? 'bg-blue-500/20 text-blue-400' :
                            meta.base_class === 'ESTIMADO' ? 'bg-amber-500/20 text-amber-400' :
                            'bg-red-500/20 text-red-400'
                          }`}>
                            {typeName}
                            {meta.third_party_verified && ' ✓'}
                          </span>
                        </td>
                        <td className="text-right py-2 text-text-muted text-xs">{lastUpdated}</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </Section>
      )}

      {/* KPI Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <div className="bg-gradient-to-br from-blue-500/10 to-blue-600/5 rounded-xl p-3 border border-blue-500/20">
          <p className="text-xs text-blue-400 uppercase tracking-wider">Carbono</p>
          <p className="text-lg font-bold text-text-primary">{data.carbon_footprint_kg_co2eq?.toFixed(1)} <span className="text-xs font-normal">kg CO₂e</span></p>
        </div>
        <div className="bg-gradient-to-br from-cyan-500/10 to-cyan-600/5 rounded-xl p-3 border border-cyan-500/20">
          <p className="text-xs text-cyan-400 uppercase tracking-wider">Água</p>
          <p className="text-lg font-bold text-text-primary">{data.water_consumption_m3?.toFixed(1)} <span className="text-xs font-normal">m³</span></p>
        </div>
        <div className="bg-gradient-to-br from-green-500/10 to-green-600/5 rounded-xl p-3 border border-green-500/20">
          <p className="text-xs text-green-400 uppercase tracking-wider">Reciclável</p>
          <p className="text-lg font-bold text-text-primary">{data.recyclability_percent?.toFixed(0)}%</p>
        </div>
        <div className="bg-gradient-to-br from-purple-500/10 to-purple-600/5 rounded-xl p-3 border border-purple-500/20">
          <p className="text-xs text-purple-400 uppercase tracking-wider">Completude</p>
          <p className="text-lg font-bold text-text-primary">{data.data_completeness_percent?.toFixed(0)}%</p>
        </div>
      </div>

      <div className="grid md:grid-cols-2 gap-4">
        {/* Identificação */}
        <Section title="Identificação" icon="🏭">
          <div className="grid grid-cols-2 gap-2 text-sm">
            <div>
              <p className="text-text-muted text-xs">Fabricante</p>
              <p className="text-text-primary">{data.manufacturer_name || '—'}</p>
            </div>
            <div>
              <p className="text-text-muted text-xs">País Origem</p>
              <p className="text-text-primary">{data.country_of_origin || '—'}</p>
            </div>
            <div>
              <p className="text-text-muted text-xs">EORI</p>
              <p className="text-text-primary font-mono text-xs">{data.manufacturer_eori || '—'}</p>
            </div>
            <div>
              <p className="text-text-muted text-xs">Site ID</p>
              <p className="text-text-primary font-mono text-xs">{data.manufacturing_site_id || '—'}</p>
            </div>
            <div className="col-span-2">
              <p className="text-text-muted text-xs">Lote/Série</p>
              <p className="text-text-primary font-mono">{data.serial_or_lot || '—'}</p>
            </div>
          </div>
        </Section>

        {/* Impacto Ambiental */}
        <Section title="Impacto Ambiental" icon="🌍">
          <div className="space-y-2">
            <div className="flex justify-between items-center text-sm">
              <span className="text-text-muted">Fabrico</span>
              <span className="text-text-primary">{data.impact_breakdown?.manufacturing_kg_co2eq?.toFixed(1) || 0} kg CO₂e</span>
            </div>
            <div className="flex justify-between items-center text-sm">
              <span className="text-text-muted">Distribuição</span>
              <span className="text-text-primary">{data.impact_breakdown?.distribution_kg_co2eq?.toFixed(1) || 0} kg CO₂e</span>
            </div>
            <div className="flex justify-between items-center text-sm">
              <span className="text-text-muted">Fim de Vida</span>
              <span className="text-text-primary">{data.impact_breakdown?.end_of_life_kg_co2eq?.toFixed(1) || 0} kg CO₂e</span>
            </div>
            <div className="border-t border-border/50 pt-2 mt-2">
              <div className="flex justify-between items-center text-sm">
                <span className="text-text-muted">Energia</span>
                <span className="text-text-primary">{data.energy_consumption_kwh || 0} kWh</span>
              </div>
            </div>
          </div>
        </Section>
      </div>

      {/* Materiais */}
      <Section title="Composição de Materiais" icon="⚗️">
        <div className="space-y-2">
          {data.materials?.map((mat: any, idx: number) => (
            <div key={idx} className="flex items-center gap-3">
              <div className="flex-1">
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-text-primary">{mat.material_name}</span>
                  <span className="text-text-muted">{mat.percentage}%</span>
                </div>
                <div className="h-2 bg-border/30 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-nikufra to-nikufra-light"
                    style={{ width: `${mat.percentage}%` }}
                  />
                </div>
              </div>
              <span className="text-xs text-text-muted w-16 text-right">{mat.mass_kg?.toFixed(1) || '—'} kg</span>
            </div>
          ))}
        </div>
      </Section>

      {/* Componentes */}
      {data.components?.length > 0 && (
        <Section title="Componentes" icon="🔧">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
            {data.components.map((comp: any, idx: number) => (
              <div key={idx} className="flex justify-between items-center bg-background/50 rounded-lg px-3 py-2 text-sm">
                <div>
                  <p className="text-text-primary">{comp.component_name}</p>
                  <p className="text-xs text-text-muted">{comp.supplier_name || 'Interno'}</p>
                </div>
                <span className="text-text-muted">{comp.weight_kg?.toFixed(1)} kg</span>
              </div>
            ))}
          </div>
        </Section>
      )}

      <div className="grid md:grid-cols-2 gap-4">
        {/* Circularidade */}
        <Section title="Circularidade & Durabilidade" icon="♻️">
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-sm text-text-muted">Conteúdo Reciclado</span>
              <span className="text-sm text-text-primary font-medium">{data.recycled_content_percent || 0}%</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-text-muted">Reciclabilidade</span>
              <span className="text-sm text-text-primary font-medium">{data.recyclability_percent || 0}%</span>
            </div>
            <div className="pt-2 space-y-2">
              <ScoreBadge value={data.durability_score || 5} label="Durabilidade" />
              <ScoreBadge value={data.reparability_score || 5} label="Reparabilidade" />
            </div>
          </div>
        </Section>

        {/* Substâncias Perigosas */}
        <Section title="Substâncias Perigosas" icon="⚠️">
          {data.hazardous_substances?.length > 0 ? (
            <div className="space-y-2">
              {data.hazardous_substances.map((sub: any, idx: number) => (
                <div key={idx} className="flex justify-between items-center text-sm bg-background/50 rounded-lg px-3 py-2">
                  <div>
                    <p className="text-text-primary">{sub.substance_name}</p>
                    <p className="text-xs text-text-muted">{sub.regulation}</p>
                  </div>
                  <span className={`px-2 py-0.5 rounded text-xs ${
                    sub.status === 'below_limit' ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
                  }`}>
                    {sub.status === 'below_limit' ? '✓ Conforme' : '⚠ Acima'}
                  </span>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-sm text-green-400">✓ Sem substâncias perigosas declaradas</p>
          )}
        </Section>
      </div>

      {/* Certificações */}
      <Section title="Certificações & Normas" icon="🏅">
        {data.certifications?.length > 0 ? (
          <div className="flex flex-wrap gap-2">
            {data.certifications.map((cert: any, idx: number) => (
              <div key={idx} className="bg-gradient-to-br from-amber-500/10 to-amber-600/5 border border-amber-500/30 rounded-lg px-3 py-2">
                <p className="text-sm font-medium text-amber-400">{cert.scheme}</p>
                <p className="text-xs text-text-muted">{cert.issuer}</p>
                {cert.valid_until && (
                  <p className="text-xs text-text-muted">Válido até {new Date(cert.valid_until).toLocaleDateString('pt-PT')}</p>
                )}
              </div>
            ))}
          </div>
        ) : (
          <p className="text-sm text-text-muted">Sem certificações registadas</p>
        )}
      </Section>

      {/* Auditorias */}
      {data.third_party_audits?.length > 0 && (
        <Section title="Auditorias de Terceiros" icon="🔍">
          <div className="space-y-2">
            {data.third_party_audits.map((audit: any, idx: number) => (
              <div key={idx} className="flex justify-between items-center bg-background/50 rounded-lg px-3 py-2">
                <div>
                  <p className="text-sm text-text-primary">{audit.auditor_name}</p>
                  <p className="text-xs text-text-muted">{audit.scope}</p>
                </div>
                <div className="text-right">
                  <span className="text-xs px-2 py-0.5 bg-green-500/20 text-green-400 rounded">{audit.result}</span>
                  <p className="text-xs text-text-muted mt-1">{new Date(audit.date).toLocaleDateString('pt-PT')}</p>
                </div>
              </div>
            ))}
          </div>
        </Section>
      )}

      {/* Compliance Radar */}
      {compliance && (
        <Section title="Compliance Radar" icon="🛡️">
          <div className="space-y-4">
            {/* Scores - Gauges */}
            <div className={`grid gap-4 ${compliance.cbam_score !== null ? 'grid-cols-3' : 'grid-cols-2'}`}>
              {/* ESPR Gauge */}
              <div className="text-center">
                <div className="relative w-24 h-24 mx-auto mb-2">
                  <svg className="transform -rotate-90 w-24 h-24">
                    <circle
                      cx="48"
                      cy="48"
                      r="40"
                      stroke="currentColor"
                      strokeWidth="8"
                      fill="none"
                      className="text-border/30"
                    />
                    <circle
                      cx="48"
                      cy="48"
                      r="40"
                      stroke="currentColor"
                      strokeWidth="8"
                      fill="none"
                      strokeDasharray={`${2 * Math.PI * 40}`}
                      strokeDashoffset={`${2 * Math.PI * 40 * (1 - compliance.espr_score / 100)}`}
                      className={`${
                        compliance.espr_score >= 80 ? 'text-green-400' :
                        compliance.espr_score >= 60 ? 'text-amber-400' : 'text-red-400'
                      }`}
                    />
                  </svg>
                  <div className="absolute inset-0 flex items-center justify-center">
                    <span className={`text-lg font-bold ${
                      compliance.espr_score >= 80 ? 'text-green-400' :
                      compliance.espr_score >= 60 ? 'text-amber-400' : 'text-red-400'
                    }`}>
                      {compliance.espr_score.toFixed(0)}
                    </span>
                  </div>
                </div>
                <p className="text-xs text-text-muted font-medium">ESPR</p>
              </div>

              {/* CBAM Gauge (if applicable) */}
              {compliance.cbam_score !== null && (
                <div className="text-center">
                  <div className="relative w-24 h-24 mx-auto mb-2">
                    <svg className="transform -rotate-90 w-24 h-24">
                      <circle
                        cx="48"
                        cy="48"
                        r="40"
                        stroke="currentColor"
                        strokeWidth="8"
                        fill="none"
                        className="text-border/30"
                      />
                      <circle
                        cx="48"
                        cy="48"
                        r="40"
                        stroke="currentColor"
                        strokeWidth="8"
                        fill="none"
                        strokeDasharray={`${2 * Math.PI * 40}`}
                        strokeDashoffset={`${2 * Math.PI * 40 * (1 - compliance.cbam_score / 100)}`}
                        className={`${
                          compliance.cbam_score >= 80 ? 'text-green-400' :
                          compliance.cbam_score >= 60 ? 'text-amber-400' : 'text-red-400'
                        }`}
                      />
                    </svg>
                    <div className="absolute inset-0 flex items-center justify-center">
                      <span className={`text-lg font-bold ${
                        compliance.cbam_score >= 80 ? 'text-green-400' :
                        compliance.cbam_score >= 60 ? 'text-amber-400' : 'text-red-400'
                      }`}>
                        {compliance.cbam_score.toFixed(0)}
                      </span>
                    </div>
                  </div>
                  <p className="text-xs text-text-muted font-medium">CBAM</p>
                </div>
              )}

              {/* CSRD Gauge */}
              <div className="text-center">
                <div className="relative w-24 h-24 mx-auto mb-2">
                  <svg className="transform -rotate-90 w-24 h-24">
                    <circle
                      cx="48"
                      cy="48"
                      r="40"
                      stroke="currentColor"
                      strokeWidth="8"
                      fill="none"
                      className="text-border/30"
                    />
                    <circle
                      cx="48"
                      cy="48"
                      r="40"
                      stroke="currentColor"
                      strokeWidth="8"
                      fill="none"
                      strokeDasharray={`${2 * Math.PI * 40}`}
                      strokeDashoffset={`${2 * Math.PI * 40 * (1 - compliance.csrd_score / 100)}`}
                      className={`${
                        compliance.csrd_score >= 80 ? 'text-green-400' :
                        compliance.csrd_score >= 60 ? 'text-amber-400' : 'text-red-400'
                      }`}
                    />
                  </svg>
                  <div className="absolute inset-0 flex items-center justify-center">
                    <span className={`text-lg font-bold ${
                      compliance.csrd_score >= 80 ? 'text-green-400' :
                      compliance.csrd_score >= 60 ? 'text-amber-400' : 'text-red-400'
                    }`}>
                      {compliance.csrd_score.toFixed(0)}
                    </span>
                  </div>
                </div>
                <p className="text-xs text-text-muted font-medium">CSRD</p>
              </div>
            </div>

            {/* Critical Gaps */}
            {compliance.critical_gaps && compliance.critical_gaps.length > 0 && (
              <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4">
                <h5 className="font-semibold text-red-400 mb-2 flex items-center gap-2">
                  <AlertCircle className="w-4 h-4" />
                  Gaps Críticos
                </h5>
                <ul className="space-y-1">
                  {compliance.critical_gaps.map((gap: string, idx: number) => (
                    <li key={idx} className="text-sm text-text-muted">• {gap}</li>
                  ))}
                </ul>
              </div>
            )}

            {/* Recommended Actions */}
            {compliance.recommended_actions && compliance.recommended_actions.length > 0 && (
              <div className="bg-amber-500/10 border border-amber-500/30 rounded-lg p-4">
                <h5 className="font-semibold text-amber-400 mb-2 flex items-center gap-2">
                  <Shield className="w-4 h-4" />
                  Ações Recomendadas
                </h5>
                <ul className="space-y-1">
                  {compliance.recommended_actions.map((action: string, idx: number) => (
                    <li key={idx} className="text-sm text-text-muted">• {action}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </Section>
      )}

      {/* Footer Metadata */}
      <div className="border-t border-border/50 pt-4 flex justify-between items-center text-xs text-text-muted">
        <div>
          <p>DPP ID: <span className="font-mono">{data.dpp_id}</span></p>
          <p>Criado: {new Date(data.created_at).toLocaleString('pt-PT')}</p>
        </div>
        <div className="text-right">
          <p>Atualizado: {new Date(data.updated_at).toLocaleString('pt-PT')}</p>
          <p>QR Slug: <span className="font-mono">{data.qr_slug}</span></p>
        </div>
      </div>
    </div>
  )
}
