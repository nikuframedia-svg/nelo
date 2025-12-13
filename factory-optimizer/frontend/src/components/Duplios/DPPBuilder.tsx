import { useState } from 'react'
import { apiCreateDPP } from '../../services/dupliosApi'

const steps = ['Identificação', 'Composição', 'Impacto', 'Circularidade', 'Verificação']

const initialForm = {
  gtin: '',
  product_name: '',
  product_category: '',
  manufacturer_name: '',
  manufacturer_eori: '',
  manufacturing_site_id: '',
  country_of_origin: '',
  serial_or_lot: '',
  materials: [
    { material_name: '', material_type: '', percentage: 100, mass_kg: 1 },
  ],
  components: [],
  carbon_footprint_kg_co2eq: 0,
  impact_breakdown: {
    manufacturing_kg_co2eq: 0,
    distribution_kg_co2eq: 0,
    end_of_life_kg_co2eq: 0,
  },
  water_consumption_m3: 0,
  energy_consumption_kwh: 0,
  recycled_content_percent: 0,
  recyclability_percent: 0,
  durability_score: 5,
  reparability_score: 5,
  hazardous_substances: [],
  certifications: [],
  third_party_audits: [],
  trust_index: 60,
  data_completeness_percent: 60,
}

export const DPPBuilder = ({ onCreated }: { onCreated?: () => void }) => {
  const [step, setStep] = useState(0)
  const [form, setForm] = useState(initialForm)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const next = () => setStep((prev) => Math.min(prev + 1, steps.length - 1))
  const prev = () => setStep((prev) => Math.max(prev - 1, 0))

  const handleSubmit = async () => {
    setLoading(true)
    setError(null)
    try {
      await apiCreateDPP(form)
      onCreated?.()
      setStep(0)
      setForm(initialForm)
    } catch (err) {
      setError('Falha ao criar DPP')
    } finally {
      setLoading(false)
    }
  }

  const renderStep = () => {
    switch (step) {
      case 0:
        return (
          <div className="grid gap-3">
            <input className="input" placeholder="GTIN" value={form.gtin} onChange={(e) => setForm({ ...form, gtin: e.target.value })} />
            <input className="input" placeholder="Nome do produto" value={form.product_name} onChange={(e) => setForm({ ...form, product_name: e.target.value })} />
            <input className="input" placeholder="Categoria" value={form.product_category} onChange={(e) => setForm({ ...form, product_category: e.target.value })} />
            <input className="input" placeholder="Fabricante" value={form.manufacturer_name} onChange={(e) => setForm({ ...form, manufacturer_name: e.target.value })} />
            <input className="input" placeholder="País de origem" value={form.country_of_origin} onChange={(e) => setForm({ ...form, country_of_origin: e.target.value })} />
          </div>
        )
      case 1:
        return (
          <div>
            {form.materials.map((mat, idx) => (
              <div key={idx} className="grid grid-cols-4 gap-2 mb-2">
                <input className="input" placeholder="Material" value={mat.material_name} onChange={(e) => {
                  const materials = [...form.materials]
                  materials[idx].material_name = e.target.value
                  setForm({ ...form, materials })
                }} />
                <input className="input" placeholder="Tipo" value={mat.material_type} onChange={(e) => {
                  const materials = [...form.materials]
                  materials[idx].material_type = e.target.value
                  setForm({ ...form, materials })
                }} />
                <input className="input" type="number" value={mat.percentage} onChange={(e) => {
                  const materials = [...form.materials]
                  materials[idx].percentage = Number(e.target.value)
                  setForm({ ...form, materials })
                }} />
                <input className="input" type="number" value={mat.mass_kg || 0} onChange={(e) => {
                  const materials = [...form.materials]
                  materials[idx].mass_kg = Number(e.target.value)
                  setForm({ ...form, materials })
                }} />
              </div>
            ))}
          </div>
        )
      case 2:
        return (
          <div className="grid gap-3">
            <input className="input" type="number" placeholder="Carbono total" value={form.carbon_footprint_kg_co2eq} onChange={(e) => setForm({ ...form, carbon_footprint_kg_co2eq: Number(e.target.value) })} />
            <input className="input" type="number" placeholder="Água" value={form.water_consumption_m3} onChange={(e) => setForm({ ...form, water_consumption_m3: Number(e.target.value) })} />
            <input className="input" type="number" placeholder="Energia" value={form.energy_consumption_kwh} onChange={(e) => setForm({ ...form, energy_consumption_kwh: Number(e.target.value) })} />
          </div>
        )
      case 3:
        return (
          <div className="grid gap-3">
            <input className="input" type="number" placeholder="Conteúdo reciclado" value={form.recycled_content_percent} onChange={(e) => setForm({ ...form, recycled_content_percent: Number(e.target.value) })} />
            <input className="input" type="number" placeholder="Reciclabilidade" value={form.recyclability_percent} onChange={(e) => setForm({ ...form, recyclability_percent: Number(e.target.value) })} />
            <input className="input" type="number" placeholder="Durabilidade" value={form.durability_score} onChange={(e) => setForm({ ...form, durability_score: Number(e.target.value) })} />
          </div>
        )
      case 4:
        return (
          <div className="space-y-3">
            <p>Revê os dados antes de publicar.</p>
            <pre className="bg-background/50 p-3 rounded text-xs h-32 overflow-auto">{JSON.stringify(form, null, 2)}</pre>
          </div>
        )
      default:
        return null
    }
  }

  return (
    <div className="rounded-2xl border border-border bg-surface p-6 space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-text-primary">DPP Builder</h3>
        <span className="text-xs text-text-muted">Passo {step + 1} / {steps.length} – {steps[step]}</span>
      </div>
      {renderStep()}
      {error && <p className="text-xs text-red-400">{error}</p>}
      <div className="flex justify-between">
        <button onClick={prev} disabled={step === 0} className="btn-secondary">Anterior</button>
        {step < steps.length - 1 ? (
          <button onClick={next} className="btn-primary">Seguinte</button>
        ) : (
          <button onClick={handleSubmit} disabled={loading} className="btn-primary">
            {loading ? 'A criar...' : 'Criar DPP'}
          </button>
        )}
      </div>
    </div>
  )
}
