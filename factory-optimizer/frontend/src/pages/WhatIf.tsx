import { useState } from 'react'
import toast from 'react-hot-toast'
import { ScenarioPreview, type ScenarioPreviewData } from '../components/ScenarioPreview'
import { ScenarioComparison, type ScenarioComparisonData } from '../components/ScenarioComparison'
import { apiDescribeScenario, apiCompareScenario } from '../services/nikufraApi'
import ZDMSimulator from './ZDMSimulator'
import { FlaskConical, Shield } from 'lucide-react'

type TabId = 'whatif' | 'zdm'

export const WhatIf = () => {
  const [activeTab, setActiveTab] = useState<TabId>('whatif')
  const [scenarioText, setScenarioText] = useState('')
  const [preview, setPreview] = useState<ScenarioPreviewData | null>(null)
  const [comparison, setComparison] = useState<ScenarioComparisonData | null>(null)
  const [isDescribing, setIsDescribing] = useState(false)
  const [isComparing, setIsComparing] = useState(false)

  const handleDescribe = async () => {
    if (!scenarioText.trim()) {
      toast.error('Escreve um cenário antes de pedir a descrição.')
      return
    }
    try {
      setIsDescribing(true)
      const data = await apiDescribeScenario(scenarioText.trim())
      setPreview(data)
      toast.success('Cenário descrito com sucesso.')
    } catch (error) {
      console.error(error)
      toast.error('Não foi possível descrever o cenário.')
    } finally {
      setIsDescribing(false)
    }
  }

  const handleCompare = async () => {
    if (!scenarioText.trim()) {
      toast.error('Escreve um cenário antes de comparar.')
      return
    }
    try {
      setIsComparing(true)
      const data = await apiCompareScenario(scenarioText.trim())
      setComparison(data)
      toast.success('Comparação concluída.')
    } catch (error) {
      console.error(error)
      toast.error('Não foi possível comparar o cenário.')
    } finally {
      setIsComparing(false)
    }
  }

  const tabs = [
    { id: 'whatif' as const, label: 'What-If', icon: <FlaskConical className="w-4 h-4" /> },
    { id: 'zdm' as const, label: 'ZDM Resiliência', icon: <Shield className="w-4 h-4" /> },
  ]

  return (
    <div className="space-y-6">
      <header className="space-y-2">
        <p className="text-xs uppercase tracking-[0.4em] text-text-muted">Laboratório de cenários</p>
        <h2 className="text-2xl font-semibold text-text-primary">What-if & Simulação de Resiliência</h2>
        <p className="max-w-3xl text-sm text-text-muted">
          Analisa cenários hipotéticos e simula a resiliência do plano face a falhas e perturbações.
        </p>
      </header>

      {/* Tab Navigation */}
      <div className="border-b border-border">
        <nav className="flex gap-1 -mb-px">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`
                group flex items-center gap-2 px-4 py-3 text-sm font-medium 
                border-b-2 transition-all duration-200
                ${activeTab === tab.id
                  ? 'border-nikufra text-nikufra'
                  : 'border-transparent text-text-muted hover:text-text-primary hover:border-border'
                }
              `}
            >
              <span className={activeTab === tab.id ? 'text-nikufra' : 'text-text-muted group-hover:text-text-primary'}>
                {tab.icon}
              </span>
              {tab.label}
            </button>
          ))}
        </nav>
      </div>

      {/* Tab Content */}
      {activeTab === 'whatif' && (
        <div className="space-y-8">
          <section className="rounded-2xl border border-border bg-surface p-6 shadow-glow">
            <label className="flex w-full flex-col gap-3 text-xs font-semibold uppercase tracking-[0.3em] text-text-muted">
              Descrição do cenário
              <textarea
                value={scenarioText}
                onChange={(event) => setScenarioText(event.target.value)}
                placeholder="Ex.: Quero introduzir a máquina M-999 com -20% de tempo no corte dos artigos ART-300 e ART-500."
                className="h-40 rounded-2xl border border-border bg-background px-4 py-3 text-sm text-text-primary outline-none transition focus:border-nikufra"
              />
            </label>
            <div className="mt-4 flex flex-wrap gap-3">
              <button
                onClick={handleDescribe}
                disabled={isDescribing}
                className="rounded-2xl border border-nikufra bg-nikufra px-4 py-2 text-sm font-semibold text-background transition hover:bg-nikufra-hover disabled:opacity-60"
              >
                {isDescribing ? 'A descrever...' : 'Descrever cenário'}
              </button>
              <button
                onClick={handleCompare}
                disabled={isComparing}
                className="rounded-2xl border border-border bg-background px-4 py-2 text-sm font-semibold text-text-primary transition hover:border-nikufra hover:text-nikufra disabled:opacity-60"
              >
                {isComparing ? 'A comparar...' : 'Comparar cenário'}
              </button>
            </div>
          </section>

          <div className="grid gap-6 lg:grid-cols-2">
            <ScenarioPreview data={preview} loading={isDescribing} />
            <ScenarioComparison data={comparison} loading={isComparing} />
          </div>
        </div>
      )}

      {activeTab === 'zdm' && <ZDMSimulator />}
    </div>
  )
}

