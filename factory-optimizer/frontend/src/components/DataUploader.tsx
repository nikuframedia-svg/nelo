import { useState } from 'react'
import { Upload, X, CheckCircle, AlertCircle, FileSpreadsheet, Loader2 } from 'lucide-react'
import { API_BASE_URL } from '../config/api'

interface UploadCardProps {
  title: string
  description: string
  endpoint: string
  icon: string
}

const UploadCard = ({ title, description, endpoint, icon }: UploadCardProps) => {
  const [file, setFile] = useState<File | null>(null)
  const [uploading, setUploading] = useState(false)
  const [result, setResult] = useState<any | null>(null)
  const [lastUpload, setLastUpload] = useState<{ file: string; date: string; status: string } | null>(null)

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0])
      setResult(null)
    }
  }

  const handleUpload = async () => {
    if (!file) return

    setUploading(true)
    setResult(null)

    try {
      const formData = new FormData()
      formData.append('file', file)

      const res = await fetch(`${API_BASE_URL}${endpoint}`, {
        method: 'POST',
        body: formData,
      })

      if (!res.ok) {
        const error = await res.json()
        throw new Error(error.detail || 'Erro ao fazer upload')
      }

      const data = await res.json()
      setResult(data)
      
      // Atualizar último upload
      setLastUpload({
        file: file.name,
        date: new Date().toLocaleString('pt-PT'),
        status: data.success ? 'OK' : 'Com avisos',
      })
    } catch (error: any) {
      setResult({
        success: false,
        errors: [error.message || 'Erro desconhecido'],
      })
    } finally {
      setUploading(false)
    }
  }

  return (
    <div className="bg-surface border border-border rounded-xl p-6 space-y-4">
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-3">
          <div className="text-3xl">{icon}</div>
          <div>
            <h3 className="text-lg font-semibold text-text-primary">{title}</h3>
            <p className="text-sm text-text-muted">{description}</p>
          </div>
        </div>
      </div>

      {/* Upload Area */}
      <div className="space-y-3">
        <div className="flex items-center gap-3">
          <label className="flex-1 cursor-pointer">
            <input
              type="file"
              accept=".xlsx,.xls"
              onChange={handleFileChange}
              className="hidden"
              disabled={uploading}
            />
            <div className="flex items-center gap-2 px-4 py-2 bg-background border border-border rounded-lg hover:bg-background/80 transition">
              <FileSpreadsheet className="w-4 h-4 text-text-muted" />
              <span className="text-sm text-text-primary">
                {file ? file.name : 'Selecionar ficheiro Excel...'}
              </span>
            </div>
          </label>
          <button
            onClick={handleUpload}
            disabled={!file || uploading}
            className="px-4 py-2 bg-nikufra hover:bg-nikufra/90 text-white rounded-lg transition disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
          >
            {uploading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                <span>A carregar...</span>
              </>
            ) : (
              <>
                <Upload className="w-4 h-4" />
                <span>Carregar</span>
              </>
            )}
          </button>
        </div>

        {/* Last Upload Info */}
        {lastUpload && (
          <div className="text-xs text-text-muted flex items-center gap-2">
            <CheckCircle className="w-3 h-3" />
            <span>
              Último: {lastUpload.file} ({lastUpload.date}) - {lastUpload.status}
            </span>
          </div>
        )}

        {/* Result */}
        {result && (
          <div className={`rounded-lg p-4 ${
            result.success
              ? 'bg-green-500/10 border border-green-500/30'
              : 'bg-red-500/10 border border-red-500/30'
          }`}>
            <div className="flex items-start gap-2 mb-2">
              {result.success ? (
                <CheckCircle className="w-5 h-5 text-green-400 flex-shrink-0 mt-0.5" />
              ) : (
                <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
              )}
              <div className="flex-1">
                <p className={`font-semibold ${
                  result.success ? 'text-green-400' : 'text-red-400'
                }`}>
                  {result.success ? 'Importação bem-sucedida' : 'Importação com erros'}
                </p>
                <p className="text-sm text-text-muted mt-1">
                  {result.imported_count} linhas importadas
                  {result.failed_count > 0 && `, ${result.failed_count} falhadas`}
                </p>
              </div>
            </div>

            {/* Warnings */}
            {result.warnings && result.warnings.length > 0 && (
              <div className="mt-3 space-y-1">
                <p className="text-xs font-semibold text-amber-400">Avisos:</p>
                <ul className="text-xs text-text-muted space-y-1 max-h-32 overflow-y-auto">
                  {result.warnings.slice(0, 5).map((warning: string, idx: number) => (
                    <li key={idx}>• {warning}</li>
                  ))}
                </ul>
                {result.warnings.length > 5 && (
                  <p className="text-xs text-text-muted">
                    +{result.warnings.length - 5} mais avisos
                  </p>
                )}
              </div>
            )}

            {/* Errors */}
            {result.errors && result.errors.length > 0 && (
              <div className="mt-3 space-y-1">
                <p className="text-xs font-semibold text-red-400">Erros:</p>
                <ul className="text-xs text-text-muted space-y-1 max-h-32 overflow-y-auto">
                  {result.errors.slice(0, 5).map((error: string, idx: number) => (
                    <li key={idx}>• {error}</li>
                  ))}
                </ul>
                {result.errors.length > 5 && (
                  <p className="text-xs text-text-muted">
                    +{result.errors.length - 5} mais erros
                  </p>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

interface DataUploaderProps {
  isOpen: boolean
  onClose: () => void
}

export const DataUploader = ({ isOpen, onClose }: DataUploaderProps) => {
  if (!isOpen) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
      <div className="bg-surface border border-border rounded-2xl w-full max-w-4xl max-h-[90vh] overflow-y-auto m-4">
        {/* Header */}
        <div className="sticky top-0 bg-surface border-b border-border p-6 flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold text-text-primary">Carregar Dados Operacionais</h2>
            <p className="text-sm text-text-muted mt-1">
              Importe dados de Excel para o sistema (Ordens, Movimentos, RH, Máquinas)
            </p>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-background rounded-lg transition"
          >
            <X className="w-5 h-5 text-text-muted" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 space-y-6">
          {/* Orders */}
          <UploadCard
            title="Ordens de Produção"
            description="Importa ordens de produção com routing e quantidades"
            endpoint="/ops-ingestion/orders/excel"
            icon="📋"
          />

          {/* Inventory Moves */}
          <UploadCard
            title="Movimentos Internos"
            description="Importa movimentos de inventário/WIP entre estações"
            endpoint="/ops-ingestion/inventory-moves/excel"
            icon="🔄"
          />

          {/* HR */}
          <UploadCard
            title="Recursos Humanos"
            description="Importa dados de colaboradores, skills e turnos"
            endpoint="/ops-ingestion/hr/excel"
            icon="👥"
          />

          {/* Machines */}
          <UploadCard
            title="Máquinas & Linhas"
            description="Importa dados de máquinas, capacidade e manutenção"
            endpoint="/ops-ingestion/machines/excel"
            icon="⚙️"
          />
        </div>

        {/* Footer */}
        <div className="sticky bottom-0 bg-surface border-t border-border p-4 flex justify-end">
          <button
            onClick={onClose}
            className="px-6 py-2 bg-background border border-border rounded-lg hover:bg-background/80 transition"
          >
            Fechar
          </button>
        </div>
      </div>
    </div>
  )
}
