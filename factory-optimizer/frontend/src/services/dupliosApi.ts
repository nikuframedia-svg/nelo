import { API_BASE_URL } from '../config/api'

export async function apiCreateDPP(payload: any) {
  const res = await fetch(`${API_BASE_URL}/duplios/dpp`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!res.ok) throw new Error('Erro ao criar DPP')
  return res.json()
}

export async function apiListDPPs() {
  const res = await fetch(`${API_BASE_URL}/duplios/dpp`)
  if (!res.ok) throw new Error('Erro ao listar DPPs')
  return res.json()
}

export async function apiGetDPP(dppId: string) {
  const res = await fetch(`${API_BASE_URL}/duplios/dpp/${dppId}`)
  if (!res.ok) throw new Error('Erro ao obter DPP')
  return res.json()
}

export async function apiGetDPPByGTIN(gtin: string) {
  const res = await fetch(`${API_BASE_URL}/duplios/dpp/by-gtin/${gtin}`)
  if (!res.ok) throw new Error('Erro ao obter DPP pelo GTIN')
  return res.json()
}

export async function apiGetDPPQRCode(dppId: string): Promise<string> {
  const res = await fetch(`${API_BASE_URL}/duplios/dpp/${dppId}/qrcode`)
  if (!res.ok) throw new Error('Erro ao obter QR Code')
  const blob = await res.blob()
  return URL.createObjectURL(blob)
}

export async function apiGapFillLite(dppId: string, force: boolean = false) {
  const res = await fetch(`${API_BASE_URL}/duplios/dpp/${dppId}/gap-fill-lite`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ force }),
  })
  if (!res.ok) throw new Error('Erro ao preencher campos automaticamente')
  return res.json()
}

export async function apiGetComplianceRadar(dppId: string) {
  const res = await fetch(`${API_BASE_URL}/duplios/dpp/${dppId}/compliance-radar`)
  if (!res.ok) throw new Error('Erro ao obter compliance radar')
  return res.json()
}

export async function apiGetComplianceSummary(dppId: string) {
  const res = await fetch(`${API_BASE_URL}/duplios/dpp/${dppId}/compliance-summary`)
  if (!res.ok) throw new Error('Erro ao obter compliance summary')
  return res.json()
}

export async function apiGetTrustIndex(dppId: string) {
  const res = await fetch(`${API_BASE_URL}/duplios/dpp/${dppId}/trust-index`)
  if (!res.ok) throw new Error('Erro ao obter Trust Index')
  return res.json()
}

export async function apiRecalculateTrustIndex(dppId: string) {
  const res = await fetch(`${API_BASE_URL}/duplios/dpp/${dppId}/trust-index/recalculate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
  })
  if (!res.ok) throw new Error('Erro ao recalcular Trust Index')
  return res.json()
}
