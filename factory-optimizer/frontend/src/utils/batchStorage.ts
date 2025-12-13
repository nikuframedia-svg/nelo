/**
 * Persistência do batch_id no localStorage
 * Garante que o Excel não é esquecido após refresh
 */

const BATCH_ID_KEY = 'prodplan_batch_id'
const BATCH_TIMESTAMP_KEY = 'prodplan_batch_timestamp'

export const batchStorage = {
  get(): string | null {
    if (typeof window === 'undefined') return null
    const batchId = localStorage.getItem(BATCH_ID_KEY)
    const timestamp = localStorage.getItem(BATCH_TIMESTAMP_KEY)
    
    // Se o batch_id tem mais de 7 dias, limpar (dados antigos)
    if (batchId && timestamp) {
      const age = Date.now() - parseInt(timestamp, 10)
      const maxAge = 7 * 24 * 60 * 60 * 1000 // 7 dias
      if (age > maxAge) {
        localStorage.removeItem(BATCH_ID_KEY)
        localStorage.removeItem(BATCH_TIMESTAMP_KEY)
        return null
      }
    }
    
    return batchId
  },
  
  set(batchId: string): void {
    if (typeof window === 'undefined') return
    localStorage.setItem(BATCH_ID_KEY, batchId)
    localStorage.setItem(BATCH_TIMESTAMP_KEY, Date.now().toString())
  },
  
  clear(): void {
    if (typeof window === 'undefined') return
    localStorage.removeItem(BATCH_ID_KEY)
    localStorage.removeItem(BATCH_TIMESTAMP_KEY)
  },
  
  exists(): boolean {
    return this.get() !== null
  },
}

