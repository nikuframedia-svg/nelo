import ReactMarkdown from 'react-markdown'
import { useQuery } from '@tanstack/react-query'
import remarkGfm from 'remark-gfm'
import Skeleton from 'react-loading-skeleton'
import 'react-loading-skeleton/dist/skeleton.css'
import clsx from 'clsx'
import api from '../utils/api'

interface InsightBannerProps {
  className?: string
  refreshIntervalMs?: number
  title?: string
  mode?: 'planeamento' | 'resumo'
}

interface InsightSummary {
  summary: string
  model: string
  generated_at: string
  detail?: string
}

// Função para processar o texto e juntar linhas que fazem parte da mesma frase
const processTextForDisplay = (text: string): string => {
  if (!text) return ''
  
  // Dividir o texto em linhas
  const lines = text.split('\n').map(line => line.trim()).filter(line => line.length > 0)
  
  if (lines.length === 0) return ''
  
  const processed: string[] = []
  let currentParagraph = ''
  let inList = false
  
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]
    const nextLine = i < lines.length - 1 ? lines[i + 1] : null
    const prevLine = i > 0 ? lines[i - 1] : null
    
    // Detectar se é item de lista com marcador (começa com -, *, •, ou número)
    const isListItemWithMarker = /^[-*•]\s/.test(line) || /^\d+[.)]\s/.test(line)
    
    // Detectar se é item de lista sem marcador (linha que termina com vírgula e está após cabeçalho de lista ou outro item)
    const isListItemWithoutMarker = line.endsWith(',') && 
                                    line.length < 80 && 
                                    prevLine && 
                                    (prevLine.endsWith(':') || prevLine.endsWith(','))
    
    // Detectar se é último item de lista (termina com ponto/vírgula e está após itens com vírgula, ou é linha curta após cabeçalho)
    const isLastListItem = (line.endsWith('.') || line.endsWith(',')) && 
                          line.length < 100 && 
                          prevLine && 
                          (prevLine.endsWith(',') || prevLine.endsWith(':'))
    
    // Detectar cabeçalho de lista (termina com : e próxima linha é item de lista)
    const isListHeader = line.endsWith(':') && 
                        nextLine && 
                        (isListItemWithMarker || nextLine.endsWith(','))
    
    // Se for cabeçalho de lista, finalizar parágrafo atual e começar lista
    if (isListHeader) {
      if (currentParagraph) {
        processed.push(currentParagraph.trim())
        currentParagraph = ''
      }
      processed.push(line)
      inList = true
      continue
    }
    
    // Se for item de lista (com ou sem marcador), manter separado
    if (isListItemWithMarker || isListItemWithoutMarker || isLastListItem) {
      if (currentParagraph) {
        processed.push(currentParagraph.trim())
        currentParagraph = ''
      }
      processed.push(line)
      inList = true
      continue
    }
    
    // Se saiu da lista (linha não é item de lista mas estava em lista), finalizar lista
    if (inList && !isListItemWithMarker && !isListItemWithoutMarker && !isLastListItem) {
      inList = false
    }
    
    // Para linhas normais (não são itens de lista)
    const endsWithPunctuation = /[.!?]$/.test(line)
    const isShortLine = line.length < 50 && !endsWithPunctuation
    
    // Se for linha curta sem pontuação, provavelmente continua na próxima linha
    if (isShortLine && !endsWithPunctuation) {
      if (currentParagraph) {
        currentParagraph += ' ' + line
      } else {
        currentParagraph = line
      }
    } else if (endsWithPunctuation) {
      // Linha que termina com pontuação - finalizar parágrafo
      if (currentParagraph) {
        currentParagraph += ' ' + line
        processed.push(currentParagraph.trim())
        currentParagraph = ''
      } else {
        processed.push(line)
      }
    } else {
      // Linha normal - juntar se não terminar com pontuação
      if (currentParagraph) {
        currentParagraph += ' ' + line
      } else {
        currentParagraph = line
      }
    }
  }
  
  // Adicionar último parágrafo se existir
  if (currentParagraph) {
    processed.push(currentParagraph.trim())
  }
  
  return processed.join('\n\n')
}

export const InsightBanner = ({ className, refreshIntervalMs = 600_000, title = '🧠 Resumo do Sistema', mode = 'resumo' }: InsightBannerProps) => {
  const { data, isLoading, error } = useQuery<InsightSummary>({
    queryKey: ['insight', mode === 'planeamento' ? 'planeamento' : 'resumo'],
    queryFn: async () => {
      const response = await api.get('/insights/generate', { params: { mode: mode === 'planeamento' ? 'planeamento' : 'resumo' } })
      // Se a resposta contém detail (mensagem do backend), retornar com detail
      return {
        summary: response.data.insight || response.data.answer || '',
        model: response.data.model || 'offline',
        generated_at: new Date().toISOString(),
        detail: response.data.detail, // Mensagem do backend (ex: "Modelo offline — iniciar Ollama.")
      }
    },
    refetchInterval: refreshIntervalMs,
    retry: 1,
  })

  if (isLoading) {
    return <Skeleton height={110} baseColor="#101010" highlightColor="#1b1b1b" className={className} />
  }

  if (error || data?.detail) {
    // Mostrar a mensagem do backend (data.detail) ou a mensagem de erro
    const message = data?.detail || error?.message || 'Erro ao carregar resumo'
    return (
      <div
        className={clsx(
          'rounded-2xl border border-warning/40 bg-warning/10 p-4 text-sm text-warning shadow-glow',
          className,
        )}
      >
        {message}
      </div>
    )
  }

  if (!data) return null

  return (
    <div
      className={clsx(
        'relative overflow-hidden rounded-2xl border border-border bg-gradient-to-br from-[#0b1411] via-[#0f1f1a] to-[#142b24] p-5 shadow-glow',
        className,
      )}
    >
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_left,rgba(110,247,185,0.25),transparent_55%)]" />
      <div className="relative space-y-4">
        <p className="text-xs uppercase tracking-[0.35em] text-nikufra/80 mb-4">{title}</p>
        <div className="text-sm text-text-primary leading-relaxed">
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            className="prose prose-invert prose-sm max-w-none [&>p]:mb-3 [&>p]:leading-[1.7] [&>p]:text-justify [&>ul]:mb-3 [&>ul]:mt-2 [&>ul]:space-y-1.5 [&>ul]:list-disc [&>ul]:ml-5 [&>li]:leading-relaxed [&>li]:mb-1"
            components={{
              p: ({ children }) => (
                <p className="mb-3 leading-[1.7] text-justify text-sm text-text-primary break-words">
                  {children}
                </p>
              ),
              ul: ({ children }) => (
                <ul className="mb-3 mt-2 space-y-1.5 list-disc ml-5 text-sm text-text-primary">
                  {children}
                </ul>
              ),
              li: ({ children }) => (
                <li className="mb-1 leading-relaxed text-text-primary">
                  {children}
                </li>
              ),
            }}
          >
            {processTextForDisplay(data.summary)}
          </ReactMarkdown>
        </div>
        <p className="text-xs text-text-muted/80 pt-3 border-t border-border/40 mt-4">
          Atualizado: {new Date(data.generated_at).toLocaleString('pt-PT')}
        </p>
      </div>
    </div>
  )
}
