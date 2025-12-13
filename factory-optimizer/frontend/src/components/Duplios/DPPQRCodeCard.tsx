import { useEffect, useState } from 'react'
import { apiGetDPPQRCode } from '../../services/dupliosApi'

interface Props {
  dppId: string
  size?: 'sm' | 'md' | 'lg'
  showDownload?: boolean
}

export const DPPQRCodeCard = ({ dppId, size = 'sm', showDownload = true }: Props) => {
  const [url, setUrl] = useState<string>('')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    setLoading(true)
    apiGetDPPQRCode(dppId)
      .then(setUrl)
      .catch(() => setUrl(''))
      .finally(() => setLoading(false))
  }, [dppId])

  const sizeClasses = {
    sm: 'w-16 h-16',
    md: 'w-24 h-24',
    lg: 'w-32 h-32',
  }

  if (loading) {
    return (
      <div className={`${sizeClasses[size]} bg-background/50 rounded-lg flex items-center justify-center`}>
        <div className="animate-spin w-4 h-4 border-2 border-nikufra border-t-transparent rounded-full" />
      </div>
    )
  }

  if (!url) {
    return (
      <div className={`${sizeClasses[size]} bg-background/50 rounded-lg flex items-center justify-center`}>
        <span className="text-xs text-text-muted">QR N/A</span>
      </div>
    )
  }

  return (
    <div className="flex flex-col items-center gap-1">
      <div className="bg-white p-1 rounded-lg shadow-lg">
        <img src={url} alt="QR Code DPP" className={`${sizeClasses[size]} rounded`} />
      </div>
      {showDownload && (
        <a
          href={url}
          download={`dpp-${dppId}.png`}
          className="text-xs text-nikufra hover:text-nikufra-light transition-colors flex items-center gap-1"
        >
          <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
          </svg>
          Download
        </a>
      )}
    </div>
  )
}
