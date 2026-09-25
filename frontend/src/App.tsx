import { useCallback, useEffect, useState } from 'react'
import { ApiError, getHealth } from './api'
import { AskPanel } from './AskPanel'
import { UploadPanel } from './UploadPanel'

const HEALTH_RETRY_MS = 3000

export default function App() {
  const [chunks, setChunks] = useState<number | null>(null)
  const [apiError, setApiError] = useState<string | null>(null)
  // Bumped when a document finishes indexing (or to retry after an error): re-fetches /health.
  const [healthVersion, setHealthVersion] = useState(0)
  const onIndexed = useCallback(() => setHealthVersion((v) => v + 1), [])

  useEffect(() => {
    let active = true
    let retry: ReturnType<typeof setTimeout> | undefined
    getHealth().then(
      (health) => {
        if (!active) return
        setChunks(health.chunks_indexed)
        setApiError(null)
      },
      (err: unknown) => {
        if (!active) return
        setApiError(err instanceof ApiError ? err.message : 'API unavailable')
        retry = setTimeout(() => setHealthVersion((v) => v + 1), HEALTH_RETRY_MS)
      },
    )
    return () => {
      active = false
      clearTimeout(retry)
    }
  }, [healthVersion])

  return (
    <main>
      <header>
        <h1>RAG Document Assistant</h1>
        <p className="status" role="status">
          {apiError ?? (chunks === null ? 'Connecting…' : `${chunks} chunks indexed`)}
        </p>
      </header>
      <UploadPanel onIndexed={onIndexed} />
      <AskPanel />
    </main>
  )
}
