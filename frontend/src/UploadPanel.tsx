import { useEffect, useRef, useState, type FormEvent } from 'react'
import { ApiError, getJob, isFinished, uploadDocument, type Job } from './api'

const POLL_MS = 1000

interface Upload {
  key: number
  filename: string
  job: Job | null
  error: string | null
}

interface Props {
  onIndexed: () => void
}

export function UploadPanel({ onIndexed }: Props) {
  const [uploads, setUploads] = useState<Upload[]>([])
  const [busy, setBusy] = useState(false)
  const input = useRef<HTMLInputElement>(null)
  const nextKey = useRef(0)

  function update(key: number, patch: Partial<Upload>) {
    setUploads((list) => list.map((u) => (u.key === key ? { ...u, ...patch } : u)))
  }

  async function handleSubmit(event: FormEvent) {
    event.preventDefault()
    const file = input.current?.files?.[0]
    if (!file) return

    const key = nextKey.current++
    setUploads((list) => [{ key, filename: file.name, job: null, error: null }, ...list])
    setBusy(true)
    try {
      const accepted = await uploadDocument(file)
      update(key, { filename: accepted.filename, job: await getJob(accepted.job_id) })
      if (input.current) input.current.value = ''
    } catch (err) {
      update(key, { error: err instanceof ApiError ? err.message : 'Upload failed' })
    } finally {
      setBusy(false)
    }
  }

  return (
    <section className="panel">
      <h2>1. Upload a document</h2>
      <form onSubmit={handleSubmit} className="upload-form">
        <label>
          <span className="visually-hidden">Document</span>
          <input ref={input} type="file" accept=".pdf,.txt,.md" disabled={busy} required />
        </label>
        <button type="submit" disabled={busy}>
          {busy ? 'Uploading…' : 'Upload'}
        </button>
      </form>
      <p className="hint">PDF, .txt or .md. Indexing runs in the background.</p>

      {uploads.length > 0 && (
        <ul className="uploads" aria-live="polite">
          {uploads.map((u) => (
            <UploadRow key={u.key} upload={u} onChange={update} onIndexed={onIndexed} />
          ))}
        </ul>
      )}
    </section>
  )
}

interface RowProps {
  upload: Upload
  onChange: (key: number, patch: Partial<Upload>) => void
  onIndexed: () => void
}

function UploadRow({ upload, onChange, onIndexed }: RowProps) {
  const { key, job } = upload

  // Poll the job until it finishes. The timer is cleared if the row unmounts.
  useEffect(() => {
    if (!job || isFinished(job.status)) return
    const timer = setTimeout(async () => {
      try {
        const next = await getJob(job.job_id)
        onChange(key, { job: next })
        if (next.status === 'succeeded') onIndexed()
      } catch (err) {
        onChange(key, { error: err instanceof ApiError ? err.message : 'Lost track of the job' })
      }
    }, POLL_MS)
    return () => clearTimeout(timer)
  }, [job, key, onChange, onIndexed])

  return (
    <li className="upload">
      <span className="filename">{upload.filename}</span>
      <Status upload={upload} />
    </li>
  )
}

function Status({ upload }: { upload: Upload }) {
  if (upload.error) return <span className="badge failed">{upload.error}</span>
  const { job } = upload
  if (!job) return <span className="badge running">uploading</span>
  if (job.status === 'failed') return <span className="badge failed">failed: {job.error}</span>
  if (job.status !== 'succeeded' || !job.result) {
    return <span className={`badge ${job.status}`}>{job.status}</span>
  }
  const { status, chunks, filename } = job.result
  const detail =
    status === 'unchanged'
      ? `already indexed${filename !== upload.filename ? ` as ${filename}` : ''}`
      : `${status}, ${chunks} chunks`
  return <span className="badge succeeded">{detail}</span>
}
