// Typed client for the FastAPI backend. Shapes mirror api/schemas.py.

export type JobStatus = 'queued' | 'running' | 'succeeded' | 'failed'
export type IngestStatus = 'indexed' | 'replaced' | 'unchanged'

export interface Health {
  status: 'ok'
  chunks_indexed: number
}

export interface JobAccepted {
  job_id: string
  filename: string
  status: JobStatus
}

export interface Job {
  job_id: string
  filename: string
  status: JobStatus
  created_at: string
  started_at: string | null
  finished_at: string | null
  error: string | null
  result: {
    document_id: string
    filename: string
    chunks: number
    status: IngestStatus
  } | null
}

export interface Source {
  source: string
  chunk: number
  score: number
  text: string
}

export interface Answer {
  answer: string
  sources: Source[]
}

export class ApiError extends Error {}

const BASE = '/api'

export function getHealth(): Promise<Health> {
  return request('/health')
}

export function uploadDocument(file: File): Promise<JobAccepted> {
  const form = new FormData()
  form.append('file', file)
  return request('/documents', { method: 'POST', body: form })
}

export function getJob(jobId: string): Promise<Job> {
  return request(`/jobs/${encodeURIComponent(jobId)}`)
}

export function askQuestion(question: string): Promise<Answer> {
  return request('/query', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question }),
  })
}

export function isFinished(status: JobStatus): boolean {
  return status === 'succeeded' || status === 'failed'
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(BASE + path, init)
  } catch {
    throw new ApiError('Cannot reach the API. Is the server running?')
  }
  const body: unknown = await response.json().catch(() => null)
  if (!response.ok) {
    throw new ApiError(errorMessage(body) ?? fallbackMessage(response.status))
  }
  return body as T
}

function fallbackMessage(status: number): string {
  // With no JSON body, a gateway error comes from the proxy (Vite or nginx), not the API:
  // usually the API isn't up yet because it is still loading the models.
  if (status === 502 || status === 503 || status === 504) {
    return 'API not reachable yet. If it just started, it may still be loading the models.'
  }
  return `Request failed (HTTP ${status})`
}

// FastAPI errors are {"detail": "message"} or, for validation errors,
// {"detail": [{"msg": "...", ...}]}.
function errorMessage(body: unknown): string | null {
  if (typeof body !== 'object' || body === null || !('detail' in body)) return null
  const { detail } = body as { detail: unknown }
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) {
    return detail.map((d) => (d as { msg?: string }).msg ?? 'Invalid request').join('; ')
  }
  return null
}
