import { useState, type FormEvent } from 'react'
import { ApiError, askQuestion, type Answer } from './api'

export function AskPanel() {
  const [question, setQuestion] = useState('')
  const [asked, setAsked] = useState('')
  const [answer, setAnswer] = useState<Answer | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  async function handleSubmit(event: FormEvent) {
    event.preventDefault()
    const text = question.trim()
    if (!text) return
    setLoading(true)
    setError(null)
    setAnswer(null)
    try {
      setAnswer(await askQuestion(text))
      setAsked(text)
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Something went wrong')
    } finally {
      setLoading(false)
    }
  }

  return (
    <section className="panel">
      <h2>2. Ask a question</h2>
      <form onSubmit={handleSubmit} className="ask-form">
        <label htmlFor="question" className="visually-hidden">
          Question
        </label>
        <textarea
          id="question"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="How much debt does Netflix have?"
          maxLength={1000}
          rows={3}
          disabled={loading}
        />
        <button type="submit" disabled={loading || !question.trim()}>
          {loading ? 'Thinking…' : 'Ask'}
        </button>
      </form>

      <div aria-live="polite">
        {error && <p className="error">{error}</p>}
        {answer && (
          <article className="answer">
            <p className="asked">{asked}</p>
            <p className="answer-text">{answer.answer}</p>
            <Sources sources={answer.sources} />
          </article>
        )}
      </div>
    </section>
  )
}

function Sources({ sources }: { sources: Answer['sources'] }) {
  if (sources.length === 0) return null
  return (
    <div className="sources">
      <h3>Sources, most relevant first</h3>
      <ol>
        {sources.map((s) => (
          <li key={`${s.source}#${s.chunk}`}>
            <details>
              <summary>
                <span className="filename">{s.source}</span>
                <span className="meta">
                  chunk {s.chunk} · similarity {s.score.toFixed(2)}
                </span>
              </summary>
              <pre className="chunk">{s.text}</pre>
            </details>
          </li>
        ))}
      </ol>
    </div>
  )
}
