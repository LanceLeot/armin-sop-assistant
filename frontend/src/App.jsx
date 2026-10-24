
import { useState } from 'react'
import './App.css'

function App() {
  const [question, setQuestion] = useState('')
  const [isSending, setIsSending] = useState(false)

  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      text: 'Hello! I am Armin, your SOP Assistant. Ask me anything about organizational procedures.'
    }
  ])

  async function handleSend() {
    const trimmedQuestion = question.trim()

    if (!trimmedQuestion || isSending) return

    setQuestion('')

    setMessages((previous) => [
      ...previous,
      { role: 'user', text: trimmedQuestion }
    ])

    setIsSending(true)

    try {
      const response = await fetch('/ask', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          question: trimmedQuestion
        })
      })

      if (!response.ok) {
        throw new Error(`Request failed (${response.status})`)
      }

      const data = await response.json()

      setMessages((previous) => [
        ...previous,
        {
          role: 'assistant',
          text: data.answer || 'The backend returned no answer.',
          sources: Array.isArray(data.sources) ? data.sources : []
        }
      ])
    } catch (error) {
      console.error('SOP backend error:', error)

      let errorMessage

      if (error.message?.includes('Failed to fetch')) {
        errorMessage =
          'I could not reach the backend. Please check that the backend is running and CORS is configured correctly.'
      } else {
        errorMessage = `Something went wrong: ${error.message}`
      }

      setMessages((previous) => [
        ...previous,
        {
          role: 'assistant',
          text: errorMessage
        }
      ])
    } finally {
      setIsSending(false)
    }
  }

  function renderMessage(text) {
    const lines = text.split('\n')

    return lines.map((line, index) => {
      const stepMatch = line.match(
        /^(Step \d+(?:\s*\([^)]*\))?:)\s*(.*)$/
      )

      // Display procedure steps with a separate heading.
      if (stepMatch) {
        const stepLabel = stepMatch[1]
        const stepContent = stepMatch[2]
        const parts = stepContent.split(/\s*;\s*/)
        const stepTitle = parts.shift() || ''

        return (
          <div className="procedure-step" key={index}>
            <p className="step-heading">
              <strong>
                {stepLabel} {stepTitle}
              </strong>
            </p>

            {parts
              .filter((instruction) => instruction.trim())
              .map((instruction, instructionIndex) => (
                <p
                  className="step-instruction"
                  key={instructionIndex}
                >
                  • {instruction.trim()}
                </p>
              ))}
          </div>
        )
      }

      // Display ordinary message lines.
      if (line.trim()) {
        return <p key={index}>{line}</p>
      }

      return <div key={index} className="message-spacer" />
    })
  }

  function getSourceName(source) {
    if (typeof source === 'string') {
      return source
    }

    return (
      source.document ||
      source.filename ||
      source.source ||
      source.name ||
      'SOP document'
    )
  }

  function getSourcePage(source) {
    if (!source || typeof source !== 'object') {
      return null
    }

    return source.page ?? source.page_number ?? source.pageNumber ?? null
  }

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand-icon">S</div>

        <div>
          <h1>Armin</h1>
          <p>Organization Procedure Helpdesk</p>
        </div>

        <span className="status"></span>
      </header>

      <main className="chat-container">
        <section className="welcome">
          <div className="welcome-icon">✦</div>

          <h2>Hi! I'm Armin, your Organizational SOP Assistant.</h2>

          <p>
            Ask a question about company procedures, policies, or
            standard operating procedures.
          </p>
        </section>

        <section className="messages">
          {messages.map((message, index) => (
            <div
              key={index}
              className={`message ${message.role}`}
            >
              <strong>
                {message.role === 'user'
                  ? 'You'
                  : 'Armin — Your Organizational SOP Assistant'}
              </strong>

              <div className="message-text">
                {renderMessage(message.text)}
              </div>

              {message.role === 'assistant' &&
                message.sources &&
                message.sources.length > 0 && (
                  <div className="message-sources">
                    <strong>Sources</strong>

                    {message.sources.map((source, sourceIndex) => {
                      const sourceName = getSourceName(source)
                      const sourcePage = getSourcePage(source)

                      return (
                        <p
                          key={`${sourceIndex}-${sourceName}`}
                          className="source-item"
                        >
                          📄 {sourceName}
                          {sourcePage != null &&
                            ` — Page ${sourcePage}`}
                        </p>
                      )
                    })}
                  </div>
                )}
            </div>
          ))}

          {isSending && (
            <div className="message assistant">
              <strong>Armin</strong>
              <p>Looking through the SOP documents...</p>
            </div>
          )}
        </section>

        <form
          className="input-area"
          onSubmit={(event) => {
            event.preventDefault()
            handleSend()
          }}
        >
          <input
            type="text"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="Ask about a procedure..."
            aria-label="Your question"
            disabled={isSending}
          />

          <button
            type="submit"
            disabled={isSending || !question.trim()}
          >
            {isSending ? 'Sending…' : 'Send ↑'}
          </button>
        </form>

        <p className="disclaimer">
          Prototype — answers are retrieved from the loaded SOP documents.
        </p>
      </main>
    </div>
  )
}

export default App
