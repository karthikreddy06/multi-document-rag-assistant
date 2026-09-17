import { useEffect, useState } from 'react';
import { api, type HealthResponse, type DocumentsResponse, type ChatResponse, type Source } from './api/client';
import './App.css';

type Message = {
  role: 'user' | 'assistant';
  content: string;
  sources?: Source[];
};

const EXAMPLE_QUESTIONS = [
  'What is the main character in The Old Man and the Sea?',
  'What are all my skills?',
  'What are the main features of TravelTrack?',
  'What is the main character in Alice in Wonderland?',
];

function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [documents, setDocuments] = useState<DocumentsResponse | null>(null);
  const [loadingInit, setLoadingInit] = useState(true);
  const [initError, setInitError] = useState<string | null>(null);

  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [sending, setSending] = useState(false);
  const [sendError, setSendError] = useState<string | null>(null);

  useEffect(() => {
    const load = async () => {
      try {
        setLoadingInit(true);
        const [healthRes, docsRes] = await Promise.all([
          api.health(),
          api.documents(),
        ]);
        setHealth(healthRes);
        setDocuments(docsRes);
      } catch (err) {
        setInitError(err instanceof Error ? err.message : 'Failed to connect');
      } finally {
        setLoadingInit(false);
      }
    };
    load();
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const query = input.trim();
    if (!query || sending) return;

    setSendError(null);
    setSending(true);
    const userMsg: Message = { role: 'user', content: query };
    setMessages((prev) => [...prev, userMsg]);
    setInput('');

    try {
      const res: ChatResponse = await api.chat(query);
      const assistantMsg: Message = {
        role: 'assistant',
        content: res.answer,
        sources: res.sources,
      };
      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err) {
      setSendError(err instanceof Error ? err.message : 'Failed to get answer');
    } finally {
      setSending(false);
    }
  };

  const handleExampleClick = (q: string) => {
    setInput(q);
    // auto-submit
    const form = document.getElementById('chat-form') as HTMLFormElement | null;
    if (form) form.requestSubmit();
  };

  if (loadingInit) {
    return <div className="app">Loading...</div>;
  }

  return (
    <div className="app">
      <header>
        <h1>Multi-Document RAG Assistant</h1>
        <p className="subtitle">Ask questions across your indexed documents.</p>
      </header>

      <main>
        {/* Keep a compact status bar */}
        <section className="status-bar">
          <span className={health ? 'ok' : 'error'}>
            {health ? `Backend: ${health.status}` : 'Backend: disconnected'}
          </span>
          {documents && (
            <span>{documents.total_documents} documents · {documents.total_chunks} chunks</span>
          )}
        </section>

        <section className="chat">
          <div className="messages" id="messages">
            {messages.length === 0 && (
              <div className="welcome">
                <p>Start by asking a question or pick an example below.</p>
                <div className="examples">
                  {EXAMPLE_QUESTIONS.map((q) => (
                    <button key={q} type="button" onClick={() => handleExampleClick(q)} disabled={sending}>
                      {q}
                    </button>
                  ))}
                </div>
              </div>
            )}
            {messages.map((msg, idx) => (
              <div key={idx} className={`msg ${msg.role}`}>
                <div className="bubble">{msg.content}</div>
                {msg.role === 'assistant' && msg.sources && msg.sources.length > 0 && (
                  <details className="sources">
                    <summary>Sources ({msg.sources.length})</summary>
                    <ul>
                      {msg.sources.map((src, si) => (
                        <li key={si}>
                          <strong>{src.filename}</strong>
                          {src.page != null && <span> · p.{src.page}</span>}
                          {src.section && <span> · {src.section}</span>}
                          {src.score != null && <span> · score: {src.score.toFixed(3)}</span>}
                        </li>
                      ))}
                    </ul>
                  </details>
                )}
              </div>
            ))}
            {sending && <div className="thinking">Thinking…</div>}
          </div>

          <form id="chat-form" onSubmit={handleSubmit} className="input-form">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Type your question…"
              disabled={sending}
              aria-label="Your question"
            />
            <button type="submit" disabled={sending || !input.trim()}>
              {sending ? 'Sending…' : 'Send'}
            </button>
          </form>
          {sendError && <div className="error-inline">{sendError}</div>}
        </section>
      </main>

      {initError && <footer className="error">Error: {initError}</footer>}
    </div>
  );
}

export default App;