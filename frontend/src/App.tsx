import { useEffect, useRef, useState } from 'react';
import { api, type HealthResponse, type DocumentsResponse, type ChatResponse, type Source } from './api/client';
import './App.css';

type Message = {
  role: 'user' | 'assistant';
  content: string;
  sources?: Source[];
  timestamp?: string;
};

type ActiveTab = 'chat' | 'documents' | 'about';

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

  const [activeTab, setActiveTab] = useState<ActiveTab>('chat');
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [sending, setSending] = useState(false);
  const [sendError, setSendError] = useState<string | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

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
        setInitError(err instanceof Error ? err.message : 'Failed to connect to backend service');
      } finally {
        setLoadingInit(false);
      }
    };
    load();
  }, []);

  useEffect(() => {
    if (activeTab === 'chat') {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, sending, activeTab]);

  const executeQuery = async (queryText: string) => {
    const query = queryText.trim();
    if (!query || sending) return;

    setActiveTab('chat');
    setSendError(null);
    setSending(true);

    const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const userMsg: Message = { role: 'user', content: query, timestamp: timeStr };
    setMessages((prev) => [...prev, userMsg]);
    setInput('');

    try {
      const res: ChatResponse = await api.chat(query);
      const assistantTime = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      const assistantMsg: Message = {
        role: 'assistant',
        content: res.answer,
        sources: res.sources,
        timestamp: assistantTime,
      };
      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err) {
      setSendError(err instanceof Error ? err.message : 'Failed to retrieve answer from server');
    } finally {
      setSending(false);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    executeQuery(input);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      executeQuery(input);
    }
  };

  const handleExampleClick = (q: string) => {
    executeQuery(q);
  };

  const docCount = documents ? documents.total_documents : 4;
  const chunkCount = documents ? documents.total_chunks : 529;

  return (
    <div className="layout-root">
      {/* ==================================================
          LEFT SIDEBAR (Terracotta Theme 250-280px)
         ================================================== */}
      <aside className={`sidebar ${mobileMenuOpen ? 'mobile-open' : ''}`}>
        <div className="sidebar-top">
          <div className="sidebar-brand">
            <div className="brand-icon-box" aria-hidden="true">
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                <polyline points="14 2 14 8 20 8"></polyline>
                <line x1="16" y1="13" x2="8" y2="13"></line>
                <line x1="16" y1="17" x2="8" y2="17"></line>
                <polyline points="10 9 9 9 8 9"></polyline>
              </svg>
            </div>
            <div className="brand-info">
              <h1 className="brand-name">
                Multi-Document<br />
                <span>RAG Assistant</span>
              </h1>
              <p className="brand-subtitle">AI Knowledge Assistant</p>
            </div>
          </div>

          <nav className="sidebar-nav" aria-label="Primary navigation">
            <button
              type="button"
              className={`nav-item ${activeTab === 'chat' ? 'active' : ''}`}
              onClick={() => { setActiveTab('chat'); setMobileMenuOpen(false); }}
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
              </svg>
              <span>Chat</span>
              {messages.length > 0 && <span className="nav-badge">{messages.length}</span>}
            </button>

            <button
              type="button"
              className={`nav-item ${activeTab === 'documents' ? 'active' : ''}`}
              onClick={() => { setActiveTab('documents'); setMobileMenuOpen(false); }}
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"></path>
                <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"></path>
              </svg>
              <span>Documents</span>
              <span className="nav-badge">{docCount}</span>
            </button>

            <button
              type="button"
              className={`nav-item ${activeTab === 'about' ? 'active' : ''}`}
              onClick={() => { setActiveTab('about'); setMobileMenuOpen(false); }}
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="10"></circle>
                <line x1="12" y1="16" x2="12" y2="12"></line>
                <line x1="12" y1="8" x2="12.01" y2="8"></line>
              </svg>
              <span>About</span>
            </button>
          </nav>
        </div>

        <div className="sidebar-bottom">
          <p className="sidebar-tagline">
            Find answers<br />
            across your documents.
          </p>
          <div className="sidebar-divider" />
          <div className="sidebar-footer-note">
            <span className="sparkle-icon">✦</span>
            <span>Powered by RAG</span>
          </div>
        </div>
      </aside>

      {/* Backdrop for mobile drawer */}
      {mobileMenuOpen && (
        <div
          className="mobile-backdrop"
          onClick={() => setMobileMenuOpen(false)}
          aria-hidden="true"
        />
      )}

      {/* ==================================================
          MAIN CONTENT AREA (Warm Cream / Editorial)
         ================================================== */}
      <div className="main-content">
        {/* TOP STATUS BAR */}
        <header className="top-status-bar">
          <div className="top-left-group">
            <button
              type="button"
              className="mobile-menu-toggle"
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              aria-label="Toggle navigation sidebar"
            >
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <line x1="3" y1="12" x2="21" y2="12"></line>
                <line x1="3" y1="6" x2="21" y2="6"></line>
                <line x1="3" y1="18" x2="21" y2="18"></line>
              </svg>
            </button>
            <span className="current-view-title">
              {activeTab === 'chat' && 'Knowledge Workspace'}
              {activeTab === 'documents' && 'Indexed Document Catalog'}
              {activeTab === 'about' && 'System Architecture'}
            </span>
          </div>

          <div className="status-badges-container">
            {/* Connected Badge */}
            <div className={`status-badge-pill ${health?.status === 'ok' ? 'is-online' : 'is-offline'}`}>
              <span className="status-indicator-dot" />
              <span className="status-badge-label">
                {health?.status === 'ok' ? 'Connected' : 'Offline'}
              </span>
            </div>

            {/* Documents Badge */}
            <div className="status-badge-pill info-pill">
              <svg className="badge-svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2">
                <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"></path>
                <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"></path>
              </svg>
              <span>{docCount} Documents</span>
            </div>

            {/* Chunks Badge */}
            <div className="status-badge-pill info-pill chunk-pill-fix">
              <svg className="badge-svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2">
                <polygon points="12 2 2 7 12 12 22 7 12 2"></polygon>
                <polyline points="2 17 12 22 22 17"></polyline>
                <polyline points="2 12 12 17 22 12"></polyline>
              </svg>
              <span>{chunkCount} Chunks</span>
            </div>
          </div>
        </header>

        {/* ERROR NOTIFICATION BANNER */}
        {initError && (
          <div className="alert-banner alert-warning" role="alert">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="10"></circle>
              <line x1="12" y1="8" x2="12" y2="12"></line>
              <line x1="12" y1="16" x2="12.01" y2="16"></line>
            </svg>
            <span>{initError}. Please verify FastAPI backend on port 8000.</span>
          </div>
        )}

        {/* TAB 1: CHAT VIEW */}
        {activeTab === 'chat' && (
          <main className="chat-viewport">
            {loadingInit ? (
              <div className="loading-container">
                <div className="editorial-spinner" />
                <p>Initializing knowledge index…</p>
              </div>
            ) : messages.length === 0 ? (
              /* HERO / EMPTY STATE */
              <div className="hero-empty-state">
                {/* Subtle pure-CSS warm landscape/abstract art graphic */}
                <div className="hero-abstract-art" aria-hidden="true">
                  <div className="art-sun" />
                  <div className="art-hill-1" />
                  <div className="art-hill-2" />
                  <div className="art-arch" />
                </div>

                <div className="hero-heading-group">
                  <h2 className="hero-title">
                    Ask anything<br />
                    about your documents
                  </h2>
                  <p className="hero-subtitle">
                    Search across multiple documents using retrieval-augmented generation.
                  </p>
                </div>

                <div className="example-cards-wrapper">
                  <div className="example-cards-grid">
                    {EXAMPLE_QUESTIONS.map((q, idx) => (
                      <button
                        key={idx}
                        type="button"
                        className="example-question-card"
                        onClick={() => handleExampleClick(q)}
                        disabled={sending}
                      >
                        <span className="card-question-text">{q}</span>
                        <span className="card-arrow-icon" aria-hidden="true">
                          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                            <line x1="5" y1="12" x2="19" y2="12"></line>
                            <polyline points="12 5 19 12 12 19"></polyline>
                          </svg>
                        </span>
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            ) : (
              /* ACTIVE MESSAGES STREAM */
              <div className="messages-stream">
                {messages.map((msg, idx) => (
                  <div key={idx} className={`message-row ${msg.role}`}>
                    {msg.role === 'assistant' && (
                      <div className="assistant-avatar" aria-hidden="true">
                        <span>✦</span>
                      </div>
                    )}

                    <div className="message-body-wrap">
                      <div className={`message-bubble ${msg.role}`}>
                        <div className="bubble-text">{msg.content}</div>
                        {msg.timestamp && (
                          <div className="message-timestamp">{msg.timestamp}</div>
                        )}
                      </div>

                      {/* SOURCES ACCORDION */}
                      {msg.role === 'assistant' && msg.sources && msg.sources.length > 0 && (
                        <div className="sources-wrapper">
                          <details className="sources-details" open>
                            <summary className="sources-toggle">
                              <span className="sources-label">
                                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                                  <polyline points="14 2 14 8 20 8"></polyline>
                                </svg>
                                Sources ({msg.sources.length})
                              </span>
                              <span className="sources-chevron" aria-hidden="true">▾</span>
                            </summary>

                            <div className="sources-cards-list">
                              {msg.sources.map((src, si) => (
                                <div key={si} className="source-card-item">
                                  <div className="source-card-top">
                                    <span className="source-filename" title={src.filename}>
                                      {src.filename}
                                    </span>
                                    {src.score != null && (
                                      <span className="source-score-badge">
                                        Score {src.score.toFixed(2)}
                                      </span>
                                    )}
                                  </div>
                                  <div className="source-card-meta">
                                    {src.page != null && (
                                      <span className="source-meta-page">Page {src.page}</span>
                                    )}
                                    {src.section && (
                                      <span className="source-meta-section" title={src.section}>
                                        · Section: {src.section}
                                      </span>
                                    )}
                                  </div>
                                </div>
                              ))}
                            </div>
                          </details>
                        </div>
                      )}
                    </div>
                  </div>
                ))}

                {/* THINKING STATE */}
                {sending && (
                  <div className="message-row assistant">
                    <div className="assistant-avatar" aria-hidden="true">
                      <span>✦</span>
                    </div>
                    <div className="message-body-wrap">
                      <div className="message-bubble assistant thinking-bubble">
                        <span className="thinking-text">Thinking</span>
                        <span className="thinking-dots">
                          <span className="dot dot-1">.</span>
                          <span className="dot dot-2">.</span>
                          <span className="dot dot-3">.</span>
                        </span>
                      </div>
                    </div>
                  </div>
                )}

                <div ref={messagesEndRef} />
              </div>
            )}
          </main>
        )}

        {/* TAB 2: DOCUMENTS VIEW */}
        {activeTab === 'documents' && (
          <main className="tab-view-container">
            <div className="tab-view-header">
              <h2 className="tab-view-title">Indexed Documents</h2>
              <p className="tab-view-subtitle">
                The multi-document RAG system currently indexes {docCount} documents totaling {chunkCount} chunk embeddings in ChromaDB.
              </p>
            </div>

            <div className="documents-catalog-grid">
              {documents?.documents.map((doc, idx) => (
                <div key={idx} className="doc-catalog-card">
                  <div className="doc-card-head">
                    <div className="doc-file-icon">
                      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                        <polyline points="14 2 14 8 20 8"></polyline>
                      </svg>
                    </div>
                    <div>
                      <h3 className="doc-name">{doc.filename}</h3>
                      <span className="doc-meta-counts">{doc.page_count} Pages · {doc.chunk_count} Chunks</span>
                    </div>
                  </div>
                  <div className="doc-sections-list">
                    <span className="sections-title">Indexed Sections:</span>
                    <div className="sections-chips">
                      {doc.sections.map((sec, sidx) => (
                        <span key={sidx} className="section-chip">{sec}</span>
                      ))}
                    </div>
                  </div>
                </div>
              ))}
            </div>

            <div className="tab-cta-wrap">
              <button
                type="button"
                className="cta-terracotta-btn"
                onClick={() => setActiveTab('chat')}
              >
                ← Return to Chat & Ask Questions
              </button>
            </div>
          </main>
        )}

        {/* TAB 3: ABOUT VIEW */}
        {activeTab === 'about' && (
          <main className="tab-view-container">
            <div className="tab-view-header">
              <h2 className="tab-view-title">About This Architecture</h2>
              <p className="tab-view-subtitle">
                A multi-document retrieval-augmented generation assistant designed for grounded reasoning across disparate corpora.
              </p>
            </div>

            <div className="about-features-grid">
              <div className="about-feature-card">
                <div className="feature-icon">🔍</div>
                <h3>Hybrid RAG Retrieval</h3>
                <p>Vector similarity scoring via Nomic Embed Text coupled with sub-query decomposition, semantic boundary chunking, and reciprocal rank fusion.</p>
              </div>
              <div className="about-feature-card">
                <div className="feature-icon">⚡</div>
                <h3>FastAPI & ChromaDB</h3>
                <p>High-performance asynchronous Python backend with strictly typed Pydantic v2 schemas and persistent Chroma vector storage.</p>
              </div>
              <div className="about-feature-card">
                <div className="feature-icon">🧠</div>
                <h3>Ollama Local Inference</h3>
                <p>Zero data exfiltration using local LLaMA 3.2 models with strict anti-hallucination prompts and document citation requirements.</p>
              </div>
            </div>

            <div className="tab-cta-wrap">
              <button
                type="button"
                className="cta-terracotta-btn"
                onClick={() => setActiveTab('chat')}
              >
                ← Start Asking Questions
              </button>
            </div>
          </main>
        )}

        {/* ==================================================
            BOTTOM CHAT INPUT AREA (Fixed/Comfortable)
           ================================================== */}
        {activeTab === 'chat' && (
          <footer className="chat-input-footer">
            <div className="input-box-container">
              {sendError && (
                <div className="alert-banner inline-alert" role="alert">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <circle cx="12" cy="12" r="10"></circle>
                    <line x1="12" y1="8" x2="12" y2="12"></line>
                    <line x1="12" y1="16" x2="12.01" y2="16"></line>
                  </svg>
                  <span>{sendError}</span>
                </div>
              )}

              <form onSubmit={handleSubmit} className="input-form">
                <textarea
                  ref={textareaRef}
                  rows={1}
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="Ask a question about your documents..."
                  disabled={sending}
                  aria-label="Ask a question about your documents"
                  className="editorial-textarea"
                />
                <button
                  type="submit"
                  disabled={sending || !input.trim()}
                  className="send-terracotta-btn"
                  aria-label="Send query"
                >
                  {sending ? (
                    <span className="btn-spinner-icon" aria-hidden="true" />
                  ) : (
                    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                      <line x1="22" y1="2" x2="11" y2="13"></line>
                      <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
                    </svg>
                  )}
                  <span>Send</span>
                </button>
              </form>

              <div className="input-caption-hint">
                Press <kbd>Enter</kbd> to send, <kbd>Shift + Enter</kbd> for a new line
              </div>
            </div>
          </footer>
        )}
      </div>
    </div>
  );
}

export default App;