import { useEffect, useRef, useState } from 'react';
import {
  api,
  type HealthResponse,
  type DocumentsResponse,
  type Source,
  type ChatSession,
  type DocumentRecord,
} from './api/client';
import { useAuth } from './context/AuthContext';
import { AuthPage } from './components/AuthPage';
import './App.css';

type Message = {
  id?: string;
  role: 'user' | 'assistant';
  content: string;
  sources?: Source[];
  timestamp?: string;
};

type ActiveTab = 'chat' | 'documents' | 'about';

const EXAMPLE_QUESTIONS = [
  'What is the main character in The Old Man and the Sea?',
  'What is the main character in Alice in Wonderland?',
  'How many days did Santiago go without catching a fish?',
  'What are the details described in the sample PDF document?',
];

/** Return a coloured label chip for the document card based on file extension. */
function getFileTypeLabel(filename: string): { label: string; color: string } {
  const ext = filename.split('.').pop()?.toLowerCase() ?? '';
  const map: Record<string, { label: string; color: string }> = {
    pdf:      { label: 'PDF',      color: '#e05252' },
    docx:     { label: 'DOCX',     color: '#4a90d9' },
    doc:      { label: 'DOC',      color: '#4a90d9' },
    pptx:     { label: 'PPTX',     color: '#e07b39' },
    ppt:      { label: 'PPT',      color: '#e07b39' },
    xlsx:     { label: 'XLSX',     color: '#3aaa6a' },
    xls:      { label: 'XLS',      color: '#3aaa6a' },
    csv:      { label: 'CSV',      color: '#2ab8a8' },
    txt:      { label: 'TXT',      color: '#8a8a9a' },
    md:       { label: 'MD',       color: '#7b68ee' },
    markdown: { label: 'MD',       color: '#7b68ee' },
    png:      { label: 'PNG',      color: '#c060c0' },
    jpg:      { label: 'JPG',      color: '#c060c0' },
    jpeg:     { label: 'JPEG',     color: '#c060c0' },
    webp:     { label: 'WEBP',     color: '#c060c0' },
  };
  return map[ext] ?? { label: ext.toUpperCase() || 'FILE', color: '#666' };
}

function App() {
  const { user, isAuthenticated, isLoading, logout } = useAuth();

  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [documents, setDocuments] = useState<DocumentsResponse | null>(null);
  const [loadingInit, setLoadingInit] = useState(true);
  const [initError, setInitError] = useState<string | null>(null);

  const [activeTab, setActiveTab] = useState<ActiveTab>('chat');
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  // Chat sessions state
  const [chats, setChats] = useState<ChatSession[]>([]);
  const [activeChatId, setActiveChatId] = useState<string | null>(null);
  const [editingChatId, setEditingChatId] = useState<string | null>(null);
  const [editTitleInput, setEditTitleInput] = useState('');
  const [deletingChatId, setDeletingChatId] = useState<string | null>(null);

  // Active chat document scope and message state
  const [chatDocs, setChatDocs] = useState<DocumentRecord[]>([]);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [sending, setSending] = useState(false);
  const [sendError, setSendError] = useState<string | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Helper: Load messages and documents for a specific chat
  const loadChatDetails = async (chatId: string) => {
    setActiveChatId(chatId);
    if (user) {
      localStorage.setItem(`rag_active_chat_${user.id}`, chatId);
    }
    setSendError(null);
    setUploadError(null);

    try {
      const [msgsRes, docsRes] = await Promise.all([
        api.listMessages(chatId).catch(() => []),
        api.listChatDocuments(chatId).catch(() => []),
      ]);

      const formattedMsgs: Message[] = msgsRes.map((m) => ({
        id: m.id,
        role: m.role,
        content: m.content,
        sources: m.sources,
        timestamp: new Date(m.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      }));

      setMessages(formattedMsgs);
      setChatDocs(docsRes);
    } catch (err) {
      console.error('Failed to load chat details:', err);
    }
  };

  const handleLogout = () => {
    setChats([]);
    setChatDocs([]);
    setMessages([]);
    setActiveChatId(null);
    setDocuments(null);
    logout();
  };

  // 1. Initial Load: Health, Global Documents, and Chat Sessions
  useEffect(() => {
    if (!isAuthenticated || !user) {
      return;
    }

    const initializeApp = async () => {
      try {
        setLoadingInit(true);
        const [healthRes, docsRes, chatsRes] = await Promise.all([
          api.health().catch(() => ({ status: 'offline' })),
          api.documents().catch(() => null),
          api.listChats().catch(() => []),
        ]);

        setHealth(healthRes);
        setDocuments(docsRes);

        let initialChatId: string | null = null;
        if (chatsRes.length > 0) {
          // Check localStorage preference scoped by user id
          const savedChatId = localStorage.getItem(`rag_active_chat_${user.id}`);
          const found = chatsRes.find((c) => c.id === savedChatId);
          initialChatId = found ? found.id : chatsRes[0].id;
          setChats(chatsRes);
        } else {
          // Auto-create initial default chat session
          const newChat = await api.createChat('New Chat');
          setChats([newChat]);
          initialChatId = newChat.id;
        }

        if (initialChatId) {
          await loadChatDetails(initialChatId);
        }
      } catch (err) {
        setInitError(err instanceof Error ? err.message : 'Failed to connect to backend service');
      } finally {
        setLoadingInit(false);
      }
    };

    initializeApp();
  }, [user?.id, isAuthenticated]);

  // 2. Auto-scroll on new messages
  useEffect(() => {
    if (activeTab === 'chat') {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, sending, activeTab]);

  // Switch Chat
  const handleSelectChat = async (chatId: string) => {
    if (chatId === activeChatId) return;
    setActiveTab('chat');
    setMobileMenuOpen(false);
    await loadChatDetails(chatId);
  };

  // Create New Chat
  const handleCreateNewChat = async () => {
    try {
      const newChat = await api.createChat('New Chat');
      setChats((prev) => [newChat, ...prev]);
      setActiveTab('chat');
      setMobileMenuOpen(false);
      await loadChatDetails(newChat.id);
    } catch (err) {
      setSendError(err instanceof Error ? err.message : 'Failed to create new chat');
    }
  };

  // Start Rename
  const handleStartRename = (chat: ChatSession) => {
    setEditingChatId(chat.id);
    setEditTitleInput(chat.title);
  };

  // Submit Rename
  const handleSaveRename = async () => {
    if (!editingChatId) return;
    const cleanTitle = editTitleInput.trim();
    if (!cleanTitle) {
      setEditingChatId(null);
      return;
    }

    try {
      const updated = await api.renameChat(editingChatId, cleanTitle);
      setChats((prev) =>
        prev.map((c) => (c.id === editingChatId ? { ...c, title: updated.title, updated_at: updated.updated_at } : c))
      );
      setEditingChatId(null);
    } catch (err) {
      console.error('Failed to rename chat:', err);
    }
  };

  // Confirm and Execute Delete Chat
  const handleDeleteChat = async (chatId: string) => {
    try {
      await api.deleteChat(chatId);
      const remaining = chats.filter((c) => c.id !== chatId);
      setChats(remaining);
      setDeletingChatId(null);

      if (activeChatId === chatId) {
        if (remaining.length > 0) {
          await loadChatDetails(remaining[0].id);
        } else {
          const freshChat = await api.createChat('New Chat');
          setChats([freshChat]);
          await loadChatDetails(freshChat.id);
        }
      }
    } catch (err) {
      console.error('Failed to delete chat:', err);
    }
  };

  // Supported file extensions (must match backend FileTypeDetector)
  const SUPPORTED_EXTENSIONS = [
    '.pdf', '.docx', '.pptx', '.xlsx', '.xls',
    '.csv', '.txt', '.md', '.markdown',
    '.png', '.jpg', '.jpeg', '.webp',
  ];

  // File Upload (Single or Multiple Documents — all supported formats)
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    if (!files.length || !activeChatId) return;

    setUploading(true);
    setUploadError(null);

    try {
      for (const file of files) {
        const ext = '.' + file.name.split('.').pop()?.toLowerCase();
        if (!SUPPORTED_EXTENSIONS.includes(ext)) {
          setUploadError(
            `Unsupported file type '${ext}'. Allowed: ${SUPPORTED_EXTENSIONS.join(', ')}`
          );
          continue;
        }
        await api.uploadChatDocument(activeChatId, file);
      }

      // Refresh chat documents and global catalog
      const [updatedChatDocs, updatedGlobalDocs] = await Promise.all([
        api.listChatDocuments(activeChatId),
        api.documents(),
      ]);
      setChatDocs(updatedChatDocs);
      setDocuments(updatedGlobalDocs);
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : 'Upload failed');
    } finally {
      setUploading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  // Remove Document from Active Chat
  const handleRemoveDocument = async (documentId: string) => {
    if (!activeChatId) return;

    try {
      await api.removeChatDocument(activeChatId, documentId);
      const updated = await api.listChatDocuments(activeChatId);
      setChatDocs(updated);
    } catch (err) {
      console.error('Failed to remove document:', err);
    }
  };

  // Execute Question in Active Chat Session
  const executeQuery = async (queryText: string) => {
    const query = queryText.trim();
    if (!query || sending || !activeChatId) return;

    setActiveTab('chat');
    setSendError(null);
    setSending(true);

    const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const userMsg: Message = { role: 'user', content: query, timestamp: timeStr };
    setMessages((prev) => [...prev, userMsg]);
    setInput('');

    try {
      const assistantTime = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      // Insert initial empty assistant message for streaming
      const initialAssistantMsg: Message = {
        role: 'assistant',
        content: '',
        sources: [],
        timestamp: assistantTime,
      };
      setMessages((prev) => [...prev, initialAssistantMsg]);

      await api.chatInSessionStream(
        activeChatId,
        query,
        (token: string) => {
          setMessages((prev) => {
            if (prev.length === 0) return prev;
            const next = [...prev];
            const lastIdx = next.length - 1;
            const lastMsg = next[lastIdx];
            if (lastMsg.role === 'assistant') {
              next[lastIdx] = {
                ...lastMsg,
                content: lastMsg.content + token,
              };
            }
            return next;
          });
        },
        (sources: Source[]) => {
          setMessages((prev) => {
            if (prev.length === 0) return prev;
            const next = [...prev];
            const lastIdx = next.length - 1;
            const lastMsg = next[lastIdx];
            if (lastMsg.role === 'assistant') {
              next[lastIdx] = {
                ...lastMsg,
                sources,
              };
            }
            return next;
          });
        },
        true
      );

      // Re-sort chats so active chat moves to top
      const nowIso = new Date().toISOString();
      setChats((prev) => {
        const updated = prev.map((c) => (c.id === activeChatId ? { ...c, updated_at: nowIso } : c));
        return [...updated].sort((a, b) => new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime());
      });
    } catch (err) {
      console.warn('Streaming failed, attempting non-streaming fallback:', err);
      try {
        const fallbackRes = await api.chatInSession(activeChatId, query, true);
        const assistantTime = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        setMessages((prev) => {
          const next = [...prev];
          const lastIdx = next.length - 1;
          if (next.length > 0 && next[lastIdx].role === 'assistant') {
            next[lastIdx] = {
              role: 'assistant',
              content: fallbackRes.answer,
              sources: fallbackRes.sources,
              timestamp: assistantTime,
            };
            return next;
          } else {
            return [
              ...next,
              {
                role: 'assistant',
                content: fallbackRes.answer,
                sources: fallbackRes.sources,
                timestamp: assistantTime,
              },
            ];
          }
        });
      } catch (fallbackErr) {
        setSendError(fallbackErr instanceof Error ? fallbackErr.message : 'Failed to retrieve answer from server');
      }
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

  const activeChat = chats.find((c) => c.id === activeChatId);
  const docCount = documents ? documents.total_documents : 0;
  const chunkCount = documents ? documents.total_chunks : 0;

  if (isLoading) {
    return (
      <div className="auth-page-root">
        <div className="auth-loading-spinner-box">
          <div className="auth-spinner large" />
          <p className="auth-loading-text">Loading workspace...</p>
        </div>
      </div>
    );
  }

  if (!isAuthenticated || !user) {
    return <AuthPage />;
  }

  return (
    <div className="layout-root">
      {/* ==================================================
          LEFT SIDEBAR (Terracotta Theme)
         ================================================== */}
      <aside className={`sidebar ${mobileMenuOpen ? 'mobile-open' : ''}`}>
        <div className="sidebar-top">
          {/* Brand Identity */}
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

          {/* New Chat Action Button */}
          <button
            type="button"
            className="new-chat-btn"
            onClick={handleCreateNewChat}
            aria-label="Create new chat"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <line x1="12" y1="5" x2="12" y2="19"></line>
              <line x1="5" y1="12" x2="19" y2="12"></line>
            </svg>
            <span>New Chat</span>
          </button>

          {/* Recent Chats Section */}
          <div className="sidebar-chats-container">
            <div className="sidebar-section-header">
              <span className="section-title-label">Recent Chats</span>
              <span className="chats-count-badge">{chats.length}</span>
            </div>

            <div className="recent-chats-list" role="list">
              {chats.map((chat) => (
                <div
                  key={chat.id}
                  className={`chat-row-item ${chat.id === activeChatId ? 'active' : ''}`}
                  onClick={() => handleSelectChat(chat.id)}
                  role="listitem"
                  tabIndex={0}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') handleSelectChat(chat.id);
                  }}
                >
                  <svg className="chat-item-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
                  </svg>

                  {editingChatId === chat.id ? (
                    <input
                      type="text"
                      className="chat-rename-input"
                      value={editTitleInput}
                      autoFocus
                      onChange={(e) => setEditTitleInput(e.target.value)}
                      onBlur={handleSaveRename}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') handleSaveRename();
                        if (e.key === 'Escape') setEditingChatId(null);
                      }}
                      onClick={(e) => e.stopPropagation()}
                    />
                  ) : (
                    <span className="chat-title-text" title={chat.title}>
                      {chat.title}
                    </span>
                  )}

                  <div className="chat-item-actions">
                    <button
                      type="button"
                      className="item-action-btn"
                      title="Rename chat"
                      onClick={(e) => {
                        e.stopPropagation();
                        handleStartRename(chat);
                      }}
                    >
                      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"></path>
                        <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"></path>
                      </svg>
                    </button>
                    <button
                      type="button"
                      className="item-action-btn delete-btn"
                      title="Delete chat"
                      onClick={(e) => {
                        e.stopPropagation();
                        setDeletingChatId(chat.id);
                      }}
                    >
                      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <polyline points="3 6 5 6 21 6"></polyline>
                        <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                      </svg>
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Secondary Navigation */}
          <nav className="sidebar-nav" aria-label="Secondary navigation">
            <button
              type="button"
              className={`nav-item ${activeTab === 'documents' ? 'active' : ''}`}
              onClick={() => {
                setActiveTab('documents');
                setMobileMenuOpen(false);
              }}
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"></path>
                <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"></path>
              </svg>
              <span>Catalog</span>
              <span className="nav-badge">{docCount}</span>
            </button>

            <button
              type="button"
              className={`nav-item ${activeTab === 'about' ? 'active' : ''}`}
              onClick={() => {
                setActiveTab('about');
                setMobileMenuOpen(false);
              }}
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
          {user && (
            <div className="sidebar-user-card">
              <div className="user-avatar-pill">
                <div className="user-initial">{user.email.charAt(0).toUpperCase()}</div>
                <span className="user-email-label" title={user.email}>{user.email}</span>
              </div>
              <button
                type="button"
                className="user-logout-btn"
                onClick={handleLogout}
                title="Sign out of workspace"
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"></path>
                  <polyline points="16 17 21 12 16 7"></polyline>
                  <line x1="21" y1="12" x2="9" y2="12"></line>
                </svg>
                <span>Sign Out</span>
              </button>
            </div>
          )}

          <p className="sidebar-tagline">
            Scoped knowledge<br />
            across your documents.
          </p>
          <div className="sidebar-divider" />
          <div className="sidebar-footer-note">
            <span className="sparkle-icon">✦</span>
            <span>Multi-User Isolated</span>
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
          MAIN CONTENT AREA (Warm Editorial Theme)
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
            <div className="header-chat-title-box">
              <span className="current-view-title">
                {activeTab === 'chat' && (activeChat?.title || 'Knowledge Workspace')}
                {activeTab === 'documents' && 'Global Document Catalog'}
                {activeTab === 'about' && 'System Architecture'}
              </span>
              {activeTab === 'chat' && (
                <span className="header-scope-note">
                  {chatDocs.length} {chatDocs.length === 1 ? 'document' : 'documents'} attached
                </span>
              )}
            </div>
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
              <span>{docCount} In Catalog</span>
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

            {/* User Profile Pill */}
            {user && (
              <div className="user-profile-pill" title={`Logged in as ${user.email}`}>
                <span className="user-pill-avatar">{user.email.charAt(0).toUpperCase()}</span>
                <span className="user-pill-email">{user.email}</span>
                <button
                  type="button"
                  className="user-pill-logout"
                  onClick={handleLogout}
                  title="Sign out"
                  aria-label="Sign out"
                >
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"></path>
                    <polyline points="16 17 21 12 16 7"></polyline>
                    <line x1="21" y1="12" x2="9" y2="12"></line>
                  </svg>
                </button>
              </div>
            )}
          </div>
        </header>

        {/* ATTACHED DOCUMENTS BAR (Chat View Only) */}
        {activeTab === 'chat' && (
          <section className="attached-docs-toolbar" aria-label="Attached chat documents">
            <div className="toolbar-inner">
              <div className="toolbar-left">
                <span className="docs-bar-heading">Attached to this chat:</span>
                <div className="docs-chips-container">
                  {chatDocs.map((doc) => (
                    <div key={doc.id} className={`doc-scope-chip status-${doc.status}`}>
                      {(() => { const ft = getFileTypeLabel(doc.filename); return (
                        <span className="doc-type-badge" style={{ background: ft.color }}>{ft.label}</span>
                      ); })()}
                      <span className="doc-chip-filename" title={doc.filename}>
                        {doc.filename}
                      </span>
                      <span className={`doc-status-tag tag-${doc.status}`}>
                        {doc.status === 'processing' && <span className="spinner-dot" />}
                        {doc.status}
                      </span>
                      {doc.chunk_count != null && doc.chunk_count > 0 && (
                        <span className="doc-chunk-count">{doc.chunk_count}c</span>
                      )}
                      <button
                        type="button"
                        className="doc-chip-detach"
                        onClick={() => handleRemoveDocument(doc.id)}
                        title="Remove document from this chat"
                        aria-label={`Remove ${doc.filename} from chat`}
                      >
                        ✕
                      </button>
                    </div>
                  ))}
                  {chatDocs.length === 0 && (
                    <span className="no-docs-hint">No documents attached yet. Upload a PDF, DOCX, CSV, TXT, MD, PPTX, XLSX, or image to start!</span>
                  )}
                </div>
              </div>

              <div className="toolbar-actions">
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".pdf,.docx,.pptx,.xlsx,.xls,.csv,.txt,.md,.markdown,.png,.jpg,.jpeg,.webp"
                  multiple
                  onChange={handleFileUpload}
                  style={{ display: 'none' }}
                  id="doc-upload-input"
                />
                <button
                  type="button"
                  className="upload-trigger-btn"
                  disabled={uploading}
                  onClick={() => fileInputRef.current?.click()}
                >
                  {uploading ? (
                    <>
                      <span className="btn-spinner-icon" aria-hidden="true" />
                      <span>Processing…</span>
                    </>
                  ) : (
                    <>
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2">
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                        <polyline points="17 8 12 3 7 8"></polyline>
                        <line x1="12" y1="3" x2="12" y2="15"></line>
                      </svg>
                      <span>Upload File(s)</span>
                    </>
                  )}
                </button>
              </div>
            </div>

            {uploadError && (
              <div className="upload-error-banner" role="alert">
                <span>⚠️ {uploadError}</span>
                <button type="button" onClick={() => setUploadError(null)}>✕</button>
              </div>
            )}
          </section>
        )}

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
                <p>Initializing workspace…</p>
              </div>
            ) : messages.length === 0 ? (
              /* HERO / EMPTY STATE */
              <div className="hero-empty-state">
                <div className="hero-abstract-art" aria-hidden="true">
                  <div className="art-sun" />
                  <div className="art-hill-1" />
                  <div className="art-hill-2" />
                  <div className="art-arch" />
                </div>

                <div className="hero-heading-group">
                  <h2 className="hero-title">
                    Ask anything<br />
                    about your attached documents
                  </h2>
                  <p className="hero-subtitle">
                    Upload PDF, DOCX, PPTX, XLSX, CSV, TXT, Markdown, or image files — then ask questions with hybrid retrieval and LLaMA 3.2 synthesis.
                  </p>
                </div>

                <div className="example-cards-wrapper">
                  <div className="example-cards-grid">
                    {EXAMPLE_QUESTIONS.map((q, idx) => (
                      <button
                        key={idx}
                        type="button"
                        className="example-question-card"
                        onClick={() => executeQuery(q)}
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
                      {msg.role === 'assistant' && msg.sources && msg.sources.length > 0 && (() => {
                        const map = new Map<string, Source[]>();
                        for (const src of msg.sources) {
                          const fn = src.filename || 'Unknown Document';
                          if (!map.has(fn)) {
                            map.set(fn, []);
                          }
                          map.get(fn)!.push(src);
                        }
                        const groups: { filename: string; chunks: Source[] }[] = [];
                        map.forEach((chunks, filename) => {
                          groups.push({ filename, chunks });
                        });

                        return (
                          <div className="sources-wrapper">
                            <details className="sources-details" open>
                              <summary className="sources-toggle">
                                <span className="sources-label">
                                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                                    <polyline points="14 2 14 8 20 8"></polyline>
                                  </svg>
                                  Sources ({groups.length} {groups.length === 1 ? 'Document' : 'Documents'} · {msg.sources.length} {msg.sources.length === 1 ? 'Chunk' : 'Chunks'})
                                </span>
                                <span className="sources-chevron" aria-hidden="true">▾</span>
                              </summary>

                              <div className="sources-cards-list">
                                {groups.map((group, gi) => (
                                  <div key={gi} className="source-group-card">
                                    <div className="source-group-header">
                                      <span className="source-filename" title={group.filename}>
                                        📄 {group.filename}
                                      </span>
                                      <span className="source-count-badge">
                                        {group.chunks.length} {group.chunks.length === 1 ? 'chunk' : 'chunks'}
                                      </span>
                                    </div>
                                    <div className="source-group-tree">
                                      {group.chunks.map((src, ci) => {
                                        const isLast = ci === group.chunks.length - 1;
                                        const prefix = group.chunks.length === 1 ? '• ' : (isLast ? '└─ ' : '├─ ');
                                        const locParts = [];
                                        if (src.page != null) locParts.push(`Page ${src.page}`);
                                        if (src.slide != null) locParts.push(`Slide ${src.slide}`);
                                        if (src.sheet != null) locParts.push(`Sheet ${src.sheet}`);
                                        if (src.section) locParts.push(`Section: ${src.section}`);
                                        const locStr = locParts.length > 0 ? locParts.join(' · ') : `Chunk ${(src.chunk_index ?? ci) + 1}`;
                                        return (
                                          <div key={ci} className="source-tree-item">
                                            <span className="tree-prefix">{prefix}</span>
                                            <span className="tree-loc">{locStr}</span>
                                            {src.score != null && (
                                              <span className="source-score-badge">Score {src.score.toFixed(2)}</span>
                                            )}
                                          </div>
                                        );
                                      })}
                                    </div>
                                  </div>
                                ))}
                              </div>
                            </details>
                          </div>
                        );
                      })()}
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
                        <span className="thinking-text">Retrieving and synthesizing</span>
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

        {/* TAB 2: GLOBAL DOCUMENTS VIEW */}
        {activeTab === 'documents' && (
          <main className="tab-view-container">
            <div className="tab-view-header">
              <h2 className="tab-view-title">Global Document Catalog</h2>
              <p className="tab-view-subtitle">
                The multi-document RAG assistant indexes {docCount} documents totaling {chunkCount} chunk embeddings in ChromaDB.
              </p>
            </div>

            <div className="documents-catalog-grid">
              {documents?.documents.map((doc, idx) => (
                <div key={idx} className="doc-catalog-card">
                  <div className="doc-card-head">
                    {(() => { const ft = getFileTypeLabel(doc.filename); return (
                      <div className="doc-file-icon-label" style={{ background: ft.color }}>
                        {ft.label}
                      </div>
                    ); })()}
                    <div>
                      <h3 className="doc-name">{doc.filename}</h3>
                      <span className="doc-meta-counts">
                        {doc.page_count != null ? `${doc.page_count} Pages/Slides/Sheets · ` : ''}{doc.chunk_count} Chunks
                      </span>
                    </div>
                  </div>
                  {doc.sections && doc.sections.length > 0 && (
                    <div className="doc-sections-list">
                      <span className="sections-title">Indexed Sections:</span>
                      <div className="sections-chips">
                        {doc.sections.map((sec, sidx) => (
                          <span key={sidx} className="section-chip">{sec}</span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>

            <div className="tab-cta-wrap">
              <button
                type="button"
                className="cta-terracotta-btn"
                onClick={() => setActiveTab('chat')}
              >
                ← Return to Active Chat
              </button>
            </div>
          </main>
        )}

        {/* TAB 3: ABOUT VIEW */}
        {activeTab === 'about' && (
          <main className="tab-view-container">
            <div className="tab-view-header">
              <h2 className="tab-view-title">System Architecture</h2>
              <p className="tab-view-subtitle">
                A multi-session, multi-document retrieval-augmented generation assistant designed for chat-scoped retrieval and document reuse.
              </p>
            </div>

            <div className="about-features-grid">
              <div className="about-feature-card">
                <div className="feature-icon">🔒</div>
                <h3>Chat-Scoped Retrieval</h3>
                <p>Retrieval queries are strictly isolated to documents attached to the active chat session using Chroma metadata filtering.</p>
              </div>
              <div className="about-feature-card">
                <div className="feature-icon">♻️</div>
                <h3>Deduplication & Reuse</h3>
                <p>Files are hashed with SHA-256 and indexed once. Multiple chats attach to the global catalog without re-embedding.</p>
              </div>
              <div className="about-feature-card">
                <div className="feature-icon">💾</div>
                <h3>Full SQLite Persistence</h3>
                <p>Chats, attached document relationships, and user/assistant messages with provenance citations persist across browser refreshes.</p>
              </div>
            </div>

            <div className="tab-cta-wrap">
              <button
                type="button"
                className="cta-terracotta-btn"
                onClick={() => setActiveTab('chat')}
              >
                ← Return to Active Chat
              </button>
            </div>
          </main>
        )}

        {/* ==================================================
            BOTTOM CHAT INPUT AREA
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
                  placeholder={
                    chatDocs.length === 0
                      ? 'Upload a PDF above to ask questions in this chat...'
                      : 'Ask a question about the attached documents...'
                  }
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

      {/* DELETE CHAT CONFIRMATION MODAL */}
      {deletingChatId && (
        <div className="modal-backdrop" onClick={() => setDeletingChatId(null)}>
          <div className="editorial-modal" onClick={(e) => e.stopPropagation()} role="dialog" aria-modal="true">
            <h3 className="modal-title">Delete Chat Session?</h3>
            <p className="modal-body">
              This will remove this chat session and its message history. Attached documents will remain safely available in the global catalog.
            </p>
            <div className="modal-actions">
              <button
                type="button"
                className="modal-cancel-btn"
                onClick={() => setDeletingChatId(null)}
              >
                Cancel
              </button>
              <button
                type="button"
                className="modal-danger-btn"
                onClick={() => handleDeleteChat(deletingChatId)}
              >
                Delete Chat
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;