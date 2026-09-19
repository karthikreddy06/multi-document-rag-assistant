const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

export interface HealthResponse {
  status: string;
}

export interface DocumentInfo {
  filename: string;
  file_hash: string;
  chunk_count: number;
  page_count: number;
  sections: string[];
}

export interface DocumentsResponse {
  total_documents: number;
  total_chunks: number;
  documents: DocumentInfo[];
}

export interface ChatSession {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface DocumentRecord {
  id: string;
  filename: string;
  file_hash: string;
  file_size: number;
  page_count?: number | null;
  chunk_count?: number | null;
  storage_path?: string | null;
  status: 'pending' | 'processing' | 'ready' | 'failed';
  error_message?: string | null;
  created_at: string;
  attached_at?: string | null;
}

export interface DocumentUploadResponse {
  message: string;
  document: DocumentRecord;
}

export interface Source {
  filename: string;
  page?: number | null;
  slide?: number | null;
  sheet?: number | null;
  section?: string | null;
  chunk_id?: string | null;
  chunk_index?: number | null;
  score?: number | null;
  text?: string | null;
}

export interface ChatResponse {
  answer: string;
  sources: Source[];
}

export interface ChatMessage {
  id: string;
  chat_id: string;
  role: 'user' | 'assistant';
  content: string;
  sources: Source[];
  created_at: string;
}

export interface IngestResponse {
  status: string;
  documents_processed: number;
  chunks_stored: number;
  message: string;
}

async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
    ...options,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }

  return response.json();
}

export const api = {
  health: () => fetchJson<HealthResponse>(`${API_BASE_URL}/api/health`),

  documents: () => fetchJson<DocumentsResponse>(`${API_BASE_URL}/api/documents`),

  // Legacy global chat
  chat: (query: string) =>
    fetchJson<ChatResponse>(`${API_BASE_URL}/api/chat`, {
      method: 'POST',
      body: JSON.stringify({ query }),
    }),

  ingest: () =>
    fetchJson<IngestResponse>(`${API_BASE_URL}/api/ingest`, {
      method: 'POST',
    }),

  // Chat Sessions CRUD
  listChats: () => fetchJson<ChatSession[]>(`${API_BASE_URL}/api/chats`),

  createChat: (title?: string) =>
    fetchJson<ChatSession>(`${API_BASE_URL}/api/chats`, {
      method: 'POST',
      body: JSON.stringify(title ? { title } : {}),
    }),

  getChat: (chatId: string) => fetchJson<ChatSession>(`${API_BASE_URL}/api/chats/${chatId}`),

  renameChat: (chatId: string, title: string) =>
    fetchJson<ChatSession>(`${API_BASE_URL}/api/chats/${chatId}`, {
      method: 'PATCH',
      body: JSON.stringify({ title }),
    }),

  deleteChat: async (chatId: string): Promise<void> => {
    const response = await fetch(`${API_BASE_URL}/api/chats/${chatId}`, {
      method: 'DELETE',
    });
    if (!response.ok) {
      const err = await response.json().catch(() => ({ detail: 'Failed to delete chat' }));
      throw new Error(err.detail || `HTTP ${response.status}`);
    }
  },

  // Chat Documents
  listChatDocuments: (chatId: string) =>
    fetchJson<DocumentRecord[]>(`${API_BASE_URL}/api/chats/${chatId}/documents`),

  uploadChatDocument: async (chatId: string, file: File): Promise<DocumentUploadResponse> => {
    const formData = new FormData();
    formData.append('file', file);
    const response = await fetch(`${API_BASE_URL}/api/chats/${chatId}/documents`, {
      method: 'POST',
      body: formData,
    });
    if (!response.ok) {
      const err = await response.json().catch(() => ({ detail: 'Failed to upload document' }));
      throw new Error(err.detail || `HTTP ${response.status}`);
    }
    return response.json();
  },

  removeChatDocument: async (chatId: string, documentId: string): Promise<void> => {
    const response = await fetch(`${API_BASE_URL}/api/chats/${chatId}/documents/${documentId}`, {
      method: 'DELETE',
    });
    if (!response.ok) {
      const err = await response.json().catch(() => ({ detail: 'Failed to remove document' }));
      throw new Error(err.detail || `HTTP ${response.status}`);
    }
  },

  // Session-Scoped Chat
  chatInSession: (chatId: string, query: string, showContext: boolean = true) =>
    fetchJson<ChatResponse>(`${API_BASE_URL}/api/chats/${chatId}/chat`, {
      method: 'POST',
      body: JSON.stringify({ query, show_context: showContext }),
    }),

  // Session-Scoped Chat Streaming (SSE)
  chatInSessionStream: async (
    chatId: string,
    query: string,
    onToken: (token: string) => void,
    onSources: (sources: Source[]) => void,
    showContext: boolean = true
  ): Promise<void> => {
    const response = await fetch(`${API_BASE_URL}/api/chats/${chatId}/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ query, show_context: showContext }),
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({ detail: 'Failed to stream response' }));
      throw new Error(err.detail || `HTTP ${response.status}`);
    }

    const reader = response.body?.getReader();
    if (!reader) throw new Error('Response body is not readable');

    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed || !trimmed.startsWith('data: ')) continue;
        const dataStr = trimmed.slice(6);
        if (dataStr === '[DONE]') break;

        try {
          const parsed = JSON.parse(dataStr);
          if (parsed.type === 'token') {
            onToken(parsed.token);
          } else if (parsed.type === 'sources') {
            onSources(parsed.sources);
          } else if (parsed.type === 'error') {
            throw new Error(parsed.error);
          }
        } catch (e) {
          if (e instanceof Error && e.message !== 'Unexpected end of JSON input') {
            throw e;
          }
        }
      }
    }
  },

  // Message History
  listMessages: (chatId: string) =>
    fetchJson<ChatMessage[]>(`${API_BASE_URL}/api/chats/${chatId}/messages`),
};