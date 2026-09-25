const rawApiUrl = import.meta.env.VITE_API_URL;
const sanitizedApiUrl = typeof rawApiUrl === 'string' ? rawApiUrl.replace(/^["'\s]+|["'\s]+$/g, '') : '';
const API_BASE_URL = sanitizedApiUrl || (import.meta.env.DEV ? 'http://127.0.0.1:8000' : '');

export const TOKEN_STORAGE_KEY = 'rag_auth_token';

export function getAuthToken(): string | null {
  return localStorage.getItem(TOKEN_STORAGE_KEY);
}

export function setAuthToken(token: string | null): void {
  if (token) {
    localStorage.setItem(TOKEN_STORAGE_KEY, token);
  } else {
    localStorage.removeItem(TOKEN_STORAGE_KEY);
  }
}

// Global 401 listener callback
let onUnauthorizedCallback: (() => void) | null = null;

export function setOnUnauthorized(callback: (() => void) | null) {
  onUnauthorizedCallback = callback;
}

function notifyUnauthorized() {
  setAuthToken(null);
  if (onUnauthorizedCallback) {
    onUnauthorizedCallback();
  }
}

/**
 * Custom fetch wrapper that automatically adds the Authorization header
 * and intercepts 401 Unauthorized responses to trigger session invalidation.
 */
export async function authenticatedFetch(url: string, options: RequestInit = {}): Promise<Response> {
  const token = getAuthToken();
  const headers = new Headers(options.headers || {});

  if (token && !headers.has('Authorization')) {
    headers.set('Authorization', `Bearer ${token}`);
  }

  const response = await fetch(url, {
    ...options,
    headers,
  });

  if (response.status === 401) {
    // Exclude login and register requests from triggering global redirect
    if (!url.includes('/api/auth/login') && !url.includes('/api/auth/register')) {
      notifyUnauthorized();
    }
  }

  return response;
}

export interface User {
  id: string;
  email: string;
  created_at: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
  user: User;
}

export interface HealthResponse {
  status: string;
}

export interface DocumentInfo {
  id?: string;
  filename: string;
  file_type?: string;
  file_size?: number;
  file_hash: string;
  chunk_count: number;
  page_count: number | null;
  sections: string[];
  status?: string;
  created_at?: string;
  storage_path?: string | null;
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

async function parseResponseError(response: Response, defaultMsg?: string): Promise<string> {
  try {
    const errorData = await response.json();
    if (typeof errorData?.detail === 'string') {
      return errorData.detail;
    }
    if (Array.isArray(errorData?.detail)) {
      return errorData.detail
        .map((d: { msg?: string }) => d.msg || JSON.stringify(d))
        .join('; ');
    }
    if (typeof errorData?.message === 'string') {
      return errorData.message;
    }
  } catch {
    // Non-JSON response body
  }
  return defaultMsg || `HTTP ${response.status}`;
}

async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
  const headers = new Headers(options?.headers || {});
  if (!headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json');
  }

  const response = await authenticatedFetch(url, {
    ...options,
    headers,
  });

  if (!response.ok) {
    const msg = await parseResponseError(response);
    throw new Error(msg);
  }

  return response.json();
}

export const api = {
  // Authentication Endpoints
  register: (email: string, password: string) =>
    fetchJson<User>(`${API_BASE_URL}/api/auth/register`, {
      method: 'POST',
      body: JSON.stringify({ email, password }),
    }),

  login: (email: string, password: string) =>
    fetchJson<TokenResponse>(`${API_BASE_URL}/api/auth/login`, {
      method: 'POST',
      body: JSON.stringify({ email, password }),
    }),

  getMe: () => fetchJson<User>(`${API_BASE_URL}/api/auth/me`),

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
    const response = await authenticatedFetch(`${API_BASE_URL}/api/chats/${chatId}`, {
      method: 'DELETE',
    });
    if (!response.ok) {
      const msg = await parseResponseError(response, 'Failed to delete chat');
      throw new Error(msg);
    }
  },

  // Chat Documents
  listChatDocuments: (chatId: string) =>
    fetchJson<DocumentRecord[]>(`${API_BASE_URL}/api/chats/${chatId}/documents`),

  uploadChatDocument: async (chatId: string, file: File): Promise<DocumentUploadResponse> => {
    const formData = new FormData();
    formData.append('file', file);
    const response = await authenticatedFetch(`${API_BASE_URL}/api/chats/${chatId}/documents`, {
      method: 'POST',
      body: formData,
    });
    if (!response.ok) {
      const msg = await parseResponseError(response, 'Failed to upload document');
      throw new Error(msg);
    }
    return response.json();
  },

  removeChatDocument: async (chatId: string, documentId: string): Promise<void> => {
    const response = await authenticatedFetch(`${API_BASE_URL}/api/chats/${chatId}/documents/${documentId}`, {
      method: 'DELETE',
    });
    if (!response.ok) {
      const msg = await parseResponseError(response, 'Failed to remove document');
      throw new Error(msg);
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
    const response = await authenticatedFetch(`${API_BASE_URL}/api/chats/${chatId}/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ query, show_context: showContext }),
    });

    if (!response.ok) {
      const msg = await parseResponseError(response, 'Failed to stream response');
      throw new Error(msg);
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
        if (!trimmed || !trimmed.startsWith('data:')) continue;
        const dataStr = trimmed.replace(/^data:\s*/, '');
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

  // File Library View / Download / Delete
  viewDocumentBlobUrl: async (documentId: string): Promise<string> => {
    const response = await authenticatedFetch(`${API_BASE_URL}/api/documents/${documentId}/view`);
    if (!response.ok) {
      const msg = await parseResponseError(response, 'Failed to view document');
      throw new Error(msg);
    }
    const blob = await response.blob();
    return URL.createObjectURL(blob);
  },

  downloadDocument: async (documentId: string, filename: string): Promise<void> => {
    const response = await authenticatedFetch(`${API_BASE_URL}/api/documents/${documentId}/download`);
    if (!response.ok) {
      const msg = await parseResponseError(response, 'Failed to download document');
      throw new Error(msg);
    }
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  },

  deleteDocument: async (documentId: string): Promise<void> => {
    const response = await authenticatedFetch(`${API_BASE_URL}/api/documents/${documentId}`, {
      method: 'DELETE',
    });
    if (!response.ok) {
      const msg = await parseResponseError(response, 'Failed to delete document');
      throw new Error(msg);
    }
  },

  attachDocumentToChat: (chatId: string, documentId: string) =>
    fetchJson<DocumentRecord>(`${API_BASE_URL}/api/chats/${chatId}/documents/${documentId}`, {
      method: 'POST',
    }),

  // File Conversion
  getConversionTargets: (documentId: string) =>
    fetchJson<{ document_id: string; filename: string; current_format: string; supported_targets: string[] }>(
      `${API_BASE_URL}/api/documents/${documentId}/convert/targets`
    ),

  convertDocument: async (documentId: string, targetFormat: string, fallbackFilename: string): Promise<void> => {
    const response = await authenticatedFetch(`${API_BASE_URL}/api/documents/${documentId}/convert`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ target_format: targetFormat }),
    });
    if (!response.ok) {
      const msg = await parseResponseError(response, 'Failed to convert document');
      throw new Error(msg);
    }
    const disposition = response.headers.get('Content-Disposition') || '';
    let filename = fallbackFilename;
    const match = disposition.match(/filename="?([^";]+)"?/i);
    if (match && match[1]) {
      filename = match[1].trim();
    }
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  },
};