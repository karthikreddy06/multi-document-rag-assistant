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

export interface ChatRequest {
  query: string;
}

export interface Source {
  filename: string;
  page: number;
  section: string;
  chunk_id: string;
  score: number;
  text: string | null;
}

export interface ChatResponse {
  answer: string;
  sources: Source[];
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

  chat: (query: string) =>
    fetchJson<ChatResponse>(`${API_BASE_URL}/api/chat`, {
      method: 'POST',
      body: JSON.stringify({ query }),
    }),

  ingest: () =>
    fetchJson<IngestResponse>(`${API_BASE_URL}/api/ingest`, {
      method: 'POST',
    }),
};