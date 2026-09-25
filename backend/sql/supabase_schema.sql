-- =============================================================================
-- Supabase PostgreSQL Schema for Multi-Document RAG Assistant
-- Project Identifier: https://ldytwnvxskfajjwcxxtb.supabase.co
-- Description: Core relational schema for user authentication, chat sessions,
--              document catalogs, chat-document links, and message history.
-- =============================================================================

-- -----------------------------------------------------------------------------
-- 0. Extensions
-- -----------------------------------------------------------------------------
-- Enable CITEXT (Case-Insensitive Text) for user emails to ensure case-insensitive
-- uniqueness without requiring manual LOWER() transformations.
CREATE EXTENSION IF NOT EXISTS citext;


-- -----------------------------------------------------------------------------
-- 1. Users Table
-- -----------------------------------------------------------------------------
-- Stores authenticated user credentials. Primary keys use application-generated
-- prefixed IDs (e.g., 'usr_...'). Email is case-insensitive and unique.
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    email CITEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);


-- -----------------------------------------------------------------------------
-- 2. Chats Table
-- -----------------------------------------------------------------------------
-- Stores user-scoped conversation threads. Cascades deletion when the owning
-- user is deleted.
CREATE TABLE IF NOT EXISTS chats (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title TEXT NOT NULL DEFAULT 'New Chat',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);


-- -----------------------------------------------------------------------------
-- 3. Documents Table
-- -----------------------------------------------------------------------------
-- Stores user-scoped document metadata. Enforces unique (user_id, file_hash)
-- to prevent duplicate file uploads per user. Cascades deletion when the owning
-- user is deleted.
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    filename TEXT NOT NULL,
    file_hash TEXT NOT NULL,
    file_size BIGINT NOT NULL,
    page_count INTEGER,
    chunk_count INTEGER,
    storage_path TEXT,
    status TEXT NOT NULL CHECK (
        status IN ('pending', 'processing', 'ready', 'failed')
    ),
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_user_document_hash UNIQUE (user_id, file_hash)
);


-- -----------------------------------------------------------------------------
-- 4. Chat-Documents Table (Junction)
-- -----------------------------------------------------------------------------
-- Associates catalog documents with specific chat sessions. Composite primary
-- key (chat_id, document_id) prevents duplicate attachments. Cascades deletion
-- when either the chat or the document is removed.
CREATE TABLE IF NOT EXISTS chat_documents (
    chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (chat_id, document_id)
);


-- -----------------------------------------------------------------------------
-- 5. Messages Table
-- -----------------------------------------------------------------------------
-- Stores conversation messages associated with chat sessions. Cascades deletion
-- when the parent chat is removed. Source chunk provenance is stored in native
-- JSONB for structured querying and performance.
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (
        role IN ('user', 'assistant', 'system')
    ),
    content TEXT NOT NULL,
    sources_json JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);


-- -----------------------------------------------------------------------------
-- 6. Performance Indexes
-- -----------------------------------------------------------------------------
-- Scoping index for user chats
CREATE INDEX IF NOT EXISTS idx_chats_user_id
    ON chats(user_id);

-- Ordering index for recent chat lists
CREATE INDEX IF NOT EXISTS idx_chats_updated_at
    ON chats(updated_at DESC);

-- Scoping index for user document catalog
CREATE INDEX IF NOT EXISTS idx_documents_user_id
    ON documents(user_id);

-- Lookup index for duplicate file hash detection
CREATE INDEX IF NOT EXISTS idx_documents_file_hash
    ON documents(file_hash);

-- Retrieval index for chat session document resolution
CREATE INDEX IF NOT EXISTS idx_chat_documents_chat_id
    ON chat_documents(chat_id);

-- Lookup index for document-to-chat relationships
CREATE INDEX IF NOT EXISTS idx_chat_documents_document_id
    ON chat_documents(document_id);

-- Scoping index for chat message history retrieval
CREATE INDEX IF NOT EXISTS idx_messages_chat_id
    ON messages(chat_id);

-- Ordering index for chronological message stream
CREATE INDEX IF NOT EXISTS idx_messages_created_at
    ON messages(created_at ASC);

-- -----------------------------------------------------------------------------
-- 7. Document Chunks & pgvector Storage
-- -----------------------------------------------------------------------------
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS document_chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    embedding vector(384) NOT NULL,
    filename TEXT NOT NULL DEFAULT '',
    page_number INTEGER,
    section TEXT,
    metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_doc_chunks_doc_id ON document_chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_doc_chunks_user_id ON document_chunks(user_id);
CREATE INDEX IF NOT EXISTS idx_doc_chunks_embedding ON document_chunks USING hnsw (embedding vector_cosine_ops);

-- -----------------------------------------------------------------------------
-- 8. Storage Bucket & RLS Policy
-- -----------------------------------------------------------------------------
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.schemata WHERE schema_name = 'storage') THEN
        IF NOT EXISTS (
            SELECT 1 FROM pg_policies
            WHERE schemaname = 'storage' AND tablename = 'objects' AND policyname = 'rag_files_bucket_policy'
        ) THEN
            CREATE POLICY rag_files_bucket_policy
            ON storage.objects
            FOR ALL
            TO public, anon, authenticated, service_role
            USING (bucket_id = 'rag-files')
            WITH CHECK (bucket_id = 'rag-files');
        END IF;
    END IF;
END
$$;

