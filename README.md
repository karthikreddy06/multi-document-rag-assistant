# Multi-Document RAG Assistant

A robust, production-ready local Retrieval-Augmented Generation (RAG) assistant built with Python, Ollama, ChromaDB, and hybrid search. The system is designed to ingest heterogeneous PDF documents, accurately retrieve information across multiple documents and sections, and generate strictly grounded answers without hallucinations.

---

## Features

- **Multi-Document PDF Ingestion**: Extracts text, page numbers, and structural metadata from heterogeneous PDF documents using `pypdf`.
- **Semantic & Vector Retrieval**: Dense vector embeddings generated locally via `nomic-embed-text` and indexed in ChromaDB.
- **Hybrid Retrieval**: Combines dense semantic vector search with boundary-aware lexical keyword matching, token coverage ratios, and exact phrase scoring.
- **Query Decomposition**: Decomposes multi-part, conjunctive, and comparative questions into distinct information needs while preserving compound titles and introductory query scope.
- **Multi-Part Question Handling**: Guarantees aspect coverage across multiple distinct clauses so that co-located or multi-clause facts are not crowded out.
- **Cross-Document Comparison**: Automatically distributes comparison attributes across compared subjects and retrieves balanced evidence across different documents.
- **Grounded Generation**: Synthesizes responses strictly from retrieved document chunks using local `llama3.2` with zero external API calls.
- **Local Vector Storage (ChromaDB)**: Embeddings and metadata are persisted locally without requiring cloud database infrastructure.
- **Local LLM Execution (Ollama)**: Runs completely offline using local models via Ollama.
- **Incremental & Idempotent Ingestion**: Tracks SHA-256 document content hashes to avoid redundant re-indexing and automatically purges deleted documents from the vector store.
- **Source & Page Provenance**: Every retrieved context chunk retains source document name, page number, section header, and deterministic chunk sequence ID.
- **Error Handling, Retries & Timeouts**: Configurable client timeouts and exponential backoff retry loops protect external Ollama calls from socket drops and model-swapping latency.

---

## Architecture

```text
┌────────────────────────────────────────────────────────┐
│                   PDF Documents                        │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│               PDF Loader & Text Cleaner                │
│    (pypdf extraction + typography & kerning repair)    │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│         Hierarchical Document Chunking                 │
│      (Section headers, titled sub-entries)             │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│               Ollama Embeddings Service                │
│      (nomic-embed-text dense vector generation)        │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│                 ChromaDB Vector Store                  │
│       (Local persistence + SHA-256 hash tracking)      │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│               Generic Query Decomposer                 │
│   (Clause splitting, scope preservation, attributes)   │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│                 Hybrid Retriever                       │
│ (Dense search + Word-boundary lexical + Aspect RRF +   │
│           Document-locality spatial bonus)             │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│               Context Construction                     │
│      (Provenance headers + Grounding guardrails)       │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│               Ollama LLM Generator                     │
│         (llama3.2 local grounded synthesis)            │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│              Accurate, Grounded Answer                 │
└────────────────────────────────────────────────────────┘
```

---

## Tech Stack

- **Language**: Python 3.11+
- **Vector Database**: ChromaDB (`chromadb>=0.5.0`)
- **Embeddings & LLM**: Ollama (`ollama>=0.3.0`)
  - Embedding Model: `nomic-embed-text`
  - Generation Model: `llama3.2`
- **PDF Extraction**: `pypdf>=5.0.0`
- **Text Splitting**: `langchain-text-splitters>=0.2.0`
- **Configuration & Validation**: Pydantic & Pydantic Settings (`pydantic>=2.0.0`, `pydantic-settings>=2.0.0`)
- **Testing**: `pytest>=8.0.0`

---

## Project Structure

```text
rag-project/
├── .env.example                  # Environment configuration template
├── .gitignore                    # Git exclusion rules for Python, ChromaDB & caches
├── pytest.ini                    # Pytest configuration and marker registration
├── README.md                     # Project documentation
├── requirements.txt              # Production Python dependencies
├── app/
│   ├── __init__.py               # Application package marker
│   ├── config.py                 # Centralized configuration (Pydantic Settings)
│   ├── main.py                   # Main CLI entrypoint and RAGApplication orchestrator
│   ├── models.py                 # Core dataclasses (Document, Chunk, RetrievedChunk)
│   ├── rag.py                    # Backward-compatible entrypoint
│   ├── embeddings/
│   │   ├── __init__.py
│   │   └── service.py            # Ollama embedding wrapper with retry logic
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── generator.py          # Ollama LLM generator with retry/timeout logic
│   │   └── prompts.py            # Grounded prompt templates and context formatting
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── chunker.py            # Hierarchical section- and entry-aware text chunker
│   │   ├── loader.py             # PDF loader with metadata extraction
│   │   └── pipeline.py           # Ingestion pipeline with hash-based synchronization
│   ├── retrieval/
│   │   ├── __init__.py
│   │   └── retriever.py          # QueryDecomposer and HybridRetriever with RRF
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logger.py             # Structured application logger
│   │   └── text_cleaner.py       # Generic text, drop-cap, and whitespace cleaners
│   └── vectorstore/
│       ├── __init__.py
│       └── store.py              # ChromaDB vector store wrapper
├── documents/                    # PDF documents directory
│   ├── Alice_in_Wonderland.pdf   # Public domain sample book
│   ├── oldmansea.pdf             # Sample literary text
│   └── sample.pdf                # Sample LaTeX / PDF reference document
└── tests/
    ├── __init__.py
    ├── test_chunker.py           # Chunking and section detection tests
    ├── test_embeddings.py        # Embedding service unit tests
    ├── test_loader.py            # PDF loading and cleaning tests
    ├── test_prompts.py           # Prompt formatting and guardrail tests
    ├── test_rag_e2e.py           # End-to-end evaluation queries (marked: llm)
    ├── test_rag_regression_suite.py # 15-category regression suite (marked: llm for generation)
    ├── test_retrieval_generic.py # Generic multi-document retrieval tests
    ├── test_retriever.py         # Retriever unit tests
    └── test_vectorstore.py       # ChromaDB upsert, query, and idempotency tests
```

---

## Installation

### 1. Clone Repository
```powershell
git clone https://github.com/karthikreddy06/multi-document-rag-assistant.git
cd multi-document-rag-assistant
```

### 2. Create and Activate Virtual Environment
On Windows (PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

On Linux/macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

---

## Ollama Setup

1. **Download & Install Ollama**:
   Install Ollama from [ollama.com](https://ollama.com) and ensure the Ollama background service is running.

2. **Pull Required Models**:
   Open a terminal and download the configured embedding and generation models:
   ```powershell
   ollama pull nomic-embed-text
   ollama pull llama3.2
   ```

3. **Verify Ollama is Running**:
   ```powershell
   curl http://localhost:11434/api/tags
   ```

---

## Environment Setup

Copy the example environment configuration file:
```powershell
cp .env.example .env
```

Review `.env` settings if your setup uses custom ports or directories:
```ini
OLLAMA_HOST=http://localhost:11434
EMBEDDING_MODEL=nomic-embed-text
LLM_MODEL=llama3.2
OLLAMA_TIMEOUT=60.0
CHROMA_PATH=./chroma_db
COLLECTION_NAME=documents
DOCUMENTS_DIR=./documents
CHUNK_SIZE=800
CHUNK_OVERLAP=100
TOP_K=5
LOG_LEVEL=INFO
```

---

## Adding Documents & Ingestion

1. Place your `.pdf` files inside the `documents/` directory.
2. The application automatically detects and indexes new or modified PDFs on startup.
3. Unchanged documents are recognized by their SHA-256 hash and will not be re-indexed unnecessarily.
4. If a document is deleted from `documents/`, its corresponding chunks are automatically removed from the ChromaDB collection.

To explicitly trigger ingestion via CLI:
```powershell
python -m app.main --ingest
```

To clear and re-index all documents from scratch:
```powershell
python -m app.main --reindex
```

---

## Running the Application

### 1. Start FastAPI Backend Server
From the project root:
```powershell
cd backend
python -m uvicorn app.api.routes:app --host 127.0.0.1 --port 8000
```
- API Base URL: `http://127.0.0.1:8000`
- OpenAPI Swagger UI: `http://127.0.0.1:8000/docs`
- Health Endpoint: `http://127.0.0.1:8000/api/health`

### 2. Start Vite React Frontend UI
From a separate terminal window in the project root:
```powershell
cd frontend
npm install
npm run dev
```
- Web Application URL: `http://localhost:5173`

### 3. CLI Interactive Session (Optional)
Run the interactive CLI interface directly:
```powershell
cd backend
python -m app.main
```
Type your question at the prompt and press Enter. Type `exit` or `quit` to end the session.

### 4. One-Shot Query Execution (CLI)
Execute a single query directly from the command line:
```powershell
cd backend
python -m app.main --query "What is the primary methodology described in the document?" --show-context
```

---

## Testing

The test suite is partitioned using pytest markers to separate fast unit/retrieval verification from slower local LLM generation tests.

### Fast Tests (Unit, Ingestion, VectorStore, Retrieval)
Runs all unit and retrieval tests without invoking local LLM generation:
```powershell
pytest -q -m "not llm"
```
*(50 tests run and pass in ~40 seconds)*

### LLM End-to-End Tests
Executes the full end-to-end tests involving local `llama3.2` generation:
```powershell
pytest -q -m llm
```
*Note: LLM test execution speed depends on your local GPU/CPU hardware and model swapping between `nomic-embed-text` and `llama3.2`.*

### Specific Test Modules
```powershell
pytest -q tests/test_prompts.py
pytest -q tests/test_retrieval_generic.py
pytest -q tests/test_rag_regression_suite.py::test_regression_a_old_man_character_and_duration
```

---

## Example Questions

- **Single-Fact Factual Question**:
  `"How do you compile a .tex document to a PDF file?"`
- **Multi-Part Question**:
  `"What is the name of the main character in The Old Man and the Sea, and how long had he gone without catching a fish?"`
- **Cross-Document Comparison**:
  `"Compare the main characters in Alice in Wonderland and The Old Man and the Sea."`
- **Broad Overview Question**:
  `"What is the main topic of the Sample PDF Document?"`
- **Unsupported / Unknown Question (Guardrail Verification)**:
  `"What was the weather like in New York on January 1st, 1800 according to the documents?"`
  *(Expected behavior: The assistant safely declines to answer due to lack of evidence in the provided context).*

---

## Grounding & Limitations

- **Strict Grounding**: The assistant is strictly constrained to the retrieved context chunks. If the necessary information is not present in the indexed documents, the assistant will explicitly state that it does not have enough information rather than hallucinating.
- **Local Inference Performance**: Response latency is governed by local hardware specifications. When running both embeddings and LLM on consumer hardware, model weights may be swapped in VRAM by Ollama, which can add a brief latency buffer between retrieval and generation.
- **Format Support**: The ingestion pipeline currently processes PDF documents (`.pdf`). Support for additional formats (e.g., DOCX, Markdown) can be added by extending the document loader.

---

## License

No license has been selected for this repository yet. (TODO: Select an appropriate open-source license such as MIT or Apache-2.0 if making this project publicly open source).
