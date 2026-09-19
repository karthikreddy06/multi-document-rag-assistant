# Dockerfile for Multi-Document RAG Assistant Backend
FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffer outputs
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy backend dependencies
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend application code
COPY . .

# Create writable data directories
RUN mkdir -p /app/data/uploads /app/data /app/chroma_db /app/documents

EXPOSE 8000

# Start FastAPI server binding to 0.0.0.0 and PORT env variable
CMD ["sh", "-c", "uvicorn app.api.routes:app --host 0.0.0.0 --port ${PORT:-8000}"]
