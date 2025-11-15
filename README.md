# Policy Document Summarization Assistant

A complete backend-only system for ingesting long insurance policy PDFs, extracting text, chunking, generating embeddings, summarizing, performing hallucination checks, and returning final summaries via REST APIs.

## Features

- **Document Upload**: Upload PDF or DOCX files and extract text
- **Intelligent Chunking**: Token-aware text chunking using LangChain
- **Embedding Generation**: Generate embeddings using OpenAI's text-embedding-3-large
- **Vector Storage**: Store embeddings in FAISS (local) or MongoDB Atlas Vector Search
- **AI Summarization**: Generate multiple types of summaries using GPT-4o-mini:
  - High-level overview (2-4 paragraphs)
  - Bullet-point summary (5-10 key points)
  - Section-level summaries
- **Hallucination Detection**: Validate summaries against original content using embedding similarity
- **Export**: Download summaries as PDF or JSON

## Tech Stack

- **Python 3.11**
- **FastAPI** - Modern, fast web framework
- **LangChain** - Text processing and chunking
- **PyMuPDF / pdfplumber** - PDF text extraction
- **OpenAI API** - Embeddings and summarization
- **FAISS / MongoDB** - Vector storage
- **AWS S3** - Document storage
- **ReportLab** - PDF generation
- **Pytest** - Testing framework

## Project Structure

```
backend/
    api/                    # API endpoints
        upload.py          # Document upload endpoint
        chunk.py           # Chunking endpoint
        embedding.py       # Embedding generation endpoint
        summarize.py       # Summarization endpoint
        qa_check.py        # QA validation endpoint
        download.py        # Download endpoint
    services/              # Business logic services
        s3_service.py      # AWS S3 operations
        extractor.py       # Text extraction
        chunker.py         # Text chunking
        embedder.py        # Embedding generation
        vector_store.py    # Vector database operations
        summarizer.py      # AI summarization
        qa_validator.py    # Hallucination detection
    core/                  # Core utilities
        config.py          # Configuration management
        utils.py           # Utility functions
        schemas.py         # Pydantic models
    tests/                 # Test suite
        test_upload.py
        test_summary.py
        test_embeddings.py
    main.py                # FastAPI application
```

## Installation

### Prerequisites

- Python 3.11+
- AWS account with S3 bucket (or use LocalStack for local development)
- OpenAI API key

### Setup

1. **Clone the repository**:
```bash
git clone <repository-url>
cd project27
```

2. **Create virtual environment**:
```bash
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Configure environment variables**:
```bash
cp .env.example .env
# Edit .env with your credentials
```

5. **Set up environment variables** (create `.env` file):
```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-large

AWS_ACCESS_KEY_ID=your_aws_access_key_id
AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key
AWS_REGION=us-east-1
AWS_BUCKET=policy-documents

VECTOR_DB=faiss  # or "mongo" for MongoDB
FAISS_INDEX_PATH=./data/faiss_index

# Optional: MongoDB configuration (if using MongoDB)
# MONGODB_URI=mongodb+srv://user:password@cluster.mongodb.net/
# MONGODB_DB_NAME=policy_docs
# MONGODB_COLLECTION=embeddings
```

## Usage

### Running the Server

```bash
uvicorn backend.main:app --reload
```

The API will be available at `http://localhost:8000`

### API Documentation

Once the server is running, visit:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### API Endpoints

#### 1. Upload Document
```bash
POST /upload
Content-Type: multipart/form-data

curl -X POST "http://localhost:8000/upload" \
  -F "file=@policy_document.pdf"
```

**Response**:
```json
{
  "document_id": "uuid-here",
  "filename": "policy_document.pdf",
  "file_size": 1024000,
  "file_type": "pdf",
  "status": "uploaded",
  "message": "Document uploaded and text extracted successfully"
}
```

#### 2. Chunk Document
```bash
POST /chunk/{document_id}

curl -X POST "http://localhost:8000/chunk/{document_id}"
```

**Response**:
```json
{
  "document_id": "uuid-here",
  "total_chunks": 15,
  "chunks": [...],
  "status": "chunked"
}
```

#### 3. Generate Embeddings
```bash
POST /embed/{document_id}

curl -X POST "http://localhost:8000/embed/{document_id}"
```

**Response**:
```json
{
  "document_id": "uuid-here",
  "total_embeddings": 15,
  "embedding_dimension": 3072,
  "vector_store": "faiss",
  "status": "embedded"
}
```

#### 4. Summarize Document
```bash
POST /summarize/{document_id}

curl -X POST "http://localhost:8000/summarize/{document_id}"
```

**Response**:
```json
{
  "document_id": "uuid-here",
  "overview": "This policy covers...",
  "bullets": ["Point 1", "Point 2", ...],
  "sections": [...],
  "status": "summarized"
}
```

#### 5. QA Check / Hallucination Detection
```bash
POST /qa-check/{document_id}

curl -X POST "http://localhost:8000/qa-check/{document_id}"
```

**Response**:
```json
{
  "document_id": "uuid-here",
  "overview_validation": {...},
  "bullets_validation": {...},
  "overall_valid": true,
  "status": "validated"
}
```

#### 6. Download Summary
```bash
GET /download/{document_id}?format=pdf
GET /download/{document_id}?format=json

curl -X GET "http://localhost:8000/download/{document_id}?format=pdf" \
  -o summary.pdf
```

### Complete Workflow Example

```bash
# 1. Upload document
UPLOAD_RESPONSE=$(curl -X POST "http://localhost:8000/upload" \
  -F "file=@policy.pdf")
DOCUMENT_ID=$(echo $UPLOAD_RESPONSE | jq -r '.document_id')

# 2. Chunk document
curl -X POST "http://localhost:8000/chunk/$DOCUMENT_ID"

# 3. Generate embeddings
curl -X POST "http://localhost:8000/embed/$DOCUMENT_ID"

# 4. Generate summaries
curl -X POST "http://localhost:8000/summarize/$DOCUMENT_ID"

# 5. Perform QA check
curl -X POST "http://localhost:8000/qa-check/$DOCUMENT_ID"

# 6. Download PDF summary
curl -X GET "http://localhost:8000/download/$DOCUMENT_ID?format=pdf" \
  -o summary.pdf
```

## Testing

Run tests with pytest:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=backend --cov-report=html

# Run specific test file
pytest backend/tests/test_upload.py
```

## Docker

### Build Docker Image

```bash
docker build -t policy-summarizer .
```

### Run Docker Container

```bash
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=your_key \
  -e AWS_ACCESS_KEY_ID=your_key \
  -e AWS_SECRET_ACCESS_KEY=your_secret \
  -e AWS_BUCKET=your_bucket \
  policy-summarizer
```

### Docker Compose (Optional)

Create `docker-compose.yml`:

```yaml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
      - AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
      - AWS_BUCKET=${AWS_BUCKET}
    volumes:
      - ./data:/app/data
```

## Configuration

All configuration is managed through environment variables. Key settings:

- **Chunking**: `CHUNK_SIZE`, `CHUNK_OVERLAP`, `MIN_CHUNK_SIZE`, `MAX_CHUNK_SIZE`
- **Summarization**: `SUMMARY_TEMPERATURE`, `MAX_TOKENS`
- **QA Validation**: `SIMILARITY_THRESHOLD`
- **File Limits**: `MAX_FILE_SIZE_MB`

## Architecture

The system follows a modular architecture:

1. **API Layer** (`api/`): FastAPI routers handling HTTP requests
2. **Service Layer** (`services/`): Business logic and external integrations
3. **Core Layer** (`core/`): Configuration, utilities, and data models

### Data Flow

1. Document uploaded → Text extracted → Stored in S3
2. Text chunked → Chunks stored locally and in S3
3. Embeddings generated → Stored in FAISS/MongoDB
4. Summaries generated → Stored in S3
5. QA validation performed → Results stored in S3
6. Summary downloaded as PDF/JSON

## Error Handling

The API includes comprehensive error handling:
- File format validation
- File size limits
- Text extraction errors
- API rate limiting (handled by OpenAI SDK)
- Vector database errors

## Performance Considerations

- **Large Files**: Supports files up to 30MB
- **Batch Processing**: Embeddings generated in batches
- **Caching**: Chunks and summaries cached in S3
- **Async Operations**: FastAPI async endpoints for better performance

## Security

- Environment variables for sensitive credentials
- File type validation
- File size limits
- Input sanitization

## Limitations

- Maximum file size: 30MB
- Requires OpenAI API access
- Requires AWS S3 bucket
- FAISS index stored locally (consider MongoDB for distributed systems)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

See LICENSE file for details.

## Support

For issues and questions, please open an issue on GitHub.
