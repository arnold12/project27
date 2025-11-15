"""
Pydantic models for request/response schemas.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


# Upload API Schemas
class UploadResponse(BaseModel):
    """Response schema for document upload."""
    document_id: str
    filename: str
    file_size: int
    file_type: str
    status: str = "uploaded"
    message: str = "Document uploaded successfully"
    metadata: Optional[Dict[str, Any]] = None


# Chunking API Schemas
class ChunkMetadata(BaseModel):
    """Schema for chunk metadata."""
    chunk_id: str
    chunk_index: int
    text_length: int
    token_count: int


class ChunkResponse(BaseModel):
    """Response schema for chunking."""
    document_id: str
    total_chunks: int
    chunks: List[ChunkMetadata]
    status: str = "chunked"
    message: str = "Document chunked successfully"


# Embedding API Schemas
class EmbedResponse(BaseModel):
    """Response schema for embedding generation."""
    document_id: str
    total_embeddings: int
    embedding_dimension: int
    vector_store: str
    status: str = "embedded"
    message: str = "Embeddings generated and stored successfully"


# Summarization API Schemas
class SectionSummary(BaseModel):
    """Schema for section-level summary."""
    chunk_id: str
    chunk_index: int
    summary: str
    text_length: int


class SummaryResponse(BaseModel):
    """Response schema for summarization."""
    document_id: str
    overview: str
    bullets: List[str]
    sections: List[SectionSummary]
    metadata: Optional[Dict[str, Any]] = None
    status: str = "summarized"
    message: str = "Document summarized successfully"


# QA Check API Schemas
class HallucinationIssue(BaseModel):
    """Schema for hallucination issue."""
    type: str
    severity: str
    description: str
    threshold: Optional[float] = None
    actual_similarity: Optional[float] = None
    issues: Optional[List[Dict[str, Any]]] = None


class SimilarChunk(BaseModel):
    """Schema for similar chunk reference."""
    chunk_id: str
    similarity: float
    text_preview: str


class ValidationResult(BaseModel):
    """Schema for validation result."""
    valid: bool
    overall_similarity: float
    threshold: float
    hallucinations: List[HallucinationIssue]
    similar_chunks_count: int
    top_similar_chunks: List[SimilarChunk]
    summary_type: str
    error: Optional[str] = None


class QACheckResponse(BaseModel):
    """Response schema for QA check."""
    document_id: str
    overview_validation: Optional[ValidationResult] = None
    bullets_validation: Optional[ValidationResult] = None
    sections_validation: Optional[ValidationResult] = None
    overall_valid: bool
    status: str = "validated"
    message: str = "QA check completed"


# Download API Schemas
class DownloadResponse(BaseModel):
    """Response schema for download."""
    document_id: str
    download_url: Optional[str] = None
    file_path: Optional[str] = None
    format: str
    status: str = "ready"
    message: str = "File ready for download"


# Error Schemas
class ErrorResponse(BaseModel):
    """Schema for error responses."""
    error: str
    detail: Optional[str] = None
    status_code: int

