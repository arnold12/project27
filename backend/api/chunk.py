"""
Chunking API endpoint.
"""
import json
from fastapi import APIRouter, HTTPException, Path
from typing import Optional

from backend.core.config import settings
from backend.core.utils import ensure_directory, save_json
from backend.core.schemas import ChunkResponse, ChunkMetadata, ErrorResponse
from backend.services.s3_service import s3_service
from backend.services.chunker import chunker_service


router = APIRouter(prefix="/chunk", tags=["chunk"])


@router.post("/{document_id}", response_model=ChunkResponse)
async def chunk_document(document_id: str = Path(..., description="Document ID")) -> ChunkResponse:
    """
    Chunk a document's extracted text into segments.
    
    Args:
        document_id: Document ID from upload
        
    Returns:
        ChunkResponse with chunk metadata
    """
    # Fetch extracted text from S3
    text_s3_key = f"documents/{document_id}/extracted_text.txt"
    extracted_text = s3_service.get_text_content(text_s3_key)
    
    if not extracted_text:
        raise HTTPException(
            status_code=404,
            detail=f"Document {document_id} not found or text not extracted"
        )
    
    # Fetch metadata
    metadata_s3_key = f"documents/{document_id}/metadata.json"
    metadata_json = s3_service.get_text_content(metadata_s3_key)
    metadata = json.loads(metadata_json) if metadata_json else {}
    
    # Chunk the text
    try:
        chunks = chunker_service.chunk_text(
            extracted_text,
            document_id,
            metadata
        )
        
        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="No valid chunks generated from document"
            )
        
        # Store chunks locally
        ensure_directory(settings.chunks_storage_path)
        chunks_file = f"{settings.chunks_storage_path}/{document_id}_chunks.json"
        save_json({"chunks": chunks, "document_id": document_id}, chunks_file)
        
        # Store chunks in S3 as well
        chunks_s3_key = f"documents/{document_id}/chunks.json"
        s3_service.save_text_content(json.dumps({"chunks": chunks, "document_id": document_id}, indent=2), chunks_s3_key)
        
        # Format response
        chunk_metadata = [
            ChunkMetadata(
                chunk_id=chunk["chunk_id"],
                chunk_index=chunk.get("chunk_index", idx),
                text_length=chunk["text_length"],
                token_count=chunk["token_count"],
            )
            for idx, chunk in enumerate(chunks)
        ]
        
        return ChunkResponse(
            document_id=document_id,
            total_chunks=len(chunks),
            chunks=chunk_metadata,
            status="chunked",
            message=f"Document chunked into {len(chunks)} segments",
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error chunking document: {str(e)}"
        )

