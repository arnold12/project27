"""
Embedding generation API endpoint.
"""
import json
from fastapi import APIRouter, HTTPException, Path

from backend.core.config import settings
from backend.core.schemas import EmbedResponse, ErrorResponse
from backend.services.s3_service import s3_service
from backend.services.embedder import embedder_service
from backend.services.vector_store import vector_store


router = APIRouter(prefix="/embed", tags=["embedding"])


@router.post("/{document_id}", response_model=EmbedResponse)
async def generate_embeddings(document_id: str = Path(..., description="Document ID")) -> EmbedResponse:
    """
    Generate embeddings for document chunks and store in vector database.
    
    Args:
        document_id: Document ID from upload
        
    Returns:
        EmbedResponse with embedding information
    """
    # Fetch chunks from S3
    chunks_s3_key = f"documents/{document_id}/chunks.json"
    chunks_json = s3_service.get_text_content(chunks_s3_key)
    
    if not chunks_json:
        raise HTTPException(
            status_code=404,
            detail=f"Chunks not found for document {document_id}. Please chunk the document first."
        )
    
    chunks_data = json.loads(chunks_json)
    chunks = chunks_data.get("chunks", [])
    
    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="No chunks found in document"
        )
    
    # Generate embeddings
    try:
        # Extract texts from chunks
        texts = [chunk["text"] for chunk in chunks]
        
        # Generate embeddings in batch
        embeddings = embedder_service.generate_embeddings_batch(texts)
        
        # Filter out None embeddings
        valid_embeddings = []
        valid_chunks = []
        for chunk, embedding in zip(chunks, embeddings):
            if embedding:
                valid_embeddings.append(embedding)
                valid_chunks.append(chunk)
        
        if not valid_embeddings:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate embeddings"
            )
        
        # Store in vector database
        success = vector_store.add_embeddings(
            document_id,
            valid_chunks,
            valid_embeddings
        )
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to store embeddings in vector database"
            )
        
        return EmbedResponse(
            document_id=document_id,
            total_embeddings=len(valid_embeddings),
            embedding_dimension=embedder_service.get_embedding_dimension(),
            vector_store=settings.vector_db,
            status="embedded",
            message=f"Generated and stored {len(valid_embeddings)} embeddings",
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating embeddings: {str(e)}"
        )

