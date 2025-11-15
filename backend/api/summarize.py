"""
Summarization API endpoint.
"""
import json
from fastapi import APIRouter, HTTPException, Path

from backend.core.schemas import SummaryResponse, SectionSummary, ErrorResponse
from backend.services.s3_service import s3_service
from backend.services.summarizer import summarizer_service


router = APIRouter(prefix="/summarize", tags=["summarization"])


@router.post("/{document_id}", response_model=SummaryResponse)
async def summarize_document(document_id: str = Path(..., description="Document ID")) -> SummaryResponse:
    """
    Generate summaries for a document.
    
    Args:
        document_id: Document ID from upload
        
    Returns:
        SummaryResponse with different types of summaries
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
    
    # Fetch metadata
    metadata_s3_key = f"documents/{document_id}/metadata.json"
    metadata_json = s3_service.get_text_content(metadata_s3_key)
    metadata = json.loads(metadata_json) if metadata_json else {}
    
    # Generate summaries
    try:
        summaries = summarizer_service.summarize_document(chunks, metadata)
        
        # Format section summaries
        section_summaries = [
            SectionSummary(
                chunk_id=section["chunk_id"],
                chunk_index=section.get("chunk_index", idx),
                summary=section["summary"],
                text_length=section.get("text_length", 0),
            )
            for idx, section in enumerate(summaries.get("sections", []))
        ]
        
        # Store summaries in S3
        summaries_s3_key = f"documents/{document_id}/summaries.json"
        summaries_data = {
            "document_id": document_id,
            "overview": summaries["overview"],
            "bullets": summaries["bullets"],
            "sections": [s.dict() for s in section_summaries],
            "metadata": summaries.get("metadata", {}),
        }
        s3_service.save_text_content(json.dumps(summaries_data, indent=2), summaries_s3_key)
        
        return SummaryResponse(
            document_id=document_id,
            overview=summaries["overview"],
            bullets=summaries["bullets"],
            sections=section_summaries,
            metadata=summaries.get("metadata"),
            status="summarized",
            message="Document summarized successfully",
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating summaries: {str(e)}"
        )

