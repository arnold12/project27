"""
QA check API endpoint for hallucination detection.
"""
import json
from fastapi import APIRouter, HTTPException, Path

from backend.core.schemas import QACheckResponse, ValidationResult, HallucinationIssue, SimilarChunk, ErrorResponse
from backend.services.s3_service import s3_service
from backend.services.qa_validator import qa_validator_service


router = APIRouter(prefix="/qa-check", tags=["qa"])


@router.post("/{document_id}", response_model=QACheckResponse)
async def qa_check_document(document_id: str = Path(..., description="Document ID")) -> QACheckResponse:
    """
    Perform QA check and hallucination detection on document summaries.
    
    Args:
        document_id: Document ID from upload
        
    Returns:
        QACheckResponse with validation results
    """
    # Fetch summaries from S3
    summaries_s3_key = f"documents/{document_id}/summaries.json"
    summaries_json = s3_service.get_text_content(summaries_s3_key)
    
    if not summaries_json:
        raise HTTPException(
            status_code=404,
            detail=f"Summaries not found for document {document_id}. Please summarize the document first."
        )
    
    summaries_data = json.loads(summaries_json)
    
    # Perform validation
    try:
        validation_results = qa_validator_service.validate_all_summaries(
            document_id,
            summaries_data
        )
        
        # Format response
        def format_validation_result(result: dict) -> ValidationResult:
            if not result:
                return None
            
            hallucinations = [
                HallucinationIssue(
                    type=h.get("type", "unknown"),
                    severity=h.get("severity", "medium"),
                    description=h.get("description", ""),
                    threshold=h.get("threshold"),
                    actual_similarity=h.get("actual_similarity"),
                    issues=h.get("issues"),
                )
                for h in result.get("hallucinations", [])
            ]
            
            similar_chunks = [
                SimilarChunk(
                    chunk_id=chunk["chunk_id"],
                    similarity=chunk["similarity"],
                    text_preview=chunk["text_preview"],
                )
                for chunk in result.get("top_similar_chunks", [])
            ]
            
            return ValidationResult(
                valid=result.get("valid", False),
                overall_similarity=result.get("overall_similarity", 0.0),
                threshold=result.get("threshold", 0.7),
                hallucinations=hallucinations,
                similar_chunks_count=result.get("similar_chunks_count", 0),
                top_similar_chunks=similar_chunks,
                summary_type=result.get("summary_type", "unknown"),
                error=result.get("error"),
            )
        
        overview_validation = format_validation_result(validation_results.get("overview_validation"))
        bullets_validation = format_validation_result(validation_results.get("bullets_validation"))
        sections_validation = format_validation_result(validation_results.get("sections_validation"))
        
        # Store validation results in S3
        validation_s3_key = f"documents/{document_id}/qa_validation.json"
        s3_service.save_text_content(json.dumps(validation_results, indent=2, default=str), validation_s3_key)
        
        return QACheckResponse(
            document_id=document_id,
            overview_validation=overview_validation,
            bullets_validation=bullets_validation,
            sections_validation=sections_validation,
            overall_valid=validation_results.get("overall_valid", False),
            status="validated",
            message="QA check completed successfully",
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error performing QA check: {str(e)}"
        )

