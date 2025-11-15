"""
Upload API endpoint for document ingestion.
"""
import os
import tempfile
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Optional

from backend.core.config import settings
from backend.core.utils import (
    generate_document_id,
    ensure_directory,
    validate_file_size,
    is_valid_document_format,
    format_timestamp,
)
from backend.core.schemas import UploadResponse, ErrorResponse
from backend.services.s3_service import s3_service
from backend.services.extractor import text_extractor


router = APIRouter(prefix="/upload", tags=["upload"])


@router.post("", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    """
    Upload a document (PDF or DOCX) and extract text.
    
    Args:
        file: Uploaded file (PDF or DOCX)
        
    Returns:
        UploadResponse with document_id and metadata
    """
    # Validate file format
    if not is_valid_document_format(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Only PDF and DOCX are supported."
        )
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
        try:
            # Save uploaded file
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
            
            # Validate file size
            if not validate_file_size(tmp_file_path, settings.max_file_size_mb):
                raise HTTPException(
                    status_code=400,
                    detail=f"File size exceeds maximum allowed size of {settings.max_file_size_mb}MB"
                )
            
            # Generate document ID
            document_id = generate_document_id()
            
            # Extract text
            try:
                extracted_text = text_extractor.extract_text(tmp_file_path)
                if not extracted_text or len(extracted_text.strip()) < 100:
                    raise HTTPException(
                        status_code=400,
                        detail="Failed to extract meaningful text from document"
                    )
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Error extracting text: {str(e)}"
                )
            
            # Extract metadata
            metadata = text_extractor.extract_metadata(tmp_file_path)
            metadata.update({
                "document_id": document_id,
                "upload_timestamp": format_timestamp(),
                "text_length": len(extracted_text),
            })
            
            # Upload to S3
            # Store original file
            file_s3_key = f"documents/{document_id}/{file.filename}"
            s3_service.upload_file(tmp_file_path, file_s3_key, metadata)
            
            # Store extracted text
            text_s3_key = f"documents/{document_id}/extracted_text.txt"
            s3_service.save_text_content(extracted_text, text_s3_key)
            
            # Store metadata
            metadata_s3_key = f"documents/{document_id}/metadata.json"
            import json
            s3_service.save_text_content(json.dumps(metadata, indent=2), metadata_s3_key)
            
            return UploadResponse(
                document_id=document_id,
                filename=file.filename,
                file_size=metadata["file_size"],
                file_type=metadata["file_extension"],
                status="uploaded",
                message="Document uploaded and text extracted successfully",
                metadata=metadata,
            )
        
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)

