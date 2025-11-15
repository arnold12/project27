"""
Download API endpoint for summary PDF and JSON.
"""
import json
import os
import tempfile
from fastapi import APIRouter, HTTPException, Path, Query
from fastapi.responses import FileResponse, JSONResponse
from typing import Optional, Union

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_LEFT, TA_CENTER

from backend.core.schemas import ErrorResponse
from backend.services.s3_service import s3_service


router = APIRouter(prefix="/download", tags=["download"])


@router.get("/{document_id}")
async def download_summary(
    document_id: str = Path(..., description="Document ID"),
    format: str = Query("pdf", description="Download format: pdf or json")
) -> Union[FileResponse, JSONResponse]:
    """
    Download summary in PDF or JSON format.
    
    Args:
        document_id: Document ID from upload
        format: Download format (pdf or json)
        
    Returns:
        FileResponse for PDF or JSONResponse for JSON
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
    
    # Fetch metadata
    metadata_s3_key = f"documents/{document_id}/metadata.json"
    metadata_json = s3_service.get_text_content(metadata_s3_key)
    metadata = json.loads(metadata_json) if metadata_json else {}
    
    if format.lower() == "json":
        # Return JSON response
        return JSONResponse(
            content={
                "document_id": document_id,
                "metadata": metadata,
                **summaries_data
            }
        )
    
    elif format.lower() == "pdf":
        # Generate PDF
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                pdf_path = tmp_file.name
            
            # Create PDF
            doc = SimpleDocTemplate(pdf_path, pagesize=letter)
            story = []
            styles = getSampleStyleSheet()
            
            # Custom styles
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=18,
                textColor='#1a1a1a',
                spaceAfter=30,
                alignment=TA_CENTER
            )
            
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontSize=14,
                textColor='#2c3e50',
                spaceAfter=12,
                spaceBefore=20
            )
            
            # Title
            title = f"Policy Document Summary"
            story.append(Paragraph(title, title_style))
            story.append(Spacer(1, 0.2*inch))
            
            # Document info
            if metadata:
                info_text = f"<b>Document:</b> {metadata.get('filename', 'N/A')}<br/>"
                info_text += f"<b>Document ID:</b> {document_id}<br/>"
                if metadata.get('page_count'):
                    info_text += f"<b>Pages:</b> {metadata.get('page_count')}<br/>"
                story.append(Paragraph(info_text, styles['Normal']))
                story.append(Spacer(1, 0.3*inch))
            
            # Overview
            story.append(Paragraph("Overview", heading_style))
            overview_text = summaries_data.get("overview", "No overview available.")
            story.append(Paragraph(overview_text.replace("\n", "<br/>"), styles['Normal']))
            story.append(Spacer(1, 0.3*inch))
            
            # Bullet Points
            story.append(Paragraph("Key Points", heading_style))
            bullets = summaries_data.get("bullets", [])
            if bullets:
                bullet_text = "<br/>".join([f"• {bullet}" for bullet in bullets])
                story.append(Paragraph(bullet_text, styles['Normal']))
            else:
                story.append(Paragraph("No bullet points available.", styles['Normal']))
            story.append(Spacer(1, 0.3*inch))
            
            # Section Summaries
            sections = summaries_data.get("sections", [])
            if sections:
                story.append(Paragraph("Section Summaries", heading_style))
                for idx, section in enumerate(sections[:20], 1):  # Limit to first 20 sections
                    section_title = f"Section {idx}"
                    section_summary = section.get("summary", "")
                    story.append(Paragraph(f"<b>{section_title}</b>", styles['Normal']))
                    story.append(Paragraph(section_summary.replace("\n", "<br/>"), styles['Normal']))
                    story.append(Spacer(1, 0.2*inch))
            
            # Build PDF
            doc.build(story)
            
            return FileResponse(
                pdf_path,
                media_type="application/pdf",
                filename=f"summary_{document_id}.pdf",
                background=None
            )
        
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error generating PDF: {str(e)}"
            )
    
    else:
        raise HTTPException(
            status_code=400,
            detail="Invalid format. Use 'pdf' or 'json'"
        )

