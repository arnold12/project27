"""
Tests for upload API endpoint.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import tempfile
import os

from backend.main import app
from backend.core.utils import generate_document_id


client = TestClient(app)


@pytest.fixture
def mock_s3_service():
    """Mock S3 service."""
    with patch("backend.api.upload.s3_service") as mock:
        mock.upload_file.return_value = True
        mock.save_text_content.return_value = True
        yield mock


@pytest.fixture
def mock_extractor():
    """Mock text extractor."""
    with patch("backend.api.upload.text_extractor") as mock:
        mock.extract_text.return_value = "This is sample extracted text from a policy document. " * 50
        mock.extract_metadata.return_value = {
            "filename": "test.pdf",
            "file_size": 1024,
            "file_extension": "pdf",
            "page_count": 5,
        }
        yield mock


def test_upload_pdf_success(mock_s3_service, mock_extractor):
    """Test successful PDF upload."""
    # Create a temporary PDF file
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
        tmp_file.write(b"%PDF-1.4\n")
        tmp_file_path = tmp_file.name
    
    try:
        with open(tmp_file_path, "rb") as f:
            response = client.post(
                "/upload",
                files={"file": ("test.pdf", f, "application/pdf")}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert "document_id" in data
        assert data["filename"] == "test.pdf"
        assert data["status"] == "uploaded"
        assert data["file_type"] == "pdf"
    finally:
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)


def test_upload_invalid_format():
    """Test upload with invalid file format."""
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp_file:
        tmp_file.write(b"Some text content")
        tmp_file_path = tmp_file.name
    
    try:
        with open(tmp_file_path, "rb") as f:
            response = client.post(
                "/upload",
                files={"file": ("test.txt", f, "text/plain")}
            )
        
        assert response.status_code == 400
        assert "Unsupported file format" in response.json()["detail"]
    finally:
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)


def test_upload_large_file(mock_s3_service, mock_extractor):
    """Test upload with file exceeding size limit."""
    # Create a large temporary file
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
        # Write 35MB of data
        tmp_file.write(b"0" * (35 * 1024 * 1024))
        tmp_file_path = tmp_file.name
    
    try:
        with open(tmp_file_path, "rb") as f:
            response = client.post(
                "/upload",
                files={"file": ("large.pdf", f, "application/pdf")}
            )
        
        assert response.status_code == 400
        assert "exceeds maximum" in response.json()["detail"].lower()
    finally:
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)


def test_root_endpoint():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data


def test_health_check():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

