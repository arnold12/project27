"""
Tests for summarization functionality.
"""
import pytest
from unittest.mock import patch, MagicMock
import json

from backend.services.summarizer import SummarizerService
from backend.core.config import settings


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client."""
    with patch("backend.services.summarizer.OpenAI") as mock:
        client_instance = MagicMock()
        mock.return_value = client_instance
        
        # Mock chat completion response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "This is a test summary."
        client_instance.chat.completions.create.return_value = mock_response
        
        yield client_instance


@pytest.fixture
def sample_chunks():
    """Sample chunks for testing."""
    return [
        {
            "chunk_id": "doc1_chunk_0",
            "chunk_index": 0,
            "text": "This is the first chunk of text from an insurance policy document. " * 20,
            "text_length": 1000,
            "token_count": 250,
        },
        {
            "chunk_id": "doc1_chunk_1",
            "chunk_index": 1,
            "text": "This is the second chunk of text from an insurance policy document. " * 20,
            "text_length": 1000,
            "token_count": 250,
        },
    ]


def test_summarizer_initialization():
    """Test summarizer service initialization."""
    with patch("backend.services.summarizer.OpenAI"):
        service = SummarizerService()
        assert service.model == settings.openai_model
        assert service.temperature == settings.summary_temperature


def test_generate_overview_summary(mock_openai_client):
    """Test overview summary generation."""
    service = SummarizerService()
    service.client = mock_openai_client
    
    text = "Sample policy text. " * 100
    summary = service._generate_overview_summary(text)
    
    assert summary == "This is a test summary."
    mock_openai_client.chat.completions.create.assert_called_once()


def test_generate_bullet_summary(mock_openai_client):
    """Test bullet summary generation."""
    service = SummarizerService()
    service.client = mock_openai_client
    
    # Mock response with bullets
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "- Point 1\n- Point 2\n- Point 3"
    mock_openai_client.chat.completions.create.return_value = mock_response
    
    text = "Sample policy text. " * 100
    bullets = service._generate_bullet_summary(text)
    
    assert isinstance(bullets, list)
    assert len(bullets) > 0


def test_summarize_document(mock_openai_client, sample_chunks):
    """Test full document summarization."""
    service = SummarizerService()
    service.client = mock_openai_client
    
    # Mock different responses for different summary types
    def mock_create(*args, **kwargs):
        response = MagicMock()
        response.choices = [MagicMock()]
        content = kwargs.get("messages", [{}])[1].get("content", "")
        if "overview" in content.lower() or "comprehensive" in content.lower():
            response.choices[0].message.content = "This is an overview summary."
        elif "bullet" in content.lower():
            response.choices[0].message.content = "- Point 1\n- Point 2"
        else:
            response.choices[0].message.content = "This is a section summary."
        return response
    
    mock_openai_client.chat.completions.create.side_effect = mock_create
    
    result = service.summarize_document(sample_chunks)
    
    assert "overview" in result
    assert "bullets" in result
    assert "sections" in result
    assert isinstance(result["bullets"], list)
    assert isinstance(result["sections"], list)

