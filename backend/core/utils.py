"""
Utility functions for the Policy Document Summarization Assistant.
"""
import os
import json
import hashlib
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime


def ensure_directory(path: str) -> None:
    """Ensure a directory exists, creating it if necessary."""
    Path(path).mkdir(parents=True, exist_ok=True)


def generate_document_id() -> str:
    """Generate a unique document ID."""
    return str(uuid.uuid4())


def generate_chunk_id(document_id: str, chunk_index: int) -> str:
    """Generate a unique chunk ID."""
    return f"{document_id}_chunk_{chunk_index}"


def calculate_file_hash(file_path: str) -> str:
    """Calculate SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def save_json(data: Dict[str, Any], file_path: str) -> None:
    """Save data to a JSON file."""
    ensure_directory(os.path.dirname(file_path))
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(file_path: str) -> Dict[str, Any]:
    """Load data from a JSON file."""
    if not os.path.exists(file_path):
        return {}
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def clean_text(text: str) -> str:
    """
    Clean extracted text by removing excessive whitespace, headers, footers.
    
    Args:
        text: Raw extracted text
        
    Returns:
        Cleaned text
    """
    # Remove excessive whitespace
    lines = text.split("\n")
    cleaned_lines = []
    
    for line in lines:
        # Remove lines that are mostly whitespace
        if line.strip():
            # Normalize whitespace within line
            cleaned_line = " ".join(line.split())
            cleaned_lines.append(cleaned_line)
    
    # Join lines with single newline
    cleaned_text = "\n".join(cleaned_lines)
    
    # Remove multiple consecutive newlines (more than 2)
    while "\n\n\n" in cleaned_text:
        cleaned_text = cleaned_text.replace("\n\n\n", "\n\n")
    
    # Remove leading/trailing whitespace
    cleaned_text = cleaned_text.strip()
    
    return cleaned_text


def format_timestamp() -> str:
    """Get current timestamp in ISO format."""
    return datetime.utcnow().isoformat() + "Z"


def estimate_tokens(text: str) -> int:
    """
    Rough estimation of token count (approximate: 1 token ≈ 4 characters).
    
    Args:
        text: Input text
        
    Returns:
        Estimated token count
    """
    return len(text) // 4


def validate_file_size(file_path: str, max_size_mb: int) -> bool:
    """Validate that file size is within limits."""
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    return file_size_mb <= max_size_mb


def get_file_extension(filename: str) -> str:
    """Get file extension in lowercase."""
    return os.path.splitext(filename)[1].lower().lstrip(".")


def is_valid_document_format(filename: str) -> bool:
    """Check if file format is supported (PDF or DOCX)."""
    ext = get_file_extension(filename)
    return ext in ["pdf", "docx"]

