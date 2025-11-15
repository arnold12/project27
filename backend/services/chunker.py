"""
Text chunking service using LangChain.
"""
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from backend.core.config import settings
from backend.core.utils import generate_chunk_id, estimate_tokens


class ChunkerService:
    """Service for chunking text into manageable segments."""
    
    def __init__(self):
        """Initialize the chunker with configuration."""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
    
    def chunk_text(
        self,
        text: str,
        document_id: str,
        metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Chunk text into segments with metadata.
        
        Args:
            text: Text to chunk
            document_id: Document ID
            metadata: Optional metadata to attach to chunks
            
        Returns:
            List of chunk dictionaries with metadata
        """
        # Create LangChain documents
        doc_metadata = metadata or {}
        documents = [Document(page_content=text, metadata=doc_metadata)]
        
        # Split documents
        chunks = self.text_splitter.split_documents(documents)
        
        # Filter chunks by size and create metadata
        chunk_list = []
        for idx, chunk in enumerate(chunks):
            chunk_text = chunk.page_content
            text_length = len(chunk_text)
            token_count = estimate_tokens(chunk_text)
            
            # Skip chunks that are too small or too large
            if token_count < settings.min_chunk_size:
                continue
            if token_count > settings.max_chunk_size:
                # Further split oversized chunks
                sub_chunks = self._split_oversized_chunk(chunk_text, document_id, idx)
                chunk_list.extend(sub_chunks)
                continue
            
            chunk_id = generate_chunk_id(document_id, idx)
            chunk_metadata = {
                "chunk_id": chunk_id,
                "document_id": document_id,
                "chunk_index": idx,
                "text": chunk_text,
                "text_length": text_length,
                "token_count": token_count,
                "metadata": {**chunk.metadata, **doc_metadata},
            }
            chunk_list.append(chunk_metadata)
        
        return chunk_list
    
    def _split_oversized_chunk(
        self,
        text: str,
        document_id: str,
        base_index: int
    ) -> List[Dict[str, Any]]:
        """
        Split an oversized chunk into smaller chunks.
        
        Args:
            text: Text to split
            document_id: Document ID
            base_index: Base index for chunk numbering
            
        Returns:
            List of chunk dictionaries
        """
        # Use a smaller chunk size for splitting
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.max_chunk_size - 100,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
        )
        
        sub_chunks = splitter.split_text(text)
        chunk_list = []
        
        for sub_idx, chunk_text in enumerate(sub_chunks):
            text_length = len(chunk_text)
            token_count = estimate_tokens(chunk_text)
            
            if token_count < settings.min_chunk_size:
                continue
            
            chunk_id = generate_chunk_id(document_id, f"{base_index}_{sub_idx}")
            chunk_metadata = {
                "chunk_id": chunk_id,
                "document_id": document_id,
                "chunk_index": f"{base_index}_{sub_idx}",
                "text": chunk_text,
                "text_length": text_length,
                "token_count": token_count,
                "metadata": {},
            }
            chunk_list.append(chunk_metadata)
        
        return chunk_list


# Global chunker instance
chunker_service = ChunkerService()

