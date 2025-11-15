"""
QA validation service for hallucination detection.
"""
from typing import Dict, Any, List
import numpy as np

from backend.core.config import settings
from backend.services.embedder import embedder_service
from backend.services.vector_store import vector_store


class QAValidatorService:
    """Service for validating summaries and detecting hallucinations."""
    
    def __init__(self):
        """Initialize the QA validator."""
        self.similarity_threshold = settings.similarity_threshold
    
    def validate_summary(
        self,
        document_id: str,
        summary_text: str,
        summary_type: str = "overview"
    ) -> Dict[str, Any]:
        """
        Validate a summary against the original document chunks.
        
        Args:
            document_id: Document ID
            summary_text: Summary text to validate
            summary_type: Type of summary (overview, bullets, section)
            
        Returns:
            Validation report with hallucination assessment
        """
        # Generate embedding for summary
        summary_embedding = embedder_service.generate_embedding(summary_text)
        if not summary_embedding:
            return {
                "valid": False,
                "error": "Failed to generate summary embedding",
                "hallucinations": [],
                "overall_similarity": 0.0,
            }
        
        # Search for similar chunks
        similar_chunks = vector_store.search_similar(
            summary_embedding,
            top_k=10,
            document_id=document_id
        )
        
        if not similar_chunks:
            return {
                "valid": False,
                "error": "No matching chunks found in document",
                "hallucinations": [],
                "overall_similarity": 0.0,
            }
        
        # Calculate average similarity
        similarities = [chunk["similarity"] for chunk in similar_chunks]
        avg_similarity = np.mean(similarities) if similarities else 0.0
        
        # Detect potential hallucinations
        hallucinations = []
        if avg_similarity < self.similarity_threshold:
            hallucinations.append({
                "type": "low_similarity",
                "severity": "high" if avg_similarity < 0.5 else "medium",
                "description": f"Summary has low similarity ({avg_similarity:.2f}) to original document",
                "threshold": self.similarity_threshold,
                "actual_similarity": avg_similarity,
            })
        
        # Check individual sentence similarity (simplified)
        # Split summary into sentences and check each
        sentences = self._split_into_sentences(summary_text)
        sentence_issues = []
        
        for sentence in sentences[:10]:  # Limit to first 10 sentences
            if len(sentence.strip()) < 20:  # Skip very short sentences
                continue
            
            sent_embedding = embedder_service.generate_embedding(sentence)
            if sent_embedding:
                sent_similar = vector_store.search_similar(
                    sent_embedding,
                    top_k=3,
                    document_id=document_id
                )
                
                if sent_similar:
                    max_sim = max([chunk["similarity"] for chunk in sent_similar])
                    if max_sim < self.similarity_threshold:
                        sentence_issues.append({
                            "sentence": sentence[:100],  # Truncate for display
                            "similarity": max_sim,
                            "threshold": self.similarity_threshold,
                        })
        
        if sentence_issues:
            hallucinations.append({
                "type": "sentence_level",
                "severity": "medium",
                "description": f"Found {len(sentence_issues)} sentences with low similarity",
                "issues": sentence_issues[:5],  # Limit to 5 examples
            })
        
        # Overall assessment
        is_valid = avg_similarity >= self.similarity_threshold and len(hallucinations) == 0
        
        return {
            "valid": is_valid,
            "overall_similarity": float(avg_similarity),
            "threshold": self.similarity_threshold,
            "hallucinations": hallucinations,
            "similar_chunks_count": len(similar_chunks),
            "top_similar_chunks": [
                {
                    "chunk_id": chunk["chunk_id"],
                    "similarity": chunk["similarity"],
                    "text_preview": chunk["text"][:200],
                }
                for chunk in similar_chunks[:3]
            ],
            "summary_type": summary_type,
        }
    
    def validate_all_summaries(
        self,
        document_id: str,
        summaries: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate all types of summaries for a document.
        
        Args:
            document_id: Document ID
            summaries: Dictionary with all summary types
            
        Returns:
            Comprehensive validation report
        """
        results = {
            "document_id": document_id,
            "overview_validation": None,
            "bullets_validation": None,
            "sections_validation": None,
            "overall_valid": True,
        }
        
        # Validate overview
        if "overview" in summaries:
            overview_text = summaries["overview"]
            results["overview_validation"] = self.validate_summary(
                document_id,
                overview_text,
                "overview"
            )
            if not results["overview_validation"]["valid"]:
                results["overall_valid"] = False
        
        # Validate bullets (combine into single text)
        if "bullets" in summaries:
            bullets_text = "\n".join(summaries["bullets"])
            results["bullets_validation"] = self.validate_summary(
                document_id,
                bullets_text,
                "bullets"
            )
            if not results["bullets_validation"]["valid"]:
                results["overall_valid"] = False
        
        # Validate sections (sample a few)
        if "sections" in summaries:
            section_texts = [s["summary"] for s in summaries["sections"][:5]]  # Sample first 5
            combined_sections = "\n".join(section_texts)
            results["sections_validation"] = self.validate_summary(
                document_id,
                combined_sections,
                "sections"
            )
            if not results["sections_validation"]["valid"]:
                results["overall_valid"] = False
        
        return results
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Simple sentence splitting.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        import re
        # Simple sentence splitting by periods, exclamation, question marks
        sentences = re.split(r'[.!?]+\s+', text)
        return [s.strip() for s in sentences if s.strip()]


# Global QA validator instance
qa_validator_service = QAValidatorService()

