"""
Summarization service using OpenAI GPT models.
"""
from typing import Dict, Any, List
from openai import OpenAI

from backend.core.config import settings


class SummarizerService:
    """Service for generating summaries using OpenAI GPT."""
    
    def __init__(self):
        """Initialize the summarizer with OpenAI client."""
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model
        self.temperature = settings.summary_temperature
        self.max_tokens = settings.max_tokens
    
    def summarize_document(
        self,
        chunks: List[Dict[str, Any]],
        document_metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive summaries for a document.
        
        Args:
            chunks: List of chunk dictionaries with text
            document_metadata: Optional document metadata
            
        Returns:
            Dictionary with different types of summaries
        """
        # Combine all chunk texts
        full_text = "\n\n".join([chunk["text"] for chunk in chunks])
        
        # Generate different types of summaries
        overview_summary = self._generate_overview_summary(full_text)
        bullet_summary = self._generate_bullet_summary(full_text)
        section_summaries = self._generate_section_summaries(chunks)
        
        return {
            "overview": overview_summary,
            "bullets": bullet_summary,
            "sections": section_summaries,
            "metadata": document_metadata or {},
        }
    
    def _generate_overview_summary(self, text: str) -> str:
        """
        Generate high-level overview summary (2-4 paragraphs).
        
        Args:
            text: Full document text
            
        Returns:
            Overview summary
        """
        prompt = f"""You are a helpful assistant that summarizes insurance policy documents in simple, clear language (grade 5-6 reading level).

Please provide a comprehensive overview summary of the following insurance policy document in 2-4 paragraphs. Focus on:
- What type of insurance policy this is
- Main coverage areas
- Key terms and conditions
- Important exclusions or limitations

Use simple, everyday language that anyone can understand.

Document text:
{text[:15000]}  # Limit to avoid token limits

Overview Summary:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that creates clear, simple summaries of insurance policies."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating overview summary: {e}")
            return "Error generating overview summary."
    
    def _generate_bullet_summary(self, text: str) -> List[str]:
        """
        Generate bullet-point summary (5-10 bullets).
        
        Args:
            text: Full document text
            
        Returns:
            List of bullet points
        """
        prompt = f"""You are a helpful assistant that summarizes insurance policy documents in simple, clear language (grade 5-6 reading level).

Please provide a bullet-point summary of the following insurance policy document with 5-10 key points. Each bullet should be:
- Simple and easy to understand
- Focused on important information
- Written in plain language

Document text:
{text[:15000]}  # Limit to avoid token limits

Bullet Summary (one bullet per line):"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that creates clear, simple bullet-point summaries of insurance policies."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=1000,
            )
            
            # Parse bullets from response
            content = response.choices[0].message.content.strip()
            bullets = [line.strip() for line in content.split("\n") if line.strip() and (line.strip().startswith("-") or line.strip().startswith("•") or line.strip()[0].isdigit())]
            
            # Clean up bullets
            cleaned_bullets = []
            for bullet in bullets:
                # Remove bullet markers
                cleaned = bullet.lstrip("-•*0123456789. ")
                if cleaned:
                    cleaned_bullets.append(cleaned)
            
            return cleaned_bullets[:10]  # Limit to 10 bullets
        except Exception as e:
            print(f"Error generating bullet summary: {e}")
            return ["Error generating bullet summary."]
    
    def _generate_section_summaries(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate section-level (chunk-wise) summaries.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            List of section summaries with metadata
        """
        section_summaries = []
        
        # Process chunks in batches to avoid token limits
        for chunk in chunks:
            chunk_text = chunk["text"]
            chunk_id = chunk.get("chunk_id", "")
            
            # Skip very short chunks
            if len(chunk_text) < 100:
                continue
            
            summary = self._summarize_chunk(chunk_text)
            
            section_summaries.append({
                "chunk_id": chunk_id,
                "chunk_index": chunk.get("chunk_index", 0),
                "summary": summary,
                "text_length": chunk.get("text_length", len(chunk_text)),
            })
        
        return section_summaries
    
    def _summarize_chunk(self, chunk_text: str) -> str:
        """
        Summarize a single chunk.
        
        Args:
            chunk_text: Text of the chunk
            
        Returns:
            Summary of the chunk
        """
        prompt = f"""You are a helpful assistant that summarizes sections of insurance policy documents in simple, clear language (grade 5-6 reading level).

Please provide a brief summary of the following section in 1-2 sentences. Use simple, everyday language.

Section text:
{chunk_text[:3000]}  # Limit chunk size

Summary:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that creates clear, simple summaries of insurance policy sections."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=200,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error summarizing chunk: {e}")
            return "Error generating section summary."


# Global summarizer instance
summarizer_service = SummarizerService()

