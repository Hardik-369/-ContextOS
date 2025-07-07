"""
Smart compactors for context optimization
"""

import os
import together
from typing import Optional
from .models import CompactStrategy


class Compactor:
    """Base class for context compactors"""
    
    def __init__(self, api_key=None):
        api_key = api_key or os.getenv("TOGETHER_API_KEY")
        self.together_client = together.Together(api_key=api_key) if api_key else None
    
    def compact(self, text: str, strategy: CompactStrategy, max_tokens: int = 2000) -> str:
        """Compact text using specified strategy"""
        if strategy == CompactStrategy.TRUNCATE:
            return self._truncate(text, max_tokens)
        elif strategy == CompactStrategy.LLM_SUMMARY:
            return self._llm_summary(text, max_tokens)
        else:
            raise ValueError(f"Unknown compact strategy: {strategy}")
    
    def _truncate(self, text: str, max_tokens: int) -> str:
        """Simple truncation by approximate token count"""
        # Rough approximation: 1 token ≈ 4 characters
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "..."
    
    def _llm_summary(self, text: str, max_tokens: int) -> str:
        """LLM-powered summarization via Together.ai"""
        if not self.together_client:
            print("Warning: No API key provided, falling back to truncation")
            return self._truncate(text, max_tokens)
        
        # If text is already short enough, return as-is
        if len(text) <= max_tokens * 4:
            return text
        
        prompt = f"""Summarize the following while preserving key entities, important details, and main concepts. Keep the summary concise but comprehensive:

{text}

Summary:"""
        
        try:
            response = self.together_client.chat.completions.create(
                model="mistralai/Mixtral-8x7B-Instruct-v0.1",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Warning: LLM summary failed ({e}), falling back to truncation")
            return self._truncate(text, max_tokens)
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)"""
        return len(text) // 4
