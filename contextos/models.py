"""
Core models and data structures for ContextOS
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
from enum import Enum


class CompactStrategy(str, Enum):
    """Strategies for compacting context"""
    TRUNCATE = "truncate"
    LLM_SUMMARY = "llm_summary"


class ContextComponent(str, Enum):
    """Available context components"""
    SYSTEM_INSTRUCTION = "system_instruction"
    USER_INPUT = "user_input"
    DOCUMENT_TEXT = "document_text"
    RAG_CHUNKS = "rag_chunks"
    FEW_SHOT = "few_shot"
    CHAT_HISTORY = "chat_history"


class Recipe(BaseModel):
    """Context recipe configuration"""
    task: str = Field(..., description="Name of the task")
    include: List[ContextComponent] = Field(..., description="Components to include in context")
    compact_strategy: CompactStrategy = Field(default=CompactStrategy.TRUNCATE, description="Strategy for compacting context")
    model: str = Field(..., description="Together.ai model to use")
    max_tokens: int = Field(default=4000, description="Maximum tokens for context")
    temperature: float = Field(default=0.7, description="Temperature for LLM generation")
    top_k: int = Field(default=3, description="Number of RAG chunks to retrieve")
    system_instruction: Optional[str] = Field(default=None, description="System instruction text")
    few_shot_examples: Optional[List[Dict[str, str]]] = Field(default=None, description="Few-shot examples")


class RAGChunk(BaseModel):
    """RAG chunk with metadata"""
    content: str
    source: str
    similarity: float


class ContextWindow(BaseModel):
    """Assembled context window"""
    prompt: str
    components: Dict[str, Any]
    token_count: int
    model: str


class LLMResponse(BaseModel):
    """LLM response with metadata"""
    response: str
    token_count: int
    model: str
    context_window: ContextWindow
