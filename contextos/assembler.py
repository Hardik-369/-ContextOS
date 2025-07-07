"""
Context assembler and main orchestrator
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
import together
from .models import Recipe, ContextComponent, ContextWindow, LLMResponse, RAGChunk
from .compactors import Compactor
from .rag import RAGConnector


class ContextAssembler:
    """Main orchestrator for context assembly and LLM interaction"""
    
    def __init__(self, recipes_path: str = "recipes", rag_index_path: str = "rag_index", api_key: str = None):
        self.recipes_path = Path(recipes_path)
        self.rag_index_path = rag_index_path
        self.api_key = api_key or os.getenv("TOGETHER_API_KEY")
        
        # Initialize components
        self.compactor = Compactor(api_key=self.api_key)
        self.rag_connector = RAGConnector(index_path=rag_index_path)
        self.together_client = together.Together(api_key=self.api_key) if self.api_key else None
        
        # Load recipes
        self.recipes: Dict[str, Recipe] = {}
        self._load_recipes()
    
    def _load_recipes(self):
        """Load all recipes from the recipes directory"""
        if not self.recipes_path.exists():
            print(f"Warning: Recipes directory {self.recipes_path} does not exist")
            return
        
        for recipe_file in self.recipes_path.glob("*.yaml"):
            try:
                with open(recipe_file, 'r') as f:
                    recipe_data = yaml.safe_load(f)
                
                recipe = Recipe(**recipe_data)
                self.recipes[recipe.task] = recipe
                print(f"Loaded recipe: {recipe.task}")
            except Exception as e:
                print(f"Error loading recipe {recipe_file}: {e}")
    
    def get_available_recipes(self) -> List[str]:
        """Get list of available recipe names"""
        return list(self.recipes.keys())
    
    def assemble_context(self, recipe_name: str, user_input: str, 
                        document_text: str = "", chat_history: List[Dict[str, str]] = None) -> ContextWindow:
        """Assemble context window according to recipe"""
        if recipe_name not in self.recipes:
            raise ValueError(f"Recipe '{recipe_name}' not found")
        
        recipe = self.recipes[recipe_name]
        components = {}
        context_parts = []
        
        # Process each component
        for component in recipe.include:
            if component == ContextComponent.SYSTEM_INSTRUCTION:
                if recipe.system_instruction:
                    instruction = recipe.system_instruction
                else:
                    instruction = self._get_default_system_instruction(recipe.task)
                
                components[component] = instruction
                context_parts.append(f"<system>\n{instruction}\n</system>")
            
            elif component == ContextComponent.USER_INPUT:
                components[component] = user_input
                context_parts.append(f"<user_input>\n{user_input}\n</user_input>")
            
            elif component == ContextComponent.DOCUMENT_TEXT:
                if document_text:
                    # Apply compacting strategy
                    compacted_text = self.compactor.compact(
                        document_text, 
                        recipe.compact_strategy, 
                        max_tokens=recipe.max_tokens // 2
                    )
                    components[component] = compacted_text
                    context_parts.append(f"<document>\n{compacted_text}\n</document>")
            
            elif component == ContextComponent.RAG_CHUNKS:
                rag_chunks = self.rag_connector.query(user_input, top_k=recipe.top_k)
                if rag_chunks:
                    chunks_text = self._format_rag_chunks(rag_chunks)
                    components[component] = rag_chunks
                    context_parts.append(f"<relevant_context>\n{chunks_text}\n</relevant_context>")
            
            elif component == ContextComponent.FEW_SHOT:
                if recipe.few_shot_examples:
                    few_shot_text = self._format_few_shot_examples(recipe.few_shot_examples)
                    components[component] = recipe.few_shot_examples
                    context_parts.append(f"<examples>\n{few_shot_text}\n</examples>")
            
            elif component == ContextComponent.CHAT_HISTORY:
                if chat_history:
                    history_text = self._format_chat_history(chat_history)
                    components[component] = chat_history
                    context_parts.append(f"<chat_history>\n{history_text}\n</chat_history>")
        
        # Assemble final prompt
        prompt = "\n\n".join(context_parts)
        
        # Estimate token count
        token_count = self.compactor.estimate_tokens(prompt)
        
        return ContextWindow(
            prompt=prompt,
            components=components,
            token_count=token_count,
            model=recipe.model
        )
    
    def generate_response(self, context_window: ContextWindow, temperature: float = None) -> LLMResponse:
        """Generate LLM response using Together.ai"""
        if not self.together_client:
            raise ValueError("Together.ai API key not provided. Please set API key.")
        
        recipe = self.recipes.get(context_window.model)
        temp = temperature if temperature is not None else (recipe.temperature if recipe else 0.7)
        
        try:
            response = self.together_client.chat.completions.create(
                model=context_window.model,
                messages=[{"role": "user", "content": context_window.prompt}],
                temperature=temp,
                max_tokens=2000
            )
            
            response_text = response.choices[0].message.content
            response_tokens = self.compactor.estimate_tokens(response_text)
            
            return LLMResponse(
                response=response_text,
                token_count=response_tokens,
                model=context_window.model,
                context_window=context_window
            )
        except Exception as e:
            raise Exception(f"Error generating response: {e}")
    
    def process_query(self, recipe_name: str, user_input: str, 
                     document_text: str = "", chat_history: List[Dict[str, str]] = None,
                     temperature: float = None) -> LLMResponse:
        """End-to-end processing: assemble context and generate response"""
        context_window = self.assemble_context(recipe_name, user_input, document_text, chat_history)
        return self.generate_response(context_window, temperature)
    
    def _get_default_system_instruction(self, task: str) -> str:
        """Get default system instruction for a task"""
        instructions = {
            "summarization": "You are a helpful assistant that creates concise, accurate summaries.",
            "question_answering": "You are a helpful assistant that answers questions based on the provided context.",
            "code_review": "You are an expert code reviewer. Provide constructive feedback and suggestions.",
            "creative_writing": "You are a creative writing assistant. Help with storytelling and creative content.",
            "analysis": "You are an analytical assistant. Provide thorough analysis and insights."
        }
        return instructions.get(task, "You are a helpful assistant.")
    
    def _format_rag_chunks(self, chunks: List[RAGChunk]) -> str:
        """Format RAG chunks for context"""
        formatted_chunks = []
        for i, chunk in enumerate(chunks, 1):
            formatted_chunks.append(f"Chunk {i} (similarity: {chunk.similarity:.3f}, source: {Path(chunk.source).name}):\n{chunk.content}")
        return "\n\n".join(formatted_chunks)
    
    def _format_few_shot_examples(self, examples: List[Dict[str, str]]) -> str:
        """Format few-shot examples for context"""
        formatted_examples = []
        for i, example in enumerate(examples, 1):
            formatted_examples.append(f"Example {i}:\nInput: {example.get('input', '')}\nOutput: {example.get('output', '')}")
        return "\n\n".join(formatted_examples)
    
    def _format_chat_history(self, history: List[Dict[str, str]]) -> str:
        """Format chat history for context"""
        formatted_history = []
        for msg in history:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            formatted_history.append(f"{role.capitalize()}: {content}")
        return "\n".join(formatted_history)
    
    def get_recipe_info(self, recipe_name: str) -> Dict[str, Any]:
        """Get detailed information about a recipe"""
        if recipe_name not in self.recipes:
            raise ValueError(f"Recipe '{recipe_name}' not found")
        
        recipe = self.recipes[recipe_name]
        return {
            "task": recipe.task,
            "model": recipe.model,
            "components": [comp.value for comp in recipe.include],
            "compact_strategy": recipe.compact_strategy.value,
            "max_tokens": recipe.max_tokens,
            "temperature": recipe.temperature,
            "top_k": recipe.top_k
        }
