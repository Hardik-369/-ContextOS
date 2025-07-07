"""
CLI tool for ContextOS
"""

import os
import typer
from typing import Optional, List
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown
from .assembler import ContextAssembler
from .rag import RAGConnector

app = typer.Typer(name="contextos", help="ContextOS - Developer tool for constructing optimized context windows for LLMs")
console = Console()

# Global assembler instance
assembler = None
global_api_key = None

def get_assembler(api_key: str = None):
    """Get or create assembler instance"""
    global assembler, global_api_key
    
    # If API key changed, reset assembler
    if api_key and api_key != global_api_key:
        assembler = None
        global_api_key = api_key
    
    if assembler is None:
        # Try API key parameter, then environment variable
        key_to_use = api_key or global_api_key or os.getenv("TOGETHER_API_KEY")
        if not key_to_use:
            console.print("❌ No API key provided. Use --api-key parameter or set TOGETHER_API_KEY environment variable.", style="red")
            raise typer.Exit(1)
        
        try:
            assembler = ContextAssembler(api_key=key_to_use)
            global_api_key = key_to_use
        except Exception as e:
            console.print(f"❌ Error initializing ContextOS: {e}", style="red")
            raise typer.Exit(1)
    
    return assembler


@app.command()
def list_recipes(
    api_key: Optional[str] = typer.Option(None, "--api-key", "-k", help="Together.ai API key")
):
    """List all available recipes"""
    asm = get_assembler(api_key)
    recipes = asm.get_available_recipes()
    
    if not recipes:
        console.print("❌ No recipes found. Create some recipes in the 'recipes/' directory first.", style="red")
        return
    
    table = Table(title="Available Recipes")
    table.add_column("Recipe", style="cyan")
    table.add_column("Model", style="green")
    table.add_column("Components", style="yellow")
    
    for recipe_name in recipes:
        info = asm.get_recipe_info(recipe_name)
        components = ", ".join(info["components"])
        table.add_row(recipe_name, info["model"], components)
    
    console.print(table)


@app.command()
def recipe_info(
    recipe_name: str,
    api_key: Optional[str] = typer.Option(None, "--api-key", "-k", help="Together.ai API key")
):
    """Get detailed information about a recipe"""
    asm = get_assembler(api_key)
    
    try:
        info = asm.get_recipe_info(recipe_name)
        
        console.print(Panel(f"""
**Task**: {info['task']}
**Model**: {info['model']}
**Components**: {', '.join(info['components'])}
**Compact Strategy**: {info['compact_strategy']}
**Max Tokens**: {info['max_tokens']}
**Temperature**: {info['temperature']}
**Top-K (RAG)**: {info['top_k']}
        """, title=f"Recipe: {recipe_name}", title_align="left"))
        
    except ValueError as e:
        console.print(f"❌ {e}", style="red")


@app.command()
def query(
    recipe_name: str,
    user_input: str,
    document_file: Optional[str] = typer.Option(None, "--document", "-d", help="Path to document file"),
    temperature: Optional[float] = typer.Option(None, "--temperature", "-t", help="Temperature for generation"),
    show_context: bool = typer.Option(False, "--show-context", "-c", help="Show assembled context"),
    output_file: Optional[str] = typer.Option(None, "--output", "-o", help="Save response to file"),
    api_key: Optional[str] = typer.Option(None, "--api-key", "-k", help="Together.ai API key")
):
    """Process a query using the specified recipe"""
    asm = get_assembler(api_key)
    
    # Check if recipe exists
    if recipe_name not in asm.get_available_recipes():
        console.print(f"❌ Recipe '{recipe_name}' not found", style="red")
        return
    
    # Load document if provided
    document_text = ""
    if document_file:
        doc_path = Path(document_file)
        if not doc_path.exists():
            console.print(f"❌ Document file not found: {document_file}", style="red")
            return
        
        try:
            with open(doc_path, 'r', encoding='utf-8') as f:
                document_text = f.read()
            console.print(f"📄 Loaded document: {document_file}")
        except Exception as e:
            console.print(f"❌ Error loading document: {e}", style="red")
            return
    
    # Process query
    try:
        with console.status("[bold green]Processing query..."):
            response = asm.process_query(
                recipe_name=recipe_name,
                user_input=user_input,
                document_text=document_text,
                temperature=temperature
            )
        
        # Show context if requested
        if show_context:
            console.print(Panel(
                response.context_window.prompt,
                title="📋 Assembled Context",
                title_align="left"
            ))
            console.print()
        
        # Show response
        console.print(Panel(
            Markdown(response.response),
            title=f"🤖 Response (Model: {response.model})",
            title_align="left"
        ))
        
        # Show token counts
        console.print(f"📊 Context tokens: {response.context_window.token_count}")
        console.print(f"📊 Response tokens: {response.token_count}")
        
        # Save to file if requested
        if output_file:
            output_path = Path(output_file)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"# Query: {user_input}\n\n")
                f.write(f"**Recipe**: {recipe_name}\n")
                f.write(f"**Model**: {response.model}\n\n")
                f.write("## Response\n\n")
                f.write(response.response)
            
            console.print(f"💾 Response saved to: {output_file}")
        
    except Exception as e:
        console.print(f"❌ Error processing query: {e}", style="red")


@app.command()
def index_docs(
    documents_path: str,
    chunk_size: int = typer.Option(512, "--chunk-size", help="Size of text chunks"),
    overlap: int = typer.Option(50, "--overlap", help="Overlap between chunks")
):
    """Index documents for RAG"""
    docs_path = Path(documents_path)
    
    if not docs_path.exists():
        console.print(f"❌ Documents path does not exist: {documents_path}", style="red")
        return
    
    try:
        rag_connector = RAGConnector()
        
        with console.status("[bold green]Indexing documents..."):
            rag_connector.index_documents(documents_path, chunk_size, overlap)
        
        stats = rag_connector.get_stats()
        console.print(f"✅ Indexed {stats['total_chunks']} chunks from {len(stats['sources'])} files")
        
    except Exception as e:
        console.print(f"❌ Error indexing documents: {e}", style="red")


@app.command()
def rag_stats():
    """Show RAG index statistics"""
    try:
        rag_connector = RAGConnector()
        stats = rag_connector.get_stats()
        
        console.print(Panel(f"""
**Total Chunks**: {stats['total_chunks']}
**Embedding Dimension**: {stats['embedding_dim']}
**Model**: {stats['model_name']}
**Sources**: {len(stats['sources'])}
        """, title="RAG Index Statistics", title_align="left"))
        
        if stats['sources']:
            console.print("\n📂 **Indexed Sources:**")
            for source in stats['sources']:
                console.print(f"  • {Path(source).name}")
        
    except Exception as e:
        console.print(f"❌ Error getting RAG stats: {e}", style="red")


@app.command()
def setup_api_key():
    """Interactively set up Together.ai API key"""
    console.print("🔑 Together.ai API Key Setup", style="bold blue")
    console.print("\nGet your API key from: https://api.together.xyz/")
    
    api_key = typer.prompt("Enter your Together.ai API Key", hide_input=True)
    
    if api_key:
        # Test the API key
        try:
            test_assembler = ContextAssembler(api_key=api_key)
            console.print("✅ API key is valid!", style="green")
            
            # Save to .env file
            env_path = Path(".env")
            if env_path.exists():
                # Read existing .env
                with open(env_path, 'r') as f:
                    lines = f.readlines()
                
                # Update or add TOGETHER_API_KEY
                updated = False
                for i, line in enumerate(lines):
                    if line.startswith("TOGETHER_API_KEY="):
                        lines[i] = f"TOGETHER_API_KEY={api_key}\n"
                        updated = True
                        break
                
                if not updated:
                    lines.append(f"TOGETHER_API_KEY={api_key}\n")
                
                with open(env_path, 'w') as f:
                    f.writelines(lines)
            else:
                # Create new .env file
                with open(env_path, 'w') as f:
                    f.write(f"TOGETHER_API_KEY={api_key}\n")
            
            console.print(f"💾 API key saved to .env file", style="green")
            console.print("\n🚀 You're ready to use ContextOS!")
            
        except Exception as e:
            console.print(f"❌ Error testing API key: {e}", style="red")
    else:
        console.print("❌ No API key provided", style="red")


@app.command()
def init():
    """Initialize ContextOS project structure"""
    # Create directories
    dirs = ["recipes", "documents", "rag_index"]
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
        console.print(f"📁 Created directory: {dir_name}")
    
    # Create example recipe
    example_recipe = """task: summarization
include:
  - system_instruction
  - document_text
  - user_input
compact_strategy: llm_summary
model: mistralai/Mixtral-8x7B-Instruct-v0.1
max_tokens: 4000
temperature: 0.7
top_k: 3
system_instruction: "You are a helpful assistant that creates concise, accurate summaries while preserving key information."
"""
    
    recipe_path = Path("recipes/summarization.yaml")
    if not recipe_path.exists():
        with open(recipe_path, 'w') as f:
            f.write(example_recipe)
        console.print(f"📄 Created example recipe: {recipe_path}")
    
    # Create .env example
    env_example = """# Together.ai API Key
TOGETHER_API_KEY=your_api_key_here
"""
    
    env_path = Path(".env.example")
    if not env_path.exists():
        with open(env_path, 'w') as f:
            f.write(env_example)
        console.print(f"📄 Created example environment file: {env_path}")
    
    console.print("\n✅ ContextOS project initialized!")
    console.print("💡 Don't forget to:")
    console.print("  1. Copy .env.example to .env and add your Together.ai API key")
    console.print("  2. Add documents to the 'documents/' directory")
    console.print("  3. Run 'contextos index-docs documents/' to index your documents")


if __name__ == "__main__":
    app()
