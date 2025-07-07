"""
Streamlit web UI for ContextOS
"""

import os
import streamlit as st
from pathlib import Path
import yaml
from dotenv import load_dotenv
from contextos.assembler import ContextAssembler
from contextos.rag import RAGConnector

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="ContextOS",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "api_key" not in st.session_state:
    st.session_state.api_key = os.getenv("TOGETHER_API_KEY", "")

if "assembler" not in st.session_state:
    st.session_state.assembler = None

if "rag_connector" not in st.session_state:
    st.session_state.rag_connector = RAGConnector()

def main():
    st.title("🧠 ContextOS")
    st.markdown("*Developer tool for constructing optimized context windows for LLMs*")
    
    # Sidebar
    with st.sidebar:
        st.header("🛠️ Configuration")
        
        # API Key input
        st.subheader("🔑 Together.ai API Key")
        api_key_input = st.text_input(
            "Enter your Together.ai API Key",
            value=st.session_state.api_key,
            type="password",
            help="Get your API key from https://api.together.xyz/"
        )
        
        if api_key_input != st.session_state.api_key:
            st.session_state.api_key = api_key_input
            st.session_state.assembler = None  # Reset assembler
            st.rerun()
        
        if st.session_state.api_key:
            if st.session_state.assembler is None:
                try:
                    st.session_state.assembler = ContextAssembler(api_key=st.session_state.api_key)
                    st.success("✅ API key loaded successfully!")
                except Exception as e:
                    st.error(f"❌ Error initializing with API key: {e}")
            else:
                st.success("✅ Ready to use!")
        else:
            st.warning("⚠️ Please enter your Together.ai API key to get started")
        
        # Recipe selection
        st.subheader("📋 Recipes")
        if st.session_state.assembler:
            recipes = st.session_state.assembler.get_available_recipes()
            
            if not recipes:
                st.warning("No recipes found. Create recipes in the 'recipes/' directory.")
                selected_recipe = None
            else:
                selected_recipe = st.selectbox("Select Recipe", recipes)
        else:
            selected_recipe = None
            st.info("Enter API key to see available recipes")
        
        # RAG Stats
        st.subheader("🔍 RAG Index")
        try:
            rag_stats = st.session_state.rag_connector.get_stats()
            st.metric("Total Chunks", rag_stats["total_chunks"])
            st.metric("Sources", len(rag_stats["sources"]))
        except Exception as e:
            st.error(f"Error loading RAG stats: {e}")
    
    # Main content
    if selected_recipe:
        show_recipe_interface(selected_recipe)
    else:
        show_setup_page()

def show_setup_page():
    """Show setup instructions when no recipes are available"""
    st.header("🚀 Getting Started")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Set up API Key")
        st.code("export TOGETHER_API_KEY=your_api_key_here", language="bash")
        
        st.subheader("2. Create Recipe Directory")
        if st.button("Create recipes/ directory"):
            Path("recipes").mkdir(exist_ok=True)
            st.success("Created recipes/ directory")
        
        st.subheader("3. Index Documents")
        docs_path = st.text_input("Documents path", value="documents")
        if st.button("Index Documents"):
            if Path(docs_path).exists():
                try:
                    with st.spinner("Indexing documents..."):
                        st.session_state.rag_connector.index_documents(docs_path)
                    st.success("Documents indexed successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error indexing documents: {e}")
            else:
                st.error(f"Directory {docs_path} does not exist")
    
    with col2:
        st.subheader("📄 Example Recipe")
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
system_instruction: "You are a helpful assistant that creates concise, accurate summaries."
"""
        st.code(example_recipe, language="yaml")
        
        if st.button("Create Example Recipe"):
            Path("recipes").mkdir(exist_ok=True)
            with open("recipes/summarization.yaml", "w") as f:
                f.write(example_recipe)
            st.success("Created example recipe!")
            st.rerun()

def show_recipe_interface(recipe_name: str):
    """Show the main recipe interface"""
    
    # Recipe info
    try:
        recipe_info = st.session_state.assembler.get_recipe_info(recipe_name)
        
        with st.expander(f"📋 Recipe Details: {recipe_name}"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Model", recipe_info["model"])
                st.metric("Max Tokens", recipe_info["max_tokens"])
            
            with col2:
                st.metric("Temperature", recipe_info["temperature"])
                st.metric("Top-K (RAG)", recipe_info["top_k"])
            
            with col3:
                st.write("**Components:**")
                for comp in recipe_info["components"]:
                    st.write(f"• {comp}")
                st.write(f"**Compact Strategy:** {recipe_info['compact_strategy']}")
    
    except Exception as e:
        st.error(f"Error loading recipe info: {e}")
        return
    
    # Main interface
    st.header("💬 Query Interface")
    
    # Input section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        user_input = st.text_area("User Input", height=100, placeholder="Enter your query here...")
        
        # Document upload/input
        st.subheader("📄 Document (Optional)")
        doc_option = st.radio("Document Input Method", ["Upload File", "Paste Text", "None"])
        
        document_text = ""
        if doc_option == "Upload File":
            uploaded_file = st.file_uploader("Choose a file", type=['txt', 'md'])
            if uploaded_file:
                document_text = str(uploaded_file.read(), "utf-8")
                st.success(f"Loaded document: {uploaded_file.name}")
        
        elif doc_option == "Paste Text":
            document_text = st.text_area("Document Text", height=150)
    
    with col2:
        st.subheader("⚙️ Settings")
        temperature = st.slider("Temperature", 0.0, 2.0, recipe_info["temperature"], 0.1)
        show_context = st.checkbox("Show Context Window")
        
        # Process button
        process_button = st.button("🚀 Process Query", type="primary", use_container_width=True)
    
    # Process query
    if process_button and user_input.strip():
        try:
            with st.spinner("Processing query..."):
                response = st.session_state.assembler.process_query(
                    recipe_name=recipe_name,
                    user_input=user_input,
                    document_text=document_text,
                    temperature=temperature
                )
            
            # Show context if requested
            if show_context:
                st.subheader("📋 Context Window")
                with st.expander("View Assembled Context", expanded=False):
                    st.code(response.context_window.prompt, language="text")
            
            # Show response
            st.subheader("🤖 Response")
            st.markdown(response.response)
            
            # Show metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Context Tokens", response.context_window.token_count)
            with col2:
                st.metric("Response Tokens", response.token_count)
            with col3:
                st.metric("Total Tokens", response.context_window.token_count + response.token_count)
            
            # Download response
            st.download_button(
                label="💾 Download Response",
                data=f"# Query: {user_input}\n\n**Recipe**: {recipe_name}\n**Model**: {response.model}\n\n## Response\n\n{response.response}",
                file_name=f"contextos_response_{recipe_name}.md",
                mime="text/markdown"
            )
            
        except Exception as e:
            st.error(f"Error processing query: {e}")
    
    elif process_button:
        st.warning("Please enter a query before processing.")

if __name__ == "__main__":
    main()
