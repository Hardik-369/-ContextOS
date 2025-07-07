# ContextOS 🧠

A powerful developer tool for constructing optimized context windows for Large Language Models using **Together.ai**. ContextOS helps you build smarter prompts through YAML-based recipes, smart context compaction, RAG integration, and both CLI and web interfaces.

## ✨ Features

- **📋 YAML-based Context Recipes**: Define reusable context construction templates
- **🔧 Smart Compactors**: Intelligent text truncation and LLM-powered summarization  
- **🔍 RAG Integration**: FAISS + HuggingFace embeddings for document retrieval
- **⚡ Context Assembly**: Automated prompt construction and optimization
- **🖥️ CLI Tool**: Feature-rich command-line interface with Typer
- **🌐 Web UI**: Beautiful Streamlit interface for interactive use
- **🤖 Together.ai Integration**: Direct LLM interaction with token tracking

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/Hardik-369/-ContextOS
cd -ContextOs

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env and add your Together.ai API key
```

### 2. Initialize Project

```bash
# Initialize project structure
python main.py init

# Or manually create directories
mkdir recipes documents rag_index
```

### 3. Get Your Together.ai API Key

1. Visit [Together.ai](https://api.together.xyz/)
2. Sign up and get your API key
3. Add it to your `.env` file:
   ```
   TOGETHER_API_KEY=your_api_key_here
   ```

### 4. Add Documents (Optional)

```bash
# Add your .txt or .md files to the documents/ directory
cp your_documents/*.txt documents/
cp your_documents/*.md documents/

# Index documents for RAG
python main.py index-docs documents/
```

### 5. Run Your First Query

```bash
# List available recipes
python main.py list-recipes

# Run a query using the summarization recipe
python main.py query summarization "Summarize the main points" --document path/to/your/document.txt

# Or use the web interface
streamlit run streamlit_app.py
```

## 📋 Recipe System

Recipes are YAML files that define how to construct context windows. Here's the anatomy of a recipe:

```yaml
task: summarization                    # Task name/identifier
include:                              # Components to include in context
  - system_instruction
  - document_text
  - user_input
compact_strategy: llm_summary         # How to handle long content
model: mistralai/Mixtral-8x7B-Instruct-v0.1  # Together.ai model
max_tokens: 4000                      # Maximum context tokens
temperature: 0.7                      # Generation temperature
top_k: 3                             # Number of RAG chunks to retrieve
system_instruction: "Custom instruction..."  # Optional custom instruction
```

### Available Components

- `system_instruction`: System/role instruction for the LLM
- `user_input`: The user's query or input
- `document_text`: Document content (with compaction)
- `rag_chunks`: Retrieved relevant chunks from indexed documents
- `few_shot`: Few-shot examples for in-context learning
- `chat_history`: Previous conversation context

### Compact Strategies

- `truncate`: Simple truncation to fit token limits
- `llm_summary`: AI-powered summarization while preserving key information

## 🖥️ CLI Usage

### Recipe Management

```bash
# List all available recipes
python main.py list-recipes

# Get detailed recipe information  
python main.py recipe-info summarization
```

### Document Indexing

```bash
# Index documents for RAG
python main.py index-docs documents/ --chunk-size 512 --overlap 50

# Check RAG index statistics
python main.py rag-stats
```

### Query Processing

```bash
# Basic query
python main.py query question_answering "What is machine learning?"

# Query with document
python main.py query summarization "Summarize this" --document report.txt

# Query with custom temperature
python main.py query code_review "Review this code" --document code.py --temperature 0.3

# Show assembled context
python main.py query summarization "Summarize" --show-context

# Save response to file
python main.py query summarization "Summarize" --output response.md
```

## 🌐 Web Interface

Launch the Streamlit web interface:

```bash
streamlit run streamlit_app.py
```

Features:
- 📋 Recipe selection and configuration
- 📄 Document upload or text pasting
- ⚙️ Real-time parameter adjustment
- 📊 Token usage monitoring
- 💾 Response download
- 🔍 Context window inspection

## 📁 Project Structure

```
contextos/
├── contextos/              # Main package
│   ├── __init__.py
│   ├── models.py          # Pydantic models
│   ├── compactors.py      # Text compaction strategies
│   ├── rag.py             # RAG connector with FAISS
│   ├── assembler.py       # Context assembly orchestrator
│   └── cli.py             # CLI interface
├── recipes/               # YAML recipe definitions
│   ├── summarization.yaml
│   ├── question_answering.yaml
│   └── code_review.yaml
├── documents/             # Documents for RAG indexing
├── rag_index/            # FAISS index storage
├── main.py               # CLI entry point
├── streamlit_app.py      # Web UI
├── requirements.txt      # Python dependencies
├── .env.example         # Environment variables template
└── README.md           # This file
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file:

```bash
# Required
TOGETHER_API_KEY=your_api_key_here

# Optional paths (defaults shown)
RECIPES_PATH=recipes
RAG_INDEX_PATH=rag_index
DOCUMENTS_PATH=documents
```

### Creating Custom Recipes

1. Create a new YAML file in `recipes/`:

```yaml
task: my_custom_task
include:
  - system_instruction
  - user_input
  - rag_chunks
compact_strategy: truncate
model: mistralai/Mixtral-8x7B-Instruct-v0.1
max_tokens: 4000
temperature: 0.5
top_k: 3
system_instruction: "You are a helpful assistant specialized in..."
```

2. The recipe will be automatically loaded and available in both CLI and web interface.

## 🤖 Supported Models

ContextOS works with all Together.ai models. Popular choices:

- `mistralai/Mixtral-8x7B-Instruct-v0.1` - Great general purpose model
- `mistralai/Mistral-7B-Instruct-v0.1` - Faster, lighter option
- `meta-llama/Llama-2-7b-chat-hf` - Alternative option
- `togethercomputer/RedPajama-INCITE-Chat-3B-v1` - Smallest option

## 🔍 RAG System

The RAG (Retrieval-Augmented Generation) system uses:

- **FAISS**: Fast similarity search
- **sentence-transformers**: High-quality embeddings
- **Smart chunking**: Overlapping text chunks with word-boundary awareness

### Indexing Documents

```bash
# Index with custom parameters
python main.py index-docs documents/ --chunk-size 1024 --overlap 100
```

### RAG in Recipes

Include `rag_chunks` in your recipe to automatically retrieve relevant context:

```yaml
include:
  - system_instruction
  - rag_chunks  # Will retrieve top_k relevant chunks
  - user_input
top_k: 5        # Number of chunks to retrieve
```

## 📊 Token Management

ContextOS provides intelligent token management:

- **Estimation**: Rough token counting (1 token ≈ 4 characters)
- **Compaction**: Automatic content compression when needed
- **Monitoring**: Real-time token usage tracking
- **Optimization**: Smart component prioritization

## 🎯 Example Workflows

### Document Summarization

```bash
# 1. Create/check summarization recipe
python main.py recipe-info summarization

# 2. Summarize a document
python main.py query summarization "Create a concise summary" --document report.pdf

# 3. Save the summary
python main.py query summarization "Create a concise summary" --document report.pdf --output summary.md
```

### RAG-based Q&A

```bash
# 1. Index your knowledge base
python main.py index-docs knowledge_base/

# 2. Ask questions
python main.py query question_answering "What are the benefits of renewable energy?"

# 3. Check what was retrieved
python main.py query question_answering "What are the benefits of renewable energy?" --show-context
```

### Code Review

```bash
# 1. Review code with few-shot examples
python main.py query code_review "Please review this Python code for best practices" --document my_script.py

# 2. Lower temperature for more focused feedback
python main.py query code_review "Check for security issues" --document app.py --temperature 0.2
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Submit a pull request

## 📄 License

This project is open source. Feel free to use, modify, and distribute according to your needs.

## 🆘 Troubleshooting

### Common Issues

**Import Errors**
```bash
# Make sure you're in the right directory and dependencies are installed
pip install -r requirements.txt
```

**API Key Issues**  
```bash
# Check your .env file
cat .env
# Make sure TOGETHER_API_KEY is set correctly
```

**RAG Index Issues**
```bash
# Clear and rebuild index
rm -rf rag_index/
python main.py index-docs documents/
```

**Recipe Not Found**
```bash
# Check recipe syntax
python -c "import yaml; print(yaml.safe_load(open('recipes/your_recipe.yaml')))"
```

## 🚀 What's Next?

- **Advanced Compaction**: More sophisticated summarization strategies
- **Multi-modal Support**: Image and audio processing
- **Plugin System**: Extensible component architecture  
- **Caching**: Response and context caching for efficiency
- **Analytics**: Usage analytics and optimization insights

---

**Happy Context Building!** 🎉

For questions, issues, or contributions, please visit our [GitHub repository](https://github.com/Hardik-369/-ContextOS).
