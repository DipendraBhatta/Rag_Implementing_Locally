
```markdown
# Local RAG System with Ollama + Phi-3 Mini

**Fully Offline | No API Keys | 100% Local Execution**

---

### 1. Task

Develop a **production-ready, fully local Retrieval-Augmented Generation (RAG)** system using **Ollama** that can answer questions based on 5 company text documents without using any external APIs (Groq, Hugging Face, OpenAI, etc.).

#### Key Requirements Implemented:
- LLM: **Phi-3 Mini** (`phi3:mini`)
- Embedding Model: `nomic-embed-text`
- Chunk Size: **1200 characters**
- Chunk Overlap: **~100 words** (500 characters)
- Vector Store: **ChromaDB** with **Cosine Similarity**
- Embedding Batch Size: **64**
- Strict context-only answers with source citation

---

#### 2. Introduction

This project implements a complete **offline RAG pipeline** for company knowledge base (Google, Tesla, Nvidia, Microsoft, SpaceX). 

Everything runs locally via **Ollama**:
- Document loading & chunking
- Embedding generation
- Vector storage & retrieval
- Final answer generation

**Advantages**:
- Complete privacy (no data leaves your machine)
- Zero recurring cost
- Works offline after initial setup
- Fast performance with Phi-3 Mini

---

#### 3. Tech Stack
    
    | Component              | Technology                              | Specification |
    |------------------------|-----------------------------------------|-------------|
    | **LLM**                | `phi3:mini` (Ollama)                    | 3.8B parameters, fast & capable |
    | **Embedding Model**    | `nomic-embed-text` (Ollama)             | 768 dimensions, excellent retrieval |
    | **Chunking**           | RecursiveCharacterTextSplitter          | 1200 chars, ~100 words overlap |
    | **Vector Database**    | **ChromaDB**                            | Persistent, local |
    | **Similarity Metric**  | **Cosine Similarity**                   | HNSW index |
    | **Batch Size**         | 64                                      | For embeddings |
    | **Framework**          | LangChain + LangChain-Ollama            | Python |
    | **Documents**          | 5 `.txt` files                          | Located in `data/` folder |

    

---

## 4. Project Structure

    ```
    rag-local-ollama/
    ├── data/                    # ← Put your  company .txt files here
    ├── chroma_db/               # Auto-created vector database
    ├── rag/
    │   ├── __init__.py
    │   ├── loader.py
    │   ├── splitter.py
    │   ├── embeddings.py
    │   ├── vectorstore.py
    │   ├── retriever.py
    │   ├── rag_prompts.py
    │   ├── qa_chain.py
    │   └── main.py
    ├── pull_models.py
    ├── requirements.txt
    ├── DOCUMENTATION.md
    └── README.md
    ```

    ---

## 5. Setup Instructions

### Step 1: Install Ollama
- Download and install Ollama from: [https://ollama.com](https://ollama.com)
- Start Ollama service (it runs in background)

### Step 2: Install Python Dependencies

```bash
pip install -r requirements.txt
```

#### Step 3: Pull Ollama Models

```bash
python pull_models.py
```

#### Step 4: Create Vector Database (Indexing)

```bash
python -m rag.vectorstore
```

#### Step 5: Run the RAG System

```bash
python -m rag.main
```

---

#### 6. How to Use

1. Type your question about any of the 5 companies
2. System will:
   - Embed the query locally
   - Retrieve most relevant chunks (cosine similarity + threshold)
   - Generate accurate answer using **Phi-3 Mini**
3. Type `exit`, `quit`, or `q` to stop

---

#### 7. File Descriptions

- **`rag/loader.py`** – Loads `.txt` files from `data/` folder
- **`rag/splitter.py`** – Splits documents (1200 chars, 100-word overlap)
- **`rag/embeddings.py`** – Uses `nomic-embed-text` via Ollama (batch=64)
- **`rag/vectorstore.py`** – Creates ChromaDB with cosine similarity
- **`rag/retriever.py`** – Retrieves top-6 relevant chunks with score filter
- **`rag/qa_chain.py`** – Main RAG pipeline (context + prompt + LLM)
- **`rag/main.py`** – Interactive chat interface
- **`pull_models.py`** – Downloads required Ollama models
- **`rag/rag_prompts.py`** – High-quality system prompt

---

#### 8. Performance Tips

- For **faster response**: Use `phi3:mini`
- For **better quality**: Later switch to `llama3.2:3b` or `phi3:medium`
- First run (indexing) will take time. Subsequent runs are fast.
- Recommended RAM: 8GB+ (16GB ideal)

---

#### 9. Future Improvements (Optional)

- Add metadata filtering
- Implement reranking
- Add conversation memory
- Web UI with Gradio/Streamlit
- Support for PDF & Markdown files

---

#### 10. Troubleshooting

**Common Issues**:
- Ollama not running → Start Ollama app
- Model not found → Run `pull_models.py` again
- ChromaDB error → Delete `chroma_db/` folder and re-index
- Slow embedding → Ensure Ollama is using GPU (if available)

---

**Project Status**: Complete & Ready for Production Use  
**Author**: LLM Engineer Implementation  
**Date**: April 2026

---

**You are now running a fully local, private, and powerful RAG system!** 

---