# RAG Playground

A hands-on sandbox for experimenting with Retrieval-Augmented Generation (RAG) workflows using modern LLM, embeddings, and vector databases.

---

## 🛠 Features

The repository covers essential components of a RAG pipeline:

### 1. Document Ingestion & Processing
- Load, preprocess and chunk documents.
- Powered by **Chroma DB** for fast similarity search and retrieval.
- Stores document embeddings with associated metadata.

### 2. Embedding Generation
- Uses **`all-MiniLM-L6-v2`** sentence transformer for efficient embeddings.
- Supports experimentation with bag-of-words text representation.
- TSNE visualization of embeddings for exploratory analysis.

### 3. Retrieval & LLM Integration
- Retrieval-augmented responses using **`gpt-4o-mini`**.
- Modular setup to experiment with different retrieval strategies.
  - Keyword Overlap
  - Cosine Distance

### 4. main_agent_driver.py
- Serves as the **entry point** to the RAG Playground, handling user queries and coordinating the retrieval and generation workflow.  
- Retrieves relevant document chunks (via embeddings) and generates answers using the LLM.
---

## ⚙️ Configuration

Before running the project, create a `config.ini` in the root directory:

```ini
[keys]
openrouter_api_key = [YOUR_OPENROUTER_KEY]
