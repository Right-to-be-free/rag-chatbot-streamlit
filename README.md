# 🧠 RAG Chatbot (Streamlit + Pinecone/FAISS/Chroma/Qdrant/MongoDB)

This project implements a **Retrieval-Augmented Generation (RAG) chatbot** with multi-database support and a Streamlit user interface.  
It includes semantic text chunking, document ingestion, metadata tracking, and querying for context-aware answers.

---

## ✨ Features

- 🔗 **Multiple Vector DBs supported**:
  - Pinecone  
  - FAISS  
  - Chroma  
  - MongoDB Atlas (Vector Search)  
  - Qdrant  

- 📄 **Supports multiple file types**: `.txt`, `.pdf`, `.csv`, `.xlsx`  
- 🧩 **Semantic text chunking** with token-aware splitting (`chunk_text_semantic`)  
- ♻️ **Deduplication** based on file hashes (avoids duplicate ingestion)  
- 👀 **Watch mode** – automatically ingest/update/remove documents on file changes  
- 💬 **Streamlit UI** with retrieved context and chatbot answers  

---

## 🗂️ Project Structure

```
.
├── app.py                # Streamlit chatbot UI
├── chunker.py            # Text splitting (semantic-aware, token-limited)
├── document_manager.py   # Manages ingestion, deduplication, metadata
├── embedding_model.py    # Embedding model wrapper
├── file_utils.py         # File loaders (PDF, CSV, XLSX, TXT)
├── ingest.py             # CLI for ingestion
├── vector_db/            # Vector DB connectors
│   ├── pinecone_db.py
│   ├── faiss_db.py
│   ├── chroma_db.py
│   ├── mongodb_db.py
│   └── qdrant_db.py
└── requirements.txt      # Python dependencies
```

---

## ⚙️ Setup

### 1. Clone the repository
```bash
git clone https://github.com/Right-to-be-free/rag-chatbot-streamlit.git
cd rag-chatbot-streamlit
```

### 2. Create a virtual environment & install dependencies
```bash
python -m venv .venv
# Activate it
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure environment variables
Create a `.env` file in the project root:

```
PINECONE_API_KEY=your-pinecone-key
PINECONE_ENV=your-pinecone-environment
MONGO_URI=your-mongodb-uri
EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

---

## 📚 Ingesting Documents

To ingest a document into your vector database:

```bash
python ingest.py Files/mydoc.pdf
```

- Documents are split into semantic chunks (`chunker.py`).  
- Embeddings are generated (`embedding_model.py`).  
- Chunks are stored in the configured vector DB with metadata (`document_manager.py`).  

Deduplication ensures unchanged files are skipped.

---

## 🔍 Querying

You can query your ingested data directly:

```python
from document_manager import DocumentManager

dm = DocumentManager(db_type="pinecone", model_name="sentence-transformers/all-MiniLM-L6-v2")
results = dm.query("What is arbitration?", top_k=3)

for r in results:
    md = r.get("metadata", {})
    print(md.get("chunk_text"))
```

---

## 👀 Watch Mode

Enable continuous ingestion by watching a folder:

```python
dm = DocumentManager(db_type="pinecone", model_name="sentence-transformers/all-MiniLM-L6-v2")
observer = dm.watch_folder("Files")
```

- 🌟 New files → automatically ingested  
- ✏️ Modified files → re-ingested  
- 🗑️ Deleted files → removed from vector DB  

---

## 💬 Run the Chatbot

Start the Streamlit app:

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.  
You’ll see:
- Retrieved context (top-k relevant chunks)  
- Generated answer from your LLM  

---

## 🚀 Roadmap

- Add Hugging Face Inference API support  
- Deploy to Streamlit Cloud / Hugging Face Spaces  
- Add connectors for Google Drive, Notion, Slack  
- Hybrid search (sparse + dense retrieval)  

---

## 🤝 Contributing

Pull requests are welcome!  
For significant changes, please open an issue first to discuss your ideas.

---

## 📜 License

This project is licensed under the MIT License.
