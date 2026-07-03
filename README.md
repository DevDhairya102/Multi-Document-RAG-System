# Research RAG — Paper Assistant

A Streamlit-based RAG (Retrieval-Augmented Generation) app for querying research papers using FAISS + Gemini.

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add your Google API key
Rename `.env` and fill in your key:
```
GOOGLE_API_KEY=your_google_api_key_here
```

### 3. Run the app
```bash
streamlit run app.py
```

## How it works

1. Upload any PDF via the sidebar
2. The app extracts text, splits into 300-token chunks, and encodes them with `all-MiniLM-L6-v2` (384-dim)
3. Chunks are stored in a FAISS `IndexFlatL2` vector index
4. When you ask a question, the top-5 nearest chunks are retrieved and sent to Gemini 2.5 Flash as context
5. Gemini answers strictly from the retrieved context

## Project structure
```
rag_app/
├── app.py              # Main Streamlit app
├── requirements.txt    # Python dependencies
├── .env                # API keys (never commit this)
├── .gitignore
└── data/
    └── papers/         # Optional: pre-load PDFs here
```

## Config (in app.py)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `chunk_size` | 300 | Tokens per chunk |
| `chunk_overlap` | 50 | Overlap between chunks |
| `top_k` | 5 | Chunks retrieved per query |
| Model | `all-MiniLM-L6-v2` | Embedding model |
| LLM | `gemini-2.5-flash` | Generation model |
