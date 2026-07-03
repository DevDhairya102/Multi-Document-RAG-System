# 🧠 Multi-Document RAG System

> A research paper Q&A assistant powered by FAISS vector search + Google Gemini 2.5 Flash

![Python](https://img.shields.io/badge/Python-3.11-b8aa97?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-c3c5bc?style=flat-square&logo=streamlit&logoColor=white)
![Gemini](https://img.shields.io/badge/Gemini-2.5%20Flash-d8dbd2?style=flat-square&logo=google&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-f5f2e3?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-b8aa97?style=flat-square)

---

## ✨ What it does

Upload any research paper PDF and instantly ask questions about it. The app chunks the paper, encodes it into vector embeddings, stores them in FAISS, and retrieves the most relevant sections to answer your question using Gemini — all without hallucinating beyond the paper's content.

---

## 🖥️ Screenshots

### App ready — upload any PDF
<img width="1526" height="704" alt="image" src="https://github.com/user-attachments/assets/a5342a85-207f-4781-be13-af7ec578e3a7" />


### Ask questions and get answers
<img width="1517" height="696" alt="image" src="https://github.com/user-attachments/assets/bfa0a198-e91f-4f89-87be-16cd9d0c21d4" />


### Dataset questions answered accurately
<img width="1078" height="193" alt="image" src="https://github.com/user-attachments/assets/b426bde1-c4f3-4e32-94e5-3f53466f71c6" />


### Retrieved context inspector
<img width="1525" height="709" alt="image" src="https://github.com/user-attachments/assets/2d404534-0c51-4954-aca0-7177e812cb72" />


---

## 🚀 How it works

```
📄 Upload PDF  →  🔪 Chunk (300 tokens)  →  🧮 Embed (384-dim)
      ↓
🗄️ FAISS Index  ←→  🔍 Query  →  📚 Top-5 Chunks  →  ✨ Gemini Answer
```

---

## ✅ Features

- 📄 **Upload any PDF** — research papers, reports, textbooks
- 🔪 **Smart chunking** — 300-token chunks with 50-token overlap
- 🧮 **MiniLM embeddings** — `all-MiniLM-L6-v2` (384 dimensions)
- 🗄️ **FAISS vector store** — fast similarity search with `IndexFlatL2`
- ✨ **Gemini 2.5 Flash** — answers strictly from retrieved context
- 💬 **Chat interface** — full conversation history per session
- 🔍 **Context inspector** — toggle to see exactly which chunks were used
- 📊 **Session stats** — live chunk count, dimensions, top-k, page count

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| PDF parsing | pypdf |
| Text splitting | LangChain RecursiveCharacterTextSplitter |
| Embeddings | sentence-transformers / all-MiniLM-L6-v2 |
| Vector store | FAISS (IndexFlatL2) |
| LLM | Google Gemini 2.5 Flash via LangChain |
| Environment | python-dotenv |

---

## ⚡ Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/DevDhairya102/Multi-Document-RAG-System.git
cd Multi-Document-RAG-System
```

### 2. Create virtual environment
```bash
python -m venv venv

# Mac/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your Google API key
Create a `.env` file in the root:
```
GOOGLE_API_KEY=your_google_api_key_here
```
Get your free key at → https://aistudio.google.com/app/apikey

### 5. Run the app
```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501** 🎉

---

## 📁 Project Structure

```
Multi-Document-RAG-System/
│
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── .env                    # API key (never commit this)
├── .gitignore              # Ignores .env and cache files
└── README.md               # You are here
```

---

## 🔧 Configuration

| Parameter | Default | Description |
|---|---|---|
| chunk_size | 300 | Tokens per chunk |
| chunk_overlap | 50 | Overlap between consecutive chunks |
| top_k | 5 | Number of chunks retrieved per query |
| Embedding model | all-MiniLM-L6-v2 | 384-dim sentence embeddings |
| LLM | gemini-2.5-flash | Generation model |

---

## 💡 Example Questions to Ask

| Paper | Question |
|---|---|
| Attention Is All You Need | "What is the self-attention mechanism?" |
| BERT | "How is BERT pre-trained?" |
| Any ML paper | "What dataset was used?" |
| Any paper | "Summarize the key findings" |
| Any paper | "What models were compared?" |

---

## 📄 Good papers to try

- **Attention Is All You Need** → https://arxiv.org/pdf/1706.03762
- **BERT** → https://arxiv.org/pdf/1810.04805
- **GPT-3** → https://arxiv.org/pdf/2005.14165
- **XGBoost** → https://arxiv.org/pdf/1603.02754
- **ResNet** → https://arxiv.org/pdf/1512.03385

---

## 🌐 Deploy to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. In **Advanced Settings → Secrets**, add:
```toml
GOOGLE_API_KEY = "your_api_key_here"
```
5. Click **Deploy** — your app will be live in minutes!

---

## 🔐 Security

- ✅ `.env` is in `.gitignore` — API key never pushed to GitHub
- ✅ Streamlit Cloud uses encrypted secrets panel
- ✅ Gemini answers only from retrieved context — no hallucination beyond the paper

---

## 📜 License

MIT License — free to use, modify, and distribute.

---

## 🙋 Author

**Dhairya** · [@DevDhairya102](https://github.com/DevDhairya102)

---

<p align="center">Built with ❤️ using Streamlit, FAISS, and Google Gemini</p>
