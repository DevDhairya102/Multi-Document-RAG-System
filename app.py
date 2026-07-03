import streamlit as st
import numpy as np
import faiss
import os
import time
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Research RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500&display=swap');

:root {
    --sand: #b8aa97;
    --stone: #c3c5bc;
    --mist: #d8dbd2;
    --cream: #f5f2e3;
    --sand-dark: #7a6e62;
    --sand-mid: #9a8e81;
    --text-main: #3a342c;
    --text-muted: #7a7068;
    --text-hint: #a09890;
    --border: #d0cac1;
    --border-light: #e2ddd6;
    --sidebar-bg: #ede9de;
    --main-bg: #faf8f2;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: var(--text-main);
}

.stApp {
    background-color: var(--cream);
}

section[data-testid="stSidebar"] {
    background-color: var(--sidebar-bg) !important;
    border-right: 0.5px solid var(--border);
}

section[data-testid="stSidebar"] > div {
    background-color: var(--sidebar-bg) !important;
}

.stTextArea textarea, .stTextInput input {
    background-color: #ede9de !important;
    border: 0.5px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text-main) !important;
    font-family: 'Inter', sans-serif !important;
}

.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: var(--sand) !important;
    box-shadow: none !important;
}

.stButton > button {
    background-color: var(--sand) !important;
    color: #faf8f2 !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 500 !important;
    transition: background 0.15s !important;
}

.stButton > button:hover {
    background-color: var(--sand-dark) !important;
    color: #faf8f2 !important;
}

.stFileUploader {
    background-color: #f5f2e8 !important;
    border: 0.5px dashed var(--sand) !important;
    border-radius: 10px !important;
}

.metric-card {
    background: #ede9de;
    border-radius: 10px;
    padding: 12px 16px;
    text-align: center;
    border: 0.5px solid var(--border);
}

.metric-val {
    font-size: 22px;
    font-weight: 500;
    color: var(--text-main);
    line-height: 1.2;
}

.metric-lbl {
    font-size: 11px;
    color: var(--text-hint);
    margin-top: 2px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

.chat-bubble-user {
    background: var(--sand);
    color: #faf8f2;
    padding: 10px 14px;
    border-radius: 12px 12px 4px 12px;
    font-size: 14px;
    line-height: 1.6;
    max-width: 78%;
    margin-left: auto;
    margin-bottom: 4px;
}

.chat-bubble-ai {
    background: #ede9de;
    color: var(--text-main);
    padding: 10px 14px;
    border-radius: 12px 12px 12px 4px;
    font-size: 14px;
    line-height: 1.6;
    max-width: 78%;
    margin-bottom: 4px;
}

.chat-meta {
    font-size: 11px;
    color: var(--text-hint);
    margin-top: 4px;
    margin-bottom: 14px;
}

.chip {
    display: inline-block;
    font-size: 11px;
    background: #e2ddd6;
    color: var(--text-muted);
    padding: 3px 9px;
    border-radius: 20px;
    border: 0.5px solid var(--border);
    margin-right: 5px;
}

.section-label {
    font-size: 11px;
    font-weight: 500;
    color: var(--text-hint);
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-bottom: 8px;
}

.context-box {
    background: #f0ede4;
    border: 0.5px solid var(--border);
    border-radius: 10px;
    padding: 12px 14px;
    font-size: 12.5px;
    color: var(--text-muted);
    line-height: 1.6;
    margin-top: 8px;
    font-family: 'Inter', sans-serif;
}

.doc-pill {
    background: #e2ddd6;
    border: 0.5px solid var(--border);
    border-radius: 8px;
    padding: 7px 10px;
    margin-bottom: 4px;
    font-size: 12.5px;
    color: var(--text-main);
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.badge {
    font-size: 10px;
    background: var(--stone);
    color: var(--sand-dark);
    padding: 1px 8px;
    border-radius: 10px;
}

div[data-testid="stExpander"] {
    border: 0.5px solid var(--border) !important;
    border-radius: 10px !important;
    background: #f0ede4 !important;
}

.stProgress > div > div {
    background-color: var(--sand) !important;
}

hr {
    border-color: var(--border-light) !important;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text, len(reader.pages)


def build_index(text, model):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )
    chunks = splitter.split_text(text)
    embeddings = model.encode(chunks, show_progress_bar=False)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return chunks, index


def retrieve_context(query, model, chunks, index, top_k=5):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    context = ""
    retrieved = []
    for i, idx in enumerate(indices[0]):
        chunk = chunks[idx]
        context += chunk + "\n\n"
        retrieved.append({"chunk": chunk, "distance": float(distances[0][i])})
    return context, retrieved


def ask_gemini(context, query):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    prompt = f"""You are a research paper analysis assistant.

Answer ONLY using the provided context.

If the answer is not found in the context, say:
"The information is not available in the retrieved sections."

Give a concise, academic answer.

Context:
{context}

Question:
{query}
"""
    response = llm.invoke(prompt)
    return response.content


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "index" not in st.session_state:
    st.session_state.index = None
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None
if "page_count" not in st.session_state:
    st.session_state.page_count = 0
if "show_context" not in st.session_state:
    st.session_state.show_context = False

model = load_embedding_model()

with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;padding-bottom:16px;border-bottom:0.5px solid var(--border);margin-bottom:16px">
        <div style="width:32px;height:32px;background:var(--sand);border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:16px">🧠</div>
        <div>
            <div style="font-size:14px;font-weight:500;color:var(--text-main)">Research RAG</div>
            <div style="font-size:11px;color:var(--text-muted)">Paper assistant</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Upload paper</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drop a PDF",
        type=["pdf"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        if st.session_state.pdf_name != uploaded_file.name:
            with st.spinner("Indexing paper…"):
                text, pages = extract_text_from_pdf(uploaded_file)
                chunks, index = build_index(text, model)
                st.session_state.chunks = chunks
                st.session_state.index = index
                st.session_state.pdf_name = uploaded_file.name
                st.session_state.page_count = pages
                st.session_state.chat_history = []
                st.success("Ready!")

    st.markdown("<br>", unsafe_allow_html=True)

    if st.session_state.pdf_name:
        st.markdown('<div class="section-label">Indexed paper</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="doc-pill">
            <span>📄 {st.session_state.pdf_name}</span>
            <span class="badge">{len(st.session_state.chunks) if st.session_state.chunks else 0}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Session stats</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{len(st.session_state.chunks) if st.session_state.chunks else "—"}</div>
            <div class="metric-lbl">Chunks</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">384</div>
            <div class="metric-lbl">Dims</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">5</div>
            <div class="metric-lbl">Top-k</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{st.session_state.page_count if st.session_state.page_count else "—"}</div>
            <div class="metric-lbl">Pages</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Options</div>', unsafe_allow_html=True)
    st.session_state.show_context = st.toggle("Show retrieved context", value=st.session_state.show_context)

    if st.session_state.chat_history:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑 Clear chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

st.markdown("""
<div style="display:flex;align-items:center;justify-content:space-between;padding-bottom:14px;border-bottom:0.5px solid var(--border-light);margin-bottom:20px">
    <div style="font-size:16px;font-weight:500;color:var(--text-main)">
        """ + (f"📄 {st.session_state.pdf_name}" if st.session_state.pdf_name else "Research RAG — Paper Q&A") + """
    </div>
    <div style="font-size:11px;background:#e2ddd6;color:var(--text-muted);padding:4px 12px;border-radius:20px">
        ✨ Gemini 2.5 Flash
    </div>
</div>
""", unsafe_allow_html=True)

if not st.session_state.pdf_name:
    st.markdown("""
    <div style="text-align:center;padding:60px 20px;color:var(--text-hint)">
        <div style="font-size:48px;margin-bottom:16px">📄</div>
        <div style="font-size:15px;font-weight:500;color:var(--text-muted);margin-bottom:8px">Upload a PDF to get started</div>
        <div style="font-size:13px">Drop any research paper in the sidebar — it will be chunked, embedded, and indexed automatically.</div>
    </div>
    """, unsafe_allow_html=True)
else:
    if not st.session_state.chat_history:
        st.markdown(f"""
        <div class="chat-bubble-ai">
            Hello! I've indexed <strong>{st.session_state.pdf_name}</strong> into <strong>{len(st.session_state.chunks)}</strong> chunks using MiniLM-L6-v2 embeddings and stored them in a FAISS index. Ask me anything about the paper.
        </div>
        <div class="chat-meta">
            <span class="chip">🗄 FAISS ready</span>
            <span class="chip">✅ {len(st.session_state.chunks)} chunks indexed</span>
        </div>
        """, unsafe_allow_html=True)

    for entry in st.session_state.chat_history:
        st.markdown(f'<div class="chat-bubble-user">{entry["question"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="chat-bubble-ai">{entry["answer"]}</div>', unsafe_allow_html=True)
        avg_dist = sum(r["distance"] for r in entry["retrieved"]) / len(entry["retrieved"])
        st.markdown(f"""
        <div class="chat-meta">
            <span class="chip">📚 {len(entry["retrieved"])} chunks retrieved</span>
            <span class="chip">📏 Avg dist: {avg_dist:.3f}</span>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.show_context:
            with st.expander("View retrieved context"):
                for i, r in enumerate(entry["retrieved"]):
                    st.markdown(f'<div class="context-box"><strong>Chunk {i+1}</strong> · distance {r["distance"]:.4f}<br><br>{r["chunk"]}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Quick questions</div>', unsafe_allow_html=True)
    qcols = st.columns(4)
    quick_qs = ["Summarize the abstract", "What dataset was used?", "Key findings", "Models compared"]
    selected_quick = None
    for i, q in enumerate(quick_qs):
        with qcols[i]:
            if st.button(q, use_container_width=True, key=f"quick_{i}"):
                selected_quick = q

    st.markdown("<br>", unsafe_allow_html=True)
    with st.form("query_form", clear_on_submit=True):
        col_input, col_btn = st.columns([5, 1])
        with col_input:
            user_query = st.text_area(
                "Question",
                placeholder="Ask a question about the paper…",
                label_visibility="collapsed",
                height=72,
                key="query_input"
            )
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("Send ↑", use_container_width=True)

    final_query = selected_quick or (user_query.strip() if submitted and user_query.strip() else None)

    if final_query:
        with st.spinner("Retrieving context and generating answer…"):
            context, retrieved = retrieve_context(
                final_query, model,
                st.session_state.chunks,
                st.session_state.index
            )
            answer = ask_gemini(context, final_query)

        st.session_state.chat_history.append({
            "question": final_query,
            "answer": answer,
            "retrieved": retrieved
        })
        st.rerun()
