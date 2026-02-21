"""
app.py
──────
Streamlit UI only.
All RAG logic  → rag_pipeline.py
All CSS styles → styles.css
"""

import streamlit as st
import requests
from pathlib import Path
from rag_pipeline import (
    get_embeddings,
    load_pdfs,
    chunk_documents,
    build_vectorstore,
    build_rag_chain,
    run_query
)

# ── Page Config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Local RAG Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Load CSS ─────────────────────────────────────────────────────
def load_css():
    css_path = Path(__file__).parent / "styles.css"
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ── Session State ────────────────────────────────────────────────
for key, default in {
    "chat_history": [],
    "vectorstore":  None,
    "chain":        None,
    "processed":    False,
    "stats":        {},
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── Cache Embeddings ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def cached_embeddings():
    return get_embeddings()

# ── Get Ollama Models ────────────────────────────────────────────
@st.cache_data(ttl=30)
def get_ollama_models():
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=3)
        if resp.status_code == 200:
            return [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        pass
    return []

# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 Local RAG Chatbot")
    st.markdown("*100% Free · Runs Offline · No API Keys*")
    st.markdown("---")

    # Ollama status check
    models = get_ollama_models()
    if models:
        st.success(f"✅ Ollama running — {len(models)} model(s) found")
        selected_model = st.selectbox(
            "🤖 Choose LLM", models,
            help="Pull models via: ollama pull llama3.2"
        )
    else:
        st.error("❌ Ollama not running")
        st.markdown("""
        **Fix:**
        1. Download [ollama.com](https://ollama.com)
        2. Run in terminal:
        ```
        ollama serve
        ollama pull llama3.2
        ```
        """)
        selected_model = None

    st.markdown("---")
    st.markdown("## 📄 Upload PDFs")
    uploaded_files = st.file_uploader(
        "Drop PDFs here",
        type=["pdf"],
        accept_multiple_files=True
    )

    st.markdown("#### ⚙️ RAG Settings")
    chunk_size    = st.slider("Chunk Size",    200, 1500, 600, 50)
    chunk_overlap = st.slider("Chunk Overlap",   0,  300,  80, 10)
    top_k         = st.slider("Top-K Chunks",    1,   10,   4)

    process_btn = st.button("🚀 Process & Index PDFs", use_container_width=True)

    if st.session_state.processed and st.session_state.stats:
        st.markdown("---")
        st.markdown("### 📊 Index Stats")
        s = st.session_state.stats
        c1, c2 = st.columns(2)
        c1.metric("PDFs",   s.get("pdfs",   0))
        c1.metric("Pages",  s.get("pages",  0))
        c2.metric("Chunks", s.get("chunks", 0))
        c2.metric("Model",  s.get("model",  "—")[:10])

    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("---")
    st.markdown("""
    **Stack:**
    🦙 Ollama · 🔗 LangChain
    🗄️ ChromaDB · 🤗 HuggingFace
    🌐 Streamlit
    """)

# ── Process PDFs ─────────────────────────────────────────────────
if process_btn:
    if not selected_model:
        st.error("⚠️ Ollama is not running. Please start Ollama first.")
    elif not uploaded_files:
        st.error("⚠️ Please upload at least one PDF.")
    else:
        with st.spinner("📖 Reading PDFs..."):
            all_docs = load_pdfs(uploaded_files)

        with st.spinner("✂️ Chunking text..."):
            chunks = chunk_documents(all_docs, chunk_size, chunk_overlap)

        with st.spinner("🧠 Creating embeddings (first run downloads ~90MB)..."):
            embeddings  = cached_embeddings()
            vectorstore = build_vectorstore(chunks, embeddings)

        with st.spinner(f"🔗 Building RAG chain with {selected_model}..."):
            chain = build_rag_chain(vectorstore, selected_model, top_k)

        st.session_state.vectorstore = vectorstore
        st.session_state.chain       = chain
        st.session_state.processed   = True
        st.session_state.chat_history = []
        st.session_state.stats = {
            "pdfs":   len(uploaded_files),
            "pages":  len(all_docs),
            "chunks": len(chunks),
            "model":  selected_model
        }
        st.success(f"✅ Done! {len(chunks)} chunks indexed. Start chatting!")
        st.rerun()

# ── Hero Header ──────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🧠 Local RAG Chatbot</h1>
    <p>Chat with your PDFs · 100% Free · Runs on your Machine · No Internet Needed</p>
    <div style="margin-top:1rem">
        <span class="badge">🦙 Ollama LLM</span>
        <span class="badge">🔗 LangChain</span>
        <span class="badge">🗄️ ChromaDB</span>
        <span class="badge">🤗 HuggingFace Embeddings</span>
        <span class="badge">🌐 Streamlit</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Chat Interface ───────────────────────────────────────────────
if st.session_state.processed:
    st.markdown("### 💬 Chat with your Documents")

    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="user-msg">
                <strong>🧑 You:</strong><br>{msg["content"]}
            </div>""", unsafe_allow_html=True)
        else:
            chips = "".join(
                f'<span class="source-chip">📄 {s}</span>'
                for s in msg.get("sources", [])
            )
            sources_html = f"<br><br><strong>📚 Sources:</strong> {chips}" if chips else ""
            st.markdown(f"""
            <div class="bot-msg">
                <strong>🤖 Assistant ({st.session_state.stats.get('model','')}):</strong><br>
                {msg["content"]}{sources_html}
            </div>""", unsafe_allow_html=True)

    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        with col1:
            user_input = st.text_input(
                "Ask anything about your PDFs...",
                placeholder="e.g. What are the key findings?",
                label_visibility="collapsed"
            )
        with col2:
            submit = st.form_submit_button("Send ➤", use_container_width=True)

    if submit and user_input.strip():
        with st.spinner("🤔 Thinking... (local LLM may take 10–30 sec)"):
            try:
                answer, sources = run_query(st.session_state.chain, user_input)
                st.session_state.chat_history.append({"role": "user",      "content": user_input})
                st.session_state.chat_history.append({"role": "assistant", "content": answer, "sources": sources})
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error: {e}")
                st.info("Make sure Ollama is still running: ollama serve")

    if st.session_state.chat_history:
        with st.expander("🔍 View Retrieved Chunks (last query)"):
            try:
                last_q = next(
                    (m["content"] for m in reversed(st.session_state.chat_history)
                     if m["role"] == "user"), None
                )
                if last_q:
                    docs = st.session_state.vectorstore.similarity_search(last_q, k=top_k)
                    for i, doc in enumerate(docs):
                        src  = doc.metadata.get("source_file", "Unknown")
                        page = doc.metadata.get("page", "?")
                        st.markdown(f"**Chunk {i+1}** — 📄 `{src}` | Page `{page}`")
                        st.text_area("", doc.page_content[:600], height=110,
                                     disabled=True, label_visibility="collapsed",
                                     key=f"chunk_{i}")
            except Exception:
                pass

# ── Welcome Screen ───────────────────────────────────────────────
else:
    c1, c2, c3, c4 = st.columns(4)
    cards = [
        ("⬇️", "Step 1", "Install Ollama & run <code>ollama pull llama3.2</code>"),
        ("📄", "Step 2", "Upload your PDF files using the sidebar"),
        ("🚀", "Step 3", "Click <strong>Process & Index PDFs</strong>"),
        ("💬", "Step 4", "Ask any question about your documents!"),
    ]
    for col, (icon, title, desc) in zip([c1, c2, c3, c4], cards):
        with col:
            st.markdown(f"""
            <div class="step-card">
                <div style="font-size:2rem">{icon}</div>
                <h3>{title}</h3>
                <p style="font-size:0.85rem;color:#555">{desc}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🏗️ How It Works")
    st.markdown("""
    <div class="arch-box">
PDF Upload ──► PyPDFLoader ──► RecursiveCharacterTextSplitter
                                          │
                                          ▼
                          HuggingFace Embeddings (local, free)
                          sentence-transformers/all-MiniLM-L6-v2
                                          │
                                          ▼
                               ChromaDB Vector Store
                               (stored locally on disk)
                                          │
         ┌────────────────────────────────┘
         │
User Query ──► Embed Query ──► Similarity Search ──► Top-K Chunks
                                                           │
                                                           ▼
                                         LangChain Prompt Builder
                                    [System + Context + Chat History]
                                                           │
                                                           ▼
                                      Ollama (Llama 3.2 / Mistral)
                                        Runs 100% on your CPU/GPU
                                                           │
                                                           ▼
                                     Answer + Source Citations
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 💡 Recommended Free Models via Ollama")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown('<div class="stat-card"><h4>🦙 llama3.2</h4><p>Best all-rounder<br><code>ollama pull llama3.2</code><br>~2GB</p></div>', unsafe_allow_html=True)
    with m2:
        st.markdown('<div class="stat-card"><h4>🌬️ mistral</h4><p>Fast & accurate<br><code>ollama pull mistral</code><br>~4GB</p></div>', unsafe_allow_html=True)
    with m3:
        st.markdown('<div class="stat-card"><h4>💎 gemma2</h4><p>Google open model<br><code>ollama pull gemma2</code><br>~5GB</p></div>', unsafe_allow_html=True)