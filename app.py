import streamlit as st
import os
import tempfile
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.set_page_config(
    page_title="Local RAG Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .hero {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(48, 43, 99, 0.4);
    }
    .hero h1 { font-size: 2.2rem; margin: 0; font-weight: 700; }
    .hero p  { font-size: 1rem; opacity: 0.85; margin-top: 0.5rem; }
    .badge {
        display: inline-block;
        background: rgba(255,255,255,0.15);
        border: 1px solid rgba(255,255,255,0.25);
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        margin: 3px;
        color: white;
    }
    .step-card {
        background: #f8f9ff;
        border: 1px solid #e0e4ff;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        height: 100%;
    }
    .step-card h3 { color: #302b63; margin-bottom: 0.4rem; }
    .user-msg {
        background: #2d2b55;
        border-left: 4px solid #4f46e5;
        padding: 0.9rem 1.2rem;
        border-radius: 0 10px 10px 0;
        margin: 0.6rem 0;
        color: #ffffff;
    }
    .bot-msg {
        background: #1e1e2e;
        border-left: 4px solid #16a34a;
        padding: 0.9rem 1.2rem;
        border-radius: 0 10px 10px 0;
        margin: 0.6rem 0;
        color: #ffffff;
    }
    .source-chip {
        display: inline-block;
        background: #4f46e5;
        color: white;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.72rem;
        margin: 2px;
    }
    .stat-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 0.8rem;
        text-align: center;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    .arch-box {
        background: #0f0c29;
        color: #a5f3fc;
        padding: 1.5rem;
        border-radius: 12px;
        font-family: monospace;
        font-size: 0.85rem;
        line-height: 1.8;
        white-space: pre;
    }
    .stButton > button {
        background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    .stButton > button:hover {
        opacity: 0.9 !important;
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)

for key, default in {
    "chat_history": [],
    "vectorstore": None,
    "chain": None,
    "processed": False,
    "stats": {},
    "ollama_models": []
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

@st.cache_data(ttl=30)
def get_ollama_models():
    try:
        import requests
        resp = requests.get("http://localhost:11434/api/tags", timeout=3)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            return [m["name"] for m in models]
    except Exception:
        pass
    return []

@st.cache_resource(show_spinner=False)
def load_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

with st.sidebar:
    st.markdown("## 🧠 Local RAG Chatbot")
    st.markdown("*100% Free · Runs Offline · No API Keys*")
    st.markdown("---")

    models = get_ollama_models()
    if models:
        st.success(f"✅ Ollama running — {len(models)} model(s) found")
        selected_model = st.selectbox("🤖 Choose LLM", models,
                                      help="Models pulled via `ollama pull <name>`")
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
        c2.metric("Model",  s.get("model",  "—")[:8])

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

if process_btn:
    if not selected_model:
        st.error("⚠️ Ollama is not running. Please start Ollama first.")
    elif not uploaded_files:
        st.error("⚠️ Please upload at least one PDF.")
    else:
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_community.vectorstores import Chroma
        from langchain_ollama import ChatOllama
        
        from langchain.chains import ConversationalRetrievalChain
        from langchain.memory import ConversationBufferMemory

        with st.spinner("📖 Reading PDFs..."):
            all_docs = []
            tmp_dir  = tempfile.mkdtemp()
            for f in uploaded_files:
                tmp_path = Path(tmp_dir) / f.name
                tmp_path.write_bytes(f.read())
                loader = PyPDFLoader(str(tmp_path))
                docs   = loader.load()
                for doc in docs:
                    doc.metadata["source_file"] = f.name
                all_docs.extend(docs)

        with st.spinner("✂️ Chunking text..."):
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            chunks = splitter.split_documents(all_docs)

        with st.spinner("🧠 Creating embeddings (first run downloads ~90MB)..."):
            embeddings  = load_embeddings()
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                collection_name="local_rag_collection"
            )

        with st.spinner(f"🔗 Building RAG chain with {selected_model}..."):
            llm = ChatOllama(
                model=selected_model,
                temperature=0.1,
                num_predict=1024,
            )
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
            chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": top_k}
                ),
                memory=memory,
                return_source_documents=True,
                verbose=False
            )

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
                result  = st.session_state.chain({"question": user_input})
                answer  = result["answer"]
                sources = list(set([
                    doc.metadata.get("source_file",
                    doc.metadata.get("source", "Unknown"))
                    for doc in result.get("source_documents", [])
                ]))
                st.session_state.chat_history.append({"role": "user",      "content": user_input})
                st.session_state.chat_history.append({"role": "assistant", "content": answer, "sources": sources})
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error: {e}")
                st.info("Make sure Ollama is still running: `ollama serve`")

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

else:
    c1, c2, c3, c4 = st.columns(4)
    cards = [
        ("⬇️", "Step 1", "Install Ollama from ollama.com & run <code>ollama pull llama3.2</code>"),
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
        st.markdown("""
        <div class="stat-card">
            <h4>🦙 llama3.2</h4>
            <p>Best all-rounder<br><code>ollama pull llama3.2</code><br>~2GB</p>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown("""
        <div class="stat-card">
            <h4>🌬️ mistral</h4>
            <p>Fast & accurate<br><code>ollama pull mistral</code><br>~4GB</p>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown("""
        <div class="stat-card">
            <h4>💎 gemma2</h4>
            <p>Google's open model<br><code>ollama pull gemma2</code><br>~5GB</p>
        </div>""", unsafe_allow_html=True)