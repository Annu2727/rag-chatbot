import streamlit as st
from pathlib import Path
from datetime import datetime
from groq import Groq
from rag_pipeline import (
    get_embeddings, load_pdfs, chunk_documents,
    build_vectorstore, build_rag_chain, run_query
)

st.set_page_config(page_title="AI Assistant", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")

def load_css():
    css_path = Path(__file__).parent / "styles.css"
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
load_css()

groq_api_key = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=groq_api_key)

SYSTEM_PROMPT = """You are an expert AI Assistant specializing in:
- Machine Learning (supervised, unsupervised, reinforcement learning, algorithms, model evaluation)
- Deep Learning (CNNs, RNNs, LSTMs, Transformers, PyTorch, TensorFlow)
- Generative AI (LLMs, diffusion models, GANs, VAEs, prompt engineering, RAG, fine-tuning)
- Agentic AI (AI agents, tool use, ReAct, LangChain agents, AutoGPT, multi-agent systems)
- Data Analysis (pandas, numpy, EDA, visualization, statistics, feature engineering)
Give clear, structured, detailed answers with bullet points and code examples when helpful."""

# Session state
for key, val in {
    "all_sessions": {}, "current_session": None,
    "vectorstore": None, "rag_chain": None,
    "pdf_processed": False, "pdf_names": []
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

def new_session():
    sid = datetime.now().strftime("%Y%m%d%H%M%S%f")
    st.session_state.all_sessions[sid] = {"title": "New Chat", "messages": [], "mode": "chat"}
    st.session_state.current_session = sid
    st.session_state.pdf_processed = False
    st.session_state.vectorstore = None
    st.session_state.rag_chain = None
    st.session_state.pdf_names = []
    return sid

def get_session():
    if not st.session_state.current_session:
        new_session()
    return st.session_state.all_sessions[st.session_state.current_session]

def add_message(role, content, sources=None):
    s = get_session()
    msg = {"role": role, "content": content}
    if sources: msg["sources"] = sources
    s["messages"].append(msg)
    if role == "user" and s["title"] == "New Chat":
        s["title"] = content[:40] + ("..." if len(content) > 40 else "")

@st.cache_resource(show_spinner=False)
def cached_embeddings():
    return get_embeddings()

def chat_with_groq(messages):
    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in messages:
        history.append({"role": m["role"], "content": m["content"]})
    resp = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=history, temperature=0.7, max_tokens=2048)
    return resp.choices[0].message.content

# ── SIDEBAR ──────────────────────────────────────────────────────
with st.sidebar:
    if st.button("✏️  New Chat", use_container_width=True):
        new_session()
        st.rerun()
    st.markdown("---")
    if st.session_state.all_sessions:
        st.markdown("<div class='history-section-label'>Recent</div>", unsafe_allow_html=True)
        for sid, data in reversed(list(st.session_state.all_sessions.items())):
            icon = "📄" if data["mode"] == "pdf" else "💬"
            if st.button(f"{icon} {data['title']}", key=f"h_{sid}", use_container_width=True):
                st.session_state.current_session = sid
                st.rerun()
    else:
        st.markdown("<small style='color:#8e8ea0'>No chat history yet</small>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**📎 Attach PDF**")
    uploaded_pdf = st.file_uploader("PDF", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed", key="pdf_uploader")
    st.markdown("---")
    st.markdown("<small style='color:#8e8ea0'>⚡ Groq · Llama3 70B · LangChain · ChromaDB</small>", unsafe_allow_html=True)

# ── MAIN ─────────────────────────────────────────────────────────
session = get_session()

# Mode badge
if session["mode"] == "pdf":
    names = ", ".join(st.session_state.pdf_names) if st.session_state.pdf_names else "PDF"
    st.markdown(f"<div class='mode-badge pdf-badge'>📄 PDF Mode · {names}</div>", unsafe_allow_html=True)
else:
    st.markdown("<div class='mode-badge'>🤖 AI Assistant · ML · DL · GenAI · Agentic AI</div>", unsafe_allow_html=True)

# Welcome screen
if not session["messages"]:
    st.markdown("""
    <div class='welcome-wrap'>
        <div class='welcome-title'>What can I help you with?</div>
        <div class='welcome-sub'>Ask anything about ML, DL, GenAI, Agentic AI<br>or attach a PDF from the sidebar</div>
    </div>
    """, unsafe_allow_html=True)

# Chat messages
for msg in session["messages"]:
    if msg["role"] == "user":
        st.markdown(f"""
        <div class='user-msg'>
            <div class='avatar-user'>U</div>
            <div class='msg-content'>{msg["content"]}</div>
        </div>""", unsafe_allow_html=True)
    else:
        chips = ""
        if msg.get("sources"):
            chips = "<br><small>" + "".join(f'<span class="source-chip">📄 {s}</span>' for s in msg["sources"]) + "</small>"
        content = msg["content"].replace("\n", "<br>")
        st.markdown(f"""
        <div class='bot-msg'>
            <div class='avatar-bot'>AI</div>
            <div class='msg-content'>{content}{chips}</div>
        </div>""", unsafe_allow_html=True)

# Retrieved chunks
if session["mode"] == "pdf" and session["messages"]:
    with st.expander("🔍 View Retrieved Chunks"):
        try:
            last_q = next((m["content"] for m in reversed(session["messages"]) if m["role"] == "user"), None)
            if last_q and st.session_state.vectorstore:
                docs = st.session_state.vectorstore.similarity_search(last_q, k=4)
                for i, doc in enumerate(docs):
                    st.markdown(f"**Chunk {i+1}** — 📄 `{doc.metadata.get('source_file','?')}` | Page `{doc.metadata.get('page','?')}`")
                    st.text_area("", doc.page_content[:500], height=100, disabled=True, label_visibility="collapsed", key=f"chunk_{i}")
        except Exception:
            pass

st.markdown("---")

# ── INPUT BAR (at bottom naturally) ──────────────────────────────
with st.form("chat_form", clear_on_submit=True):
    col1, col2 = st.columns([8, 1])
    with col1:
        user_input = st.text_input("Message", placeholder="Ask about ML, DL, GenAI, Agentic AI...", label_visibility="collapsed")
    with col2:
        submit = st.form_submit_button("Send", use_container_width=True)

# ── PDF PROCESSING ────────────────────────────────────────────────
if uploaded_pdf and not st.session_state.pdf_processed:
    with st.spinner("📖 Reading PDFs..."):
        all_docs = load_pdfs(uploaded_pdf)
    with st.spinner("✂️ Chunking..."):
        chunks = chunk_documents(all_docs, chunk_size=800, chunk_overlap=100)
    with st.spinner("🧠 Creating embeddings..."):
        embeddings = cached_embeddings()
        vectorstore = build_vectorstore(chunks, embeddings)
    with st.spinner("🔗 Building chain..."):
        rag_chain = build_rag_chain(vectorstore, groq_api_key, "llama3-70b-8192", 5)
    st.session_state.vectorstore = vectorstore
    st.session_state.rag_chain = rag_chain
    st.session_state.pdf_processed = True
    st.session_state.pdf_names = [f.name for f in uploaded_pdf]
    session["mode"] = "pdf"
    st.success(f"✅ {len(chunks)} chunks indexed! Ask questions about your PDF.")
    st.rerun()

# ── HANDLE SUBMIT ─────────────────────────────────────────────────
if submit and user_input.strip():
    with st.spinner("Thinking..."):
        try:
            if session["mode"] == "pdf" and st.session_state.rag_chain:
                answer, sources = run_query(st.session_state.rag_chain, user_input.strip(), session["messages"])
                add_message("user", user_input.strip())
                add_message("assistant", answer, sources)
            else:
                add_message("user", user_input.strip())
                answer = chat_with_groq(session["messages"])
                add_message("assistant", answer)
            st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")