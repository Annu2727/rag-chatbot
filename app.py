import streamlit as st
from pathlib import Path
from datetime import datetime
from groq import Groq
from rag_pipeline import (
    get_embeddings, load_pdfs, chunk_documents,
    build_vectorstore, build_rag_chain, run_query
)

st.set_page_config(page_title="AI Assistant", page_icon="🤖", layout="centered")

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

# ── Session State ─────────────────────────────────────────────────
for key, val in {
    "all_sessions": {},          # {id: {title, messages, pdf_names}}
    "current_session": None,
    "all_vectorstores": [],
    "rag_chain": None,
    "pdf_processed": False,
    "pdf_names": [],
    "show_uploader": False,
    "uploaded_file_names": set(),
    "show_history": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

@st.cache_resource(show_spinner=False)
def cached_embeddings():
    return get_embeddings()

def new_session():
    sid = datetime.now().strftime("%Y%m%d%H%M%S%f")
    st.session_state.all_sessions[sid] = {
        "title": "New Chat",
        "messages": [],
        "pdf_names": []
    }
    st.session_state.current_session = sid
    st.session_state.messages = []
    st.session_state.pdf_processed = False
    st.session_state.all_vectorstores = []
    st.session_state.rag_chain = None
    st.session_state.pdf_names = []
    st.session_state.show_uploader = False
    st.session_state.uploaded_file_names = set()

def save_current_session():
    sid = st.session_state.current_session
    if sid and sid in st.session_state.all_sessions:
        st.session_state.all_sessions[sid]["messages"]  = st.session_state.messages
        st.session_state.all_sessions[sid]["pdf_names"] = st.session_state.pdf_names
        if st.session_state.messages:
            first_user = next((m["content"] for m in st.session_state.messages if m["role"] == "user"), None)
            if first_user:
                st.session_state.all_sessions[sid]["title"] = first_user[:40] + ("..." if len(first_user) > 40 else "")

def load_session(sid):
    s = st.session_state.all_sessions[sid]
    st.session_state.current_session  = sid
    st.session_state.messages         = s["messages"]
    st.session_state.pdf_names        = s["pdf_names"]
    st.session_state.pdf_processed    = len(s["pdf_names"]) > 0
    st.session_state.show_history     = False
    st.session_state.all_vectorstores = []
    st.session_state.rag_chain        = None

# Init first session
if not st.session_state.current_session:
    new_session()

def chat_with_groq(messages):
    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in messages:
        history.append({"role": m["role"], "content": m["content"]})
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile", messages=history, temperature=0.7, max_tokens=2048
    )
    return resp.choices[0].message.content

def auto_summarize(pdf_names):
    question = f"Please provide a comprehensive summary of the uploaded document(s): {', '.join(pdf_names)}. Include key topics, main points, and important findings."
    answer, sources = run_query(st.session_state.rag_chain, question, [])
    return answer, sources

def process_pdfs(uploaded_pdf):
    truly_new = [f for f in uploaded_pdf if f.name not in st.session_state.uploaded_file_names]
    if not truly_new:
        return
    with st.spinner(f"📖 Reading {len(truly_new)} PDF(s)..."):
        all_docs = load_pdfs(truly_new)
    with st.spinner("✂️ Chunking..."):
        chunks = chunk_documents(all_docs, chunk_size=400, chunk_overlap=50)
    with st.spinner("🧠 Creating embeddings..."):
        embeddings = cached_embeddings()
        new_vs = build_vectorstore(chunks, embeddings)
    st.session_state.all_vectorstores.append(new_vs)
    with st.spinner("🔗 Building chain..."):
        rag_chain = build_rag_chain(new_vs, groq_api_key, "llama-3.3-70b-versatile", 5)
    st.session_state.rag_chain    = rag_chain
    st.session_state.pdf_processed = True
    st.session_state.show_uploader = False
    for f in truly_new:
        st.session_state.uploaded_file_names.add(f.name)
    st.session_state.pdf_names = list(st.session_state.uploaded_file_names)
    with st.spinner("✨ Generating summary..."):
        summary, sources = auto_summarize([f.name for f in truly_new])
    st.session_state.messages.append({
        "role": "assistant",
        "content": f"📄 **Analyzed: {', '.join([f.name for f in truly_new])}**\n\n{summary}",
        "sources": sources
    })
    save_current_session()
    st.rerun()

# ── TOPBAR ────────────────────────────────────────────────────────
col_hist, col_badge, col_new, col_clear = st.columns([1, 6, 1, 1])

with col_hist:
    history_label = "📋" if not st.session_state.show_history else "✖️"
    if st.button(history_label, help="Chat History", use_container_width=True):
        st.session_state.show_history = not st.session_state.show_history
        st.rerun()

with col_badge:
    if st.session_state.pdf_processed:
        names = ", ".join(st.session_state.pdf_names)
        st.markdown(f"<div class='mode-badge pdf-badge'>📄 {names}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='mode-badge'>🤖 AI Assistant · ML · DL · GenAI · Agentic AI</div>", unsafe_allow_html=True)

with col_new:
    if st.button("✏️", help="New Chat", use_container_width=True):
        save_current_session()
        new_session()
        st.rerun()

with col_clear:
    if st.button("🗑️", help="Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pdf_processed = False
        st.session_state.all_vectorstores = []
        st.session_state.rag_chain = None
        st.session_state.pdf_names = []
        st.session_state.show_uploader = False
        st.session_state.uploaded_file_names = set()
        save_current_session()
        st.rerun()

# ── HISTORY PANEL ─────────────────────────────────────────────────
if st.session_state.show_history:
    st.markdown("<div class='history-panel'>", unsafe_allow_html=True)
    st.markdown("#### 🕘 Chat History")
    if st.session_state.all_sessions:
        for sid, data in reversed(list(st.session_state.all_sessions.items())):
            is_active = sid == st.session_state.current_session
            icon  = "📄" if data["pdf_names"] else "💬"
            label = f"{icon} {data['title']}"
            col_btn, col_del = st.columns([9, 1])
            with col_btn:
                btn_style = "**" if is_active else ""
                if st.button(label, key=f"h_{sid}", use_container_width=True):
                    save_current_session()
                    load_session(sid)
                    st.rerun()
            with col_del:
                if st.button("✖", key=f"d_{sid}", use_container_width=True):
                    del st.session_state.all_sessions[sid]
                    if st.session_state.current_session == sid:
                        new_session()
                    st.rerun()
    else:
        st.markdown("<small style='color:#8e8ea0'>No history yet</small>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("---")

# ── WELCOME ───────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div class='welcome-wrap'>
        <div class='welcome-title'>What can I help you with?</div>
        <div class='welcome-sub'>Ask anything about ML, DL, GenAI, Agentic AI<br>or click 📎 to analyze a PDF</div>
    </div>
    """, unsafe_allow_html=True)

# ── CHAT MESSAGES ─────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"""
        <div class='user-msg'>
            <div class='avatar-user'>U</div>
            <div class='msg-content'>{msg["content"]}</div>
        </div>""", unsafe_allow_html=True)
    else:
        chips = ""
        if msg.get("sources"):
            chips = "<br><small>" + "".join(
                f'<span class="source-chip">📄 {s}</span>' for s in msg["sources"]
            ) + "</small>"
        content = msg["content"].replace("\n", "<br>")
        st.markdown(f"""
        <div class='bot-msg'>
            <div class='avatar-bot'>AI</div>
            <div class='msg-content'>{content}{chips}</div>
        </div>""", unsafe_allow_html=True)

if st.session_state.pdf_processed and st.session_state.messages:
    with st.expander("🔍 View Retrieved Chunks"):
        try:
            last_q = next((m["content"] for m in reversed(st.session_state.messages) if m["role"] == "user"), None)
            if last_q and st.session_state.all_vectorstores:
                docs = st.session_state.all_vectorstores[-1].similarity_search(last_q, k=4)
                for i, doc in enumerate(docs):
                    st.markdown(f"**Chunk {i+1}** — 📄 `{doc.metadata.get('source_file','?')}` | Page `{doc.metadata.get('page','?')}`")
                    st.text_area("", doc.page_content[:500], height=100, disabled=True, label_visibility="collapsed", key=f"chunk_{i}")
        except Exception:
            pass

st.markdown("<br>", unsafe_allow_html=True)

# ── PDF UPLOADER ──────────────────────────────────────────────────
if st.session_state.show_uploader:
    uploaded_pdf = st.file_uploader(
        "Upload PDF", type=["pdf"], accept_multiple_files=True,
        label_visibility="collapsed", key="pdf_uploader"
    )
    if uploaded_pdf:
        process_pdfs(uploaded_pdf)

# ── INPUT BAR ─────────────────────────────────────────────────────
col_attach, col_input, col_send = st.columns([1, 8, 1])
with col_attach:
    if st.button("📎", help="Attach PDF", use_container_width=True):
        st.session_state.show_uploader = not st.session_state.show_uploader
        st.rerun()
with col_input:
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Message", placeholder="Ask about ML, DL, GenAI, Agentic AI...", label_visibility="collapsed")
        submit = st.form_submit_button("Send", use_container_width=False)
with col_send:
    st.write("")

# ── HANDLE SUBMIT ─────────────────────────────────────────────────
if submit and user_input.strip():
    with st.spinner("Thinking..."):
        try:
            if st.session_state.pdf_processed and st.session_state.rag_chain:
                answer, sources = run_query(st.session_state.rag_chain, user_input.strip(), st.session_state.messages)
                st.session_state.messages.append({"role": "user",      "content": user_input.strip()})
                st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
            else:
                st.session_state.messages.append({"role": "user", "content": user_input.strip()})
                answer = chat_with_groq(st.session_state.messages)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            save_current_session()
            st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")