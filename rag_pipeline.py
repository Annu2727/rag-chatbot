"""
rag_pipeline.py
───────────────
All RAG logic: PDF loading, chunking, embedding, vector store, LLM chain.

Stack:
  - LLM        : Ollama (Llama 3.2) — runs locally, 100% free
  - Embeddings : HuggingFace all-MiniLM-L6-v2 — runs locally, free
  - Vector DB  : ChromaDB — runs locally, free
  - Framework  : LangChain — connects everything
"""

import tempfile
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory


# ── 1. Embedding Model ───────────────────────────────────────────
def get_embeddings():
    """
    Loads HuggingFace sentence-transformer embedding model.
    - Runs 100% locally on CPU
    - No API key needed
    - Downloads ~90MB on first run, cached after
    Model: sentence-transformers/all-MiniLM-L6-v2
    """
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


# ── 2. PDF Loader ────────────────────────────────────────────────
def load_pdfs(uploaded_files):
    """
    Accepts Streamlit uploaded file objects.
    Saves each to a temp folder, reads with PyPDFLoader.
    Tags every page with its source filename for citations.
    Returns: list of LangChain Document objects
    """
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

    return all_docs


# ── 3. Text Chunker ──────────────────────────────────────────────
def chunk_documents(docs, chunk_size=600, chunk_overlap=80):
    """
    Splits documents into small overlapping chunks.
    Uses RecursiveCharacterTextSplitter which splits on:
      paragraphs (\n\n) → lines (\n) → sentences (.) → words ( )
    This keeps meaningful text units together.
    Returns: list of smaller LangChain Document chunks
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_documents(docs)


# ── 4. Vector Store ──────────────────────────────────────────────
def build_vectorstore(chunks, embeddings):
    """
    Converts each text chunk into a vector using HuggingFace embeddings.
    Stores all vectors in ChromaDB (local in-memory database).
    ChromaDB enables fast similarity search — finds chunks
    most similar to the user's question.
    Returns: ChromaDB vectorstore object
    """
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="local_rag_collection"
    )


# ── 5. RAG Chain ─────────────────────────────────────────────────
def build_rag_chain(vectorstore, model_name, top_k=4):
    """
    Builds the full RAG pipeline using LangChain:

    Flow:
      User Question
        → Embed question with HuggingFace
        → Search ChromaDB for top-K similar chunks
        → Build prompt: [system + chunks + chat history + question]
        → Send to Ollama (local Llama 3.2)
        → Return answer + source documents

    Args:
        vectorstore : ChromaDB vectorstore
        model_name  : Ollama model e.g. 'llama3.2', 'mistral'
        top_k       : how many chunks to retrieve per query

    Returns: LangChain ConversationalRetrievalChain
    """

    # LLM — Ollama runs the model locally on your machine
    llm = ChatOllama(
        model=model_name,
        temperature=0.1,   # low = more factual, less creative
        num_predict=1024,  # max tokens in response
    )

    # Memory — stores conversation turns so follow-up questions work
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    # Chain — wires retriever + memory + LLM together
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

    return chain


# ── 6. Query Runner ──────────────────────────────────────────────
def run_query(chain, question):
    """
    Sends user question through the RAG chain.
    Returns:
        answer  : str  — LLM generated answer
        sources : list — unique source PDF filenames cited
    """
    result  = chain({"question": question})
    answer  = result["answer"]
    sources = list(set([
        doc.metadata.get("source_file", doc.metadata.get("source", "Unknown"))
        for doc in result.get("source_documents", [])
    ]))
    return answer, sources