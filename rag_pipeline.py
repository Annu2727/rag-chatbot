import tempfile
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


def load_pdfs(uploaded_files):
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


def chunk_documents(docs, chunk_size=600, chunk_overlap=80):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_documents(docs)


def build_vectorstore(chunks, embeddings):
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="local_rag_collection"
    )


def build_rag_chain(vectorstore, groq_api_key, model_name="llama-3.1-8b-instant", top_k=4):
    llm = ChatGroq(
        model=model_name,
        groq_api_key=groq_api_key,
        temperature=0.1,
        max_tokens=1024,
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k}
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that answers questions based on PDF documents.
Use the following context to answer. If unsure, say you don't know.

Context:
{context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])

    # Store retriever alongside llm and prompt
    return {"llm": llm, "retriever": retriever, "prompt": prompt}


def run_query(chain_dict, question, chat_history=[]):
    llm       = chain_dict["llm"]
    retriever = chain_dict["retriever"]
    prompt    = chain_dict["prompt"]

    # Convert chat history to LangChain messages
    history = []
    for msg in chat_history:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history.append(AIMessage(content=msg["content"]))

    # Retrieve relevant chunks
    docs    = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    # Build and invoke the chain
    messages = prompt.format_messages(
        context=context,
        chat_history=history,
        question=question
    )
    response = llm.invoke(messages)

    # Extract answer string safely
    if hasattr(response, "content"):
        answer = response.content
    elif isinstance(response, dict):
        answer = response.get("content", str(response))
    else:
        answer = str(response)

    # Extract source filenames
    sources = list(set([
        doc.metadata.get("source_file", "Unknown")
        for doc in docs
    ]))

    return answer, sources