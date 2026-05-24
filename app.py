# app.py
import os
import json
import uuid
import pickle
from datetime import datetime
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, UnstructuredExcelLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder

# ── Session ID (must be first) ───────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

# ── Config ───────────────────────────────────────────
BASE_DIR     = os.path.join(os.path.expanduser("~"), ".docchat")
FAISS_PATH   = os.path.join(BASE_DIR, "faiss_db")
CHUNKS_PATH  = os.path.join(BASE_DIR, "faiss_db", "chunks.pkl")
PARENTS_PATH = os.path.join(BASE_DIR, "faiss_db", "parents.pkl")
CHATS_DIR    = os.path.join(BASE_DIR, "rag_chats", st.session_state.session_id)

os.makedirs(FAISS_PATH, exist_ok=True)
os.makedirs(CHATS_DIR, exist_ok=True)

try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except:
    from dotenv import load_dotenv
    load_dotenv()
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ── Page setup ───────────────────────────────────────
st.set_page_config(
    page_title="DocChat AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────
st.markdown("""
<style>
    #MainMenu, footer, header {visibility: hidden;}
    * {box-sizing: border-box;}

    .stApp {
        background: #0a0a0f;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .main .block-container {
        padding: 1rem 1.5rem 6rem 1.5rem;
        max-width: 860px;
        margin: 0 auto;
    }

    h1 {
        font-size: clamp(1.4rem, 4vw, 2rem) !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px;
        background: linear-gradient(135deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0 !important;
    }

    .stApp p, .stApp .stCaption {
        color: #6b7280 !important;
        font-size: 0.8rem !important;
    }

    [data-testid="stSidebar"] {
        background: #0f0f17 !important;
        border-right: 1px solid #1f1f2e;
        padding: 1rem 0.8rem;
    }
    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }

    [data-testid="stSidebar"] h3 {
        font-size: 0.7rem !important;
        font-weight: 600 !important;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: #4b5563 !important;
        margin: 1.2rem 0 0.5rem 0 !important;
    }

    [data-testid="stFileUploader"] {
        background: #13131f;
        border: 1.5px dashed #2d2d44;
        border-radius: 12px;
        padding: 0.8rem;
        transition: border-color 0.2s;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #a78bfa;
    }

    .stButton > button {
        background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.2s !important;
        width: 100%;
        letter-spacing: 0.3px;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 20px rgba(124, 58, 237, 0.35) !important;
    }
    .stButton > button:active {
        transform: translateY(0) !important;
    }

    [data-testid="stToggle"] {
        margin: 0.3rem 0;
    }

    hr {
        border-color: #1f1f2e !important;
        margin: 0.8rem 0 !important;
    }

    [data-testid="stSidebar"] .stButton > button {
        background: transparent !important;
        border: 1px solid #1f1f2e !important;
        border-radius: 8px !important;
        color: #9ca3af !important;
        font-size: 0.78rem !important;
        font-weight: 400 !important;
        text-align: left !important;
        padding: 0.4rem 0.7rem !important;
        margin: 0.15rem 0 !important;
        transition: all 0.15s !important;
        box-shadow: none !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: #1a1a2e !important;
        border-color: #7c3aed !important;
        color: #e2e8f0 !important;
        transform: none !important;
        box-shadow: none !important;
    }

    [data-testid="stChatMessage"] {
        background: #13131f;
        border: 1px solid #1f1f2e;
        border-radius: 16px;
        padding: 0.8rem 1rem;
        margin: 0.4rem 0;
    }

    [data-testid="stChatMessageContent"] {
        font-size: clamp(0.85rem, 2.5vw, 0.95rem) !important;
        line-height: 1.7 !important;
        color: #e2e8f0 !important;
    }

    [data-testid="stChatInput"] {
        background: #13131f !important;
        border: 1.5px solid #2d2d44 !important;
        border-radius: 14px !important;
        padding: 0.6rem 1rem !important;
        transition: border-color 0.2s !important;
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: #7c3aed !important;
        box-shadow: 0 0 0 3px rgba(124, 58, 237, 0.1) !important;
    }
    [data-testid="stChatInput"] textarea {
        font-size: 16px !important;
        color: #e2e8f0 !important;
        background: transparent !important;
    }

    [data-testid="stExpander"] {
        background: #0f0f17 !important;
        border: 1px solid #1f1f2e !important;
        border-radius: 10px !important;
        margin-top: 0.4rem !important;
    }
    [data-testid="stExpander"] summary {
        font-size: 0.78rem !important;
        color: #6b7280 !important;
        font-weight: 500 !important;
    }

    [data-testid="stAlert"] {
        background: #13131f !important;
        border: 1px solid #1f1f2e !important;
        border-radius: 12px !important;
        color: #9ca3af !important;
        font-size: 0.85rem !important;
    }

    .stCaption {
        font-size: 0.72rem !important;
        color: #4b5563 !important;
        margin-top: 0.2rem !important;
    }

    [data-testid="stSpinner"] {
        color: #7c3aed !important;
    }

    ::-webkit-scrollbar {width: 4px; height: 4px;}
    ::-webkit-scrollbar-track {background: #0a0a0f;}
    ::-webkit-scrollbar-thumb {
        background: #2d2d44;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {background: #7c3aed;}

    @media (max-width: 768px) {
        .main .block-container {
            padding: 0.8rem 0.8rem 5rem 0.8rem;
        }
        h1 {font-size: 1.3rem !important;}
        [data-testid="stChatMessage"] {
            padding: 0.6rem 0.8rem;
        }
        [data-testid="stChatMessageContent"] {
            font-size: 0.88rem !important;
        }
    }

    .welcome-box {
        text-align: center;
        padding: 3rem 1rem;
    }
    .welcome-box h2 {
        font-size: 1.1rem;
        color: #6b7280;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }
    .welcome-box p {
        font-size: 0.82rem;
        color: #374151;
        margin: 0.2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────
st.markdown("# 🧠 DocChat AI")
st.caption("Hybrid RAG · Reranking · Parent Document Retrieval")

# ── Chat persistence ─────────────────────────────────
def save_chat(chat_id, messages, title):
    path = os.path.join(CHATS_DIR, f"{chat_id}.json")
    with open(path, "w") as f:
        json.dump({
            "id": chat_id,
            "title": title,
            "messages": messages,
            "updated_at": datetime.now().isoformat()
        }, f, indent=2)

def list_chats():
    chats = []
    for fn in os.listdir(CHATS_DIR):
        if fn.endswith(".json"):
            with open(os.path.join(CHATS_DIR, fn)) as f:
                chats.append(json.load(f))
    return sorted(chats, key=lambda c: c.get("updated_at", ""), reverse=True)

def delete_chat(chat_id):
    path = os.path.join(CHATS_DIR, f"{chat_id}.json")
    if os.path.exists(path):
        os.remove(path)

# ── Embeddings + Reranker + LLM ──────────────────────
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

@st.cache_resource
def load_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

@st.cache_resource
def load_llm():
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model="openai/gpt-oss-120b",
        temperature=0,
    )

embeddings = load_embeddings()
reranker   = load_reranker()
llm        = load_llm()

# ── Query helpers ─────────────────────────────────────
def expand_query(query):
    expansion_prompt = f"""Given this search query, generate 8-10 additional keywords and phrases that would help find relevant information in a document. Return ONLY the expanded query as a single line, nothing else.

Original query: {query}

Expanded query:"""
    expanded = llm.invoke(expansion_prompt)
    if hasattr(expanded, "content"):
        expanded = expanded.content
    return f"{query} {expanded}"

def rewrite_query(question, history):
    if not history or history == "No prior conversation.":
        return question
    rewrite_prompt = f"""Given the chat history and the latest question, rewrite the question to be fully self-contained and specific. If the question is already clear, return it unchanged. Return ONLY the rewritten question, nothing else.

Chat History:
{history}

Latest Question: {question}

Rewritten Question:"""
    rewritten = llm.invoke(rewrite_prompt)
    if hasattr(rewritten, "content"):
        rewritten = rewritten.content
    return rewritten.strip()

def check_hallucination(question, answer, source_docs):
    context = "\n\n".join(doc.page_content for doc in source_docs)
    check_prompt = f"""You are a fact-checker. Given the context, question, and answer below, determine if the answer is supported by the context.

Context:
{context[:6000]}

Question: {question}

Answer: {answer}

Is the answer broadly supported by the context, without any significant fabricated claims? Note: if the answer admits it cannot find certain information, that is GROUNDED behaviour, not a hallucination. Reply with ONLY one of:
GROUNDED - answer is fully or mostly supported
PARTIAL - answer contains some claims not clearly in the context
HALLUCINATED - answer contains significant fabricated information not in the context at all"""
    result = llm.invoke(check_prompt)
    if hasattr(result, "content"):
        result = result.content
    return result.strip()

# ── Ingest files ──────────────────────────────────────
def ingest_files(file_paths, append=False):
    new_parent_chunks = []
    new_child_chunks  = []
    file_stats        = []

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )

    for file_path in file_paths:
        ext      = os.path.splitext(file_path)[-1].lower()
        filename = os.path.basename(file_path).replace("temp_", "")

        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext in [".xlsx", ".xls"]:
            loader = UnstructuredExcelLoader(file_path, mode="elements")
        elif ext in [".docx", ".doc"]:
            loader = Docx2txtLoader(file_path)
        else:
            st.warning(f"Skipping: {filename}")
            continue

        docs = loader.load()
        for d in docs:
            d.metadata["source_file"] = filename

        parents = parent_splitter.split_documents(docs)
        for i, parent in enumerate(parents):
            parent.metadata["parent_id"] = f"{filename}_{i}"
        new_parent_chunks.extend(parents)

        for i, parent in enumerate(parents):
            children = child_splitter.split_documents([parent])
            for child in children:
                child.metadata["parent_id"] = f"{filename}_{i}"
            new_child_chunks.extend(children)

        file_stats.append((filename, len(docs), len(parents), len(new_child_chunks)))

    if append and os.path.exists(CHUNKS_PATH) and os.path.exists(PARENTS_PATH):
        with open(CHUNKS_PATH, "rb") as f:
            existing_children = pickle.load(f)
        with open(PARENTS_PATH, "rb") as f:
            existing_parents = pickle.load(f)
        all_children = existing_children + new_child_chunks
        all_parents  = existing_parents  + new_parent_chunks
    else:
        all_children = new_child_chunks
        all_parents  = new_parent_chunks

    if not all_children:
        st.error("No text could be extracted. The file may be scanned or image-based.")
        return None, None, None, None, None

    parent_lookup = {p.metadata["parent_id"]: p for p in all_parents}

    db = FAISS.from_documents(all_children, embeddings)
    db.save_local(FAISS_PATH)

    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(all_children, f)
    with open(PARENTS_PATH, "wb") as f:
        pickle.dump(parent_lookup, f)

    return db, all_children, all_parents, parent_lookup, file_stats

@st.cache_resource
def load_existing_db():
    if os.path.exists(FAISS_PATH) and os.path.exists(CHUNKS_PATH) and os.path.exists(PARENTS_PATH):
        try:
            db = FAISS.load_local(
                FAISS_PATH, embeddings,
                allow_dangerous_deserialization=True
            )
            with open(CHUNKS_PATH, "rb") as f:
                children = pickle.load(f)
            with open(PARENTS_PATH, "rb") as f:
                parent_lookup = pickle.load(f)
            return db, children, parent_lookup
        except Exception:
            return None, None, None
    return None, None, None

# ── Hybrid RAG chain ──────────────────────────────────
def build_chain(db, all_children, parent_lookup):
    prompt = PromptTemplate(
        template="""You are an expert assistant analyzing documents. Answer the question thoroughly using the context below.

INSTRUCTIONS:
- Use ALL relevant information from the context
- Combine information from multiple parts if needed
- Be specific and detailed — quote facts, numbers, names
- If asked about a person, list everything: skills, projects, experience, education
- Only say "I don't have enough information" if the context is truly empty on the topic

Chat History:
{history}

Context:
{context}

Question: {question}

Detailed Answer:""",
        input_variables=["history", "context", "question"]
    )

    semantic_retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 8, "fetch_k": 20, "lambda_mult": 0.5}
    )

    bm25_retriever = BM25Retriever.from_documents(all_children)
    bm25_retriever.k = 6

    def get_parents(docs):
        seen, parents = set(), []
        for doc in docs:
            pid = doc.metadata.get("parent_id")
            if pid and pid not in seen and pid in parent_lookup:
                parents.append(parent_lookup[pid])
                seen.add(pid)
        return parents if parents else docs

    def rerank_docs(query, semantic_docs, keyword_docs, top_n=4):
        semantic_parents = get_parents(semantic_docs)
        keyword_parents  = get_parents(keyword_docs[:2])

        if semantic_parents:
            pairs  = [(query, doc.page_content) for doc in semantic_parents]
            scores = reranker.predict(pairs)
            ranked = sorted(zip(scores, semantic_parents), key=lambda x: x[0], reverse=True)
            top_semantic = [doc for _, doc in ranked[:top_n]]
        else:
            top_semantic = []

        seen, final = set(), []
        for doc in keyword_parents + top_semantic:
            key = doc.metadata.get("parent_id", doc.page_content[:80])
            if key not in seen:
                final.append(doc)
                seen.add(key)
        return final

    def format_docs(docs):
        return "\n\n".join(
            f"[Source: {d.metadata.get('source_file','?')} | Page: {d.metadata.get('page','?')}]\n{d.page_content}"
            for d in docs
        )

    chain = (
        {
            "context": lambda x: format_docs(
                rerank_docs(
                    x["question"],
                    semantic_retriever.invoke(x["expanded_query"]),
                    bm25_retriever.invoke(x["expanded_query"])
                )
            ),
            "question": lambda x: x["question"],
            "history":  lambda x: x["history"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    hybrid_retriever = EnsembleRetriever(
        retrievers=[semantic_retriever, bm25_retriever],
        weights=[0.5, 0.5]
    )

    return chain, hybrid_retriever

def format_history(messages, last_n=4):
    recent = messages[-last_n*2:]
    return "\n".join(
        f"{m['role'].capitalize()}: {m['content']}" for m in recent
    ) if recent else "No prior conversation."

# ── Session state ────────────────────────────────────
if "chat_id" not in st.session_state:
    st.session_state.chat_id = str(uuid.uuid4())[:8]
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_title" not in st.session_state:
    st.session_state.chat_title = "New Chat"

# ── Sidebar ──────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📁 Documents")
    uploaded_files = st.file_uploader(
        "Upload PDFs, Excel, or Word",
        type=["pdf", "xlsx", "xls", "docx", "doc"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    append_mode = st.toggle("➕ Append to existing docs", value=False)

    if uploaded_files and st.button("🚀 Ingest", use_container_width=True):
        saved_paths = []
        for f in uploaded_files:
            p = f"/tmp/temp_{f.name}"
            with open(p, "wb") as fp:
                fp.write(f.getbuffer())
            saved_paths.append(p)

        with st.spinner(f"Ingesting {len(saved_paths)} file(s)..."):
            result = ingest_files(saved_paths, append=append_mode)
            for p in saved_paths:
                os.remove(p)

        if result[0] is not None:
            db, all_children, all_parents, parent_lookup, file_stats = result
            chain, retriever = build_chain(db, all_children, parent_lookup)
            st.session_state.rag_chain = chain
            st.session_state.retriever = retriever
            st.success(f"✅ {len(file_stats)} file(s) ingested")
            for name, pages, parents, children in file_stats:
                st.caption(f"📄 {name} • {pages}p • {parents} parents • {children} children")

    if "rag_chain" not in st.session_state:
        existing_db, existing_children, existing_parents = load_existing_db()
        if existing_db and existing_children and existing_parents:
            chain, retriever = build_chain(existing_db, existing_children, existing_parents)
            st.session_state.rag_chain = chain
            st.session_state.retriever = retriever

    st.markdown("---")

    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.chat_id    = str(uuid.uuid4())[:8]
        st.session_state.messages   = []
        st.session_state.chat_title = "New Chat"
        st.rerun()

    st.markdown("### 💬 Chat History")
    chats = list_chats()
    if not chats:
        st.caption("No previous chats")
    else:
        for c in chats[:20]:
            col_a, col_b = st.columns([5, 1])
            with col_a:
                if st.button(
                    f"💬 {c['title'][:28]}",
                    key=f"load_{c['id']}",
                    use_container_width=True
                ):
                    st.session_state.chat_id    = c["id"]
                    st.session_state.messages   = c["messages"]
                    st.session_state.chat_title = c["title"]
                    st.rerun()
            with col_b:
                if st.button("🗑", key=f"del_{c['id']}"):
                    delete_chat(c["id"])
                    st.rerun()

# ── Main chat ────────────────────────────────────────
if "rag_chain" not in st.session_state:
    st.markdown("""
    <div class="welcome-box">
        <h2>Upload a document to get started</h2>
        <p>Supports PDF, Excel, and Word files</p>
        <p>Powered by Hybrid RAG · Reranking · Parent Document Retrieval</p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"#### {st.session_state.chat_title}")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])
            if "hallucination" in msg:
                badge = {
                    "GROUNDED":     "✅ Grounded",
                    "PARTIAL":      "⚠️ Partially grounded",
                    "HALLUCINATED": "🚨 May contain unsupported claims"
                }.get(msg["hallucination"].split()[0], "🔍 Checked")
                st.caption(f"Confidence: {badge}")
            if "sources" in msg and msg["sources"]:
                with st.expander(f"📚 {len(msg['sources'])} sources"):
                    for src in msg["sources"]:
                        st.markdown(f"**{src['file']}** — page {src['page']}")
                        st.caption(src["snippet"])

    if question := st.chat_input("Ask anything about your documents..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(question)

        if st.session_state.chat_title == "New Chat":
            st.session_state.chat_title = question[:50]

        with st.chat_message("assistant", avatar="🤖"):
            history = format_history(st.session_state.messages[:-1])

            with st.spinner("🔍 Processing query..."):
                rewritten = rewrite_query(question, history)
                expanded  = expand_query(rewritten)

            source_docs = st.session_state.retriever.invoke(expanded)

            placeholder = st.empty()
            full_answer = ""
            for chunk in st.session_state.rag_chain.stream({
                "question":       rewritten,
                "expanded_query": expanded,
                "history":        history
            }):
                full_answer += chunk
                placeholder.markdown(full_answer + "▌")
            placeholder.markdown(full_answer)

            with st.spinner("🔎 Checking answer..."):
                hallucination_status = check_hallucination(
                    question, full_answer, source_docs
                )

            badge = {
                "GROUNDED":     "✅ Grounded",
                "PARTIAL":      "⚠️ Partially grounded",
                "HALLUCINATED": "🚨 May contain unsupported claims"
            }.get(hallucination_status.split()[0], "🔍 Checked")
            st.caption(f"Confidence: {badge}")

            sources, seen = [], set()
            for doc in source_docs:
                key = (doc.metadata.get("source_file", "?"), doc.metadata.get("page", "?"))
                if key not in seen:
                    sources.append({
                        "file":    doc.metadata.get("source_file", "Unknown"),
                        "page":    doc.metadata.get("page", "?"),
                        "snippet": doc.page_content[:200] + "..."
                    })
                    seen.add(key)

            with st.expander(f"📚 {len(sources)} sources"):
                for src in sources:
                    st.markdown(f"**{src['file']}** — page {src['page']}")
                    st.caption(src["snippet"])

        st.session_state.messages.append({
            "role":          "assistant",
            "content":       full_answer,
            "hallucination": hallucination_status,
            "sources":       sources
        })

        save_chat(
            st.session_state.chat_id,
            st.session_state.messages,
            st.session_state.chat_title
        )