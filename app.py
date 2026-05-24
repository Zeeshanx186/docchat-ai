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

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

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

st.set_page_config(
    page_title="DocChat AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
#MainMenu, footer, header {visibility: hidden;}
* {box-sizing: border-box;}
.stApp {
    background: #0a0a0e;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}
.main .block-container {
    padding: 0.5rem 1rem 5rem 1rem;
    max-width: 100%;
}
h1 {
    font-size: clamp(1.2rem, 3vw, 1.6rem) !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, #f59e0b, #fbbf24);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 !important;
    line-height: 1.2 !important;
}
h2, h3, h4 {color: #e2e8f0 !important; font-weight: 600 !important;}
p, .stCaption, label {color: #6b7280 !important; font-size: 0.78rem !important;}

[data-testid="stSidebar"] {
    background: #0e0e14 !important;
    border-right: 1px solid #1c1c28 !important;
}
[data-testid="stSidebar"] > div {padding: 0 !important;}

[data-testid="stFileUploader"] {
    background: #0c0c12;
    border: 1.5px dashed #1c1c28;
    border-radius: 10px;
    padding: 0.6rem;
}
[data-testid="stFileUploader"]:hover {border-color: #f59e0b;}

.stButton > button {
    background: linear-gradient(135deg, #d97706, #f59e0b) !important;
    color: #0a0a0e !important;
    border: none !important;
    border-radius: 9px !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    padding: 0.5rem 1rem !important;
    transition: all 0.2s !important;
    width: 100%;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(245,158,11,0.3) !important;
}

[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    color: #6b7280 !important;
    border: 1px solid #1c1c28 !important;
    border-radius: 7px !important;
    font-weight: 400 !important;
    font-size: 0.78rem !important;
    padding: 0.35rem 0.6rem !important;
    box-shadow: none !important;
    transform: none !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #111119 !important;
    border-color: #f59e0b !important;
    color: #e2e8f0 !important;
    transform: none !important;
    box-shadow: none !important;
}

[data-testid="stChatMessage"] {
    background: #0e0e16;
    border: 1px solid #1c1c28;
    border-radius: 14px;
    padding: 0.8rem 1rem;
    margin: 0.3rem 0;
}
[data-testid="stChatMessageContent"] {
    font-size: clamp(0.85rem, 2vw, 0.95rem) !important;
    line-height: 1.7 !important;
    color: #e2e8f0 !important;
}
[data-testid="stChatInput"] {
    background: #0e0e16 !important;
    border: 1.5px solid #2a2a3a !important;
    border-radius: 13px !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #f59e0b !important;
    box-shadow: 0 0 0 3px rgba(245,158,11,0.1) !important;
}
[data-testid="stChatInput"] textarea {
    font-size: 15px !important;
    color: #e2e8f0 !important;
}
[data-testid="stExpander"] {
    background: #0e0e16 !important;
    border: 1px solid #1c1c28 !important;
    border-radius: 10px !important;
}
[data-testid="stToggle"] span {background: #1c1c28 !important;}
hr {border-color: #1c1c28 !important; margin: 0.5rem 0 !important;}
::-webkit-scrollbar {width: 3px;}
::-webkit-scrollbar-track {background: #0a0a0e;}
::-webkit-scrollbar-thumb {background: #2a2a3a; border-radius: 3px;}
::-webkit-scrollbar-thumb:hover {background: #f59e0b;}

.chunk-card {
    border-radius: 10px;
    padding: 0.7rem 0.9rem;
    margin: 0.4rem 0;
    border-left: 3px solid;
    border-top: 1px solid #1c1c28;
    border-right: 1px solid #1c1c28;
    border-bottom: 1px solid #1c1c28;
    background: #111119;
    font-size: 0.78rem;
}
.source-card {
    border-radius: 9px;
    padding: 0.6rem 0.8rem;
    background: #111119;
    border-top: 2px solid;
    border-left: 1px solid #1c1c28;
    border-right: 1px solid #1c1c28;
    border-bottom: 1px solid #1c1c28;
    margin: 0.3rem 0;
    font-size: 0.78rem;
}
.doc-card-active {
    background: #13131e;
    border: 1px solid #f59e0b;
    border-left: 3px solid #f59e0b;
    border-radius: 9px;
    padding: 0.6rem 0.8rem;
    margin: 0.25rem 0;
}
.doc-card {
    background: #111119;
    border: 1px solid #1c1c28;
    border-radius: 9px;
    padding: 0.6rem 0.8rem;
    margin: 0.25rem 0;
}
.chat-item-active {
    background: #13131e;
    border: 1px solid #f59e0b;
    border-left: 3px solid #f59e0b;
    border-radius: 8px;
    padding: 0.4rem 0.6rem;
    margin: 0.15rem 0;
}
.chat-item {
    background: #111119;
    border: 1px solid transparent;
    border-radius: 8px;
    padding: 0.4rem 0.6rem;
    margin: 0.15rem 0;
}
.section-label {
    font-size: 0.65rem !important;
    font-weight: 600 !important;
    letter-spacing: 1.5px !important;
    color: #3d3d52 !important;
    text-transform: uppercase;
    margin: 0.8rem 0 0.3rem 0;
    padding: 0 14px;
}
.welcome-box {
    text-align: center;
    padding: 4rem 2rem;
}
@media (max-width: 768px) {
    .main .block-container {padding: 0.5rem 0.5rem 5rem 0.5rem;}
    h1 {font-size: 1.1rem !important;}
}
</style>
""", unsafe_allow_html=True)


def save_chat(chat_id, messages, title):
    path = os.path.join(CHATS_DIR, f"{chat_id}.json")
    with open(path, "w") as f:
        json.dump({"id": chat_id, "title": title, "messages": messages,
                   "updated_at": datetime.now().isoformat()}, f, indent=2)

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

def time_label(iso):
    try:
        diff = datetime.now() - datetime.fromisoformat(iso)
        if diff.days == 0: return "TODAY"
        if diff.days == 1: return "YESTERDAY"
        return f"{diff.days} DAYS AGO"
    except:
        return ""


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
    return ChatGroq(api_key=GROQ_API_KEY, model="openai/gpt-oss-120b", temperature=0)

embeddings = load_embeddings()
reranker   = load_reranker()
llm        = load_llm()


def expand_query(query):
    prompt = f"""Generate 8-10 additional keywords to help find relevant information. Return ONLY the expanded query as a single line.

Original: {query}
Expanded:"""
    r = llm.invoke(prompt)
    text = r.content if hasattr(r, "content") else r
    return f"{query} {text}"

def rewrite_query(question, history):
    if not history or history == "No prior conversation.":
        return question
    prompt = f"""Rewrite the question to be fully self-contained. Return ONLY the rewritten question.

History:
{history}

Question: {question}
Rewritten:"""
    r = llm.invoke(prompt)
    text = r.content if hasattr(r, "content") else r
    return text.strip()

def check_hallucination(question, answer, source_docs):
    context = "\n\n".join(doc.page_content for doc in source_docs)
    prompt = f"""Fact-check: is the answer supported by the context?
Note: admitting missing info = GROUNDED.

Context:
{context[:6000]}

Question: {question}
Answer: {answer}

Reply ONLY with one of:
GROUNDED - fully or mostly supported
PARTIAL - some claims not in context
HALLUCINATED - significant fabricated info"""
    r = llm.invoke(prompt)
    text = r.content if hasattr(r, "content") else r
    return text.strip()


def ingest_files(file_paths, append=False):
    new_parents, new_children, file_stats = [], [], []
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
    child_splitter  = RecursiveCharacterTextSplitter(chunk_size=200,  chunk_overlap=20)

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
        for i, p in enumerate(parents):
            p.metadata["parent_id"] = f"{filename}_{i}"
        new_parents.extend(parents)

        for i, p in enumerate(parents):
            children = child_splitter.split_documents([p])
            for c in children:
                c.metadata["parent_id"] = f"{filename}_{i}"
            new_children.extend(children)

        file_stats.append((filename, len(docs), len(parents), len(new_children)))

    if append and os.path.exists(CHUNKS_PATH) and os.path.exists(PARENTS_PATH):
        with open(CHUNKS_PATH, "rb") as f:
            existing_children = pickle.load(f)
        with open(PARENTS_PATH, "rb") as f:
            existing_parents = pickle.load(f)
        all_children = existing_children + new_children
        all_parents  = existing_parents  + new_parents
    else:
        all_children = new_children
        all_parents  = new_parents

    if not all_children:
        st.error("No text extracted. File may be scanned/image-based.")
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
            db = FAISS.load_local(FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
            with open(CHUNKS_PATH, "rb") as f:
                children = pickle.load(f)
            with open(PARENTS_PATH, "rb") as f:
                parent_lookup = pickle.load(f)
            return db, children, parent_lookup
        except:
            return None, None, None
    return None, None, None


def build_chain(db, all_children, parent_lookup):
    prompt = PromptTemplate(
        template="""You are an expert assistant. Answer thoroughly using the context below.

INSTRUCTIONS:
- Use ALL relevant information from context
- Be specific — quote facts, numbers, names
- Only say "I don't have enough information" if context is truly empty

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
            "context":  lambda x: format_docs(rerank_docs(
                x["question"],
                semantic_retriever.invoke(x["expanded_query"]),
                bm25_retriever.invoke(x["expanded_query"])
            )),
            "question": lambda x: x["question"],
            "history":  lambda x: x["history"],
        }
        | prompt | llm | StrOutputParser()
    )

    hybrid_retriever = EnsembleRetriever(
        retrievers=[semantic_retriever, bm25_retriever],
        weights=[0.5, 0.5]
    )

    return chain, hybrid_retriever, rerank_docs, semantic_retriever, bm25_retriever

def format_history(messages, last_n=4):
    recent = messages[-last_n*2:]
    return "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in recent) if recent else "No prior conversation."


if "chat_id"         not in st.session_state: st.session_state.chat_id         = str(uuid.uuid4())[:8]
if "messages"        not in st.session_state: st.session_state.messages        = []
if "chat_title"      not in st.session_state: st.session_state.chat_title      = "New Chat"
if "ingested_files"  not in st.session_state: st.session_state.ingested_files  = []


with st.sidebar:
    st.markdown("""
    <div style="padding:14px 14px 10px 14px;border-bottom:1px solid #1c1c28;border-left:3px solid #f59e0b;margin-bottom:8px;">
        <div style="display:flex;align-items:center;gap:10px;">
            <div style="width:30px;height:30px;border-radius:8px;background:#1a1505;border:1px solid rgba(245,158,11,0.4);display:flex;align-items:center;justify-content:center;font-weight:700;color:#f59e0b;font-size:14px;">D</div>
            <div>
                <div style="color:#f1f1f4;font-weight:700;font-size:13px;line-height:1.2;">DocChat AI</div>
                <div style="color:#f59e0b;font-size:9px;">Hybrid RAG · Zero Cost</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="section-label">Workspace</p>', unsafe_allow_html=True)

    if st.session_state.ingested_files:
        for i, fname in enumerate(st.session_state.ingested_files):
            active     = i == 0
            card_class = "doc-card-active" if active else "doc-card"
            badge      = '<span style="color:#34d399;font-size:9px;">● active</span>' if active else ""
            st.markdown(f"""
            <div class="{card_class}">
                <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                    <div>
                        <div style="color:#f1f1f4;font-size:10px;font-weight:500;">📄 {fname[:26]}</div>
                        <div style="color:#3d3d52;font-size:8px;margin-top:2px;">ingested</div>
                    </div>
                    {badge}
                </div>
            </div>
            """, unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Drop files here",
        type=["pdf", "xlsx", "xls", "docx", "doc"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    append_mode = st.toggle("Append to existing docs", value=False)

    if uploaded_files and st.button("⚡ Ingest Documents", use_container_width=True):
        saved_paths = []
        for f in uploaded_files:
            p = f"/tmp/temp_{f.name}"
            with open(p, "wb") as fp:
                fp.write(f.getbuffer())
            saved_paths.append(p)

        with st.spinner("Ingesting..."):
            result = ingest_files(saved_paths, append=append_mode)
            for p in saved_paths:
                os.remove(p)

        if result[0] is not None:
            db, all_children, all_parents, parent_lookup, file_stats = result
            chain, retriever, rerank_fn, sem_ret, bm25_ret = build_chain(db, all_children, parent_lookup)
            st.session_state.rag_chain      = chain
            st.session_state.retriever      = retriever
            st.session_state.rerank_fn      = rerank_fn
            st.session_state.sem_ret        = sem_ret
            st.session_state.bm25_ret       = bm25_ret
            st.session_state.parent_lookup  = parent_lookup
            st.session_state.ingested_files = [s[0] for s in file_stats]
            st.success(f"✅ {len(file_stats)} file(s) ready")
            for name, pages, parents, children in file_stats:
                st.caption(f"📄 {name} · {pages}p · {parents} parents · {children} children")

    if "rag_chain" not in st.session_state:
        existing_db, existing_children, existing_parents = load_existing_db()
        if existing_db and existing_children and existing_parents:
            chain, retriever, rerank_fn, sem_ret, bm25_ret = build_chain(existing_db, existing_children, existing_parents)
            st.session_state.rag_chain     = chain
            st.session_state.retriever     = retriever
            st.session_state.rerank_fn     = rerank_fn
            st.session_state.sem_ret       = sem_ret
            st.session_state.bm25_ret      = bm25_ret
            st.session_state.parent_lookup = existing_parents

    st.markdown("---")

    if st.button("＋ New Conversation", use_container_width=True):
        st.session_state.chat_id    = str(uuid.uuid4())[:8]
        st.session_state.messages   = []
        st.session_state.chat_title = "New Chat"
        st.rerun()

    st.markdown('<p class="section-label">Recent Chats</p>', unsafe_allow_html=True)
    chats = list_chats()
    if not chats:
        st.markdown('<p style="color:#2a2a38;font-size:9px;padding:0 14px;">No previous chats</p>', unsafe_allow_html=True)
    else:
        for c in chats[:15]:
            is_active  = c["id"] == st.session_state.chat_id
            card_class = "chat-item-active" if is_active else "chat-item"
            label      = time_label(c.get("updated_at",""))
            st.markdown(f"""
            <div class="{card_class}">
                <div style="color:#3d3d52;font-size:7px;">{label}</div>
                <div style="color:{'#e2e8f0' if is_active else '#6b7280'};font-size:9px;font-weight:{'500' if is_active else '400'};">{c['title'][:30]}</div>
            </div>
            """, unsafe_allow_html=True)
            col_a, col_b = st.columns([4, 1])
            with col_a:
                if st.button(f"↗ Open", key=f"load_{c['id']}", use_container_width=True):
                    st.session_state.chat_id    = c["id"]
                    st.session_state.messages   = c["messages"]
                    st.session_state.chat_title = c["title"]
                    st.rerun()
            with col_b:
                if st.button("🗑", key=f"del_{c['id']}"):
                    delete_chat(c["id"])
                    st.rerun()


if "rag_chain" not in st.session_state:
    st.markdown("""
    <div class="welcome-box">
        <div style="font-size:3rem;margin-bottom:1rem;">🧠</div>
        <div style="color:#4b4b60;font-size:1.1rem;font-weight:500;margin-bottom:0.5rem;">Upload a document to get started</div>
        <div style="color:#3d3d52;font-size:0.82rem;">Supports PDF · Excel · Word</div>
        <div style="color:#2a2a38;font-size:0.78rem;margin-top:0.5rem;">Hybrid RAG · Reranking · Parent Document Retrieval · Hallucination Check</div>
    </div>
    """, unsafe_allow_html=True)
else:
    chat_col, ctx_col = st.columns([3, 1])

    with chat_col:
        st.markdown(f"""
        <div style="padding:0.5rem 0 0.8rem 0;border-bottom:1px solid #1c1c28;margin-bottom:0.8rem;">
            <div style="color:#3d3d52;font-size:8px;letter-spacing:1.2px;text-transform:uppercase;">Active Conversation</div>
            <div style="color:#e2e8f0;font-size:13px;font-weight:600;">{st.session_state.chat_title}</div>
        </div>
        """, unsafe_allow_html=True)

        SOURCE_COLORS = ["#818cf8","#f59e0b","#34d399","#f472b6"]

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
                st.markdown(msg["content"])

                if "pipeline" in msg:
                    p = msg["pipeline"]
                    steps = [
                        ("#818cf8", "① Rewrite ✓"),
                        ("#f59e0b", f"② Expand +{p.get('keywords',8)}kw ✓"),
                        ("#34d399", f"③ Retrieve {p.get('chunks',0)} ✓"),
                        ("#f472b6", "④ Rerank ✓"),
                        ("#38bdf8", f"⑤ PDR {p.get('parents',0)} ✓"),
                        ("#f59e0b", "⑥ Generate ✓"),
                    ]
                    pills = " → ".join([
                        f'<span style="background:#111119;border:1px solid {c};color:{c};padding:2px 8px;border-radius:5px;font-size:0.68rem;font-weight:600;">{l}</span>'
                        for c, l in steps
                    ])
                    st.markdown(f'<div style="margin:0.4rem 0;">{pills}</div>', unsafe_allow_html=True)

                if "hallucination" in msg:
                    status = msg["hallucination"].split()[0]
                    badge_map = {
                        "GROUNDED":     ("✓ Grounded",               "#34d399","#0d2010", 87),
                        "PARTIAL":      ("⚠ Partially grounded",     "#f59e0b","#1a1505", 62),
                        "HALLUCINATED": ("✗ May contain fabrications","#f87171","#200d0d", 30),
                    }
                    lbl, color, bg, score = badge_map.get(status, ("🔍 Checked","#6b7280","#111119", 50))
                    bw = score * 1.6
                    st.markdown(f"""
                    <div style="display:flex;align-items:center;gap:10px;margin:0.3rem 0;">
                        <span style="background:{bg};color:{color};padding:2px 8px;border-radius:5px;font-size:0.7rem;font-weight:600;">{lbl}</span>
                        <div style="flex:1;max-width:160px;background:#1c1c28;border-radius:3px;height:4px;">
                            <div style="background:{color};width:{bw}px;height:4px;border-radius:3px;opacity:0.7;"></div>
                        </div>
                        <span style="color:{color};font-size:0.68rem;">{score}%</span>
                    </div>
                    """, unsafe_allow_html=True)

                if "sources" in msg and msg["sources"]:
                    with st.expander(f"📚 {len(msg['sources'])} sources"):
                        for i, src in enumerate(msg["sources"]):
                            c  = SOURCE_COLORS[i % len(SOURCE_COLORS)]
                            sc = src.get("score", 0.5)
                            bw = int(sc * 120)
                            st.markdown(f"""
                            <div class="source-card" style="border-top-color:{c};">
                                <div style="display:flex;justify-content:space-between;align-items:center;">
                                    <span style="color:{c};font-weight:600;font-size:0.78rem;">📄 {src['file']} · p.{src['page']}</span>
                                    <span style="color:{c};font-size:0.72rem;background:#0e0e16;padding:1px 6px;border-radius:4px;">{sc:.2f}</span>
                                </div>
                                <div style="color:#6b7280;font-size:0.75rem;margin:4px 0;">{src['snippet']}</div>
                                <div style="background:#1c1c28;border-radius:2px;height:3px;margin-top:4px;">
                                    <div style="background:{c};width:{bw}px;height:3px;border-radius:2px;opacity:0.6;"></div>
                                </div>
                                <div style="color:#3d3d52;font-size:0.68rem;margin-top:3px;">{src.get('retrieval_type','semantic')}</div>
                            </div>
                            """, unsafe_allow_html=True)

        if question := st.chat_input("Ask anything about your documents..."):
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user", avatar="🧑"):
                st.markdown(question)

            if st.session_state.chat_title == "New Chat":
                st.session_state.chat_title = question[:50]

            with st.chat_message("assistant", avatar="🤖"):
                history      = format_history(st.session_state.messages[:-1])
                pipeline_info = {}

                with st.spinner("① Rewriting query..."):
                    rewritten = rewrite_query(question, history)
                    pipeline_info["rewrite"] = rewritten

                with st.spinner("② Expanding keywords..."):
                    expanded  = expand_query(rewritten)
                    pipeline_info["keywords"] = max(len(expanded.split()) - len(rewritten.split()), 8)

                with st.spinner("③ Retrieving chunks..."):
                    sem_docs  = st.session_state.sem_ret.invoke(expanded)
                    bm25_docs = st.session_state.bm25_ret.invoke(expanded)
                    pipeline_info["chunks"] = len(sem_docs) + len(bm25_docs)

                with st.spinner("④⑤ Reranking + fetching parents..."):
                    final_docs = st.session_state.rerank_fn(question, sem_docs, bm25_docs)
                    pipeline_info["parents"] = len(final_docs)
                    scores = reranker.predict([(question, d.page_content) for d in final_docs]) if final_docs else []

                source_docs = st.session_state.retriever.invoke(expanded)

                placeholder = st.empty()
                full_answer = ""
                for chunk in st.session_state.rag_chain.stream({
                    "question": rewritten, "expanded_query": expanded, "history": history
                }):
                    full_answer += chunk
                    placeholder.markdown(full_answer + "▌")
                placeholder.markdown(full_answer)

                with st.spinner("Checking for hallucinations..."):
                    hallucination_status = check_hallucination(question, full_answer, source_docs)

                # Pipeline pills
                steps = [
                    ("#818cf8", "① Rewrite ✓"),
                    ("#f59e0b", f"② Expand +{pipeline_info.get('keywords',8)}kw ✓"),
                    ("#34d399", f"③ Retrieve {pipeline_info.get('chunks',0)} ✓"),
                    ("#f472b6", "④ Rerank ✓"),
                    ("#38bdf8", f"⑤ PDR {pipeline_info.get('parents',0)} ✓"),
                    ("#f59e0b", "⑥ Generate ✓"),
                ]
                pills = " → ".join([
                    f'<span style="background:#111119;border:1px solid {c};color:{c};padding:2px 8px;border-radius:5px;font-size:0.68rem;font-weight:600;">{l}</span>'
                    for c, l in steps
                ])
                st.markdown(f'<div style="margin:0.4rem 0;">{pills}</div>', unsafe_allow_html=True)

                # Confidence badge
                status = hallucination_status.split()[0]
                badge_map = {
                    "GROUNDED":     ("✓ Grounded",               "#34d399","#0d2010", 87),
                    "PARTIAL":      ("⚠ Partially grounded",     "#f59e0b","#1a1505", 62),
                    "HALLUCINATED": ("✗ May contain fabrications","#f87171","#200d0d", 30),
                }
                lbl, color, bg, score = badge_map.get(status, ("🔍 Checked","#6b7280","#111119", 50))
                bw = score * 1.6
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:10px;margin:0.3rem 0;">
                    <span style="background:{bg};color:{color};padding:2px 8px;border-radius:5px;font-size:0.7rem;font-weight:600;">{lbl}</span>
                    <div style="flex:1;max-width:160px;background:#1c1c28;border-radius:3px;height:4px;">
                        <div style="background:{color};width:{bw}px;height:4px;border-radius:3px;opacity:0.7;"></div>
                    </div>
                    <span style="color:{color};font-size:0.68rem;">{score}%</span>
                </div>
                """, unsafe_allow_html=True)

                # Sources
                retrieval_types = (
                    ["keyword bypass"] * min(2, len(bm25_docs)) +
                    ["semantic"] * len(sem_docs)
                )
                sources, seen = [], set()
                for i, doc in enumerate(source_docs):
                    key = (doc.metadata.get("source_file","?"), doc.metadata.get("page","?"))
                    if key not in seen:
                        raw_score = float(scores[i]) if i < len(scores) else 0
                        norm_score = round(max(0, min(1, (raw_score + 10) / 20)), 2)
                        sources.append({
                            "file":           doc.metadata.get("source_file","Unknown"),
                            "page":           doc.metadata.get("page","?"),
                            "snippet":        doc.page_content[:180] + "...",
                            "score":          norm_score,
                            "retrieval_type": retrieval_types[i] if i < len(retrieval_types) else "semantic"
                        })
                        seen.add(key)

                with st.expander(f"📚 {len(sources)} sources"):
                    for i, src in enumerate(sources):
                        c  = SOURCE_COLORS[i % len(SOURCE_COLORS)]
                        sc = src.get("score", 0.5)
                        bw = int(sc * 120)
                        st.markdown(f"""
                        <div class="source-card" style="border-top-color:{c};">
                            <div style="display:flex;justify-content:space-between;align-items:center;">
                                <span style="color:{c};font-weight:600;font-size:0.78rem;">📄 {src['file']} · p.{src['page']}</span>
                                <span style="color:{c};font-size:0.72rem;background:#0e0e16;padding:1px 6px;border-radius:4px;">{sc:.2f}</span>
                            </div>
                            <div style="color:#6b7280;font-size:0.75rem;margin:4px 0;">{src['snippet']}</div>
                            <div style="background:#1c1c28;border-radius:2px;height:3px;margin-top:4px;">
                                <div style="background:{c};width:{bw}px;height:3px;border-radius:2px;opacity:0.6;"></div>
                            </div>
                            <div style="color:#3d3d52;font-size:0.68rem;margin-top:3px;">{src.get('retrieval_type','semantic')}</div>
                        </div>
                        """, unsafe_allow_html=True)

                st.session_state.messages.append({
                    "role":          "assistant",
                    "content":       full_answer,
                    "hallucination": hallucination_status,
                    "pipeline":      pipeline_info,
                    "sources":       sources,
                    "context_chunks": [
                        {
                            "file":  d.metadata.get("source_file","?"),
                            "page":  d.metadata.get("page","?"),
                            "text":  d.page_content[:400],
                            "score": round(float(scores[i]) if i < len(scores) else 0, 3),
                            "type":  retrieval_types[i] if i < len(retrieval_types) else "semantic"
                        }
                        for i, d in enumerate(final_docs)
                    ]
                })

                save_chat(st.session_state.chat_id, st.session_state.messages, st.session_state.chat_title)
                st.session_state.last_chunks          = final_docs
                st.session_state.last_scores          = scores
                st.session_state.last_retrieval_types = retrieval_types
                st.rerun()

    with ctx_col:
        st.markdown("""
        <div style="padding:0.5rem 0 0.8rem 0.5rem;border-bottom:1px solid #1c1c28;border-left:3px solid #38bdf8;margin-bottom:0.8rem;">
            <div style="color:#38bdf8;font-size:9px;letter-spacing:1.2px;font-weight:600;text-transform:uppercase;">Context Window</div>
            <div style="color:#3d3d52;font-size:8px;">Chunks fed to LLM</div>
        </div>
        """, unsafe_allow_html=True)

        CHUNK_COLORS = {
            "keyword bypass": "#f59e0b",
            "semantic":       "#818cf8",
            "reranked":       "#34d399",
            "keyword match":  "#f472b6",
        }

        last_msg = next(
            (m for m in reversed(st.session_state.messages) if m["role"] == "assistant"),
            None
        )

        if last_msg and "context_chunks" in last_msg:
            chunks = last_msg["context_chunks"]
            total_chars = sum(len(c["text"]) for c in chunks)
            est_tokens  = total_chars // 4
            bar_w = min(int(est_tokens / 80), 100)

            st.markdown(f"""
            <div style="background:#111119;border:1px solid #1c1c28;border-radius:8px;padding:0.6rem;margin-bottom:0.5rem;">
                <div style="color:#3d3d52;font-size:8px;letter-spacing:1px;margin-bottom:6px;">TOKEN USAGE</div>
                <div style="background:#1c1c28;border-radius:3px;height:5px;margin-bottom:4px;">
                    <div style="background:#38bdf8;width:{bar_w}%;height:5px;border-radius:3px;opacity:0.7;"></div>
                </div>
                <div style="color:#4b4b60;font-size:8px;">~{est_tokens} context tokens · {len(chunks)} parent chunks</div>
            </div>
            """, unsafe_allow_html=True)

            for i, chunk in enumerate(chunks):
                color      = CHUNK_COLORS.get(chunk["type"], "#818cf8")
                type_label = chunk["type"]
                raw_score  = chunk["score"]
                score_bar  = int(min(abs(raw_score) / 15 * 100, 100))

                st.markdown(f"""
                <div class="chunk-card" style="border-left-color:{color};">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
                        <span style="color:{color};font-weight:600;font-size:0.75rem;">Chunk {i+1} · p.{chunk['page']}</span>
                        <span style="color:{color};font-size:0.7rem;background:#0e0e16;padding:1px 5px;border-radius:3px;">{raw_score:.2f}</span>
                    </div>
                    <div style="color:#4b4b60;font-size:0.72rem;margin-bottom:4px;">{chunk['file'][:24]}</div>
                    <div style="color:#6b7280;font-size:0.75rem;line-height:1.5;">{chunk['text'][:220]}...</div>
                    <div style="background:#1c1c28;border-radius:2px;height:3px;margin-top:6px;">
                        <div style="background:{color};width:{score_bar}%;height:3px;border-radius:2px;opacity:0.5;"></div>
                    </div>
                    <div style="color:#3d3d52;font-size:0.68rem;margin-top:3px;">{type_label}</div>
                </div>
                """, unsafe_allow_html=True)

            all_text = "\n\n---\n\n".join(
                f"[Chunk {i+1} | {c['file']} p.{c['page']} | {c['type']}]\n{c['text']}"
                for i, c in enumerate(chunks)
            )
            st.download_button(
                label="⬇ Export chunks",
                data=all_text,
                file_name=f"context_{st.session_state.chat_id}.txt",
                mime="text/plain",
                use_container_width=True
            )
        else:
            st.markdown("""
            <div style="text-align:center;padding:2rem 0.5rem;color:#2a2a38;">
                <div style="font-size:1.5rem;margin-bottom:0.5rem;">📭</div>
                <div style="font-size:0.78rem;">Ask a question to see<br>chunks fed to the LLM</div>
            </div>
            """, unsafe_allow_html=True)