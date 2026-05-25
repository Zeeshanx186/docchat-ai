<div align="center">

# 🧠 DocChat AI

**Chat with your documents. Powered by a production-grade Hybrid RAG pipeline. Completely free.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-0.2+-green?style=flat)](https://langchain.com)
[![Cost](https://img.shields.io/badge/Cost-$0%2Fmonth-brightgreen?style=flat)](https://console.groq.com)

</div>

---

## What is this?

DocChat AI is not a wrapper around a single `similarity_search()` call. It's a multi-stage retrieval pipeline that combines semantic search, keyword search, cross-encoder reranking, parent document retrieval, query rewriting, and hallucination detection — all running locally or on free-tier APIs.

Upload a PDF, Excel, or Word file. Ask anything. Get precise, cited answers.

---

## Pipeline Architecture

```
Document Upload
    │
    ▼
Chunking (Parent + Child splits)
    │
    ├──► FAISS index (dense vectors)
    └──► BM25 index (sparse keyword)
                        │
User Query ─► Rewrite + Expand ─► Hybrid Retrieval
                                        │
                                  ┌─────┴──────┐
                               Semantic      Keyword
                               (MMR·FAISS)  (BM25 bypass)
                                  └─────┬──────┘
                                        │
                               Cross-encoder Rerank
                                        │
                               Parent Doc Retrieval
                                        │
                               LLM Generate (streaming)
                                        │
                               Hallucination Check ✓
```

**Why this matters:** Most RAG demos retrieve 3-5 chunks and call it done. This pipeline retrieves small child chunks for precision, then fetches their large parent chunks for rich context — so the LLM gets the full picture, not just a fragment.

---

## Features

| Feature | What it does |
|---|---|
| **Hybrid retrieval** | FAISS semantic (MMR) + BM25 keyword search combined with ensemble weighting |
| **Parent Document Retrieval** | Indexes small chunks, retrieves large parent context for the LLM |
| **Cross-encoder reranking** | `ms-marco-MiniLM-L-6-v2` re-scores results post-retrieval for higher relevance |
| **Query rewriting** | Rewrites ambiguous follow-up questions using conversation history |
| **Keyword expansion** | Expands query with 8-10 semantically related terms before retrieval |
| **Hallucination detection** | Every response is fact-checked against source documents (GROUNDED / PARTIAL / HALLUCINATED) |
| **Source citations** | Every answer cites file name, page number, and confidence score |
| **Context window viewer** | See exactly which chunks were fed to the LLM, with retrieval type and scores |
| **Streaming responses** | Token-by-token output, no waiting for the full response |
| **Persistent chat history** | Conversations saved per session, resumable anytime |
| **Incremental ingestion** | Add new documents to an existing index without rebuilding from scratch |
| **Multi-format support** | PDF, Excel (.xlsx/.xls), Word (.docx/.doc) |

---

## Tech Stack — $0/month

| Layer | Tool | Why |
|---|---|---|
| LLM | [Groq](https://console.groq.com) (gpt-oss-120b) | Fastest free inference available |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` | Runs locally, no API cost |
| Vector DB | FAISS | Local, persistent, no server needed |
| Keyword Search | BM25Retriever | Exact keyword matching with no overhead |
| Reranker | CrossEncoder `ms-marco-MiniLM-L-6-v2` | Local reranking, significant precision boost |
| Framework | LangChain | Retriever orchestration |
| UI | Streamlit | Clean chat interface with sidebar |

---

## Setup

### 1. Clone
```bash
git clone https://github.com/Zeeshanx186/docchat-ai.git
cd docchat-ai
```

### 2. Virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your Groq API key
```bash
cp .env.example .env
# Open .env and set: GROQ_API_KEY=your_key_here
```

### 5. Run
```bash
streamlit run app.py
```

---

## Get a Free Groq API Key

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up (free, no credit card)
3. Create an API key
4. Paste it into `.env`

Groq's free tier is fast enough for production use.

---

## How the RAG pipeline works

**1. Ingestion**
Documents are split into two levels: large *parent* chunks (3000 tokens) for context richness, and small *child* chunks (200 tokens) for precision retrieval. Only child chunks are indexed; parent chunks are stored separately by ID.

**2. Query processing**
Before retrieval, the query is:
- Rewritten to resolve pronouns and references using conversation history
- Expanded with 8-10 related keywords to improve recall

**3. Hybrid retrieval**
Two retrievers run in parallel:
- **FAISS (MMR)**: Finds semantically similar chunks, with diversity to avoid redundancy
- **BM25**: Finds chunks with exact keyword matches — critical for technical terms, names, numbers

**4. Reranking + parent fetch**
A cross-encoder scores every retrieved chunk against the original question. The top-ranked child chunks are mapped back to their parent chunks, giving the LLM full paragraphs instead of fragments.

**5. Generation + verification**
The LLM generates a streamed answer. A second LLM call then checks whether the answer is grounded in the retrieved context, labeling it GROUNDED, PARTIAL, or HALLUCINATED.

---

## Project structure

```
docchat-ai/
├── app.py              # Main Streamlit application
├── requirements.txt
├── .env.example
└── README.md
```

---

## Author

**Zeeshan Yaqoob** — AI Engineer  
[GitHub](https://github.com/Zeeshanx186) · [LinkedIn](https://linkedin.com/in/zeeshanyaqoob) · [zeeshanyaqoob999@gmail.com](mailto:zeeshanyaqoob999@gmail.com)

---

<div align="center">
<sub>Built as a portfolio project demonstrating production-grade RAG architecture. Zero cloud costs.</sub>
</div>
