# 🧠 DocChat AI

A production-grade RAG (Retrieval Augmented Generation) application that lets you chat with your documents — completely free, zero cost.

## Features

- 📁 Multi-file upload (PDF, Excel, Word)
- 🤝 Hybrid retrieval (FAISS semantic + BM25 keyword)
- 🎯 Cross-encoder reranking with BM25 bypass
- 📄 Parent Document Retriever (precise search, rich context)
- 🔍 Query expansion + conversation-aware rewriting
- ✅ Hallucination checking
- 💬 Persistent chat history
- 📚 Source citations with page numbers
- ➕ Incremental document ingestion
- ⚡ Streaming responses

## Tech Stack

| Layer | Tool | Cost |
|---|---|---|
| LLM | Groq (gpt-oss-120b) | Free |
| Embeddings | HuggingFace (MiniLM) | Free |
| Vector DB | FAISS | Free |
| Keyword Search | BM25 | Free |
| Reranker | CrossEncoder (MiniLM) | Free |
| Framework | LangChain | Free |
| UI | Streamlit | Free |

**Total cost: $0/month**

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/docchat-ai.git
cd docchat-ai
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
```bash
cp .env.example .env
# Edit .env and add your Groq API key
```

### 5. Run the app
```bash
streamlit run app.py
```

## Get a Free Groq API Key
1. Go to [console.groq.com](https://console.groq.com)
2. Sign up for free
3. Create an API key
4. Paste it in your `.env` file

## Architecture

