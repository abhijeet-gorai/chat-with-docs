# Chat With Your Docs

A document question-answering system built using Retrieval-Augmented Generation (RAG) with two-stage retrieval (vector similarity + reranking) for improved answer quality.

## Quick Setup

### Prerequisites
- Python 3.12+
- [uv](https://docs.astral.sh/uv/getting-started/installation)
- Groq API key (free tier available at [console.groq.com](https://console.groq.com))

### Installation

```bash
# Clone the repository
git clone https://github.com/abhijeet-gorai/chat-with-docs
cd chat-with-docs

# Create and activate virtual environment with uv
uv sync
```

### Configuration

1. Create a `.env` file in the project root:
```bash
GROQ_APIKEY=your_groq_api_key_here
```

2. (Optional) Modify `rag_pipelines/reranker_rag/config.yaml` to adjust:
   - LLM model and parameters
   - Embedding model
   - Chunking strategy
   - Reranker type (ColBERT or Cross-Encoder)

### Running the Application

```bash
uv run app.py
```

The Gradio interface will be available at `http://localhost:7860`

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                    Gradio UI                                    │
│  ┌─────────────────────┐    ┌─────────────────────────────────────────────────┐ │
│  │   Sidebar           │    │   Main Area                                     │ │
│  │   ─────────         │    │   ─────────                                     │ │
│  │   • Model Select    │    │   [Q&A Tab]          [Add Documents Tab]        │ │
│  │   • Temperature     │    │   • Collection       • Collection Select        │ │
│  │   • Max Tokens      │    │   • Query Input      • New Collection Name      │ │
│  │   • Top P           │    │   • Answer Output    • File Upload (PDF/DOCX)   │ │
│  │   • Apply Settings  │    │   • Source Display   • Upload Status            │ │
│  └─────────────────────┘    └─────────────────────────────────────────────────┘ │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              RerankerRAG Pipeline                               │
│                                                                                 │
│   ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐          │
│   │   Document       │    │   Two-Stage      │    │   Answer         │          │
│   │   Ingestion      │    │   Retrieval      │    │   Generation     │          │
│   │   ─────────────  │    │   ─────────────  │    │   ─────────────  │          │
│   │   • File Reader  │    │   • Vector       │    │   • Context      │          │
│   │   • Chunking     │───▶│     Similarity   │───▶│     Formatting   │          │
│   │   • Dedup        │    │   • Reranking    │    │   • LLM Response │          │
│   │   • Embedding    │    │     (k*3 → k)    │    │   • Streaming    │          │
│   └──────────────────┘    └──────────────────┘    └──────────────────┘          │
└──────────┬──────────────────────┬──────────────────────────┬────────────────────┘
           │                      │                          │
           ▼                      ▼                          ▼
┌──────────────────┐    ┌────────────────────┐       ┌──────────────────┐
│   ChromaDB       │    │   Rerankers        │       │   Groq API       │
│   (Vector Store) │    │   ───────────────  │       │   (LLM Service)  │
│                  │    │   • Cross-Encoder  │       │                  │
│   sentence-      │    │   • ColBERT        │       │   OpenAI/Llama   │
│   transformers/  │    │                    │       │   models         │
│   all-MiniLM-    │    │                    │       │                  │
│   L6-v2          │    │                    │       │                  │
└──────────────────┘    └────────────────────┘       └──────────────────┘
```

### Component Overview

| Component | Technology | Purpose |
|-----------|------------|---------|
| UI | Gradio | Web interface for rapid prototyping |
| RAG Pipeline | Custom Python | Orchestrates retrieval and generation |
| Vector Store | ChromaDB | Persistent local vector database |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 | Document/query embedding |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 | Second-stage relevance scoring |
| LLM | Groq API (openai/gpt-oss-120b) | Answer generation |

> **Note**: Gradio was chosen for rapid prototyping. For production or scaling, I would use **FastAPI** for backend API endpoints and **React** for the frontend.

---

## Productionization & Scaling (AWS/GCP/Azure)

To deploy this solution at scale on a cloud provider:

### Infrastructure Changes

| Component | Current | Production Alternative |
|-----------|---------|------------------------|
| UI | Gradio | FastAPI (backend) + React (frontend) |
| Vector DB | Local ChromaDB | Milvus, Elasticsearch |
| LLM | Groq API | AWS Bedrock, Azure OpenAI, GCP Vertex AI, or self-hosted |
| Observability | None | Langfuse for LLM tracing and monitoring |

### Scaling Considerations

1. **Horizontal Scaling**: The stateless Gradio app can scale behind a load balancer. Vector DB and LLM are external services that handle their own scaling.

2. **Caching**: Add Redis/Memcached for:
   - Embedding cache (same documents don't need re-embedding)
   - LLM response cache for repeated queries

3. **Async Processing**: For large document uploads, use a job queue (Celery/SQS) for background processing.

4. **Monitoring**: Add Langfuse for LLM observability, tracing, and cost tracking.

5. **Auth/Multi-tenancy**: Add authentication layer and collection-per-user isolation.

---

## RAG/LLM Approach & Decisions

### Choices Considered and Final Decisions

#### LLM Provider
| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| OpenAI API | Best quality, widely used | Cost, rate limits | ❌ |
| Anthropic | Great reasoning | Cost | ❌ |
| Gemini API | Good quality, generous free tier | API key requires Google Cloud | ❌ |
| **Groq** | Free tier, ultra-fast inference, Llama/OpenAI OSS models | Smaller model selection | ✅ Selected |
| Ollama (local) | No API costs, privacy | Requires good hardware, slower | ❌ |

**Rationale**: Groq provides free-tier access with excellent inference speed and easy API key generation. The availability of Llama 4 and OpenAI GPT OSS models made it ideal for a demonstration. For production, I'd recommend AWS Bedrock or Azure OpenAI for SLAs and enterprise support.

#### Embedding Model
| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| **all-MiniLM-L6-v2** | Small, fast, good quality, local | Not best-in-class | ✅ Selected |

**Rationale**: `all-MiniLM-L6-v2` is small (~80MB) and fast, making it ideal for quick prototyping. HuggingFace offers many embedding models—the optimal choice depends on the data type (code, legal, medical, etc.). For domain-specific use cases, I'd evaluate models from the [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard).

#### Vector Database
| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| Milvus | Scalable, production-ready | Complex setup, resource heavy | For production |
| Elasticsearch | Full-text + vector search | Heavier, more config | For hybrid search |
| FAISS | Fast, in-memory | No persistence OOTB | For benchmarking |
| **ChromaDB** | Free, persistent, simple API | Single-node only | ✅ Selected |

**Rationale**: ChromaDB provides persistence with zero configuration, making it perfect for a single-node prototype. For production, I'd use Milvus or Elasticsearch depending on scale and hybrid search requirements.

#### Orchestration Framework
| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| LangGraph | Great for agentic/conversational RAG | Overkill for simple RAG | For future |
| LlamaIndex | RAG-focused | Different paradigm | ❌ |
| **LangChain** | Flexible, good abstractions | Can be verbose | ✅ Selected |

**Rationale**: Used LangChain selectively—just the document loaders, text splitters, and message types. Avoided high-level chains to keep full control. For agentic RAG or conversational RAG with memory, I'd use LangGraph.

#### Reranking Strategy
| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| No reranking | Simpler | Lower precision | ❌ |
| LLM-as-reranker | High quality | Slow, expensive | ❌ |
| **Cross-Encoder** | Fast, accurate | Needs model download | ✅ Default |
| ColBERT | Token-level matching | More complex | ✅ Available |

**Rationale**: Two-stage retrieval (vector + reranker) significantly improves relevance. I retrieve 3× the needed documents, then rerank to top-k, balancing recall and precision.

### Chunking Strategy

```yaml
chunk_size: 512
chunk_overlap: 50
separators: ["\n\n", " "]
```

**Rationale**:
- 512 tokens balances context window usage with retrieval granularity
- 50-token overlap prevents information loss at chunk boundaries
- `RecursiveCharacterTextSplitter` respects paragraph boundaries when possible

**Alternative Chunking Strategies** (not implemented, but worth considering):
| Strategy | Best For | Trade-off |
|----------|----------|----------|
| Semantic Chunking | Variable-length logical units | Requires embedding calls during chunking |
| Sentence-based | Q&A with precise answers | May lose context across sentences |
| Document-structure aware | PDFs with clear sections/headers | Requires document parsing logic |
| Agentic Chunking | LLM decides chunk boundaries | Expensive, slow |

### Prompt Engineering

The system uses a carefully crafted prompt (`prompts/rag_prompts.py`):

```
Key prompt principles:
1. Ground responses in provided context only
2. Explicit instruction to not fabricate information
3. Built-in guardrail against harmful content
4. Natural response formatting (don't mention "the context says...")
5. Graceful fallback when answer isn't found
```

### Context Management

- Retrieved documents are concatenated with double newlines as separators
- Source documents are tracked and displayed to users
- Metadata (filename, page number) preserved through the pipeline

### Guardrails

1. **Input validation**: Empty queries rejected at UI level
2. **Content grounding**: Prompt instructs LLM to only use provided context
3. **Harmful content**: System prompt explicitly prohibits hate speech/profanity
4. **Graceful degradation**: "I don't have the answer" response for missing information

### Quality Controls

1. **Deduplication**: Documents are deduplicated by content hash before insertion
2. **Reranking**: Two-stage retrieval improves precision
3. **Streaming**: Real-time response streaming for better UX
4. **Source attribution**: Retrieved documents shown to users for verification

---

## Key Technical Decisions

### 1. Two-Stage Retrieval Architecture
**Decision**: Retrieve 3× documents with vector search, then rerank to top-k.

**Why**: Vector similarity alone often returns semantically similar but not necessarily relevant documents. Reranking with a cross-encoder model significantly improves answer quality with minimal latency impact (~50-100ms).

### 2. Local Models Where Possible
**Decision**: Use local embeddings (HuggingFace) and rerankers, only call API for LLM.

**Why**: Reduces API costs, eliminates rate limits for document processing, and provides offline capability for ingestion.

### 3. YAML-Based Configuration
**Decision**: All pipeline parameters configurable via `config.yaml`.

**Why**: Separation of configuration from code enables:
- Easy experimentation with different models
- Environment-specific configs (dev/staging/prod)
- No code changes for parameter tuning

### 4. Collection-Based Document Organization
**Decision**: Support multiple collections in ChromaDB.

**Why**: Enables use cases like:
- Different document sets for different contexts
- A/B testing retrieval strategies
- Multi-tenant isolation (with auth layer)

### 5. Streaming Responses
**Decision**: Stream LLM responses to UI.

**Why**: Reduces perceived latency significantly. Users see the first tokens within ~200ms instead of waiting 2-5 seconds for complete response.

### 6. Modular Reranker Interface
**Decision**: Abstract reranker behind a common interface.

**Why**: Easy to swap between ColBERT and Cross-Encoder, or add new rerankers (e.g., Cohere Rerank API) without changing pipeline code.

---

## Engineering Standards

### Followed

- **Type hints**: All function signatures include type annotations
- **Docstrings**: Comprehensive docstrings for all public methods
- **Configuration as code**: YAML config with TypedDict schema validation
- **Separation of concerns**: Clear boundaries between services, utils, and pipelines
- **Dependency management**: Modern `pyproject.toml` with `uv` for fast, reproducible installs
- **Git hygiene**: Comprehensive `.gitignore`, no secrets in repo
- **Error handling**: Try/except blocks with user-friendly error messages

### Skipped (Time Constraints)

- **Unit tests**: No test suite (would add pytest with mock LLM responses)
- **CI/CD**: No GitHub Actions pipeline
- **FastAPI + React**: Used Gradio for rapid prototyping; would use FastAPI backend + React frontend for production
- **Agentic RAG**: Would implement with LangGraph for multi-step reasoning and tool use

---

## What I'd Do Differently With More Time

### Priority 1: Testing & Quality
- [ ] Add pytest test suite with mock LLM/embeddings
- [ ] Add integration tests with a small test collection

### Priority 2: Observability
- [ ] Langfuse integration for LLM tracing and cost tracking
- [ ] Structured logging with correlation IDs
- [ ] Retrieval quality metrics dashboard

### Priority 3: Features
- [ ] Chat history / conversation memory
- [ ] Hybrid search (keyword + semantic)
- [ ] Document metadata filtering in queries
- [ ] Support for more file types (HTML, Markdown, CSV)
- [ ] Chunk visualization for debugging

### Priority 4: Production Readiness
- [ ] FastAPI backend + React frontend
- [ ] GitHub Actions CI pipeline
- [ ] Health check endpoint
- [ ] Rate limiting
- [ ] Authentication layer
- [ ] Async document processing with background jobs

### Priority 5: Advanced RAG
- [ ] Agentic RAG with LangGraph for multi-step reasoning
- [ ] Conversational RAG with chat history
- [ ] Hypothetical document embeddings (HyDE)
- [ ] Query expansion/reformulation

---

## Known Edge Cases / Limitations

1. **Large documents**: Memory constraints with very large PDFs on machines with limited RAM
2. **Table extraction**: Table content from PDFs may not preserve structure well
3. **Concurrent writes**: ChromaDB single-node doesn't handle concurrent writes gracefully
4. **Non-English content**: Current embedding model optimized for English

---

## Project Structure

```
chat-with-docs/
├── app.py                          # Gradio UI entry point
├── pyproject.toml                  # Dependencies and project metadata
├── config.yaml                     # Pipeline configuration (in rag_pipelines/)
├── prompts/
│   └── rag_prompts.py              # System and user prompt templates
├── rag_pipelines/
│   └── reranker_rag/
│       ├── __init__.py
│       ├── config.yaml             # Default configuration
│       ├── config_schema.py        # TypedDict schemas for config
│       └── rag_with_reranker.py    # Main RAG pipeline class
├── services/
│   ├── groq.py                     # Groq LLM initialization
│   └── vector_db/
│       └── chroma.py               # ChromaDB wrapper class
├── utils/
│   ├── file_readers.py             # PDF/DOCX/TXT document loaders
│   └── rerankers/
│       ├── colbert.py              # ColBERT reranker implementation
│       └── cross_encoder.py        # Cross-encoder reranker
├── data/                           # Sample documents (not committed)
└── chroma_db/                      # Persistent vector store (gitignored)
```

---

## License

MIT License - feel free to use and modify.
