# Agentic RAG API

Production-level API for an Agentic RAG (Retrieval-Augmented Generation) system with web crawling and intelligent retrieval capabilities.

## Features

- 🤖 **Intelligent Agent**: LangGraph-powered agent that autonomously crawls websites and retrieves information
- 🌐 **Web Crawling**: Advanced web scraping using Crawl4AI with JavaScript support
- 🔍 **RAG Retrieval**: Vector-based semantic search using HuggingFace embeddings
- ⚡ **FastAPI**: Production-ready REST API with automatic OpenAPI documentation
- 🛡️ **Robust Error Handling**: Comprehensive error handling and logging throughout
- ⚙️ **Configuration Management**: Pydantic-based settings with environment variable support

## Architecture

```
```
WebRetrieve_Autonoma/
├── api/                    # FastAPI application
│   ├── main.py            # API endpoints and app setup
│   ├── models.py          # Pydantic request/response models
│   └── __init__.py
├── utils/                 # Core utilities
│   ├── nodes.py          # LangGraph nodes
│   ├── tools.py          # Agent tools (crawler, RAG)
│   └── state.py          # Agent state definition
├── agent.py              # LangGraph workflow definition
└── config.py             # Configuration management
```

## Quick Start

### Installation

```bash
# Install dependencies using uv (recommended)
uv sync # Install required browser dependencies
uv run crawl4ai-setup

# Verify the installation and system compatibility
uv run crawl4ai-doctor

# Or using pip
pip install -e .
```

### Configuration

Create a `.env` file in the project root:

```env
# Required
GROQ_API_KEY=your_groq_api_key_here

# Optional - defaults shown
LLM_MODEL_NAME=llama-3.3-70b-versatile
LLM_TEMPERATURE=0.0
LLM_RATE_LIMIT_RPS=0.1
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
CHUNK_SIZE=4000
CHUNK_OVERLAP=200
RAG_TOP_K=5
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
```

### Running the API

```bash
# Using Python module with uv
uv run python -m WebRetrieve_Autonoma.api.main

# Or using Uvicorn directly with uv
uv run uvicorn WebRetrieve_Autonoma.api.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Endpoints

### Health Check

```bash
GET /health
```

Response:
```json
{
  "status": "healthy",
  "timestamp": "2026-01-15T17:42:00Z",
  "version": "0.1.0"
}
```

### Query Agent

```bash
POST /query
Content-Type: application/json

{
  "query": "Go to https://crawl4ai.com/ and tell me about its main features."
}
```

Response:
```json
{
  "success": true,
  "messages": [...],
  "final_answer": "Based on crawling crawl4ai.com, here are the main features...",
  "timestamp": "2026-01-15T17:42:00Z"
}
```

## Usage Example

### Using cURL

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "Go to https://example.com and summarize the content"}'
```

### Using Python

```python
import httpx
import asyncio

async def query_agent():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/query",
            json={"query": "Tell me about https://crawl4ai.com features"}
        )
        print(response.json()["final_answer"])

asyncio.run(query_agent())
```

## Testing

Run the test suite:

```bash
# Make sure the API server is running first
uv run python -m WebRetrieve_Autonoma.api.main

# In another terminal, run tests
uv run python tests/test_api.py
```

## How It Works

1. **Query Submission**: User submits a research query via `/query` endpoint
2. **LLM Planning**: Agent analyzes the query and decides which tools to use
3. **Web Crawling**: Uses `web_crawler_tool` to fetch and index website content
4. **Link Discovery**: Identifies relevant internal links for deeper research
5. **RAG Retrieval**: Uses `rag_retrieval_tool` to search indexed content
6. **Synthesis**: Agent synthesizes information and provides cited answers

## Configuration Options

All settings can be configured via environment variables or `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | (required) | Groq API key for LLM access |
| `LLM_MODEL_NAME` | `llama-3.3-70b-versatile` | LLM model to use |
| `LLM_TEMPERATURE` | `0.0` | LLM temperature for responses |
| `LLM_RATE_LIMIT_RPS` | `0.1` | Requests per second for LLM |
| `EMBEDDING_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `CHUNK_SIZE` | `4000` | Text chunk size for splitting |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `RAG_TOP_K` | `5` | Number of documents to retrieve |
| `API_HOST` | `0.0.0.0` | API server host |
| `API_PORT` | `8000` | API server port |
| `LOG_LEVEL` | `INFO` | Logging level |

## Production Deployment

For production deployment:

1. Set appropriate CORS origins in `WebRetrieve_Autonoma/api/main.py`
2. Use a production ASGI server like Gunicorn with Uvicorn workers
3. Set up proper logging and monitoring
4. Configure rate limiting and authentication as needed
5. Use environment variables for sensitive configuration

Example production command:
```bash
gunicorn WebRetrieve_Autonoma.api.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --log-level info
```

## Development

The codebase follows production-level best practices:

- **Type Hints**: Comprehensive type annotations throughout
- **Error Handling**: Try-except blocks with proper logging
- **Configuration**: Centralized settings management with Pydantic
- **Logging**: Structured logging at appropriate levels
- **Validation**: Pydantic models for request/response validation
- **Documentation**: Docstrings and inline documentation

## License

MIT
