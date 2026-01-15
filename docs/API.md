# API Documentation

## Overview

The Agentic RAG API provides intelligent web research capabilities through a REST interface. The agent autonomously crawls websites, indexes content, and provides comprehensive answers to research queries.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, no authentication is required. For production deployments, implement appropriate authentication mechanisms.

## Endpoints

### GET /health

Health check endpoint to verify API availability.

**Response**
```json
{
  "status": "healthy",
  "timestamp": "2026-01-15T17:42:00.000Z",
  "version": "0.1.0"
}
```

**Status Codes**
- `200 OK`: Service is healthy

---

### POST /query

Submit a research query to the agent.

**Request Body**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| query | string | Yes | Research query (1-5000 characters) |

**Example Request**
```json
{
  "query": "Go to https://crawl4ai.com/ and tell me about its features and pricing"
}
```

**Response**

| Field | Type | Description |
|-------|------|-------------|
| success | boolean | Whether the query was successful |
| messages | array | All messages in the agent conversation |
| final_answer | string | The final answer from the agent (last AI message) |
| error | string | Error message if failed (null if successful) |
| timestamp | string | ISO 8601 timestamp of the response |

**Example Response**
```json
{
  "success": true,
  "messages": [
    {
      "role": "human",
      "content": "Go to https://crawl4ai.com/ and tell me about its features",
      "tool_calls": null
    },
    {
      "role": "ai",
      "content": "",
      "tool_calls": [
        {
          "name": "web_crawler_tool",
          "args": {"url": "https://crawl4ai.com/"},
          "id": "call_123"
        }
      ]
    },
    {
      "role": "tool",
      "content": "✓ Successfully crawled and indexed https://crawl4ai.com/...",
      "tool_calls": null
    },
    {
      "role": "ai",
      "content": "Based on my analysis of crawl4ai.com, here are the main features...",
      "tool_calls": null
    }
  ],
  "final_answer": "Based on my analysis of crawl4ai.com, here are the main features...",
  "error": null,
  "timestamp": "2026-01-15T17:42:00.000Z"
}
```

**Status Codes**
- `200 OK`: Query processed successfully
- `422 Unprocessable Entity`: Invalid request body
- `500 Internal Server Error`: Server error during processing

---

### GET /

Root endpoint with API information.

**Response**
```json
{
  "message": "Agentic RAG API",
  "version": "0.1.0",
  "endpoints": {
    "health": "/health",
    "query": "/query",
    "docs": "/docs"
  }
}
```

## Agent Behavior

The agent follows this strategy:

1. **CRAWL**: When given a URL, uses `web_crawler_tool` to fetch and index content
2. **DEEP DIVE**: Analyzes internal links and crawls relevant pages for comprehensive research
3. **RETRIEVE**: Uses `rag_retrieval_tool` to search across all indexed content
4. **SYNTHESIZE**: Provides comprehensive answers with source citations

## Rate Limiting

The API implements LLM rate limiting:
- Default: 0.1 requests per second to the LLM
- Configurable via `LLM_RATE_LIMIT_RPS` environment variable

## Error Handling

All errors are returned in a consistent format:

```json
{
  "success": false,
  "messages": [],
  "final_answer": null,
  "error": "Error description",
  "timestamp": "2026-01-15T17:42:00.000Z"
}
```

## Interactive Documentation

Visit these URLs when the server is running:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

Both provide interactive API exploration and testing capabilities.
