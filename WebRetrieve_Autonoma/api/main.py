import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from WebRetrieve_Autonoma.agent import graph
from WebRetrieve_Autonoma.config import settings
from WebRetrieve_Autonoma.api.models import (
    HealthResponse,
    StreamingQueryRequest,
)


logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events."""
    logger.info("Starting up Agentic RAG API")
    yield
    logger.info("Shutting down Agentic RAG API")


app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description="Production-level API for Agentic RAG system with web crawling and retrieval capabilities",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",  # For development
        "http://localhost:5173",  # Vite default
        "http://localhost:3000",  # Alternative React port
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint to verify the API is running.

    Returns:
        HealthResponse with status and version information
    """
    logger.info("Health check requested")
    return HealthResponse(status="healthy", version=settings.api_version)


@app.get("/models", tags=["Configuration"])
async def get_models():
    """
    Get list of supported models from Groq, GitHub, and OpenRouter.
    """
    try:
        import json
        import os
        
        models_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models.json")
        
        if os.path.exists(models_file_path):
            with open(models_file_path, 'r') as f:
                models_data = json.load(f)
                return models_data
        else:
            logger.warning(f"Models file not found at {models_file_path}")
            models = [
                {"id": "groq/llama-3.1-8b-instant", "name": "Llama 3.1 8B Instant", "provider": "Groq"},
                {"id": "github/gpt-4o", "name": "GPT-4o", "provider": "GitHub"},
            ]
            return {"models": models}
            
    except Exception as e:
        logger.error(f"Error reading models file: {e}")
        # Fallback to basic models on error
        models = [
            {"id": "groq/llama-3.1-8b-instant", "name": "Llama 3.1 8B Instant", "provider": "Groq"},
            {"id": "github/gpt-4o", "name": "GPT-4o", "provider": "GitHub"},
        ]
        return {"models": models}


@app.post("/stream-query", tags=["Agent"])
async def stream_query(request: StreamingQueryRequest):
    """
    Stream query execution with real-time progress updates using Server-Sent Events.

    This endpoint streams progress events as they occur during agent execution,
    including tool calls, content fetching, embedding creation, and more.

    Args:
        request: StreamingQueryRequest with query and optional session_id

    Returns:
        StreamingResponse with SSE events
    """
    logger.info(f"Received streaming query: {request.query[:100]}... Model: {request.model or 'DEFAULT'}")

    async def event_generator():
        try:
            import json

            progress_data = json.dumps(
                {
                    "event_type": "status",
                    "message": "Starting agent...",
                    "progress_percent": 10,
                }
            )
            yield f"event: progress\ndata: {progress_data}\n\n"

            inputs = {"messages": [HumanMessage(content=request.query)]}
            config = {"configurable": {"thread_id": request.thread_id}}
            
            if request.model:
                 config["configurable"]["model"] = request.model

            progress = 20

            # Pass config with recursion_limit here
            final_config = {
                "configurable": config["configurable"],
                "recursion_limit": 30 # Increased to allow for deep dive research
            }

            # Initial signal the stream has started
            init_data = json.dumps({"event_type": "status", "message": "Thinking...", "progress_percent": 10})
            yield f"event: progress\ndata: {init_data}\n\n"

            async for event in graph.astream_events(inputs, config=final_config, version="v2"):
                kind = event["event"]
                
                if kind == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    if hasattr(chunk, "content") and chunk.content:
                        token_data = json.dumps({
                            "event_type": "token",
                            "message": str(chunk.content),
                            "role": "ai"
                        })
                        yield f"event: token\ndata: {token_data}\n\n"
                
                elif kind == "on_tool_start":
                    tool_name = event["name"]
                    tool_data = json.dumps({
                        "event_type": "tool_start",
                        "message": f"Using {tool_name}...",
                        "tool_name": tool_name,
                        "progress_percent": 60,
                    })
                    yield f"event: progress\ndata: {tool_data}\n\n"
                
                elif kind == "on_tool_end":
                    tool_name = event["name"]
                    tool_data = json.dumps({
                        "event_type": "tool_complete",
                        "message": f"Finished using {tool_name}",
                        "tool_name": tool_name,
                        "progress_percent": 80,
                    })
                    yield f"event: progress\ndata: {tool_data}\n\n"
                
                elif kind == "on_chat_model_end":
                    # Final message from a model call (e.g. tool calls or final answer)
                    msg = event["data"]["output"]
                    # Be inclusive of 'ai' or 'assistant' types
                    if hasattr(msg, "type") and msg.type in ["ai", "assistant"]:
                        message_data = json.dumps({
                            "role": "ai",
                            "content": msg.content,
                            "model": request.model,
                            "tool_calls": getattr(msg, "tool_calls", None),
                        })
                        yield f"event: message\ndata: {message_data}\n\n"

            # Final completion signal
            complete_data = json.dumps(
                {
                    "event_type": "complete",
                    "message": "Query complete",
                    "progress_percent": 100,
                }
            )
            yield f"event: progress\ndata: {complete_data}\n\n"
            yield f"event: done\ndata: {json.dumps({'status': 'success'})}\n\n"

        except Exception as e:
            logger.exception("Error during streaming")
            error_data = json.dumps(
                {"event_type": "error", "message": f"Error: {str(e)}"}
            )
            yield f"event: error\ndata: {error_data}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Agentic RAG API",
        "version": settings.api_version,
        "endpoints": {
            "health": "/health",
            "health": "/health",
            "stream_query": "/stream-query",
            "docs": "/docs",
        },
    }



if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "WebRetrieve_Autonoma.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level=settings.log_level.lower(),
    )
