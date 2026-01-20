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
    Get list of supported models from Groq and local Ollama.
    """
    models = [
       
    {"id": "llama-3.1-8b-instant", "name": "Llama 3.1 8B Instant", "provider": "Groq"},
    {"id": "llama-3.3-70b-versatile", "name": "Llama 3.3 70B Versatile", "provider": "Groq"},
    {"id": "moonshotai/kimi-k2-instruct", "name": "Kimi K2 Instruct", "provider": "Groq"},
    {"id": "openai/gpt-oss-120b", "name": "GPT-OSS 120B", "provider": "Groq"},
    {"id": "openai/gpt-oss-20b", "name": "GPT-OSS 20B", "provider": "Groq"},
    {"id": "openai/gpt-oss-safeguard-20b", "name": "GPT-OSS Safeguard 20B", "provider": "Groq"},
    {"id": "qwen/qwen3-32b", "name": "Qwen 3 32B", "provider": "Groq"},
    {"id": "gpt-4o", "name": "GPT-4o", "provider": "GitHub"},
    {"id": "gpt-4o-mini", "name": "GPT-4o Mini", "provider": "GitHub"}
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

            async for output in graph.astream(inputs, config=final_config, stream_mode="updates"):
                for node_name, state_update in output.items():
                    logger.info(f"Node {node_name} executed")

                    if node_name == "tool_node":
                        node_progress = json.dumps(
                            {
                                "event_type": "tool_start",
                                "message": "Executing tools...",
                                "progress_percent": 60,
                            }
                        )
                        yield f"event: progress\ndata: {node_progress}\n\n"
                    elif node_name == "llm_call":
                        node_progress = json.dumps(
                            {
                                "event_type": "status",
                                "message": "LLM thinking...",
                                "progress_percent": 30,
                            }
                        )
                        yield f"event: progress\ndata: {node_progress}\n\n"
                    
                    if "generation" in state_update:
                        # We still keep the check for final answer extraction if needed by other components,
                        # but message sending is handled by the "messages" block below.
                        pass

                    if "messages" in state_update:
                        for msg in state_update["messages"]:
                            msg_content = (
                                str(msg.content)
                                if hasattr(msg, "content")
                                else str(msg)
                            )
                            msg_type = getattr(msg, "type", "unknown")

                            # Send tool call events
                            if hasattr(msg, "tool_calls") and msg.tool_calls:
                                for tool_call in msg.tool_calls:
                                    tool_name = tool_call.get("name", "unknown")
                                    tool_event = json.dumps(
                                        {
                                            "event_type": "tool_start",
                                            "message": f"Using {tool_name}...",
                                            "tool_name": tool_name,
                                            "progress_percent": progress,
                                        }
                                    )
                                    yield f"event: progress\ndata: {tool_event}\n\n"

                            if msg_type in ["ai", "human", "tool"]:
                                message_data = json.dumps(
                                    {
                                        "role": msg_type,
                                        "content": msg_content,
                                        "tool_calls": (
                                            msg.tool_calls
                                            if hasattr(msg, "tool_calls")
                                            else None
                                        ),
                                    }
                                )
                                yield f"event: message\ndata: {message_data}\n\n"

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
