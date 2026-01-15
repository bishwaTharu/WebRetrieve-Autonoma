import logging
import uuid
from contextlib import asynccontextmanager
from typing import Dict
from datetime import datetime
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from my_agent.agent import graph
from my_agent.config import settings
from my_agent.api.models import (
    QueryRequest,
    QueryResponse,
    MessageResponse,
    HealthResponse,
    LinkSubmissionRequest,
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


sessions: Dict[str, Dict] = {}

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


@app.post("/query", response_model=QueryResponse, tags=["Agent"])
async def query_agent(request: QueryRequest):
    """
    Send a research query to the agent.

    The agent will:
    1. Crawl provided URLs using web_crawler_tool
    2. Analyze content and discover internal links
    3. Use RAG retrieval to answer specific questions
    4. Provide comprehensive, cited answers

    Args:
        request: QueryRequest containing the research query

    Returns:
        QueryResponse with the agent's analysis and answer

    Raises:
        HTTPException: If the query fails
    """
    logger.info(f"Received query: {request.query[:100]}...")

    try:
        inputs = {"messages": [HumanMessage(content=request.query)]}

        all_messages = []
        async for output in graph.astream(inputs, stream_mode="updates"):
            for node_name, state_update in output.items():
                logger.info(f"Node {node_name} executed")
                if "messages" in state_update:
                    all_messages.extend(state_update["messages"])

        messages_response = []
        final_answer = None

        for msg in all_messages:
            msg_dict = {
                "role": getattr(msg, "type", "unknown"),
                "content": str(msg.content) if hasattr(msg, "content") else str(msg),
            }

            if hasattr(msg, "tool_calls") and msg.tool_calls:
                msg_dict["tool_calls"] = msg.tool_calls

            messages_response.append(MessageResponse(**msg_dict))

            if msg_dict["role"] == "ai" and msg_dict.get("content"):
                final_answer = msg_dict["content"]

        logger.info("Query completed successfully")

        return QueryResponse(
            success=True, messages=messages_response, final_answer=final_answer
        )

    except Exception as e:
        logger.exception("Error processing query")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}",
        )


@app.post("/submit-links", tags=["Links"])
async def submit_links(request: LinkSubmissionRequest):
    """
    Submit initial URLs and immediately start crawling them.

    This endpoint accepts URLs, creates a session, and streams the crawling progress
    via Server-Sent Events. The frontend should display progress while crawling.

    Args:
        request: LinkSubmissionRequest with list of URLs

    Returns:
        StreamingResponse with crawling progress events
    """
    logger.info(f"Received link submission with {len(request.urls)} URLs")

    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "urls": request.urls,
        "created_at": datetime.utcnow().isoformat(),
    }

    async def crawl_generator():
        try:
            import json

            session_data = json.dumps(
                {"session_id": session_id, "urls_count": len(request.urls)}
            )
            yield f"event: session\ndata: {session_data}\n\n"

            from my_agent.utils.tools import AgentTools

            tools_instance = AgentTools()
            crawled_summaries = []

            total_urls = len(request.urls)
            for idx, url in enumerate(request.urls, 1):
                # Send progress event
                progress = int((idx - 1) / total_urls * 100)
                progress_data = json.dumps(
                    {
                        "message": f"Crawling {url}...",
                        "progress_percent": progress,
                        "url": url,
                    }
                )
                yield f"event: progress\ndata: {progress_data}\n\n"

                try:
                    result = await tools_instance._web_crawler_logic(url)
                    if result and len(result) > 0:
                        summary = result[:500] if len(result) > 500 else result
                        crawled_summaries.append({"url": url, "summary": summary})
                    success_data = json.dumps(
                        {
                            "url": url,
                            "success": True,
                            "message": "Successfully crawled and indexed",
                        }
                    )
                    yield f"event: crawl_complete\ndata: {success_data}\n\n"

                except Exception as e:
                    logger.exception(f"Error crawling {url}")
                    error_data = json.dumps({"url": url, "error": str(e)})
                    yield f"event: crawl_error\ndata: {error_data}\n\n"
            try:
                from langchain_groq import ChatGroq
                from my_agent.config import settings
                import re

                llm = ChatGroq(
                    model=settings.llm_model_name,
                    api_key=settings.groq_api_key,
                    temperature=0.7,
                )

                urls_text = "\n".join([f"- {url}" for url in request.urls])
                summaries_text = "\n\n".join(
                    [
                        f"URL: {s['url']}\nContent Preview: {s['summary'][:300]}..."
                        for s in crawled_summaries
                    ]
                )

                prompt = f"""Based on these URLs and their content, generate 4 intelligent, specific questions that would help users explore the information:

                            URLs:
                            {urls_text}

                            Content Previews:
                            {summaries_text}

                            Generate 4 concise questions (max 10 words each) that are:
                            1. Specific to the actual content
                            2. Interesting and useful
                            3. Different from each other

                            IMPORTANT: Return ONLY a valid JSON array, nothing else. No thinking, no explanation, no markdown.
                            Format: ["Question 1?", "Question 2?", "Question 3?", "Question 4?"]
                            """

                # Get questions from LLM
                response = await llm.ainvoke(prompt)
                questions_text = response.content.strip()

                # Parse JSON response with robust extraction
                try:
                    # Remove thinking tags if present
                    questions_text = re.sub(
                        r"<tool_call>.*?<tool_call>",
                        "",
                        questions_text,
                        flags=re.DOTALL,
                    )
                    if "```" in questions_text:
                        match = re.search(
                            r"```(?:json)?\s*(\[.*?\])\s*```", questions_text, re.DOTALL
                        )
                        if match:
                            questions_text = match.group(1)
                        else:
                            questions_text = questions_text.replace(
                                "```json", ""
                            ).replace("```", "")
                    json_match = re.search(r"\[.*?\]", questions_text, re.DOTALL)
                    if json_match:
                        questions_text = json_match.group(0)

                    # Clean up the text
                    questions_text = questions_text.strip()
                    questions = json.loads(questions_text)
                    if isinstance(questions, list) and len(questions) > 0:
                        valid_questions = [
                            q
                            for q in questions
                            if isinstance(q, str) and len(q.strip()) > 0
                        ][:4]

                        if len(valid_questions) > 0:
                            sessions[session_id][
                                "suggested_questions"
                            ] = valid_questions
                            questions_data = json.dumps({"questions": valid_questions})
                            yield f"event: questions\ndata: {questions_data}\n\n"
                            logger.info(
                                f"Generated {len(valid_questions)} questions successfully"
                            )
                        else:
                            raise ValueError("No valid questions extracted")
                    else:
                        raise ValueError("Response is not a valid list")

                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(
                        f"Failed to parse questions JSON: {questions_text[:200]}... Error: {str(e)}"
                    )
                    domain = (
                        request.urls[0].split("/")[2]
                        if len(request.urls) > 0
                        else "this website"
                    )
                    fallback_questions = [
                        f"What is {domain}?",
                        "What are the main features?",
                        "How do I get started?",
                        "Where can I find documentation?",
                    ]
                    sessions[session_id]["suggested_questions"] = fallback_questions
                    questions_data = json.dumps({"questions": fallback_questions})
                    yield f"event: questions\ndata: {questions_data}\n\n"

            except Exception as e:
                logger.exception("Error generating questions")

            complete_data = json.dumps(
                {
                    "message": "All URLs crawled successfully",
                    "progress_percent": 100,
                    "session_id": session_id,
                }
            )
            yield f"event: complete\ndata: {complete_data}\n\n"
            yield f"event: done\ndata: {json.dumps({'status': 'success'})}\n\n"

        except Exception as e:
            logger.exception("Error during crawling")
            error_data = json.dumps({"message": f"Error: {str(e)}"})
            yield f"event: error\ndata: {error_data}\n\n"

    return StreamingResponse(
        crawl_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


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
    logger.info(f"Received streaming query: {request.query[:100]}...")

    async def event_generator():
        try:
            import json

            # Send initial status
            progress_data = json.dumps(
                {
                    "event_type": "status",
                    "message": "Starting agent...",
                    "progress_percent": 10,
                }
            )
            yield f"event: progress\ndata: {progress_data}\n\n"

            inputs = {"messages": [HumanMessage(content=request.query)]}
            progress = 20

            async for output in graph.astream(inputs, stream_mode="updates"):
                for node_name, state_update in output.items():
                    logger.info(f"Node {node_name} executed")

                    # Send node execution event
                    if node_name == "tool_node":
                        node_progress = json.dumps(
                            {
                                "event_type": "tool_start",
                                "message": "Executing tools...",
                                "progress_percent": progress,
                            }
                        )
                        yield f"event: progress\ndata: {node_progress}\n\n"
                        progress = min(progress + 20, 80)
                    elif node_name == "llm_call":
                        node_progress = json.dumps(
                            {
                                "event_type": "status",
                                "message": "LLM thinking...",
                                "progress_percent": progress,
                            }
                        )
                        yield f"event: progress\ndata: {node_progress}\n\n"
                        progress = min(progress + 10, 90)

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
            "query": "/query",
            "submit_links": "/submit-links",
            "stream_query": "/stream-query",
            "docs": "/docs",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "my_agent.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level=settings.log_level.lower(),
    )
