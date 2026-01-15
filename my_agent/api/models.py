from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field
from datetime import datetime
import uuid


class QueryRequest(BaseModel):
    """Request model for agent query endpoint."""
    
    query: str = Field(
        ...,
        description="The research query to send to the agent",
        min_length=1,
        max_length=5000,
        examples=["Tell me about crawl4ai features from crawl4ai.com"]
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "Go to https://crawl4ai.com/ and tell me how it handles JavaScript execution."
            }
        }


class LinkSubmissionRequest(BaseModel):
    """Request model for submitting initial links to research."""
    
    urls: List[str] = Field(
        ...,
        description="List of URLs to crawl and research",
        min_length=1,
        max_length=10,
        examples=[["https://crawl4ai.com", "https://crawl4ai.com/docs"]]
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "urls": ["https://crawl4ai.com", "https://example.com/docs"]
            }
        }


class LinkSubmissionResponse(BaseModel):
    """Response model for link submission."""
    
    session_id: str = Field(..., description="Unique session identifier")
    success: bool = Field(..., description="Whether the submission was successful")
    message: str = Field(..., description="Response message")
    urls_count: int = Field(..., description="Number of URLs accepted")


class StreamingQueryRequest(BaseModel):
    """Request model for streaming query endpoint."""
    
    query: str = Field(
        ...,
        description="The research query to send to the agent",
        min_length=1,
        max_length=5000
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Optional session ID for context"
    )


class ProgressEvent(BaseModel):
    """Model for progress events during agent execution."""
    
    event_type: Literal[
        "status", "tool_start", "tool_progress", "tool_complete", 
        "message", "error", "complete"
    ] = Field(..., description="Type of progress event")
    
    message: str = Field(..., description="Progress message")
    
    tool_name: Optional[str] = Field(
        default=None,
        description="Name of the tool being executed"
    )
    
    progress_percent: Optional[int] = Field(
        default=None,
        description="Progress percentage (0-100)"
    )
    
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Event timestamp"
    )


class StreamEvent(BaseModel):
    """Wrapper for Server-Sent Events."""
    
    event: str = Field(default="message", description="SSE event type")
    data: ProgressEvent = Field(..., description="Event data")
    
    def to_sse(self) -> str:
        """Convert to SSE format."""
        return f"event: {self.event}\ndata: {self.data.model_dump_json()}\n\n"


class MessageResponse(BaseModel):
    """Single message in the conversation."""
    
    role: str = Field(..., description="Role of the message sender")
    content: str = Field(..., description="Content of the message")
    tool_calls: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Tool calls made by the LLM"
    )


class QueryResponse(BaseModel):
    """Response model for agent query endpoint."""
    
    success: bool = Field(..., description="Whether the query was successful")
    messages: List[MessageResponse] = Field(
        ...,
        description="All messages in the conversation"
    )
    final_answer: Optional[str] = Field(
        default=None,
        description="The final answer from the agent"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if the query failed"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of the response"
    )


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    
    status: str = Field(..., description="Health status")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of the health check"
    )
    version: str = Field(..., description="API version")
