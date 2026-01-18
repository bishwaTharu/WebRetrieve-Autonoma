from typing import  Optional, Dict, Any, Literal
from pydantic import BaseModel, Field
from datetime import datetime
import uuid


class StreamingQueryRequest(BaseModel):
    """Request model for streaming query endpoint."""
    
    query: str = Field(
        ...,
        description="The research query to send to the agent",
        min_length=1,
        max_length=5000
    )
    thread_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique thread ID for conversation persistence"
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Optional session ID for context"
    )
    model: Optional[str] = Field(
        default=None,
        description="ID of the model to use for this request"
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



class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    
    status: str = Field(..., description="Health status")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of the health check"
    )
    version: str = Field(..., description="API version")
