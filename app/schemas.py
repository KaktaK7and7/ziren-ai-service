from typing import List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    user_id: int
    message: str = Field(min_length=1, max_length=10000)
    session_id: Optional[int] = None


class ChatResponse(BaseModel):
    answer: str
    session_id: int
    memory_updated: bool
    summary_updated: bool
    memory_logs: List[str]


class HealthResponse(BaseModel):
    status: str
    app: str


class PersonaPresetRequest(BaseModel):
    preset_name: str


class MemoryItemCreateRequest(BaseModel):
    type: str = Field(default="semantic", min_length=1, max_length=50)
    category: str = Field(default="general", min_length=1, max_length=100)
    content: str = Field(min_length=1, max_length=10000)
    source_message: Optional[str] = Field(default=None, max_length=10000)
    importance: float = Field(default=0.5, ge=0, le=1)
    confidence: float = Field(default=0.8, ge=0, le=1)
    sensitivity: str = Field(default="normal", min_length=1, max_length=50)
    status: str = Field(default="active", min_length=1, max_length=50)


class MemoryItemUpdateRequest(BaseModel):
    type: Optional[str] = Field(default=None, min_length=1, max_length=50)
    category: Optional[str] = Field(default=None, min_length=1, max_length=100)
    content: Optional[str] = Field(default=None, min_length=1, max_length=10000)
    source_message: Optional[str] = Field(default=None, max_length=10000)
    importance: Optional[float] = Field(default=None, ge=0, le=1)
    confidence: Optional[float] = Field(default=None, ge=0, le=1)
    sensitivity: Optional[str] = Field(default=None, min_length=1, max_length=50)
    status: Optional[str] = Field(default=None, min_length=1, max_length=50)
