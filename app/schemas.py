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