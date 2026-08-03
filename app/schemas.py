from typing import Annotated, Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

DeliveredCompanionLine = Annotated[
    str,
    Field(min_length=1, max_length=600),
]


class ChatRequest(BaseModel):
    user_id: int
    message: str = Field(min_length=1, max_length=10000)
    session_id: Optional[int] = None
    drawing_enabled: bool = False
    story_mode_enabled: bool = True
    companion_name: Optional[str] = Field(
        default=None,
        min_length=2,
        max_length=32,
    )
    preceding_assistant_lines: List[DeliveredCompanionLine] = Field(
        default_factory=list,
        max_length=2,
    )
    story_context: Optional[str] = Field(default=None, max_length=6000)
    activity_context: Optional[str] = Field(default=None, max_length=3000)
    capability_context: Optional[str] = Field(default=None, max_length=5000)


class ScreenAnalysisRequest(BaseModel):
    user_id: int
    message: str = Field(min_length=1, max_length=2000)
    image_data_url: str = Field(min_length=100, max_length=1_800_000)
    session_id: Optional[int] = None
    story_mode_enabled: bool = True
    companion_name: Optional[str] = Field(
        default=None,
        min_length=2,
        max_length=32,
    )
    preceding_assistant_lines: List[DeliveredCompanionLine] = Field(
        default_factory=list,
        max_length=2,
    )
    story_context: Optional[str] = Field(default=None, max_length=6000)
    activity_context: Optional[str] = Field(default=None, max_length=3000)
    capability_context: Optional[str] = Field(default=None, max_length=5000)


class ScreenAnnotation(BaseModel):
    id: str = Field(min_length=1, max_length=40)
    label: str = Field(min_length=1, max_length=100)
    kind: Literal["target", "step", "text", "warning"]
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    width: float = Field(gt=0, le=1)
    height: float = Field(gt=0, le=1)
    step: int = Field(ge=0, le=8)


class ScreenActionProposal(BaseModel):
    type: Literal["none", "click"]
    target_id: str = Field(default="", max_length=40)
    label: str = Field(default="", max_length=100)
    risk: Literal["safe", "blocked"]
    reason: str = Field(default="", max_length=240)


class ScreenAnalysisPlan(BaseModel):
    answer: str = Field(min_length=1, max_length=5000)
    mode: Literal[
        "explain",
        "translate",
        "guide",
        "annotate",
    ]
    annotations: List[ScreenAnnotation] = Field(max_length=8)
    action: ScreenActionProposal


class ChatResponse(BaseModel):
    answer: str
    session_id: int
    memory_updated: bool
    summary_updated: bool
    memory_logs: List[str]
    story_signal: Optional[Dict[str, Any]] = None
    drawing_request: Optional[Dict[str, Any]] = None


class ScreenAnalysisResponse(ChatResponse):
    mode: Literal[
        "explain",
        "translate",
        "guide",
        "annotate",
    ]
    annotations: List[ScreenAnnotation] = Field(default_factory=list, max_length=8)
    action: ScreenActionProposal


class DrawingGenerateRequest(BaseModel):
    user_id: int
    kind: Literal["sketch", "technical", "story"] = "sketch"
    title: str = Field(min_length=1, max_length=80)
    prompt: str = Field(min_length=3, max_length=1600)
    story_relevant: bool = False
    completion_line: str = Field(default="", max_length=240)


class DrawingGenerateResponse(BaseModel):
    image_data_url: str
    mime_type: Literal["image/png"] = "image/png"
    model: str
    sha256: str


class CommandReactionRequest(BaseModel):
    user_id: int
    feature_id: str = Field(min_length=1, max_length=100)
    subject_label: str = Field(default="", max_length=120)
    result_text: str = Field(default="", max_length=240)
    session_id: Optional[int] = None
    story_mode_enabled: bool = True
    companion_name: Optional[str] = Field(
        default=None,
        min_length=2,
        max_length=32,
    )
    story_context: Optional[str] = Field(default=None, max_length=6000)
    activity_context: Optional[str] = Field(default=None, max_length=3000)
    capability_context: Optional[str] = Field(default=None, max_length=5000)


class ProactiveRequest(BaseModel):
    user_id: int
    idle_minutes: int = Field(ge=1, le=1440)
    session_id: Optional[int] = None
    story_mode_enabled: bool = True
    companion_name: Optional[str] = Field(
        default=None,
        min_length=2,
        max_length=32,
    )
    story_context: Optional[str] = Field(default=None, max_length=6000)
    activity_context: Optional[str] = Field(default=None, max_length=3000)
    capability_context: Optional[str] = Field(default=None, max_length=5000)


class CompanionLineResponse(BaseModel):
    text: str
    session_id: int


class HealthResponse(BaseModel):
    status: str
    app: str


class AppLauncherCandidate(BaseModel):
    index: int
    name: str = Field(min_length=1, max_length=500)
    type: str = Field(default="", max_length=50)
    source: str = Field(default="", max_length=100)
    aliases: List[str] = Field(default_factory=list, max_length=10)
    appid: Optional[str] = Field(default=None, max_length=100)
    path_basename: str = Field(default="", max_length=500)


class AppLauncherResolveRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    candidates: List[AppLauncherCandidate] = Field(default_factory=list, max_length=40)


class AppLauncherResolveResponse(BaseModel):
    selected_index: Optional[int] = None
    confidence: float = Field(default=0.0, ge=0, le=1)
    spoken_name: Optional[str] = None
    reason: str = ""


class PersonaNameRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)


class PersonaPresetRequest(BaseModel):
    preset_name: str = Field(min_length=1, max_length=100)


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
