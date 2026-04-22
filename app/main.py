from fastapi import FastAPI, HTTPException

from app.chat_service import ChatService
from app.config import settings
from app.persona_service import PersonaService
from app.memory_service import MemoryService
from app.schemas import ChatRequest, ChatResponse, HealthResponse, PersonaPresetRequest


app = FastAPI(title=settings.APP_NAME)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", app=settings.APP_NAME)


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest):
    try:
        answer, session_id, memory_updated, summary_updated, memory_logs, _ = ChatService.chat(
            user_id=payload.user_id,
            message=payload.message,
            session_id=payload.session_id,
        )
        return ChatResponse(
            answer=answer,
            session_id=session_id,
            memory_updated=memory_updated,
            summary_updated=summary_updated,
            memory_logs=memory_logs,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/persona/{user_id}")
def get_persona(user_id: int):
    try:
        return PersonaService.ensure_persona(user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/persona/{user_id}/preset")
def apply_persona_preset(user_id: int, payload: PersonaPresetRequest):
    try:
        return PersonaService.apply_preset(user_id, payload.preset_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memory/{user_id}")
def get_memory(user_id: int):
    try:
        return MemoryService.ensure_memory(user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))