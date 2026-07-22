from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse

from app.chat_service import ChatService
from app.app_launcher_service import AppLauncherService
from app.config import settings
from app.persona_service import PersonaService
from app.memory_service import MemoryService
from app.internal_auth import INTERNAL_TOKEN_HEADER, get_internal_auth_error
from app.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    AppLauncherResolveRequest,
    AppLauncherResolveResponse,
    MemoryItemCreateRequest,
    MemoryItemUpdateRequest,
    PersonaNameRequest,
    PersonaPresetRequest,
)


app = FastAPI(title=settings.APP_NAME)


def internal_server_error(context: str, error: Exception) -> HTTPException:
    print(f"[{context}] {type(error).__name__}: {error}")
    return HTTPException(status_code=500, detail="Internal server error")


@app.middleware("http")
async def require_internal_auth(request: Request, call_next):
    if request.url.path == "/health":
        return await call_next(request)

    provided_token = request.headers.get(INTERNAL_TOKEN_HEADER, "")
    auth_error = get_internal_auth_error(
        provided_token,
        settings.AI_INTERNAL_TOKEN,
    )

    if auth_error is not None:
        status_code, detail = auth_error
        return JSONResponse(
            status_code=status_code,
            content={"detail": detail},
        )

    return await call_next(request)


@app.get("/health", response_model=HealthResponse)
def health(response: Response):
    required_values = (
        settings.AI_INTERNAL_TOKEN,
        settings.OPENAI_API_KEY,
        settings.DATABASE_URL,
    )

    if not all(required_values):
        response.status_code = 503
        return HealthResponse(status="misconfigured", app=settings.APP_NAME)

    return HealthResponse(status="ok", app=settings.APP_NAME)


@app.post("/persona/{user_id}/name")
def update_name(user_id: int, payload: PersonaNameRequest):
    try:
        return PersonaService.update_name(user_id, payload.name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise internal_server_error("persona.name", e) from e

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
        raise internal_server_error("chat", e) from e


@app.get("/persona/{user_id}")
def get_persona(user_id: int):
    try:
        return PersonaService.ensure_persona(user_id)
    except Exception as e:
        raise internal_server_error("persona.get", e) from e


@app.get("/messages/{user_id}")
def get_messages(user_id: int):
    try:
        return ChatService.get_last_session_messages(user_id)
    except Exception as e:
        raise internal_server_error("messages.get", e) from e


@app.post("/persona/{user_id}/preset")
def apply_persona_preset(user_id: int, payload: PersonaPresetRequest):
    try:
        return PersonaService.apply_preset(user_id, payload.preset_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise internal_server_error("persona.preset", e) from e


@app.get("/memory/{user_id}")
def get_memory(user_id: int):
    try:
        return MemoryService.ensure_memory(user_id)
    except Exception as e:
        raise internal_server_error("memory.get", e) from e


@app.post("/app-launcher/resolve", response_model=AppLauncherResolveResponse)
def app_launcher_resolve(payload: AppLauncherResolveRequest):
    return AppLauncherService.resolve(payload)


@app.post("/memory/{user_id}/clear")
def clear_memory(user_id: int):
    try:
        return MemoryService.clear_all_memory(user_id)
    except Exception as e:
        raise internal_server_error("memory.clear", e) from e


@app.delete("/memory/{user_id}/all")
def delete_all_memory(user_id: int):
    try:
        return MemoryService.clear_all_memory(user_id)
    except Exception as e:
        raise internal_server_error("memory.delete_all", e) from e


@app.get("/memory-items/{user_id}")
def list_memory_items(user_id: int):
    try:
        return MemoryService.list_memory_items(user_id)
    except Exception as e:
        raise internal_server_error("memory_items.list", e) from e


@app.post("/memory-items/{user_id}")
def create_memory_item(user_id: int, payload: MemoryItemCreateRequest):
    try:
        return MemoryService.create_memory_item(
            user_id,
            payload.model_dump(),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise internal_server_error("memory_items.create", e) from e


@app.patch("/memory-items/{user_id}/{item_id}")
def update_memory_item(user_id: int, item_id: int, payload: MemoryItemUpdateRequest):
    try:
        item = MemoryService.update_memory_item(
            user_id,
            item_id,
            payload.model_dump(exclude_unset=True),
        )
        if not item:
            raise HTTPException(status_code=404, detail="memory item not found")
        return item
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise internal_server_error("memory_items.update", e) from e


@app.delete("/memory-items/{user_id}/{item_id}")
def delete_memory_item(user_id: int, item_id: int):
    try:
        deleted = MemoryService.delete_memory_item(user_id, item_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="memory item not found")
        return {"deleted": True, "id": item_id}
    except HTTPException:
        raise
    except Exception as e:
        raise internal_server_error("memory_items.delete", e) from e
