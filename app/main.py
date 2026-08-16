import secrets

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from app.chat_service import ChatService
from app.app_launcher_service import AppLauncherService
from app.command_router_service import CommandRouterService
from app.config import settings
from app.persona_service import PersonaService
from app.memory_service import MemoryService
from app.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    CommandRouteRequest,
    CommandRouteResponse,
    AppLauncherResolveRequest,
    AppLauncherResolveResponse,
    MemoryItemCreateRequest,
    MemoryItemUpdateRequest,
    PersonaPresetRequest,
)


app = FastAPI(title=settings.APP_NAME)


@app.middleware("http")
async def require_internal_gateway_token(request: Request, call_next):
    if request.url.path == "/health":
        return await call_next(request)

    configured = settings.AI_INTERNAL_TOKEN
    if not configured:
        return JSONResponse(
            status_code=503,
            content={"detail": "AI internal gateway token is not configured"},
        )

    supplied = str(request.headers.get("X-Ziren-Internal-Token") or "")
    if not supplied or not secrets.compare_digest(supplied, configured):
        return JSONResponse(
            status_code=401,
            content={"detail": "Invalid internal gateway token"},
        )

    return await call_next(request)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", app=settings.APP_NAME)


@app.post("/persona/{user_id}/name")
def update_name(user_id: int, data: dict):
    try:
        return PersonaService.update_name(user_id, data.get("name", ""))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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


@app.post("/command-route", response_model=CommandRouteResponse)
def command_route(payload: CommandRouteRequest):
    try:
        return CommandRouterService.resolve(payload)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/persona/{user_id}")
def get_persona(user_id: int):
    try:
        return PersonaService.ensure_persona(user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/messages/{user_id}")
def get_messages(user_id: int):
    try:
        return ChatService.get_last_session_messages(user_id)
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


@app.post("/app-launcher/resolve", response_model=AppLauncherResolveResponse)
def app_launcher_resolve(payload: AppLauncherResolveRequest):
    return AppLauncherService.resolve(payload)


@app.post("/memory/{user_id}/clear")
def clear_memory(user_id: int):
    try:
        return MemoryService.clear_all_memory(user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/memory/{user_id}/all")
def delete_all_memory(user_id: int):
    try:
        return MemoryService.clear_all_memory(user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memory-items/{user_id}")
def list_memory_items(user_id: int):
    try:
        return MemoryService.list_memory_items(user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
        raise HTTPException(status_code=500, detail=str(e))


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
        raise HTTPException(status_code=500, detail=str(e))


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
        raise HTTPException(status_code=500, detail=str(e))
