import logging
import secrets

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from app.chat_service import ChatService
from app.app_launcher_service import AppLauncherService
from app.command_router_service import CommandRouterService
from app.config import settings
from app.db import db_cursor
from app.persona_service import PersonaService
from app.memory_service import MemoryService
from app.subscription_service import SubscriptionAccessError, SubscriptionService
from app.usage_context import ai_usage_context
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


logger = logging.getLogger("ziren-ai-service")
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


def _subscription_error(error: SubscriptionAccessError) -> HTTPException:
    return HTTPException(
        status_code=error.status_code,
        detail={"code": error.code, "message": str(error)},
    )


def _internal_error(context: str, error: Exception) -> HTTPException:
    logger.exception("%s failed", context, exc_info=error)
    return HTTPException(status_code=500, detail="Internal service error")


@app.get("/health", response_model=HealthResponse)
def health():
    """Public process liveness only. It intentionally exposes no secrets."""
    return HealthResponse(status="ok", app=settings.APP_NAME)


@app.get("/ready")
def ready():
    """Authenticated readiness: configuration exists and PostgreSQL responds."""
    if not settings.OPENAI_API_KEY or not settings.DATABASE_URL:
        raise HTTPException(status_code=503, detail="Service configuration incomplete")

    try:
        with db_cursor() as cursor:
            cursor.execute("SELECT 1 AS ready")
            row = cursor.fetchone()
        if not row:
            raise RuntimeError("Database readiness query returned no row")
    except Exception as error:
        logger.exception("AI readiness database check failed", exc_info=error)
        raise HTTPException(status_code=503, detail="Database unavailable") from None

    return {"status": "ready", "app": settings.APP_NAME}


@app.get("/subscription/{user_id}")
def subscription_status(user_id: int):
    try:
        return {"ok": True, "subscription": SubscriptionService.get_status(user_id)}
    except Exception as error:
        raise _internal_error("subscription status", error) from None


@app.post("/persona/{user_id}/name")
def update_name(user_id: int, data: dict):
    try:
        return PersonaService.update_name(user_id, data.get("name", ""))
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception as error:
        raise _internal_error("persona name update", error) from None


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest):
    try:
        SubscriptionService.require_ai_access(payload.user_id)
        with ai_usage_context(payload.user_id, "chat"):
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
    except SubscriptionAccessError as error:
        raise _subscription_error(error)
    except Exception as error:
        raise _internal_error("chat", error) from None


@app.post("/command-route", response_model=CommandRouteResponse)
def command_route(payload: CommandRouteRequest):
    try:
        SubscriptionService.require_ai_access(payload.user_id)
        with ai_usage_context(payload.user_id, "command_route"):
            return CommandRouterService.resolve(payload)
    except SubscriptionAccessError as error:
        raise _subscription_error(error)
    except Exception as error:
        raise _internal_error("command route", error) from None


@app.get("/persona/{user_id}")
def get_persona(user_id: int):
    try:
        return PersonaService.ensure_persona(user_id)
    except Exception as error:
        raise _internal_error("persona read", error) from None


@app.get("/messages/{user_id}")
def get_messages(user_id: int):
    try:
        return ChatService.get_last_session_messages(user_id)
    except Exception as error:
        raise _internal_error("message history", error) from None


@app.post("/persona/{user_id}/preset")
def apply_persona_preset(user_id: int, payload: PersonaPresetRequest):
    try:
        return PersonaService.apply_preset(user_id, payload.preset_name)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception as error:
        raise _internal_error("persona preset", error) from None


@app.get("/memory/{user_id}")
def get_memory(user_id: int):
    try:
        return MemoryService.ensure_memory(user_id)
    except Exception as error:
        raise _internal_error("memory read", error) from None


@app.post("/app-launcher/resolve", response_model=AppLauncherResolveResponse)
def app_launcher_resolve(payload: AppLauncherResolveRequest):
    # This legacy resolver is kept for compatibility. New command routing uses
    # /command-route and is metered/gated there. The resolver itself never
    # executes a program; Core still validates and launches local candidates.
    return AppLauncherService.resolve(payload)


@app.post("/memory/{user_id}/clear")
def clear_memory(user_id: int):
    try:
        return MemoryService.clear_all_memory(user_id)
    except Exception as error:
        raise _internal_error("memory clear", error) from None


@app.delete("/memory/{user_id}/all")
def delete_all_memory(user_id: int):
    try:
        return MemoryService.clear_all_memory(user_id)
    except Exception as error:
        raise _internal_error("memory delete all", error) from None


@app.get("/memory-items/{user_id}")
def list_memory_items(user_id: int):
    try:
        return MemoryService.list_memory_items(user_id)
    except Exception as error:
        raise _internal_error("memory items list", error) from None


@app.post("/memory-items/{user_id}")
def create_memory_item(user_id: int, payload: MemoryItemCreateRequest):
    try:
        return MemoryService.create_memory_item(
            user_id,
            payload.model_dump(),
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception as error:
        raise _internal_error("memory item create", error) from None


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
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception as error:
        raise _internal_error("memory item update", error) from None


@app.delete("/memory-items/{user_id}/{item_id}")
def delete_memory_item(user_id: int, item_id: int):
    try:
        deleted = MemoryService.delete_memory_item(user_id, item_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="memory item not found")
        return {"deleted": True, "id": item_id}
    except HTTPException:
        raise
    except Exception as error:
        raise _internal_error("memory item delete", error) from None
