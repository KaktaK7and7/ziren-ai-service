from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar


_current_user_id: ContextVar[int | None] = ContextVar("ai_usage_user_id", default=None)
_current_operation: ContextVar[str] = ContextVar("ai_usage_operation", default="ai")


@contextmanager
def ai_usage_context(user_id: int, operation: str):
    user_token = _current_user_id.set(int(user_id))
    operation_token = _current_operation.set(str(operation or "ai")[:64])
    try:
        yield
    finally:
        _current_operation.reset(operation_token)
        _current_user_id.reset(user_token)


def get_ai_usage_context() -> tuple[int | None, str]:
    return _current_user_id.get(), _current_operation.get()
