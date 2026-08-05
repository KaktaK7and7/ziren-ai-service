from __future__ import annotations

import re
from typing import Any

from app.chat_service import ChatService
from app.memory_service import MemoryService
from app.persona_service import PersonaService
from app.schemas import ScreenAnalysisPlan


TRANSIENT_SCREEN_RETRY_MARKER = (
    "Служебное уточнение: это увеличенный фрагмент предыдущего снимка"
)


def is_transient_screen_retry(message: object) -> bool:
    """Recognize the internal enlarged-crop retry emitted by the desktop core."""
    normalized = " ".join(str(message or "").split())
    return TRANSIENT_SCREEN_RETRY_MARKER in normalized


def analyze_screen_without_history(
    user_id: int,
    message: str,
    image_data_url: str,
    session_id: int | None = None,
    preceding_assistant_lines: list[str] | None = None,
    story_mode_enabled: bool = True,
    companion_name: str | None = None,
    story_context: str | None = None,
    activity_context: str | None = None,
    capability_context: str | None = None,
) -> tuple[ScreenAnalysisPlan, int]:
    """Run a locator retry against chat context without persisting service text."""
    persona = PersonaService.ensure_persona(user_id)
    memory_row = MemoryService.ensure_memory(user_id)
    actual_session_id = ChatService.get_or_create_session(user_id, session_id)
    recent_messages = ChatService.get_recent_messages(actual_session_id, limit=6)
    delivered_lines = [
        re.sub(r"[\x00-\x1f\x7f]", " ", str(line)).strip()[:600]
        for line in (preceding_assistant_lines or [])[:2]
        if str(line).strip()
    ]
    messages = ChatService.build_screen_analysis_messages(
        persona=persona,
        memory_row=memory_row,
        recent_messages=[
            *recent_messages,
            *(
                {"role": "assistant", "content": line}
                for line in delivered_lines
            ),
        ],
        message=message,
        image_data_url=image_data_url,
        story_mode_enabled=story_mode_enabled,
        companion_name=companion_name,
        story_context=story_context,
        activity_context=activity_context,
        capability_context=capability_context,
    )
    plan = ChatService.generate_screen_analysis_plan(
        messages,
        story_mode_enabled=story_mode_enabled,
        message=message,
    )
    return plan, actual_session_id
