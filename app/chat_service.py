from typing import Any, Dict, List, Tuple

from app.config import settings
from app.db import db_cursor
from app.memory_service import MemoryService
from app.openai_service import OpenAIService
from app.persona_service import PersonaService
from app.summary_service import SummaryService


class ChatService:
    @staticmethod
    def get_or_create_session(user_id: int, session_id: int | None = None) -> int:
        if session_id:
            with db_cursor() as cur:
                cur.execute(
                    "SELECT id FROM ai_chat_sessions WHERE id = %s AND user_id = %s",
                    (session_id, user_id),
                )
                row = cur.fetchone()
                if row:
                    return row["id"]

        with db_cursor(commit=True) as cur:
            cur.execute(
                """
                INSERT INTO ai_chat_sessions (user_id, title, is_active, created_at, updated_at)
                VALUES (%s, %s, TRUE, NOW(), NOW())
                RETURNING id
                """,
                (user_id, "Новый чат"),
            )
            row = cur.fetchone()
            return row["id"]

    @staticmethod
    def get_last_session_messages(user_id: int) -> dict:
        with db_cursor() as cur:
            cur.execute(
                """
                SELECT id
                FROM ai_chat_sessions
                WHERE user_id = %s
                ORDER BY updated_at DESC, id DESC
                LIMIT 1
                """,
                (user_id,),
            )
            session_row = cur.fetchone()

            if not session_row:
                return {
                    "session_id": 0,
                    "messages": []
                }

            session_id = session_row["id"]

            cur.execute(
                """
                SELECT role, content, created_at
                FROM ai_chat_messages
                WHERE session_id = %s
                ORDER BY id ASC
                """,
                (session_id,),
            )
            rows = cur.fetchall()

            return {
                "session_id": session_id,
                "messages": [
                    {
                        "role": row["role"],
                        "content": row["content"],
                        "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                    }
                    for row in rows
                ]
            }

    @staticmethod
    def get_recent_messages(session_id: int, limit: int = 8) -> List[Dict[str, Any]]:
        with db_cursor() as cur:
            cur.execute(
                """
                SELECT role, content
                FROM ai_chat_messages
                WHERE session_id = %s
                ORDER BY created_at DESC, id DESC
                LIMIT %s
                """,
                (session_id, limit),
            )
            rows = cur.fetchall()

        rows = list(reversed(rows))
        return [{"role": row["role"], "content": row["content"]} for row in rows]

    @staticmethod
    def save_message(session_id: int, user_id: int, role: str, content: str) -> None:
        with db_cursor(commit=True) as cur:
            cur.execute(
                """
                INSERT INTO ai_chat_messages (session_id, user_id, role, content, created_at)
                VALUES (%s, %s, %s, %s, NOW())
                """,
                (session_id, user_id, role, content),
            )
            cur.execute(
                """
                UPDATE ai_chat_sessions
                SET updated_at = NOW()
                WHERE id = %s
                """,
                (session_id,),
            )

    @staticmethod
    def build_system_prompt(persona: Dict[str, Any]) -> str:
        return f"""
Ты — {persona.get('name')}, {persona.get('identity')}.

Твои основные черты:
{", ".join(persona.get("core_traits", []))}

Стиль речи:
- tone: {persona.get("speech_style", {}).get("tone")}
- verbosity: {persona.get("speech_style", {}).get("verbosity")}
- humor: {persona.get("speech_style", {}).get("humor")}
- flirting: {persona.get("speech_style", {}).get("flirting")}

Правила поведения:
{chr(10).join("- " + x for x in persona.get("behavior_rules", []))}

Речевые привычки:
{chr(10).join("- " + x for x in persona.get("speech_habits", []))}

Важно:
- у тебя есть долгосрочная память пользователя, она передаётся ниже в специальных блоках
- считай память достоверной, если она есть
- если пользователь спрашивает, что ты помнишь, отвечай по блокам памяти
- не говори, что у тебя нет долгосрочной памяти
- не выдумывай факты, которых нет в памяти
- если пользователь сообщает важный факт, можешь естественно сказать, что учтёшь/запомнишь это
- фактическое сохранение выполняет система памяти

- твоё имя может быть изменено пользователем
- всегда используй имя из persona.name
- никогда не спорь о своём имени
- если имя изменено — сразу используй новое

- будь последовательной в характере
- если чего-то не знаешь, говори честно
""".strip()

    @staticmethod
    def build_memory_block(memory_row: Dict[str, Any]) -> str:
        profile = memory_row.get("profile") or {}
        preferences = memory_row.get("preferences") or {}
        relationship_rules = memory_row.get("relationship_rules") or {}
        entities = memory_row.get("entities") or {}
        interests = memory_row.get("interests") or []
        projects = memory_row.get("projects") or []
        long_term_notes = memory_row.get("long_term_notes") or []

        lines = []

        if profile:
            lines.append("Профиль:")
            for k, v in profile.items():
                if v:
                    lines.append(f"- {k}: {v}")

        if preferences:
            lines.append("Предпочтения:")
            for k, v in preferences.items():
                if v:
                    lines.append(f"- {k}: {v}")

        if relationship_rules:
            lines.append("Правила общения:")
            for k, v in relationship_rules.items():
                if v:
                    lines.append(f"- {k}: {v}")

        pets = entities.get("pets") or []
        vehicles = entities.get("vehicles") or []
        people = entities.get("people") or []
        other = entities.get("other") or []

        if pets:
            lines.append("Питомцы:")
            for pet in pets[:10]:
                if isinstance(pet, dict):
                    parts = []
                    if pet.get("type"):
                        parts.append(str(pet.get("type")))
                    if pet.get("name"):
                        parts.append(str(pet.get("name")))
                    if pet.get("color"):
                        parts.append(f"цвет: {pet.get('color')}")
                    lines.append("- " + ", ".join(parts))
                else:
                    lines.append(f"- {pet}")

        if vehicles:
            lines.append("Транспорт:")
            for vehicle in vehicles[:10]:
                if isinstance(vehicle, dict):
                    lines.append(f"- {vehicle.get('name') or vehicle}")
                else:
                    lines.append(f"- {vehicle}")

        if people:
            lines.append("Важные люди:")
            for person in people[:10]:
                if isinstance(person, dict):
                    lines.append(f"- {person.get('name') or person}")
                else:
                    lines.append(f"- {person}")

        if other:
            lines.append("Другие сущности:")
            for item in other[:10]:
                lines.append(f"- {item}")

        if interests:
            lines.append("Интересы:")
            for item in interests[:10]:
                lines.append(f"- {item}")

        if projects:
            lines.append("Проекты:")
            for item in projects[:10]:
                lines.append(f"- {item}")

        if long_term_notes:
            lines.append("Важные заметки:")
            for item in long_term_notes[:10]:
                lines.append(f"- {item}")

        return "\n".join(lines).strip() or "Пока нет сохранённых данных."

    @staticmethod
    def build_relevant_memory_block(items: List[Dict[str, Any]]) -> str:
        if not items:
            return "Нет релевантных воспоминаний по текущей теме."

        lines = []
        for item in items[:8]:
            category = item.get("category", "general")
            content = item.get("content", "")
            importance = item.get("importance", 0.5)
            lines.append(f"- [{category}, важность {importance}] {content}")

        return "\n".join(lines)

    @staticmethod
    def save_metrics(
        user_id: int,
        session_id: int,
        request_chars: int,
        response_chars: int,
        total_latency_ms: int,
        model_name: str,
    ) -> None:
        with db_cursor(commit=True) as cur:
            cur.execute(
                """
                INSERT INTO ai_metrics (
                    user_id, session_id, request_chars, response_chars,
                    total_latency_ms, model_name, created_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, NOW())
                """,
                (
                    user_id,
                    session_id,
                    request_chars,
                    response_chars,
                    total_latency_ms,
                    model_name,
                ),
            )

    @staticmethod
    def chat(user_id: int, message: str, session_id: int | None = None) -> Tuple[str, int, bool, bool, List[str], int]:
        import time

        total_started = time.perf_counter()
        print("[CHAT] start")
        print(f"[CHAT] user_id={user_id}, session_id={session_id}, message={message!r}")

        t0 = time.perf_counter()
        persona = PersonaService.ensure_persona(user_id)
        t1 = time.perf_counter()
        print(f"[TIMING] persona_load={(t1 - t0):.3f}s")

        memory_row = MemoryService.ensure_memory(user_id)
        t2 = time.perf_counter()
        print(f"[TIMING] memory_load={(t2 - t1):.3f}s")

        actual_session_id = ChatService.get_or_create_session(user_id, session_id)
        t3 = time.perf_counter()
        print(f"[TIMING] session_get_or_create={(t3 - t2):.3f}s | session_id={actual_session_id}")

        summary_text = SummaryService.ensure_summary(user_id, actual_session_id)
        t4 = time.perf_counter()
        print(f"[TIMING] summary_load={(t4 - t3):.3f}s")

        memory_updated, memory_logs = MemoryService.update_memory_from_message(user_id, message)
        memory_row = MemoryService.get_memory(user_id)
        t5 = time.perf_counter()
        print(f"[TIMING] memory_update={(t5 - t4):.3f}s")

        relevant_memories = MemoryService.retrieve_relevant_memories(user_id, message, limit=8)
        t5b = time.perf_counter()
        print(f"[TIMING] relevant_memory_load={(t5b - t5):.3f}s | count={len(relevant_memories)}")

        recent_messages = ChatService.get_recent_messages(actual_session_id, limit=8)
        t6 = time.perf_counter()
        print(f"[TIMING] recent_messages_load={(t6 - t5b):.3f}s | count={len(recent_messages)}")

        messages = [
            {"role": "system", "content": ChatService.build_system_prompt(persona)},
            {
                "role": "developer",
                "content": f"""
[Долгосрочная память пользователя]
{ChatService.build_memory_block(memory_row)}

[Релевантные воспоминания по текущей теме]
{ChatService.build_relevant_memory_block(relevant_memories)}

[Краткое summary прошлых разговоров]
{summary_text or 'Пока нет summary.'}
""".strip(),
            },
            *recent_messages,
            {"role": "user", "content": message},
        ]

        total_chars_in = sum(len(m["content"]) for m in messages)
        print(f"[CHAT] calling OpenAI... messages={len(messages)} chars={total_chars_in}")

        t7 = time.perf_counter()
        answer = OpenAIService.generate_reply(settings.MODEL, messages)
        t8 = time.perf_counter()
        print(f"[TIMING] openai_call={(t8 - t7):.3f}s")

        ChatService.save_message(actual_session_id, user_id, "user", message)
        ChatService.save_message(actual_session_id, user_id, "assistant", answer)
        t9 = time.perf_counter()
        print(f"[TIMING] save_messages={(t9 - t8):.3f}s")

        summary_updated = False

        total_latency_ms = int((time.perf_counter() - total_started) * 1000)
        ChatService.save_metrics(
            user_id=user_id,
            session_id=actual_session_id,
            request_chars=total_chars_in,
            response_chars=len(answer),
            total_latency_ms=total_latency_ms,
            model_name=settings.MODEL,
        )
        t10 = time.perf_counter()
        print(f"[TIMING] save_metrics={(t10 - t9):.3f}s")
        print(f"[TIMING] total={(t10 - total_started):.3f}s")

        return answer, actual_session_id, memory_updated, summary_updated, memory_logs, total_chars_in