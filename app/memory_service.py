import json
import re
from typing import Any, Dict, List, Tuple

from app.db import db_cursor


DEFAULT_MEMORY = {
    "profile": {},
    "preferences": {},
    "relationship_rules": {},
    "entities": {
        "pets": [],
        "vehicles": [],
        "people": [],
        "other": [],
    },
    "interests": [],
    "projects": [],
    "long_term_notes": [],
}


def normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text).strip())


def capitalize_first(value: str) -> str:
    value = normalize_text(value)
    return value[:1].upper() + value[1:] if value else value


class MemoryService:
    @staticmethod
    def ensure_memory(user_id: int) -> Dict[str, Any]:
        with db_cursor(commit=True) as cur:
            cur.execute(
                """
                INSERT INTO ai_user_memory (
                    user_id, profile, preferences, relationship_rules,
                    entities, interests, projects, long_term_notes
                )
                VALUES (%s, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb)
                ON CONFLICT (user_id) DO NOTHING
                """,
                (
                    user_id,
                    json.dumps(DEFAULT_MEMORY["profile"], ensure_ascii=False),
                    json.dumps(DEFAULT_MEMORY["preferences"], ensure_ascii=False),
                    json.dumps(DEFAULT_MEMORY["relationship_rules"], ensure_ascii=False),
                    json.dumps(DEFAULT_MEMORY["entities"], ensure_ascii=False),
                    json.dumps(DEFAULT_MEMORY["interests"], ensure_ascii=False),
                    json.dumps(DEFAULT_MEMORY["projects"], ensure_ascii=False),
                    json.dumps(DEFAULT_MEMORY["long_term_notes"], ensure_ascii=False),
                ),
            )
        return MemoryService.get_memory(user_id)

    @staticmethod
    def get_memory(user_id: int) -> Dict[str, Any]:
        with db_cursor() as cur:
            cur.execute("SELECT * FROM ai_user_memory WHERE user_id = %s", (user_id,))
            row = cur.fetchone()
            if not row:
                return MemoryService.ensure_memory(user_id)
            return dict(row)

    @staticmethod
    def update_memory_from_message(user_id: int, message: str) -> Tuple[bool, List[str]]:
        memory_row = MemoryService.get_memory(user_id)

        profile = memory_row["profile"] or {}
        interests = memory_row["interests"] or []
        projects = memory_row["projects"] or []

        changed = False
        logs: List[str] = []
        low = normalize_text(message).lower()

        m = re.search(r"(?:меня зовут|мо[её] имя)\s+([A-Za-zА-Яа-яЁё\-]{2,40})", low, re.IGNORECASE)
        if m:
            name = capitalize_first(m.group(1))
            if profile.get("name") != name:
                profile["name"] = name
                changed = True
                logs.append(f"profile.name = {name}")

        m = re.search(r"(?:я живу в|мой город)\s+([A-Za-zА-Яа-яЁё\-\s]{2,60})", low, re.IGNORECASE)
        if m:
            city = capitalize_first(m.group(1).strip(" .,!?:;"))
            if profile.get("city") != city:
                profile["city"] = city
                changed = True
                logs.append(f"profile.city = {city}")

        m = re.search(r"(?:я говорю на|я говорю по|мой язык)\s+([A-Za-zА-Яа-яЁё\-]{2,40})", low, re.IGNORECASE)
        if m:
            language = normalize_text(m.group(1))
            if profile.get("language") != language:
                profile["language"] = language
                changed = True
                logs.append(f"profile.language = {language}")

        for pattern in [
            r"(?:я люблю)\s+(.+)$",
            r"(?:мне нравится)\s+(.+)$",
            r"(?:я увлекаюсь)\s+(.+)$",
        ]:
            m = re.search(pattern, low, re.IGNORECASE)
            if m:
                interest = capitalize_first(m.group(1).strip(" .,!?:;"))
                if interest and interest not in interests:
                    interests.append(interest)
                    changed = True
                    logs.append(f"interests += {interest}")
                break

        for pattern in [
            r"(?:я делаю)\s+(.+)$",
            r"(?:я создаю)\s+(.+)$",
            r"(?:я пишу)\s+(.+)$",
            r"(?:я работаю над)\s+(.+)$",
        ]:
            m = re.search(pattern, low, re.IGNORECASE)
            if m:
                project = capitalize_first(m.group(1).strip(" .,!?:;"))
                if project and project not in projects:
                    projects.append(project)
                    changed = True
                    logs.append(f"projects += {project}")
                break

        if changed:
            with db_cursor(commit=True) as cur:
                cur.execute(
                    """
                    UPDATE ai_user_memory
                    SET profile = %s::jsonb,
                        interests = %s::jsonb,
                        projects = %s::jsonb,
                        updated_at = NOW()
                    WHERE user_id = %s
                    """,
                    (
                        json.dumps(profile, ensure_ascii=False),
                        json.dumps(interests, ensure_ascii=False),
                        json.dumps(projects, ensure_ascii=False),
                        user_id,
                    ),
                )

        return changed, logs