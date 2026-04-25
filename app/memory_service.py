import json
import re
from typing import Any, Dict, List, Tuple

from app.db import db_cursor
from app.openai_service import OpenAIService
from app.config import settings


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


def normalize_memory_content_for_dedupe(content: Any) -> str:
    text = normalize_text(content).lower()

    prefix_patterns = [
        r"^пользователь\s+интересуется\s+",
        r"^пользователь\s+любит\s+",
        r"^пользователь\s+",
        r"^меня\s+зовут\s+",
        r"^моё\s+имя\s+",
        r"^мое\s+имя\s+",
    ]
    for pattern in prefix_patterns:
        updated = re.sub(pattern, "", text, count=1)
        if updated != text:
            text = updated
            break

    text = re.sub(r"[^\w\s]+", " ", text, flags=re.UNICODE)
    text = text.replace("_", " ")
    return normalize_text(text)


def capitalize_first(value: str) -> str:
    value = normalize_text(value)
    return value[:1].upper() + value[1:] if value else value


def list_has_text(items: List[Any], text: str) -> bool:
    target = normalize_memory_content_for_dedupe(text)
    for item in items:
        if normalize_memory_content_for_dedupe(item) == target:
            return True
    return False


def clone_default_memory() -> Dict[str, Any]:
    return json.loads(json.dumps(DEFAULT_MEMORY, ensure_ascii=False))


def classify_memory_category(category: str, content: str) -> str:
    category = normalize_text(category or "general") or "general"
    text = normalize_text(content).lower()

    if category != "general":
        return category

    if "меня зовут" in text or "имя:" in text:
        return "profile.name"
    if "я говорю" in text or "язык" in text:
        return "profile.language"
    if "я живу" in text or "город" in text:
        return "profile.city"
    if any(word in text for word in ["машин", "автомоб", "велосипед", "мотоцикл", "самокат"]):
        return "entities.vehicles"
    if any(word in text for word in ["кот", "кошк", "собак", "питомец", "питомц"]):
        return "entities.pets"

    return category


def extract_profile_value(key: str, content: str) -> str:
    value = normalize_text(content)

    patterns = {
        "name": [
            r"(?:меня зовут|мо[её]\s+имя|имя:)\s*([A-Za-zА-Яа-яЁё\-]{2,80})",
            r"(?:пользователя зовут|пользователь\s+)\s*([A-Za-zА-Яа-яЁё\-]{2,80})",
        ],
        "city": [
            r"(?:я живу в|я переехал в|я переехала в|мой город|город:|жив[её]т в)\s*([A-Za-zА-Яа-яЁё\-\s]{2,80})",
        ],
        "language": [
            r"(?:я говорю на|я говорю по|мой язык|язык:|говорит на|говорит по)\s*([A-Za-zА-Яа-яЁё\-]{2,80})",
        ],
    }

    for pattern in patterns.get(key, []):
        match = re.search(pattern, value, re.IGNORECASE)
        if match:
            value = match.group(1)
            break
    else:
        value = re.sub(r"^Пользователь\s+", "", value, flags=re.IGNORECASE)
        value = re.sub(r"^(живёт в|живет в|говорит на|говорит по|работает в|работает как|зовут)\s+", "", value, flags=re.IGNORECASE)

    return capitalize_first(value.strip(" .,!?:;"))


def entity_exists(items: List[Dict[str, Any]], name: str) -> bool:
    target = normalize_memory_content_for_dedupe(name)
    for item in items:
        if normalize_memory_content_for_dedupe(item.get("name", "")) == target:
            return True
    return False


class MemoryService:
    MEMORY_ITEM_FIELDS = {
        "type",
        "category",
        "content",
        "source_message",
        "importance",
        "confidence",
        "sensitivity",
        "status",
    }

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
    def list_memory_items(user_id: int) -> List[Dict[str, Any]]:
        with db_cursor() as cur:
            cur.execute(
                """
                SELECT
                    id, user_id, type, category, content, source_message,
                    importance, confidence, sensitivity, status,
                    created_at, updated_at, last_accessed_at, access_count
                FROM ai_memory_items
                WHERE user_id = %s
                  AND status = 'active'
                ORDER BY importance DESC, updated_at DESC, id DESC
                """,
                (user_id,),
            )
            rows = cur.fetchall()

        return [dict(row) for row in rows]

    @staticmethod
    def create_memory_item(user_id: int, data: Dict[str, Any]) -> Dict[str, Any]:
        content = normalize_text(data.get("content", ""))
        if not content:
            raise ValueError("content is required")

        item_type = normalize_text(data.get("type", "semantic")) or "semantic"
        category = classify_memory_category(data.get("category", "general"), content)
        source_message = data.get("source_message")
        source_message = normalize_text(source_message) if source_message is not None else None
        sensitivity = normalize_text(data.get("sensitivity", "normal")) or "normal"
        status = normalize_text(data.get("status", "active")) or "active"

        with db_cursor(commit=True) as cur:
            if status == "active":
                cur.execute(
                    """
                    SELECT
                        id, user_id, type, category, content, source_message,
                        importance, confidence, sensitivity, status,
                        created_at, updated_at, last_accessed_at, access_count
                    FROM ai_memory_items
                    WHERE user_id = %s
                      AND status = 'active'
                    ORDER BY importance DESC, updated_at DESC, id DESC
                    """,
                    (user_id,),
                )
                target_content = content.lower()
                target_normalized = normalize_memory_content_for_dedupe(content)
                for existing in cur.fetchall():
                    existing_content = normalize_text(existing.get("content", ""))
                    existing_normalized = normalize_memory_content_for_dedupe(existing_content)
                    if existing_content.lower() == target_content or (
                        target_normalized and existing_normalized == target_normalized
                    ):
                        return dict(existing)

            cur.execute(
                """
                INSERT INTO ai_memory_items (
                    user_id, type, category, content, source_message,
                    importance, confidence, sensitivity, status,
                    created_at, updated_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
                RETURNING
                    id, user_id, type, category, content, source_message,
                    importance, confidence, sensitivity, status,
                    created_at, updated_at, last_accessed_at, access_count
                """,
                (
                    user_id,
                    item_type,
                    category,
                    content,
                    source_message,
                    float(data.get("importance", 0.5)),
                    float(data.get("confidence", 0.8)),
                    sensitivity,
                    status,
                ),
            )
            row = cur.fetchone()

        item = dict(row)
        if item.get("status") == "active":
            MemoryService.rebuild_structured_memory_from_items(user_id)

        return item

    @staticmethod
    def update_memory_item(user_id: int, item_id: int, data: Dict[str, Any]) -> Dict[str, Any] | None:
        updates = {
            key: value
            for key, value in data.items()
            if key in MemoryService.MEMORY_ITEM_FIELDS and (value is not None or key == "source_message")
        }

        if not updates:
            raise ValueError("no fields to update")

        for text_field in ["type", "category", "content", "source_message", "sensitivity", "status"]:
            if text_field in updates and updates[text_field] is not None:
                updates[text_field] = normalize_text(updates[text_field])

        if "content" in updates and not updates["content"]:
            raise ValueError("content cannot be empty")

        assignments = []
        values = []
        for field in MemoryService.MEMORY_ITEM_FIELDS:
            if field in updates:
                assignments.append(f"{field} = %s")
                values.append(updates[field])

        values.extend([user_id, item_id])

        with db_cursor(commit=True) as cur:
            cur.execute(
                f"""
                UPDATE ai_memory_items
                SET {", ".join(assignments)},
                    updated_at = NOW()
                WHERE user_id = %s
                  AND id = %s
                RETURNING
                    id, user_id, type, category, content, source_message,
                    importance, confidence, sensitivity, status,
                    created_at, updated_at, last_accessed_at, access_count
                """,
                tuple(values),
            )
            row = cur.fetchone()

        if not row:
            return None

        item = dict(row)
        MemoryService.rebuild_structured_memory_from_items(user_id)
        return item

    @staticmethod
    def delete_memory_item(user_id: int, item_id: int) -> bool:
        with db_cursor(commit=True) as cur:
            cur.execute(
                """
                UPDATE ai_memory_items
                SET status = 'deleted',
                    updated_at = NOW()
                WHERE user_id = %s
                  AND id = %s
                  AND status <> 'deleted'
                RETURNING id
                """,
                (user_id, item_id),
            )
            row = cur.fetchone()

        deleted = bool(row)
        if deleted:
            MemoryService.rebuild_structured_memory_from_items(user_id)

        return deleted

    @staticmethod
    def clear_all_memory(user_id: int) -> Dict[str, bool]:
        empty_memory = clone_default_memory()
        MemoryService.ensure_memory(user_id)

        with db_cursor(commit=True) as cur:
            cur.execute(
                """
                UPDATE ai_memory_items
                SET status = 'deleted',
                    updated_at = NOW()
                WHERE user_id = %s
                  AND status <> 'deleted'
                """,
                (user_id,),
            )
            cur.execute(
                """
                UPDATE ai_user_memory
                SET profile = %s::jsonb,
                    preferences = %s::jsonb,
                    relationship_rules = %s::jsonb,
                    entities = %s::jsonb,
                    interests = %s::jsonb,
                    projects = %s::jsonb,
                    long_term_notes = %s::jsonb,
                    updated_at = NOW()
                WHERE user_id = %s
                """,
                (
                    json.dumps(empty_memory["profile"], ensure_ascii=False),
                    json.dumps(empty_memory["preferences"], ensure_ascii=False),
                    json.dumps(empty_memory["relationship_rules"], ensure_ascii=False),
                    json.dumps(empty_memory["entities"], ensure_ascii=False),
                    json.dumps(empty_memory["interests"], ensure_ascii=False),
                    json.dumps(empty_memory["projects"], ensure_ascii=False),
                    json.dumps(empty_memory["long_term_notes"], ensure_ascii=False),
                    user_id,
                ),
            )

        return {"ok": True}

    @staticmethod
    def should_run_ai_memory_analysis(message: str, regex_changed: bool) -> bool:
        text = normalize_text(message).lower()

        triggers = [
            "запомни",
            "сохрани",
            "я стал",
            "я стала",
            "я был",
            "я была",
            "я живу",
            "я переехал",
            "меня зовут",
            "у меня есть",
            "мой кот",
            "моя кошка",
            "мой питомец",
            "моя машина",
            "мой автомобиль",
            "я работаю",
            "я уволился",
            "я заболел",
            "мне тяжело",
            "я люблю",
            "я не люблю",
            "мне нравится",
            "я хочу чтобы ты",
            "общайся",
            "не делай",
            "моя девушка",
            "мой парень",
            "моя мама",
            "мой папа",
            "мой брат",
            "моя сестра",
            "мой друг",
            "моя семья",
            "я начал",
            "мои планы",
            "я научился",
            "мой проект",
            "я занимаюсь",
        ]

        if any(t in text for t in triggers):
            return True

        if not regex_changed and len(text) > 80 and any(x in text for x in ["я ", "мой ", "моя ", "мне ", "у меня"]):
            return True

        return False

    @staticmethod
    def extract_memory_with_ai(user_id: int, message: str) -> dict:
        prompt = """
Ты — модуль долгосрочной памяти персонального ИИ-ассистента.
Отвечай СТРОГО валидным JSON.
Не добавляй текст вне JSON.
Не используй markdown.

Твоя задача — определить, есть ли в сообщении пользователя важная информация,
которая может пригодиться в будущих разговорах и поможет ассистенту лучше понимать человека.

Сохраняй долгосрочно полезные факты:
- имя, город, язык
- семья, друзья, партнёры, важные люди
- питомцы
- работа, увольнение, учёба, профессия
- проекты
- интересы, хобби, любимые темы
- здоровье и важное состояние
- конфликты, отношения, важные события
- правила общения с ассистентом
- предпочтения пользователя
- важные жизненные изменения

Не сохраняй:
- случайные фразы
- одноразовые просьбы
- шутки без явного намерения сохранить
- временное настроение без долгосрочного значения
- пароли, ключи, секреты
- опасные инструкции

Если пользователь явно говорит "запомни" или "сохрани", это сильный сигнал важности.

Верни только JSON:
{
  "should_save": true,
  "items": [
    {
      "type": "semantic",
      "category": "profile.city",
      "content": "Пользователь живёт во Владивостоке",
      "importance": 0.8,
      "confidence": 0.95,
      "sensitivity": "normal"
    }
  ]
}

Допустимые type:
- semantic
- episodic
- preference
- relationship

Примеры category:
- profile.name
- profile.city
- profile.language
- entities.pets
- entities.vehicles
- entities.people
- interests
- projects
- relationship_rules
- preferences
- long_term_notes
- life_events
- health
- work
""".strip()

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": message},
        ]

        result = OpenAIService.generate_json(
            model=settings.MEMORY_MODEL,
            messages=messages,
        )

        if not result:
            return {"should_save": False, "items": []}

        result.setdefault("should_save", False)
        result.setdefault("items", [])
        return result

    @staticmethod
    def save_memory_items(user_id: int, items: List[Dict[str, Any]], source_message: str) -> Tuple[bool, List[str]]:
        if not items:
            return False, []

        logs: List[str] = []

        with db_cursor(commit=True) as cur:
            cur.execute(
                """
                SELECT content
                FROM ai_memory_items
                WHERE user_id = %s
                  AND status = 'active'
                """,
                (user_id,),
            )
            active_contents = [normalize_text(row.get("content", "")) for row in cur.fetchall()]

            for item in items:
                content = normalize_text(item.get("content", ""))
                if not content:
                    continue

                category = classify_memory_category(item.get("category", "general"), content)

                target_content = content.lower()
                target_normalized = normalize_memory_content_for_dedupe(content)
                duplicate = any(
                    existing.lower() == target_content
                    or (target_normalized and normalize_memory_content_for_dedupe(existing) == target_normalized)
                    for existing in active_contents
                )
                if duplicate:
                    continue

                cur.execute(
                    """
                    INSERT INTO ai_memory_items (
                        user_id, type, category, content, source_message,
                        importance, confidence, sensitivity, status,
                        created_at, updated_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'active', NOW(), NOW())
                    """,
                    (
                        user_id,
                        item.get("type", "semantic"),
                        category,
                        content,
                        source_message,
                        float(item.get("importance", 0.5)),
                        float(item.get("confidence", 0.8)),
                        item.get("sensitivity", "normal"),
                    ),
                )

                logs.append(f"memory_item += {content}")
                active_contents.append(content)

        return bool(logs), logs

    @staticmethod
    def apply_ai_items_to_structured_memory(
        memory_row: Dict[str, Any],
        items: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], bool, List[str]]:
        profile = memory_row["profile"] or {}
        preferences = memory_row["preferences"] or {}
        relationship_rules = memory_row["relationship_rules"] or {}
        entities = memory_row["entities"] or DEFAULT_MEMORY["entities"].copy()
        interests = memory_row["interests"] or []
        projects = memory_row["projects"] or []
        long_term_notes = memory_row["long_term_notes"] or []

        entities.setdefault("pets", [])
        entities.setdefault("vehicles", [])
        entities.setdefault("people", [])
        entities.setdefault("other", [])

        changed = False
        logs: List[str] = []

        for item in items:
            category = classify_memory_category(item.get("category", ""), item.get("content", ""))
            content = normalize_text(item.get("content", ""))

            if not content:
                continue

            if category == "interests":
                clean = content.replace("Пользователь любит ", "").replace("Пользователь интересуется ", "")
                clean = capitalize_first(clean)
                if clean and not list_has_text(interests, clean):
                    interests.append(clean)
                    changed = True
                    logs.append(f"interests += {clean}")

            elif category == "projects":
                clean = content.replace("Пользователь работает над ", "").replace("Пользователь делает ", "")
                clean = capitalize_first(clean)
                if clean and not list_has_text(projects, clean):
                    projects.append(clean)
                    changed = True
                    logs.append(f"projects += {clean}")

            elif category == "relationship_rules":
                if "communication_style" not in relationship_rules:
                    relationship_rules["communication_style"] = content
                    changed = True
                    logs.append("relationship_rules.communication_style updated")
                else:
                    old = normalize_text(relationship_rules["communication_style"])
                    if content.lower() not in old.lower():
                        relationship_rules["communication_style"] = f"{old}; {content}"
                        changed = True
                        logs.append("relationship_rules.communication_style extended")

            elif category == "preferences":
                if not list_has_text(list(preferences.values()), content):
                    key = f"note_{len(preferences) + 1}"
                    preferences[key] = content
                    changed = True
                    logs.append(f"preferences.{key} = {content}")

            elif category.startswith("profile."):
                key = category.split(".", 1)[1]
                if key in ["name", "city", "language", "work", "job"]:
                    value = extract_profile_value(key, content)
                    if value and profile.get(key) != value:
                        profile[key] = value
                        changed = True
                        logs.append(f"profile.{key} = {value}")

            elif category == "entities.pets":
                pet_name = None
                pet_type = None
                color = None

                m = re.search(r"(кот|кошка|пёс|пес|собака|питомец)\s+([A-Za-zА-Яа-яЁё\-]{2,40})", content, re.IGNORECASE)
                if m:
                    pet_type = m.group(1).lower()
                    pet_name = capitalize_first(m.group(2))

                color_match = re.search(r"(серый|серая|белый|белая|чёрный|черный|чёрная|черная|рыжий|рыжая|чёрно-белый|черно-белый)", content, re.IGNORECASE)
                if color_match:
                    color = color_match.group(1).lower()

                pet = {
                    "type": pet_type or "питомец",
                    "name": pet_name or content,
                }
                if color:
                    pet["color"] = color

                if not entity_exists(entities["pets"], pet["name"]):
                    entities["pets"].append(pet)
                    changed = True
                    logs.append(f"entities.pets += {pet}")

            elif category == "entities.vehicles":
                vehicle = {"name": content}
                if not entity_exists(entities["vehicles"], vehicle["name"]):
                    entities["vehicles"].append(vehicle)
                    changed = True
                    logs.append(f"entities.vehicles += {content}")

            elif category == "entities.people":
                person = {"name": content}
                if not entity_exists(entities["people"], person["name"]):
                    entities["people"].append(person)
                    changed = True
                    logs.append(f"entities.people += {content}")

            else:
                if not list_has_text(long_term_notes, content):
                    long_term_notes.append(content)
                    changed = True
                    logs.append(f"long_term_notes += {content}")

        updated = {
            **memory_row,
            "profile": profile,
            "preferences": preferences,
            "relationship_rules": relationship_rules,
            "entities": entities,
            "interests": interests[:100],
            "projects": projects[:100],
            "long_term_notes": long_term_notes[:200],
        }

        return updated, changed, logs

    @staticmethod
    def save_structured_memory(user_id: int, memory_row: Dict[str, Any]) -> None:
        with db_cursor(commit=True) as cur:
            cur.execute(
                """
                UPDATE ai_user_memory
                SET profile = %s::jsonb,
                    preferences = %s::jsonb,
                    relationship_rules = %s::jsonb,
                    entities = %s::jsonb,
                    interests = %s::jsonb,
                    projects = %s::jsonb,
                    long_term_notes = %s::jsonb,
                    updated_at = NOW()
                WHERE user_id = %s
                """,
                (
                    json.dumps(memory_row["profile"] or {}, ensure_ascii=False),
                    json.dumps(memory_row["preferences"] or {}, ensure_ascii=False),
                    json.dumps(memory_row["relationship_rules"] or {}, ensure_ascii=False),
                    json.dumps(memory_row["entities"] or DEFAULT_MEMORY["entities"], ensure_ascii=False),
                    json.dumps(memory_row["interests"] or [], ensure_ascii=False),
                    json.dumps(memory_row["projects"] or [], ensure_ascii=False),
                    json.dumps(memory_row["long_term_notes"] or [], ensure_ascii=False),
                    user_id,
                ),
            )

    @staticmethod
    def rebuild_structured_memory_from_items(user_id: int) -> Tuple[Dict[str, Any], bool, List[str]]:
        current_memory = MemoryService.ensure_memory(user_id)

        active_items = MemoryService.list_memory_items(user_id)

        rebuilt_memory = {
            **current_memory,
            "profile": {},
            "preferences": {},
            "relationship_rules": {},
            "entities": clone_default_memory()["entities"],
            "interests": [],
            "projects": [],
            "long_term_notes": [],
        }

        updated_memory, _, logs = MemoryService.apply_ai_items_to_structured_memory(
            rebuilt_memory,
            active_items,
        )

        changed = any(
            (current_memory.get(field) or DEFAULT_MEMORY.get(field)) != updated_memory.get(field)
            for field in [
                "profile",
                "preferences",
                "relationship_rules",
                "entities",
                "interests",
                "projects",
                "long_term_notes",
            ]
        )

        if changed:
            MemoryService.save_structured_memory(user_id, updated_memory)

        return updated_memory, changed, logs

    @staticmethod
    def retrieve_relevant_memories(user_id: int, message: str, limit: int = 8) -> List[Dict[str, Any]]:
        text = normalize_text(message).lower()
        memory_query = any(
            phrase in text
            for phrase in [
                "что ты помнишь",
                "что помнишь",
                "что ты знаешь обо мне",
                "что знаешь обо мне",
                "память",
                "мои факты",
                "мои данные",
                "мои воспоминания",
                "покажи память",
                "покажи факты",
                "РїР°РјСЏС‚СЊ",
                "РјРѕРё С„Р°РєС‚С‹",
                "РјРѕРё РґР°РЅРЅС‹Рµ",
            ]
        )
        effective_limit = max(limit, 20) if memory_query else limit

        category_hints = []
        if any(x in text for x in ["кот", "кошка", "питом", "животн"]):
            category_hints.append("entities.pets")
        if any(x in text for x in ["машин", "авто", "subaru", "impreza"]):
            category_hints.append("entities.vehicles")
        if any(x in text for x in ["работ", "увол", "студ", "професс"]):
            category_hints.append("work")
        if any(x in text for x in ["проект", "ziren", "ассистент"]):
            category_hints.append("projects")
        if any(x in text for x in ["люблю", "интерес", "хобби", "нравится"]):
            category_hints.append("interests")
        if any(x in text for x in ["семь", "мама", "папа", "брат", "сестра", "друг", "девушка", "парень"]):
            category_hints.append("entities.people")
        if any(x in text for x in ["помнишь", "запомнила", "что ты знаешь", "что помнишь"]):
            category_hints.extend(["profile.name", "profile.city", "entities.pets", "projects", "interests", "long_term_notes"])

        with db_cursor(commit=True) as cur:
            if memory_query:
                cur.execute(
                    """
                    SELECT id, type, category, content, importance, confidence, sensitivity, created_at
                    FROM ai_memory_items
                    WHERE user_id = %s
                      AND status = 'active'
                    ORDER BY importance DESC, updated_at DESC
                    LIMIT %s
                    """,
                    (user_id, effective_limit),
                )
                rows = cur.fetchall()
            elif category_hints:
                cur.execute(
                    """
                    SELECT id, type, category, content, importance, confidence, sensitivity, created_at
                    FROM ai_memory_items
                    WHERE user_id = %s
                      AND status = 'active'
                      AND (category = ANY(%s) OR category = 'general')
                    ORDER BY importance DESC, updated_at DESC
                    LIMIT %s
                    """,
                    (user_id, category_hints, effective_limit),
                )
                rows = cur.fetchall()
            else:
                keywords = [
                    word
                    for word in re.findall(r"[A-Za-zА-Яа-яЁё0-9]{3,}", text)
                    if word not in {
                        "что",
                        "как",
                        "где",
                        "про",
                        "мне",
                        "тебя",
                        "это",
                        "мои",
                        "мой",
                        "моя",
                        "the",
                        "and",
                        "for",
                        "with",
                    }
                ][:8]

                rows = []
                if keywords:
                    conditions = " OR ".join(["content ILIKE %s"] * len(keywords))
                    cur.execute(
                        f"""
                        SELECT id, type, category, content, importance, confidence, sensitivity, created_at
                        FROM ai_memory_items
                        WHERE user_id = %s
                          AND status = 'active'
                          AND ({conditions})
                        ORDER BY importance DESC, updated_at DESC
                        LIMIT %s
                        """,
                        tuple([user_id, *[f"%{word}%" for word in keywords], effective_limit]),
                    )
                    rows = cur.fetchall()

                if not rows:
                    cur.execute(
                        """
                        SELECT id, type, category, content, importance, confidence, sensitivity, created_at
                        FROM ai_memory_items
                        WHERE user_id = %s
                          AND status = 'active'
                        ORDER BY importance DESC, updated_at DESC
                        LIMIT %s
                        """,
                        (user_id, effective_limit),
                    )
                    rows = cur.fetchall()

            ids = [row["id"] for row in rows]
            if ids:
                cur.execute(
                    """
                    UPDATE ai_memory_items
                    SET last_accessed_at = NOW(),
                        access_count = access_count + 1
                    WHERE id = ANY(%s)
                    """,
                    (ids,),
                )

        return [dict(row) for row in rows]

    @staticmethod
    def update_memory_from_message(user_id: int, message: str) -> Tuple[bool, List[str]]:
        memory_row = MemoryService.get_memory(user_id)

        profile = memory_row["profile"] or {}
        interests = memory_row["interests"] or []
        projects = memory_row["projects"] or []
        entities = memory_row["entities"] or DEFAULT_MEMORY["entities"].copy()

        entities.setdefault("pets", [])
        entities.setdefault("vehicles", [])
        entities.setdefault("people", [])
        entities.setdefault("other", [])

        changed = False
        logs: List[str] = []
        regex_items: List[Dict[str, Any]] = []
        low = normalize_text(message).lower()

        m = re.search(r"(?:меня зовут|мо[её] имя)\s+([A-Za-zА-Яа-яЁё\-]{2,40})", low, re.IGNORECASE)
        if m:
            name = capitalize_first(m.group(1))
            if profile.get("name") != name:
                profile["name"] = name
                changed = True
                regex_items.append({"type": "semantic", "category": "profile.name", "content": f"Меня зовут {name}", "importance": 0.8, "confidence": 0.95})
                logs.append(f"profile.name = {name}")

        m = re.search(r"(?:я живу в|я переехал в|я переехала в|мой город)\s+([A-Za-zА-Яа-яЁё\-\s]{2,60})", low, re.IGNORECASE)
        if m:
            city = capitalize_first(m.group(1).strip(" .,!?:;"))
            if profile.get("city") != city:
                profile["city"] = city
                changed = True
                regex_items.append({"type": "semantic", "category": "profile.city", "content": f"Я живу в {city}", "importance": 0.8, "confidence": 0.95})
                logs.append(f"profile.city = {city}")

        m = re.search(r"(?:я говорю на|я говорю по|мой язык)\s+([A-Za-zА-Яа-яЁё\-]{2,40})", low, re.IGNORECASE)
        if m:
            language = normalize_text(m.group(1))
            if profile.get("language") != language:
                profile["language"] = language
                changed = True
                regex_items.append({"type": "semantic", "category": "profile.language", "content": f"Я говорю на {language}", "importance": 0.8, "confidence": 0.95})
                logs.append(f"profile.language = {language}")

        m = re.search(
            r"(?:у меня есть|мой|моя)\s+(кот|кошка|пёс|пес|собака|питомец)\s+([A-Za-zА-Яа-яЁё\-]{2,40})(?:.*?(серый|серая|белый|белая|чёрный|черный|чёрная|черная|рыжий|рыжая))?",
            low,
            re.IGNORECASE,
        )
        if m:
            pet_type = m.group(1).lower()
            pet_name = capitalize_first(m.group(2))
            pet_color = m.group(3).lower() if m.group(3) else None

            pet = {"type": pet_type, "name": pet_name}
            if pet_color:
                pet["color"] = pet_color

            if not entity_exists(entities["pets"], pet_name):
                entities["pets"].append(pet)
                changed = True
                pet_content = f"{pet_type} {pet_name}"
                if pet_color:
                    pet_content = f"{pet_content}, {pet_color}"
                regex_items.append({"type": "semantic", "category": "entities.pets", "content": pet_content, "importance": 0.7, "confidence": 0.9})
                logs.append(f"entities.pets += {pet}")

        m = re.search(r"(?:у меня|моя машина|мой автомобиль)\s+(?:есть\s+)?([A-Za-zА-Яа-яЁё0-9\-\s]{2,80})", low, re.IGNORECASE)
        if m and any(x in low for x in ["машина", "автомобиль", "subaru", "impreza"]):
            vehicle_name = capitalize_first(m.group(1).strip(" .,!?:;"))
            vehicle = {"name": vehicle_name}
            if vehicle_name and not entity_exists(entities["vehicles"], vehicle_name):
                entities["vehicles"].append(vehicle)
                changed = True
                regex_items.append({"type": "semantic", "category": "entities.vehicles", "content": vehicle_name, "importance": 0.7, "confidence": 0.9})
                logs.append(f"entities.vehicles += {vehicle_name}")

        for pattern in [
            r"(?:я люблю)\s+(.+)$",
            r"(?:мне нравится)\s+(.+)$",
            r"(?:я увлекаюсь)\s+(.+)$",
            r"(?:мое хобби|моё хобби)\s+(.+)$",
        ]:
            m = re.search(pattern, low, re.IGNORECASE)
            if m:
                interest = capitalize_first(m.group(1).strip(" .,!?:;"))
                if interest and not list_has_text(interests, interest):
                    interests.append(interest)
                    changed = True
                    regex_items.append({"type": "semantic", "category": "interests", "content": interest, "importance": 0.6, "confidence": 0.85})
                    logs.append(f"interests += {interest}")
                break

        for pattern in [
            r"(?:я делаю)\s+(.+)$",
            r"(?:я создаю)\s+(.+)$",
            r"(?:я пишу)\s+(.+)$",
            r"(?:я работаю над)\s+(.+)$",
            r"(?:занимаюсь проектом)\s+(.+)$",
        ]:
            m = re.search(pattern, low, re.IGNORECASE)
            if m:
                project = capitalize_first(m.group(1).strip(" .,!?:;"))
                if project and not list_has_text(projects, project):
                    projects.append(project)
                    changed = True
                    regex_items.append({"type": "semantic", "category": "projects", "content": project, "importance": 0.7, "confidence": 0.85})
                    logs.append(f"projects += {project}")
                break

        if regex_items:
            regex_items_changed, regex_item_logs = MemoryService.save_memory_items(user_id, regex_items, message)
            _, structured_changed, structured_logs = MemoryService.rebuild_structured_memory_from_items(user_id)
            if regex_items_changed or structured_changed:
                changed = True
                logs.extend(regex_item_logs)
                logs.extend(structured_logs)

        if MemoryService.should_run_ai_memory_analysis(message, changed):
            ai_result = MemoryService.extract_memory_with_ai(user_id, message)
            items = ai_result.get("items", []) if ai_result.get("should_save") else []

            if items:
                ai_items_changed, ai_item_logs = MemoryService.save_memory_items(user_id, items, message)
                _, structured_changed, structured_logs = MemoryService.rebuild_structured_memory_from_items(user_id)

                if ai_items_changed or structured_changed:
                    changed = True
                    logs.extend(ai_item_logs)
                    logs.extend(structured_logs)

        return changed, logs
