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


def capitalize_first(value: str) -> str:
    value = normalize_text(value)
    return value[:1].upper() + value[1:] if value else value


def list_has_text(items: List[Any], text: str) -> bool:
    target = normalize_text(text).lower()
    for item in items:
        if normalize_text(item).lower() == target:
            return True
    return False


def entity_exists(items: List[Dict[str, Any]], name: str) -> bool:
    target = normalize_text(name).lower()
    for item in items:
        if normalize_text(item.get("name", "")).lower() == target:
            return True
    return False


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
            for item in items:
                content = normalize_text(item.get("content", ""))
                if not content:
                    continue

                category = normalize_text(item.get("category", "general")) or "general"

                cur.execute(
                    """
                    SELECT id
                    FROM ai_memory_items
                    WHERE user_id = %s
                      AND status = 'active'
                      AND lower(content) = lower(%s)
                    LIMIT 1
                    """,
                    (user_id, content),
                )
                duplicate = cur.fetchone()
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
            category = normalize_text(item.get("category", ""))
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
                    value = content
                    value = re.sub(r"^Пользователь\s+", "", value, flags=re.IGNORECASE)
                    value = re.sub(r"^(живёт в|говорит на|говорит по|работает в|работает как|зовут)\s+", "", value, flags=re.IGNORECASE)
                    value = capitalize_first(value)
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
    def retrieve_relevant_memories(user_id: int, message: str, limit: int = 8) -> List[Dict[str, Any]]:
        text = normalize_text(message).lower()

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
            if category_hints:
                cur.execute(
                    """
                    SELECT id, type, category, content, importance, confidence, sensitivity, created_at
                    FROM ai_memory_items
                    WHERE user_id = %s
                      AND status = 'active'
                      AND category = ANY(%s)
                    ORDER BY importance DESC, updated_at DESC
                    LIMIT %s
                    """,
                    (user_id, category_hints, limit),
                )
            else:
                cur.execute(
                    """
                    SELECT id, type, category, content, importance, confidence, sensitivity, created_at
                    FROM ai_memory_items
                    WHERE user_id = %s
                      AND status = 'active'
                    ORDER BY importance DESC, updated_at DESC
                    LIMIT %s
                    """,
                    (user_id, limit),
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
        low = normalize_text(message).lower()

        m = re.search(r"(?:меня зовут|мо[её] имя)\s+([A-Za-zА-Яа-яЁё\-]{2,40})", low, re.IGNORECASE)
        if m:
            name = capitalize_first(m.group(1))
            if profile.get("name") != name:
                profile["name"] = name
                changed = True
                logs.append(f"profile.name = {name}")

        m = re.search(r"(?:я живу в|я переехал в|я переехала в|мой город)\s+([A-Za-zА-Яа-яЁё\-\s]{2,60})", low, re.IGNORECASE)
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
                logs.append(f"entities.pets += {pet}")

        m = re.search(r"(?:у меня|моя машина|мой автомобиль)\s+(?:есть\s+)?([A-Za-zА-Яа-яЁё0-9\-\s]{2,80})", low, re.IGNORECASE)
        if m and any(x in low for x in ["машина", "автомобиль", "subaru", "impreza"]):
            vehicle_name = capitalize_first(m.group(1).strip(" .,!?:;"))
            vehicle = {"name": vehicle_name}
            if vehicle_name and not entity_exists(entities["vehicles"], vehicle_name):
                entities["vehicles"].append(vehicle)
                changed = True
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
                    logs.append(f"projects += {project}")
                break

        if changed:
            memory_row["profile"] = profile
            memory_row["interests"] = interests
            memory_row["projects"] = projects
            memory_row["entities"] = entities
            MemoryService.save_structured_memory(user_id, memory_row)

        if MemoryService.should_run_ai_memory_analysis(message, changed):
            ai_result = MemoryService.extract_memory_with_ai(user_id, message)
            items = ai_result.get("items", []) if ai_result.get("should_save") else []

            if items:
                ai_items_changed, ai_item_logs = MemoryService.save_memory_items(user_id, items, message)
                updated_memory, structured_changed, structured_logs = MemoryService.apply_ai_items_to_structured_memory(
                    MemoryService.get_memory(user_id),
                    items,
                )

                if structured_changed:
                    MemoryService.save_structured_memory(user_id, updated_memory)

                if ai_items_changed or structured_changed:
                    changed = True
                    logs.extend(ai_item_logs)
                    logs.extend(structured_logs)

        return changed, logs