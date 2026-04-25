import json
from typing import Any, Dict

from app.db import db_cursor
from app.presets import DEFAULT_PERSONA, PERSONA_PRESETS


class PersonaService:
    @staticmethod
    def ensure_persona(user_id: int) -> Dict[str, Any]:
        with db_cursor(commit=True) as cur:
            cur.execute(
                """
                INSERT INTO ai_personas (
                    user_id, preset_name, name, identity,
                    core_traits, speech_style, behavior_rules, speech_habits
                )
                VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb)
                ON CONFLICT (user_id) DO NOTHING
                """,
                (
                    user_id,
                    DEFAULT_PERSONA["preset_name"],
                    DEFAULT_PERSONA["name"],
                    DEFAULT_PERSONA["identity"],
                    json.dumps(DEFAULT_PERSONA["core_traits"], ensure_ascii=False),
                    json.dumps(DEFAULT_PERSONA["speech_style"], ensure_ascii=False),
                    json.dumps(DEFAULT_PERSONA["behavior_rules"], ensure_ascii=False),
                    json.dumps(DEFAULT_PERSONA["speech_habits"], ensure_ascii=False),
                ),
            )

        return PersonaService.get_persona(user_id)

    @staticmethod
    def get_persona(user_id: int) -> Dict[str, Any]:
        with db_cursor() as cur:
            cur.execute("SELECT * FROM ai_personas WHERE user_id = %s", (user_id,))
            row = cur.fetchone()
            if not row:
                return PersonaService.ensure_persona(user_id)
            return dict(row)
        

    @staticmethod
    def update_name(user_id: int, name: str):
        name = str(name).strip()
        if not name:
            raise ValueError("name is required")

        PersonaService.ensure_persona(user_id)

        with db_cursor(commit=True) as cur:
            cur.execute("""
                UPDATE ai_personas
                SET name = %s, updated_at = NOW()
                WHERE user_id = %s
                RETURNING *
            """, (name, user_id))

            return cur.fetchone()


    @staticmethod
    def apply_preset(user_id: int, preset_name: str) -> Dict[str, Any]:
        preset = PERSONA_PRESETS.get(preset_name)
        if not preset:
            raise ValueError("Preset not found")

        with db_cursor(commit=True) as cur:
            cur.execute(
                """
                INSERT INTO ai_personas (
                    user_id, preset_name, name, identity,
                    core_traits, speech_style, behavior_rules, speech_habits, updated_at
                )
                VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, NOW())
                ON CONFLICT (user_id) DO UPDATE SET
                    preset_name = EXCLUDED.preset_name,
                    identity = EXCLUDED.identity,
                    core_traits = EXCLUDED.core_traits,
                    speech_style = EXCLUDED.speech_style,
                    behavior_rules = EXCLUDED.behavior_rules,
                    speech_habits = EXCLUDED.speech_habits,
                    updated_at = NOW()
                """,
                (
                    user_id,
                    preset["preset_name"],
                    preset["name"],
                    preset["identity"],
                    json.dumps(preset["core_traits"], ensure_ascii=False),
                    json.dumps(preset["speech_style"], ensure_ascii=False),
                    json.dumps(preset["behavior_rules"], ensure_ascii=False),
                    json.dumps(preset["speech_habits"], ensure_ascii=False),
                ),
            )

        return PersonaService.get_persona(user_id)
