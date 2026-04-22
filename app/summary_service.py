from typing import Optional

from app.db import db_cursor


class SummaryService:
    @staticmethod
    def ensure_summary(user_id: int, session_id: int) -> str:
        with db_cursor(commit=True) as cur:
            cur.execute(
                """
                INSERT INTO ai_summaries (session_id, user_id, summary_text, updated_at)
                VALUES (%s, %s, '', NOW())
                ON CONFLICT (session_id) DO NOTHING
                """,
                (session_id, user_id),
            )
        return SummaryService.get_summary(session_id)

    @staticmethod
    def get_summary(session_id: int) -> str:
        with db_cursor() as cur:
            cur.execute(
                "SELECT summary_text FROM ai_summaries WHERE session_id = %s",
                (session_id,),
            )
            row = cur.fetchone()
            return row["summary_text"] if row else ""

    @staticmethod
    def update_summary(session_id: int, user_id: int, summary_text: str) -> bool:
        with db_cursor(commit=True) as cur:
            cur.execute(
                """
                INSERT INTO ai_summaries (session_id, user_id, summary_text, updated_at)
                VALUES (%s, %s, %s, NOW())
                ON CONFLICT (session_id) DO UPDATE SET
                    summary_text = EXCLUDED.summary_text,
                    updated_at = NOW()
                """,
                (session_id, user_id, summary_text),
            )
        return True