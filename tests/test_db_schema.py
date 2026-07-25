import unittest
from contextlib import contextmanager
from unittest.mock import patch

from app import db_schema


class RecordingCursor:
    def __init__(self) -> None:
        self.statements = []

    def execute(self, statement: str) -> None:
        self.statements.append(statement)


class DatabaseSchemaTests(unittest.TestCase):
    def test_ensure_schema_creates_every_table_idempotently(self) -> None:
        cursor = RecordingCursor()
        commit_values = []

        @contextmanager
        def recording_db_cursor(commit: bool = False):
            commit_values.append(commit)
            yield cursor

        with patch.object(db_schema, "db_cursor", recording_db_cursor):
            db_schema.ensure_schema()

        combined_sql = "\n".join(cursor.statements)
        required_tables = (
            "ai_personas",
            "ai_user_memory",
            "ai_memory_items",
            "ai_chat_sessions",
            "ai_chat_messages",
            "ai_summaries",
            "ai_metrics",
        )

        self.assertEqual(commit_values, [True])
        for table_name in required_tables:
            with self.subTest(table_name=table_name):
                self.assertIn(
                    f"CREATE TABLE IF NOT EXISTS {table_name}",
                    combined_sql,
                )

        self.assertNotIn("DROP TABLE", combined_sql.upper())


if __name__ == "__main__":
    unittest.main()
