import os
import unittest
from contextlib import contextmanager
from unittest.mock import patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("AI_INTERNAL_TOKEN", "test-internal-token")

from fastapi.testclient import TestClient

from app.main import app


AUTH = {"X-Ziren-Internal-Token": "test-internal-token"}


class _Cursor:
    def execute(self, _query):
        return None

    def fetchone(self):
        return {"ready": 1}


@contextmanager
def _healthy_db_cursor():
    yield _Cursor()


class ReleaseReadinessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_readiness_is_not_public(self):
        response = self.client.get("/ready")
        self.assertEqual(response.status_code, 401)

    def test_readiness_fails_closed_when_configuration_is_incomplete(self):
        with patch("app.main.settings.DATABASE_URL", ""):
            response = self.client.get("/ready", headers=AUTH)
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json().get("detail"), "Service configuration incomplete")

    def test_readiness_checks_database_when_configured(self):
        with patch("app.main.settings.OPENAI_API_KEY", "test-key"), patch(
            "app.main.settings.DATABASE_URL", "postgresql://example.invalid/test"
        ), patch("app.main.db_cursor", _healthy_db_cursor):
            response = self.client.get("/ready", headers=AUTH)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json().get("status"), "ready")

    def test_internal_exception_text_is_not_returned_to_gateway(self):
        secret_detail = "postgresql://user:super-secret@example.invalid/db"
        with patch(
            "app.main.PersonaService.ensure_persona",
            side_effect=RuntimeError(secret_detail),
        ):
            response = self.client.get("/persona/1", headers=AUTH)

        self.assertEqual(response.status_code, 500)
        body = response.text
        self.assertIn("Internal service error", body)
        self.assertNotIn("super-secret", body)


if __name__ == "__main__":
    unittest.main()
