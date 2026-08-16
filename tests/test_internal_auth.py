import os
import unittest

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("AI_INTERNAL_TOKEN", "test-internal-token")

from fastapi.testclient import TestClient

from app.main import app


class InternalGatewayAuthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health_remains_public(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json().get("status"), "ok")

    def test_missing_internal_token_is_rejected_before_route_validation(self):
        response = self.client.post("/command-route", json={})
        self.assertEqual(response.status_code, 401)

    def test_wrong_internal_token_is_rejected(self):
        response = self.client.post(
            "/command-route",
            headers={"X-Ziren-Internal-Token": "wrong-token"},
            json={},
        )
        self.assertEqual(response.status_code, 401)

    def test_correct_internal_token_reaches_route_validation(self):
        response = self.client.post(
            "/command-route",
            headers={"X-Ziren-Internal-Token": "test-internal-token"},
            json={},
        )
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
