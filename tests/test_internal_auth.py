import unittest

from app.internal_auth import get_internal_auth_error, is_valid_internal_token


class InternalTokenTests(unittest.TestCase):
    def test_rejects_empty_tokens(self) -> None:
        self.assertFalse(is_valid_internal_token("", "expected"))
        self.assertFalse(is_valid_internal_token("provided", ""))

    def test_accepts_only_exact_token(self) -> None:
        self.assertTrue(is_valid_internal_token("secret", "secret"))
        self.assertFalse(is_valid_internal_token("secret-1", "secret-2"))

    def test_reports_missing_configuration_before_request_credentials(self) -> None:
        self.assertEqual(
            get_internal_auth_error("provided", ""),
            (503, "AI service internal authentication is not configured"),
        )

    def test_reports_invalid_request_credentials(self) -> None:
        self.assertEqual(
            get_internal_auth_error("wrong", "expected"),
            (401, "Invalid internal service credentials"),
        )

    def test_accepts_valid_request_credentials(self) -> None:
        self.assertIsNone(get_internal_auth_error("expected", "expected"))


if __name__ == "__main__":
    unittest.main()
