import hmac


INTERNAL_TOKEN_HEADER = "X-Ziren-Internal-Token"


def is_valid_internal_token(provided_token: str, expected_token: str) -> bool:
    provided = str(provided_token or "").strip()
    expected = str(expected_token or "").strip()

    if not provided or not expected:
        return False

    return hmac.compare_digest(
        provided.encode("utf-8"),
        expected.encode("utf-8"),
    )


def get_internal_auth_error(
    provided_token: str,
    expected_token: str,
) -> tuple[int, str] | None:
    if not str(expected_token or "").strip():
        return 503, "AI service internal authentication is not configured"

    if not is_valid_internal_token(provided_token, expected_token):
        return 401, "Invalid internal service credentials"

    return None
