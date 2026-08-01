import base64
import unittest

from fastapi import HTTPException

from app.main import (
    MAX_SCREENSHOT_BYTES,
    SCREENSHOT_DATA_URL_PREFIX,
    validate_screenshot_data_url,
)


class ScreenshotValidationTests(unittest.TestCase):
    @staticmethod
    def data_url(payload: bytes) -> str:
        encoded = base64.b64encode(payload).decode("ascii")
        return f"{SCREENSHOT_DATA_URL_PREFIX}{encoded}"

    def test_accepts_bounded_jpeg(self) -> None:
        self.assertIsNone(
            validate_screenshot_data_url(
                self.data_url(b"\xff\xd8\xff\xe0\x00\x10"),
            ),
        )

    def test_rejects_wrong_format_bad_base64_and_oversized_payload(self) -> None:
        invalid_values = (
            "data:image/png;base64,iVBORw0KGgo=",
            f"{SCREENSHOT_DATA_URL_PREFIX}not base64!",
            self.data_url(b"\x89PNG"),
            self.data_url(
                b"\xff\xd8\xff" + b"0" * MAX_SCREENSHOT_BYTES,
            ),
        )

        for value in invalid_values:
            with self.subTest(value=value[:48]):
                with self.assertRaises(HTTPException):
                    validate_screenshot_data_url(value)


if __name__ == "__main__":
    unittest.main()
