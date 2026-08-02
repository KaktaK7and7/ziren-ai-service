import base64
import unittest
from unittest.mock import patch

from app.main import build_drawing_prompt, generate_drawing
from app.schemas import DrawingGenerateRequest


class DrawingGenerationTests(unittest.TestCase):
    def test_style_prompt_is_rough_original_and_marks_technical_limits(
        self,
    ) -> None:
        prompt = build_drawing_prompt(DrawingGenerateRequest(
            user_id=1,
            kind="technical",
            title="Робо-рука",
            prompt="Манипулятор с тремя суставами",
        ))

        self.assertIn("graphite-pencil rough draft", prompt)
        self.assertIn("construction lines", prompt)
        self.assertIn("original", prompt)
        self.assertIn("never invent exact dimensions", prompt)
        self.assertIn("not fabrication-ready", prompt)

    def test_generated_png_is_returned_with_checksum(self) -> None:
        image_bytes = b"\x89PNG\r\n\x1a\n" + b"test-image"

        with patch(
            "app.main.OpenAIService.generate_image",
            return_value={
                "image_base64": base64.b64encode(image_bytes).decode("ascii"),
            },
        ):
            response = generate_drawing(DrawingGenerateRequest(
                user_id=1,
                title="Набросок",
                prompt="Небольшой механизм",
            ))

        self.assertTrue(
            response.image_data_url.startswith("data:image/png;base64,"),
        )
        self.assertEqual(len(response.sha256), 64)


if __name__ == "__main__":
    unittest.main()
