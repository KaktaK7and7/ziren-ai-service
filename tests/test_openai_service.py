import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.openai_service import OpenAIService


class OpenAIServiceTests(unittest.TestCase):
    def test_structured_generation_uses_a_strict_json_schema(self) -> None:
        create = Mock(return_value=SimpleNamespace(
            output_text=json.dumps({"answer": "Готово"}),
        ))
        fake_client = SimpleNamespace(
            responses=SimpleNamespace(create=create),
        )
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["answer"],
            "properties": {"answer": {"type": "string"}},
        }

        with patch("app.openai_service.client", fake_client):
            result = OpenAIService.generate_structured(
                "model-test",
                [{"role": "user", "content": "test"}],
                "screen_test",
                schema,
            )

        self.assertEqual(result, {"answer": "Готово"})
        request = create.call_args.kwargs
        self.assertTrue(request["text"]["format"]["strict"])
        self.assertEqual(
            request["text"]["format"]["schema"],
            schema,
        )


if __name__ == "__main__":
    unittest.main()
