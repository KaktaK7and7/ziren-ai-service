import os
import unittest
from unittest.mock import patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from app.command_router_service import CommandRouterService
from app.openai_service import OpenAIService
from app.schemas import CommandRouteRequest


CAPABILITIES = [
    {
        "feature_id": "system.text_input",
        "actions": [
            {
                "action_id": "text.type",
                "display_name": "Ввести текст",
                "argument_hint": "arguments.text",
            }
        ],
    },
    {
        "feature_id": "system.volume",
        "actions": [
            {
                "action_id": "volume.set",
                "display_name": "Громкость",
                "argument_hint": "arguments.percent",
            }
        ],
    },
]


class CommandRouterServiceTests(unittest.TestCase):
    def _resolve(self, raw):
        payload = CommandRouteRequest(
            message="тестовая команда",
            capabilities=CAPABILITIES,
        )
        with patch.object(OpenAIService, "generate_json", return_value=raw):
            return CommandRouterService.resolve(payload)

    def test_allowed_action_is_returned_with_safe_arguments_only(self):
        result = self._resolve(
            {
                "command_like": True,
                "matched": True,
                "feature_id": "system.text_input",
                "action_id": "text.type",
                "arguments": {
                    "text": "привет",
                    "powershell": "Remove-Item C:\\*",
                    "nested": {"bad": True},
                },
                "confidence": 0.97,
                "reason": "user asked to type text",
            }
        )
        self.assertTrue(result.command_like)
        self.assertTrue(result.matched)
        self.assertEqual(result.feature_id, "system.text_input")
        self.assertEqual(result.action_id, "text.type")
        self.assertEqual(result.arguments, {"text": "привет"})
        self.assertTrue(result.reason.startswith("command:"))

    def test_hallucinated_action_is_rejected(self):
        result = self._resolve(
            {
                "command_like": True,
                "matched": True,
                "feature_id": "system.shell",
                "action_id": "shell.run",
                "arguments": {"text": "whoami"},
                "confidence": 0.99,
            }
        )
        self.assertTrue(result.command_like)
        self.assertFalse(result.matched)
        self.assertTrue(result.reason.startswith("command:"))

    def test_low_confidence_action_is_rejected_but_stays_command_like(self):
        result = self._resolve(
            {
                "command_like": True,
                "matched": True,
                "feature_id": "system.volume",
                "action_id": "volume.set",
                "arguments": {"percent": 40},
                "confidence": 0.55,
            }
        )
        self.assertTrue(result.command_like)
        self.assertFalse(result.matched)
        self.assertEqual(result.confidence, 0.55)

    def test_unmatched_command_does_not_become_chat(self):
        result = self._resolve(
            {
                "command_like": True,
                "matched": False,
                "confidence": 0.4,
                "reason": "no supported local action",
            }
        )
        self.assertTrue(result.command_like)
        self.assertFalse(result.matched)
        self.assertTrue(result.reason.startswith("command:"))

    def test_ordinary_conversation_is_marked_as_chat(self):
        result = self._resolve(
            {
                "command_like": False,
                "matched": False,
                "confidence": 0.99,
                "reason": "ordinary conversation",
            }
        )
        self.assertFalse(result.command_like)
        self.assertFalse(result.matched)
        self.assertTrue(result.reason.startswith("chat:"))

    def test_chat_classification_wins_over_inconsistent_matched_fields(self):
        result = self._resolve(
            {
                "command_like": False,
                "matched": True,
                "feature_id": "system.volume",
                "action_id": "volume.set",
                "arguments": {"percent": 100},
                "confidence": 0.99,
                "reason": "ordinary conversation",
            }
        )
        self.assertFalse(result.command_like)
        self.assertFalse(result.matched)
        self.assertEqual(result.feature_id, "")
        self.assertEqual(result.action_id, "")
        self.assertEqual(result.arguments, {})
        self.assertTrue(result.reason.startswith("chat:"))


if __name__ == "__main__":
    unittest.main()
