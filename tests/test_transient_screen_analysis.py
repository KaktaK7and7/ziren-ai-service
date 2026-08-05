import unittest
from unittest.mock import Mock, patch

from app.transient_screen_analysis import (
    TRANSIENT_SCREEN_RETRY_MARKER,
    analyze_screen_without_history,
    is_transient_screen_retry,
)


class TransientScreenAnalysisTests(unittest.TestCase):
    def test_marker_survives_whitespace_normalization(self) -> None:
        message = (
            "покажи кнопку\n\n"
            "Служебное уточнение: это увеличенный фрагмент "
            "предыдущего снимка вокруг вероятной цели."
        )

        self.assertTrue(is_transient_screen_retry(message))
        self.assertFalse(is_transient_screen_retry("покажи кнопку ассистент"))
        self.assertIn("увеличенный фрагмент", TRANSIENT_SCREEN_RETRY_MARKER)

    @patch("app.transient_screen_analysis.ChatService.save_message")
    @patch("app.transient_screen_analysis.ChatService.generate_screen_analysis_plan")
    @patch("app.transient_screen_analysis.ChatService.build_screen_analysis_messages")
    @patch("app.transient_screen_analysis.ChatService.get_recent_messages")
    @patch("app.transient_screen_analysis.ChatService.get_or_create_session")
    @patch("app.transient_screen_analysis.MemoryService.ensure_memory")
    @patch("app.transient_screen_analysis.PersonaService.ensure_persona")
    def test_retry_uses_context_without_writing_history(
        self,
        ensure_persona: Mock,
        ensure_memory: Mock,
        get_or_create_session: Mock,
        get_recent_messages: Mock,
        build_messages: Mock,
        generate_plan: Mock,
        save_message: Mock,
    ) -> None:
        expected_plan = Mock()
        ensure_persona.return_value = {"name": "Мелисса"}
        ensure_memory.return_value = {"profile": {}}
        get_or_create_session.return_value = 42
        get_recent_messages.return_value = [
            {"role": "user", "content": "покажи кнопку"},
        ]
        build_messages.return_value = [{"role": "system", "content": "test"}]
        generate_plan.return_value = expected_plan

        plan, session_id = analyze_screen_without_history(
            user_id=7,
            message=(
                "покажи кнопку\n\n"
                "Служебное уточнение: это увеличенный фрагмент "
                "предыдущего снимка вокруг вероятной цели."
            ),
            image_data_url="data:image/jpeg;base64,test",
            session_id=42,
        )

        self.assertIs(plan, expected_plan)
        self.assertEqual(session_id, 42)
        get_recent_messages.assert_called_once_with(42, limit=6)
        generate_plan.assert_called_once()
        save_message.assert_not_called()


if __name__ == "__main__":
    unittest.main()
