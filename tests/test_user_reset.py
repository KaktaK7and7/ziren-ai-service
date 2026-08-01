import unittest
from unittest.mock import MagicMock, patch

from app.memory_service import MemoryService


class UserResetTests(unittest.TestCase):
    def test_reset_removes_companion_data_but_not_account_data(self) -> None:
        cursor = MagicMock()
        context = MagicMock()
        context.__enter__.return_value = cursor
        context.__exit__.return_value = False

        with patch("app.memory_service.db_cursor", return_value=context) as db_cursor:
            result = MemoryService.reset_all_user_data(17)

        self.assertEqual(result, {"ok": True})
        db_cursor.assert_called_once_with(commit=True)
        statements = [call.args[0] for call in cursor.execute.call_args_list]
        self.assertEqual(len(statements), 4)
        self.assertTrue(any("ai_memory_items" in statement for statement in statements))
        self.assertTrue(any("ai_chat_sessions" in statement for statement in statements))
        self.assertTrue(any("ai_user_memory" in statement for statement in statements))
        self.assertTrue(any("ai_personas" in statement for statement in statements))
        for call in cursor.execute.call_args_list:
            self.assertEqual(call.args[1], (17,))


if __name__ == "__main__":
    unittest.main()
