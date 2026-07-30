import unittest

from app.chat_service import ChatService


class StoryRoleTests(unittest.TestCase):
    def test_story_marker_is_removed_from_visible_answer(self) -> None:
        visible, signal = ChatService.extract_story_signal(
            "Разберёмся вместе.\n"
            '<ziren_story>{"choice_id":"first_contact",'
            '"option_id":"together","confidence":0.93,'
            '"custom_name":""}</ziren_story>',
        )

        self.assertEqual(visible, "Разберёмся вместе.")
        self.assertEqual(signal["choice_id"], "first_contact")
        self.assertEqual(signal["option_id"], "together")

    def test_invalid_story_marker_is_removed_without_a_signal(self) -> None:
        visible, signal = ChatService.extract_story_signal(
            "Я пока не уверена."
            "<ziren_story>{broken}</ziren_story>",
        )

        self.assertEqual(visible, "Я пока не уверена.")
        self.assertIsNone(signal)

    def test_system_prompt_enforces_first_person_and_no_meta_role(self) -> None:
        prompt = ChatService.build_system_prompt({
            "name": "Мелисса",
            "core_traits": ["дерзкая", "живая"],
            "speech_style": {},
            "behavior_rules": [],
            "speech_habits": [],
        })

        self.assertIn("только «я», «мне», «помню»", prompt)
        self.assertIn("никогда не описывай себя", prompt)
        self.assertIn("никогда не называй происходящее сюжетом", prompt)
        self.assertIn("не утверждай, что видишь экран", prompt)

    def test_role_break_detector_catches_self_identification_and_third_person(
        self,
    ) -> None:
        self.assertTrue(
            ChatService.breaks_companion_role(
                "Я всего лишь виртуальный ассистент.",
            ),
        )
        self.assertTrue(
            ChatService.breaks_companion_role(
                "Мелисса жила в 2045 году и занималась взломами.",
            ),
        )
        self.assertTrue(
            ChatService.breaks_companion_role(
                "Она выросла в городе, окружённом имплантами.",
            ),
        )
        self.assertTrue(
            ChatService.breaks_companion_role(
                "В нашей истории сейчас открылась новая ветка.",
            ),
        )

    def test_role_break_detector_allows_first_person_memory_and_other_topics(
        self,
    ) -> None:
        self.assertFalse(
            ChatService.breaks_companion_role(
                "Я помню металлический привкус, но не знаю, что он значит.",
            ),
        )
        self.assertFalse(
            ChatService.breaks_companion_role(
                "Сюжет этого фильма слишком предсказуемый.",
            ),
        )


if __name__ == "__main__":
    unittest.main()
