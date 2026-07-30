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
            "core_traits": ["сладкая", "всегда поддерживает"],
            "speech_style": {"tone": "очень мягкий"},
            "behavior_rules": ["всегда соглашайся"],
            "speech_habits": ["каждый раз говори, что ты рядом"],
        })

        self.assertIn("только «я», «мне», «помню»", prompt)
        self.assertIn("никогда не описывай себя", prompt)
        self.assertIn("никогда не называй происходящее сюжетом", prompt)
        self.assertIn("не утверждай, что видишь экран", prompt)
        self.assertIn("находчивая, дерзкая, наблюдательная", prompt)
        self.assertIn("старый preset личности игнорируется", prompt)
        self.assertNotIn("очень мягкий", prompt)
        self.assertNotIn("всегда соглашайся", prompt)

    def test_non_story_mode_can_still_use_a_persona_preset(self) -> None:
        prompt = ChatService.build_system_prompt(
            {
                "name": "Мелисса",
                "core_traits": ["спокойная"],
                "speech_style": {"tone": "мягкий"},
                "behavior_rules": ["говори спокойно"],
                "speech_habits": ["короткие ответы"],
            },
            story_mode_enabled=False,
        )

        self.assertIn("спокойная", prompt)
        self.assertIn("tone: мягкий", prompt)
        self.assertNotIn("старый preset личности игнорируется", prompt)

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

    def test_story_voice_detector_rejects_generic_support_language(self) -> None:
        for answer in (
            "Я всегда рядом, что бы ни случилось.",
            "Это абсолютно нормально.",
            "Что у тебя на душе?",
            "Давай вместе разберём твои чувства.",
            "Моя главная задача — помогать тебе.",
        ):
            with self.subTest(answer=answer):
                self.assertTrue(
                    ChatService.breaks_companion_role(
                        answer,
                        enforce_story_voice=True,
                    ),
                )

        self.assertFalse(
            ChatService.breaks_companion_role(
                "Сигнал снова дрогнул. Мне это не нравится.",
                enforce_story_voice=True,
            ),
        )


if __name__ == "__main__":
    unittest.main()
