import unittest
from unittest.mock import patch

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

    def test_living_story_name_overrides_stale_persona_name(self) -> None:
        prompt = ChatService.build_system_prompt(
            {
                "name": "Мелисса",
                "core_traits": [],
                "speech_style": {},
                "behavior_rules": [],
                "speech_habits": [],
            },
            story_mode_enabled=True,
            companion_name="Искра",
        )

        self.assertIn('"Искра"', prompt)
        self.assertNotIn('"Мелисса"', prompt)

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

    def test_command_reaction_task_follows_chat_history(self) -> None:
        messages = ChatService.build_companion_line_messages(
            persona={
                "name": "Мелисса",
                "core_traits": [],
                "speech_style": {},
                "behavior_rules": [],
                "speech_habits": [],
            },
            memory_row={},
            recent_messages=[
                {
                    "role": "user",
                    "content": "Почему прошлый запуск занял так много времени?",
                },
            ],
            instruction=(
                'Отреагируй на текущую команду: '
                '{"recognized_command":"открыть Cyberpunk 2077",'
                '"local_result":"игра запущена"}'
            ),
        )

        self.assertEqual(messages[-2]["role"], "user")
        self.assertEqual(messages[-1]["role"], "developer")
        self.assertIn("открыть Cyberpunk 2077", messages[-1]["content"])
        self.assertIn("только как фон", messages[-1]["content"])
        self.assertIn("говори только по теме этой команды", messages[-1]["content"])

    def test_command_reaction_can_exclude_stale_chat_history(self) -> None:
        with (
            patch(
                "app.chat_service.PersonaService.ensure_persona",
                return_value={
                    "name": "Мелисса",
                    "core_traits": [],
                    "speech_style": {},
                    "behavior_rules": [],
                    "speech_habits": [],
                },
            ),
            patch(
                "app.chat_service.MemoryService.ensure_memory",
                return_value={},
            ),
            patch.object(
                ChatService,
                "get_or_create_session",
                return_value=17,
            ),
            patch.object(ChatService, "get_recent_messages") as recent_messages,
            patch.object(
                ChatService,
                "generate_role_safe_reply",
                return_value=("Запускаешь игру без разминки. Смело.", None),
            ),
        ):
            answer, session_id = ChatService.generate_companion_line(
                user_id=3,
                instruction="Отреагируй на запуск игры.",
                include_recent_messages=False,
            )

        self.assertEqual(answer, "Запускаешь игру без разминки. Смело.")
        self.assertEqual(session_id, 17)
        recent_messages.assert_not_called()


if __name__ == "__main__":
    unittest.main()
