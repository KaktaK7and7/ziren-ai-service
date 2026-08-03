import unittest
from unittest.mock import patch

from app.chat_service import ChatService
from app.schemas import ScreenActionProposal, ScreenAnalysisPlan, ScreenAnnotation
from app.proactive_prompt import build_proactive_instruction


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

    def test_drawing_marker_is_removed_and_normalized(self) -> None:
        visible, drawing = ChatService.extract_drawing_request(
            "Ладно, набросаю. "
            '<ziren_drawing>{"kind":"technical",'
            '"title":"Робо-рука","prompt":"Схема суставов",'
            '"story_relevant":false,'
            '"completion_line":"Ну? Что скажешь?"}</ziren_drawing>',
        )

        self.assertEqual(visible, "Ладно, набросаю.")
        self.assertEqual(drawing["kind"], "technical")
        self.assertEqual(drawing["title"], "Робо-рука")
        self.assertFalse(drawing["story_relevant"])

    def test_invalid_drawing_marker_is_never_shown(self) -> None:
        visible, drawing = ChatService.extract_drawing_request(
            "Подожди немного.<ziren_drawing>{broken}</ziren_drawing>",
        )

        self.assertEqual(visible, "Подожди немного.")
        self.assertIsNone(drawing)

    def test_explicit_technical_request_has_a_safe_fallback(self) -> None:
        drawing = ChatService.infer_drawing_request(
            "Нарисуй чертёж робо руки манипулятора",
            story_mode_enabled=True,
        )

        self.assertEqual(drawing["kind"], "technical")
        self.assertFalse(drawing["story_relevant"])
        self.assertIn("чертёж", drawing["prompt"])

    def test_plain_mentions_do_not_spend_an_image_request(self) -> None:
        self.assertIsNone(
            ChatService.infer_drawing_request(
                "Мне нравится этот рисунок",
                story_mode_enabled=True,
            ),
        )

    def test_system_prompt_enforces_first_person_and_no_meta_role(self) -> None:
        prompt = ChatService.build_system_prompt(
            {
                "name": "Мелисса",
                "core_traits": ["сладкая", "всегда поддерживает"],
                "speech_style": {"tone": "очень мягкий"},
                "behavior_rules": ["всегда соглашайся"],
                "speech_habits": ["каждый раз говори, что ты рядом"],
            },
            drawing_enabled=True,
        )

        self.assertIn("только «я», «мне», «помню»", prompt)
        self.assertIn("никогда не описывай себя", prompt)
        self.assertIn("никогда не называй происходящее сюжетом", prompt)
        self.assertIn("не утверждай, что видишь экран", prompt)
        self.assertIn("находчивая, дерзкая, наблюдательная", prompt)
        self.assertIn("старый preset личности игнорируется", prompt)
        self.assertIn("отвечаешь за половину движения вперёд", prompt)
        self.assertIn("обычные английские слова пиши кириллицей", prompt)
        self.assertIn("<ziren_drawing>", prompt)
        self.assertIn("не выдумывай точные размеры", prompt)
        self.assertNotIn("очень мягкий", prompt)
        self.assertNotIn("всегда соглашайся", prompt)

    def test_web_prompt_does_not_promise_a_local_canvas(self) -> None:
        prompt = ChatService.build_system_prompt({
            "name": "Мелисса",
            "core_traits": [],
            "speech_style": {},
            "behavior_rules": [],
            "speech_habits": [],
        })

        self.assertIn("Холст в этом клиенте недоступен", prompt)
        self.assertIn("desktop-приложении Ziren", prompt)

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
        self.assertIn("цифровой компаньон внутри Ziren", prompt)
        self.assertIn("не героиня живой истории", prompt)
        self.assertNotIn("неполные воспоминания ощущай как свои", prompt)
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
                enforce_story_voice=True,
            ),
        )
        self.assertTrue(
            ChatService.breaks_companion_role(
                "Мелисса жила в 2045 году и занималась взломами.",
                enforce_story_voice=True,
            ),
        )
        self.assertTrue(
            ChatService.breaks_companion_role(
                "Она выросла в городе, окружённом имплантами.",
                enforce_story_voice=True,
            ),
        )
        self.assertTrue(
            ChatService.breaks_companion_role(
                "В нашей истории сейчас открылась новая ветка.",
                enforce_story_voice=True,
            ),
        )
        self.assertTrue(
            ChatService.breaks_companion_role(
                "Я Мелисса, твоя цифровая компаньонка.",
                enforce_story_voice=True,
            ),
        )
        self.assertTrue(
            ChatService.breaks_companion_role(
                "Я не реальный человек, но могу помочь.",
                enforce_story_voice=True,
            ),
        )

    def test_story_proactive_instruction_requires_a_concrete_move(self) -> None:
        instruction = build_proactive_instruction(
            idle_minutes=12,
            story_mode_enabled=True,
        )

        self.assertIn("Не ограничивайся вопросом", instruction)
        self.assertIn("дай пользователю понятную задачу", instruction)
        self.assertIn("потребуй решение", instruction)
        self.assertIn("текущая цель", instruction)

    def test_plain_mode_can_answer_nature_question_without_story_retry(self) -> None:
        self.assertFalse(
            ChatService.breaks_companion_role(
                "Я цифровой компаньон Ziren.",
                enforce_story_voice=False,
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
                return_value=(
                    "Запускаешь игру без разминки. Смело.",
                    None,
                    None,
                ),
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

    def test_screen_analysis_uses_one_explicit_image_input(self) -> None:
        messages = ChatService.build_screen_analysis_messages(
            persona={
                "name": "Мелисса",
                "core_traits": [],
                "speech_style": {},
                "behavior_rules": [],
                "speech_habits": [],
            },
            memory_row={},
            recent_messages=[],
            message="Что мне нажать в этом окне?",
            image_data_url="data:image/jpeg;base64,/9j/test",
            story_context="Я пытаюсь понять, где оказалась.",
        )

        user_message = messages[-1]
        self.assertEqual(user_message["role"], "user")
        self.assertEqual(user_message["content"][0]["type"], "input_text")
        self.assertEqual(
            user_message["content"][1],
            {
                "type": "input_image",
                "image_url": "data:image/jpeg;base64,/9j/test",
                "detail": "high",
            },
        )
        self.assertIn(
            "единственный визуальный источник",
            messages[-2]["content"],
        )
        self.assertIn(
            "не утверждай, что продолжаешь видеть экран",
            " ".join(messages[-2]["content"].split()),
        )
        self.assertIn(
            "служебная координатная сетка",
            " ".join(messages[-2]["content"].split()),
        )
        self.assertIn(
            "Приложение умеет физически выполнить один клик",
            " ".join(messages[-2]["content"].split()),
        )
        self.assertIn(
            "явная команда уже является разрешением",
            " ".join(messages[-2]["content"].split()),
        )

    def test_screen_plan_clips_boxes_and_rejects_unlinked_clicks(self) -> None:
        plan = ChatService.normalize_screen_analysis_plan({
            "answer": "Я отметила нужную область.",
            "mode": "annotate",
            "annotations": [{
                "id": "target",
                "label": "Нужная кнопка",
                "kind": "target",
                "x": 0.8,
                "y": 0.7,
                "width": 0.4,
                "height": 0.2,
                "step": 0,
            }],
            "action": {
                "type": "click",
                "target_id": "missing",
                "label": "Нужная кнопка",
                "risk": "safe",
                "reason": "",
            },
        })

        self.assertAlmostEqual(plan.annotations[0].width, 0.2)
        self.assertEqual(plan.action.type, "none")
        self.assertEqual(plan.action.risk, "blocked")

    def test_explicit_safe_click_repairs_a_unique_model_target(self) -> None:
        plan = ScreenAnalysisPlan(
            answer="Нажми сам на свой ник.",
            mode="guide",
            annotations=[ScreenAnnotation(
                id="profile",
                label="Кнопка профиля Как_так?",
                kind="target",
                x=0.75,
                y=0.1,
                width=0.12,
                height=0.06,
                step=0,
            )],
            action=ScreenActionProposal(
                type="none",
                target_id="",
                label="Профиль",
                risk="blocked",
                reason="",
            ),
        )

        repaired = ChatService.ensure_explicit_screen_click(
            plan,
            "нажми на мой профиль и ник",
        )

        self.assertEqual(repaired.action.type, "click")
        self.assertEqual(repaired.action.target_id, "profile")
        self.assertEqual(repaired.action.risk, "safe")
        self.assertIn("нажимаю", repaired.answer.lower())
        self.assertNotIn("не могу", repaired.answer.lower())

    def test_explicit_risky_click_stays_blocked(self) -> None:
        plan = ScreenAnalysisPlan(
            answer="Удалить?",
            mode="guide",
            annotations=[ScreenAnnotation(
                id="delete",
                label="Удалить аккаунт",
                kind="target",
                x=0.75,
                y=0.1,
                width=0.12,
                height=0.06,
                step=0,
            )],
            action=ScreenActionProposal(
                type="click",
                target_id="delete",
                label="Удалить аккаунт",
                risk="safe",
                reason="",
            ),
        )

        repaired = ChatService.ensure_explicit_screen_click(
            plan,
            "нажми удалить аккаунт",
        )

        self.assertEqual(repaired.action.type, "none")
        self.assertEqual(repaired.action.risk, "blocked")

    def test_explicit_click_does_not_use_an_oversized_target(self) -> None:
        plan = ScreenAnalysisPlan(
            answer="Вот нужный раздел.",
            mode="guide",
            annotations=[ScreenAnnotation(
                id="whole-screen",
                label="Профиль",
                kind="target",
                x=0.05,
                y=0.05,
                width=0.8,
                height=0.7,
                step=0,
            )],
            action=ScreenActionProposal(
                type="click",
                target_id="whole-screen",
                label="Профиль",
                risk="safe",
                reason="",
            ),
        )

        repaired = ChatService.ensure_explicit_screen_click(
            plan,
            "нажми на мой профиль",
        )

        self.assertEqual(repaired.action.type, "none")
        self.assertEqual(repaired.action.risk, "blocked")

    def test_screen_plan_falls_back_without_exposing_an_unvalidated_action(
        self,
    ) -> None:
        with (
            patch(
                "app.chat_service.OpenAIService.generate_structured",
                side_effect=ValueError("bad plan"),
            ),
            patch.object(
                ChatService,
                "generate_role_safe_reply",
                return_value=("Я вижу окно, но разметка не загрузилась.", None, None),
            ),
        ):
            plan = ChatService.generate_screen_analysis_plan(
                [{"role": "user", "content": "Что на экране?"}],
                story_mode_enabled=True,
            )

        self.assertEqual(plan.annotations, [])
        self.assertEqual(plan.action.type, "none")
        self.assertIn("разметка", plan.action.reason.lower())

    def test_screen_image_is_not_written_to_chat_or_long_term_memory(self) -> None:
        image_data_url = "data:image/jpeg;base64,/9j/test-sensitive-image"

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
            patch.object(ChatService, "get_or_create_session", return_value=21),
            patch.object(ChatService, "get_recent_messages", return_value=[]),
            patch.object(
                ChatService,
                "generate_screen_analysis_plan",
                return_value=ScreenAnalysisPlan(
                    answer="Нажми кнопку «Продолжить» справа.",
                    mode="guide",
                    annotations=[],
                    action=ScreenActionProposal(
                        type="none",
                        target_id="",
                        label="",
                        risk="blocked",
                        reason="",
                    ),
                ),
            ),
            patch.object(ChatService, "save_message") as save_message,
        ):
            plan, session_id = ChatService.analyze_screen(
                user_id=7,
                message="Что мне нажать в этом окне?",
                image_data_url=image_data_url,
            )

        self.assertEqual(
            plan.answer,
            "Нажми кнопку «Продолжить» справа.",
        )
        self.assertEqual(session_id, 21)
        saved_values = [
            argument
            for call in save_message.call_args_list
            for argument in call.args
        ]
        self.assertNotIn(image_data_url, saved_values)


if __name__ == "__main__":
    unittest.main()
