import unittest

from pydantic import ValidationError

from app.schemas import (
    AppLauncherCandidate,
    AppLauncherResolveRequest,
    ChatRequest,
    PersonaNameRequest,
    PersonaPresetRequest,
)


class SchemaLimitTests(unittest.TestCase):
    def test_app_launcher_accepts_at_most_forty_candidates(self) -> None:
        candidates = [
            AppLauncherCandidate(index=index, name=f"App {index}")
            for index in range(40)
        ]
        payload = AppLauncherResolveRequest(query="app", candidates=candidates)

        self.assertEqual(len(payload.candidates), 40)

        with self.assertRaises(ValidationError):
            AppLauncherResolveRequest(
                query="app",
                candidates=candidates + [AppLauncherCandidate(index=40, name="Overflow")],
            )

    def test_app_launcher_accepts_at_most_ten_aliases(self) -> None:
        aliases = [f"alias-{index}" for index in range(10)]
        candidate = AppLauncherCandidate(index=0, name="App", aliases=aliases)

        self.assertEqual(candidate.aliases, aliases)

        with self.assertRaises(ValidationError):
            AppLauncherCandidate(index=0, name="App", aliases=aliases + ["overflow"])

    def test_persona_text_fields_are_bounded(self) -> None:
        self.assertEqual(PersonaNameRequest(name="Melissa").name, "Melissa")
        self.assertEqual(PersonaPresetRequest(preset_name="default").preset_name, "default")

        for payload_type, field_name in (
            (PersonaNameRequest, "name"),
            (PersonaPresetRequest, "preset_name"),
        ):
            with self.subTest(payload_type=payload_type.__name__):
                with self.assertRaises(ValidationError):
                    payload_type(**{field_name: "x" * 101})

    def test_story_context_is_optional_and_bounded(self) -> None:
        payload = ChatRequest(
            user_id=1,
            message="Привет",
            story_context="сюжет",
        )

        self.assertEqual(payload.story_context, "сюжет")

        with self.assertRaises(ValidationError):
            ChatRequest(
                user_id=1,
                message="Привет",
                story_context="x" * 6001,
            )


if __name__ == "__main__":
    unittest.main()
