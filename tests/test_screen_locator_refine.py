import unittest

from app.schemas import ScreenAnnotation


class ScreenLocatorRefineTests(unittest.TestCase):
    def test_tight_target_receives_high_confidence(self) -> None:
        annotation = ScreenAnnotation.model_validate({
            "id": "assistant-tab",
            "label": "Кнопка Ассистент",
            "kind": "target",
            "x": 0.6,
            "y": 0.1,
            "width": 0.09,
            "height": 0.05,
            "step": 0,
        })

        self.assertEqual(annotation.confidence, 0.92)

    def test_broad_target_is_marked_for_locator_retry(self) -> None:
        annotation = ScreenAnnotation.model_validate({
            "id": "assistant-section",
            "label": "Раздел Ассистент",
            "kind": "target",
            "x": 0.4,
            "y": 0.1,
            "width": 0.5,
            "height": 0.3,
            "step": 0,
        })

        self.assertLess(annotation.confidence, 0.78)

    def test_text_annotation_does_not_trigger_target_retry(self) -> None:
        annotation = ScreenAnnotation.model_validate({
            "id": "caption",
            "label": "Текст подсказки",
            "kind": "text",
            "x": 0.1,
            "y": 0.1,
            "width": 0.6,
            "height": 0.4,
            "step": 0,
        })

        self.assertEqual(annotation.confidence, 1.0)

    def test_explicit_confidence_is_preserved(self) -> None:
        annotation = ScreenAnnotation.model_validate({
            "id": "profile",
            "label": "Профиль",
            "kind": "target",
            "x": 0.8,
            "y": 0.1,
            "width": 0.08,
            "height": 0.05,
            "step": 0,
            "confidence": 0.81,
        })

        self.assertEqual(annotation.confidence, 0.81)


if __name__ == "__main__":
    unittest.main()
