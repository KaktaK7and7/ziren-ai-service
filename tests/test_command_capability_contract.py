import unittest

from app.command_router_service import (
    MAX_ACTIONS_PER_FEATURE,
    CommandRouterService,
)


class CommandCapabilityContractTests(unittest.TestCase):
    def test_classifier_keeps_bounded_action_catalog_and_voice_examples(self):
        actions = [
            {
                "action_id": f"feature.action_{index}",
                "display_name": f"Action {index}",
                "argument_hint": "arguments.target — test",
                "voice_examples": [
                    f"пример {index}",
                    f"воск вариант {index}",
                ],
            }
            for index in range(MAX_ACTIONS_PER_FEATURE + 1)
        ]
        catalog = CommandRouterService._sanitize_catalog(
            [{"feature_id": "system.test_contract", "actions": actions}]
        )

        self.assertEqual(len(catalog), 1)
        self.assertEqual(len(catalog[0]["actions"]), MAX_ACTIONS_PER_FEATURE)
        first = catalog[0]["actions"][0]
        self.assertEqual(first["voice_examples"], ["пример 0", "воск вариант 0"])
        self.assertEqual(first["argument_hint"], "arguments.target — test")

    def test_action_beyond_contract_limit_never_becomes_selectable(self):
        actions = [
            {
                "action_id": f"feature.action_{index}",
                "display_name": f"Action {index}",
            }
            for index in range(MAX_ACTIONS_PER_FEATURE + 1)
        ]
        catalog = CommandRouterService._sanitize_catalog(
            [{"feature_id": "system.test_contract", "actions": actions}]
        )
        ids = {action["action_id"] for action in catalog[0]["actions"]}

        self.assertIn(f"feature.action_{MAX_ACTIONS_PER_FEATURE - 1}", ids)
        self.assertNotIn(f"feature.action_{MAX_ACTIONS_PER_FEATURE}", ids)


if __name__ == "__main__":
    unittest.main()
