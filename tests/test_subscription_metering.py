import unittest
from datetime import datetime, timezone
from types import SimpleNamespace

from app.subscription_service import (
    AiUsage,
    calculate_cost_microusd,
    monthly_quota_window,
    usage_from_response,
)


class SubscriptionMeteringTests(unittest.TestCase):
    def test_gpt_4_1_mini_cost_accounts_for_cached_tokens(self):
        usage = AiUsage(
            input_tokens=10_000,
            cached_input_tokens=4_000,
            output_tokens=1_000,
        )
        # 6k * $0.40/M + 4k * $0.10/M + 1k * $1.60/M = $0.0044
        self.assertEqual(calculate_cost_microusd("gpt-4.1-mini", usage), 4_400)

    def test_gpt_5_nano_is_metered_with_its_own_prices(self):
        usage = AiUsage(input_tokens=10_000, output_tokens=1_000)
        # 10k * $0.05/M + 1k * $0.40/M = $0.0009
        self.assertEqual(calculate_cost_microusd("gpt-5-nano", usage), 900)

    def test_usage_is_read_from_responses_api_shape(self):
        response = SimpleNamespace(
            usage=SimpleNamespace(
                input_tokens=1234,
                output_tokens=321,
                input_tokens_details=SimpleNamespace(cached_tokens=512),
            )
        )
        usage = usage_from_response(response)
        self.assertEqual(usage.input_tokens, 1234)
        self.assertEqual(usage.cached_input_tokens, 512)
        self.assertEqual(usage.output_tokens, 321)

    def test_unknown_model_never_gets_invented_price(self):
        with self.assertRaises(ValueError):
            calculate_cost_microusd("future-unknown-model", AiUsage(input_tokens=1))

    def test_ai_quota_refreshes_by_calendar_month_even_for_annual_billing(self):
        start, end = monthly_quota_window(
            datetime(2026, 8, 17, 23, 59, tzinfo=timezone.utc)
        )
        self.assertEqual(start, datetime(2026, 8, 1, tzinfo=timezone.utc))
        self.assertEqual(end, datetime(2026, 9, 1, tzinfo=timezone.utc))

        start, end = monthly_quota_window(
            datetime(2026, 12, 31, 12, 0, tzinfo=timezone.utc)
        )
        self.assertEqual(start, datetime(2026, 12, 1, tzinfo=timezone.utc))
        self.assertEqual(end, datetime(2027, 1, 1, tzinfo=timezone.utc))


if __name__ == "__main__":
    unittest.main()
