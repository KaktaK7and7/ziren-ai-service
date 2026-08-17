from __future__ import annotations

from openai import OpenAI

from app.config import settings
from app.subscription_service import SubscriptionService, usage_from_response
from app.usage_context import get_ai_usage_context


client = OpenAI(
    api_key=settings.OPENAI_API_KEY,
    timeout=30.0,
)


def _response_text(response) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return str(text).strip()

    parts: list[str] = []
    try:
        for item in response.output:
            if getattr(item, "type", None) == "message":
                for content in getattr(item, "content", []):
                    if getattr(content, "type", None) == "output_text":
                        parts.append(content.text)
    except Exception:
        pass
    return "\n".join(parts).strip()


def _record_usage(response, model: str, suffix: str) -> None:
    user_id, base_operation = get_ai_usage_context()
    if user_id is None:
        return

    usage = usage_from_response(response)
    try:
        cost = SubscriptionService.record_usage(
            user_id=user_id,
            operation=f"{base_operation}.{suffix}",
            model=model,
            usage=usage,
        )
        print(
            "[AI_USAGE]",
            f"user={user_id}",
            f"operation={base_operation}.{suffix}",
            f"model={model}",
            f"input={usage.input_tokens}",
            f"cached={usage.cached_input_tokens}",
            f"output={usage.output_tokens}",
            f"cost_microusd={cost}",
        )
    except Exception as error:
        # The model request has already been paid for. Do not discard a valid
        # user response solely because telemetry persistence failed; surface a
        # loud server log so operations can reconcile it.
        print("[AI_USAGE][ERROR] metering failed:", error)


class OpenAIService:
    @staticmethod
    def generate_json(model: str, messages: list[dict]) -> dict:
        print("[OpenAI][JSON] sending request...")
        print(f"[OpenAI][JSON] model={model}")
        print(f"[OpenAI][JSON] messages_count={len(messages)}")

        response = client.responses.create(
            model=model,
            input=messages,
        )
        _record_usage(response, model, "json")
        print("[OpenAI][JSON] response received")

        text = _response_text(response)
        try:
            import json
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else {}
        except Exception as error:
            print("[OpenAI][JSON] parse error:", error)
            return {"should_save": False, "items": []}

    @staticmethod
    def generate_reply(model: str, messages: list[dict]) -> str:
        print("[OpenAI] sending request...")
        print(f"[OpenAI] model={model}")
        print(f"[OpenAI] messages_count={len(messages)}")

        response = client.responses.create(
            model=model,
            input=messages,
        )
        _record_usage(response, model, "reply")
        print("[OpenAI] response received")
        return _response_text(response)
