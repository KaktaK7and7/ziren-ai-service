from __future__ import annotations

from openai import OpenAI

from app.config import settings
from app.subscription_service import (
    AiBudgetReservation,
    SubscriptionService,
    usage_from_response,
)
from app.usage_context import get_ai_usage_context


client = OpenAI(
    api_key=settings.OPENAI_API_KEY,
    timeout=30.0,
)

JSON_MAX_OUTPUT_TOKENS = 1200
REPLY_MAX_OUTPUT_TOKENS = 1000


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


def _reserve_usage(
    model: str,
    messages: list[dict],
    suffix: str,
    max_output_tokens: int,
) -> tuple[int, str, AiBudgetReservation] | None:
    user_id, base_operation = get_ai_usage_context()
    if user_id is None:
        return None
    operation = f"{base_operation}.{suffix}"
    reservation = SubscriptionService.reserve_request(
        user_id=user_id,
        operation=operation,
        model=model,
        messages=messages,
        max_output_tokens=max_output_tokens,
    )
    print(
        "[AI_BUDGET][RESERVED]",
        f"user={user_id}",
        f"operation={operation}",
        f"model={model}",
        f"reserved_microusd={reservation.reserved_microusd}",
    )
    return user_id, operation, reservation


def _release_usage(reserved: tuple[int, str, AiBudgetReservation] | None) -> None:
    if reserved is None:
        return
    user_id, _operation, reservation = reserved
    try:
        SubscriptionService.release_reservation(
            user_id=user_id,
            reservation_id=reservation.reservation_id,
        )
    except Exception as error:
        # Expired reservations are removed automatically before the next budget
        # reservation, so a failed cleanup cannot create unlimited spend.
        print("[AI_BUDGET][ERROR] reservation cleanup failed:", error)


def _record_usage(
    response,
    model: str,
    suffix: str,
    reserved: tuple[int, str, AiBudgetReservation] | None,
) -> None:
    user_id, base_operation = get_ai_usage_context()
    if user_id is None:
        return

    operation = f"{base_operation}.{suffix}"
    usage = usage_from_response(response)
    try:
        if reserved is not None:
            reserved_user_id, reserved_operation, reservation = reserved
            cost = SubscriptionService.finalize_usage(
                user_id=reserved_user_id,
                reservation_id=reservation.reservation_id,
                operation=reserved_operation,
                model=model,
                usage=usage,
            )
        else:
            cost = SubscriptionService.record_usage(
                user_id=user_id,
                operation=operation,
                model=model,
                usage=usage,
            )
        print(
            "[AI_USAGE]",
            f"user={user_id}",
            f"operation={operation}",
            f"model={model}",
            f"input={usage.input_tokens}",
            f"cached={usage.cached_input_tokens}",
            f"output={usage.output_tokens}",
            f"cost_microusd={cost}",
        )
    except Exception as error:
        # The provider request has already happened. Keep the reservation until
        # its TTL if atomic finalization failed; that fails closed for budget
        # purposes instead of giving the next request free capacity.
        print("[AI_USAGE][ERROR] metering finalization failed:", error)


class OpenAIService:
    @staticmethod
    def generate_json(model: str, messages: list[dict]) -> dict:
        print("[OpenAI][JSON] sending request...")
        print(f"[OpenAI][JSON] model={model}")
        print(f"[OpenAI][JSON] messages_count={len(messages)}")

        reserved = _reserve_usage(
            model,
            messages,
            "json",
            JSON_MAX_OUTPUT_TOKENS,
        )
        try:
            response = client.responses.create(
                model=model,
                input=messages,
                max_output_tokens=JSON_MAX_OUTPUT_TOKENS,
            )
        except Exception:
            _release_usage(reserved)
            raise

        _record_usage(response, model, "json", reserved)
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

        reserved = _reserve_usage(
            model,
            messages,
            "reply",
            REPLY_MAX_OUTPUT_TOKENS,
        )
        try:
            response = client.responses.create(
                model=model,
                input=messages,
                max_output_tokens=REPLY_MAX_OUTPUT_TOKENS,
            )
        except Exception:
            _release_usage(reserved)
            raise

        _record_usage(response, model, "reply", reserved)
        print("[OpenAI] response received")
        return _response_text(response)
