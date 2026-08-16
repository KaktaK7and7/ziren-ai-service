from __future__ import annotations

import json
from typing import Any

from app.config import settings
from app.openai_service import OpenAIService
from app.schemas import CommandRouteRequest, CommandRouteResponse


MAX_CAPABILITIES = 80
MAX_ACTIONS_PER_FEATURE = 40
ALLOWED_ARGUMENT_KEYS = {
    "text",
    "target",
    "folder_id",
    "minutes",
    "hours",
    "time",
    "label",
    "percent",
    "monitor",
    "direction",
    "recipient",
    "message",
}


class CommandRouterService:
    @staticmethod
    def resolve(payload: CommandRouteRequest) -> CommandRouteResponse:
        catalog = CommandRouterService._sanitize_catalog(payload.capabilities)
        if not catalog:
            return CommandRouteResponse(matched=False, reason="empty capability catalog")

        system_prompt = (
            "Ты — только классификатор локальных команд Windows для Ziren. "
            "Никогда не отвечай пользователю разговорным текстом и никогда не утверждай, "
            "что действие выполнено. Выбирай действие ТОЛЬКО из переданного каталога. "
            "Если сообщение является обычным разговором, вопросом или ты не уверена — matched=false. "
            "Не придумывай feature_id/action_id. Не генерируй shell, PowerShell, CMD, пути или сочетания клавиш. "
            "Аргументы извлекай только из текста пользователя. Для ввода текста используй arguments.text; "
            "для окон arguments.target; для процентов arguments.percent. "
            "Верни только JSON: {matched:boolean, feature_id:string, action_id:string, arguments:object, "
            "confidence:number, reason:string}. Confidence ниже 0.78 означает matched=false."
        )
        user_payload = {
            "message": payload.message,
            "capabilities": catalog,
        }
        raw = OpenAIService.generate_json(
            settings.MODEL,
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
        )
        return CommandRouterService._validate_result(raw, catalog)

    @staticmethod
    def _sanitize_catalog(capabilities: list[dict[str, Any]]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for feature in capabilities[:MAX_CAPABILITIES]:
            if not isinstance(feature, dict):
                continue
            feature_id = str(feature.get("feature_id") or "").strip()[:100]
            if not feature_id:
                continue
            actions: list[dict[str, str]] = []
            for action in feature.get("actions", [])[:MAX_ACTIONS_PER_FEATURE]:
                if isinstance(action, dict):
                    action_id = str(action.get("action_id") or "").strip()[:120]
                    label = str(action.get("display_name") or action_id).strip()[:160]
                    argument_hint = str(action.get("argument_hint") or "").strip()[:160]
                else:
                    continue
                if action_id:
                    actions.append({
                        "action_id": action_id,
                        "display_name": label,
                        "argument_hint": argument_hint,
                    })
            if actions:
                result.append({"feature_id": feature_id, "actions": actions})
        return result

    @staticmethod
    def _validate_result(raw: object, catalog: list[dict[str, Any]]) -> CommandRouteResponse:
        if not isinstance(raw, dict) or raw.get("matched") is not True:
            return CommandRouteResponse(matched=False, reason="no semantic command match")

        feature_id = str(raw.get("feature_id") or "").strip()
        action_id = str(raw.get("action_id") or "").strip()
        try:
            confidence = float(raw.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        allowed_pairs = {
            (feature["feature_id"], action["action_id"])
            for feature in catalog
            for action in feature["actions"]
        }
        if confidence < 0.78 or (feature_id, action_id) not in allowed_pairs:
            return CommandRouteResponse(
                matched=False,
                confidence=confidence,
                reason="low confidence or action is outside capability catalog",
            )

        raw_arguments = raw.get("arguments")
        arguments: dict[str, Any] = {}
        if isinstance(raw_arguments, dict):
            for key, value in raw_arguments.items():
                if key not in ALLOWED_ARGUMENT_KEYS:
                    continue
                if isinstance(value, (str, int, float, bool)):
                    arguments[key] = value

        return CommandRouteResponse(
            matched=True,
            feature_id=feature_id,
            action_id=action_id,
            arguments=arguments,
            confidence=confidence,
            reason=str(raw.get("reason") or "")[:300],
        )
