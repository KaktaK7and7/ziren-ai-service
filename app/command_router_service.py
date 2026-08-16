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


def _reason(prefix: str, value: object, fallback: str) -> str:
    text = " ".join(str(value or fallback).split())[:280]
    return f"{prefix}: {text}"[:300]


class CommandRouterService:
    @staticmethod
    def resolve(payload: CommandRouteRequest) -> CommandRouteResponse:
        catalog = CommandRouterService._sanitize_catalog(payload.capabilities)
        if not catalog:
            return CommandRouteResponse(
                matched=False,
                command_like=False,
                reason="system: empty capability catalog",
            )

        system_prompt = (
            "Ты — только классификатор локальных команд Windows и функций Ziren. "
            "Никогда не отвечай пользователю разговорным текстом и никогда не утверждай, "
            "что действие выполнено. Сначала определи command_like: true, если пользователь "
            "просит управлять компьютером, приложением, окном, файлами, мультимедиа, расписанием "
            "или другой функцией Ziren; false, если это обычный разговор, вопрос, мнение или просьба "
            "объяснить что-либо без выполнения действия. Если command_like=false, matched=false. "
            "Если command_like=true, выбирай действие ТОЛЬКО из переданного каталога. "
            "Если подходящего действия нет или ты не уверена — matched=false, но command_like=true. "
            "Не придумывай feature_id/action_id. Не генерируй shell, PowerShell, CMD, пути или "
            "произвольные сочетания клавиш. Аргументы извлекай только из текста пользователя. "
            "Для ввода текста используй arguments.text; для окон и приложений arguments.target; "
            "для процентов arguments.percent; для номера монитора arguments.monitor. "
            "Верни только JSON: {command_like:boolean, matched:boolean, feature_id:string, "
            "action_id:string, arguments:object, confidence:number, reason:string}. "
            "Confidence ниже 0.78 означает matched=false."
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
        if not isinstance(raw, dict):
            return CommandRouteResponse(
                matched=False,
                command_like=False,
                reason="system: invalid classifier response",
            )

        command_like = raw.get("command_like") is True
        if not command_like:
            return CommandRouteResponse(
                matched=False,
                command_like=False,
                reason=_reason("chat", raw.get("reason"), "ordinary conversation"),
            )

        feature_id = str(raw.get("feature_id") or "").strip()
        action_id = str(raw.get("action_id") or "").strip()
        try:
            confidence = float(raw.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        if raw.get("matched") is not True:
            return CommandRouteResponse(
                matched=False,
                command_like=True,
                confidence=confidence,
                reason=_reason(
                    "command",
                    raw.get("reason"),
                    "no safe capability match",
                ),
            )

        allowed_pairs = {
            (feature["feature_id"], action["action_id"])
            for feature in catalog
            for action in feature["actions"]
        }
        if confidence < 0.78 or (feature_id, action_id) not in allowed_pairs:
            return CommandRouteResponse(
                matched=False,
                command_like=True,
                confidence=confidence,
                reason="command: low confidence or action is outside capability catalog",
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
            command_like=True,
            feature_id=feature_id,
            action_id=action_id,
            arguments=arguments,
            confidence=confidence,
            reason=_reason("command", raw.get("reason"), "matched safe capability"),
        )
