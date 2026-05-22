import json
import re
from typing import Any

from openai import OpenAI

from app.config import settings
from app.schemas import (
    AppLauncherCandidate,
    AppLauncherResolveRequest,
    AppLauncherResolveResponse,
)


app_launcher_client = OpenAI(
    api_key=settings.OPENAI_API_KEY,
    timeout=8.0,
)


class AppLauncherService:
    MIN_CONFIDENCE = 0.82

    @staticmethod
    def resolve(payload: AppLauncherResolveRequest) -> AppLauncherResolveResponse:
        print("[APP_LAUNCHER_RESOLVE] query", payload.query)
        print("[APP_LAUNCHER_RESOLVE] candidates_count", len(payload.candidates))

        if not payload.candidates:
            return AppLauncherResolveResponse(
                selected_index=None,
                confidence=0.0,
                spoken_name=None,
                reason="No candidates",
            )

        valid_indexes = {candidate.index for candidate in payload.candidates}

        try:
            result = AppLauncherService._call_openai(payload)
            selected_index = result.get("selected_index")
            confidence = float(result.get("confidence", 0.0) or 0.0)
            spoken_name = result.get("spoken_name")
            reason = str(result.get("reason", ""))

            if spoken_name is not None:
                spoken_name = str(spoken_name).strip() or None

            if (
                selected_index is None
                or int(selected_index) not in valid_indexes
                or confidence < AppLauncherService.MIN_CONFIDENCE
            ):
                print("[APP_LAUNCHER_RESOLVE] fallback", reason)
                return AppLauncherResolveResponse(
                    selected_index=None,
                    confidence=max(0.0, min(1.0, confidence)),
                    spoken_name=None,
                    reason=reason or "Недостаточно уверенности",
                )

            selected_index = int(selected_index)
            print(
                "[APP_LAUNCHER_RESOLVE] selected",
                selected_index,
                confidence,
                spoken_name,
            )
            return AppLauncherResolveResponse(
                selected_index=selected_index,
                confidence=max(0.0, min(1.0, confidence)),
                spoken_name=spoken_name,
                reason=reason,
            )
        except Exception as error:
            reason = f"AI error: {str(error)[:220]}"
            print("[APP_LAUNCHER_RESOLVE] fallback", reason)
            return AppLauncherResolveResponse(
                selected_index=None,
                confidence=0.0,
                spoken_name=None,
                reason=reason,
            )

    @staticmethod
    def _call_openai(payload: AppLauncherResolveRequest) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": AppLauncherService._system_prompt()},
            {"role": "user", "content": AppLauncherService._user_prompt(payload)},
        ]

        try:
            response = app_launcher_client.responses.create(
                model=settings.MODEL,
                input=messages,
                response_format={"type": "json_object"},
            )
        except Exception as error:
            if "response_format" not in str(error):
                raise

            response = app_launcher_client.responses.create(
                model=settings.MODEL,
                input=messages,
            )

        text = getattr(response, "output_text", None) or ""

        if not text:
            parts = []
            try:
                for item in response.output:
                    if getattr(item, "type", None) == "message":
                        for content in getattr(item, "content", []):
                            if getattr(content, "type", None) == "output_text":
                                parts.append(content.text)
            except Exception:
                pass
            text = "\n".join(parts).strip()

        return AppLauncherService._extract_json(text)

    @staticmethod
    def _system_prompt() -> str:
        return """Ты — semantic matcher для локального запуска приложений.
Пользователь говорит по-русски.
Он может произносить английские названия русскими буквами.
Твоя задача — выбрать, какое приложение из списка candidates он имел в виду.

Примеры:
- "пабг", "пабг батл граундс", "пабы к батл граунд с" -> PUBG: BATTLEGROUNDS
- "дед целс", "ддт цел с", "дед селс" -> Dead Cells
- "евро трак симулятор" -> Euro Truck Simulator 2
- "нид фор спид", "нфс" -> Need for Speed
- "волпейпер", "обои" -> Wallpaper Engine
- "раст" -> Rust
- "дискорд" -> Discord
- "телега" -> Telegram
- "кска", "контра" -> Counter-Strike 2
- "бесконечное лето" -> Everlasting Summer

Очень важно:
- выбирай только из candidates
- не придумывай приложения
- не придумывай путь
- не придумывай exe
- если не уверен — selected_index null
- если несколько вариантов одинаково подходят — selected_index null
- spoken_name нужен для русского TTS

Ответ строго JSON без markdown:
{
  "selected_index": 13,
  "confidence": 0.96,
  "spoken_name": "Пабг Батлграундс",
  "reason": "..."
}"""

    @staticmethod
    def _user_prompt(payload: AppLauncherResolveRequest) -> str:
        candidates = [
            AppLauncherService._candidate_to_prompt(candidate)
            for candidate in payload.candidates
        ]
        return json.dumps(
            {
                "query": payload.query,
                "candidates": candidates,
            },
            ensure_ascii=False,
        )

    @staticmethod
    def _candidate_to_prompt(candidate: AppLauncherCandidate) -> dict[str, Any]:
        return {
            "index": candidate.index,
            "name": candidate.name,
            "type": candidate.type,
            "source": candidate.source,
            "aliases": candidate.aliases[:10],
            "appid": candidate.appid or "",
            "path_basename": candidate.path_basename,
        }

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        try:
            data = json.loads(text)
            return data if isinstance(data, dict) else {}
        except Exception:
            pass

        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {}

        try:
            data = json.loads(match.group(0))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}
