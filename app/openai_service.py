from openai import OpenAI

from app.config import settings

client = OpenAI(
    api_key=settings.OPENAI_API_KEY,
    timeout=30.0,
)


class OpenAIService:
    @staticmethod
    def generate_json(model: str, messages: list[dict]) -> dict:
        print("[OpenAI][JSON] sending request...")
        print(f"[OpenAI][JSON] model={model}")
        print(f"[OpenAI][JSON] messages_count={len(messages)}")

        response = client.responses.create(
            model=model,
            input=messages,
            response_format={"type": "json_object"},
        )

        print("[OpenAI][JSON] response received")

        text = getattr(response, "output_text", None)

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

        try:
            import json
            return json.loads(text)            
        except Exception as e:
            print("[OpenAI][JSON] parse error:", e)
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

        print("[OpenAI] response received")

        text = getattr(response, "output_text", None)
        if text:
            return text.strip()

        parts = []
        try:
            for item in response.output:
                if getattr(item, "type", None) == "message":
                    for content in getattr(item, "content", []):
                        if getattr(content, "type", None) == "output_text":
                            parts.append(content.text)
        except Exception:
            pass

        return "\n".join(parts).strip()
    
    