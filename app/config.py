import os
from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return max(minimum, int(raw.strip()))
    except (TypeError, ValueError):
        return default


class Settings:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")
    AI_INTERNAL_TOKEN: str = os.getenv("AI_INTERNAL_TOKEN", "").strip()

    # Основная модель для живого диалога Мелиссы.
    MODEL: str = os.getenv("MODEL", "gpt-4.1-mini")

    # Дешёвые служебные AI-задачи не должны сжигать бюджет основной модели.
    COMMAND_ROUTE_MODEL: str = os.getenv("COMMAND_ROUTE_MODEL", "gpt-5-nano")
    MEMORY_MODEL: str = os.getenv("MEMORY_MODEL", "gpt-5-nano")

    # Пока платёжный checkout не подключён, staging/production можно держать
    # в beta-режиме. При true Free полностью теряет доступ к облачной Мелиссе,
    # но локальная Змея продолжает работать без AI-service.
    SUBSCRIPTIONS_ENFORCED: bool = _env_bool("SUBSCRIPTIONS_ENFORCED", False)

    # Beta must not mean unlimited provider spend. These are provider-cost
    # safety ceilings in micro-USD (1_000_000 = $1), not user-visible prices.
    BETA_AI_MONTHLY_SAFETY_BUDGET_MICROUSD: int = _env_int(
        "BETA_AI_MONTHLY_SAFETY_BUDGET_MICROUSD",
        2_000_000,
    )
    BETA_AI_DAILY_SAFETY_BUDGET_MICROUSD: int = _env_int(
        "BETA_AI_DAILY_SAFETY_BUDGET_MICROUSD",
        400_000,
    )
    AI_REQUEST_TEXT_CHAR_LIMIT: int = _env_int(
        "AI_REQUEST_TEXT_CHAR_LIMIT",
        120_000,
        minimum=8_000,
    )
    AI_RESERVATION_TTL_SECONDS: int = _env_int(
        "AI_RESERVATION_TTL_SECONDS",
        120,
        minimum=30,
    )

    APP_NAME: str = os.getenv("APP_NAME", "ziren-ai-service")


settings = Settings()
