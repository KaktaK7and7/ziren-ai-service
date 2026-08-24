from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from app.config import settings
from app.db import db_cursor


PLAN_CATALOG = {
    "free": {
        "rank": 0,
        "ai_budget_microusd": 0,
    },
    "plus": {
        "rank": 1,
        "ai_budget_microusd": 1_500_000,
    },
    "pro": {
        "rank": 2,
        "ai_budget_microusd": 4_000_000,
    },
}

# Provider prices per one million text tokens, in USD.
# Unknown models are intentionally rejected: silently guessing a future model
# price is how subscription economics drift into the red.
MODEL_PRICING_USD_PER_MILLION = {
    "gpt-4.1-mini": {
        "input": 0.40,
        "cached_input": 0.10,
        "output": 1.60,
    },
    "gpt-5-nano": {
        "input": 0.05,
        "cached_input": 0.005,
        "output": 0.40,
    },
}

_MEDIA_KEYS = {
    "image",
    "image_url",
    "input_image",
    "audio",
    "input_audio",
    "file",
    "file_data",
}


class SubscriptionAccessError(RuntimeError):
    def __init__(self, code: str, message: str, status_code: int = 402):
        super().__init__(message)
        self.code = code
        self.status_code = status_code


@dataclass(frozen=True)
class AiUsage:
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class AiBudgetReservation:
    reservation_id: str
    reserved_microusd: int
    max_output_tokens: int


def _model_family(model: str) -> str | None:
    value = str(model or "").strip().lower()
    for family in MODEL_PRICING_USD_PER_MILLION:
        if value == family or value.startswith(f"{family}-"):
            return family
    return None


def calculate_cost_microusd(model: str, usage: AiUsage) -> int:
    family = _model_family(model)
    if family is None:
        raise ValueError(f"No pricing configured for model: {model}")

    price = MODEL_PRICING_USD_PER_MILLION[family]
    cached = max(0, min(usage.input_tokens, usage.cached_input_tokens))
    uncached = max(0, usage.input_tokens - cached)
    dollars = (
        uncached * price["input"]
        + cached * price["cached_input"]
        + max(0, usage.output_tokens) * price["output"]
    ) / 1_000_000
    return max(0, int(round(dollars * 1_000_000)))


def usage_from_response(response: Any) -> AiUsage:
    raw = getattr(response, "usage", None)
    if raw is None:
        return AiUsage()

    input_tokens = int(getattr(raw, "input_tokens", 0) or 0)
    output_tokens = int(getattr(raw, "output_tokens", 0) or 0)
    details = getattr(raw, "input_tokens_details", None)
    cached_tokens = int(getattr(details, "cached_tokens", 0) or 0) if details else 0
    return AiUsage(
        input_tokens=max(0, input_tokens),
        cached_input_tokens=max(0, cached_tokens),
        output_tokens=max(0, output_tokens),
    )


def monthly_quota_window(now: datetime) -> tuple[datetime, datetime]:
    """Return the UTC calendar-month window used by all paid plans."""
    current = now.astimezone(timezone.utc)
    start = datetime(current.year, current.month, 1, tzinfo=timezone.utc)
    if current.month == 12:
        end = datetime(current.year + 1, 1, 1, tzinfo=timezone.utc)
    else:
        end = datetime(current.year, current.month + 1, 1, tzinfo=timezone.utc)
    return start, end


def daily_quota_window(now: datetime) -> tuple[datetime, datetime]:
    current = now.astimezone(timezone.utc)
    start = datetime(
        current.year,
        current.month,
        current.day,
        tzinfo=timezone.utc,
    )
    return start, start + timedelta(days=1)


def usage_level(percent: int) -> str:
    value = max(0, int(percent))
    if value >= 100:
        return "exhausted"
    if value >= 90:
        return "critical"
    if value >= 70:
        return "warning"
    return "normal"


def _text_character_count(value: Any, *, parent_key: str = "") -> int:
    """Conservatively count text without treating base64/media as text tokens."""
    if parent_key.lower() in _MEDIA_KEYS:
        return 0
    if isinstance(value, str):
        return len(value)
    if isinstance(value, (list, tuple)):
        return sum(_text_character_count(item) for item in value)
    if isinstance(value, dict):
        total = 0
        for key, item in value.items():
            clean_key = str(key).lower()
            if clean_key in _MEDIA_KEYS:
                continue
            total += _text_character_count(item, parent_key=clean_key)
        return total
    return 0


def estimate_request_cost_ceiling_microusd(
    model: str,
    messages: Any,
    max_output_tokens: int,
) -> tuple[int, int]:
    """Return conservative provider-cost ceiling and counted text characters.

    We intentionally use one text character as at most one input token. This is
    pessimistic for normal Russian/English text and therefore suitable for
    reserving budget before the provider request. Actual usage replaces the
    reservation after the response arrives.
    """
    text_chars = _text_character_count(messages)
    if text_chars > settings.AI_REQUEST_TEXT_CHAR_LIMIT:
        raise SubscriptionAccessError(
            "ai_request_too_large",
            "Запрос слишком большой для безопасной обработки. Сократи контекст и попробуй ещё раз.",
            status_code=413,
        )

    estimated_usage = AiUsage(
        input_tokens=text_chars + 256,
        cached_input_tokens=0,
        output_tokens=max(1, int(max_output_tokens)),
    )
    return calculate_cost_microusd(model, estimated_usage), text_chars


class SubscriptionService:
    _schema_ready = False

    @classmethod
    def ensure_schema(cls) -> None:
        if cls._schema_ready:
            return

        with db_cursor(commit=True) as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS user_subscriptions (
                    user_id INT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
                    plan VARCHAR(16) NOT NULL DEFAULT 'free',
                    status VARCHAR(24) NOT NULL DEFAULT 'active',
                    current_period_start TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    current_period_end TIMESTAMPTZ,
                    ai_budget_override_microusd BIGINT,
                    cancel_at_period_end BOOLEAN NOT NULL DEFAULT FALSE,
                    provider VARCHAR(40),
                    provider_customer_id VARCHAR(160),
                    provider_subscription_id VARCHAR(160),
                    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    CHECK (plan IN ('free', 'plus', 'pro'))
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS ai_usage_events (
                    id BIGSERIAL PRIMARY KEY,
                    user_id INT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    operation VARCHAR(80) NOT NULL,
                    model VARCHAR(100) NOT NULL,
                    input_tokens INT NOT NULL DEFAULT 0,
                    cached_input_tokens INT NOT NULL DEFAULT 0,
                    output_tokens INT NOT NULL DEFAULT 0,
                    cost_microusd BIGINT NOT NULL DEFAULT 0,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_ai_usage_user_created
                ON ai_usage_events(user_id, created_at DESC);
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS ai_usage_reservations (
                    reservation_id VARCHAR(40) PRIMARY KEY,
                    user_id INT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    operation VARCHAR(80) NOT NULL,
                    model VARCHAR(100) NOT NULL,
                    reserved_microusd BIGINT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMPTZ NOT NULL
                );
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_ai_reservations_user_expiry
                ON ai_usage_reservations(user_id, expires_at);
                """
            )
        cls._schema_ready = True

    @classmethod
    def _raw_subscription(cls, user_id: int) -> dict[str, Any] | None:
        cls.ensure_schema()
        with db_cursor() as cur:
            cur.execute(
                """
                SELECT user_id, plan, status, current_period_start,
                       current_period_end, ai_budget_override_microusd,
                       cancel_at_period_end, provider
                FROM user_subscriptions
                WHERE user_id = %s
                LIMIT 1
                """,
                (user_id,),
            )
            row = cur.fetchone()
        return dict(row) if row else None

    @classmethod
    def get_status(cls, user_id: int) -> dict[str, Any]:
        row = cls._raw_subscription(user_id)
        now = datetime.now(timezone.utc)

        plan = "free"
        status = "active"
        period_start = None
        period_end = None
        cancel_at_period_end = False
        budget_override = None

        if row:
            candidate = str(row.get("plan") or "free").lower()
            row_status = str(row.get("status") or "active").lower()
            row_end = row.get("current_period_end")
            is_expired = bool(row_end and row_end <= now)
            if candidate in PLAN_CATALOG and row_status in {"active", "trialing"} and not is_expired:
                plan = candidate
                status = row_status
                period_start = row.get("current_period_start")
                period_end = row_end
                cancel_at_period_end = bool(row.get("cancel_at_period_end"))
                budget_override = row.get("ai_budget_override_microusd")
            elif row_status not in {"active", "trialing"} or is_expired:
                status = "expired" if is_expired else row_status

        default_budget = int(PLAN_CATALOG[plan]["ai_budget_microusd"])
        budget = (
            max(0, int(budget_override))
            if budget_override is not None
            else default_budget
        )

        quota_start, quota_end = monthly_quota_window(now)
        effective_quota_end = min(quota_end, period_end) if period_end else quota_end
        day_start, day_end = daily_quota_window(now)
        with db_cursor() as cur:
            cur.execute(
                """
                SELECT
                    COALESCE(SUM(cost_microusd), 0)::bigint AS cost_microusd,
                    COALESCE(SUM(input_tokens), 0)::bigint AS input_tokens,
                    COALESCE(SUM(cached_input_tokens), 0)::bigint AS cached_input_tokens,
                    COALESCE(SUM(output_tokens), 0)::bigint AS output_tokens,
                    COUNT(*)::int AS requests
                FROM ai_usage_events
                WHERE user_id = %s
                  AND created_at >= %s
                  AND created_at < %s
                """,
                (user_id, quota_start, effective_quota_end),
            )
            usage = dict(cur.fetchone() or {})
            cur.execute(
                """
                SELECT COALESCE(SUM(cost_microusd), 0)::bigint AS cost_microusd
                FROM ai_usage_events
                WHERE user_id = %s
                  AND created_at >= %s
                  AND created_at < %s
                """,
                (user_id, day_start, day_end),
            )
            daily_usage = dict(cur.fetchone() or {})

        spent = max(0, int(usage.get("cost_microusd") or 0))
        remaining = max(0, budget - spent)
        percent = 0 if budget <= 0 else min(100, round(spent / budget * 100))

        beta_override = not bool(settings.SUBSCRIPTIONS_ENFORCED)
        ai_enabled = beta_override or (plan in {"plus", "pro"} and remaining > 0)

        return {
            "plan": plan,
            "status": status,
            "ai_enabled": ai_enabled,
            "beta_override": beta_override,
            "ai_budget_microusd": budget,
            "ai_spent_microusd": spent,
            "ai_remaining_microusd": remaining,
            "ai_usage_percent": percent,
            "ai_usage_level": usage_level(percent),
            "ai_daily_spent_microusd": max(0, int(daily_usage.get("cost_microusd") or 0)),
            "requests": int(usage.get("requests") or 0),
            "input_tokens": int(usage.get("input_tokens") or 0),
            "cached_input_tokens": int(usage.get("cached_input_tokens") or 0),
            "output_tokens": int(usage.get("output_tokens") or 0),
            "ai_quota_period_start": quota_start.isoformat(),
            "ai_quota_period_end": effective_quota_end.isoformat(),
            "current_period_start": period_start.isoformat() if period_start else None,
            "current_period_end": period_end.isoformat() if period_end else None,
            "cancel_at_period_end": cancel_at_period_end,
        }

    @classmethod
    def require_ai_access(cls, user_id: int) -> dict[str, Any]:
        status = cls.get_status(user_id)
        if status["ai_enabled"]:
            return status
        if status["plan"] == "free":
            raise SubscriptionAccessError(
                "melissa_requires_subscription",
                "Мелисса доступна на Plus и Pro. Змея и локальные команды остаются бесплатными.",
            )
        raise SubscriptionAccessError(
            "ai_budget_exhausted",
            "AI-ресурс на этот календарный месяц израсходован. Локальные команды Змеи продолжают работать.",
        )

    @classmethod
    def reserve_request(
        cls,
        *,
        user_id: int,
        operation: str,
        model: str,
        messages: Any,
        max_output_tokens: int,
    ) -> AiBudgetReservation:
        """Atomically reserve a conservative provider-cost ceiling.

        The per-user PostgreSQL advisory transaction lock prevents concurrent
        requests from both spending the same remaining subscription budget.
        """
        status = cls.require_ai_access(user_id)
        estimated_cost, _text_chars = estimate_request_cost_ceiling_microusd(
            model,
            messages,
            max_output_tokens,
        )
        now = datetime.now(timezone.utc)
        quota_start, quota_end = monthly_quota_window(now)
        day_start, day_end = daily_quota_window(now)

        if status["beta_override"] and status["plan"] == "free":
            monthly_limit = settings.BETA_AI_MONTHLY_SAFETY_BUDGET_MICROUSD
            daily_limit = settings.BETA_AI_DAILY_SAFETY_BUDGET_MICROUSD
        else:
            monthly_limit = max(0, int(status["ai_budget_microusd"]))
            daily_limit = max(100_000, int(monthly_limit * 0.30))

        per_request_limit = min(
            250_000,
            max(50_000, int(monthly_limit * 0.10)),
        )
        if estimated_cost > per_request_limit:
            raise SubscriptionAccessError(
                "ai_request_budget_limit",
                "Этот запрос слишком дорогой для одного обращения. Сократи контекст или разбей задачу на несколько шагов.",
                status_code=429,
            )

        cls.ensure_schema()
        reservation_id = uuid.uuid4().hex
        expires_at = now + timedelta(seconds=settings.AI_RESERVATION_TTL_SECONDS)

        with db_cursor(commit=True) as cur:
            cur.execute("SELECT pg_advisory_xact_lock(%s)", (int(user_id),))
            cur.execute(
                "DELETE FROM ai_usage_reservations WHERE expires_at <= %s",
                (now,),
            )
            cur.execute(
                """
                SELECT COALESCE(SUM(cost_microusd), 0)::bigint AS spent
                FROM ai_usage_events
                WHERE user_id = %s AND created_at >= %s AND created_at < %s
                """,
                (user_id, quota_start, quota_end),
            )
            monthly_spent = int((cur.fetchone() or {}).get("spent") or 0)
            cur.execute(
                """
                SELECT COALESCE(SUM(reserved_microusd), 0)::bigint AS reserved
                FROM ai_usage_reservations
                WHERE user_id = %s AND expires_at > %s
                """,
                (user_id, now),
            )
            active_reserved = int((cur.fetchone() or {}).get("reserved") or 0)

            if monthly_spent + active_reserved + estimated_cost > monthly_limit:
                code = (
                    "ai_beta_safety_budget_exhausted"
                    if status["beta_override"] and status["plan"] == "free"
                    else "ai_budget_exhausted"
                )
                raise SubscriptionAccessError(
                    code,
                    "AI-ресурс на текущий период почти исчерпан. Змея и локальные команды продолжают работать.",
                    status_code=429,
                )

            cur.execute(
                """
                SELECT COALESCE(SUM(cost_microusd), 0)::bigint AS spent
                FROM ai_usage_events
                WHERE user_id = %s AND created_at >= %s AND created_at < %s
                """,
                (user_id, day_start, day_end),
            )
            daily_spent = int((cur.fetchone() or {}).get("spent") or 0)
            cur.execute(
                """
                SELECT COALESCE(SUM(reserved_microusd), 0)::bigint AS reserved
                FROM ai_usage_reservations
                WHERE user_id = %s
                  AND created_at >= %s
                  AND created_at < %s
                  AND expires_at > %s
                """,
                (user_id, day_start, day_end, now),
            )
            daily_reserved = int((cur.fetchone() or {}).get("reserved") or 0)
            if daily_spent + daily_reserved + estimated_cost > daily_limit:
                raise SubscriptionAccessError(
                    "ai_daily_safety_limit",
                    "На сегодня достигнут защитный лимит облачного AI. Это предотвращает случайный перерасход; локальная Змея продолжает работать.",
                    status_code=429,
                )

            cur.execute(
                """
                INSERT INTO ai_usage_reservations (
                    reservation_id, user_id, operation, model,
                    reserved_microusd, expires_at
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    reservation_id,
                    user_id,
                    str(operation or "ai")[:80],
                    str(model or "unknown")[:100],
                    estimated_cost,
                    expires_at,
                ),
            )

        return AiBudgetReservation(
            reservation_id=reservation_id,
            reserved_microusd=estimated_cost,
            max_output_tokens=max_output_tokens,
        )

    @classmethod
    def release_reservation(cls, *, user_id: int, reservation_id: str) -> None:
        if not reservation_id:
            return
        cls.ensure_schema()
        with db_cursor(commit=True) as cur:
            cur.execute(
                "DELETE FROM ai_usage_reservations WHERE reservation_id = %s AND user_id = %s",
                (reservation_id, user_id),
            )

    @classmethod
    def finalize_usage(
        cls,
        *,
        user_id: int,
        reservation_id: str,
        operation: str,
        model: str,
        usage: AiUsage,
    ) -> int:
        cls.ensure_schema()
        cost = calculate_cost_microusd(model, usage)
        with db_cursor(commit=True) as cur:
            cur.execute("SELECT pg_advisory_xact_lock(%s)", (int(user_id),))
            cur.execute(
                """
                INSERT INTO ai_usage_events (
                    user_id, operation, model, input_tokens,
                    cached_input_tokens, output_tokens, cost_microusd
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    user_id,
                    str(operation or "unknown")[:80],
                    str(model or "unknown")[:100],
                    usage.input_tokens,
                    usage.cached_input_tokens,
                    usage.output_tokens,
                    cost,
                ),
            )
            cur.execute(
                "DELETE FROM ai_usage_reservations WHERE reservation_id = %s AND user_id = %s",
                (reservation_id, user_id),
            )
        return cost

    @classmethod
    def record_usage(
        cls,
        *,
        user_id: int,
        operation: str,
        model: str,
        usage: AiUsage,
    ) -> int:
        """Backward-compatible direct metering path for non-reserved callers."""
        cls.ensure_schema()
        cost = calculate_cost_microusd(model, usage)
        with db_cursor(commit=True) as cur:
            cur.execute(
                """
                INSERT INTO ai_usage_events (
                    user_id, operation, model, input_tokens,
                    cached_input_tokens, output_tokens, cost_microusd
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    user_id,
                    str(operation or "unknown")[:80],
                    str(model or "unknown")[:100],
                    usage.input_tokens,
                    usage.cached_input_tokens,
                    usage.output_tokens,
                    cost,
                ),
            )
        return cost
